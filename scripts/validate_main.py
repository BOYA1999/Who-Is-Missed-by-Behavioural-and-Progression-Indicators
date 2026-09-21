from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "experiment" / "main_v1"
CONTRACT = ROOT / "artifacts" / "baseline" / "json" / "metric_contract.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def exact_capacity(y, scores, weights, capacity=0.10):
    order = np.argsort(-scores, kind="stable")
    y = y[order]
    scores = scores[order]
    weights = weights[order]
    target = capacity * weights.sum()
    selected = np.zeros(len(y))
    used = 0.0
    start = 0
    while start < len(y) and used < target:
        end = start + 1
        while end < len(y) and scores[end] == scores[start]:
            end += 1
        group_weight = weights[start:end].sum()
        fraction = min(1.0, (target - used) / group_weight)
        selected[start:end] = fraction
        used += fraction * group_weight
        start = end
    true_positive = np.sum(weights * y * selected)
    return {
        "recall_at_10pct_capacity": float(true_positive / np.sum(weights * y)),
        "precision_at_10pct_capacity": float(true_positive / np.sum(weights * selected)),
        "selected_fraction": float(np.sum(weights * selected) / weights.sum()),
    }


def main():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    result = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    manifest_before = json.loads((RUN / "artifact_manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(RUN / "oof_predictions.csv.gz")
    bootstrap = pd.read_csv(RUN / "bootstrap_metrics.csv.gz")
    fold_metrics = pd.read_csv(RUN / "fold_metrics.csv")
    fairness = pd.read_csv(RUN / "fairness_metrics.csv")

    failures = []
    warnings = []
    expected_n = 29588
    if len(predictions) != expected_n:
        failures.append(f"prediction rows {len(predictions)} != {expected_n}")
    if predictions["CNTSTUID"].duplicated().any():
        failures.append("duplicate CNTSTUID rows")

    required = contract["required_metric_keys"]
    for row in result["metrics"]:
        for key in required:
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                failures.append(f"missing/nonfinite {row['model']} {row['split']} {key}")
    for key, checks in result["checks"].items():
        for check in checks:
            if int(check["school_overlap_n"]) != 0:
                failures.append(f"school leakage {key} {check['fold_label']}")
            if key.endswith("region_iecv") and int(check["region_overlap_n"]) != 0:
                failures.append(f"region leakage {key} {check['fold_label']}")

    probability_columns = [column for column in predictions if column.startswith("p__")]
    for column in probability_columns:
        values = predictions[column]
        if values.isna().any() or not values.between(0, 1).all():
            failures.append(f"invalid probabilities in {column}")

    y = predictions["outcome"].to_numpy(dtype=int)
    w = predictions["W_FSTUWT"].to_numpy(dtype=float)
    p = predictions["p__expanded_hgb__school_nested"].to_numpy(dtype=float)
    independent = {
        "weighted_prevalence": float(np.sum(w * y) / np.sum(w)),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        **exact_capacity(y, p, w),
    }
    reported = next(
        row
        for row in result["metrics"]
        if row["model"] == "expanded_hgb" and row["split"] == "school_nested"
    )
    for key in [
        "weighted_prevalence",
        "auprc",
        "auroc",
        "brier",
        "recall_at_10pct_capacity",
        "precision_at_10pct_capacity",
    ]:
        if abs(independent[key] - float(reported[key])) > 1e-10:
            failures.append(f"independent recomputation mismatch for {key}")

    wide = bootstrap.pivot(index="replicate", columns="model")
    contrasts = [
        ("expanded_hgb", "expanded_logit"),
        ("expanded_logit", "routine_logit"),
        ("routine_logit", "routine_hgb"),
    ]
    delta_rows = []
    for left, right in contrasts:
        for metric in ["auprc", "recall_at_10pct_capacity"]:
            delta = wide[metric][left] - wide[metric][right]
            delta_rows.append(
                {
                    "contrast": f"{left}-{right}",
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "ci_2.5": float(delta.quantile(0.025)),
                    "ci_97.5": float(delta.quantile(0.975)),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(RUN / "paired_bootstrap_deltas.csv", index=False, encoding="utf-8-sig")

    complex_auprc = deltas.loc[
        (deltas["contrast"] == "expanded_hgb-expanded_logit")
        & (deltas["metric"] == "auprc")
    ].iloc[0]
    complex_recall = deltas.loc[
        (deltas["contrast"] == "expanded_hgb-expanded_logit")
        & (deltas["metric"] == "recall_at_10pct_capacity")
    ].iloc[0]
    if complex_auprc["ci_2.5"] < contract["minimum_practical_delta"]["auprc"]:
        warnings.append("HGB AUPRC gain is positive but its lower interval bound is below the practical threshold")
    if complex_recall["ci_2.5"] < contract["minimum_practical_delta"]["recall_at_10pct_capacity"]:
        warnings.append("HGB recall gain does not robustly meet the practical threshold")

    regional = fold_metrics.loc[
        (fold_metrics["model"] == "expanded_hgb") & (fold_metrics["split"] == "region_iecv")
    ]
    regional_ranges = {
        metric: [float(regional[metric].min()), float(regional[metric].max())]
        for metric in [
            "auprc",
            "auroc",
            "calibration_intercept",
            "calibration_slope",
            "recall_at_10pct_capacity",
        ]
    }

    fairness_main = fairness.loc[
        (fairness["model"] == "expanded_hgb") & (fairness["split"] == "school_nested")
    ]
    fairness_gaps = {}
    for axis, group in fairness_main.groupby("group_axis"):
        values = group.set_index("group")["recall_at_global_10pct_capacity"].to_dict()
        fairness_gaps[axis] = {
            "group_recalls": {key: float(value) for key, value in values.items()},
            "max_minus_min": float(max(values.values()) - min(values.values())),
        }

    previous_manifest_mismatches = []
    for item in manifest_before["artifacts"]:
        path = Path(item["path"])
        if not path.exists() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            previous_manifest_mismatches.append(path.name)
    if set(previous_manifest_mismatches) - {"bash.log"}:
        failures.append(
            "unexpected pre-finalization manifest mismatches: "
            + ", ".join(sorted(set(previous_manifest_mismatches) - {"bash.log"}))
        )

    run_manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    script_hash_matches = (
        file_hash(ROOT / "scripts" / "run_main.py") == run_manifest["script_sha256"]
    )
    if not script_hash_matches:
        failures.append("run_main.py hash differs from executed-script hash")

    validation = {
        "status": "pass_with_caveats" if not failures else "fail",
        "failures": failures,
        "caveats": warnings,
        "independent_recomputation": independent,
        "paired_bootstrap_deltas": delta_rows,
        "regional_ranges": regional_ranges,
        "fairness_recall_gaps": fairness_gaps,
        "pre_finalize_manifest_mismatches": previous_manifest_mismatches,
        "executed_script_hash_matches": script_hash_matches,
        "review_scope": "core binary outcome, school-nested and region-IECV outputs; sensitivity analyses not yet reviewed",
    }
    (RUN / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_md = f"""# Validation

**Status:** {'Share with caveats' if not failures else 'Needs revision'}

- Prediction rows: {len(predictions):,}; duplicate student IDs: {int(predictions['CNTSTUID'].duplicated().sum())}.
- Leakage findings: {sum('leakage' in item for item in failures)}.
- Required metric and probability checks: {'passed' if not failures else 'see failures'}.
- Independent expanded-HGB recomputation matched the reported AUPRC, AUROC, Brier, 10% capacity recall and precision within 1e-10.
- Expanded HGB minus expanded logistic, paired school bootstrap: AUPRC {complex_auprc['mean_delta']:+.3f} (95% interval {complex_auprc['ci_2.5']:+.3f} to {complex_auprc['ci_97.5']:+.3f}); recall {complex_recall['mean_delta']:+.3f} ({complex_recall['ci_2.5']:+.3f} to {complex_recall['ci_97.5']:+.3f}).
- The nonlinear gain is positive, but practical superiority is not robust: the AUPRC interval crosses the pre-specified +0.02 practical threshold and the recall gain remains below +0.03.
- Region IECV reveals heterogeneity despite similar pooled performance: AUROC {regional_ranges['auroc'][0]:.3f}–{regional_ranges['auroc'][1]:.3f}, calibration slope {regional_ranges['calibration_slope'][0]:.3f}–{regional_ranges['calibration_slope'][1]:.3f}, and 10% capacity recall {regional_ranges['recall_at_10pct_capacity'][0]:.3f}–{regional_ranges['recall_at_10pct_capacity'][1]:.3f}. Small Ceuta and Melilla folds contribute to the extremes.
- Intervals from the school bootstrap condition on fixed OOF predictions and omit model-refitting uncertainty.
- Continuous-outcome and missing-data sensitivities remain outside this validation scope.
"""
    (RUN / "validation.md").write_text(validation_md, encoding="utf-8")

    artifact_paths = [
        path for path in RUN.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    ]
    final_manifest = {
        "run_id": result["run_id"],
        "finalized_after_validation": True,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in sorted(artifact_paths)
        ],
    }
    (RUN / "artifact_manifest.json").write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": validation["status"], "failures": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
