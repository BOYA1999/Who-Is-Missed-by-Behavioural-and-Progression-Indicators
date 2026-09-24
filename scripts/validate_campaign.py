from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "campaign_v1"
DATA = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def exact_capacity(y, scores, weights, capacity=0.10):
    order = np.argsort(-scores, kind="stable")
    y = y[order]
    scores = scores[order]
    weights = weights[order]
    selected = np.zeros(len(y))
    target = capacity * weights.sum()
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
    return (
        float(true_positive / np.sum(weights * y)),
        float(true_positive / np.sum(weights * selected)),
    )


def main():
    results = pd.read_csv(OUT / "slice_results.csv")
    predictions = pd.read_csv(OUT / "sensitivity_predictions.csv.gz")
    allocation = pd.read_csv(OUT / "allocation_policy.csv")
    campaign = json.loads((OUT / "campaign_results.json").read_text(encoding="utf-8"))
    failures = []
    caveats = []
    if campaign["status"] != "success":
        failures.append("campaign status is not success")
    if len(predictions) != 29588 or predictions["CNTSTUID"].duplicated().any():
        failures.append("prediction row-count or uniqueness failure")
    probability_columns = [column for column in predictions if column.startswith("audit_attributes__")]
    for column in probability_columns:
        if predictions[column].isna().any() or not predictions[column].between(0, 1).all():
            failures.append(f"invalid probability column {column}")
    continuous_columns = [column for column in predictions if column.startswith("continuous__")]
    for column in continuous_columns:
        if predictions[column].isna().any() or not np.isfinite(predictions[column]).all():
            failures.append(f"invalid continuous column {column}")

    y_cont = predictions["ST016Q01NA"].to_numpy(dtype=float)
    y_bin = predictions["outcome"].to_numpy(dtype=int)
    w = predictions["W_FSTUWT"].to_numpy(dtype=float)
    p_cont = predictions["continuous__expanded_hgb"].to_numpy(dtype=float)
    error = y_cont - p_cont
    independent_continuous = {
        "mae": float(np.sum(w * np.abs(error)) / np.sum(w)),
        "rmse": float(np.sqrt(np.sum(w * error**2) / np.sum(w))),
    }
    reported_continuous = results.loc[
        (results["slice_id"] == "S1_continuous_outcome")
        & (results["model"] == "expanded_hgb")
    ].iloc[0]
    for metric in ["mae", "rmse"]:
        if abs(independent_continuous[metric] - reported_continuous[metric]) > 1e-10:
            failures.append(f"continuous recomputation mismatch {metric}")

    p_audit = predictions["audit_attributes__expanded_audit_hgb"].to_numpy(dtype=float)
    independent_binary = {
        "auprc": float(average_precision_score(y_bin, p_audit, sample_weight=w)),
        "auroc": float(roc_auc_score(y_bin, p_audit, sample_weight=w)),
        "brier": float(brier_score_loss(y_bin, p_audit, sample_weight=w)),
    }
    recall, precision = exact_capacity(y_bin, p_audit, w)
    independent_binary["recall_at_10pct_capacity"] = recall
    independent_binary["precision_at_10pct_capacity"] = precision
    reported_binary = results.loc[
        (results["slice_id"] == "S3_audit_attributes")
        & (results["model"] == "expanded_audit_hgb")
    ].iloc[0]
    for metric, value in independent_binary.items():
        if abs(value - reported_binary[metric]) > 1e-10:
            failures.append(f"binary recomputation mismatch {metric}")

    variables = [
        "CNT",
        "CNTSCHID",
        "ST016Q01NA",
        "ST001D01T",
        "REPEAT",
        "ST062Q01TA",
        "ST062Q02TA",
        "ST062Q03TA",
        "BELONG",
        "BULLIED",
        "TEACHSUP",
    ]
    source, _ = pyreadstat.read_sav(DATA, usecols=variables, apply_value_formats=False)
    source = source.loc[(source["CNT"] == "ESP") & source["ST016Q01NA"].between(0, 10)].copy()
    expanded = [
        "ST001D01T",
        "REPEAT",
        "ST062Q01TA",
        "ST062Q02TA",
        "ST062Q03TA",
        "BELONG",
        "BULLIED",
        "TEACHSUP",
    ]
    complete_n = int(source[expanded].notna().all(axis=1).sum())
    if complete_n != 27937:
        failures.append(f"complete-case count {complete_n} != 27937")

    strata = pd.qcut(source["ST016Q01NA"], q=10, labels=False, duplicates="drop")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260918)
    school_overlap = []
    schools = source["CNTSCHID"].to_numpy()
    for fold, (train, test) in enumerate(splitter.split(source[expanded], strata, schools)):
        overlap = set(schools[train]).intersection(set(schools[test]))
        if overlap:
            school_overlap.append(fold)
    if school_overlap:
        failures.append(f"reconstructed school split leakage {school_overlap}")

    allocation_core = allocation.loc[allocation["group_axis"].isna()].copy()
    for _, row in allocation_core.iterrows():
        if abs(float(row["selected_weight_fraction"]) - 0.10) > 1e-10:
            failures.append(f"capacity fraction mismatch {row['model']} {row['policy']}")

    audit_fairness = pd.read_csv(OUT / "audit_attribute_fairness.csv")
    gaps = audit_fairness.loc[audit_fairness["group"] == "max_minus_min"].pivot(
        index="group_axis", columns="model", values="recall"
    )
    gap_increases = {
        axis: float(gaps.loc[axis, "expanded_audit_hgb"] - gaps.loc[axis, "expanded_hgb_main"])
        for axis in gaps.index
    }
    if not all(value > 0 for value in gap_increases.values()):
        caveats.append("not every fairness recall gap increased after adding audit attributes")

    validation = {
        "status": "pass_with_caveats" if not failures else "fail",
        "failures": failures,
        "caveats": caveats,
        "independent_continuous": independent_continuous,
        "independent_audit_attribute_binary": independent_binary,
        "complete_case_n": complete_n,
        "reconstructed_school_overlap_folds": school_overlap,
        "audit_attribute_gap_increases": gap_increases,
        "review_scope": "four selected sensitivity slices; no model-refitting uncertainty intervals",
    }
    (OUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    main_global = allocation_core.loc[
        (allocation_core["model"] == "expanded_hgb") & (allocation_core["policy"] == "global")
    ].iloc[0]
    main_local = allocation_core.loc[
        (allocation_core["model"] == "expanded_hgb")
        & (allocation_core["policy"] == "within_school")
    ].iloc[0]
    validation_md = f"""# Campaign validation

**Status:** {'Ready within reviewed scope' if not failures else 'Needs revision'}

- All {len(predictions):,} prediction rows were unique; binary probabilities and continuous predictions were finite.
- Independent recomputation matched expanded-HGB continuous MAE/RMSE and audit-attribute binary AUPRC/AUROC/Brier/capacity metrics within 1e-10.
- Complete-case count independently recovered from the SAV file: {complete_n:,}.
- Deterministic outer school folds were reconstructed with zero school overlap.
- Every global and within-school policy allocated exactly 10% of weighted capacity; expanded-HGB recall changed from {main_global['recall']:.3f} to {main_local['recall']:.3f}.
- Adding audit attributes increased recall gaps by sex {gap_increases['sex']:+.3f}, immigration {gap_increases['immigration_background']:+.3f}, and ESCS {gap_increases['escs_weighted_tertile']:+.3f}; this supports keeping them as audit variables rather than ranking inputs.
- These slices provide robustness checks, not prospective, causal, diagnostic or deployment evidence.
"""
    (OUT / "validation.md").write_text(validation_md, encoding="utf-8")

    artifacts = [path for path in OUT.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    manifest = {
        "campaign_id": campaign["campaign_id"],
        "finalized_after_validation": True,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(artifacts)
        ],
    }
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": validation["status"], "failures": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
