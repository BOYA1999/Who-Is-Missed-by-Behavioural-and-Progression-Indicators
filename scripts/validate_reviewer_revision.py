from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from run_pilot import selection_fraction


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "reviewer_revision_v1"
MAIN = ROOT / "artifacts" / "experiment" / "main_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures = []
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((OUT / "validation.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(OUT / "revision_predictions.csv.gz")
    main_oof = pd.read_csv(MAIN / "oof_predictions.csv.gz")
    frame = main_oof[["CNTSTUID", "outcome", "W_FSTUWT"]].merge(
        predictions, on="CNTSTUID", how="inner", validate="one_to_one"
    )
    metrics = pd.read_csv(OUT / "model_metrics.csv").set_index("model")
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    model_columns = [column for column in predictions if column not in {"CNTSTUID", "CNTSCHID", "fold__school_nested"}]
    recomputed = {}
    for model in model_columns:
        p = frame[model].to_numpy(dtype=float)
        selected = selection_fraction(p, w, 0.10)
        tp = float(np.sum(w * y * selected))
        values = {
            "auprc": float(average_precision_score(y, p, sample_weight=w)),
            "recall": tp / float(np.sum(w * y)),
            "precision": tp / float(np.sum(w * selected)),
            "capacity": float(np.sum(w * selected) / np.sum(w)),
        }
        recomputed[model] = values
        stored = metrics.loc[model]
        if abs(values["auprc"] - stored["auprc"]) > 1e-12:
            failures.append(f"AUPRC mismatch for {model}")
        if abs(values["recall"] - stored["recall_at_10pct_capacity"]) > 1e-12:
            failures.append(f"recall mismatch for {model}")
        if abs(values["precision"] - stored["precision_at_10pct_capacity"]) > 1e-12:
            failures.append(f"precision mismatch for {model}")
        if abs(values["capacity"] - 0.10) > 1e-12:
            failures.append(f"capacity mismatch for {model}")

    main_metrics = pd.read_csv(MAIN / "metrics.csv")
    main_metrics = main_metrics.loc[main_metrics["split"].eq("school_nested")].set_index("model")
    for model in ["routine_logit", "expanded_logit", "expanded_hgb"]:
        for metric in ["auprc", "recall_at_10pct_capacity", "precision_at_10pct_capacity"]:
            if abs(metrics.loc[model, metric] - main_metrics.loc[model, metric]) > 1e-12:
                failures.append(f"baseline metric mismatch for {model} {metric}")

    brr = pd.read_csv(OUT / "brr_performance_replicates.csv.gz")
    if len(brr) != 80 * len(model_columns):
        failures.append("BRR row count mismatch")
    if float((brr["selected_weight_fraction"] - 0.10).abs().max()) > 1e-12:
        failures.append("BRR capacity was not preserved")
    brr_deltas = pd.read_csv(OUT / "brr_deltas.csv")
    target = brr_deltas.loc[
        brr_deltas["model"].eq("expanded_logit")
        & brr_deltas["reference"].eq("routine_logit")
        & brr_deltas["metric"].eq("recall_at_10pct_capacity")
    ].iloc[0]
    independent_delta = float(metrics.loc["expanded_logit", "recall_at_10pct_capacity"] - metrics.loc["routine_logit", "recall_at_10pct_capacity"])
    if abs(target["estimate_delta"] - independent_delta) > 1e-12:
        failures.append("BRR delta estimate mismatch")

    overlap = pd.read_csv(OUT / "queue_overlap.csv")
    for population in overlap["population"].unique():
        cells = overlap.loc[
            overlap["population"].eq(population)
            & overlap["cell"].isin(["both", "routine_only", "expanded_only", "neither"]),
            "weighted_share",
        ]
        if abs(float(cells.sum()) - 1.0) > 1e-12:
            failures.append(f"overlap cells do not sum to one for {population}")
    all_overlap = overlap.loc[overlap["population"].eq("all_students")].set_index("cell")
    if abs(all_overlap.loc["routine_only", "weighted_share"] - all_overlap.loc["expanded_only", "weighted_share"]) > 1e-12:
        failures.append("equal-capacity displacement symmetry failed")
    if abs(all_overlap.loc["union", "weighted_share"] - summary["all_student_union_capacity"]) > 1e-12:
        failures.append("union capacity mismatch")

    coefficients = pd.read_csv(OUT / "logistic_coefficients.csv")
    if set(coefficients["fold"].unique()) != {0, 1, 2, 3, 4}:
        failures.append("coefficient folds incomplete")
    hgb = pd.read_csv(OUT / "hgb_no_earlystop_folds.csv")
    if not ((hgb["early_stopping"] == False).all() and (hgb["n_iter"] == 200).all()):
        failures.append("HGB early-stopping sensitivity settings mismatch")
    region = pd.read_csv(OUT / "region_uncertainty.csv")
    if len(region) != 19 or int(region["students"].sum()) != 29588:
        failures.append("regional accounting mismatch")
    sex_curve = pd.read_csv(OUT / "sex_calibration_curve.csv")
    if set(sex_curve["group"]) != {"female", "male"} or sex_curve.groupby("group")["bin"].nunique().min() < 8:
        failures.append("sex calibration curve incomplete")

    manifest = json.loads((OUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = ROOT / item["path"]
        if not path.exists() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append(f"manifest mismatch: {item['path']}")

    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": {
            "rows": int(len(frame)),
            "models": len(model_columns),
            "brr_rows": int(len(brr)),
            "regions": int(len(region)),
            "manifest_entries_checked": len(manifest["artifacts"]),
        },
        "independent_recomputed_metrics": recomputed,
    }
    (OUT / "independent_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifacts = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    (OUT / "artifact_manifest.json").write_text(
        json.dumps({"analysis_id": "reviewer_revision_v1", "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
