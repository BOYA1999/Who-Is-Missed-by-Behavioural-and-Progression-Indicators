from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_main import nested_oof
from run_pilot import selection_fraction

CONFIG_PATH = ROOT / "artifacts" / "experiment" / "main_v1" / "config.json"
DATA_PATH = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
STORED_PATH = ROOT / "artifacts" / "experiment" / "main_v1" / "oof_predictions.csv.gz"
SEEDS = [20260918, 20260923, 20260924]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    routine_features = config["routine_features"]
    expanded_features = routine_features + config["school_experience_features"]
    columns = list(dict.fromkeys([
        "CNT", "CNTSCHID", "CNTSTUID", "STRATUM", "ST016Q01NA", "W_FSTUWT", *expanded_features
    ]))
    log_lines = []

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        log_lines.append(line)
        print(line, flush=True)
        (OUT / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("Loading selected PISA fields")
    frame, meta = pyreadstat.read_sav(DATA_PATH, usecols=columns, apply_value_formats=False)
    frame = frame.loc[(frame["CNT"] == config["country"]) & frame["ST016Q01NA"].between(0, 10)].copy()
    frame.reset_index(drop=True, inplace=True)
    frame["outcome"] = frame["ST016Q01NA"].between(0, 4).astype(int)
    stratum_labels = meta.variable_value_labels["STRATUM"]
    frame["region"] = frame["STRATUM"].map(
        lambda value: stratum_labels.get(value, str(value)).split(":", 1)[-1].split(",", 1)[0].strip()
    )
    assert len(frame) == 29588 and frame["CNTSCHID"].nunique() == 965
    assert frame["outcome"].sum() == 4057
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    total_w = float(w.sum())
    low_w = float((w * y).sum())
    stored = pd.read_csv(STORED_PATH, usecols=[
        "CNTSTUID", "CNTSCHID", "outcome", "W_FSTUWT",
        "p__routine_logit__school_nested", "p__expanded_logit__school_nested", "fold__school_nested"
    ])
    for column in ["CNTSTUID", "CNTSCHID", "outcome", "W_FSTUWT"]:
        assert np.array_equal(frame[column].to_numpy(), stored[column].to_numpy()), column

    runs = {}
    metric_rows = []
    checks = {}
    for seed in SEEDS:
        run_config = {**config, "seed": seed}
        run = {}
        folds = {}
        for name, features in [("routine", routine_features), ("expanded", expanded_features)]:
            log(f"Seed {seed}: {name} school-nested logistic model")
            p, fold_ids, fold_checks, tuning_rows = nested_oof(
                frame, features, "logit", "school_nested", run_config, log
            )
            assert all(item["school_overlap_n"] == 0 for item in fold_checks)
            assert all(item["train_n"] + item["test_n"] == len(frame) for item in fold_checks)
            assert np.isfinite(p).all()
            folds[name] = fold_ids
            selected = selection_fraction(p, w, 0.10)
            assert np.isclose(np.sum(w * selected) / total_w, 0.10, atol=1e-12)
            run[name] = selected
            metric_rows.append({
                "seed": seed, "information_set": name,
                "recall": float(np.sum(w * y * selected) / low_w),
                "selected_weight_share": float(np.sum(w * selected) / total_w),
            })
            checks[f"{seed}_{name}"] = {
                "fold_train_test_school_overlap": [c["school_overlap_n"] for c in fold_checks],
                "fold_selected_parameters": [c["selected_params"] for c in fold_checks],
                "inner_tuning_rows": len(tuning_rows),
            }
            if seed == SEEDS[0]:
                source_name = f"p__{name}_logit__school_nested"
                max_delta = float(np.max(np.abs(p - stored[source_name].to_numpy(dtype=float))))
                checks[f"{seed}_{name}"]["max_abs_stored_prediction_difference"] = max_delta
                assert max_delta < 1e-10, (name, max_delta)
                assert np.array_equal(fold_ids, stored["fold__school_nested"].to_numpy(dtype=int))
        assert np.array_equal(folds["routine"], folds["expanded"]), seed
        runs[seed] = run

    def compare(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
        both = first * second
        first_only = first * (1 - second)
        second_only = (1 - first) * second
        neither = (1 - first) * (1 - second)
        assert np.allclose(both + first_only + second_only + neither, 1, atol=1e-12)
        overlap_all = float(np.sum(w * both) / total_w)
        first_only_low = float(np.sum(w * y * first_only))
        first_low = float(np.sum(w * y * first))
        second_only_low = float(np.sum(w * y * second_only))
        second_low = float(np.sum(w * y * second))
        return {
            "overlap_all_student_weight_share": overlap_all,
            "queue_replacement_fraction": 1 - overlap_all / 0.10,
            "first_only_all_student_weight_share": float(np.sum(w * first_only) / total_w),
            "second_only_all_student_weight_share": float(np.sum(w * second_only) / total_w),
            "first_covered_low_life_satisfaction_displaced_fraction": first_only_low / first_low,
            "second_covered_low_life_satisfaction_displaced_fraction": second_only_low / second_low,
            "missed_by_both_low_life_satisfaction_share": float(np.sum(w * y * neither) / low_w),
        }

    metrics = pd.DataFrame(metric_rows)
    cross_rows = []
    for seed in SEEDS:
        routine_recall = metrics.loc[(metrics.seed == seed) & (metrics.information_set == "routine"), "recall"].item()
        expanded_recall = metrics.loc[(metrics.seed == seed) & (metrics.information_set == "expanded"), "recall"].item()
        cross_rows.append({
            "seed": seed, "recall_routine": routine_recall, "recall_expanded": expanded_recall,
            "recall_gain": expanded_recall - routine_recall,
            **compare(runs[seed]["routine"], runs[seed]["expanded"]),
        })
    cross = pd.DataFrame(cross_rows)
    first = cross.iloc[0]
    assert round(first.recall_routine * 100, 1) == 16.3
    assert round(first.recall_expanded * 100, 1) == 26.5
    assert round(first.overlap_all_student_weight_share * 100, 2) == 3.63
    assert round(first.first_covered_low_life_satisfaction_displaced_fraction * 100, 1) == 40.4
    within_rows = []
    for name in ["routine", "expanded"]:
        for seed_a, seed_b in itertools.combinations(SEEDS, 2):
            within_rows.append({
                "information_set": name, "seed_a": seed_a, "seed_b": seed_b,
                **compare(runs[seed_a][name], runs[seed_b][name]),
            })
    within = pd.DataFrame(within_rows)
    assert np.isfinite(cross.select_dtypes(include="number").to_numpy()).all()
    assert np.isfinite(within.select_dtypes(include="number").to_numpy()).all()
    metrics.to_csv(OUT / "run_metrics_aggregate.csv", index=False)
    cross.to_csv(OUT / "cross_information_aggregate.csv", index=False)
    within.to_csv(OUT / "within_information_aggregate.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    manifest = {
        "command": ".venv\\Scripts\\python.exe artifacts\\stability_revision_20260923\\run_stability.py",
        "seeds": SEEDS, "same_students": len(frame), "schools": int(frame.CNTSCHID.nunique()),
        "outcome_events": int(y.sum()), "data_sav_sha256": sha256(DATA_PATH),
        "config_sha256": sha256(CONFIG_PATH), "source_script_sha256": sha256(Path(__file__)),
        "run_main_sha256": sha256(ROOT / "scripts" / "run_main.py"),
        "run_pilot_sha256": sha256(ROOT / "scripts" / "run_pilot.py"),
        "stored_oof_sha256": sha256(STORED_PATH),
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "pyreadstat", "scikit-learn"
        ]},
        "outputs_are_aggregate_only": True,
        "scope": "Descriptive three-seed school-split sensitivity; no full model-uncertainty CI",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("Completed: original-seed gate and three-seed aggregate stability comparison passed")


if __name__ == "__main__":
    main()
