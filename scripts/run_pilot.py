from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pyreadstat
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "experiment" / "pilot_v1"
CONFIG_PATH = RUN_DIR / "config.json"
DATA_PATH = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def selection_fraction(scores: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
    target_weight = capacity * float(weights.sum())
    selected = np.zeros(len(scores), dtype=float)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    start = 0
    used = 0.0
    while start < len(order) and used < target_weight:
        end = start + 1
        while end < len(order) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        group = order[start:end]
        group_weight = float(weights[group].sum())
        remaining = target_weight - used
        if group_weight <= remaining + 1e-12:
            selected[group] = 1.0
            used += group_weight
        else:
            selected[group] = max(0.0, remaining / group_weight)
            used = target_weight
        start = end
    return selected


def calibration_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    x = logit(clipped).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(x, y, sample_weight=w / w.mean())
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def core_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, capacities: list[float]) -> dict:
    intercept, slope = calibration_metrics(y, p, w)
    result = {
        "n": int(len(y)),
        "weighted_prevalence": weighted_mean(y, w),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    positive_weight = float(np.sum(w * y))
    for capacity in capacities:
        selected = selection_fraction(p, w, capacity)
        true_positive_weight = float(np.sum(w * y * selected))
        selected_weight = float(np.sum(w * selected))
        suffix = f"{int(capacity * 100)}pct_capacity"
        result[f"recall_at_{suffix}"] = true_positive_weight / positive_weight
        result[f"precision_at_{suffix}"] = true_positive_weight / selected_weight
        result[f"selected_weight_fraction_at_{suffix}"] = selected_weight / float(w.sum())
    return result


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probabilities: list[float]) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= sorted_weights.sum()
    return np.interp(probabilities, cumulative, sorted_values)


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "logit":
        classifier = LogisticRegression(
            C=1.0,
            l1_ratio=0.1,
            solver="saga",
            max_iter=3000,
            tol=1e-5,
            random_state=seed,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", classifier),
        ]
    )


def run_oof(
    frame: pd.DataFrame,
    features: list[str],
    kind: str,
    split_name: str,
    seed: int,
    n_splits: int,
    log,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    groups = frame["CNTSCHID"].to_numpy()
    if split_name == "school_grouped":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(frame[features], y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(frame[features], y)
    predictions = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=int)
    fold_checks = []
    for fold, (train_index, test_index) in enumerate(splits):
        overlap = set(groups[train_index]).intersection(set(groups[test_index]))
        if split_name == "school_grouped" and overlap:
            raise RuntimeError(f"school leakage in fold {fold}")
        model = make_model(kind, seed + fold)
        fit_weight = w[train_index] / w[train_index].mean()
        model.fit(
            frame.iloc[train_index][features],
            y[train_index],
            classifier__sample_weight=fit_weight,
        )
        predictions[test_index] = model.predict_proba(frame.iloc[test_index][features])[:, 1]
        fold_ids[test_index] = fold
        fold_checks.append(
            {
                "fold": fold,
                "train_n": int(len(train_index)),
                "test_n": int(len(test_index)),
                "train_schools": int(pd.Series(groups[train_index]).nunique()),
                "test_schools": int(pd.Series(groups[test_index]).nunique()),
                "school_overlap_n": int(len(overlap)),
                "test_events": int(y[test_index].sum()),
            }
        )
        log(f"{split_name} {kind} fold {fold + 1}/{n_splits} complete")
    if np.isnan(predictions).any() or np.any(fold_ids < 0):
        raise RuntimeError("incomplete out-of-fold predictions")
    return predictions, fold_ids, fold_checks


def safe_group_metrics(y, p, w, selected, mask) -> dict | None:
    if int(mask.sum()) < 50 or len(np.unique(y[mask])) < 2:
        return None
    group_y = y[mask]
    group_p = p[mask]
    group_w = w[mask]
    group_selected = selected[mask]
    intercept, slope = calibration_metrics(group_y, group_p, group_w)
    positives = float(np.sum(group_w * group_y))
    selected_weight = float(np.sum(group_w * group_selected))
    true_positives = float(np.sum(group_w * group_y * group_selected))
    return {
        "n": int(mask.sum()),
        "events": int(group_y.sum()),
        "weighted_prevalence": weighted_mean(group_y, group_w),
        "auprc": float(average_precision_score(group_y, group_p, sample_weight=group_w)),
        "auroc": float(roc_auc_score(group_y, group_p, sample_weight=group_w)),
        "brier": float(brier_score_loss(group_y, group_p, sample_weight=group_w)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "recall_at_global_10pct_capacity": true_positives / positives,
        "precision_at_global_10pct_capacity": true_positives / selected_weight if selected_weight else None,
        "selection_rate": selected_weight / float(group_w.sum()),
    }


def dataframe_markdown(frame: pd.DataFrame, decimals: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{decimals}f}")
    headers = list(formatted.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / ("smoke.log" if args.smoke else "bash.log")
    log_lines = []

    def log(message: str) -> None:
        stamped = f"{datetime.now(timezone.utc).isoformat()} {message}"
        log_lines.append(stamped)
        print(stamped, flush=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    started = datetime.now(timezone.utc)
    variables = list(
        dict.fromkeys(
            [
                "CNT",
                "CNTSCHID",
                "CNTSTUID",
                "STRATUM",
                "ST016Q01NA",
                "W_FSTUWT",
                *config["routine_features"],
                *config["school_experience_features"],
                *config["audit_features"],
            ]
        )
    )
    log("loading selected columns")
    frame, meta = pyreadstat.read_sav(DATA_PATH, usecols=variables, apply_value_formats=False)
    frame = frame.loc[(frame["CNT"] == config["country"]) & frame["ST016Q01NA"].between(0, 10)].copy()
    frame.reset_index(drop=True, inplace=True)
    frame["outcome"] = frame["ST016Q01NA"].between(0, 4).astype(int)
    labels = meta.variable_value_labels["STRATUM"]
    frame["region"] = frame["STRATUM"].map(
        lambda value: labels.get(value, str(value)).split(":", 1)[-1].split(",", 1)[0].strip()
    )
    if args.smoke:
        smoke_schools = frame["CNTSCHID"].drop_duplicates().iloc[:100]
        frame = frame.loc[frame["CNTSCHID"].isin(smoke_schools)].copy().reset_index(drop=True)
        config = {**config, "n_splits": 2}
        model_specs = [("routine_logit", config["routine_features"], "logit")]
        split_names = ["school_grouped"]
    else:
        model_specs = [
            ("routine_logit", config["routine_features"], "logit"),
            ("routine_hgb", config["routine_features"], "hgb"),
            (
                "expanded_logit",
                config["routine_features"] + config["school_experience_features"],
                "logit",
            ),
            (
                "expanded_hgb",
                config["routine_features"] + config["school_experience_features"],
                "hgb",
            ),
        ]
        split_names = ["school_grouped", "student_random"]
    log(f"analytic rows={len(frame)} schools={frame['CNTSCHID'].nunique()} events={frame['outcome'].sum()}")
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    prediction_frame = frame[
        [
            "CNTSTUID",
            "CNTSCHID",
            "STRATUM",
            "region",
            "ST016Q01NA",
            "outcome",
            "W_FSTUWT",
            "ST004D01T",
            "IMMIG",
            "ESCS",
        ]
    ].copy()
    metric_rows = []
    fold_metric_rows = []
    fold_checks = {}
    for model_name, features, kind in model_specs:
        for split_name in split_names:
            log(f"starting {model_name} / {split_name}")
            predictions, fold_ids, checks = run_oof(
                frame,
                features,
                kind,
                split_name,
                config["seed"],
                config["n_splits"],
                log,
            )
            key = f"{model_name}__{split_name}"
            prediction_frame[f"p__{key}"] = predictions
            prediction_frame[f"fold__{split_name}"] = fold_ids
            metrics = core_metrics(y, predictions, w, config["capacities"])
            metric_rows.append({"model": model_name, "split": split_name, **metrics})
            fold_checks[key] = checks
            for fold in range(config["n_splits"]):
                mask = fold_ids == fold
                fold_metric_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "fold": fold,
                        **core_metrics(y[mask], predictions[mask], w[mask], config["capacities"]),
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows)
    fold_metrics_frame = pd.DataFrame(fold_metric_rows)
    if args.smoke:
        smoke_output = {
            "status": "success",
            "rows": len(frame),
            "schools": int(frame["CNTSCHID"].nunique()),
            "metrics": metric_rows,
            "checks": fold_checks,
        }
        (RUN_DIR / "smoke_metrics.json").write_text(
            json.dumps(smoke_output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log("smoke run complete")
        return

    region_audit = (
        frame.groupby("region", dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "n": len(group),
                    "schools": group["CNTSCHID"].nunique(),
                    "events": group["outcome"].sum(),
                    "weighted_prevalence": weighted_mean(
                        group["outcome"].to_numpy(), group["W_FSTUWT"].to_numpy()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    region_audit.to_csv(RUN_DIR / "spain_region_audit.csv", index=False, encoding="utf-8-sig")

    escs_valid = frame["ESCS"].notna().to_numpy()
    escs_cuts = weighted_quantiles(
        frame.loc[escs_valid, "ESCS"].to_numpy(),
        frame.loc[escs_valid, "W_FSTUWT"].to_numpy(),
        [1 / 3, 2 / 3],
    )
    group_definitions = {
        "sex": {
            "female": frame["ST004D01T"].eq(1).to_numpy(),
            "male": frame["ST004D01T"].eq(2).to_numpy(),
        },
        "immigration_background": {
            "native": frame["IMMIG"].eq(1).to_numpy(),
            "immigrant": frame["IMMIG"].isin([2, 3]).to_numpy(),
        },
        "escs_weighted_tertile": {
            "low": frame["ESCS"].le(escs_cuts[0]).to_numpy(),
            "middle": frame["ESCS"].gt(escs_cuts[0]).to_numpy()
            & frame["ESCS"].le(escs_cuts[1]).to_numpy(),
            "high": frame["ESCS"].gt(escs_cuts[1]).to_numpy(),
        },
    }
    fairness_rows = []
    for model_name, _, _ in model_specs:
        p = prediction_frame[f"p__{model_name}__school_grouped"].to_numpy()
        selected = selection_fraction(p, w, 0.10)
        for group_axis, levels in group_definitions.items():
            for level, mask in levels.items():
                values = safe_group_metrics(y, p, w, selected, mask)
                if values:
                    fairness_rows.append(
                        {"model": model_name, "group_axis": group_axis, "group": level, **values}
                    )
    fairness_frame = pd.DataFrame(fairness_rows)

    predictions_path = RUN_DIR / "oof_predictions.csv.gz"
    folds_path = RUN_DIR / "fold_assignments.csv.gz"
    metrics_csv_path = RUN_DIR / "metrics.csv"
    fold_metrics_path = RUN_DIR / "fold_metrics.csv"
    fairness_path = RUN_DIR / "fairness_metrics.csv"
    prediction_frame.to_csv(predictions_path, index=False, compression="gzip")
    prediction_frame[
        ["CNTSTUID", "CNTSCHID", "region", "fold__school_grouped", "fold__student_random"]
    ].to_csv(folds_path, index=False, compression="gzip")
    metrics_frame.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    fold_metrics_frame.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    fairness_frame.to_csv(fairness_path, index=False, encoding="utf-8-sig")

    prevalence = weighted_mean(y, w)
    baselines = {
        "random_capacity": {
            f"recall_at_{int(q * 100)}pct_capacity": q for q in config["capacities"]
        },
        "no_skill_auprc": prevalence,
        "constant_probability_brier": prevalence * (1 - prevalence),
        "oracle_recall_at_10pct_capacity": min(1.0, 0.10 / prevalence),
    }
    result = {
        "run_id": config["run_id"],
        "status": "success",
        "stage": config["stage"],
        "metric_contract": str(ROOT / "artifacts" / "baseline" / "json" / "metric_contract.json"),
        "baselines": baselines,
        "metrics": metric_rows,
        "fold_checks": fold_checks,
        "escs_weighted_tertile_cuts": escs_cuts.tolist(),
    }
    (RUN_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display_columns = [
        "model",
        "split",
        "weighted_prevalence",
        "auprc",
        "auroc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "recall_at_10pct_capacity",
        "precision_at_10pct_capacity",
    ]
    metrics_md = "# Pilot metrics\n\n" + dataframe_markdown(metrics_frame[display_columns]) + "\n"
    (RUN_DIR / "metrics.md").write_text(metrics_md, encoding="utf-8")

    grouped = metrics_frame.loc[metrics_frame["split"] == "school_grouped"].set_index("model")
    random = metrics_frame.loc[metrics_frame["split"] == "student_random"].set_index("model")
    routine = grouped.loc["routine_logit"]
    expanded = grouped.loc["expanded_logit"]
    complex_delta_recall = float(
        grouped.loc["expanded_hgb", "recall_at_10pct_capacity"]
        - grouped.loc["expanded_logit", "recall_at_10pct_capacity"]
    )
    complex_delta_auprc = float(
        grouped.loc["expanded_hgb", "auprc"] - grouped.loc["expanded_logit", "auprc"]
    )
    random_optimism = float(random.loc["expanded_hgb", "auprc"] - grouped.loc["expanded_hgb", "auprc"])
    summary = f"""# Pilot summary

## Outcome

The pilot completed on {len(frame):,} students from {frame['CNTSCHID'].nunique():,} schools. The weighted prevalence of low life satisfaction was {prevalence:.3f}.

- Routine-information elastic-net: school-grouped AUPRC {routine['auprc']:.3f}; recall at 10% capacity {routine['recall_at_10pct_capacity']:.3f}.
- Adding school-experience scales: elastic-net AUPRC {expanded['auprc']:.3f}; recall at 10% capacity {expanded['recall_at_10pct_capacity']:.3f}.
- Expanded HGB minus expanded elastic-net: AUPRC {complex_delta_auprc:+.3f}; recall at 10% capacity {complex_delta_recall:+.3f}.
- Student-random minus school-grouped AUPRC for expanded HGB: {random_optimism:+.3f}.

## Interpretation boundary

This is an auxiliary/dev pilot with fixed hyperparameters. It validates the analysis path and provides directional evidence; it is not the final nested-validation estimate and does not support diagnosis, causal claims, intervention effects, or deployment.

## Next action

Proceed to the frozen main analysis if fold checks, calibration values, and subgroup counts pass independent review. If routine information remains close to the random-capacity baseline, preserve that negative result and frame the paper as an audit showing why direct universal self-report should not be replaced by visible proxies.
"""
    (RUN_DIR / "summary.md").write_text(summary, encoding="utf-8")
    claim_validation = f"""# Claim validation

| Claim | Metric | Expected direction | Pilot observation | Verdict |
|---|---|---|---|---|
| Routine visible information adds coverage beyond random capacity | recall at 10% capacity | > 0.10 | {routine['recall_at_10pct_capacity']:.3f} | {'supported directionally' if routine['recall_at_10pct_capacity'] > 0.10 else 'refuted directionally'} |
| Extra school-experience scales add information | grouped AUPRC | expanded > routine | {expanded['auprc'] - routine['auprc']:+.3f} | {'supported directionally' if expanded['auprc'] > routine['auprc'] else 'not supported'} |
| Complex model crosses descriptive point-margin | researcher-set AUPRC +0.02 or recall +0.03 | point-margin comparison only | AUPRC {complex_delta_auprc:+.3f}; recall {complex_delta_recall:+.3f} | {'point margin crossed; practical value untested' if complex_delta_auprc >= 0.02 or complex_delta_recall >= 0.03 else 'point margin not crossed'} |
| Student-random split is optimistic | AUPRC random - grouped | > 0 | {random_optimism:+.3f} | {'supported directionally' if random_optimism > 0 else 'not supported in pilot'} |

All verdicts are pilot-level and remain inconclusive for the manuscript until nested grouped validation and cluster uncertainty analysis are completed.
"""
    (RUN_DIR / "claim_validation.md").write_text(claim_validation, encoding="utf-8")

    ended = datetime.now(timezone.utc)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyreadstat", "scipy", "scikit-learn"]
        },
    }
    (RUN_DIR / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_manifest = {
        "run_id": config["run_id"],
        "stage": config["stage"],
        "status": "success",
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "command": ".venv\\Scripts\\python.exe scripts\\run_pilot.py",
        "config": str(CONFIG_PATH),
        "metric_contract": str(ROOT / "artifacts" / "baseline" / "json" / "metric_contract.json"),
        "dataset": str(DATA_PATH),
        "dataset_zip_sha256": config["data_sha256"],
        "seed": config["seed"],
        "environment": str(RUN_DIR / "environment.json"),
        "tool_note": "bash_exec and artifact.record_main_experiment were unavailable; local durable equivalents were written. This run remains auxiliary/dev.",
    }
    (RUN_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runlog_summary = "# Run log summary\n\n" + "\n".join(f"- {line}" for line in log_lines) + "\n"
    (RUN_DIR / "runlog.summary.md").write_text(runlog_summary, encoding="utf-8")

    artifact_paths = [
        path
        for path in RUN_DIR.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json"}
    ]
    artifact_manifest = {
        "run_id": config["run_id"],
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(artifact_paths)
        ],
    }
    (RUN_DIR / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("pilot run complete")


if __name__ == "__main__":
    main()
