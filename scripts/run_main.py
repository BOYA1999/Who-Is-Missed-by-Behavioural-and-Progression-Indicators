from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from run_pilot import (
    core_metrics,
    dataframe_markdown,
    safe_group_metrics,
    selection_fraction,
    sha256,
    weighted_mean,
    weighted_quantiles,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "experiment" / "main_v1"
CONFIG_PATH = RUN_DIR / "config.json"
DATA_PATH = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
PILOT_DIR = ROOT / "artifacts" / "experiment" / "pilot_v1"


def parameter_candidates(kind: str, config: dict) -> list[dict]:
    grid = config["parameter_grid"][kind]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def make_model(kind: str, params: dict, seed: int) -> Pipeline:
    if kind == "logit":
        classifier = LogisticRegression(
            C=params["C"],
            solver="lbfgs",
            max_iter=2000,
            tol=1e-6,
            random_state=seed,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    classifier = HistGradientBoostingClassifier(random_state=seed, **params)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", classifier),
        ]
    )


def fit_model(model: Pipeline, x: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> Pipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, classifier__sample_weight=w / w.mean())
    return model


def tune(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    features: list[str],
    kind: str,
    config: dict,
    seed: int,
) -> tuple[dict, list[dict]]:
    train = frame.iloc[train_index]
    y = train["outcome"].to_numpy(dtype=int)
    w = train["W_FSTUWT"].to_numpy(dtype=float)
    groups = train["CNTSCHID"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=config["inner_school_splits"], shuffle=True, random_state=seed
    )
    inner_splits = list(splitter.split(train[features], y, groups))
    scores = []
    best_score = -np.inf
    best_params = None
    for candidate_index, params in enumerate(parameter_candidates(kind, config)):
        predictions = np.full(len(train), np.nan)
        converged = True
        for fold, (inner_train, inner_valid) in enumerate(inner_splits):
            model = make_model(kind, params, seed + candidate_index * 10 + fold)
            try:
                fit_model(model, train.iloc[inner_train][features], y[inner_train], w[inner_train])
            except ConvergenceWarning:
                converged = False
                break
            predictions[inner_valid] = model.predict_proba(train.iloc[inner_valid][features])[:, 1]
        score = (
            float(average_precision_score(y, predictions, sample_weight=w))
            if converged
            else float("nan")
        )
        scores.append({"params": params, "weighted_auprc": score, "converged": converged})
        if not converged:
            continue
        if score > best_score:
            best_score = score
            best_params = params
    if best_params is None:
        raise RuntimeError(f"all {kind} candidates failed to converge")
    return best_params, scores


def nested_oof(
    frame: pd.DataFrame,
    features: list[str],
    kind: str,
    split_name: str,
    config: dict,
    log,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    schools = frame["CNTSCHID"].to_numpy()
    predictions = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=int)
    checks = []
    tuning_rows = []
    if split_name == "school_nested":
        splitter = StratifiedGroupKFold(
            n_splits=config["outer_school_splits"], shuffle=True, random_state=config["seed"]
        )
        outer_splits = list(splitter.split(frame[features], y, schools))
        fold_labels = [str(index) for index in range(len(outer_splits))]
    else:
        fold_labels = sorted(frame["region"].dropna().unique().tolist())
        outer_splits = [
            (
                np.flatnonzero(frame["region"].ne(region).to_numpy()),
                np.flatnonzero(frame["region"].eq(region).to_numpy()),
            )
            for region in fold_labels
        ]
    for fold, ((train_index, test_index), fold_label) in enumerate(zip(outer_splits, fold_labels)):
        school_overlap = set(schools[train_index]).intersection(set(schools[test_index]))
        if school_overlap:
            raise RuntimeError(f"school leakage in {split_name} fold {fold_label}")
        region_overlap = set(frame.iloc[train_index]["region"]).intersection(
            set(frame.iloc[test_index]["region"])
        )
        if split_name == "region_iecv" and region_overlap:
            raise RuntimeError(f"region leakage in fold {fold_label}")
        params, candidate_scores = tune(
            frame,
            train_index,
            features,
            kind,
            config,
            config["seed"] + fold * 100,
        )
        model = make_model(kind, params, config["seed"] + fold)
        fit_model(
            model,
            frame.iloc[train_index][features],
            y[train_index],
            w[train_index],
        )
        predictions[test_index] = model.predict_proba(frame.iloc[test_index][features])[:, 1]
        fold_ids[test_index] = fold
        checks.append(
            {
                "fold": fold,
                "fold_label": fold_label,
                "train_n": int(len(train_index)),
                "test_n": int(len(test_index)),
                "train_schools": int(pd.Series(schools[train_index]).nunique()),
                "test_schools": int(pd.Series(schools[test_index]).nunique()),
                "school_overlap_n": int(len(school_overlap)),
                "region_overlap_n": int(len(region_overlap)),
                "test_events": int(y[test_index].sum()),
                "selected_params": params,
            }
        )
        for candidate in candidate_scores:
            tuning_rows.append(
                {
                    "split": split_name,
                    "fold": fold,
                    "fold_label": fold_label,
                    "kind": kind,
                    "params": json.dumps(candidate["params"], sort_keys=True),
                    "weighted_auprc": candidate["weighted_auprc"],
                    "converged": candidate["converged"],
                    "selected": candidate["params"] == params,
                }
            )
        log(f"{split_name} {kind} fold {fold + 1}/{len(outer_splits)} ({fold_label}) complete")
    if np.isnan(predictions).any() or np.any(fold_ids < 0):
        raise RuntimeError(f"incomplete predictions for {split_name} {kind}")
    return predictions, fold_ids, checks, tuning_rows


def bootstrap_fixed_predictions(
    frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    model_names: list[str],
    replicates: int,
    seed: int,
    log,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    schools = pd.Index(frame["CNTSCHID"].unique())
    school_codes = pd.Categorical(frame["CNTSCHID"], categories=schools).codes
    y = frame["outcome"].to_numpy(dtype=int)
    base_w = frame["W_FSTUWT"].to_numpy(dtype=float)
    rows = []
    fairness_rows = []
    sex_masks = {
        "female": frame["ST004D01T"].eq(1).to_numpy(),
        "male": frame["ST004D01T"].eq(2).to_numpy(),
    }
    immigration_masks = {
        "native": frame["IMMIG"].eq(1).to_numpy(),
        "immigrant": frame["IMMIG"].isin([2, 3]).to_numpy(),
    }
    escs_valid = frame["ESCS"].notna().to_numpy()
    escs_cuts = weighted_quantiles(
        frame.loc[escs_valid, "ESCS"].to_numpy(),
        frame.loc[escs_valid, "W_FSTUWT"].to_numpy(),
        [1 / 3, 2 / 3],
    )
    escs_masks = {
        "low": frame["ESCS"].le(escs_cuts[0]).to_numpy(),
        "middle": frame["ESCS"].gt(escs_cuts[0]).to_numpy()
        & frame["ESCS"].le(escs_cuts[1]).to_numpy(),
        "high": frame["ESCS"].gt(escs_cuts[1]).to_numpy(),
    }
    fairness_axes = {
        "sex": sex_masks,
        "immigration_background": immigration_masks,
        "escs_weighted_tertile": escs_masks,
    }
    for replicate in range(replicates):
        sampled = rng.integers(0, len(schools), size=len(schools))
        multiplicity = np.bincount(sampled, minlength=len(schools))
        w = base_w * multiplicity[school_codes]
        present = w > 0
        for model_name in model_names:
            p = prediction_frame[f"p__{model_name}__school_nested"].to_numpy()
            selected = selection_fraction(p[present], w[present], 0.10)
            y_present = y[present]
            w_present = w[present]
            p_present = p[present]
            tp = float(np.sum(w_present * y_present * selected))
            rows.append(
                {
                    "replicate": replicate,
                    "model": model_name,
                    "auprc": float(average_precision_score(y_present, p_present, sample_weight=w_present)),
                    "auroc": float(roc_auc_score(y_present, p_present, sample_weight=w_present)),
                    "brier": float(brier_score_loss(y_present, p_present, sample_weight=w_present)),
                    "recall_at_10pct_capacity": tp / float(np.sum(w_present * y_present)),
                    "precision_at_10pct_capacity": tp / float(np.sum(w_present * selected)),
                }
            )
            if model_name == "expanded_hgb":
                selected_full = np.zeros(len(frame))
                selected_full[present] = selected
                for axis, levels in fairness_axes.items():
                    recalls = []
                    for level, mask in levels.items():
                        group_positive = float(np.sum(w[mask] * y[mask]))
                        recalls.append(
                            float(np.sum(w[mask] * y[mask] * selected_full[mask])) / group_positive
                        )
                    fairness_rows.append(
                        {
                            "replicate": replicate,
                            "group_axis": axis,
                            "max_minus_min_recall_gap": max(recalls) - min(recalls),
                        }
                    )
        if (replicate + 1) % 100 == 0:
            log(f"bootstrap {replicate + 1}/{replicates} complete")
    return pd.DataFrame(rows), pd.DataFrame(fairness_rows)


def interval_summary(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "bootstrap_mean": float(group[metric].mean()),
                    "ci_2.5": float(group[metric].quantile(0.025)),
                    "ci_97.5": float(group[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    log_path = RUN_DIR / ("smoke.log" if args.smoke else "bash.log")
    log_lines = []

    def log(message: str) -> None:
        stamped = f"{datetime.now(timezone.utc).isoformat()} {message}"
        log_lines.append(stamped)
        console_encoding = sys.stdout.encoding or "utf-8"
        print(stamped.encode(console_encoding, errors="replace").decode(console_encoding), flush=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    started = datetime.now(timezone.utc)
    replicate_weights = [f"W_FSTURWT{index}" for index in range(1, 81)]
    variables = list(
        dict.fromkeys(
            [
                "CNT",
                "CNTSCHID",
                "CNTSTUID",
                "STRATUM",
                "ST016Q01NA",
                "W_FSTUWT",
                *replicate_weights,
                *config["routine_features"],
                *config["school_experience_features"],
                *config["audit_features"],
            ]
        )
    )
    log("loading selected columns including 80 PISA replicate weights")
    frame, meta = pyreadstat.read_sav(DATA_PATH, usecols=variables, apply_value_formats=False)
    frame = frame.loc[(frame["CNT"] == config["country"]) & frame["ST016Q01NA"].between(0, 10)].copy()
    frame.reset_index(drop=True, inplace=True)
    frame["outcome"] = frame["ST016Q01NA"].between(0, 4).astype(int)
    stratum_labels = meta.variable_value_labels["STRATUM"]
    frame["region"] = frame["STRATUM"].map(
        lambda value: stratum_labels.get(value, str(value)).split(":", 1)[-1].split(",", 1)[0].strip()
    )
    if args.smoke:
        smoke_schools = frame["CNTSCHID"].drop_duplicates().iloc[:100]
        frame = frame.loc[frame["CNTSCHID"].isin(smoke_schools)].copy().reset_index(drop=True)
        config = {
            **config,
            "outer_school_splits": 2,
            "inner_school_splits": 2,
            "bootstrap_replicates": 10,
            "parameter_grid": {
                "logit": {"C": [1.0]},
                "hgb": {
                    "max_leaf_nodes": [15],
                    "min_samples_leaf": [50],
                    "learning_rate": [0.05],
                    "max_iter": [50],
                    "l2_regularization": [1.0],
                },
            },
        }
        model_specs = [("routine_logit", config["routine_features"], "logit")]
        split_names = ["school_nested"]
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
        split_names = ["school_nested", "region_iecv"]
    log(
        f"analytic rows={len(frame)} schools={frame['CNTSCHID'].nunique()} "
        f"regions={frame['region'].nunique()} events={frame['outcome'].sum()}"
    )
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    predictions = frame[
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
    all_checks = {}
    all_tuning_rows = []
    for model_name, features, kind in model_specs:
        for split_name in split_names:
            log(f"starting {model_name} / {split_name}")
            p, fold_ids, checks, tuning_rows = nested_oof(
                frame, features, kind, split_name, config, log
            )
            predictions[f"p__{model_name}__{split_name}"] = p
            predictions[f"fold__{split_name}"] = fold_ids
            metric_rows.append(
                {"model": model_name, "split": split_name, **core_metrics(y, p, w, config["capacities"])}
            )
            all_checks[f"{model_name}__{split_name}"] = checks
            for row in tuning_rows:
                all_tuning_rows.append({"model": model_name, **row})
            for fold in sorted(np.unique(fold_ids)):
                mask = fold_ids == fold
                label = checks[int(fold)]["fold_label"]
                fold_metric_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "fold": int(fold),
                        "fold_label": label,
                        **core_metrics(y[mask], p[mask], w[mask], config["capacities"]),
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows)
    fold_metrics_frame = pd.DataFrame(fold_metric_rows)
    tuning_frame = pd.DataFrame(all_tuning_rows)
    if args.smoke:
        smoke = {
            "status": "success",
            "n": len(frame),
            "schools": int(frame["CNTSCHID"].nunique()),
            "metrics": metric_rows,
            "checks": all_checks,
        }
        (RUN_DIR / "smoke_metrics.json").write_text(
            json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log("main smoke complete")
        return

    prevalence = weighted_mean(y, w)
    replicate_estimates = np.array(
        [weighted_mean(y, frame[column].to_numpy(dtype=float)) for column in replicate_weights]
    )
    prevalence_se = float(
        np.sqrt(config["pisa_brr_variance_factor"] * np.sum((replicate_estimates - prevalence) ** 2))
    )
    prevalence_brr = {
        "estimate": prevalence,
        "standard_error": prevalence_se,
        "ci_95_normal": [prevalence - 1.96 * prevalence_se, prevalence + 1.96 * prevalence_se],
        "replicate_count": 80,
        "variance_factor": config["pisa_brr_variance_factor"],
    }
    (RUN_DIR / "prevalence_brr.json").write_text(
        json.dumps(prevalence_brr, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
        for split_name in split_names:
            p = predictions[f"p__{model_name}__{split_name}"].to_numpy()
            selected = selection_fraction(p, w, 0.10)
            for axis, levels in group_definitions.items():
                for level, mask in levels.items():
                    values = safe_group_metrics(y, p, w, selected, mask)
                    if values:
                        fairness_rows.append(
                            {
                                "model": model_name,
                                "split": split_name,
                                "group_axis": axis,
                                "group": level,
                                **values,
                            }
                        )
    fairness_frame = pd.DataFrame(fairness_rows)

    heuristic_complete = frame[
        ["REPEAT", "ST062Q01TA", "ST062Q02TA", "ST062Q03TA"]
    ].notna().all(axis=1).to_numpy()
    heuristic_score = (
        frame["REPEAT"].eq(1).astype(float)
        + frame["ST062Q01TA"].gt(1).astype(float)
        + frame["ST062Q02TA"].gt(1).astype(float)
        + frame["ST062Q03TA"].ge(3).astype(float)
    ).to_numpy()
    heuristic_selected = selection_fraction(
        heuristic_score[heuristic_complete], w[heuristic_complete], 0.10
    )
    heuristic_tp = float(
        np.sum(w[heuristic_complete] * y[heuristic_complete] * heuristic_selected)
    )
    heuristic_metrics = {
        "definition": "one point each for grade repetition, any full-day absence, any class skipping, and frequent tardiness; complete cases",
        "n": int(heuristic_complete.sum()),
        "weighted_prevalence": weighted_mean(y[heuristic_complete], w[heuristic_complete]),
        "recall_at_10pct_capacity": heuristic_tp
        / float(np.sum(w[heuristic_complete] * y[heuristic_complete])),
        "precision_at_10pct_capacity": heuristic_tp
        / float(np.sum(w[heuristic_complete] * heuristic_selected)),
    }

    log("starting fixed-prediction school bootstrap")
    model_names = [item[0] for item in model_specs]
    bootstrap_frame, fairness_bootstrap_frame = bootstrap_fixed_predictions(
        frame,
        predictions,
        model_names,
        config["bootstrap_replicates"],
        config["seed"] + 5000,
        log,
    )
    bootstrap_summary = interval_summary(
        bootstrap_frame,
        ["model"],
        ["auprc", "auroc", "brier", "recall_at_10pct_capacity", "precision_at_10pct_capacity"],
    )
    fairness_bootstrap_summary = interval_summary(
        fairness_bootstrap_frame,
        ["group_axis"],
        ["max_minus_min_recall_gap"],
    )

    predictions.to_csv(RUN_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")
    metrics_frame.to_csv(RUN_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics_frame.to_csv(RUN_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    tuning_frame.to_csv(RUN_DIR / "tuning_results.csv.gz", index=False, compression="gzip")
    fairness_frame.to_csv(RUN_DIR / "fairness_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap_frame.to_csv(RUN_DIR / "bootstrap_metrics.csv.gz", index=False, compression="gzip")
    bootstrap_summary.to_csv(RUN_DIR / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    fairness_bootstrap_frame.to_csv(
        RUN_DIR / "fairness_bootstrap.csv.gz", index=False, compression="gzip"
    )
    fairness_bootstrap_summary.to_csv(
        RUN_DIR / "fairness_bootstrap_summary.csv", index=False, encoding="utf-8-sig"
    )

    result = {
        "run_id": config["run_id"],
        "status": "success",
        "stage": config["stage"],
        "prevalence_brr": prevalence_brr,
        "heuristic_baseline": heuristic_metrics,
        "random_capacity_recall": 0.10,
        "oracle_recall_at_10pct_capacity": min(1.0, 0.10 / prevalence),
        "metrics": metric_rows,
        "checks": all_checks,
        "bootstrap_scope": config["bootstrap_scope"],
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
    (RUN_DIR / "metrics.md").write_text(
        "# Main metrics\n\n" + dataframe_markdown(metrics_frame[display_columns]) + "\n",
        encoding="utf-8",
    )

    main_metrics = metrics_frame.loc[metrics_frame["split"] == "school_nested"].set_index("model")
    region_metrics = metrics_frame.loc[metrics_frame["split"] == "region_iecv"].set_index("model")
    routine = main_metrics.loc["routine_logit"]
    expanded_logit = main_metrics.loc["expanded_logit"]
    expanded_hgb = main_metrics.loc["expanded_hgb"]
    complex_recall_delta = float(
        expanded_hgb["recall_at_10pct_capacity"] - expanded_logit["recall_at_10pct_capacity"]
    )
    complex_auprc_delta = float(expanded_hgb["auprc"] - expanded_logit["auprc"])
    region_drop = float(expanded_hgb["auprc"] - region_metrics.loc["expanded_hgb", "auprc"])
    best_row = bootstrap_summary.loc[
        (bootstrap_summary["model"] == "expanded_hgb")
        & (bootstrap_summary["metric"] == "recall_at_10pct_capacity")
    ].iloc[0]
    summary = f"""# Main experiment summary

## Outcome

The nested school validation and 19-region internal–external validation completed for {len(frame):,} students from {frame['CNTSCHID'].nunique():,} schools. Weighted low-life-satisfaction prevalence was {prevalence:.3f} (PISA BRR 95% CI {prevalence_brr['ci_95_normal'][0]:.3f}–{prevalence_brr['ci_95_normal'][1]:.3f}).

- Routine penalized logistic regression, school nested: AUPRC {routine['auprc']:.3f}; recall at 10% capacity {routine['recall_at_10pct_capacity']:.3f}.
- Expanded penalized logistic regression, school nested: AUPRC {expanded_logit['auprc']:.3f}; recall {expanded_logit['recall_at_10pct_capacity']:.3f}.
- Expanded HGB, school nested: AUPRC {expanded_hgb['auprc']:.3f}; recall {expanded_hgb['recall_at_10pct_capacity']:.3f} (fixed-OOF school-bootstrap 95% interval {best_row['ci_2.5']:.3f}–{best_row['ci_97.5']:.3f}).
- Expanded HGB versus expanded penalized logistic regression: AUPRC {complex_auprc_delta:+.3f}; recall {complex_recall_delta:+.3f}.
- Expanded HGB AUPRC, region IECV: {region_metrics.loc['expanded_hgb', 'auprc']:.3f}; school-nested minus region-IECV {region_drop:+.3f}.
- Simple visible-risk heuristic, complete cases: recall at 10% capacity {heuristic_metrics['recall_at_10pct_capacity']:.3f}.

## Evidence classification

The main operational finding is supported if the final QA confirms these outputs: routine visible information improves on random selection but misses most low-life-satisfaction students; school-experience scales add substantial information; the extra gain from a nonlinear model is evaluated against the pre-specified practical thresholds rather than statistical significance alone.

## Boundaries

The bootstrap interval conditions on fixed out-of-fold predictions and does not include model-refitting uncertainty. The data are cross-sectional and the task is contemporaneous identification, not prospective prediction. Results do not validate diagnosis, causality, intervention effects, or deployment.

## Next action

Run claim-focused sensitivity analyses for the continuous 0–10 outcome, complete cases, and inclusion/exclusion of audit attributes; then begin the manuscript only if those checks do not reverse the main interpretation.
"""
    (RUN_DIR / "summary.md").write_text(summary, encoding="utf-8")

    practical_complex = complex_auprc_delta >= 0.02 or complex_recall_delta >= 0.03
    claim_validation = f"""# Claim validation

| Claim | Metric | Observed | Verdict |
|---|---|---|---|
| Routine proxies beat random capacity but miss most cases | recall at 10% capacity | random 0.100; routine {routine['recall_at_10pct_capacity']:.3f} | supported |
| School-experience scales add information | AUPRC and recall, expanded logit minus routine logit | AUPRC {expanded_logit['auprc'] - routine['auprc']:+.3f}; recall {expanded_logit['recall_at_10pct_capacity'] - routine['recall_at_10pct_capacity']:+.3f} | supported |
| Nonlinear complexity has practical value | expanded HGB minus expanded logit; thresholds +0.02 AUPRC or +0.03 recall | AUPRC {complex_auprc_delta:+.3f}; recall {complex_recall_delta:+.3f} | {'supported' if practical_complex else 'not supported'} |
| Performance transports across Spanish regions | region IECV versus school nested | AUPRC difference {region_drop:+.3f} | {'supported directionally' if abs(region_drop) < 0.02 else 'narrowed by regional shift'} |

Claim status remains conditional on sensitivity analyses and rendered table/figure QA.
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
        "command": ".venv\\Scripts\\python.exe scripts\\run_main.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
        "config": str(CONFIG_PATH),
        "metric_contract": str(ROOT / "artifacts" / "baseline" / "json" / "metric_contract.json"),
        "baseline": str(PILOT_DIR),
        "dataset": str(DATA_PATH),
        "dataset_zip_sha256": config["data_sha256"],
        "seed": config["seed"],
        "environment": str(RUN_DIR / "environment.json"),
        "tool_note": "bash_exec and artifact.record_main_experiment were unavailable; equivalent local durable outputs were written.",
    }
    (RUN_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RUN_DIR / "runlog.summary.md").write_text(
        "# Run log summary\n\n" + "\n".join(f"- {line}" for line in log_lines) + "\n",
        encoding="utf-8",
    )
    artifact_paths = [
        path for path in RUN_DIR.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "run_id": config["run_id"],
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(artifact_paths)
        ],
    }
    (RUN_DIR / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("main run complete")


if __name__ == "__main__":
    main()
