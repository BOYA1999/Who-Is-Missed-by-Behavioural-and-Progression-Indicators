from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from run_main import fit_model, make_model, tune
from run_pilot import (
    calibration_metrics,
    core_metrics,
    selection_fraction,
    weighted_mean,
    weighted_quantiles,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "reviewer_revision_v1"
DATA = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
MAIN = ROOT / "artifacts" / "experiment" / "main_v1"
CAMPAIGN = ROOT / "artifacts" / "analysis" / "campaign_v1"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
MAIN_CONFIG = json.loads((MAIN / "config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary(series: pd.Series, positive: set[float], valid: set[float]) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    mask = series.isin(valid)
    result.loc[mask] = series.loc[mask].isin(positive).astype(float)
    return result


def brr_interval(full: float, replicates: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(replicates, dtype=float)
    se = float(np.sqrt(CONFIG["pisa_brr_variance_factor"] * np.sum((values - full) ** 2)))
    return se, full - 1.96 * se, full + 1.96 * se


def fixed_outer_logit(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    log,
    fixed_params: dict[int, dict] | None = None,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    folds = frame["fold__school_nested"].to_numpy(dtype=int)
    predictions = np.full(len(frame), np.nan)
    coefficient_rows = []
    tuning_rows = []
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        test_index = np.flatnonzero(folds == fold)
        if fixed_params is None:
            params, candidates = tune(
                frame,
                train_index,
                features,
                "logit",
                MAIN_CONFIG,
                MAIN_CONFIG["seed"] + int(fold) * 100,
            )
        else:
            params = fixed_params[int(fold)]
            candidates = []
        model = make_model("logit", params, MAIN_CONFIG["seed"] + int(fold))
        fit_model(model, frame.iloc[train_index][features], y[train_index], w[train_index])
        predictions[test_index] = model.predict_proba(frame.iloc[test_index][features])[:, 1]
        imputer = model.named_steps["imputer"]
        scaler = model.named_steps["scaler"]
        classifier = model.named_steps["classifier"]
        transformed_names = imputer.get_feature_names_out(features)
        medians = dict(zip(features, imputer.statistics_))
        for index, transformed_name in enumerate(transformed_names):
            coefficient_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "selected_C": params["C"],
                    "term": transformed_name,
                    "coefficient_standardized": float(classifier.coef_[0, index]),
                    "scaler_mean": float(scaler.mean_[index]),
                    "scaler_scale": float(scaler.scale_[index]),
                    "imputation_median": float(medians[transformed_name])
                    if transformed_name in medians and np.isfinite(medians[transformed_name])
                    else np.nan,
                }
            )
        for candidate in candidates:
            tuning_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "params": json.dumps(candidate["params"], sort_keys=True),
                    "weighted_average_precision": candidate["weighted_auprc"],
                    "selected": candidate["params"] == params,
                }
            )
        log(f"{model_name} fold {fold + 1}/5 complete")
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"non-finite predictions for {model_name}")
    return predictions, coefficient_rows, tuning_rows


def hgb_no_earlystop(
    frame: pd.DataFrame,
    params_by_fold: dict[int, dict],
    log,
) -> tuple[np.ndarray, list[dict]]:
    features = MAIN_CONFIG["routine_features"] + MAIN_CONFIG["school_experience_features"]
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    folds = frame["fold__school_nested"].to_numpy(dtype=int)
    predictions = np.full(len(frame), np.nan)
    rows = []
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        test_index = np.flatnonzero(folds == fold)
        params = params_by_fold[int(fold)]
        classifier = HistGradientBoostingClassifier(
            random_state=MAIN_CONFIG["seed"] + int(fold),
            early_stopping=False,
            **params,
        )
        model = Pipeline(
            [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("classifier", classifier)]
        )
        fit_model(model, frame.iloc[train_index][features], y[train_index], w[train_index])
        predictions[test_index] = model.predict_proba(frame.iloc[test_index][features])[:, 1]
        rows.append(
            {
                "fold": int(fold),
                "early_stopping": False,
                "n_iter": int(model.named_steps["classifier"].n_iter_),
                "params": json.dumps(params, sort_keys=True),
            }
        )
        log(f"expanded_hgb_no_earlystop fold {fold + 1}/5 complete")
    return predictions, rows


def performance_bootstrap(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    log,
) -> pd.DataFrame:
    rng = np.random.default_rng(CONFIG["seed"] + 1000)
    schools = pd.Index(frame["CNTSCHID"].unique())
    school_codes = pd.Categorical(frame["CNTSCHID"], categories=schools).codes
    y = frame["outcome"].to_numpy(dtype=int)
    base_w = frame["W_FSTUWT"].to_numpy(dtype=float)
    rows = []
    for replicate in range(CONFIG["bootstrap_replicates"]):
        sampled = rng.integers(0, len(schools), len(schools))
        multiplicity = np.bincount(sampled, minlength=len(schools))
        w = base_w * multiplicity[school_codes]
        present = w > 0
        for model_name, p in predictions.items():
            selected = selection_fraction(p[present], w[present], CONFIG["capacity"])
            yp = y[present]
            wp = w[present]
            tp = float(np.sum(wp * yp * selected))
            rows.append(
                {
                    "replicate": replicate,
                    "model": model_name,
                    "weighted_average_precision": float(
                        average_precision_score(yp, p[present], sample_weight=wp)
                    ),
                    "recall_at_10pct_capacity": tp / float(np.sum(wp * yp)),
                    "precision_at_10pct_capacity": tp / float(np.sum(wp * selected)),
                    "selected_weight_fraction": float(np.sum(wp * selected) / np.sum(wp)),
                }
            )
        if (replicate + 1) % 100 == 0:
            log(f"school bootstrap {replicate + 1}/{CONFIG['bootstrap_replicates']} complete")
    return pd.DataFrame(rows)


def delta_summary(
    replicate_metrics: pd.DataFrame,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    wide = replicate_metrics.pivot(index="replicate", columns="model")
    rows = []
    for model, reference in comparisons:
        for metric in ["weighted_average_precision", "recall_at_10pct_capacity", "precision_at_10pct_capacity"]:
            delta = wide[(metric, model)] - wide[(metric, reference)]
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": metric,
                    "bootstrap_mean_delta": float(delta.mean()),
                    "ci_2.5": float(delta.quantile(0.025)),
                    "ci_97.5": float(delta.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def brr_delta_summary(
    replicate_metrics: pd.DataFrame,
    full_metrics: pd.DataFrame,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    wide = replicate_metrics.pivot(index="replicate", columns="model")
    full = full_metrics.set_index("model")
    metric_map = {
        "weighted_average_precision": "auprc",
        "recall_at_10pct_capacity": "recall_at_10pct_capacity",
        "precision_at_10pct_capacity": "precision_at_10pct_capacity",
    }
    rows = []
    for model, reference in comparisons:
        for metric, full_column in metric_map.items():
            estimate = float(full.loc[model, full_column] - full.loc[reference, full_column])
            replicate_delta = wide[(metric, model)] - wide[(metric, reference)]
            se, lower, upper = brr_interval(estimate, replicate_delta.to_numpy())
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": metric,
                    "estimate_delta": estimate,
                    "brr_se": se,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def brr_performance(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    full_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["outcome"].to_numpy(dtype=int)
    replicate_rows = []
    for replicate in range(1, 81):
        w = frame[f"W_FSTURWT{replicate}"].to_numpy(dtype=float)
        for model_name, p in predictions.items():
            selected = selection_fraction(p, w, CONFIG["capacity"])
            tp = float(np.sum(w * y * selected))
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "model": model_name,
                    "weighted_average_precision": float(average_precision_score(y, p, sample_weight=w)),
                    "recall_at_10pct_capacity": tp / float(np.sum(w * y)),
                    "precision_at_10pct_capacity": tp / float(np.sum(w * selected)),
                    "selected_weight_fraction": float(np.sum(w * selected) / np.sum(w)),
                }
            )
    replicates = pd.DataFrame(replicate_rows)
    rows = []
    full = full_metrics.set_index("model")
    metric_map = {
        "weighted_average_precision": "auprc",
        "recall_at_10pct_capacity": "recall_at_10pct_capacity",
        "precision_at_10pct_capacity": "precision_at_10pct_capacity",
        "selected_weight_fraction": "selected_weight_fraction_at_10pct_capacity",
    }
    for model_name in predictions:
        subset = replicates.loc[replicates["model"].eq(model_name)]
        for metric, full_column in metric_map.items():
            estimate = float(full.loc[model_name, full_column])
            se, lower, upper = brr_interval(estimate, subset[metric].to_numpy())
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "estimate": estimate,
                    "brr_se": se,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return replicates, pd.DataFrame(rows)


def overlap_membership(routine: np.ndarray, expanded: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "both": routine * expanded,
        "routine_only": routine * (1.0 - expanded),
        "expanded_only": (1.0 - routine) * expanded,
        "neither": (1.0 - routine) * (1.0 - expanded),
    }


def overlap_table(
    frame: pd.DataFrame,
    routine_p: np.ndarray,
    expanded_p: np.ndarray,
    weight_column: str,
) -> pd.DataFrame:
    w = frame[weight_column].to_numpy(dtype=float)
    y = frame["outcome"].to_numpy(dtype=int)
    routine = selection_fraction(routine_p, w, CONFIG["capacity"])
    expanded = selection_fraction(expanded_p, w, CONFIG["capacity"])
    memberships = overlap_membership(routine, expanded)
    rows = []
    for population, mask in [("all_students", np.ones(len(frame), dtype=bool)), ("low_life_satisfaction", y == 1)]:
        denominator = float(np.sum(w[mask]))
        for cell, membership in memberships.items():
            rows.append(
                {
                    "population": population,
                    "cell": cell,
                    "weighted_share": float(np.sum(w[mask] * membership[mask]) / denominator),
                }
            )
        rows.append(
            {
                "population": population,
                "cell": "union",
                "weighted_share": float(
                    np.sum(w[mask] * (1.0 - memberships["neither"][mask])) / denominator
                ),
            }
        )
    return pd.DataFrame(rows)


def replacement_profiles(
    frame: pd.DataFrame,
    routine_p: np.ndarray,
    expanded_p: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    y = frame["outcome"].to_numpy(dtype=int)
    routine = selection_fraction(routine_p, w, CONFIG["capacity"])
    expanded = selection_fraction(expanded_p, w, CONFIG["capacity"])
    low = y == 1
    memberships = {key: value[low] for key, value in overlap_membership(routine, expanded).items()}
    low_frame = frame.loc[low].reset_index(drop=True)
    low_w = low_frame["W_FSTUWT"].to_numpy(dtype=float)
    schools = pd.Index(frame["CNTSCHID"].unique())
    school_codes = pd.Categorical(low_frame["CNTSCHID"], categories=schools).codes
    rng = np.random.default_rng(CONFIG["seed"] + 2000)
    counts = rng.multinomial(
        len(schools), np.full(len(schools), 1.0 / len(schools)), size=CONFIG["bootstrap_replicates"]
    )
    variables = [
        "ST016Q01NA",
        "BELONG",
        "BULLIED",
        "TEACHSUP",
        "ESCS",
        "female",
        "immigrant",
        "repeated_grade",
        "any_whole_day_skipping",
        "any_class_skipping",
        "any_tardiness",
    ]
    rows = []
    boots = {}
    for group, membership in memberships.items():
        for variable in variables:
            values = low_frame[variable].to_numpy(dtype=float)
            valid = np.isfinite(values)
            effective = low_w * membership * valid
            denominator = np.bincount(school_codes, weights=effective, minlength=len(schools))
            numerator = np.bincount(
                school_codes,
                weights=effective * np.nan_to_num(values, nan=0.0),
                minlength=len(schools),
            )
            estimate = float(numerator.sum() / denominator.sum())
            boot = (counts @ numerator) / (counts @ denominator)
            boots[(group, variable)] = boot
            rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "estimate": estimate,
                    "ci_lower": float(np.quantile(boot, 0.025)),
                    "ci_upper": float(np.quantile(boot, 0.975)),
                }
            )
    contrasts = []
    estimates = pd.DataFrame(rows).set_index(["group", "variable"])
    for variable in variables:
        delta = boots[("expanded_only", variable)] - boots[("routine_only", variable)]
        contrasts.append(
            {
                "contrast": "expanded_only_minus_routine_only",
                "variable": variable,
                "difference": float(
                    estimates.loc[("expanded_only", variable), "estimate"]
                    - estimates.loc[("routine_only", variable), "estimate"]
                ),
                "ci_lower": float(np.quantile(delta, 0.025)),
                "ci_upper": float(np.quantile(delta, 0.975)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def weighted_stat(frame: pd.DataFrame, value: str, mask: np.ndarray) -> tuple[int, float, float]:
    valid = mask & frame[value].notna().to_numpy()
    values = frame.loc[valid, value].to_numpy(dtype=float)
    weights = frame.loc[valid, "W_FSTUWT"].to_numpy(dtype=float)
    return int(valid.sum()), float(weights.sum()), weighted_mean(values, weights)


def inclusion_audit(all_spain: pd.DataFrame) -> pd.DataFrame:
    included = all_spain["ST016Q01NA"].between(0, 10).to_numpy()
    rows = []
    for group, mask in [("included", included), ("excluded_invalid_or_missing_outcome", ~included)]:
        for variable in [
            "female",
            "immigrant",
            "ESCS",
            "repeated_grade",
            "any_whole_day_skipping",
            "any_class_skipping",
            "any_tardiness",
            "BELONG",
            "BULLIED",
            "TEACHSUP",
        ]:
            n_valid, valid_weight, estimate = weighted_stat(all_spain, variable, mask)
            rows.append(
                {
                    "group": group,
                    "group_n": int(mask.sum()),
                    "variable": variable,
                    "valid_n": n_valid,
                    "valid_weight": valid_weight,
                    "weighted_mean_or_proportion": estimate,
                }
            )
    return pd.DataFrame(rows)


def score_miss_profile(
    frame: pd.DataFrame,
    routine_p: np.ndarray,
    expanded_p: np.ndarray,
) -> pd.DataFrame:
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    score = frame["ST016Q01NA"].to_numpy(dtype=float)
    routine = selection_fraction(routine_p, w, CONFIG["capacity"])
    expanded = selection_fraction(expanded_p, w, CONFIG["capacity"])
    neither = (1.0 - routine) * (1.0 - expanded)
    total_neither_low = float(np.sum(w[(score <= 4)] * neither[(score <= 4)]))
    rows = []
    for value in range(5):
        mask = score == value
        rows.append(
            {
                "life_satisfaction_score": value,
                "unweighted_n": int(mask.sum()),
                "routine_miss_rate": float(np.sum(w[mask] * (1.0 - routine[mask])) / np.sum(w[mask])),
                "expanded_miss_rate": float(np.sum(w[mask] * (1.0 - expanded[mask])) / np.sum(w[mask])),
                "neither_selected_rate": float(np.sum(w[mask] * neither[mask]) / np.sum(w[mask])),
                "share_of_neither_low_group": float(np.sum(w[mask] * neither[mask]) / total_neither_low),
            }
        )
    return pd.DataFrame(rows)


def reference_profiles(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mask = frame["ST016Q01NA"].between(5, 10).to_numpy()
    for variable in profiles["variable"].unique():
        _, _, estimate = weighted_stat(frame, variable, mask)
        rows.append({"group": "life_satisfaction_5_to_10", "variable": variable, "estimate": estimate})
    return pd.concat([profiles, pd.DataFrame(rows)], ignore_index=True, sort=False)


def group_metrics(y, p, w, selected, mask) -> dict:
    intercept, slope = calibration_metrics(y[mask], p[mask], w[mask])
    positives = float(np.sum(w[mask] * y[mask]))
    chosen = float(np.sum(w[mask] * selected[mask]))
    true_positive = float(np.sum(w[mask] * y[mask] * selected[mask]))
    return {
        "weighted_prevalence": weighted_mean(y[mask], w[mask]),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "recall": true_positive / positives,
        "precision": true_positive / chosen,
        "selection_rate": chosen / float(np.sum(w[mask])),
    }


def sex_calibration(frame: pd.DataFrame, p: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["outcome"].to_numpy(dtype=int)
    full_w = frame["W_FSTUWT"].to_numpy(dtype=float)
    groups = {"female": frame["ST004D01T"].eq(1).to_numpy(), "male": frame["ST004D01T"].eq(2).to_numpy()}
    full_selected = selection_fraction(p, full_w, CONFIG["capacity"])
    summary_rows = []
    curve_rows = []
    for group, mask in groups.items():
        full_values = group_metrics(y, p, full_w, full_selected, mask)
        replicate_values = {metric: [] for metric in full_values}
        for replicate in range(1, 81):
            w = frame[f"W_FSTURWT{replicate}"].to_numpy(dtype=float)
            selected = selection_fraction(p, w, CONFIG["capacity"])
            values = group_metrics(y, p, w, selected, mask)
            for metric, value in values.items():
                replicate_values[metric].append(value)
        for metric, estimate in full_values.items():
            se, lower, upper = brr_interval(estimate, np.array(replicate_values[metric]))
            summary_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "estimate": estimate,
                    "brr_se": se,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
        cuts = weighted_quantiles(p[mask], full_w[mask], [index / 8 for index in range(1, 8)])
        bins = np.digitize(p, np.unique(cuts), right=True)
        for bin_id in sorted(np.unique(bins[mask])):
            cell = mask & (bins == bin_id)
            predicted = weighted_mean(p[cell], full_w[cell])
            observed = weighted_mean(y[cell], full_w[cell])
            pred_rep = []
            obs_rep = []
            for replicate in range(1, 81):
                w = frame[f"W_FSTURWT{replicate}"].to_numpy(dtype=float)
                pred_rep.append(weighted_mean(p[cell], w[cell]))
                obs_rep.append(weighted_mean(y[cell], w[cell]))
            _, pred_lower, pred_upper = brr_interval(predicted, np.array(pred_rep))
            _, obs_lower, obs_upper = brr_interval(observed, np.array(obs_rep))
            curve_rows.append(
                {
                    "group": group,
                    "bin": int(bin_id + 1),
                    "n": int(cell.sum()),
                    "mean_predicted": predicted,
                    "predicted_ci_lower": pred_lower,
                    "predicted_ci_upper": pred_upper,
                    "observed_rate": observed,
                    "observed_ci_lower": obs_lower,
                    "observed_ci_upper": obs_upper,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def audit_attribute_table(frame: pd.DataFrame, main_p: np.ndarray) -> pd.DataFrame:
    campaign = pd.read_csv(CAMPAIGN / "sensitivity_predictions.csv.gz")
    audit = frame[["CNTSTUID"]].merge(
        campaign[["CNTSTUID", "audit_attributes__expanded_audit_logit"]],
        on="CNTSTUID",
        how="left",
        validate="one_to_one",
    )["audit_attributes__expanded_audit_logit"].to_numpy(dtype=float)
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    axes = {
        "sex": {"female": frame["ST004D01T"].eq(1).to_numpy(), "male": frame["ST004D01T"].eq(2).to_numpy()},
        "immigration_background": {
            "native": frame["IMMIG"].eq(1).to_numpy(),
            "immigrant": frame["IMMIG"].isin([2, 3]).to_numpy(),
        },
    }
    valid_escs = frame["ESCS"].notna().to_numpy()
    cuts = weighted_quantiles(frame.loc[valid_escs, "ESCS"].to_numpy(), w[valid_escs], [1 / 3, 2 / 3])
    axes["escs_weighted_tertile"] = {
        "low": frame["ESCS"].le(cuts[0]).to_numpy(),
        "middle": (frame["ESCS"].gt(cuts[0]) & frame["ESCS"].le(cuts[1])).to_numpy(),
        "high": frame["ESCS"].gt(cuts[1]).to_numpy(),
    }
    rows = []
    for model_name, p in [("expanded_logit_main", main_p), ("expanded_audit_logit", audit)]:
        selected = selection_fraction(p, w, CONFIG["capacity"])
        for axis, levels in axes.items():
            for group, mask in levels.items():
                rows.append(
                    {
                        "model": model_name,
                        "group_axis": axis,
                        "group": group,
                        "n": int(mask.sum()),
                        **group_metrics(y, p, w, selected, mask),
                    }
                )
    return pd.DataFrame(rows)


def region_uncertainty(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    y = frame["outcome"].to_numpy(dtype=int)
    rows = []
    for region in sorted(frame["region"].dropna().unique()):
        mask = frame["region"].eq(region).to_numpy()
        full_w = frame.loc[mask, "W_FSTUWT"].to_numpy(dtype=float)
        selected = selection_fraction(p[mask], full_w, CONFIG["capacity"])
        recall = float(np.sum(full_w * y[mask] * selected) / np.sum(full_w * y[mask]))
        intercept, slope = calibration_metrics(y[mask], p[mask], full_w)
        replicate_recall = []
        replicate_intercept = []
        replicate_slope = []
        for replicate in range(1, 81):
            w = frame.loc[mask, f"W_FSTURWT{replicate}"].to_numpy(dtype=float)
            chosen = selection_fraction(p[mask], w, CONFIG["capacity"])
            replicate_recall.append(float(np.sum(w * y[mask] * chosen) / np.sum(w * y[mask])))
            ci, cs = calibration_metrics(y[mask], p[mask], w)
            replicate_intercept.append(ci)
            replicate_slope.append(cs)
        recall_se, recall_lower, recall_upper = brr_interval(recall, np.array(replicate_recall))
        intercept_se, intercept_lower, intercept_upper = brr_interval(intercept, np.array(replicate_intercept))
        slope_se, slope_lower, slope_upper = brr_interval(slope, np.array(replicate_slope))
        rows.append(
            {
                "region": region,
                "students": int(mask.sum()),
                "schools": int(frame.loc[mask, "CNTSCHID"].nunique()),
                "low_life_satisfaction_cases": int(y[mask].sum()),
                "weighted_prevalence": weighted_mean(y[mask], full_w),
                "recall": recall,
                "recall_brr_se": recall_se,
                "recall_ci_lower": recall_lower,
                "recall_ci_upper": recall_upper,
                "calibration_intercept": intercept,
                "intercept_brr_se": intercept_se,
                "intercept_ci_lower": intercept_lower,
                "intercept_ci_upper": intercept_upper,
                "calibration_slope": slope,
                "slope_brr_se": slope_se,
                "slope_ci_lower": slope_lower,
                "slope_ci_upper": slope_upper,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_lines = []

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        log_lines.append(line)
        print(line, flush=True)
        (OUT / "analysis.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

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
                *MAIN_CONFIG["routine_features"],
                *MAIN_CONFIG["school_experience_features"],
                *MAIN_CONFIG["audit_features"],
                *CONFIG["alternative_bullying_items"],
            ]
        )
    )
    log("loading Spain records and PISA replicate weights")
    all_spain, meta = pyreadstat.read_sav(DATA, usecols=variables, apply_value_formats=False)
    all_spain = all_spain.loc[all_spain["CNT"].eq(CONFIG["country"])].copy().reset_index(drop=True)
    all_spain["female"] = binary(all_spain["ST004D01T"], {1.0}, {1.0, 2.0})
    all_spain["immigrant"] = binary(all_spain["IMMIG"], {2.0, 3.0}, {1.0, 2.0, 3.0})
    all_spain["repeated_grade"] = binary(all_spain["REPEAT"], {1.0}, {0.0, 1.0})
    for source, target in [
        ("ST062Q01TA", "any_whole_day_skipping"),
        ("ST062Q02TA", "any_class_skipping"),
        ("ST062Q03TA", "any_tardiness"),
    ]:
        all_spain[target] = binary(all_spain[source], {2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0})
    item_values = all_spain[CONFIG["alternative_bullying_items"]].where(
        all_spain[CONFIG["alternative_bullying_items"]].apply(lambda column: column.between(1, 4))
    )
    item_values = item_values - 1.0
    valid_items = item_values.notna().sum(axis=1)
    all_spain[CONFIG["alternative_bullying_name"]] = item_values.mean(axis=1).where(
        valid_items >= CONFIG["alternative_bullying_min_valid_items"]
    )
    frame = all_spain.loc[all_spain["ST016Q01NA"].between(0, 10)].copy().reset_index(drop=True)
    frame["outcome"] = frame["ST016Q01NA"].between(0, 4).astype(int)
    stratum_labels = meta.variable_value_labels["STRATUM"]
    frame["region"] = frame["STRATUM"].map(
        lambda value: stratum_labels.get(value, str(value)).split(":", 1)[-1].split(",", 1)[0].strip()
    )
    oof = pd.read_csv(MAIN / "oof_predictions.csv.gz")
    needed_oof = [
        "CNTSTUID",
        "fold__school_nested",
        "p__routine_logit__school_nested",
        "p__expanded_logit__school_nested",
        "p__expanded_hgb__school_nested",
        "p__expanded_logit__region_iecv",
    ]
    frame = frame.merge(oof[needed_oof], on="CNTSTUID", how="left", validate="one_to_one")
    if len(frame) != 29588 or frame["fold__school_nested"].isna().any():
        raise RuntimeError("analytic sample or frozen-fold merge mismatch")
    frame["fold__school_nested"] = frame["fold__school_nested"].astype(int)
    log(f"analytic rows={len(frame)} schools={frame['CNTSCHID'].nunique()}")

    tuning = pd.read_csv(MAIN / "tuning_results.csv.gz")
    selected = tuning.loc[tuning["selected"].astype(bool) & tuning["split"].eq("school_nested")]
    params = {
        model: {
            int(row.fold): json.loads(row.params)
            for row in selected.loc[selected["model"].eq(model)].itertuples(index=False)
        }
        for model in ["routine_logit", "expanded_logit", "expanded_hgb"]
    }
    model_features = {
        "routine_logit_refit": MAIN_CONFIG["routine_features"],
        "routine_plus_belong_logit": MAIN_CONFIG["routine_features"] + ["BELONG"],
        "routine_plus_bullied_logit": MAIN_CONFIG["routine_features"] + ["BULLIED"],
        "routine_plus_teachsup_logit": MAIN_CONFIG["routine_features"] + ["TEACHSUP"],
        "experiences_only_logit": MAIN_CONFIG["school_experience_features"],
        "expanded_logit_refit": MAIN_CONFIG["routine_features"] + MAIN_CONFIG["school_experience_features"],
        "expanded_altbully_logit": MAIN_CONFIG["routine_features"]
        + ["BELONG", CONFIG["alternative_bullying_name"], "TEACHSUP"],
    }
    predictions = {
        "routine_logit": frame["p__routine_logit__school_nested"].to_numpy(dtype=float),
        "expanded_logit": frame["p__expanded_logit__school_nested"].to_numpy(dtype=float),
        "expanded_hgb": frame["p__expanded_hgb__school_nested"].to_numpy(dtype=float),
    }
    coefficient_rows = []
    tuning_rows = []
    for model_name, features in model_features.items():
        fixed = None
        if model_name == "routine_logit_refit":
            fixed = params["routine_logit"]
        elif model_name == "expanded_logit_refit":
            fixed = params["expanded_logit"]
        p, coefficients, model_tuning = fixed_outer_logit(frame, features, model_name, log, fixed)
        predictions[model_name] = p
        coefficient_rows.extend(coefficients)
        tuning_rows.extend(model_tuning)
    predictions["expanded_hgb_no_earlystop"], hgb_rows = hgb_no_earlystop(
        frame, params["expanded_hgb"], log
    )
    refit_differences = {
        "routine_logit": float(np.max(np.abs(predictions["routine_logit_refit"] - predictions["routine_logit"]))),
        "expanded_logit": float(np.max(np.abs(predictions["expanded_logit_refit"] - predictions["expanded_logit"]))),
    }
    if max(refit_differences.values()) > 1e-10:
        raise RuntimeError(f"baseline refit mismatch: {refit_differences}")

    reported_predictions = {
        key: value
        for key, value in predictions.items()
        if key not in {"routine_logit_refit", "expanded_logit_refit"}
    }
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    metric_rows = []
    for model_name, p in reported_predictions.items():
        metric_rows.append(
            {"model": model_name, **core_metrics(y, p, w, MAIN_CONFIG["capacities"])}
        )
    metrics = pd.DataFrame(metric_rows)
    bootstrap = performance_bootstrap(frame, reported_predictions, log)
    comparisons = [
        ("routine_plus_belong_logit", "routine_logit"),
        ("routine_plus_bullied_logit", "routine_logit"),
        ("routine_plus_teachsup_logit", "routine_logit"),
        ("experiences_only_logit", "routine_logit"),
        ("expanded_logit", "routine_logit"),
        ("expanded_logit", "experiences_only_logit"),
        ("expanded_altbully_logit", "routine_logit"),
        ("expanded_altbully_logit", "expanded_logit"),
        ("expanded_hgb_no_earlystop", "expanded_hgb"),
    ]
    bootstrap_deltas = delta_summary(bootstrap, comparisons)
    brr_replicates, brr_summary = brr_performance(frame, reported_predictions, metrics)
    brr_deltas = brr_delta_summary(brr_replicates, metrics, comparisons)

    overlap = overlap_table(
        frame, predictions["routine_logit"], predictions["expanded_logit"], "W_FSTUWT"
    )
    overlap_brr_parts = []
    for replicate in range(1, 81):
        part = overlap_table(
            frame,
            predictions["routine_logit"],
            predictions["expanded_logit"],
            f"W_FSTURWT{replicate}",
        )
        part["replicate"] = replicate
        overlap_brr_parts.append(part)
    overlap_brr = pd.concat(overlap_brr_parts, ignore_index=True)
    overlap_ci_rows = []
    for row in overlap.itertuples(index=False):
        rep = overlap_brr.loc[
            overlap_brr["population"].eq(row.population) & overlap_brr["cell"].eq(row.cell),
            "weighted_share",
        ].to_numpy()
        se, lower, upper = brr_interval(row.weighted_share, rep)
        overlap_ci_rows.append(
            {
                "population": row.population,
                "cell": row.cell,
                "weighted_share": row.weighted_share,
                "brr_se": se,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    overlap_ci = pd.DataFrame(overlap_ci_rows)
    profiles, contrasts = replacement_profiles(
        frame, predictions["routine_logit"], predictions["expanded_logit"]
    )
    score_profile = score_miss_profile(
        frame, predictions["routine_logit"], predictions["expanded_logit"]
    )
    references = reference_profiles(frame, profiles)
    sex_summary, sex_curve = sex_calibration(frame, predictions["expanded_logit"])
    audit_table = audit_attribute_table(frame, predictions["expanded_logit"])
    region = region_uncertainty(
        frame, frame["p__expanded_logit__region_iecv"].to_numpy(dtype=float)
    )
    inclusion = inclusion_audit(all_spain)

    predictions_out = frame[["CNTSTUID", "CNTSCHID", "fold__school_nested"]].copy()
    for model_name, p in reported_predictions.items():
        predictions_out[model_name] = p
    predictions_out.to_csv(OUT / "revision_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(OUT / "logistic_coefficients.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "ablation_tuning.csv", index=False)
    pd.DataFrame(hgb_rows).to_csv(OUT / "hgb_no_earlystop_folds.csv", index=False)
    bootstrap.to_csv(OUT / "bootstrap_metrics.csv.gz", index=False, compression="gzip")
    bootstrap_deltas.to_csv(OUT / "bootstrap_deltas.csv", index=False)
    brr_replicates.to_csv(OUT / "brr_performance_replicates.csv.gz", index=False, compression="gzip")
    brr_summary.to_csv(OUT / "brr_performance_summary.csv", index=False)
    brr_deltas.to_csv(OUT / "brr_deltas.csv", index=False)
    overlap_ci.to_csv(OUT / "queue_overlap.csv", index=False)
    overlap_brr.to_csv(OUT / "queue_overlap_brr.csv.gz", index=False, compression="gzip")
    profiles.to_csv(OUT / "replacement_profiles.csv", index=False)
    contrasts.to_csv(OUT / "replacement_contrasts.csv", index=False)
    score_profile.to_csv(OUT / "score_miss_profile.csv", index=False)
    references.to_csv(OUT / "reference_profiles.csv", index=False)
    sex_summary.to_csv(OUT / "sex_calibration_summary.csv", index=False)
    sex_curve.to_csv(OUT / "sex_calibration_curve.csv", index=False)
    audit_table.to_csv(OUT / "audit_attribute_logit.csv", index=False)
    region.to_csv(OUT / "region_uncertainty.csv", index=False)
    inclusion.to_csv(OUT / "inclusion_audit.csv", index=False)

    low_overlap = overlap_ci.loc[overlap_ci["population"].eq("low_life_satisfaction")].set_index("cell")
    all_overlap = overlap_ci.loc[overlap_ci["population"].eq("all_students")].set_index("cell")
    routine_covered = low_overlap.loc["both", "weighted_share"] + low_overlap.loc["routine_only", "weighted_share"]
    displaced_fraction = low_overlap.loc["routine_only", "weighted_share"] / routine_covered
    summary = {
        "analysis_id": CONFIG["analysis_id"],
        "status": "pass",
        "analytic_n": int(len(frame)),
        "spain_total_n": int(len(all_spain)),
        "excluded_outcome_n": int(len(all_spain) - len(frame)),
        "alternative_bullying_nonmissing_n": int(frame[CONFIG["alternative_bullying_name"]].notna().sum()),
        "baseline_refit_max_abs_difference": refit_differences,
        "routine_covered_cases_displaced_fraction": float(displaced_fraction),
        "all_student_union_capacity": float(all_overlap.loc["union", "weighted_share"]),
        "interpretation_boundary": CONFIG["claim_boundary"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text(
        "# Reviewer revision analysis\n\n"
        f"- Analytic sample: {len(frame):,}; excluded for invalid or missing outcome: {len(all_spain) - len(frame):,}.\n"
        f"- Alternative seven-item peer-victimisation composite available for {summary['alternative_bullying_nonmissing_n']:,} analytic students.\n"
        f"- Routine-covered low-life-satisfaction case weight displaced from the expanded queue: {displaced_fraction:.3%}.\n"
        f"- Full-sample union capacity of the two 10% queues: {summary['all_student_union_capacity']:.3%}.\n"
        "- All uncertainty analyses condition on fixed out-of-fold scores; they do not include model-refitting uncertainty.\n",
        encoding="utf-8",
    )
    validation = {
        "status": "pass",
        "checks": {
            "analytic_rows": len(frame) == 29588,
            "schools": int(frame["CNTSCHID"].nunique()) == 965,
            "folds_complete": sorted(frame["fold__school_nested"].unique().tolist()) == [0, 1, 2, 3, 4],
            "routine_refit_reproduces_baseline": refit_differences["routine_logit"] <= 1e-10,
            "expanded_refit_reproduces_baseline": refit_differences["expanded_logit"] <= 1e-10,
            "all_predictions_finite": all(np.isfinite(value).all() for value in reported_predictions.values()),
            "overlap_cells_sum_to_one": all(
                abs(
                    overlap.loc[
                        overlap["population"].eq(population) & overlap["cell"].isin(["both", "routine_only", "expanded_only", "neither"]),
                        "weighted_share",
                    ].sum()
                    - 1.0
                )
                < 1e-10
                for population in ["all_students", "low_life_satisfaction"]
            ),
            "brr_replicates": 80,
            "bootstrap_replicates": CONFIG["bootstrap_replicates"],
        },
        "claim_boundary": CONFIG["claim_boundary"],
    }
    if not all(validation["checks"].values()):
        validation["status"] = "fail"
    (OUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyreadstat", "scikit-learn"]
        },
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": sha256(DATA),
    }
    (OUT / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"completed status={validation['status']}")
    artifacts = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    (OUT / "artifact_manifest.json").write_text(
        json.dumps({"analysis_id": CONFIG["analysis_id"], "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
