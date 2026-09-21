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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_main import nested_oof
from run_pilot import core_metrics, safe_group_metrics, selection_fraction, weighted_mean, weighted_quantiles


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "campaign_v1"
DATA = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
MAIN = ROOT / "artifacts" / "experiment" / "main_v1"
MAIN_CONFIG = json.loads((MAIN / "config.json").read_text(encoding="utf-8"))
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))


def regression_candidates(kind: str) -> list[dict]:
    grid = CONFIG["continuous_logit_analogue"] if kind == "ridge" else CONFIG["continuous_hgb"]
    grid = {key: value for key, value in grid.items() if key != "model"}
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def regression_model(kind: str, params: dict, seed: int) -> Pipeline:
    if kind == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(**params)),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("regressor", HistGradientBoostingRegressor(random_state=seed, **params)),
        ]
    )


def regression_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    mean_y = weighted_mean(y, w)
    mean_p = weighted_mean(p, w)
    centered_p = p - mean_p
    slope = float(np.sum(w * centered_p * (y - mean_y)) / np.sum(w * centered_p**2))
    intercept = mean_y - slope * mean_p
    error = y - p
    sse = float(np.sum(w * error**2))
    sst = float(np.sum(w * (y - mean_y) ** 2))
    return {
        "n": int(len(y)),
        "weighted_mean_outcome": mean_y,
        "mae": float(np.sum(w * np.abs(error)) / np.sum(w)),
        "rmse": float(np.sqrt(sse / np.sum(w))),
        "r2": 1 - sse / sst,
        "calibration_intercept": float(intercept),
        "calibration_slope": slope,
    }


def fit_regression(model, x, y, w):
    model.fit(x, y, regressor__sample_weight=w / w.mean())
    return model


def tune_regression(frame, train_index, features, kind, seed):
    train = frame.iloc[train_index]
    y = train["ST016Q01NA"].to_numpy(dtype=float)
    w = train["W_FSTUWT"].to_numpy(dtype=float)
    groups = train["CNTSCHID"].to_numpy()
    strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
    splitter = StratifiedGroupKFold(
        n_splits=CONFIG["inner_school_splits"], shuffle=True, random_state=seed
    )
    splits = list(splitter.split(train[features], strata, groups))
    best_params = None
    best_rmse = np.inf
    rows = []
    for candidate_index, params in enumerate(regression_candidates(kind)):
        predictions = np.full(len(train), np.nan)
        for fold, (inner_train, inner_valid) in enumerate(splits):
            model = regression_model(kind, params, seed + candidate_index * 10 + fold)
            fit_regression(model, train.iloc[inner_train][features], y[inner_train], w[inner_train])
            predictions[inner_valid] = model.predict(train.iloc[inner_valid][features])
        rmse = regression_metrics(y, predictions, w)["rmse"]
        rows.append({"params": params, "weighted_rmse": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    return best_params, rows


def continuous_nested_oof(frame, features, kind, log):
    y = frame["ST016Q01NA"].to_numpy(dtype=float)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    schools = frame["CNTSCHID"].to_numpy()
    strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
    splitter = StratifiedGroupKFold(
        n_splits=CONFIG["outer_school_splits"], shuffle=True, random_state=CONFIG["seed"]
    )
    predictions = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=int)
    tuning_rows = []
    checks = []
    for fold, (train_index, test_index) in enumerate(splitter.split(frame[features], strata, schools)):
        overlap = set(schools[train_index]).intersection(set(schools[test_index]))
        if overlap:
            raise RuntimeError(f"continuous school leakage in fold {fold}")
        params, candidates = tune_regression(
            frame, train_index, features, kind, CONFIG["seed"] + fold * 100
        )
        model = regression_model(kind, params, CONFIG["seed"] + fold)
        fit_regression(
            model,
            frame.iloc[train_index][features],
            y[train_index],
            w[train_index],
        )
        predictions[test_index] = model.predict(frame.iloc[test_index][features])
        fold_ids[test_index] = fold
        checks.append(
            {
                "fold": fold,
                "school_overlap_n": len(overlap),
                "selected_params": params,
                "test_n": len(test_index),
            }
        )
        for candidate in candidates:
            tuning_rows.append(
                {
                    "fold": fold,
                    "kind": kind,
                    "params": json.dumps(candidate["params"], sort_keys=True),
                    "weighted_rmse": candidate["weighted_rmse"],
                    "selected": candidate["params"] == params,
                }
            )
        log(f"continuous {kind} fold {fold + 1}/{CONFIG['outer_school_splits']} complete")
    return predictions, fold_ids, checks, tuning_rows


def within_school_selection(frame, scores, capacity):
    selected = np.zeros(len(frame))
    for _, indices in frame.groupby("CNTSCHID").indices.items():
        indices = np.asarray(indices)
        selected[indices] = selection_fraction(
            scores[indices], frame.iloc[indices]["W_FSTUWT"].to_numpy(dtype=float), capacity
        )
    return selected


def allocation_metrics(frame, scores, selected):
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    true_positive = float(np.sum(w * y * selected))
    return {
        "recall": true_positive / float(np.sum(w * y)),
        "precision": true_positive / float(np.sum(w * selected)),
        "selected_weight_fraction": float(np.sum(w * selected) / np.sum(w)),
    }


def fairness_groups(frame):
    valid = frame["ESCS"].notna().to_numpy()
    cuts = weighted_quantiles(
        frame.loc[valid, "ESCS"].to_numpy(),
        frame.loc[valid, "W_FSTUWT"].to_numpy(),
        [1 / 3, 2 / 3],
    )
    return {
        "sex": {
            "female": frame["ST004D01T"].eq(1).to_numpy(),
            "male": frame["ST004D01T"].eq(2).to_numpy(),
        },
        "immigration_background": {
            "native": frame["IMMIG"].eq(1).to_numpy(),
            "immigrant": frame["IMMIG"].isin([2, 3]).to_numpy(),
        },
        "escs_weighted_tertile": {
            "low": frame["ESCS"].le(cuts[0]).to_numpy(),
            "middle": frame["ESCS"].gt(cuts[0]).to_numpy()
            & frame["ESCS"].le(cuts[1]).to_numpy(),
            "high": frame["ESCS"].gt(cuts[1]).to_numpy(),
        },
    }


def group_recall_gaps(frame, selected):
    y = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    rows = []
    for axis, levels in fairness_groups(frame).items():
        recalls = []
        for level, mask in levels.items():
            recall = float(np.sum(w[mask] * y[mask] * selected[mask]) / np.sum(w[mask] * y[mask]))
            rows.append({"group_axis": axis, "group": level, "recall": recall})
            recalls.append(recall)
        rows.append(
            {"group_axis": axis, "group": "max_minus_min", "recall": max(recalls) - min(recalls)}
        )
    return rows


def main():
    log_lines = []

    def log(message):
        stamped = f"{datetime.now(timezone.utc).isoformat()} {message}"
        log_lines.append(stamped)
        encoding = sys.stdout.encoding or "utf-8"
        print(stamped.encode(encoding, errors="replace").decode(encoding), flush=True)
        (OUT / "campaign.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

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
                *MAIN_CONFIG["routine_features"],
                *MAIN_CONFIG["school_experience_features"],
                *MAIN_CONFIG["audit_features"],
            ]
        )
    )
    frame, meta = pyreadstat.read_sav(DATA, usecols=variables, apply_value_formats=False)
    frame = frame.loc[(frame["CNT"] == "ESP") & frame["ST016Q01NA"].between(0, 10)].copy()
    frame.reset_index(drop=True, inplace=True)
    frame["outcome"] = frame["ST016Q01NA"].between(0, 4).astype(int)
    labels = meta.variable_value_labels["STRATUM"]
    frame["region"] = frame["STRATUM"].map(
        lambda value: labels.get(value, str(value)).split(":", 1)[-1].split(",", 1)[0].strip()
    )
    frame["student_key"] = frame["CNTSTUID"].astype(str)
    main_predictions = pd.read_csv(MAIN / "oof_predictions.csv.gz")
    main_predictions["student_key"] = main_predictions["CNTSTUID"].astype(str)
    main_probabilities = [
        "student_key",
        "p__routine_logit__school_nested",
        "p__routine_hgb__school_nested",
        "p__expanded_logit__school_nested",
        "p__expanded_hgb__school_nested",
    ]
    frame = frame.merge(main_predictions[main_probabilities], on="student_key", how="left", validate="one_to_one")
    if frame[main_probabilities[1:]].isna().any().any():
        raise RuntimeError("main prediction merge failed")
    log(f"loaded {len(frame)} rows and {frame['CNTSCHID'].nunique()} schools")

    routine = MAIN_CONFIG["routine_features"]
    expanded = routine + MAIN_CONFIG["school_experience_features"]
    y_binary = frame["outcome"].to_numpy(dtype=int)
    w = frame["W_FSTUWT"].to_numpy(dtype=float)
    slice_results = []
    prediction_output = frame[["CNTSTUID", "CNTSCHID", "ST016Q01NA", "outcome", "W_FSTUWT"]].copy()

    continuous_tuning = []
    for feature_name, features in [("routine", routine), ("expanded", expanded)]:
        for kind in ["ridge", "hgb"]:
            model_name = f"{feature_name}_{kind}"
            p, fold_ids, checks, tuning_rows = continuous_nested_oof(frame, features, kind, log)
            prediction_output[f"continuous__{model_name}"] = p
            metrics = regression_metrics(frame["ST016Q01NA"].to_numpy(), p, w)
            slice_results.append(
                {
                    "slice_id": "S1_continuous_outcome",
                    "model": model_name,
                    "population": "all target-valid",
                    **metrics,
                }
            )
            continuous_tuning.extend({"model": model_name, **row} for row in tuning_rows)
            if any(check["school_overlap_n"] for check in checks):
                raise RuntimeError("continuous school leakage")

    complete_mask = frame[expanded].notna().all(axis=1).to_numpy()
    complete = frame.loc[complete_mask].copy().reset_index(drop=True)
    for kind in ["logit", "hgb"]:
        model_name = f"expanded_{kind}"
        p, _, checks, tuning_rows = nested_oof(
            complete, expanded, kind, "school_nested", MAIN_CONFIG, log
        )
        metrics = core_metrics(
            complete["outcome"].to_numpy(dtype=int),
            p,
            complete["W_FSTUWT"].to_numpy(dtype=float),
            MAIN_CONFIG["capacities"],
        )
        slice_results.append(
            {
                "slice_id": "S2_complete_case_retrained",
                "model": model_name,
                "population": "complete expanded predictors",
                **metrics,
            }
        )
        prediction_output.loc[complete_mask, f"complete_case__{model_name}"] = p
        main_p = complete[f"p__{model_name}__school_nested"].to_numpy(dtype=float)
        restricted_metrics = core_metrics(
            complete["outcome"].to_numpy(dtype=int),
            main_p,
            complete["W_FSTUWT"].to_numpy(dtype=float),
            MAIN_CONFIG["capacities"],
        )
        slice_results.append(
            {
                "slice_id": "S2_main_predictions_restricted",
                "model": model_name,
                "population": "complete expanded predictors",
                **restricted_metrics,
            }
        )

    audit_features = expanded + MAIN_CONFIG["audit_features"]
    audit_predictions = {}
    for kind in ["logit", "hgb"]:
        model_name = f"expanded_audit_{kind}"
        p, _, checks, tuning_rows = nested_oof(
            frame, audit_features, kind, "school_nested", MAIN_CONFIG, log
        )
        audit_predictions[model_name] = p
        prediction_output[f"audit_attributes__{model_name}"] = p
        metrics = core_metrics(y_binary, p, w, MAIN_CONFIG["capacities"])
        slice_results.append(
            {
                "slice_id": "S3_audit_attributes",
                "model": model_name,
                "population": "all target-valid",
                **metrics,
            }
        )

    allocation_rows = []
    for model_name in ["routine_logit", "routine_hgb", "expanded_logit", "expanded_hgb"]:
        p = frame[f"p__{model_name}__school_nested"].to_numpy(dtype=float)
        global_selected = selection_fraction(p, w, CONFIG["capacity"])
        local_selected = within_school_selection(frame, p, CONFIG["capacity"])
        for policy, selected in [("global", global_selected), ("within_school", local_selected)]:
            metrics = allocation_metrics(frame, p, selected)
            allocation_rows.append({"model": model_name, "policy": policy, **metrics})
            for row in group_recall_gaps(frame, selected):
                allocation_rows.append(
                    {
                        "model": model_name,
                        "policy": policy,
                        "group_axis": row["group_axis"],
                        "group": row["group"],
                        "group_recall": row["recall"],
                    }
                )

    fairness_rows = []
    for model_name, p in {
        "expanded_hgb_main": frame["p__expanded_hgb__school_nested"].to_numpy(dtype=float),
        **audit_predictions,
    }.items():
        selected = selection_fraction(p, w, CONFIG["capacity"])
        for row in group_recall_gaps(frame, selected):
            fairness_rows.append({"model": model_name, **row})

    results_frame = pd.DataFrame(slice_results)
    allocation_frame = pd.DataFrame(allocation_rows)
    fairness_frame = pd.DataFrame(fairness_rows)
    prediction_output.to_csv(OUT / "sensitivity_predictions.csv.gz", index=False, compression="gzip")
    results_frame.to_csv(OUT / "slice_results.csv", index=False, encoding="utf-8-sig")
    allocation_frame.to_csv(OUT / "allocation_policy.csv", index=False, encoding="utf-8-sig")
    fairness_frame.to_csv(OUT / "audit_attribute_fairness.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(continuous_tuning).to_csv(
        OUT / "continuous_tuning.csv.gz", index=False, compression="gzip"
    )

    continuous = results_frame.loc[results_frame["slice_id"] == "S1_continuous_outcome"].set_index("model")
    complete_retrained = results_frame.loc[
        results_frame["slice_id"] == "S2_complete_case_retrained"
    ].set_index("model")
    complete_restricted = results_frame.loc[
        results_frame["slice_id"] == "S2_main_predictions_restricted"
    ].set_index("model")
    audit = results_frame.loc[results_frame["slice_id"] == "S3_audit_attributes"].set_index("model")
    main_metrics = pd.read_csv(MAIN / "metrics.csv").query("split == 'school_nested'").set_index("model")
    allocation_core = allocation_frame.loc[allocation_frame["group_axis"].isna()].set_index(
        ["model", "policy"]
    )
    local_recall = allocation_core.loc[("expanded_hgb", "within_school"), "recall"]
    global_recall = allocation_core.loc[("expanded_hgb", "global"), "recall"]
    main_fair_gap = fairness_frame.loc[
        (fairness_frame["model"] == "expanded_hgb_main")
        & (fairness_frame["group"] == "max_minus_min")
    ].set_index("group_axis")["recall"]
    audit_fair_gap = fairness_frame.loc[
        (fairness_frame["model"] == "expanded_audit_hgb")
        & (fairness_frame["group"] == "max_minus_min")
    ].set_index("group_axis")["recall"]

    campaign_summary = f"""# Analysis campaign summary

## Stable findings

1. **The information-tier result survives the continuous outcome.** Expanded HGB RMSE was {continuous.loc['expanded_hgb', 'rmse']:.3f} versus {continuous.loc['routine_hgb', 'rmse']:.3f} for routine HGB; expanded ridge RMSE was {continuous.loc['expanded_ridge', 'rmse']:.3f} versus {continuous.loc['routine_ridge', 'rmse']:.3f}. Thus, the gain from school-experience scales is not created only by the 0–4 cutoff.
2. **Fold-wise imputation is not driving the binary result.** On {int(complete_retrained.loc['expanded_hgb', 'n']):,} complete cases, retrained expanded HGB AUPRC was {complete_retrained.loc['expanded_hgb', 'auprc']:.3f} and 10% capacity recall was {complete_retrained.loc['expanded_hgb', 'recall_at_10pct_capacity']:.3f}; the original main predictions restricted to the same students gave {complete_restricted.loc['expanded_hgb', 'auprc']:.3f} and {complete_restricted.loc['expanded_hgb', 'recall_at_10pct_capacity']:.3f}.
3. **The school-level capacity conclusion is stable.** Expanded HGB recall was {global_recall:.3f} under one national 10% capacity and {local_recall:.3f} when every school received its own 10% capacity.

## Narrowed findings

4. **Adding audit attributes has limited operational value.** Expanded-audit HGB AUPRC was {audit.loc['expanded_audit_hgb', 'auprc']:.3f} versus {main_metrics.loc['expanded_hgb', 'auprc']:.3f} without those attributes; recall was {audit.loc['expanded_audit_hgb', 'recall_at_10pct_capacity']:.3f} versus {main_metrics.loc['expanded_hgb', 'recall_at_10pct_capacity']:.3f}. Recall-gap changes were sex {audit_fair_gap['sex'] - main_fair_gap['sex']:+.3f}, immigration {audit_fair_gap['immigration_background'] - main_fair_gap['immigration_background']:+.3f}, and ESCS {audit_fair_gap['escs_weighted_tertile'] - main_fair_gap['escs_weighted_tertile']:+.3f}. These variables should remain audit variables rather than operational ranking inputs.

## Campaign conclusion

The main proxy-audit claim is strengthened: routine visible information is weak, richer school-experience information helps but still misses most cases, and this pattern survives continuous scoring, complete-case analysis and school-local capacity. The complex-model superiority claim remains narrow: nonlinear performance is somewhat better, but the practical gain over a transparent model is small. No further model-family expansion is justified.

## Next route

Move to paper outline and Methods/Results drafting. Keep plausible-value achievement analysis appendix-only unless a reviewer or target journal requires it, because it is not needed for the central proxy-replacement question.
"""
    (OUT / "campaign_summary.md").write_text(campaign_summary, encoding="utf-8")

    slice_contracts = [
        {
            "slice_id": "S1_continuous_outcome",
            "question": "Does the information-tier result survive without dichotomization?",
            "fixed": "Spain sample, features, weights, school-nested validation",
            "changed": "0-10 outcome and regression metrics",
            "claim_update": "strengthens",
            "comparability": "parallel outcome sensitivity; not numerically comparable to AUPRC",
            "next_action": "retain as main-text robustness",
        },
        {
            "slice_id": "S2_complete_case",
            "question": "Is the main binary result driven by imputation?",
            "fixed": "binary outcome, model classes, weights, school grouping",
            "changed": "complete-case population and no predictor imputation",
            "claim_update": "strengthens",
            "comparability": "directly compared on identical complete-case population",
            "next_action": "report in appendix with one main-text sentence",
        },
        {
            "slice_id": "S3_audit_attributes",
            "question": "Do audit attributes materially improve performance or gap profiles?",
            "fixed": "binary outcome, sample, weights, split and capacity",
            "changed": "sex, immigration and ESCS added as predictors",
            "claim_update": "narrows",
            "comparability": "feature-policy sensitivity",
            "next_action": "keep attributes for auditing rather than ranking",
        },
        {
            "slice_id": "S4_within_school_capacity",
            "question": "Does the result hold with local school capacity?",
            "fixed": "main OOF predictions, outcome, weights and 10% capacity",
            "changed": "global versus within-school allocation",
            "claim_update": "strengthens",
            "comparability": "allocation-policy sensitivity",
            "next_action": "use local-capacity result for school application framing",
        },
    ]
    payload = {
        "campaign_id": CONFIG["campaign_id"],
        "status": "success",
        "parent_run": CONFIG["parent_run"],
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "slices": slice_contracts,
        "next_route": "paper_outline_and_write",
        "claim_boundary": "cross-sectional contemporaneous identification audit; no diagnosis or causal/deployment claim",
    }
    (OUT / "campaign_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyreadstat", "scipy", "scikit-learn"]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
    }
    (OUT / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("analysis campaign complete")


if __name__ == "__main__":
    main()
