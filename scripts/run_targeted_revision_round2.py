import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "artifacts" / "analysis" / "reviewer_revision_v1"
CAMPAIGN = ROOT / "artifacts" / "analysis" / "campaign_v1"
MAIN = ROOT / "artifacts" / "experiment" / "main_v1"
OUT = ROOT / "artifacts" / "analysis" / "targeted_revision_v2"
OUT.mkdir(parents=True, exist_ok=True)


def selection_fraction(scores, weights, capacity=0.10):
    target = capacity * weights.sum()
    selected = np.zeros(len(scores), dtype=float)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    start = 0
    used = 0.0
    while start < len(order) and used < target:
        end = start + 1
        while end < len(order) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        group = order[start:end]
        group_weight = weights[group].sum()
        remaining = target - used
        if group_weight <= remaining + 1e-12:
            selected[group] = 1.0
            used += group_weight
        else:
            selected[group] = max(0.0, remaining / group_weight)
            used = target
        start = end
    return selected


metrics = pd.read_csv(REV / "model_metrics.csv").set_index("model")
bootstrap_deltas = pd.read_csv(REV / "bootstrap_deltas.csv")
metric_columns = {
    "weighted_average_precision": "auprc",
    "recall_at_10pct_capacity": "recall_at_10pct_capacity",
    "precision_at_10pct_capacity": "precision_at_10pct_capacity",
}
point_rows = []
for row in bootstrap_deltas.itertuples(index=False):
    point_rows.append(
        {
            "model": row.model,
            "reference": row.reference,
            "metric": row.metric,
            "full_sample_delta": metrics.loc[row.model, metric_columns[row.metric]]
            - metrics.loc[row.reference, metric_columns[row.metric]],
            "bootstrap_mean_delta": row.bootstrap_mean_delta,
            "bootstrap_ci_lower": row._4,
            "bootstrap_ci_upper": row._5,
        }
    )
point_deltas = pd.DataFrame(point_rows)
point_deltas.to_csv(OUT / "point_delta_checks.csv", index=False)

identity_rows = []
for model, row in metrics.iterrows():
    for capacity in (5, 10, 15, 20):
        expected = row["weighted_prevalence"] * row[f"recall_at_{capacity}pct_capacity"] / (capacity / 100)
        observed = row[f"precision_at_{capacity}pct_capacity"]
        identity_rows.append(
            {
                "model": model,
                "capacity": capacity / 100,
                "observed_precision": observed,
                "identity_precision": expected,
                "absolute_residual": abs(observed - expected),
            }
        )
identity = pd.DataFrame(identity_rows)
assert identity["absolute_residual"].max() < 1e-12
identity.to_csv(OUT / "capacity_identity_checks.csv", index=False)

bootstrap = pd.read_csv(REV / "bootstrap_metrics.csv.gz")
brr = pd.read_csv(REV / "brr_performance_replicates.csv.gz")
hgb_rows = []
for metric, column in metric_columns.items():
    full_delta = metrics.loc["expanded_hgb", column] - metrics.loc["expanded_logit", column]
    boot_wide = bootstrap.pivot(index="replicate", columns="model", values=metric)
    boot_delta = boot_wide["expanded_hgb"] - boot_wide["expanded_logit"]
    brr_wide = brr.pivot(index="replicate", columns="model", values=metric)
    brr_delta = brr_wide["expanded_hgb"] - brr_wide["expanded_logit"]
    brr_se = np.sqrt(0.05 * np.square(brr_delta - full_delta).sum())
    hgb_rows.append(
        {
            "metric": metric,
            "full_sample_delta": full_delta,
            "bootstrap_ci_lower": boot_delta.quantile(0.025),
            "bootstrap_ci_upper": boot_delta.quantile(0.975),
            "brr_ci_lower": full_delta - 1.96 * brr_se,
            "brr_ci_upper": full_delta + 1.96 * brr_se,
        }
    )
pd.DataFrame(hgb_rows).to_csv(OUT / "hgb_vs_expanded_logit.csv", index=False)

queue = pd.read_csv(REV / "queue_overlap.csv")
queue_replicates = pd.read_csv(REV / "queue_overlap_brr.csv.gz")
low = queue.loc[queue["population"].eq("low_life_satisfaction")].set_index("cell")
displaced = low.loc["routine_only", "weighted_share"] / (
    low.loc["both", "weighted_share"] + low.loc["routine_only", "weighted_share"]
)
rep_wide = queue_replicates.loc[
    queue_replicates["population"].eq("low_life_satisfaction")
].pivot(index="replicate", columns="cell", values="weighted_share")
rep_displaced = rep_wide["routine_only"] / (rep_wide["both"] + rep_wide["routine_only"])
displaced_se = np.sqrt(0.05 * np.square(rep_displaced - displaced).sum())
overlap_summary = queue.copy()
overlap_summary = pd.concat(
    [
        overlap_summary,
        pd.DataFrame(
            [
                {
                    "population": "low_life_satisfaction",
                    "cell": "routine_covered_displaced_fraction",
                    "weighted_share": displaced,
                    "brr_se": displaced_se,
                    "ci_lower": displaced - 1.96 * displaced_se,
                    "ci_upper": displaced + 1.96 * displaced_se,
                }
            ]
        ),
    ],
    ignore_index=True,
)
overlap_summary.to_csv(OUT / "overlap_uncertainty.csv", index=False)

main = pd.read_csv(
    MAIN / "oof_predictions.csv.gz",
    usecols=[
        "CNTSTUID",
        "CNTSCHID",
        "outcome",
        "W_FSTUWT",
        "p__routine_logit__school_nested",
        "p__expanded_logit__school_nested",
    ],
)
revision_predictions = pd.read_csv(
    REV / "revision_predictions.csv.gz", usecols=["CNTSTUID", "routine_logit", "expanded_logit"]
)
frame = main.merge(revision_predictions, on="CNTSTUID", validate="one_to_one")
assert np.max(np.abs(frame["routine_logit"] - frame["p__routine_logit__school_nested"])) < 1e-12
assert np.max(np.abs(frame["expanded_logit"] - frame["p__expanded_logit__school_nested"])) < 1e-12
w = frame["W_FSTUWT"].to_numpy(float)
tie_rows = []
for model in ("routine_logit", "expanded_logit"):
    selected = selection_fraction(frame[model].to_numpy(float), w)
    boundary = (selected > 0) & (selected < 1)
    tie_rows.append(
        {
            "model": model,
            "boundary_students": int(boundary.sum()),
            "boundary_weight_share": w[boundary].sum() / w.sum(),
            "selected_boundary_weight_share": np.sum(w[boundary] * selected[boundary]) / w.sum(),
            "allocation_fraction": float(np.unique(selected[boundary])[0]),
        }
    )
pd.DataFrame(tie_rows).to_csv(OUT / "fractional_boundary.csv", index=False)

campaign_predictions = pd.read_csv(
    CAMPAIGN / "sensitivity_predictions.csv.gz",
    usecols=["CNTSTUID", "audit_attributes__expanded_audit_logit"],
)
frame = frame.merge(campaign_predictions, on="CNTSTUID", validate="one_to_one")
y = frame["outcome"].to_numpy(int)
audit_p = frame["audit_attributes__expanded_audit_logit"].to_numpy(float)
main_p = frame["expanded_logit"].to_numpy(float)
schools = pd.Index(frame["CNTSCHID"].unique())
school_codes = pd.Categorical(frame["CNTSCHID"], categories=schools).codes
rng = np.random.default_rng(20260920 + 1000)
audit_bootstrap_rows = []
for replicate in range(1000):
    sampled = rng.integers(0, len(schools), len(schools))
    multiplicity = np.bincount(sampled, minlength=len(schools))
    wr = w * multiplicity[school_codes]
    present = wr > 0
    for model, predictions in (("expanded_logit", main_p), ("expanded_audit_logit", audit_p)):
        selected = selection_fraction(predictions[present], wr[present])
        yp = y[present]
        wp = wr[present]
        tp = np.sum(wp * yp * selected)
        audit_bootstrap_rows.append(
            {
                "replicate": replicate,
                "model": model,
                "weighted_average_precision": average_precision_score(
                    yp, predictions[present], sample_weight=wp
                ),
                "recall_at_10pct_capacity": tp / np.sum(wp * yp),
                "precision_at_10pct_capacity": tp / np.sum(wp * selected),
            }
        )
audit_bootstrap = pd.DataFrame(audit_bootstrap_rows)
audit_bootstrap.to_csv(OUT / "audit_attribute_bootstrap_replicates.csv.gz", index=False, compression="gzip")
existing_main = bootstrap.loc[bootstrap["model"].eq("expanded_logit")].sort_values("replicate")
recomputed_main = audit_bootstrap.loc[audit_bootstrap["model"].eq("expanded_logit")].sort_values("replicate")
bootstrap_match = max(
    np.max(np.abs(existing_main[metric].to_numpy() - recomputed_main[metric].to_numpy()))
    for metric in metric_columns
)
assert bootstrap_match < 1e-12

campaign_results = pd.read_csv(CAMPAIGN / "slice_results.csv")
audit_full = campaign_results.loc[
    campaign_results["model"].eq("expanded_audit_logit")
    & campaign_results["slice_id"].eq("S3_audit_attributes")
].iloc[0]
audit_models = pd.DataFrame(
    [
        {
            "model": "Expanded logit",
            "weighted_ap": metrics.loc["expanded_logit", "auprc"],
            "recall_at_10pct_capacity": metrics.loc["expanded_logit", "recall_at_10pct_capacity"],
            "precision_at_10pct_capacity": metrics.loc["expanded_logit", "precision_at_10pct_capacity"],
            "calibration_intercept": metrics.loc["expanded_logit", "calibration_intercept"],
            "calibration_slope": metrics.loc["expanded_logit", "calibration_slope"],
        },
        {
            "model": "Expanded + audit attributes",
            "weighted_ap": audit_full["auprc"],
            "recall_at_10pct_capacity": audit_full["recall_at_10pct_capacity"],
            "precision_at_10pct_capacity": audit_full["precision_at_10pct_capacity"],
            "calibration_intercept": audit_full["calibration_intercept"],
            "calibration_slope": audit_full["calibration_slope"],
        },
    ]
)
audit_models.to_csv(OUT / "audit_attribute_overall.csv", index=False)
audit_wide = audit_bootstrap.pivot(index="replicate", columns="model")
audit_delta_rows = []
for metric, column in metric_columns.items():
    delta = audit_wide[(metric, "expanded_audit_logit")] - audit_wide[(metric, "expanded_logit")]
    audit_delta_rows.append(
        {
            "metric": metric,
            "full_sample_delta": audit_models.loc[1, column if column != "auprc" else "weighted_ap"]
            - audit_models.loc[0, column if column != "auprc" else "weighted_ap"],
            "bootstrap_ci_lower": delta.quantile(0.025),
            "bootstrap_ci_upper": delta.quantile(0.975),
        }
    )
pd.DataFrame(audit_delta_rows).to_csv(OUT / "audit_attribute_deltas.csv", index=False)

robustness = campaign_results.loc[
    campaign_results["slice_id"].isin(
        ["S1_continuous_outcome", "S2_complete_case_retrained", "S2_main_predictions_restricted"]
    )
].copy()
robustness = robustness[
    [
        "slice_id",
        "model",
        "n",
        "mae",
        "rmse",
        "r2",
        "auprc",
        "recall_at_10pct_capacity",
        "precision_at_10pct_capacity",
    ]
]
for model in ("expanded_hgb", "expanded_hgb_no_earlystop"):
    row = metrics.loc[model]
    robustness.loc[len(robustness)] = [
        "S3_hgb_early_stopping",
        model,
        row["n"],
        np.nan,
        np.nan,
        np.nan,
        row["auprc"],
        row["recall_at_10pct_capacity"],
        row["precision_at_10pct_capacity"],
    ]
robustness.to_csv(OUT / "robustness_results.csv", index=False)

validation = {
    "status": "pass",
    "max_capacity_identity_residual": identity["absolute_residual"].max(),
    "bootstrap_recalculation_max_difference": bootstrap_match,
    "four_cell_sum_all_students": queue.loc[
        queue["population"].eq("all_students") & ~queue["cell"].eq("union"), "weighted_share"
    ].sum(),
    "four_cell_sum_low_life_satisfaction": queue.loc[
        queue["population"].eq("low_life_satisfaction") & ~queue["cell"].eq("union"), "weighted_share"
    ].sum(),
}
assert abs(validation["four_cell_sum_all_students"] - 1) < 1e-12
assert abs(validation["four_cell_sum_low_life_satisfaction"] - 1) < 1e-12
(OUT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
print(json.dumps(validation))
