import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/experiment/main_v1/oof_predictions.csv.gz"
REFERENCE = ROOT / "artifacts/analysis/campaign_v1/allocation_policy.csv"
NATIONAL = ROOT / "artifacts/analysis/reviewer_revision_v1/queue_overlap.csv"
OUT = ROOT / "artifacts/stability_revision_20260923"


def select(scores, weights):
    selected = np.zeros(len(scores))
    order = np.argsort(-scores, kind="stable")
    target = 0.1 * weights.sum()
    used = 0.0
    start = 0
    while start < len(order) and used < target:
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group = order[start:end]
        group_weight = weights[group].sum()
        fraction = min(1.0, max(0.0, (target - used) / group_weight))
        selected[group] = fraction
        used += group_weight * fraction
        start = end
    return selected


cols = ["CNTSCHID", "outcome", "W_FSTUWT", "p__routine_logit__school_nested", "p__expanded_logit__school_nested"]
frame = pd.read_csv(SOURCE, usecols=cols)
assert len(frame) == 29588 and frame.CNTSCHID.nunique() == 965
assert frame[cols].notna().all().all()
weights = frame.W_FSTUWT.to_numpy(float)
positive = frame.outcome.to_numpy(float)
groups = frame.groupby("CNTSCHID").indices
memberships = {}
checks = {}
reference = pd.read_csv(REFERENCE)
for model in ("routine_logit", "expanded_logit"):
    scores = frame[f"p__{model}__school_nested"].to_numpy(float)
    selected = np.zeros(len(frame))
    for indices in groups.values():
        selected[indices] = select(scores[indices], weights[indices])
        assert abs(np.dot(weights[indices], selected[indices]) / weights[indices].sum() - 0.1) < 1e-12
    memberships[model] = selected
    recall = np.dot(weights * positive, selected) / np.dot(weights, positive)
    precision = np.dot(weights * positive, selected) / np.dot(weights, selected)
    target = reference.loc[(reference.model == model) & (reference.policy == "within_school") & reference.recall.notna()].iloc[0]
    assert abs(recall - target.recall) < 1e-12
    assert abs(precision - target.precision) < 1e-12
    checks[model] = {"recall": float(recall), "precision": float(precision), "capacity": float(np.dot(weights, selected) / weights.sum())}

a = memberships["routine_logit"]
b = memberships["expanded_logit"]
cells = {"both": a * b, "bp_only": a * (1 - b), "expanded_only": (1 - a) * b, "neither": (1 - a) * (1 - b)}
rows = []
for name, membership in cells.items():
    rows.append({"cell": name, "share_of_all_weight": float(np.dot(weights, membership) / weights.sum()), "share_of_low_ls_weight": float(np.dot(weights * positive, membership) / np.dot(weights, positive))})
assert abs(sum(row["share_of_all_weight"] for row in rows) - 1) < 1e-12
assert abs(sum(row["share_of_low_ls_weight"] for row in rows) - 1) < 1e-12
assert abs(rows[0]["share_of_all_weight"] + rows[1]["share_of_all_weight"] - 0.1) < 1e-12
assert abs(rows[0]["share_of_all_weight"] + rows[2]["share_of_all_weight"] - 0.1) < 1e-12
checks["replacement_share_of_bp_selected_low_ls"] = rows[1]["share_of_low_ls_weight"] / (rows[0]["share_of_low_ls_weight"] + rows[1]["share_of_low_ls_weight"])
national_a = select(frame.p__routine_logit__school_nested.to_numpy(float), weights)
national_b = select(frame.p__expanded_logit__school_nested.to_numpy(float), weights)
national_both = float(np.dot(weights, national_a * national_b) / weights.sum())
expected_national_both = pd.read_csv(NATIONAL).query("population == 'all_students' and cell == 'both'").weighted_share.iloc[0]
assert abs(national_both - expected_national_both) < 1e-12
checks["national_both_reproduced"] = national_both
OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "within_school_overlap_aggregate.csv", index=False)
(OUT / "within_school_overlap_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(checks, indent=2))
