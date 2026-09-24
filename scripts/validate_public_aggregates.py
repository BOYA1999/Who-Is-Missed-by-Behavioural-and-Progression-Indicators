from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

metrics = pd.read_csv(ROOT / "artifacts" / "experiment" / "main_v1" / "metrics.csv")
school = metrics.loc[metrics["split"].eq("school_nested")].set_index("model")
assert set(school.index) == {"routine_logit", "routine_hgb", "expanded_logit", "expanded_hgb"}
assert school["n"].eq(29588).all()
np.testing.assert_allclose(school["weighted_prevalence"], 0.14576531643843837, rtol=0, atol=1e-15)

expected = {
    "routine_logit": (0.19968719409216784, 0.16295458838094906),
    "expanded_logit": (0.30341680359226025, 0.26537315919045984),
    "expanded_hgb": (0.32887827551473353, 0.28094462841968787),
}
for model, (ap, recall) in expected.items():
    np.testing.assert_allclose(school.loc[model, "auprc"], ap, rtol=0, atol=1e-15)
    np.testing.assert_allclose(school.loc[model, "recall_at_10pct_capacity"], recall, rtol=0, atol=1e-15)
    identity = school.loc[model, "weighted_prevalence"] * recall / 0.10
    np.testing.assert_allclose(identity, school.loc[model, "precision_at_10pct_capacity"], rtol=0, atol=2e-15)

overlap = pd.read_csv(ROOT / "artifacts" / "analysis" / "reviewer_revision_v1" / "queue_overlap.csv")
for population in ["all_students", "low_life_satisfaction"]:
    cells = overlap.loc[overlap["population"].eq(population) & overlap["cell"].isin(["both", "routine_only", "expanded_only", "neither"])]
    np.testing.assert_allclose(cells["weighted_share"].sum(), 1.0, rtol=0, atol=2e-15)
assert np.isclose(overlap.query("population == 'all_students' and cell == 'both'")["weighted_share"].iloc[0], 0.03626423507012329)
assert np.isclose(overlap.query("population == 'all_students' and cell == 'union'")["weighted_share"].iloc[0], 0.16373576492987685)
assert np.isclose(overlap.query("population == 'low_life_satisfaction' and cell == 'neither'")["weighted_share"].iloc[0], 0.6687846019093204)

contrasts = pd.read_csv(ROOT / "artifacts" / "analysis" / "profile_bootstrap_reselected_v1" / "contrast_summary.csv")
life = contrasts.query("contrast == 'expanded_only_minus_routine_only' and variable == 'ST016Q01NA'").iloc[0]
np.testing.assert_allclose([life["difference"], life["ci_lower"], life["ci_upper"]], [-0.2831768603395117, -0.5238879553845844, -0.003991984224476726], rtol=0, atol=1e-15)
assert int(life["finite_replicates"]) == 1000

targeted = json.loads((ROOT / "artifacts" / "analysis" / "targeted_revision_v2" / "validation.json").read_text(encoding="utf-8"))
assert targeted["status"] == "pass"
assert targeted["max_capacity_identity_residual"] < 2e-15
assert targeted["bootstrap_recalculation_max_difference"] < 1e-15

stability = ROOT / "artifacts" / "stability_revision_20260923"
runs = pd.read_csv(stability / "run_metrics_aggregate.csv")
cross = pd.read_csv(stability / "cross_information_aggregate.csv")
within = pd.read_csv(stability / "within_information_aggregate.csv")
local = pd.read_csv(stability / "within_school_overlap_aggregate.csv")
assert set(runs["seed"]) == {20260918, 20260923, 20260924}
assert len(runs) == 6 and len(cross) == 3 and len(within) == 6
np.testing.assert_allclose(runs["selected_weight_share"], 0.10, rtol=0, atol=2e-15)
np.testing.assert_allclose(
    cross["overlap_all_student_weight_share"] + cross["first_only_all_student_weight_share"],
    0.10, rtol=0, atol=2e-15,
)
np.testing.assert_allclose(
    within["overlap_all_student_weight_share"] + within["first_only_all_student_weight_share"],
    0.10, rtol=0, atol=2e-15,
)
assert cross["overlap_all_student_weight_share"].between(0.0356, 0.0364).all()
assert within.query("information_set == 'routine'")["overlap_all_student_weight_share"].between(0.0906, 0.0921).all()
assert within.query("information_set == 'expanded'")["overlap_all_student_weight_share"].between(0.0946, 0.0952).all()
assert set(local["cell"]) == {"both", "bp_only", "expanded_only", "neither"}
np.testing.assert_allclose(local["share_of_all_weight"].sum(), 1.0, rtol=0, atol=2e-15)
np.testing.assert_allclose(local["share_of_low_ls_weight"].sum(), 1.0, rtol=0, atol=2e-15)

for path in ROOT.rglob("*.csv*"):
    frame = pd.read_csv(path, nrows=0)
    forbidden = {"CNTSTUID", "CNTSCHID"}.intersection(frame.columns)
    assert not forbidden, f"Unit identifiers in public table {path}: {sorted(forbidden)}"

print("PASS: public aggregate results are internally consistent.")
