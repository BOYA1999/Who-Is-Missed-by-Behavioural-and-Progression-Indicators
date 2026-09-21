from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "visibility_profiles_v1"
DATA = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
MAIN = ROOT / "artifacts" / "experiment" / "main_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


failures: list[str] = []
assignments = pd.read_csv(OUT / "visibility_assignments.csv.gz")
shares = pd.read_csv(OUT / "visibility_group_shares.csv").set_index("group")
profiles = pd.read_csv(OUT / "profile_estimates.csv").set_index(["group", "variable"])
main_metrics = pd.read_csv(MAIN / "metrics.csv").set_index(["split", "model"])

groups = ["routine_visible", "student_experience_only", "missed_by_both"]
if len(assignments) != 4057 or assignments["CNTSTUID"].duplicated().any():
    failures.append("assignment inventory mismatch")
if not np.allclose(assignments[groups].sum(axis=1), 1.0, atol=1e-12):
    failures.append("row-level memberships do not sum to one")

w = assignments["W_FSTUWT"].to_numpy(dtype=float)
for group in groups:
    observed = float(np.sum(w * assignments[group]) / np.sum(w))
    expected = float(shares.loc[group, "weighted_share_of_low_life_satisfaction"])
    if abs(observed - expected) > 1e-12:
        failures.append(f"share mismatch for {group}")

for selection, model in [
    ("selected_routine", "routine_logit"),
    ("selected_student_experience", "expanded_logit"),
]:
    observed = float(np.sum(w * assignments[selection]) / np.sum(w))
    expected = float(main_metrics.loc[("school_nested", model), "recall_at_10pct_capacity"])
    if abs(observed - expected) > 1e-12:
        failures.append(f"recall mismatch for {model}")

raw_columns = ["CNT", "CNTSTUID", "BELONG", "BULLIED", "TEACHSUP"]
raw, _ = pyreadstat.read_sav(DATA, usecols=raw_columns)
raw = raw.loc[raw["CNT"].eq("ESP")].drop(columns="CNT")
check = assignments.merge(raw, on="CNTSTUID", how="left", validate="one_to_one")
for group in groups:
    group_weight = w * check[group].to_numpy(dtype=float)
    for variable in ["BELONG", "BULLIED", "TEACHSUP"]:
        values = check[variable].to_numpy(dtype=float)
        valid = np.isfinite(values)
        observed = float(np.sum(group_weight[valid] * values[valid]) / np.sum(group_weight[valid]))
        expected = float(profiles.loc[(group, variable), "estimate"])
        if abs(observed - expected) > 1e-12:
            failures.append(f"profile mismatch for {group} {variable}")

manifest = json.loads((OUT / "artifact_manifest.json").read_text(encoding="utf-8"))
for item in manifest["artifacts"]:
    path = ROOT / item["path"]
    if not path.exists() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        failures.append(f"manifest mismatch: {item['path']}")

result = {
    "status": "pass" if not failures else "fail",
    "failures": failures,
    "checks": {
        "assignment_rows": len(assignments),
        "memberships_sum_to_one": not np.any(np.abs(assignments[groups].sum(axis=1) - 1.0) > 1e-12),
        "recalls_match_main_v1": not any("recall mismatch" in failure for failure in failures),
        "relational_profiles_recomputed": not any("profile mismatch" in failure for failure in failures),
        "artifact_manifest_verified": not any("manifest mismatch" in failure for failure in failures),
    },
}
(OUT / "independent_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if failures:
    raise SystemExit(1)
