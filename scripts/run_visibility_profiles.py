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

from run_pilot import selection_fraction


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "visibility_profiles_v1"
DATA = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
PREDICTIONS = ROOT / "artifacts" / "experiment" / "main_v1" / "oof_predictions.csv.gz"
MAIN_METRICS = ROOT / "artifacts" / "experiment" / "main_v1" / "metrics.csv"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))

PROFILE_LABELS = {
    "ST016Q01NA": ("Life satisfaction (0-4)", "mean"),
    "BELONG": ("Sense of belonging", "mean"),
    "BULLIED": ("Exposure to bullying", "mean"),
    "TEACHSUP": ("Mathematics teacher support", "mean"),
    "ESCS": ("Economic, social and cultural status", "mean"),
    "female": ("Female", "proportion"),
    "immigrant": ("First- or second-generation immigrant", "proportion"),
    "repeated_grade": ("Repeated a grade", "proportion"),
    "any_whole_day_absence": ("Any whole-day absence in previous two weeks", "proportion"),
    "any_class_skipping": ("Any class skipping in previous two weeks", "proportion"),
    "any_tardiness": ("Any tardiness in previous two weeks", "proportion"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_from_codes(series: pd.Series, positive: set[float], valid: set[float]) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[series.isin(valid)] = series.loc[series.isin(valid)].isin(positive).astype(float)
    return result


def school_contributions(
    school_codes: np.ndarray,
    weights: np.ndarray,
    membership: np.ndarray,
    values: np.ndarray,
    school_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    effective = weights * membership * valid
    denominator = np.bincount(school_codes, weights=effective, minlength=school_count)
    numerator = np.bincount(
        school_codes,
        weights=effective * np.nan_to_num(values, nan=0.0),
        minlength=school_count,
    )
    return numerator, denominator


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        log_lines.append(line)
        print(line, flush=True)
        (OUT / "analysis.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("loading fixed school-held-out predictions")
    predictions = pd.read_csv(PREDICTIONS)
    if predictions[CONFIG["student_id"]].duplicated().any():
        raise RuntimeError("prediction student identifiers are not unique")

    raw_columns = [
        "CNT",
        "CNTSTUID",
        "ST016Q01NA",
        "BELONG",
        "BULLIED",
        "TEACHSUP",
        "REPEAT",
        "ST062Q01TA",
        "ST062Q02TA",
        "ST062Q03TA",
    ]
    raw, _ = pyreadstat.read_sav(DATA, usecols=raw_columns)
    raw = raw.loc[
        raw["CNT"].eq(CONFIG["country"]) & raw["ST016Q01NA"].between(0, 10)
    ].copy()
    raw = raw.drop(columns=["CNT", "ST016Q01NA"])
    frame = predictions.merge(raw, on="CNTSTUID", how="left", validate="one_to_one")
    if len(frame) != len(predictions) or frame["BELONG"].isna().all():
        raise RuntimeError("raw-variable merge failed")

    frame["female"] = binary_from_codes(frame["ST004D01T"], {1.0}, {1.0, 2.0})
    frame["immigrant"] = binary_from_codes(frame["IMMIG"], {2.0, 3.0}, {1.0, 2.0, 3.0})
    frame["repeated_grade"] = binary_from_codes(frame["REPEAT"], {1.0}, {0.0, 1.0})
    for source, target in [
        ("ST062Q01TA", "any_whole_day_absence"),
        ("ST062Q02TA", "any_class_skipping"),
        ("ST062Q03TA", "any_tardiness"),
    ]:
        frame[target] = binary_from_codes(frame[source], {2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0})

    weights = frame[CONFIG["weight"]].to_numpy(dtype=float)
    routine = selection_fraction(
        frame[CONFIG["routine_score"]].to_numpy(dtype=float), weights, CONFIG["capacity"]
    )
    expanded = selection_fraction(
        frame[CONFIG["student_experience_score"]].to_numpy(dtype=float),
        weights,
        CONFIG["capacity"],
    )
    frame["selected_routine"] = routine
    frame["selected_student_experience"] = expanded

    low = frame.loc[frame["outcome"].eq(1)].copy().reset_index(drop=True)
    routine_low = low["selected_routine"].to_numpy(dtype=float)
    expanded_low = low["selected_student_experience"].to_numpy(dtype=float)
    groups = {
        "routine_visible": routine_low,
        "student_experience_only": (1.0 - routine_low) * expanded_low,
        "missed_by_both": (1.0 - routine_low) * (1.0 - expanded_low),
    }
    overlaps = {
        "selected_by_both": routine_low * expanded_low,
        "routine_only": routine_low * (1.0 - expanded_low),
        "student_experience_only": (1.0 - routine_low) * expanded_low,
        "neither": (1.0 - routine_low) * (1.0 - expanded_low),
    }

    membership_sum = np.sum(np.column_stack(list(groups.values())), axis=1)
    if not np.allclose(membership_sum, 1.0, atol=1e-12):
        raise RuntimeError("primary visibility memberships do not sum to one")

    low_weights = low[CONFIG["weight"]].to_numpy(dtype=float)
    schools = pd.Index(frame[CONFIG["school_id"]].unique())
    school_codes = pd.Categorical(low[CONFIG["school_id"]], categories=schools).codes
    school_count = len(schools)
    rng = np.random.default_rng(CONFIG["seed"])
    bootstrap_counts = rng.multinomial(
        school_count,
        np.full(school_count, 1.0 / school_count),
        size=CONFIG["bootstrap_replicates"],
    )
    total_low_by_school = np.bincount(school_codes, weights=low_weights, minlength=school_count)
    total_low_boot = bootstrap_counts @ total_low_by_school

    profile_rows: list[dict] = []
    profile_boot: dict[tuple[str, str], np.ndarray] = {}
    group_share_rows: list[dict] = []
    for group, membership in groups.items():
        group_weight_by_school = np.bincount(
            school_codes, weights=low_weights * membership, minlength=school_count
        )
        share = float(group_weight_by_school.sum() / total_low_by_school.sum())
        share_boot = (bootstrap_counts @ group_weight_by_school) / total_low_boot
        lower, upper = interval(share_boot)
        group_share_rows.append(
            {
                "group": group,
                "unweighted_rows_with_membership": int(np.sum(membership > 0)),
                "weighted_share_of_low_life_satisfaction": share,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
        for variable in CONFIG["profile_variables"]:
            values = low[variable].to_numpy(dtype=float)
            numerator, denominator = school_contributions(
                school_codes, low_weights, membership, values, school_count
            )
            estimate = float(numerator.sum() / denominator.sum())
            boot = (bootstrap_counts @ numerator) / (bootstrap_counts @ denominator)
            profile_boot[(group, variable)] = boot
            lower, upper = interval(boot)
            label, unit = PROFILE_LABELS[variable]
            profile_rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "label": label,
                    "unit": unit,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_weight_share_within_group": float(
                        denominator.sum() / np.sum(low_weights * membership)
                    ),
                }
            )

    contrast_rows: list[dict] = []
    contrasts = [
        ("student_experience_only", "routine_visible"),
        ("missed_by_both", "student_experience_only"),
    ]
    profiles = pd.DataFrame(profile_rows)
    for left, right in contrasts:
        for variable in CONFIG["profile_variables"]:
            left_estimate = float(
                profiles.loc[
                    profiles["group"].eq(left) & profiles["variable"].eq(variable), "estimate"
                ].iloc[0]
            )
            right_estimate = float(
                profiles.loc[
                    profiles["group"].eq(right) & profiles["variable"].eq(variable), "estimate"
                ].iloc[0]
            )
            boot = profile_boot[(left, variable)] - profile_boot[(right, variable)]
            lower, upper = interval(boot)
            contrast_rows.append(
                {
                    "contrast": f"{left}_minus_{right}",
                    "variable": variable,
                    "difference": left_estimate - right_estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )

    overlap_rows = []
    for group, membership in overlaps.items():
        share = float(np.sum(low_weights * membership) / np.sum(low_weights))
        overlap_rows.append(
            {
                "group": group,
                "weighted_share_of_low_life_satisfaction": share,
                "unweighted_rows_with_membership": int(np.sum(membership > 0)),
            }
        )

    routine_recall = float(np.sum(low_weights * routine_low) / np.sum(low_weights))
    expanded_recall = float(np.sum(low_weights * expanded_low) / np.sum(low_weights))
    main_metrics = pd.read_csv(MAIN_METRICS).set_index(["split", "model"])
    expected_routine = float(
        main_metrics.loc[("school_nested", "routine_logit"), "recall_at_10pct_capacity"]
    )
    expected_expanded = float(
        main_metrics.loc[("school_nested", "expanded_logit"), "recall_at_10pct_capacity"]
    )
    failures = []
    if abs(routine_recall - expected_routine) > 1e-12:
        failures.append("routine recall does not reproduce main_v1")
    if abs(expanded_recall - expected_expanded) > 1e-12:
        failures.append("expanded recall does not reproduce main_v1")
    if abs(sum(row["weighted_share_of_low_life_satisfaction"] for row in group_share_rows) - 1) > 1e-12:
        failures.append("visibility group shares do not sum to one")
    if not np.isfinite(profiles[["estimate", "ci_lower", "ci_upper"]].to_numpy()).all():
        failures.append("profile results contain non-finite values")

    pd.DataFrame(group_share_rows).to_csv(OUT / "visibility_group_shares.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "selection_overlap.csv", index=False)
    profiles.to_csv(OUT / "profile_estimates.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT / "profile_contrasts.csv", index=False)
    assignments = low[
        ["CNTSTUID", "CNTSCHID", "W_FSTUWT", "ST016Q01NA", "selected_routine", "selected_student_experience"]
    ].copy()
    for group, membership in groups.items():
        assignments[group] = membership
    assignments.to_csv(OUT / "visibility_assignments.csv.gz", index=False, compression="gzip")

    summary = {
        "analysis_id": CONFIG["analysis_id"],
        "status": "pass" if not failures else "fail",
        "n_low_life_satisfaction": int(len(low)),
        "schools": int(school_count),
        "routine_recall": routine_recall,
        "student_experience_recall": expanded_recall,
        "group_shares": {
            row["group"]: row["weighted_share_of_low_life_satisfaction"]
            for row in group_share_rows
        },
        "interpretation_boundary": (
            "Groups are derived from fixed-capacity rules using the profiled variables; differences are "
            "descriptive of operational visibility pathways, not independent or causal effects."
        ),
        "failures": failures,
        "next_route": "write" if not failures else "repair_analysis",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "validation.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "failures": failures,
                "checks": {
                    "prediction_ids_unique": True,
                    "raw_merge_one_to_one": True,
                    "routine_recall_matches_main": abs(routine_recall - expected_routine) <= 1e-12,
                    "expanded_recall_matches_main": abs(expanded_recall - expected_expanded) <= 1e-12,
                    "visibility_memberships_sum_to_one": True,
                    "bootstrap_replicates": CONFIG["bootstrap_replicates"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyreadstat"]
        },
        "seed": CONFIG["seed"],
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    log(f"completed with status={summary['status']} routine_recall={routine_recall:.6f} expanded_recall={expanded_recall:.6f}")
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
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
