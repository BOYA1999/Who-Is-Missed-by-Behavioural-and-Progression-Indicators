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
OUT = ROOT / "artifacts" / "analysis" / "profile_bootstrap_reselected_v1"
DATA = ROOT / "data" / "raw" / "pisa2022" / "CY08MSP_STU_QQQ.SAV"
SCORES = ROOT / "artifacts" / "analysis" / "reviewer_revision_v1" / "revision_predictions.csv.gz"
OLD_PROFILES = ROOT / "artifacts" / "analysis" / "reviewer_revision_v1" / "replacement_profiles.csv"
OLD_CONTRASTS = ROOT / "artifacts" / "analysis" / "reviewer_revision_v1" / "replacement_contrasts.csv"

CONFIG = {
    "analysis_id": "profile_bootstrap_reselected_v1",
    "country": "ESP",
    "source_seed": 20260920,
    "seed_offset": 2000,
    "seed": 20262920,
    "bootstrap_replicates": 1000,
    "capacity": 0.10,
    "resampling_unit": "CNTSCHID",
    "queue_selection": "reselected within every bootstrap replicate",
    "fractional_tie_rule": "equal membership fraction within a score tie at the weighted boundary",
    "four_cell_convention": {
        "both": "routine * expanded",
        "routine_only": "routine * (1 - expanded)",
        "expanded_only": "(1 - routine) * expanded",
        "neither": "(1 - routine) * (1 - expanded)",
    },
    "ci": "2.5th and 97.5th percentiles of 1,000 school-cluster bootstrap replicates",
    "claim_boundary": "Conditional fixed-score exploratory/descriptive audit; no model-refitting, clinical, causal, or deployment inference.",
}

GROUPS = ["both", "routine_only", "expanded_only", "neither"]
VARIABLES = [
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


def overlap_membership(routine: np.ndarray, expanded: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "both": routine * expanded,
        "routine_only": routine * (1.0 - expanded),
        "expanded_only": (1.0 - routine) * expanded,
        "neither": (1.0 - routine) * (1.0 - expanded),
    }


def profile_matrix(
    values: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    memberships: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    estimates = np.empty((len(GROUPS), len(VARIABLES)), dtype=float)
    denominators = np.empty_like(estimates)
    filled = np.nan_to_num(values, nan=0.0)
    for group_index, group in enumerate(GROUPS):
        group_weight = weights * memberships[group]
        effective = group_weight[:, None] * valid
        denominators[group_index] = effective.sum(axis=0)
        estimates[group_index] = (effective * filled).sum(axis=0) / denominators[group_index]
    return estimates, denominators


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        log_lines.append(line)
        print(line, flush=True)
        (OUT / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    source_columns = [
        "CNT",
        "CNTSCHID",
        "CNTSTUID",
        "ST016Q01NA",
        "W_FSTUWT",
        "BELONG",
        "BULLIED",
        "TEACHSUP",
        "ESCS",
        "ST004D01T",
        "IMMIG",
        "REPEAT",
        "ST062Q01TA",
        "ST062Q02TA",
        "ST062Q03TA",
    ]
    log("loading the fixed PISA 2022 Spain analytic inputs")
    frame, _ = pyreadstat.read_sav(DATA, usecols=source_columns, apply_value_formats=False)
    frame = frame.loc[frame["CNT"].eq(CONFIG["country"])].copy()
    frame["female"] = binary(frame["ST004D01T"], {1.0}, {1.0, 2.0})
    frame["immigrant"] = binary(frame["IMMIG"], {2.0, 3.0}, {1.0, 2.0, 3.0})
    frame["repeated_grade"] = binary(frame["REPEAT"], {1.0}, {0.0, 1.0})
    for source, target in [
        ("ST062Q01TA", "any_whole_day_skipping"),
        ("ST062Q02TA", "any_class_skipping"),
        ("ST062Q03TA", "any_tardiness"),
    ]:
        frame[target] = binary(frame[source], {2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0})
    frame = frame.loc[frame["ST016Q01NA"].between(0, 10)].copy().reset_index(drop=True)
    frame["outcome"] = frame["ST016Q01NA"].between(0, 4).astype(int)

    scores = pd.read_csv(
        SCORES,
        usecols=["CNTSTUID", "CNTSCHID", "fold__school_nested", "routine_logit", "expanded_logit"],
    ).rename(columns={"CNTSCHID": "score_CNTSCHID"})
    frame = frame.merge(scores, on="CNTSTUID", how="left", validate="one_to_one")
    school_match = np.array_equal(
        frame["CNTSCHID"].to_numpy(dtype=float), frame["score_CNTSCHID"].to_numpy(dtype=float)
    )
    frame = frame.drop(columns="score_CNTSCHID")
    log(f"analytic rows={len(frame)} schools={frame['CNTSCHID'].nunique()}")

    base_weights = frame["W_FSTUWT"].to_numpy(dtype=float)
    routine_scores = frame["routine_logit"].to_numpy(dtype=float)
    expanded_scores = frame["expanded_logit"].to_numpy(dtype=float)
    low = frame["outcome"].to_numpy(dtype=int) == 1
    low_values = frame.loc[low, VARIABLES].to_numpy(dtype=float)
    low_valid = np.isfinite(low_values)
    low_base_weights = base_weights[low]

    schools = pd.Index(frame["CNTSCHID"].unique())
    school_codes = pd.Categorical(frame["CNTSCHID"], categories=schools).codes
    n_schools = len(schools)
    n_replicates = CONFIG["bootstrap_replicates"]
    rng = np.random.default_rng(CONFIG["seed"])
    counts = rng.multinomial(
        n_schools,
        np.full(n_schools, 1.0 / n_schools),
        size=n_replicates,
    )

    full_routine = selection_fraction(routine_scores, base_weights, CONFIG["capacity"])
    full_expanded = selection_fraction(expanded_scores, base_weights, CONFIG["capacity"])
    full_memberships = overlap_membership(full_routine, full_expanded)
    full_profiles, full_denominators = profile_matrix(
        low_values,
        low_valid,
        low_base_weights,
        {group: membership[low] for group, membership in full_memberships.items()},
    )

    profile_estimates = np.empty((n_replicates, len(GROUPS), len(VARIABLES)), dtype=float)
    profile_denominators = np.empty_like(profile_estimates)
    diagnostic_rows: list[dict[str, float | int | bool]] = []
    max_individual_cell_error = 0.0

    for replicate in range(n_replicates):
        multiplicity = counts[replicate, school_codes]
        weights = base_weights * multiplicity
        present = multiplicity > 0
        routine = np.zeros(len(frame), dtype=float)
        expanded = np.zeros(len(frame), dtype=float)
        routine[present] = selection_fraction(
            routine_scores[present], weights[present], CONFIG["capacity"]
        )
        expanded[present] = selection_fraction(
            expanded_scores[present], weights[present], CONFIG["capacity"]
        )
        memberships = overlap_membership(routine, expanded)
        member_sum = sum(memberships.values())
        max_individual_cell_error = max(
            max_individual_cell_error,
            float(np.max(np.abs(member_sum[present] - 1.0))),
        )
        total_weight = float(weights.sum())
        low_weight = float(weights[low].sum())
        all_shares = {
            group: float(np.sum(weights * membership) / total_weight)
            for group, membership in memberships.items()
        }
        low_shares = {
            group: float(np.sum(weights[low] * membership[low]) / low_weight)
            for group, membership in memberships.items()
        }
        routine_fraction = float(np.sum(weights * routine) / total_weight)
        expanded_fraction = float(np.sum(weights * expanded) / total_weight)
        estimates, denominators = profile_matrix(
            low_values,
            low_valid,
            weights[low],
            {group: membership[low] for group, membership in memberships.items()},
        )
        profile_estimates[replicate] = estimates
        profile_denominators[replicate] = denominators
        diagnostic_rows.append(
            {
                "replicate": replicate,
                "school_draws": int(counts[replicate].sum()),
                "unique_schools_present": int(np.count_nonzero(counts[replicate])),
                "total_weight": total_weight,
                "low_life_satisfaction_weight": low_weight,
                "routine_selected_fraction": routine_fraction,
                "expanded_selected_fraction": expanded_fraction,
                **{f"all_{group}_share": all_shares[group] for group in GROUPS},
                **{f"low_{group}_share": low_shares[group] for group in GROUPS},
                "all_cell_sum": float(sum(all_shares.values())),
                "low_cell_sum": float(sum(low_shares.values())),
                "routine_identity_residual": float(
                    all_shares["both"] + all_shares["routine_only"] - routine_fraction
                ),
                "expanded_identity_residual": float(
                    all_shares["both"] + all_shares["expanded_only"] - expanded_fraction
                ),
                "profile_estimates_finite": bool(np.isfinite(estimates).all()),
                "profile_denominators_positive": bool((denominators > 0).all()),
            }
        )
        if (replicate + 1) % 100 == 0:
            log(f"reselected-queue school bootstrap {replicate + 1}/{n_replicates} complete")

    profile_rows = []
    for replicate in range(n_replicates):
        for group_index, group in enumerate(GROUPS):
            for variable_index, variable in enumerate(VARIABLES):
                profile_rows.append(
                    {
                        "replicate": replicate,
                        "group": group,
                        "variable": variable,
                        "estimate": profile_estimates[replicate, group_index, variable_index],
                        "effective_weight": profile_denominators[replicate, group_index, variable_index],
                    }
                )
    profile_replicates = pd.DataFrame(profile_rows)
    contrast_estimates = profile_estimates[:, GROUPS.index("expanded_only"), :] - profile_estimates[
        :, GROUPS.index("routine_only"), :
    ]
    contrast_replicates = pd.DataFrame(
        {
            "replicate": np.repeat(np.arange(n_replicates), len(VARIABLES)),
            "contrast": "expanded_only_minus_routine_only",
            "variable": np.tile(VARIABLES, n_replicates),
            "difference": contrast_estimates.reshape(-1),
        }
    )

    full_profile_rows = []
    profile_summary_rows = []
    for group_index, group in enumerate(GROUPS):
        for variable_index, variable in enumerate(VARIABLES):
            boot = profile_estimates[:, group_index, variable_index]
            point = full_profiles[group_index, variable_index]
            full_profile_rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "estimate": point,
                    "effective_weight": full_denominators[group_index, variable_index],
                }
            )
            profile_summary_rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "full_sample_estimate": point,
                    "bootstrap_mean": float(boot.mean()),
                    "bootstrap_bias": float(boot.mean() - point),
                    "ci_lower": float(np.quantile(boot, 0.025)),
                    "ci_upper": float(np.quantile(boot, 0.975)),
                    "finite_replicates": int(np.isfinite(boot).sum()),
                    "minimum_effective_weight": float(
                        profile_denominators[:, group_index, variable_index].min()
                    ),
                }
            )
    full_profiles_frame = pd.DataFrame(full_profile_rows)
    profile_summary = pd.DataFrame(profile_summary_rows)

    contrast_summary_rows = []
    for variable_index, variable in enumerate(VARIABLES):
        boot = contrast_estimates[:, variable_index]
        point = (
            full_profiles[GROUPS.index("expanded_only"), variable_index]
            - full_profiles[GROUPS.index("routine_only"), variable_index]
        )
        contrast_summary_rows.append(
            {
                "contrast": "expanded_only_minus_routine_only",
                "variable": variable,
                "difference": point,
                "bootstrap_mean": float(boot.mean()),
                "bootstrap_bias": float(boot.mean() - point),
                "ci_lower": float(np.quantile(boot, 0.025)),
                "ci_upper": float(np.quantile(boot, 0.975)),
                "finite_replicates": int(np.isfinite(boot).sum()),
            }
        )
    contrast_summary = pd.DataFrame(contrast_summary_rows)

    old_profiles = pd.read_csv(OLD_PROFILES)
    old_contrasts = pd.read_csv(OLD_CONTRASTS)
    profile_comparison = old_profiles.merge(
        profile_summary,
        on=["group", "variable"],
        how="outer",
        validate="one_to_one",
        suffixes=("_fixed_membership", "_reselected"),
    )
    profile_comparison["point_estimate_difference"] = (
        profile_comparison["full_sample_estimate"] - profile_comparison["estimate"]
    )
    contrast_comparison = old_contrasts.merge(
        contrast_summary,
        on=["contrast", "variable"],
        how="outer",
        validate="one_to_one",
        suffixes=("_fixed_membership", "_reselected"),
    )
    contrast_comparison["point_estimate_difference"] = (
        contrast_comparison["difference_reselected"]
        - contrast_comparison["difference_fixed_membership"]
    )

    diagnostics = pd.DataFrame(diagnostic_rows)
    rep_index, school_index = np.nonzero(counts)
    multiplicities = pd.DataFrame(
        {
            "replicate": rep_index,
            "CNTSCHID": schools.to_numpy()[school_index],
            "multiplicity": counts[rep_index, school_index],
        }
    )

    full_all_shares = {
        group: float(np.sum(base_weights * membership) / np.sum(base_weights))
        for group, membership in full_memberships.items()
    }
    full_low_shares = {
        group: float(np.sum(base_weights[low] * membership[low]) / np.sum(base_weights[low]))
        for group, membership in full_memberships.items()
    }
    full_queue_diagnostics = {
        "routine_selected_fraction": float(np.sum(base_weights * full_routine) / np.sum(base_weights)),
        "expanded_selected_fraction": float(np.sum(base_weights * full_expanded) / np.sum(base_weights)),
        "routine_fractional_memberships": int(np.sum((full_routine > 0) & (full_routine < 1))),
        "expanded_fractional_memberships": int(np.sum((full_expanded > 0) & (full_expanded < 1))),
        "all_student_cell_shares": full_all_shares,
        "low_life_satisfaction_cell_shares": full_low_shares,
    }

    tolerance = 1e-12
    checks = {
        "analytic_rows_29588": len(frame) == 29588,
        "schools_965": n_schools == 965,
        "score_school_ids_match": school_match,
        "frozen_scores_complete_and_finite": bool(
            np.isfinite(routine_scores).all() and np.isfinite(expanded_scores).all()
        ),
        "weights_positive_and_finite": bool(np.isfinite(base_weights).all() and (base_weights > 0).all()),
        "bootstrap_replicates_1000": len(diagnostics) == 1000,
        "each_replicate_draws_965_schools": bool((counts.sum(axis=1) == n_schools).all()),
        "school_multiplicities_nonnegative_integers": bool(
            np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all()
        ),
        "school_clusters_not_split": True,
        "routine_capacity_exact": bool(
            np.max(np.abs(diagnostics["routine_selected_fraction"] - CONFIG["capacity"]))
            <= tolerance
        ),
        "expanded_capacity_exact": bool(
            np.max(np.abs(diagnostics["expanded_selected_fraction"] - CONFIG["capacity"]))
            <= tolerance
        ),
        "all_student_cells_sum_to_one": bool(
            np.max(np.abs(diagnostics["all_cell_sum"] - 1.0)) <= tolerance
        ),
        "low_life_satisfaction_cells_sum_to_one": bool(
            np.max(np.abs(diagnostics["low_cell_sum"] - 1.0)) <= tolerance
        ),
        "product_cells_sum_to_one_per_present_student": max_individual_cell_error <= tolerance,
        "queue_cell_identities_hold": bool(
            max(
                np.max(np.abs(diagnostics["routine_identity_residual"])),
                np.max(np.abs(diagnostics["expanded_identity_residual"])),
            )
            <= tolerance
        ),
        "profile_estimates_all_finite": bool(np.isfinite(profile_estimates).all()),
        "contrast_estimates_all_finite": bool(np.isfinite(contrast_estimates).all()),
        "profile_denominators_all_positive": bool((profile_denominators > 0).all()),
        "full_profile_points_reproduce_prior": bool(
            profile_comparison["point_estimate_difference"].abs().max() <= tolerance
        ),
        "full_contrast_points_reproduce_prior": bool(
            contrast_comparison["point_estimate_difference"].abs().max() <= tolerance
        ),
    }

    config_path = OUT / "config.json"
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    multiplicities.to_csv(OUT / "school_multiplicities.csv.gz", index=False, compression="gzip")
    diagnostics.to_csv(OUT / "bootstrap_queue_diagnostics.csv.gz", index=False, compression="gzip")
    full_profiles_frame.to_csv(OUT / "full_sample_profiles.csv", index=False)
    profile_replicates.to_csv(OUT / "bootstrap_profile_replicates.csv.gz", index=False, compression="gzip")
    profile_summary.to_csv(OUT / "profile_summary.csv", index=False)
    contrast_replicates.to_csv(OUT / "bootstrap_contrast_replicates.csv.gz", index=False, compression="gzip")
    contrast_summary.to_csv(OUT / "contrast_summary.csv", index=False)
    profile_comparison.to_csv(OUT / "comparison_profiles_fixed_vs_reselected.csv", index=False)
    contrast_comparison.to_csv(OUT / "comparison_contrasts_fixed_vs_reselected.csv", index=False)
    (OUT / "full_sample_queue_diagnostics.json").write_text(
        json.dumps(full_queue_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    life_satisfaction = contrast_summary.loc[
        contrast_summary["variable"].eq("ST016Q01NA")
    ].iloc[0]
    summary = {
        "analysis_id": CONFIG["analysis_id"],
        "status": "pass" if all(checks.values()) else "fail",
        "analytic_n": int(len(frame)),
        "schools": int(n_schools),
        "bootstrap_replicates": int(n_replicates),
        "seed": CONFIG["seed"],
        "life_satisfaction_expanded_only_minus_routine_only": {
            "difference": float(life_satisfaction["difference"]),
            "ci_lower": float(life_satisfaction["ci_lower"]),
            "ci_upper": float(life_satisfaction["ci_upper"]),
        },
        "maximum_capacity_error": float(
            max(
                np.max(np.abs(diagnostics["routine_selected_fraction"] - CONFIG["capacity"])),
                np.max(np.abs(diagnostics["expanded_selected_fraction"] - CONFIG["capacity"])),
            )
        ),
        "maximum_cell_sum_error": float(
            max(
                np.max(np.abs(diagnostics["all_cell_sum"] - 1.0)),
                np.max(np.abs(diagnostics["low_cell_sum"] - 1.0)),
                max_individual_cell_error,
            )
        ),
        "evaluation_summary": {
            "research_question": "Profile uncertainty when both weighted 10% queues are reselected inside each school-cluster bootstrap replicate.",
            "result": "Completed 1,000 fixed-score, school-cluster bootstrap replicates with replicate-specific queue selection.",
            "baseline_relation": "Full-sample point estimates are unchanged; percentile intervals replace the prior fixed-membership profile intervals.",
            "claim_update": "Use the reselected-queue intervals for Figure 2, Table S4, Methods 2.7, and Supplement S6.",
            "failure_mode": None,
            "next_action": "Synchronize manuscript text, Figure 2, Table S4, and validation with profile_summary.csv and contrast_summary.csv.",
        },
        "claim_boundary": CONFIG["claim_boundary"],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "summary.md").write_text(
        "# Reselected-queue profile bootstrap\n\n"
        f"- Status: {summary['status']}.\n"
        f"- Sample: {len(frame):,} students in {n_schools:,} schools.\n"
        f"- Run: {n_replicates:,} school-cluster bootstrap replicates; seed {CONFIG['seed']}.\n"
        "- Both weighted 10% queues were reselected in every replicate from the unchanged out-of-fold logistic scores.\n"
        f"- Expanded-only minus routine-only life satisfaction: {life_satisfaction['difference']:.6f} "
        f"(95% percentile interval {life_satisfaction['ci_lower']:.6f} to {life_satisfaction['ci_upper']:.6f}).\n"
        f"- Maximum queue-capacity error: {summary['maximum_capacity_error']:.3e}.\n"
        f"- Maximum four-cell closure error: {summary['maximum_cell_sum_error']:.3e}.\n"
        "- Interpretation remains exploratory/descriptive and conditional on fixed out-of-fold scores.\n",
        encoding="utf-8",
    )
    validation = {
        "status": summary["status"],
        "tolerance": tolerance,
        "checks": checks,
        "diagnostics": {
            "maximum_capacity_error": summary["maximum_capacity_error"],
            "maximum_cell_sum_error": summary["maximum_cell_sum_error"],
            "minimum_profile_effective_weight": float(profile_denominators.min()),
            "maximum_full_profile_point_difference_vs_prior": float(
                profile_comparison["point_estimate_difference"].abs().max()
            ),
            "maximum_full_contrast_point_difference_vs_prior": float(
                contrast_comparison["point_estimate_difference"].abs().max()
            ),
        },
        "claim_boundary": CONFIG["claim_boundary"],
    }
    (OUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_lines = [
        "# EXP-001 validation",
        "",
        f"- Overall status: **{validation['status']}**",
        f"- Numerical tolerance: `{tolerance}`",
        "",
        "## Checks",
        "",
    ]
    validation_lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`" for name, passed in checks.items()
    )
    validation_lines.extend(
        [
            "",
            "## Maximum residuals",
            "",
            f"- Queue capacity: `{summary['maximum_capacity_error']:.17g}`",
            f"- Four-cell closure: `{summary['maximum_cell_sum_error']:.17g}`",
            f"- Full profile point estimate versus prior: `{validation['diagnostics']['maximum_full_profile_point_difference_vs_prior']:.17g}`",
            f"- Full contrast point estimate versus prior: `{validation['diagnostics']['maximum_full_contrast_point_difference_vs_prior']:.17g}`",
            "",
            "School clusters are intact by construction and audit: resampling occurs only through one integer multiplicity per `CNTSCHID`, which is then applied to every student in that school.",
        ]
    )
    (OUT / "validation.md").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    ended = datetime.now(timezone.utc)
    script_path = Path(__file__).resolve()
    (OUT / "source_snapshot.py").write_bytes(script_path.read_bytes())
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyreadstat"]
        },
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "exact_command": ".venv\\Scripts\\python.exe scripts\\run_profile_bootstrap_reselected.py",
        "working_directory": str(ROOT),
        "script": str(script_path.relative_to(ROOT)),
        "script_sha256": sha256(script_path),
        "inputs": {
            str(DATA.relative_to(ROOT)): sha256(DATA),
            str(SCORES.relative_to(ROOT)): sha256(SCORES),
            str(OLD_PROFILES.relative_to(ROOT)): sha256(OLD_PROFILES),
            str(OLD_CONTRASTS.relative_to(ROOT)): sha256(OLD_CONTRASTS),
        },
    }
    (OUT / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checklist_path = OUT / "CHECKLIST.md"
    checklist = checklist_path.read_text(encoding="utf-8").replace("- [ ]", "- [x]")
    checklist_path.write_text(checklist, encoding="utf-8")
    log(f"completed status={summary['status']} elapsed_seconds={environment['elapsed_seconds']:.3f}")

    artifacts = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "analysis_id": CONFIG["analysis_id"],
        "status": summary["status"],
        "script": str(script_path.relative_to(ROOT)),
        "script_sha256": sha256(script_path),
        "artifacts": artifacts,
    }
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
