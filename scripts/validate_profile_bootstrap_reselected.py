from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "profile_bootstrap_reselected_v1"
TOLERANCE = 1e-12
GROUPS = ["both", "routine_only", "expanded_only", "neither"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((OUT / "environment.json").read_text(encoding="utf-8"))
    existing_manifest = json.loads((OUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    multiplicities = pd.read_csv(OUT / "school_multiplicities.csv.gz")
    diagnostics = pd.read_csv(OUT / "bootstrap_queue_diagnostics.csv.gz")
    profiles = pd.read_csv(OUT / "bootstrap_profile_replicates.csv.gz")
    profile_summary = pd.read_csv(OUT / "profile_summary.csv")
    full_profiles = pd.read_csv(OUT / "full_sample_profiles.csv")
    contrasts = pd.read_csv(OUT / "bootstrap_contrast_replicates.csv.gz")
    contrast_summary = pd.read_csv(OUT / "contrast_summary.csv")

    manifest_hashes_match = all(
        (ROOT / item["path"]).is_file()
        and (ROOT / item["path"]).stat().st_size == item["bytes"]
        and sha256(ROOT / item["path"]) == item["sha256"]
        for item in existing_manifest["artifacts"]
    )
    input_hashes_match = all(
        (ROOT / path).is_file() and sha256(ROOT / path) == digest
        for path, digest in environment["inputs"].items()
    )
    script_path = ROOT / environment["script"]
    source_snapshot = OUT / "source_snapshot.py"

    draws = multiplicities.groupby("replicate", sort=True)["multiplicity"].sum()
    counts_integer = np.allclose(
        multiplicities["multiplicity"], np.round(multiplicities["multiplicity"]), atol=0, rtol=0
    )
    expected_replicates = np.arange(config["bootstrap_replicates"])
    diagnostic_replicates = diagnostics["replicate"].to_numpy(dtype=int)

    profile_wide = profiles.pivot(
        index=["replicate", "variable"], columns="group", values="estimate"
    )
    recomputed_contrast = (
        profile_wide["expanded_only"] - profile_wide["routine_only"]
    ).rename("recomputed_difference").reset_index()
    contrast_check = contrasts.merge(
        recomputed_contrast,
        on=["replicate", "variable"],
        how="outer",
        validate="one_to_one",
    )
    max_contrast_reconstruction_error = float(
        np.max(np.abs(contrast_check["difference"] - contrast_check["recomputed_difference"]))
    )

    profile_quantiles = (
        profiles.groupby(["group", "variable"])["estimate"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "recomputed_lower", 0.975: "recomputed_upper"})
    )
    profile_check = (
        profile_summary.merge(
            profile_quantiles,
            on=["group", "variable"],
            how="outer",
            validate="one_to_one",
        )
        .merge(
            full_profiles[["group", "variable", "estimate"]],
            on=["group", "variable"],
            how="outer",
            validate="one_to_one",
        )
    )
    max_profile_ci_error = float(
        max(
            np.max(np.abs(profile_check["ci_lower"] - profile_check["recomputed_lower"])),
            np.max(np.abs(profile_check["ci_upper"] - profile_check["recomputed_upper"])),
        )
    )
    max_profile_point_error = float(
        np.max(np.abs(profile_check["full_sample_estimate"] - profile_check["estimate"]))
    )

    contrast_quantiles = (
        contrasts.groupby(["contrast", "variable"])["difference"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "recomputed_lower", 0.975: "recomputed_upper"})
    )
    contrast_summary_check = contrast_summary.merge(
        contrast_quantiles,
        on=["contrast", "variable"],
        how="outer",
        validate="one_to_one",
    )
    max_contrast_ci_error = float(
        max(
            np.max(
                np.abs(
                    contrast_summary_check["ci_lower"]
                    - contrast_summary_check["recomputed_lower"]
                )
            ),
            np.max(
                np.abs(
                    contrast_summary_check["ci_upper"]
                    - contrast_summary_check["recomputed_upper"]
                )
            ),
        )
    )

    checks = {
        "config_replicates_1000": config["bootstrap_replicates"] == 1000,
        "multiplicity_replicates_complete": np.array_equal(draws.index.to_numpy(), expected_replicates),
        "each_replicate_has_965_school_draws": bool((draws == 965).all()),
        "multiplicities_are_positive_integers": bool(
            counts_integer and (multiplicities["multiplicity"] > 0).all()
        ),
        "diagnostic_replicates_complete": np.array_equal(diagnostic_replicates, expected_replicates),
        "routine_capacity_exact": bool(
            np.max(np.abs(diagnostics["routine_selected_fraction"] - 0.10)) <= TOLERANCE
        ),
        "expanded_capacity_exact": bool(
            np.max(np.abs(diagnostics["expanded_selected_fraction"] - 0.10)) <= TOLERANCE
        ),
        "all_student_cells_close": bool(
            np.max(np.abs(diagnostics["all_cell_sum"] - 1.0)) <= TOLERANCE
        ),
        "low_life_satisfaction_cells_close": bool(
            np.max(np.abs(diagnostics["low_cell_sum"] - 1.0)) <= TOLERANCE
        ),
        "queue_identities_close": bool(
            max(
                np.max(np.abs(diagnostics["routine_identity_residual"])),
                np.max(np.abs(diagnostics["expanded_identity_residual"])),
            )
            <= TOLERANCE
        ),
        "profile_shape_complete": len(profiles) == 1000 * 4 * 11,
        "profile_groups_complete": set(profiles["group"]) == set(GROUPS),
        "profile_estimates_finite": bool(np.isfinite(profiles["estimate"]).all()),
        "profile_weights_positive_finite": bool(
            np.isfinite(profiles["effective_weight"]).all()
            and (profiles["effective_weight"] > 0).all()
        ),
        "contrast_shape_complete": len(contrasts) == 1000 * 11,
        "contrast_estimates_finite": bool(np.isfinite(contrasts["difference"]).all()),
        "contrasts_reconstruct_from_profiles": max_contrast_reconstruction_error <= TOLERANCE,
        "profile_percentile_intervals_reconstruct": max_profile_ci_error <= TOLERANCE,
        "profile_full_points_reconstruct": max_profile_point_error <= TOLERANCE,
        "contrast_percentile_intervals_reconstruct": max_contrast_ci_error <= TOLERANCE,
        "recorded_input_hashes_match": input_hashes_match,
        "recorded_script_hash_matches": sha256(script_path) == environment["script_sha256"],
        "source_snapshot_matches_script": sha256(source_snapshot) == sha256(script_path),
        "prevalidation_manifest_hashes_match": manifest_hashes_match,
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "diagnostics": {
            "maximum_routine_capacity_error": float(
                np.max(np.abs(diagnostics["routine_selected_fraction"] - 0.10))
            ),
            "maximum_expanded_capacity_error": float(
                np.max(np.abs(diagnostics["expanded_selected_fraction"] - 0.10))
            ),
            "maximum_all_student_cell_sum_error": float(
                np.max(np.abs(diagnostics["all_cell_sum"] - 1.0))
            ),
            "maximum_low_life_satisfaction_cell_sum_error": float(
                np.max(np.abs(diagnostics["low_cell_sum"] - 1.0))
            ),
            "maximum_contrast_reconstruction_error": max_contrast_reconstruction_error,
            "maximum_profile_ci_reconstruction_error": max_profile_ci_error,
            "maximum_profile_point_reconstruction_error": max_profile_point_error,
            "maximum_contrast_ci_reconstruction_error": max_contrast_ci_error,
        },
    }
    (OUT / "independent_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Independent validation of EXP-001",
        "",
        f"- Status: **{result['status']}**",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in checks.items())
    lines.extend(
        [
            "",
            "The replicate files independently reproduce the recorded percentile intervals and every expanded-only minus routine-only contrast. School resampling is represented by one saved integer multiplicity per present `CNTSCHID` and replicate; all 1,000 replicate multiplicity totals equal 965 school draws.",
        ]
    )
    (OUT / "independent_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

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
        "analysis_id": config["analysis_id"],
        "status": result["status"],
        "script": environment["script"],
        "script_sha256": environment["script_sha256"],
        "validator": str(Path(__file__).resolve().relative_to(ROOT)),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "artifacts": artifacts,
    }
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
