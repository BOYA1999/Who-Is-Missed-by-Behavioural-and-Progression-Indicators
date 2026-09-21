# Independent validation of EXP-001

- Status: **pass**

- [x] `config_replicates_1000`
- [x] `multiplicity_replicates_complete`
- [x] `each_replicate_has_965_school_draws`
- [x] `multiplicities_are_positive_integers`
- [x] `diagnostic_replicates_complete`
- [x] `routine_capacity_exact`
- [x] `expanded_capacity_exact`
- [x] `all_student_cells_close`
- [x] `low_life_satisfaction_cells_close`
- [x] `queue_identities_close`
- [x] `profile_shape_complete`
- [x] `profile_groups_complete`
- [x] `profile_estimates_finite`
- [x] `profile_weights_positive_finite`
- [x] `contrast_shape_complete`
- [x] `contrast_estimates_finite`
- [x] `contrasts_reconstruct_from_profiles`
- [x] `profile_percentile_intervals_reconstruct`
- [x] `profile_full_points_reconstruct`
- [x] `contrast_percentile_intervals_reconstruct`
- [x] `recorded_input_hashes_match`
- [x] `recorded_script_hash_matches`
- [x] `source_snapshot_matches_script`
- [x] `prevalidation_manifest_hashes_match`

The replicate files independently reproduce the recorded percentile intervals and every expanded-only minus routine-only contrast. School resampling is represented by one saved integer multiplicity per present `CNTSCHID` and replicate; all 1,000 replicate multiplicity totals equal 965 school draws.
