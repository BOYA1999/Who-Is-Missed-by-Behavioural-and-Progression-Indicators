# EXP-001 validation

- Overall status: **pass**
- Numerical tolerance: `1e-12`

## Checks

- [x] `analytic_rows_29588`
- [x] `schools_965`
- [x] `score_school_ids_match`
- [x] `frozen_scores_complete_and_finite`
- [x] `weights_positive_and_finite`
- [x] `bootstrap_replicates_1000`
- [x] `each_replicate_draws_965_schools`
- [x] `school_multiplicities_nonnegative_integers`
- [x] `school_clusters_not_split`
- [x] `routine_capacity_exact`
- [x] `expanded_capacity_exact`
- [x] `all_student_cells_sum_to_one`
- [x] `low_life_satisfaction_cells_sum_to_one`
- [x] `product_cells_sum_to_one_per_present_student`
- [x] `queue_cell_identities_hold`
- [x] `profile_estimates_all_finite`
- [x] `contrast_estimates_all_finite`
- [x] `profile_denominators_all_positive`
- [x] `full_profile_points_reproduce_prior`
- [x] `full_contrast_points_reproduce_prior`

## Maximum residuals

- Queue capacity: `4.4408920985006262e-16`
- Four-cell closure: `3.3306690738754696e-16`
- Full profile point estimate versus prior: `8.8817841970012523e-16`
- Full contrast point estimate versus prior: `4.4408920985006262e-16`

School clusters are intact by construction and audit: resampling occurs only through one integer multiplicity per `CNTSCHID`, which is then applied to every student in that school.
