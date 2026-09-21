# Statistical audit for targeted revision v2

Audit date: 2026-09-20. Scope: existing outputs and code only; no new model fitted. Original manuscript, scripts and tables were not edited.

## Findings and immediate revisions

1. **Table 3 mixes estimands.** It displays model metrics evaluated on the full sample but displays their bootstrap-mean differences. This is why the BULLIED recall gain reads 0.072 rather than the actual 0.069328. The cause is `paper/tables/build_reviewer_revision_tables.py:48,54`, which formats `bootstrap_mean_delta`. That column is explicitly the mean across bootstrap contrasts in `scripts/run_reviewer_revision.py:200–217`. Replace the reported point with the full-sample paired difference; retain the percentile bootstrap interval. The same correction applies to prose using recall gain 0.103: the point is 0.102419. AP gain rounds to 0.104 either way.
2. **Precision is not an independent fixed-capacity finding.** Precision = prevalence × recall / capacity holds for every model, up to floating-point error. At the common prevalence 0.1457653164 and q=0.10, precision = 1.457653164 × recall. Report precision for interpretability, without treating it as separate convergent evidence.
3. **Overlap uses fractional allocation.** The four groups are algebraic weighted memberships, not always integer student counts. Report the rule explicitly and include uncertainty.
4. **HGB has a small positive paired gain.** Its paired intervals exclude zero for AP and recall in both school bootstrap and BRR. A statement of no clear gain would not match these results. A statement of modest gain with added complexity is supported.
5. **Regional capacity is local to each held-out region.** Figure S1 recall does not evaluate a single pooled national threshold. Pooled region-held-out metrics use the pooled 10% threshold; region-specific figure estimates select 10% within each region. Both are valid but must be distinguished.
6. **Existing complete-case sensitivity has a narrower scope than a full information-set test.** It compares expanded models retrained on 27,937 complete cases against main expanded-model scores restricted to the same cases. It does not refit routine models on the complete-case sample, so it cannot by itself demonstrate persistence of the routine-versus-expanded contrast.

## Correct point estimates with existing paired school-bootstrap 95% intervals

All differences below are proportions, not percentage points. Multiply recall/precision differences by 100 for percentage-point wording. Source: `artifacts/analysis/reviewer_revision_v1/model_metrics.csv` and `bootstrap_deltas.csv`; intervals condition on the fixed out-of-fold scores and reselect the queue under school resampling.

| Model minus reference | Metric | Correct point | 95% interval |
|---|---|---:|---|
| routine_plus_belong_logit − routine_logit | weighted_average_precision | 0.054710 | [0.045495, 0.064704] |
| routine_plus_belong_logit − routine_logit | recall_at_10pct_capacity | 0.042733 | [0.029304, 0.060080] |
| routine_plus_belong_logit − routine_logit | precision_at_10pct_capacity | 0.062290 | [0.043001, 0.086860] |
| routine_plus_bullied_logit − routine_logit | weighted_average_precision | 0.068209 | [0.056103, 0.081223] |
| routine_plus_bullied_logit − routine_logit | recall_at_10pct_capacity | 0.069328 | [0.055392, 0.089183] |
| routine_plus_bullied_logit − routine_logit | precision_at_10pct_capacity | 0.101056 | [0.081275, 0.130283] |
| routine_plus_teachsup_logit − routine_logit | weighted_average_precision | 0.008331 | [0.003539, 0.013074] |
| routine_plus_teachsup_logit − routine_logit | recall_at_10pct_capacity | 0.006749 | [-0.003554, 0.019443] |
| routine_plus_teachsup_logit − routine_logit | precision_at_10pct_capacity | 0.009837 | [-0.005027, 0.028554] |
| experiences_only_logit − routine_logit | weighted_average_precision | 0.098670 | [0.082291, 0.114782] |
| experiences_only_logit − routine_logit | recall_at_10pct_capacity | 0.094298 | [0.072861, 0.115419] |
| experiences_only_logit − routine_logit | precision_at_10pct_capacity | 0.137454 | [0.106385, 0.169257] |
| expanded_logit − routine_logit | weighted_average_precision | 0.103730 | [0.090258, 0.117246] |
| expanded_logit − routine_logit | recall_at_10pct_capacity | 0.102419 | [0.084549, 0.121697] |
| expanded_logit − routine_logit | precision_at_10pct_capacity | 0.149291 | [0.125242, 0.178787] |
| expanded_logit − experiences_only_logit | weighted_average_precision | 0.005060 | [-0.001217, 0.010962] |
| expanded_logit − experiences_only_logit | recall_at_10pct_capacity | 0.008121 | [-0.003269, 0.020576] |
| expanded_logit − experiences_only_logit | precision_at_10pct_capacity | 0.011837 | [-0.004670, 0.030143] |
| expanded_altbully_logit − routine_logit | weighted_average_precision | 0.090623 | [0.077687, 0.102968] |
| expanded_altbully_logit − routine_logit | recall_at_10pct_capacity | 0.074653 | [0.057110, 0.092801] |
| expanded_altbully_logit − routine_logit | precision_at_10pct_capacity | 0.108818 | [0.083642, 0.134407] |
| expanded_altbully_logit − expanded_logit | weighted_average_precision | -0.013107 | [-0.017270, -0.009233] |
| expanded_altbully_logit − expanded_logit | recall_at_10pct_capacity | -0.027766 | [-0.037322, -0.020590] |
| expanded_altbully_logit − expanded_logit | precision_at_10pct_capacity | -0.040473 | [-0.054117, -0.030039] |
| expanded_hgb_no_earlystop − expanded_hgb | weighted_average_precision | -0.000618 | [-0.005888, 0.004967] |
| expanded_hgb_no_earlystop − expanded_hgb | recall_at_10pct_capacity | 0.000774 | [-0.007724, 0.007282] |
| expanded_hgb_no_earlystop − expanded_hgb | precision_at_10pct_capacity | 0.001128 | [-0.011233, 0.010677] |

## HGB versus expanded logistic regression

Recalculated by subtracting models within the same replicate IDs in `reviewer_revision_v1/bootstrap_metrics.csv.gz` (1,000 paired replicates) and `brr_performance_replicates.csv.gz` (80 paired replicates). Full-sample points come from model_metrics.csv. Fay-BRR SE = sqrt(0.05 × sum((replicate contrast − full contrast)^2)); interval = point ± 1.96 SE. This uses the same source run as Table 3. The earlier main_v1 paired_bootstrap_deltas.csv has slightly different percentile endpoints because it is a different bootstrap run; do not mix runs silently.

| Metric | Point | School bootstrap 95% CI | Fay-BRR 95% CI |
|---|---:|---|---|
| weighted_average_precision | 0.025461 | [0.014329, 0.036146] | [0.015241, 0.035682] |
| recall_at_10pct_capacity | 0.015571 | [0.002002, 0.028923] | [0.002814, 0.028329] |
| precision_at_10pct_capacity | 0.022698 | [0.002973, 0.042456] | [0.004113, 0.041283] |

## Fractional queue and four-cell definition

Evidence: `scripts/run_pilot.py:39–60`; `scripts/run_reviewer_revision.py:303–344`.

For model m, total target weight is qΣw. Scores are sorted descending. Let t be the boundary score, W_above the weight above t, and W_tie the weight exactly equal to t. For every student in the tied boundary group:
c_im = (qΣw − W_above) / W_tie.
Above the boundary c_im=1; below it c_im=0. This ensures Σw c_im / Σw=q, with deterministic proportional allocation of boundary weight.

Let c_R and c_E denote routine and expanded allocation fractions. The implementation defines:
- both = c_R c_E;
- routine only = c_R(1−c_E);
- expanded only = (1−c_R)c_E;
- neither = (1−c_R)(1−c_E).

These sum to one per student. The product convention corresponds to expected overlap under independent random tie resolution where both allocations are fractional; it should not be described as observed attendance, literal mutually exclusive student counts, or a realised random allocation. In this specific sample the boundary groups are tiny. Boundary shares were independently reconstructed from `main_v1/oof_predictions.csv.gz`, using the cumulative weighted rank boundary rather than importing the production helper.

| Model | Boundary students | Boundary weight as % of population | Allocation fraction c | Allocated boundary weight as % of population |
|---|---:|---:|---:|---:|
| routine_logit | 11 | 0.075693 | 0.548317 | 0.041504 |
| expanded_logit | 1 | 0.002244 | 0.915094 | 0.002053 |
| routine_hgb | 9 | 0.035272 | 0.149732 | 0.005281 |
| expanded_hgb | 1 | 0.011362 | 0.801510 | 0.009107 |

Existing Fay-BRR four-cell intervals are already in `reviewer_revision_v1/queue_overlap.csv`. Replicate queue selection is repeated under the 80 replicate weights, with fixed scores. All values below are percentages.

| Population | Cell | Estimate % | 95% CI % |
|---|---|---:|---|
| all_students | both | 3.626 | [3.401, 3.852] |
| all_students | routine_only | 6.374 | [6.148, 6.599] |
| all_students | expanded_only | 6.374 | [6.148, 6.599] |
| all_students | neither | 83.626 | [83.401, 83.852] |
| all_students | union | 16.374 | [16.148, 16.599] |
| low_life_satisfaction | both | 9.711 | [8.337, 11.086] |
| low_life_satisfaction | routine_only | 6.584 | [5.527, 7.641] |
| low_life_satisfaction | expanded_only | 16.826 | [15.130, 18.522] |
| low_life_satisfaction | neither | 66.878 | [65.333, 68.424] |
| low_life_satisfaction | union | 33.122 | [31.576, 34.667] |

Routine-covered low-life-satisfaction weight displaced = routine_only / (both + routine_only) = **40.405%**, Fay-BRR 95% CI **34.866–45.944%**. This interval was derived from the existing paired cell shares in `queue_overlap_brr.csv.gz`, applying the ratio within each replicate. Do not obtain this interval by dividing marginal CI endpoints.

## Precision identity checks

Source: reviewer_revision_v1/model_metrics.csv for nine models, plus main_v1/metrics.csv school_nested routine_hgb. The maximum absolute residual is 2.220e-16.

| Model | Reported precision | prevalence × recall / q |
|---|---:|---:|
| routine_logit | 0.237531 | 0.237531 |
| expanded_logit | 0.386822 | 0.386822 |
| expanded_hgb | 0.409520 | 0.409520 |
| routine_plus_belong_logit | 0.299821 | 0.299821 |
| routine_plus_bullied_logit | 0.338588 | 0.338588 |
| routine_plus_teachsup_logit | 0.247368 | 0.247368 |
| experiences_only_logit | 0.374985 | 0.374985 |
| expanded_altbully_logit | 0.346349 | 0.346349 |
| expanded_hgb_no_earlystop | 0.410648 | 0.410648 |
| routine_hgb | 0.230293 | 0.230293 |

## Existing sensitivity outputs for a supplementary table

Source: `artifacts/analysis/campaign_v1/slice_results.csv`, with per-student predictions in `sensitivity_predictions.csv.gz`. Model fitting and population definitions: `scripts/run_sensitivities.py:297–353`. Continuous analyses have n=29,588, mean life satisfaction 6.883132.

| Continuous model | MAE | RMSE | R² |
|---|---:|---:|---:|
| routine_ridge | 1.750278 | 2.245404 | 0.014775 |
| routine_hgb | 1.751644 | 2.244419 | 0.015640 |
| expanded_ridge | 1.641784 | 2.122236 | 0.119897 |
| expanded_hgb | 1.604762 | 2.083171 | 0.152000 |

| Sensitivity / model | n | Prevalence | AP | Recall@10% | Precision@10% | Brier |
|---|---:|---:|---:|---:|---:|---:|
| S2_complete_case_retrained / expanded_logit | 27937 | 0.140647 | 0.299944 | 0.270677 | 0.380698 | 0.110953 |
| S2_main_predictions_restricted / expanded_logit | 27937 | 0.140647 | 0.300015 | 0.273610 | 0.384824 | 0.110957 |
| S2_complete_case_retrained / expanded_hgb | 27937 | 0.140647 | 0.326778 | 0.300336 | 0.422413 | 0.107917 |
| S2_main_predictions_restricted / expanded_hgb | 27937 | 0.140647 | 0.327763 | 0.295791 | 0.416021 | 0.108126 |
| S3_audit_attributes / expanded_audit_logit | 29588 | 0.145765 | 0.328202 | 0.273861 | 0.399194 | 0.112245 |
| S3_audit_attributes / expanded_audit_hgb | 29588 | 0.145765 | 0.348629 | 0.293317 | 0.427555 | 0.110501 |

No CIs are supplied in this existing sensitivity summary; do not invent them. Calibration intercepts/slopes and other capacity points are included in the companion JSON. Audit-attribute sensitivities add sex, immigration and ESCS; the fairness breakdown exists in campaign_v1/audit_attribute_fairness.csv and reviewer_revision_v1/audit_attribute_logit.csv.

## Regional estimand and manuscript wording

`scripts/run_reviewer_revision.py:609–637` masks a single region before calling selection_fraction, for both full weights and each BRR replicate. `scripts/run_reviewer_revision.py:841–843` supplies expanded-logit **region-held-out** predictions. Hence describe Figure S1 as:
“Each region was evaluated on predictions from a model trained outside that region; recall was calculated after allocating 10% of student weight within that region. The queue was reselected within the region for each BRR replicate.”

By contrast, `scripts/run_main.py:426–434` computes pooled metrics from all predictions and weights together; `:437–446` calculates fold/region metrics separately. Distinguish pooled national-capacity performance from within-region-capacity heterogeneity.

## Recommended manuscript corrections

- Replace all delta point estimates with full-sample metric differences; use the exact values above before rounding. Update Table 3 recall gains to 0.043, 0.069, 0.007, 0.094, 0.102, 0.075 for BELONG, BULLIED, TEACHSUP, experiences only, expanded, alternative bullying respectively.
- Table 3 BULLIED AP difference becomes 0.068 (not 0.069). Other AP differences round to 0.055, 0.008, 0.099, 0.104, 0.091.
- Expanded minus experiences-only recall is 0.008121, CI −0.003269 to 0.020576; avoid the bootstrap mean 0.009043 as the point.
- State the capacity identity near the primary performance table, and use coverage as the principal fixed-capacity outcome.
- Report four-cell BRR intervals and displacement CI; define the fractional membership product convention.
- Report HGB paired gain and intervals, qualifying the gain by magnitude and interpretability.
- Place the available continuous, complete-case and audit-attribute results in a supplement table, retaining their actual comparison scope.
- Clarify regional 10% queue construction in methods and Figure S1 caption.

## Added audit-attribute paired contrast and HGB sensitivity rows

The complete-sample expanded_audit_logit minus expanded_logit contrast is AP **0.024785280**, 95% paired school-bootstrap CI **0.017116141 to 0.034044751**; recall **0.008487469**, CI **−0.003801228 to 0.021745187**. Thus the AP improvement is supported by this fixed-score interval while the recall difference at 10% capacity remains uncertain.

This additional calculation fits no model. It uses the saved audit_attributes__expanded_audit_logit scores in campaign_v1/sensitivity_predictions.csv.gz; original ordered CNTSTUID records match main_v1/oof_predictions.csv.gz and reviewer_revision_v1/revision_predictions.csv.gz exactly. It regenerates the same 1,000 school resamples using seed 20261920 and the same ordered unique-school list as scripts/run_reviewer_revision.py:162–179, pairs against stored expanded_logit replicate metrics by replicate ID, and independently computes the capacity boundary using cumulative weighted rank. A spot check on replicate IDs 0, 1, 250, 500 and 999 reproduces stored expanded-logit AP within 5.56e−17 and recall within 1.39e−15. Bootstrap mean contrasts are 0.025152338 and 0.008754192; these are not the reported full-sample point estimates.

Use the continuous and complete-case rows in the preceding tables directly for the supplemental robustness table. Add these HGB implementation rows from reviewer_revision_v1/model_metrics.csv:

| HGB setting | n | AP | Recall@10% | Precision@10% | Brier | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Main implementation | 29,588 | 0.328878276 | 0.280944628 | 0.409519827 | 0.111792116 | 0.003244039 | 1.000413481 |
| Fixed selected hyperparameters, early_stopping=False | 29,588 | 0.328260381 | 0.281718456 | 0.410647799 | 0.112077011 | −0.073738418 | 0.949719065 |

The no-early-stop minus main point contrasts are AP −0.000617895 (paired CI −0.005888105 to 0.004967135) and recall +0.000773828 (paired CI −0.007724177 to 0.007281787). The correct recall point is slightly positive even though its bootstrap mean is slightly negative. All five no-early-stop folds use 200 boosting iterations according to reviewer_revision_v1/hgb_no_earlystop_folds.csv.
