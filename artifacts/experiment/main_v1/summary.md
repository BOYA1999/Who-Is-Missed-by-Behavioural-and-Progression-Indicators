# Main experiment summary

## Outcome

The nested school validation and 19-region internal–external validation completed for 29,588 students from 965 schools. Weighted low-life-satisfaction prevalence was 0.146 (PISA BRR 95% CI 0.139–0.153).

- Routine penalized logistic regression, school nested: AUPRC 0.200; recall at 10% capacity 0.163.
- Expanded penalized logistic regression, school nested: AUPRC 0.303; recall 0.265.
- Expanded HGB, school nested: AUPRC 0.329; recall 0.281 (fixed-OOF school-bootstrap 95% interval 0.265–0.299).
- Expanded HGB versus expanded penalized logistic regression: paired school-bootstrap AUPRC +0.026 (95% interval +0.015 to +0.036); recall +0.016 (+0.003 to +0.029). The gain is positive, while the lower interval bounds remain below the researcher-set descriptive comparison margins of +0.02 AUPRC and +0.03 recall.
- Expanded HGB AUPRC, region IECV: 0.323; school-nested minus region-IECV +0.006.
- Simple visible-risk heuristic, complete cases: recall at 10% capacity 0.160.

## Evidence classification

The main operational finding is supported: routine visible information improves on random selection but misses most low-life-satisfaction students, while school-experience scales add substantial information. The extra gain from a nonlinear model is small and secondary; the transparent expanded model remains the primary interpretable benchmark.

## Sensitivity campaign

- The information-tier result survived the continuous 0–10 outcome: expanded HGB RMSE 2.083 versus routine HGB 2.244; expanded ridge RMSE 2.122 versus routine ridge 2.245.
- Complete-case analysis (n=27,937) closely reproduced the main result; fold-wise imputation was not the driver.
- Giving every school its own 10% follow-up capacity reduced expanded-HGB recall from 0.281 to 0.267 without reversing the conclusion.
- Adding sex, immigration and ESCS increased AUPRC only from 0.329 to 0.349 while materially increasing recall gaps. They remain audit attributes, not primary ranking inputs.
- No further model-family expansion is justified.

## Boundaries

The bootstrap interval conditions on fixed out-of-fold predictions and does not include model-refitting uncertainty. The data are cross-sectional and the task is contemporaneous identification, not prospective prediction. Results do not validate diagnosis, causality, intervention effects, or deployment.

## Next action

Proceed to the paper outline and Methods/Results drafting, followed by rendered table and figure QA.
