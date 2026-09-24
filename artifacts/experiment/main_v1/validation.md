# Validation

**Status:** Share with caveats

- Prediction rows: 29,588; duplicate student IDs: 0.
- Leakage findings: 0.
- Required metric and probability checks: passed.
- Independent expanded-HGB recomputation matched the reported AUPRC, AUROC, Brier, 10% capacity recall and precision within 1e-10.
- Expanded HGB minus expanded logistic, paired school bootstrap: AUPRC +0.026 (95% interval +0.015 to +0.036); recall +0.016 (+0.003 to +0.029).
- The nonlinear gain is positive. Its paired intervals exclude zero, but their lower bounds do not exceed the researcher-set descriptive comparison margins of +0.02 AUPRC or +0.03 recall; those margins are not cost-validated.
- Region IECV reveals heterogeneity despite similar pooled performance: AUROC 0.635–0.805, calibration slope 0.598–1.362, and 10% capacity recall 0.213–0.394. Small Ceuta and Melilla folds contribute to the extremes.
- Intervals from the school bootstrap condition on fixed OOF predictions and omit model-refitting uncertainty.
- Continuous-outcome and missing-data sensitivities remain outside this validation scope.
