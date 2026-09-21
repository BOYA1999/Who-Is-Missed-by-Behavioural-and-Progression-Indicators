# Campaign validation

**Status:** Ready within reviewed scope

- All 29,588 prediction rows were unique; binary probabilities and continuous predictions were finite.
- Independent recomputation matched expanded-HGB continuous MAE/RMSE and audit-attribute binary AUPRC/AUROC/Brier/capacity metrics within 1e-10.
- Complete-case count independently recovered from the SAV file: 27,937.
- Deterministic outer school folds were reconstructed with zero school overlap.
- Every global and within-school policy allocated exactly 10% of weighted capacity; expanded-HGB recall changed from 0.281 to 0.267.
- Adding audit attributes increased recall gaps by sex +0.182, immigration +0.043, and ESCS +0.076; this supports keeping them as audit variables rather than ranking inputs.
- These slices provide robustness checks, not prospective, causal, diagnostic or deployment evidence.
