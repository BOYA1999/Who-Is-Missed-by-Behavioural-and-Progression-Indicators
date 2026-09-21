# Reproducibility guide

## Two replay levels

### 1. Aggregate verification

This level does not require the PISA unit-record file. It verifies internal identities and the public-safe tables retained in the repository:

```bash
python scripts/validate_public_aggregates.py
python scripts/package_sanity_check.py
```

This is an aggregate consistency check, not an independent reconstruction from raw data.

### 2. Full local rerun

After placing the verified OECD file at `data/raw/pisa2022/CY08MSP_STU_QQQ.SAV`, run:

```bash
python scripts/run_full_pipeline.py
```

The wrapper executes:

1. school-isolated nested validation;
2. primary-result validation;
3. continuous-outcome, complete-case, audit-attribute, and local-capacity sensitivities;
4. visibility profiles;
5. component ablations, alternative bullying composite, calibration, subgroup, regional, and inclusion audits;
6. school-cluster bootstrap with priority sets reselected in every replicate;
7. targeted consistency checks;
8. aggregate consistency checks.

## Environment

The recorded Windows run used Python 3.12.14 and the versions pinned in `requirements.txt`. The code is platform-neutral apart from shell examples.

## Locally generated restricted intermediates

The full run creates files containing unit-level scores or identifiers, including:

- `oof_predictions.csv.gz`;
- `sensitivity_predictions.csv.gz`;
- `revision_predictions.csv.gz`;
- `visibility_assignments.csv.gz`;
- `fold_assignments.csv.gz`; and
- `school_multiplicities.csv.gz`.

These files are required for some full-run validation steps but are intentionally absent from this repository and ignored by Git.

## Statistical boundary

- School-held-out predictions are used to compare ranking information sets.
- Fay-BRR intervals address designated survey-estimate variation.
- Prediction-derived Fay-BRR quantities and school-bootstrap intervals reuse stored out-of-fold scores; they do not include model-refitting or tuning uncertainty.
- The 10% capacity is an analytic constraint, not an estimate of available Spanish school resources.
- Regional and subgroup results are descriptive or exploratory and do not establish fairness, transferability, or operational validity.
