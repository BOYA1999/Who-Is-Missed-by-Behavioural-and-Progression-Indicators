# PISA 2022 Spain fixed-capacity school-support audit

This repository contains code and aggregate results for a secondary analysis of the PISA 2022 Spain student questionnaire. It compares two information sets under the same hypothetical follow-up capacity:

- behavioural and academic-progression indicators; and
- the same indicators plus student-reported belonging, bullying exposure, and mathematics-teacher support.

The analysis asks who enters, leaves, or remains outside a weighted priority set when the information used for ranking changes. It is a cross-sectional allocation audit, not a clinical screen, diagnosis, causal analysis, prospective prediction study, or validated referral protocol.

## What is included

- analysis and validation scripts with repository-relative paths;
- frozen configurations and tested package versions;
- aggregate result tables and non-identifying bootstrap summaries;
- checks that run without the unit-record data;
- instructions for a full local rerun after obtaining the OECD file independently.
- a three-seed paired school-split check of priority-list stability.

The repository excludes the PISA unit-record files, student and school identifiers, row-level predictions, fold assignments, selection assignments, and school bootstrap multiplicities.

## Quick public check — no PISA data required

```bash
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
python scripts/validate_public_aggregates.py
python scripts/package_sanity_check.py
```

The aggregate check verifies the reported sample size, weighted prevalence, model metrics, fixed-capacity identities, queue-overlap cells, and bootstrap contrast.

## Full local rerun

Download the PISA 2022 SPSS student-questionnaire public-use file from the OECD and place the extracted file at:

```text
data/raw/pisa2022/CY08MSP_STU_QQQ.SAV
```

Then run:

```bash
python scripts/run_full_pipeline.py
```

The full rerun creates unit-level intermediate files locally. They are excluded by `.gitignore` and must not be committed. Exact data hashes and acquisition notes are in [DATA.md](DATA.md); the analysis order and evidence boundaries are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Key aggregate results

For 29,588 students in 965 sampled schools, weighted low-life-satisfaction prevalence was 14.58%. At the illustrative weighted 10% capacity:

| Ranking | Weighted AP | Recall |
|---|---:|---:|
| Behavioural–progression logistic regression | 0.200 | 16.3% |
| Expanded logistic regression | 0.303 | 26.5% |
| Expanded histogram-based gradient boosting | 0.329 | 28.1% |

The two logistic-regression priority sets shared 3.63% of total student weight and required 16.37% capacity for their union. Among students reporting low life satisfaction, 66.9% were outside both sets. Across three paired school-split seeds, cross-information overlap was 3.57%–3.63% of total weight; within-information overlap across seeds was 9.07%–9.51%. These ranges are descriptive, not confidence intervals.

The nonlinear model had a modest positive gain over expanded logistic regression. Its paired AP and recall intervals excluded zero, but their lower bounds did not exceed the researcher-set +0.02 AP and +0.03 recall comparison margins. These margins are descriptive, not preregistered or validated against service costs.

These quantities describe this sample and frozen analysis. They do not show that the added experiences cause low life satisfaction or that a ranking improves access to support.

## Repository map

```text
artifacts/                 Public-safe aggregate results and frozen configurations
data/raw/pisa2022/         Empty data location plus instructions; no data are included
scripts/                   Analysis, validation, and release checks
DATA.md                    OECD acquisition and hash information
REPRODUCIBILITY.md         Full and aggregate-only replay instructions
RESULTS.md                 Claim boundaries and retained negative findings
```

## Data source

OECD, PISA 2022 Database: <https://www.oecd.org/en/data/datasets/pisa-2022-database.html>
