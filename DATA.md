# Data acquisition and boundary

## Source

Obtain the PISA 2022 SPSS student-questionnaire public-use file from the official OECD database page:

<https://www.oecd.org/en/data/datasets/pisa-2022-database.html>

The OECD page provides public-use unit-record files in SPSS and SAS formats. This repository does not redistribute those files.

## Expected files and hashes

The analysis used the following locally downloaded archive and extracted file:

| File | Bytes | SHA-256 |
|---|---:|---|
| `STU_QQQ_SPSS.zip` | 682,364,259 | `24EE020CD6315D8ACA52E07FD9AAEF21CBF4F9E5B21601734B4EE6D0928A0905` |
| `CY08MSP_STU_QQQ.SAV` | 2,096,666,903 | `9E4EDDACD25E10C145F7BFE817FC99A5B8DF544536E5EA1DFD7230C988916A8C` |

The `data_sha256` field in the frozen pilot and main configurations refers to the downloaded ZIP archive. The full rerun also checks the extracted SAV file before analysis.

Place only the extracted file at:

```text
data/raw/pisa2022/CY08MSP_STU_QQQ.SAV
```

PowerShell verification:

```powershell
Get-FileHash -Algorithm SHA256 .\data\raw\pisa2022\CY08MSP_STU_QQQ.SAV
```

macOS/Linux verification:

```bash
sha256sum data/raw/pisa2022/CY08MSP_STU_QQQ.SAV
```

## Analytic subset

The scripts select `CNT == "ESP"` and require a valid response to `ST016Q01NA`. Low life satisfaction is defined descriptively as a response from 0 through 4. This is not a clinical cutoff.

## Redistribution

Do not commit the archive, the extracted SAV file, download fragments, or derived row-level files. Users are responsible for reviewing and following the OECD terms applicable to the public-use files.

