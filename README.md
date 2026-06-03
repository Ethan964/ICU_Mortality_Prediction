# ICU Mortality Prediction

## Overview

Leveraging PhysioNet's MIMIC-IV dataset, we aim to train and develop prediction models used to find correlation between 24-hour mortality rates and various predictors contained within MIMIC-IV.

## Data Access

This project requires MIMIC-IV v2.2. You must obtain credentialed access at: https://physionet.org/content/mimiciv/. Raw data is never included in this repo.

After credentialling, download the required tables:
admissions, patients, icustats, chartevents, labevents, inputevents, prescriptions, diagnoses_icd

Place compressed files in 'data/raw/hops' and 'data/raw/icu'.

## Setup

```bash
poetry install
cp .env.example .env
```

## Reproducing Results

```bash
make preprocess
make train
make evaluate
```

## Data Governance

This project uses MIMIC-IV v3.1, a de-identified electronic health record dataset from the Beth Israel Deaconess Medical Center, accessed via PhysioNet (Johnson et al., 2023; doi:10.13026/6mm1-ek67). Access requires completion of CITI human subjects training and execution of the PhysioNet Credentialed Health Data Use Agreement 1.5.0. No data files are included in this repository. All MIMIC-IV files are stored locally behind access controls and are never committed to version control. Researchers wishing to reproduce this work must independently obtain credentialed access at https://physionet.org/content/mimiciv/3.1/. Any derived datasets or model outputs that may contain patient-level signal are treated as sensitive and are not shared publicly. This project does not attempt to re-identify any individual. All analyses use the de-identified data as provided; no linkage to external records was performed.

## Citation

Johnson, Alistair, et al. "MIMIC-IV" (version 3.1). PhysioNet (2024). RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58
