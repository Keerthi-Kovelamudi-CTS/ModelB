# ModelB — Clinical Cancer Prediction Pipeline

A machine-learning pipeline for predicting whether a patient will be diagnosed with **cancer in the next 12 months**, using historical electronic health records (EHR) from UK GP practices. The repository currently covers two cancer types:

- 🔵 **Bladder Cancer**
- 🟢 **Prostate Cancer**

> **Disclaimer:** This system is intended to support clinical decision-making only. It does **not** replace the professional judgment of a qualified clinician.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Getting Started](#getting-started)
5. [Data Requirements](#data-requirements)
6. [Output Files](#output-files)
7. [Models and Performance](#models-and-performance)
8. [Feature Engineering](#feature-engineering)
9. [Configuration](#configuration)
10. [License / Disclaimer](#license--disclaimer)

---

## Overview

### Prediction Task

| Property | Detail |
|---|---|
| **Goal** | Predict cancer diagnosis within the next 12 months |
| **INDEX_DATE (cancer patients)** | 12 months before confirmed diagnosis |
| **INDEX_DATE (controls)** | Pseudo-date — age 70 (bladder) or random 2013–2024 (prostate) |
| **Lookback window** | 13 to 1 month before INDEX_DATE (bladder) / five 12-month windows W1–W5 (prostate) |
| **Target performance** | ≥ 80 % sensitivity **and** ≥ 70 % specificity |
| **Data source** | UK GP EHR via Snowflake (EMIS system) |
| **Coding systems** | SNOMED CT (observations), DMD (medications) |

---

## Repository Structure

```
ModelB/
│
├── README.md
│
│── Root-level scripts (Bladder Cancer — legacy/iterative)
├── 1_Sanity check.py
├── 1_run_modeling.py
├── 2_Patientoverlapcheck.py
├── 3_droppatients.py
├── 4_feature_engineering.py                  ← Bladder FE v4 FINAL (18 levels)
├── 4_feature_engineering_v2_comprehensive.py
├── 5_feature_cleanup.py
├── FE_NEW.py
├── modeling_new.py
├── run_modeling.py
├── try_ensembles_for_target.py
│
│── Root-level CSV data files
├── Bladder_negative_meds.csv
├── Bladder_negative_obs.csv
├── Bladder_positive_meds.csv
├── Bladder_positive_obs.csv
├── FE_bladder_med_windowed.csv
│
├── BladderCancer/                             ← Organised bladder cancer pipeline
│   ├── Phase 1 - Feature Identification & Prioritization (SNOMED-based)/
│   │   ├── 1.1 Data Stratification/
│   │   │   ├── Top_SNOMEDs_OBS_Positive_Cohort_12mo.sql
│   │   │   ├── Top_SNOMEDs_OBS_Negative_Cohort_12mo.sql
│   │   │   ├── Top_SNOMEDs_MED_Positive_Cohort_12mo.sql
│   │   │   ├── Top_SNOMEDs_MED_Negative_Cohort_12mo.sql
│   │   │   ├── OBS_Patient_Level_Binary_Matrix.sql
│   │   │   └── MED_Patient_Level_Binary_Matrix.sql
│   │   ├── 1.2 SNOMED Extraction & Counting/
│   │   │   ├── Top_SNOMEDs_Positive_cohort_12mo.sql
│   │   │   └── Top_SNOMEDs_OBS_Negative_Cohort_12mo.sql
│   │   └── 1.3 Feature Importance Scoring/
│   │       └── Combined Scoring (ML+Stat).py
│   │
│   ├── Phase 2 - Temporal Feature Engineering/
│   │   ├── 2.1 Cohorts by Time Windows/
│   │   │   ├── OBS_Windowed.sql
│   │   │   └── MED_Windowed.sql
│   │   ├── 2.2 Feature Transformation Logic/
│   │   │   ├── 2.2.1 Sanity Check.py
│   │   │   ├── 2.2.2 Patient_Overlap_Check.py
│   │   │   ├── 2.2.3 Drop_Not_Overlapped_Patients.py
│   │   │   ├── 2.2.4 features_transformation.py
│   │   │   └── 2.2.5 Feature_Cleanup.py
│   │   └── 2.3 Quality Checks/
│   │       ├── 2.3.1 Diagnositics_Check.py
│   │       └── Quality checks & Validation.py
│   │
│   └── Phase 3 - Rule Identification and Model Training/
│       ├── 3.1 Baseline modeling/
│       │   ├── baseline modeling.py
│       │   ├── Ensemble.py
│       │   ├── Final_Analysis.py
│       │   └── Resultswriteup.py
│       └── 3.2 Rule Extraction & Generation/
│           └── rules extraction.py
│
└── Prostate/                                  ← Organised prostate cancer pipeline
    ├── Phase 1 - Feature Identification & Prioritization (SNOMED-based)/
    │   ├── 1.1 Data Stratification/
    │   │   ├── Positive Cohort for Prostate Cancer.sql
    │   │   └── Negative Cohort for Prostate Cancer.sql
    │   ├── 1.2 SNOMED Extraction & Counting/
    │   │   ├── Top_SNOMEDs_Positive_cohort_12mo.sql
    │   │   ├── Top_SNOMEDs_Positive_cohort_36mo.sql
    │   │   ├── Top_SNOMEDs_Negative_cohort_12mo.sql
    │   │   ├── Top_SNOMEDs_Negative_cohort_36mo.sql
    │   │   └── Age distribution count.sql
    │   ├── 1.3 Feature Importance Scoring/
    │   │   ├── Combined Scoring (ML+Stat).py
    │   │   ├── Feature importance via ML (L1 logistic, RF, XGBoost).py
    │   │   └── Feature importance via Statistical Significance.py
    │   └── 1.4 Clinical Review & Validation/
    │       └── Top Predictive SNOMEDs for the Prostate cancer
    │
    ├── Phase 2 - Temporal Feature Engineering/
    │   ├── 2.1 Define Time Windows/
    │   │   ├── Prostate Cancer - Positive Cohort Window.sql
    │   │   ├── Prostate Cancer - Negative Cohort Window.sql
    │   │   └── COUNT CANCER PATIENTS.sql
    │   ├── 2.2 Feature Transformation Logic/
    │   │   ├── prostate_features_transformation.py
    │   │   ├── symptom_occurance_analysis.py
    │   │   └── value_trend_analysis.py
    │   ├── 2.3 Quality Checks/
    │   │   └── Quality checks & Validation.py
    │   ├── 2.4 Create Final Transformed Dataset/
    │   │   └── Final Transformed Dataset/
    │   └── 2.5 Feature Store Integration/
    │       └── Vertex AI Feature Store/
    │
    └── Phase 3 - Rule Identification and Model Training/
        ├── 3.1 Baseline ML Model/
        │   └── prostate cancer baseline modeling.py
        ├── 3.2 Rule Extraction & Generation/
        │   └── rules extraction.py
        └── 3.4 Complex Modeling/
            └── complex modeling.py
```

---

## Pipeline Architecture

Both cancer pipelines follow the same **three-phase architecture**:

```
Phase 1                    Phase 2                       Phase 3
──────────────────         ──────────────────────────    ────────────────────────────
Feature Identification  →  Temporal Feature Engineering  →  Model Training & Rules
(SNOMED-based)
  │                          │                               │
  ├─ 1.1 Stratification      ├─ 2.1 Time Windows (SQL)      ├─ 3.1 Baseline ML Models
  ├─ 1.2 SNOMED Extraction   ├─ 2.2 Feature Transforms      └─ 3.2 Rule Extraction
  └─ 1.3 Importance Scoring  └─ 2.3 Quality Checks
```

### Phase 1 — Feature Identification & Prioritization

**Goal**: Identify the most clinically predictive SNOMED codes from raw EHR data.

| Step | What it does |
|---|---|
| **1.1 Data Stratification** | SQL queries extract the top-ranked observation and medication SNOMED codes for cancer (positive) and control (negative) cohorts in a 12-month lookback window. Excludes ethnicity codes, immunisations, cancer diagnosis codes, and palliative care patients. |
| **1.2 SNOMED Extraction & Counting** | SQL creates a patient × SNOMED feature matrix in long format — each row is `(patient_guid, label, sex, age_at_index, feature_name, feature_code, event_count)`. Combines both cohorts. |
| **1.3 Feature Importance Scoring** | Python script ranks features using: **Statistical** (prevalence difference, odds ratio with 95% CI, chi-squared with Bonferroni correction) + **ML** (L1 Logistic Regression, Random Forest 200 trees, Gradient Boosting 150 trees, 3-fold stratified CV, 5:1 neg:pos downsampling). Combined score: 30 % statistical + 20 % chi-squared + 50 % ML. Outputs top 150 features for observations, medications, and combined. |

### Phase 2 — Temporal Feature Engineering

**Goal**: Transform raw EHR events into rich time-aware features for each patient.

| Step | What it does |
|---|---|
| **2.1 Time Windows (SQL)** | Defines the INDEX_DATE and extracts observations/medications in structured time windows relative to it. |
| **2.2 Feature Transformation** | Python scripts build hundreds of features per patient (see [Feature Engineering](#feature-engineering)). Includes data sanity checks, patient-overlap checks, and feature cleanup. |
| **2.3 Quality Checks** | Validates INDEX_DATE correctness, checks for data leakage (events after INDEX_DATE), confirms control patients have sufficient follow-up. |

### Phase 3 — Model Training & Rule Extraction

**Goal**: Train ML classifiers and extract interpretable clinical prediction rules.

| Step | What it does |
|---|---|
| **3.1 Baseline ML Models** | Trains XGBoost, LightGBM, CatBoost, Random Forest, and Logistic Regression using a 60/15/25 train/val/test split with 5-fold stratified cross-validation. |
| **3.2 Rule Extraction** | Uses SHAP feature importance to generate risk-stratified prediction rules with clinical recommendations. |

---

## Getting Started

### Prerequisites

```
Python >= 3.9
Snowflake account (with access to CTSUK_BULK and analytics.cts schemas)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Keerthi-Kovelamudi-CTS/ModelB.git
cd ModelB

# Install Python dependencies
pip install pandas numpy scipy scikit-learn xgboost lightgbm catboost shap matplotlib seaborn joblib
```

### Running the Bladder Cancer Pipeline (Organised Version)

```bash
# Phase 1 — run SQL queries in Snowflake, save outputs as CSV
# Then run feature importance scoring:
python "BladderCancer/Phase 1 - Feature Identification & Prioritization (SNOMED-based)/1.3 Feature Importance Scoring/Combined Scoring (ML+Stat).py"

# Phase 2 — feature engineering (in order):
python "BladderCancer/Phase 2 - Temporal Feature Engineering/2.2 Feature Transformation Logic/2.2.1 Sanity Check.py"
python "BladderCancer/Phase 2 - Temporal Feature Engineering/2.2 Feature Transformation Logic/2.2.2 Patient_Overlap_Check.py"
python "BladderCancer/Phase 2 - Temporal Feature Engineering/2.2 Feature Transformation Logic/2.2.3 Drop_Not_Overlapped_Patients.py"
python "BladderCancer/Phase 2 - Temporal Feature Engineering/2.2 Feature Transformation Logic/2.2.4 features_transformation.py"
python "BladderCancer/Phase 2 - Temporal Feature Engineering/2.2 Feature Transformation Logic/2.2.5 Feature_Cleanup.py"

# Phase 2 — quality checks:
python "BladderCancer/Phase 2 - Temporal Feature Engineering/2.3 Quality Checks/Quality checks & Validation.py"

# Phase 3 — model training and rule extraction:
python "BladderCancer/Phase 3 - Rule Identification and Model Training/3.1 Baseline modeling/baseline modeling.py"
python "BladderCancer/Phase 3 - Rule Identification and Model Training/3.2 Rule Extraction & Generation/rules extraction.py"
```

### Running the Root-Level Bladder Cancer Scripts (Legacy)

```bash
# Feature engineering with optional time-window argument (default: 6mo):
python 4_feature_engineering.py --window 3mo   # or --window 6mo

# Model training:
python 1_run_modeling.py

# Ensemble tuning to hit sensitivity/specificity targets:
python try_ensembles_for_target.py
```

### Running the Prostate Cancer Pipeline

```bash
# Phase 1 — run SQL queries in Snowflake, save outputs, then:
python "Prostate/Phase 1 - Feature Identification & Prioritization (SNOMED-based)/1.3 Feature Importance Scoring/Combined Scoring (ML+Stat).py"

# Phase 2 — feature transformation:
python "Prostate/Phase 2 - Temporal Feature Engineering /2.2 Feature Transformation Logic/prostate_features_transformation.py"
python "Prostate/Phase 2 - Temporal Feature Engineering /2.2 Feature Transformation Logic/symptom_occurance_analysis.py"
python "Prostate/Phase 2 - Temporal Feature Engineering /2.2 Feature Transformation Logic/value_trend_analysis.py"

# Phase 2 — quality checks:
python "Prostate/Phase 2 - Temporal Feature Engineering /2.3 Quality Checks/Quality checks & Validation.py"

# Phase 3 — model training and rules:
python "Prostate/Phase 3 - Rule Identification and Model Training/3.1 Baseline ML Model/prostate cancer baseline modeling.py"
python "Prostate/Phase 3 - Rule Identification and Model Training/3.2 Rule Extraction & Generation/rules extraction.py"
```

---

## Data Requirements

All input data is sourced from Snowflake. The following tables are used:

| Table | Contents |
|---|---|
| `CTSUK_BULK.RAW.CARERECORD_OBSERVATION` | Clinical observations (SNOMED-coded) |
| `CTSUK_BULK.STAGING.PATIENTS_EMIS` | Patient demographics |
| `CTSUK_BULK.RAW.PRESCRIBING_DRUGRECORD` | Medication records |
| `CTSUK_BULK.RAW.PRESCRIBING_ISSUERECORD` | Prescription issue records |
| `CTSUK_BULK.RAW.CODING_CLINICALCODE` | SNOMED clinical code lookup |
| `CTSUK_BULK.RAW.CODING_DRUGCODE` | DMD drug code lookup |
| `analytics.cts.dim_cancer_codes` | Cancer diagnosis code definitions |

> ⚠️ **Never commit database credentials or connection strings to this repository.**  
> Use environment variables or a secrets manager for all Snowflake credentials.

### Root-Level CSV Files

These files are input to the root-level (legacy) bladder cancer scripts:

| File | Description |
|---|---|
| `Bladder_positive_obs.csv` | Observations for bladder cancer patients |
| `Bladder_negative_obs.csv` | Observations for control patients |
| `Bladder_positive_meds.csv` | Medications for bladder cancer patients |
| `Bladder_negative_meds.csv` | Medications for control patients |
| `FE_bladder_med_windowed.csv` | Windowed medication feature-engineering output |

---

## Output Files

| Phase | Output |
|---|---|
| Phase 1 — Stratification | CSV files: top SNOMED codes ranked by frequency for positive and negative cohorts |
| Phase 1 — SNOMED Matrix | CSV: patient × feature matrix in long format |
| Phase 1 — Importance Scoring | CSV: top 150 features with combined importance scores |
| Phase 2 — Feature Engineering | CSV: wide patient-level feature matrix (hundreds of columns) |
| Phase 3 — Model Training | Trained model files (`.pkl` / `.json`), calibration curves, performance metrics |
| Phase 3 — Rule Extraction | Clinical summary report: risk tiers, SHAP-driven rules, and referral recommendations |

---

## Models and Performance

### Target Metrics

| Metric | Target |
|---|---|
| Sensitivity (Recall) | ≥ 80 % |
| Specificity | ≥ 70 % |

### Models Trained (Phase 3)

| Model | Notes |
|---|---|
| **XGBoost** | Primary model; isotonic calibration applied |
| **LightGBM** | Fast gradient boosting |
| **CatBoost** | High-recall variant; favoured in ensembles |
| **Random Forest** | Stable baseline |
| **Logistic Regression** | Interpretable baseline |

### Ensemble Strategy

Weighted combinations of XGBoost, LightGBM, and CatBoost are tested to optimise for the sensitivity/specificity targets. CatBoost is given higher weight in ensembles due to its superior recall performance.

### Risk Stratification (Rule Extraction)

| Risk Tier | Probability Threshold | Recommended Action |
|---|---|---|
| Very High | > 90 % | Urgent referral |
| High | 70 – 90 % | Expedited investigation |
| Moderate-High | 50 – 70 % | Investigate |
| Moderate | 30 – 50 % | Monitor closely |
| Low | < 30 % | Routine care |

---

## Feature Engineering

The bladder cancer pipeline (`4_feature_engineering.py`) implements **18 levels** of feature engineering:

| Level | Category | Examples |
|---|---|---|
| 1 | **Demographics** | Age, sex, age bands |
| 2 | **Observation features** | Per-category counts, per-term counts, per-window counts |
| 3 | **Haematuria deep features** | Frank, painless, microscopic haematuria; acceleration flags |
| 4 | **Urine features** | Urine investigations, lab abnormalities, cytology |
| 5 | **LUTS features** | Bladder pain, hesitancy, frequency, retention |
| 6 | **Catheter / Imaging / Urological / Gynaecological** | Procedure presence and frequency |
| 7 | **Risk factors** | Smoking (heavy/light/passive/stopped), alcohol (AUDIT scores, trends) |
| 8 | **Comorbidities** | Diabetes, CKD stages, anaemia, hypertension, BPH, previous cancer, burden score |
| 9 | **Lab values** | 23 markers (eGFR, creatinine, Hb, platelets, WBC, CRP, ESR, PSA, ALT, ALP, albumin, glucose, HbA1c, MCV, ferritin, iron, RBC, haematocrit, weight, BMI, calcium, adjusted calcium) — with statistics, slopes, clinical thresholds, volatility |
| 10 | **Medications** | Per-category, per-term; UTI antibiotic depth; opioid escalation; drug combinations |
| 11 | **Temporal patterns** | Acceleration, GP visit frequency, event gaps, burstiness |
| 12 | **Event clustering** | Burst weeks, diverse-activity weeks, haematuria co-occurrence |
| 13 | **New observation categories** | Weight loss, fatigue, appetite loss, dysuria, DVT, PE, frailty severity |
| 14 | **Temporal sequences** | Event-pair timing, acceleration ratios, recurrence patterns |
| 15 | **Syndrome / composite scores** | Bleeding score, constitutional symptoms, pathway score, pain score, VTE score, UTI treatment failure |
| 16 | **Clinical pathways** | Investigation depth, multi-investigation patterns |
| 17 | **Interaction features** | Haematuria × smoking, age × symptoms, lab × clinical combinations |
| 18 | **Ratio & recency features** | First occurrence / span, cross-lab interactions, symptom diversity, lab-to-event timing, consultation patterns |

The prostate cancer pipeline uses a similar approach adapted for PSA, DRE findings, LUTS, haematuria, and erectile dysfunction over five 12-month time windows (W1–W5).

---

## Configuration

### Bladder Cancer Feature Engineering

```bash
# Choose a time-window size (default: 6mo):
python 4_feature_engineering.py --window 3mo
python 4_feature_engineering.py --window 6mo
```

### Prostate Cancer Cohort Windows

The SQL files in `Prostate/Phase 2 - Temporal Feature Engineering /2.1 Define Time Windows/` define:
- **INDEX_DATE**: 12 months before diagnosis for cancer patients; random date 2013–2024 for controls
- **107 validated SNOMED codes** covering PSA (7 codes), DRE (6 codes), LUTS, haematuria, erectile dysfunction, and other prostate-relevant findings

### Key Modelling Hyperparameters

Adjust directly in the relevant script:

| Parameter | Default | Location |
|---|---|---|
| Train / Val / Test split | 60 / 15 / 25 | `1_run_modeling.py`, `baseline modeling.py` |
| Cross-validation folds | 5 | `baseline modeling.py` |
| Neg:Pos downsample ratio | 5:1 | `Combined Scoring (ML+Stat).py` |
| Sensitivity target | 0.80 | `try_ensembles_for_target.py` |
| Specificity target | 0.70 | `try_ensembles_for_target.py` |

---

## License / Disclaimer

This project is developed for **clinical research and decision-support purposes** within a healthcare organisation.

- This tool **does not** replace clinical judgment.
- Predictions are **probabilistic** and must be interpreted by a qualified clinician.
- Patient data must be handled in accordance with applicable data protection legislation (e.g., UK GDPR, NHS data governance frameworks).
- No real patient data, database credentials, or identifiable information should be committed to this repository.
