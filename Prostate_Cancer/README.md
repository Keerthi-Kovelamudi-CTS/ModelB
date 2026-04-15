# Prostate Cancer Prediction Pipeline

Config-driven, modular pipeline for predicting prostate cancer from primary care EHR data.

All cancer-specific settings live in **`config.py`**. Generic pipeline code is shared and reusable across cancer types. To port to a new cancer, only **2 files** need to change: `config.py` and `4_cancer_features.py`.

---

## Directory Overview

```
Prostate_Cancer/
    1_Top_Snomed/              Phase 1 - SNOMED feature ranking
    2_Feature_Engineering/     Phase 2 - Feature engineering pipeline
    3_Modeling/                Phase 3 - Model training & evaluation
    4_ExpandedData_Test/       Phase 4 - Holdout / expanded data validation
    5_Inference/               Phase 5 - Production single-patient prediction
    6_Explainability/          Phase 6 - Explainability & enhancement loop
```

---

## Quick Start

```bash
# 1. Full feature engineering pipeline (end to end)
cd 2_Feature_Engineering/
python 0_run_pipeline.py

# 2. Train models
cd ../3_Modeling/
python 1_run_modeling.py

# 3. Holdout validation
cd ../4_ExpandedData_Test/
python 1_preprocess_holdout.py
python 2_run_holdout_fe.py
cd ../3_Modeling/
python 2_predict_unseen.py --window 12mo
```

---

## Phase 2: Feature Engineering (`2_Feature_Engineering/`)

### Pipeline Flow

```
Raw CSVs (from SQL)
    |
    v
[1_sanity_check.py]  -->  Patient overlap removal, dedup, sex/age filter
    |
    v
[2_clean_data.py]    -->  Lab range clamping, date parsing
    |
    v
[3_pipeline.py]      -->  500+ generic features (obs, med, lab, interactions)
    +
[4_cancer_features.py] -> Prostate-specific features (PSA, urinary, bone)
    |
    v
[5_cleanup.py]       -->  Remove noise, confounds, leakage
    |
    v
[6_text_features.py] -->  NLP: keywords + TF-IDF + BERT embeddings
    |
    v
Clean feature matrices ready for modeling
```

### Scripts

| Script | Step | What It Does |
|--------|------|-------------|
| `config.py` | -- | Single source of truth for all cancer-specific settings: categories, thresholds, lab ranges, text patterns, modeling params, paths |
| `0_run_pipeline.py` | -- | **Orchestrator.** Runs all steps in sequence. Supports `--step` flag to run individual steps |
| `1_sanity_check.py` | Step 1 | **Data validation & patient fixes.** Checks row counts, date ranges, category distributions, nulls. Removes patient overlaps (same patient in both pos/neg cohort), med-only patients, duplicates. Filters sex (male only) and age (18+). Saves master patient list |
| `2_clean_data.py` | Step 2 | **Lab range clamping.** Parses dates, converts VALUE to numeric, clamps lab outliers to NaN using physiological ranges from `config.LAB_RANGES`, encodes time windows |
| `3_pipeline.py` | Step 3 | **Generic feature engineering** (7 functions, all parameterized by config). See detailed breakdown below |
| `4_cancer_features.py` | Step 4 | **Prostate-specific features.** PSA dynamics, urinary patterns, bone/metastatic signals, treatment flags, risk score. This is the ONLY file that changes per cancer type |
| `5_cleanup.py` | Step 5 | **Feature cleanup.** Removes non-numeric, duplicates, high-null (>95%), zero/near-zero variance, utilization confounds (eGFR, lab counts), high correlation (>0.98), leakage (>0.5 label correlation). Smart NaN fill by feature type |
| `6_text_features.py` | Step 6 | **NLP pipeline.** Extracts text from ASSOCIATED_TEXT field. Builds keyword features, TF-IDF embeddings (15 dims), and BERT embeddings (15 dims) |

### Usage

```bash
python 0_run_pipeline.py                 # run everything
python 0_run_pipeline.py --step sanity   # step 1 only
python 0_run_pipeline.py --step clean    # step 2 only
python 0_run_pipeline.py --step fe       # steps 3-4 only
python 0_run_pipeline.py --step cleanup  # step 5 only
python 0_run_pipeline.py --step text     # step 6 only
```

### Step 3 Detail: `3_pipeline.py` Functions

#### `build_clinical_features()`
- **Demographics**: age, age band
- **Per-category observation counts**: count in window A, count in window B, total count
- **Flags**: has_ever, acceleration (more events in B than A), new_in_B (appeared in B but not A)
- **Lab value stats**: mean, min, max, std, first, last, delta (B minus A), linear slope over time
- **Investigation patterns**: imaging count A/B, acceleration
- **Aggregate features**: total events, unique categories, unique codes, symptom-specific counts
- **Temporal features**: event span (days), days from last event to index date

#### `build_medication_features()`
- **Per-category counts**: A, B, total, has_ever, acceleration, new_in_B
- **Quantity stats**: total quantity, mean quantity per category
- **Unique drugs** per category
- **Aggregates**: total prescriptions, unique categories/drugs, polypharmacy score

#### `build_interaction_features()`
- Creates **obs x med interaction pairs** from `config.INTERACTION_PAIRS`
  (e.g., urinary_symptoms x PSA_monitoring, bone_pain x imaging)
- Multi-symptom burden (3+ unique symptom categories)
- High investigation with symptoms

#### `build_advanced_features()`
- **Symptom clusters**: groups from `config.CLUSTER_DEFINITIONS` (urinary, PSA, bone, constitutional, imaging, sexual). Counts, breadth, cross-cluster combinations
- **Visit patterns**: unique visit dates, acceleration, events-per-visit
- **Temporal trajectories**: quarterly event counts (Q1 through Q6), increasing trend flag
- **Lab trajectories**: per-category slope, range, CV, first-last diff, declining flag
- **Medication escalation**: pain med acceleration, steroid escalation, repeat prescriptions, polypharmacy increase
- **Investigation patterns**: symptom-to-investigation gap (days)
- **Cross-domain**: diagnostic odyssey (high visits + multi-system)
- **Time-decay**: exponential-weighted symptom scores (recent events weighted higher)

#### `extract_maximum_features()`
- **Monthly bins**: event counts per 3-month block (0-3m, 3-6m, ..., 9-12m)
- **Rolling 3-month windows**: overlapping count windows
- **Per-category granular**: total events and unique codes per category
- **Co-occurrence pairs**: top 10 category pairs that co-occur in same patient
- **Rates**: events per category
- **Age interactions**: age band x cluster flags
- **Entropy**: Shannon entropy of category distribution (diversity measure)
- **Gini**: concentration measure of category distribution

#### `build_new_signal_features()`
- **Per-lab-term features**: count, mean, last value, slope for top 30 lab terms
- **Recency**: days since last event per category
- **Distinct visit dates** per category

#### `build_trend_features()`
- **Per-symptom frequency**: event count, time span, frequency per year
- **Mean interval** between events (days)
- **Worsening flag**: more events in the recent half vs earlier half
- **Per-medication recurrence**: prescription count and unique drugs per category

### Step 4 Detail: `4_cancer_features.py` (Prostate-Specific)

| Block | Features | Clinical Rationale |
|-------|----------|-------------------|
| **1. PSA Dynamics** | PSA count A/B, acceleration, latest/first value, elevated flags (>4, >10, >20, >100 ng/mL), delta, rising flag, rapid rise (>2) | PSA is the primary screening marker for prostate cancer. Rising PSA is the most common trigger for investigation |
| **2. Urinary Patterns** | Urinary symptom counts, acceleration, haematuria flag, erectile dysfunction flag | LUTS are the most common presenting symptoms. Haematuria raises suspicion |
| **3. Bone/Metastatic** | Bone pain, lower back pain, any bone symptom, bone acceleration, ALP elevation (>130 U/L) | Bone metastases are common in advanced prostate cancer. ALP elevation signals bone involvement |
| **4. Lab Prognostic** | CRP (>10 mg/L), haemoglobin (<120 g/L = anaemia), testosterone | Anaemia and low testosterone suggest advanced disease or hormonal changes |
| **5. Treatment Flags** | Alpha blockers, 5-ARI, antibiotics, pain meds, corticosteroids, BPH combo flag | BPH treatment (alpha blocker + 5-ARI) suggests managed benign disease. Escalating pain meds may signal progression |
| **6. Diagnostic Pathway** | Abnormal imaging flag, symptom-to-imaging gap (days) | Time from first urinary symptom to imaging reflects investigation urgency |
| **7. Risk Score** | Composite: age>=65 + PSA>10 + urinary symptoms + bone symptoms + abnormal imaging | Loosely based on D'Amico risk stratification |
| **8. Risk Factors** | Peak age (50-80), elderly (70+), age x PSA, age x urinary, PSA + bone combination | Epidemiological risk factors and high-risk combinations |

### Data Directory

```
data/
    3mo/                                 3-month lookback window
        prostate_3mo_obs.csv             Raw observation data (from SQL)
        prostate_3mo_obs_dropped.csv     After sanity check + cleaning
        prostate_3mo_med.csv             Raw medication data (from SQL)
        prostate_3mo_med_dropped.csv     After sanity check + cleaning
    6mo/                                 6-month lookback (same structure)
    12mo/                                12-month lookback (same structure)
```

### Results Directory

```
results/
    1_sanity_check/                      master_patients_{w}.csv
    2_clean_data/                        (logs only)
    3_feature_engineering/
        {3mo,6mo,12mo}/                  feature_matrix_final_{w}.csv
    4_cancer_features/
        {3mo,6mo,12mo}/                  (merged into step 3 output)
    5_cleanup/
        {3mo,6mo,12mo}/                  feature_matrix_clean_{w}.csv
    6_text_features/
        keywords/{3mo,6mo,12mo}/         text_features_{w}.csv
        tfidf_embeddings/{3mo,6mo,12mo}/ text_embeddings_{w}.csv
        bert_embeddings/{3mo,6mo,12mo}/  bert_embeddings_{w}.csv
```

---

## Phase 3: Modeling (`3_Modeling/`)

### Pipeline Flow

```
Clean features (from Phase 2)
    |
    v
[1_run_modeling.py]
    |
    +-- Filter to clinical features (remove utilization noise)
    +-- Merge text + TF-IDF + BERT embeddings
    |
    +-- For each window (3mo, 6mo, 12mo):
    |     For each seed (13, 42, 45, 77):
    |       |
    |       +-- Split: train 75% / val 10% / test 15%
    |       +-- Feature selection (XGBoost importance, force-include text)
    |       +-- Tune XGBoost (Optuna, 75 trials)
    |       +-- Tune LightGBM (Optuna, 75 trials)
    |       +-- Tune CatBoost (Optuna, 75 trials)
    |       +-- Ensemble: top 2 by val AUC, grid-search weights
    |       +-- Threshold: pick best tier on validation set
    |       +-- Report: metrics on held-out test set
    |       +-- Save: models, predictions, feature importances
    |
    v
Saved models + evaluation results

    |
    v
[2_predict_unseen.py]  -->  Load saved models, predict on new data
```

### Scripts

| Script | What It Does |
|--------|-------------|
| `1_run_modeling.py` | **Full training pipeline.** Loads clean matrices + text features, filters to clinical features, trains 3 boosting models with Optuna tuning, builds weighted ensemble, selects threshold using tiered criteria, evaluates on held-out test set. Saves models, predictions, and feature importances |
| `2_predict_unseen.py` | **Holdout prediction.** Loads saved models + threshold + feature list from training. Applies to new feature matrix. If labels present, calculates AUC, sensitivity, specificity |

### Threshold Tiers (from best to fallback)

| Tier | Sensitivity | Specificity | When Used |
|------|------------|-------------|-----------|
| Tier 0 | >= 80% | >= 70% | Preferred target |
| Tier 1 | >= 75% | >= 65% | Good |
| Tier 2 | >= 70% | >= 65% | Acceptable |
| Tier 2b | >= 70% | >= 60% | Minimum viable |
| Tier 3 | Balanced | Balanced | Fallback (max of min(sens, spec)) |

### Usage

```bash
# Train
python 1_run_modeling.py

# Predict on holdout
python 2_predict_unseen.py --window 12mo
python 2_predict_unseen.py --window 12mo --data /path/to/custom_features.csv
```

### Results Directory

```
results/
    1_training/
        {3mo,6mo,12mo}/
            saved_models/
                xgboost_model.pkl
                lightgbm_model.pkl
                catboost_model.pkl
                config.json          (features, threshold, ensemble weights)
            predictions_{w}.csv      (test set predictions)
            selected_features_{w}.csv
            feature_importances_{w}.csv
        final_results.csv            (summary: all windows x all seeds)
        modeling.log
    2_predictions/
        {3mo,6mo,12mo}/
            predictions_unseen_{w}.csv
            predictions_unseen_{w}_metrics.txt
```

---

## Phase 4: Expanded Data Test (`4_ExpandedData_Test/`)

Validates trained models on a larger holdout population (e.g., 300K non-cancer patients) to assess real-world false positive rate and generalization.

### Pipeline Flow

```
Raw 300K CSV (from Snowflake/BigQuery)
    |
    v
[1_preprocess_holdout.py]
    |
    +-- Exclude training patients (prevent leakage)
    +-- Map CODE_ID -> CATEGORY (using training mapping)
    +-- Window into A/B periods (same ranges as training)
    +-- Split into obs/med CSVs per window
    |
    v
[2_run_holdout_fe.py]
    |
    +-- Imports SAME 3_pipeline.py functions (no copy-paste!)
    +-- Runs identical FE steps as training
    |
    v
Holdout feature matrices

    |
    v
[3_Modeling/2_predict_unseen.py]  -->  Apply saved models, evaluate
```

### Scripts

| Script | What It Does |
|--------|-------------|
| `1_preprocess_holdout.py` | **Raw data to FE format.** Loads raw holdout CSV, excludes training patients, maps CODE_ID to CATEGORY using `code_category_mapping.json` (same as training), windows events into A/B periods, saves as `prostate_{w}_{obs,med}_dropped.csv` |
| `2_run_holdout_fe.py` | **Holdout feature engineering.** Imports the **exact same functions** from `2_Feature_Engineering/3_pipeline.py` and `4_cancer_features.py`. Zero risk of training/holdout feature drift |

### Usage

```bash
# 1. Place raw 300K CSV in data/raw/
# 2. Preprocess
python 1_preprocess_holdout.py

# 3. Run FE (same code as training)
python 2_run_holdout_fe.py

# 4. Predict (from 3_Modeling/)
cd ../3_Modeling/
python 2_predict_unseen.py --window 3mo
python 2_predict_unseen.py --window 6mo
python 2_predict_unseen.py --window 12mo
```

### Directory Structure

```
data/
    raw/                                 Drop raw 300K CSV here
    fe_input/
        {3mo,6mo,12mo}/                  prostate_{w}_{obs,med}_dropped.csv
results/
    {3mo,6mo,12mo}/                      holdout_features_{w}.csv
```

### Required Files

| File | Source | Purpose |
|------|--------|---------|
| `code_category_mapping.json` | Generated from training SQL | Maps CODE_ID to CATEGORY (ensures holdout uses same categories as training) |
| Raw holdout CSV | Snowflake/BigQuery export | Non-cancer patients for validation |

---

## Porting to a New Cancer Type

To create a pipeline for a new cancer (e.g., Lung, Colorectal):

| Step | File to Change | What to Update |
|------|---------------|---------------|
| 1 | `config.py` | CANCER_NAME, DATA_PREFIX, PREFIX, OBS_CATEGORIES, MED_CATEGORIES, LAB_CATEGORIES, CLUSTER_DEFINITIONS, SYMPTOM_CATEGORIES, INTERACTION_PAIRS, LAB_RANGES, TEXT keywords/patterns |
| 2 | `4_cancer_features.py` | Rewrite `build_cancer_specific_features()` with cancer-specific clinical logic |
| 3 | `code_category_mapping.json` | Regenerate from training SQL for the new cancer |

**Everything else stays the same** — `3_pipeline.py`, `5_cleanup.py`, `6_text_features.py`, `1_run_modeling.py`, `2_predict_unseen.py`, `2_run_holdout_fe.py` are all generic and config-driven.

---

## Phase 5: Production Inference (`5_Inference/`)

Self-contained folder for deploying single-patient predictions. **No dependency on training code** — everything is bundled.

### Setup (once after training)
```bash
python 1_package_artifacts.py --window 12mo    # copies models + pipeline code
```

### Predict (per patient)
```bash
python 2_predict_single_patient.py --input patient.json --window 12mo
```

### What Gets Deployed
```
5_Inference/                           DEPLOY THIS FOLDER ONLY
  2_predict_single_patient.py          Inference entry point
  requirements.txt                     pip install -r requirements.txt
  pipeline/                            Bundled FE code (auto-copied)
    config.py, 3_pipeline.py, 4_cancer_features.py, 6_text_features.py
  artifacts/12mo/                      Saved models + transformers
    models/                            xgboost, lightgbm, catboost, config.json
    transformers/                      tfidf_vectorizer, svd, bert_pca
    code_category_mapping.json
    selected_features.csv
  sample_input/sample_patient.json     Example input format
```

### Output
```json
{
  "patient_id": "ABC123",
  "risk_score": 0.7312,
  "prediction": "HIGH RISK",
  "top_risk_factors": [
    {"rank": 1, "feature": "PROST_LAB_psa_elevated_10", "value": 1.0},
    {"rank": 2, "feature": "PROST_urinary_acceleration", "value": 1.0}
  ]
}
```

---

## Phase 6: Explainability & Enhancement Loop (`6_Explainability/`)

Ensures every prediction can be explained to a clinician in plain language. See `6_Explainability/README.md` for full documentation.

### The Loop
```bash
# 1. Compute SHAP → generate clinician reports
python 1_explain_predictions.py --window 12mo

# 2. Audit: how many top features are unexplainable?
python 2_audit_features.py --window 12mo

# 3. Auto-remove opaque features → retrain → compare
python 3_enhancement_loop.py --window 12mo
```

### Key Files
| File | Purpose |
|------|---------|
| `feature_dictionary.py` | Maps every feature to a clinical description + explainability class (direct/indirect/opaque) |
| `1_explain_predictions.py` | SHAP analysis → per-patient clinician reports |
| `2_audit_features.py` | Flags important-but-opaque features, scores explainability |
| `3_enhancement_loop.py` | Removes opaque → retrains → checks AUC drop → repeats |

### Sample Clinician Report
```
Patient: PATIENT_456
Risk Score: 68.40%  |  Prediction: HIGH RISK

WHY THIS PATIENT IS FLAGGED:
  1. PSA above 10 ng/mL (significantly elevated)
  2. PSA rising rapidly (>2 ng/mL change)
  3. Urinary symptoms worsening over time
  4. Bone or back pain present (possible metastatic signal)
  5. Pain medication prescriptions increasing
```

---

## Dependencies

| Category | Packages |
|----------|----------|
| Core | pandas, numpy, scikit-learn, pathlib |
| Modeling | xgboost, lightgbm, catboost, optuna, joblib |
| Text/NLP | sentence-transformers (PubMedBERT) |
| Explainability | shap |
| Optional | matplotlib (plots) |
