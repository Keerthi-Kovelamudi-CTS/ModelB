# Prostate Cancer Prediction Pipeline

Config-driven, modular pipeline for predicting prostate cancer from primary-care EHR data (SNOMED-coded observations, medications, labs, and clinical text). Six phases from SNOMED ranking through deployable inference and SHAP explainability.

**Porting to another cancer**: change `config.py` and `4_cancer_features.py` (+ regenerate `code_category_mapping.json`). All other code is generic.

---

## Directory Layout

```
Prostate_Cancer/
    1_Top_Snomed/              Phase 1 — SNOMED feature ranking (exploratory)
    2_Feature_Engineering/     Phase 2 — Feature engineering pipeline
    3_Modeling/                Phase 3 — Training, tuning, ensemble, calibration
    4_ExpandedData_Test/       Phase 4 — Holdout / expanded-cohort validation
    5_Inference/               Phase 5 — Production single-patient prediction
    6_Explainability/          Phase 6 — SHAP + feature audit + enhancement loop
    README.md                  This file
    run_all.sh                 End-to-end driver (all phases sequentially)
```

---

## Pipeline Overview (end-to-end)

```
        SNOMED codes & categories               Trained models
        from clinical literature                 (for deployment)
                 │                                     ▲
                 ▼                                     │
 ┌─── Phase 1 ────────┐        ┌───────── Phase 5 ────┴──────┐
 │  Top_Snomed        │        │  Inference                   │
 │  rank SNOMEDs by   │        │  single-patient JSON →       │
 │  statistical       │        │  calibrated risk + top       │
 │  signal            │        │  risk factors                │
 └─────────┬──────────┘        └──────────────────────────────┘
           │                                     ▲
           ▼                                     │
 ┌─── Phase 2 ────────┐        ┌───────── Phase 4 ────────────┐
 │  Feature Eng.      │        │  Expanded-Data Test          │
 │  cohort → clean →  │        │  ~300K holdout patients →    │
 │  ~1000 features →  │◀──FE───│  apply saved transformers →  │
 │  cleanup → text    │  funcs │  predict (calibrator + thr)  │
 └─────────┬──────────┘        └──────────────────────────────┘
           │                                     ▲
           ▼                                     │
 ┌─── Phase 3 ────────┐        ┌───────── Phase 6 ────────────┐
 │  Modeling          │        │  Explainability              │
 │  train/val/test →  │        │  SHAP on highest-weight      │
 │  XGB+LGBM+CB Optuna│───────▶│  model → clinician reports   │
 │  ensemble (top-3)+ │        │  → audit opaque features →   │
 │  isotonic calib +  │        │  enhancement loop            │
 │  Tier-0 + Tier-90  │        └──────────────────────────────┘
 └────────────────────┘
```

---

## Design Decisions (the "why")

- **Temporal windows A vs B** are the core leakage defence. A = earlier observation period, B = later; both strictly before `INDEX_DATE`. `2_clean_data.py` enforces this with a leakage guard that drops any row where `EVENT_DATE >= INDEX_DATE`.
- **No downsampling of negatives** — natural ~1:9 positive:negative ratio is preserved. Class imbalance is handled at modeling time via `scale_pos_weight` (XGBoost/LightGBM) and `auto_class_weights='Balanced'` (CatBoost).
- **Config-driven** — everything cancer-specific lives in `config.py`. Porting to a new cancer only touches `config.py` + `4_cancer_features.py` + regenerating `code_category_mapping.json`.
- **Isotonic calibration** — ensemble scores are mapped to probabilities via `IsotonicRegression` fit on a 70% slice of val; threshold is selected on the held-out 30% to avoid fit-and-threshold-on-same-data bias.
- **Top-3 ensemble** — all three boosters (XGBoost, LightGBM, CatBoost) contribute; weights are grid-searched on a tiered sens/spec objective.
- **Tier-0 primary + Tier-90 alternative** — Tier-0 (sens ≥ 80%, spec ≥ 70%) is the main operating point; Tier-90 (sens ≥ 90%, max achievable spec) is reported alongside for high-sensitivity screening use cases.
- **Same FE code in training and holdout** — Phase 4's holdout pipeline imports FE functions directly from Phase 2 (no duplication, zero drift risk).

---

## Quick Start

```bash
# 1. Full feature engineering pipeline (all steps, all windows)
cd 2_Feature_Engineering/
python 0_run_pipeline.py                 # or --step sanity | clean | fe | cleanup | text

# 2. Train models (3mo, 6mo, 12mo × 4 seeds each)
cd ../3_Modeling/
python 1_run_modeling.py

# 3. Holdout / expanded-cohort validation
cd ../4_ExpandedData_Test/
python 1_preprocess_holdout.py
python 2_run_holdout_fe.py
cd ../3_Modeling/
python 2_predict_unseen.py --window 12mo

# 4. Package inference artifacts for deployment
cd ../5_Inference/
python 1_package_artifacts.py --window 12mo

# 5. Explainability
cd ../6_Explainability/
python 1_explain_predictions.py --window 12mo
python 2_audit_features.py --window 12mo
python 3_enhancement_loop.py --window 12mo
```

---

## Phase 1 — `1_Top_Snomed/`

Ranks raw SNOMED observation codes by statistical signal (χ², OR, information gain, mutual information, and combined rank) against cancer label. Output goes to `output/Scores_prostate/prostate_combined_top150.csv`.

**Note**: the ranking is exploratory; it feeds category definitions (in `config.OBS_CATEGORIES`) that Phase 2 uses, but is not a runtime dependency of Phases 2–6.

---

## Phase 2 — `2_Feature_Engineering/`

### Pipeline Flow

```
Raw CSVs (from SQL)
    |
    v
[1_sanity_check.py]   — patient overlap removal, dedup, sex/age filter
[2_clean_data.py]     — lab range clamping, date parsing, leakage guard
[3_pipeline.py]       — 500+ generic features across 7 blocks
     +
[4_cancer_features.py] — prostate-specific features (PSA dynamics, urinary, bone)
[5_cleanup.py]        — remove near-zero variance, high-correlation, leakage
[6_text_features.py]  — keywords + TF-IDF + BERT (PubMedBERT on GPU)
    |
    v
Clean feature matrices, text/TF-IDF/BERT embeddings, saved transformers
```

### Scripts

| Script | What it does |
|--------|--------------|
| `config.py` | Single source of truth: CANCER_NAME, OBS/MED/LAB categories, LAB_RANGES, INTERACTION_PAIRS, text patterns, modeling params, paths |
| `0_run_pipeline.py` | Orchestrator. `--step all \| sanity \| clean \| fe \| cleanup \| text` |
| `1_sanity_check.py` | Checks row counts, date ranges, category distribution, nulls. Fixes patient overlap (pos∩neg → drop), med-only patients, duplicates. Filters sex (male) and age (18+). Saves master patient list. **No downsampling — natural class ratio is preserved** |
| `2_clean_data.py` | Parses dates, clamps lab outliers using `config.LAB_RANGES`, encodes time windows. **Includes a leakage guard that drops any row where `EVENT_DATE >= INDEX_DATE`** |
| `3_pipeline.py` | Generic feature engineering (parameterised by config). 7 blocks, ~500–1000 features. See below. NaN-safe slope calculations (guards against degenerate lab series) |
| `4_cancer_features.py` | Prostate-specific block: PSA dynamics, PSA age-band percentile, urinary patterns, bone/metastatic signals, treatment flags, composite risk score |
| `5_cleanup.py` | Removes duplicate columns, high-null (>95%), zero/near-zero variance, high-correlation pairs (>0.98), and features with >0.5 label correlation (possible leakage). Smart NaN fill by feature type |
| `6_text_features.py` | NLP: extracts text from ASSOCIATED_TEXT, builds keyword flags, TF-IDF (15 dims via SVD), and BERT embeddings (15 dims via PCA). **Auto-detects CUDA** via `torch.cuda.is_available()` and uses batch_size=256 on GPU |

### Step 3: Generic Features (`3_pipeline.py`)

- `build_clinical_features` — demographics, per-category obs counts (A/B), flags, lab stats (mean/min/max/std/first/last/delta/slope), investigation patterns, aggregates, temporal features
- `build_medication_features` — per-category counts, quantity stats, unique drugs, polypharmacy
- `build_interaction_features` — obs×med pairs from `INTERACTION_PAIRS`, multi-symptom burden
- `build_advanced_features` — symptom clusters, visit patterns, temporal trajectories, lab trajectories, med escalation, cross-domain, time-decay weighted scores
- `extract_maximum_features` — monthly bins, rolling 3-month counts, granular per-category, co-occurrence pairs, entropy, Gini
- `build_new_signal_features` — per-lab-term (top 30), recency, distinct visits
- `build_trend_features` — per-symptom frequency trends, worsening flags, per-med recurrence

### Step 4: Prostate-Specific (`4_cancer_features.py`)

| Block | Features | Clinical rationale |
|-------|----------|-------------------|
| PSA dynamics | count A/B, acceleration, latest/first, elevated flags (>4/10/20/100), delta, rising, rapid rise, velocity, doubling time | Primary screening marker; rising PSA triggers investigation |
| **PSA age-band percentile (new)** | percentile within 10-yr age bands (<50, 50–59, 60–69, 70–79, 80+), top-10%, top-5% | PSA>4 means different things at 50 vs 75; within-band percentile captures that |
| PSA percentile (global) | rank-pct vs all patients, top 5/10/25% | Cohort-relative positioning |
| Urinary | symptom counts, acceleration, haematuria, ED | LUTS + haematuria raise suspicion |
| Bone / metastatic | bone pain, lower back pain, ALP elevation (>130 U/L) | Bone mets common in advanced disease |
| Lab prognostic | CRP (>10), haemoglobin (<120 = anaemia), testosterone | Anaemia / low T suggest advanced disease |
| Treatment flags | alpha blockers, 5-ARI, antibiotics, pain meds, corticosteroids, BPH combo | BPH combo suggests benign; escalating pain may signal progression |
| Diagnostic pathway | abnormal-imaging flag, symptom-to-imaging gap | Investigation urgency |
| Risk score | composite: age≥65 + PSA>10 + urinary + bone + abnormal imaging | D'Amico-adjacent |
| Age × risk | elderly (70+), age×PSA, age×urinary, PSA+bone combo | High-risk combinations |

### Data & Results Layout

```
data/{3mo,6mo,12mo}/
    prostate_{w}_{obs,med}_dropped.csv   (raw/after SQL drop)

results/
    1_sanity_check/  master_patients_{w}.csv
    2_clean_data/    (logs only)
    3_feature_engineering/{w}/   feature_matrix_final_{w}.csv
    4_cancer_features/{w}/       (merged into step 3)
    5_cleanup/{w}/               feature_matrix_clean_{w}.csv
    6_text_features/
        keywords/{w}/            text_features_{w}.csv
        tfidf_embeddings/{w}/    text_embeddings_{w}.csv
        bert_embeddings/{w}/     bert_embeddings_{w}.csv
        fitted_transformers/{w}/ tfidf_vectorizer, svd, bert_pca (pkl)
```

---

## Phase 3 — `3_Modeling/`

### Pipeline Flow

```
Clean features + text + TF-IDF + BERT
    |
    v
For each window (3mo, 6mo, 12mo):
  For each seed (13, 42, 45, 77):
    Split 75% / 10% / 15% (stratified)
    Feature selection  — rank-fused XGBoost + LightGBM importances, label-correlation filter
    Tune               — Optuna 75 trials per model: XGBoost, LightGBM, CatBoost
    Ensemble           — top-3 by val AUC, grid-search weights (tiered objective)
    Calibrate          — IsotonicRegression fit on a held-out slice of val (70% fit / 30% threshold)
    Threshold          — picked on the 30% calibrated slice (no double-dip)
    Report             — test AUC + Tier-0 sens/spec + Tier-90 alt operating point
    Save               — models + config.json (ensemble, weights, threshold, calibrator)
```

### Scripts

| Script | Purpose |
|--------|---------|
| `1_run_modeling.py` | Full training loop. Loads clean matrices + text + embeddings, filters to clinical features, trains 3 boosters with Optuna, builds weighted 3-model ensemble, fits isotonic calibration, selects threshold, reports on test. Saves models, predictions, feature importances |
| `2_predict_unseen.py` | Loads saved models + calibrator + threshold. Applies to any new feature matrix. Handles N-model ensemble + optional calibrator |

### Threshold Tiers

| Tier | Sens | Spec | Purpose |
|------|------|------|---------|
| Tier 0 | ≥80% | ≥70% | **Preferred** primary operating point |
| Tier 1 | ≥75% | ≥65% | Fallback if Tier 0 unachievable |
| Tier 2 | ≥70% | ≥65% | Weaker fallback |
| Tier 2b | ≥70% | ≥60% | Minimum viable |
| Tier 3 | max min(sens, spec) | balanced | Last resort |
| **Tier 90 (alt)** | ≥90% | max achievable | **Reported alongside** the primary tier — high-sensitivity screening use case |

### config.json schema (saved per window)

```python
{
  'selected_features': [...],
  'threshold': float,                       # primary operating point
  'ensemble_models':  ['catboost', 'xgboost', 'lightgbm'],  # up to 3
  'ensemble_weights': [w1, w2, w3],         # sum to 1
  'calibrator': <IsotonicRegression>,       # applied to ensemble output
  'tier90_threshold': float or nan,         # alt operating point
  'seed': int,
  'window': '3mo' | '6mo' | '12mo',
}
```

### Results Layout

```
results/
  1_training/
    {w}/
      saved_models/
        xgboost_model.pkl, lightgbm_model.pkl, catboost_model.pkl, config.json
      predictions_{w}.csv      (y_true, y_pred_proba_raw, y_pred_proba [calibrated], y_pred)
      selected_features_{w}.csv
      feature_importances_{w}.csv
    final_results.csv          (all windows × seeds, incl. tier90 cols)
    modeling.log
  2_predictions/
    {w}/
      predictions_unseen_{w}.csv
      predictions_unseen_{w}_metrics.txt
```

---

## Phase 4 — `4_ExpandedData_Test/`

Validates trained models on a larger, mostly-negative holdout (e.g., ~300K patients) to estimate real-world false-positive rate.

### Pipeline Flow

```
Raw 300K CSV (from SQL / Snowflake / BigQuery)
    |
    v
[1_preprocess_holdout.py]
    — Excludes training patients (prevents leakage)
    — Maps CODE_ID → CATEGORY via code_category_mapping.json
    — Windows events into A/B (same as training)
    — Saves prostate_{w}_{obs,med}_dropped.csv
    |
    v
[2_run_holdout_fe.py]
    — Imports THE SAME FE functions from 2_Feature_Engineering (zero drift risk)
    — Applies saved transformers (tfidf, svd, bert_pca) — no refitting
    |
    v
holdout_features_{w}.csv

    |
    v
[3_Modeling/2_predict_unseen.py]  (calibrator + threshold applied)
```

---

## Phase 5 — `5_Inference/`

Self-contained folder for deploying single-patient predictions. No dependency on training code — everything is bundled.

### Usage

```bash
# Once after training
python 1_package_artifacts.py --window 12mo     # copies models + pipeline code

# Per patient
python 2_predict_single_patient.py --input patient.json --window 12mo
```

### Layout (what gets deployed)

```
5_Inference/                    DEPLOY THIS FOLDER ONLY
  2_predict_single_patient.py
  requirements.txt
  pipeline/                     (bundled FE code — auto-copied by 1_package_artifacts)
    config.py, 3_pipeline.py, 4_cancer_features.py, 6_text_features.py
  artifacts/{w}/
    models/                     xgboost, lightgbm, catboost, config.json (incl. calibrator)
    transformers/               tfidf_vectorizer, svd, bert_pca
    code_category_mapping.json
    selected_features.csv
  sample_input/sample_patient.json
```

### Output

```json
{
  "patient_id": "ABC123",
  "risk_score": 0.7312,                      // calibrated probability
  "prediction": "HIGH RISK",
  "threshold": 0.4216,
  "top_risk_factors": [
    {"rank": 1, "feature": "PROST_LAB_psa_elevated_10", "value": 1.0},
    {"rank": 2, "feature": "PROST_LAB_psa_age_band_top10pct", "value": 1.0}
  ]
}
```

---

## Phase 6 — `6_Explainability/`

### The Loop

```bash
python 1_explain_predictions.py --window 12mo    # SHAP on highest-weight ensemble model
python 2_audit_features.py --window 12mo         # Flag important-but-opaque features
python 3_enhancement_loop.py --window 12mo       # Auto-remove opaque → retrain → compare
```

### Files

| File | Purpose |
|------|---------|
| `feature_dictionary.py` | Maps every feature to a clinical description + explainability class (direct / indirect / opaque). Includes new PSA age-band features |
| `1_explain_predictions.py` | SHAP analysis → per-patient clinician reports. Picks the **highest-weight** ensemble model (most representative) |
| `2_audit_features.py` | Flags important-but-opaque features, scores explainability coverage |
| `3_enhancement_loop.py` | Removes opaque features → retrains → checks AUC drop → iterates |

### Sample Clinician Report

```
Patient: PATIENT_456
Risk Score: 68.40%  |  Prediction: HIGH RISK

WHY THIS PATIENT IS FLAGGED:
  1. PSA above 10 ng/mL (significantly elevated)
  2. PSA in top 10% for the patient's age band
  3. PSA rising rapidly (>2 ng/mL change)
  4. Urinary symptoms worsening over time
  5. Bone or back pain present (possible metastatic signal)
```

---

## Porting to Another Cancer

| Step | File | What to change |
|------|------|----------------|
| 1 | `config.py` | CANCER_NAME, DATA_PREFIX, PREFIX, OBS_CATEGORIES, MED_CATEGORIES, LAB_CATEGORIES, CLUSTER_DEFINITIONS, SYMPTOM_CATEGORIES, INTERACTION_PAIRS, LAB_RANGES, TEXT patterns |
| 2 | `4_cancer_features.py` | Rewrite `build_cancer_specific_features` with the new cancer's clinical logic |
| 3 | `code_category_mapping.json` | Regenerate from the new cancer's training SQL |

Everything else (`3_pipeline.py`, `5_cleanup.py`, `6_text_features.py`, `1_run_modeling.py`, `2_predict_unseen.py`, `2_run_holdout_fe.py`) is generic and config-driven.

---

## Environment

### Python dependencies

| Category | Packages |
|----------|----------|
| Core | pandas, numpy, scikit-learn |
| Modeling | xgboost, lightgbm, catboost, optuna, joblib |
| Text / NLP | sentence-transformers (PubMedBERT), transformers, torch |
| Explainability | shap |
| Optional | matplotlib (plots) |

### GPU

- BERT embedding (Phase 2, Step 6) auto-detects CUDA and uses `batch_size=256` on GPU.
- PyTorch wheel must match the driver — for a driver at CUDA 12.2, install `torch==2.x+cu121` (**not** `+cu130`):
  ```bash
  pip install --user --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```
- Boosters (XGBoost / LightGBM / CatBoost) run on CPU in the current setup.

---

## Log of Recent Changes

- **Removed 5:1 neg:pos downsample** in `1_sanity_check.py`; natural ratio preserved.
- **Leakage guard** added in `2_clean_data.py` (drops rows with `EVENT_DATE >= INDEX_DATE`).
- **NaN-safe polyfit slopes** in `3_pipeline.py` and `4_cancer_features.py` (4 sites) — eliminates LAPACK DLASCL warning flood.
- **PSA age-band percentile** features added in `4_cancer_features.py`.
- **BERT on GPU** — `6_text_features.py` auto-detects CUDA.
- **Top-3 ensemble** (was top-2) + **isotonic calibration** + **Tier-90 alt operating point** in `1_run_modeling.py`.
- **Multi-model feature selection** — rank-fused XGBoost + LightGBM importances.
- **Expanded Optuna search ranges** (n_estimators up to 1500, learning_rate down to 0.003, + `gamma`, `bagging_temperature`, `random_strength`, `min_split_gain`).
- **Downstream consumers** updated (`2_predict_unseen.py`, `5_Inference/2_predict_single_patient.py`, `6_Explainability/1_explain_predictions.py`, `feature_dictionary.py`) for the new ensemble/config schema.
