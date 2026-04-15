# OVARIAN CANCER PROJECT — FILE GUIDE

## Directory Structure Overview

```
Ovarian_Cancer/
├── 1_Top_Snomed/          # Phase 1: SNOMED identification
├── 2_Feature_Engineering/ # Phase 2: Feature engineering pipeline
├── 3_Modeling/            # Phase 3: Model training and results
└── PROJECT_FILE_GUIDE.md  # This file
```

---

## 2_Feature_Engineering/

### Scripts (run in order)

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `1_Sanitycheck+Patientoverlapcheck.py` | Raw windowed CSVs from Snowflake | Sanity check reports, overlap reports | Validates data quality, checks patient overlap between clinical and med data |
| `3_droppatients.py` | Raw CSVs + overlap results | `data/{3mo,6mo,12mo}/FE_ovarian_dropped_patients_*.csv` | Removes patients only in med data (no clinical records) |
| `4_Feature_engineering.py` | `data/{window}/FE_ovarian_dropped_patients_*.csv` | `results/FE/{window}/feature_matrix_final_{window}.csv` | **Main FE pipeline** — runs Steps 4a through 4f sequentially (see below) |
| `4g_text_features.py` | `data/{window}/FE_ovarian_dropped_patients_clinical_*.csv` | `results/Text_Features/{window}/text_features_{window}.csv` | Extracts 10 imaging keyword features from ASSOCIATED_TEXT |
| `4h_text_embeddings.py` | Same clinical CSVs | `results/Text_Embeddings/{window}/text_embeddings_{window}.csv` | TF-IDF + SVD embeddings (15 dimensions) from clinical text |
| `4i_bert_embeddings.py` | Same clinical CSVs | `results/BERT_Embeddings/{window}/bert_embeddings_{window}.csv` | PubMedBERT embeddings (15 dimensions) from clinical text |
| `5_feature_cleanup.py` | `results/FE/{window}/feature_matrix_final_{window}.csv` | `results/Cleanup_Finalresults/{window}/feature_matrix_clean_{window}.csv` | Removes duplicates, high-null, zero-variance, near-zero-variance, high-correlation features. Sanitises column names. |

### Feature Engineering Steps (inside 4_Feature_engineering.py)

| Step | What it does | Features added |
|------|-------------|----------------|
| **4a** | Base features: demographics, symptom counts (A/B windows), lab values, comorbidities, risk factors, investigations, medications, interaction features | ~577 |
| **4b** | Advanced: symptom clusters, visit patterns, temporal trajectories (quarterly), lab slopes, medication escalation, cross-domain interactions, time-decay scores | ~110 |
| **4c** | Maximum extraction: monthly time bins, per-category granular features, per-lab-term deep features, pairwise co-occurrence, sequence features, recurrence, rates, age interactions, comorbidity interactions, entropy | ~1,300 |
| **4d** | Ovarian-specific: NICE cardinal symptoms, IBS/UTI mimic patterns, ovarian cyst interactions, risk factors, lab thresholds (Hb, CRP, albumin, platelets, eGFR, ESR), diagnostic pathway, treatment patterns | ~50 |
| **4e** | Signal features: top 50 SNOMED binary features, top 30 medication terms, per-symptom recency scores, distinct visit dates, recency x persistence interactions | ~158 |
| **4f** | Lung-style trends (adapted from AI-UK lung pipeline): per-symptom frequency per month, trend slope, intervals, first/second half frequency, IS_WORSENING, per-medication trends, lab percent change, trend R-squared, trend direction | ~316 |

### Text Feature Pipeline (separate scripts)

| Step | Script | Features | Description |
|------|--------|----------|-------------|
| **4g** | `4g_text_features.py` | 10 | Regex keyword extraction from imaging text: complex cyst, solid, mass, ascites, enlarged ovary, heterogenous, fibroid, endometrial thickening, simple cyst, imaging finding count |
| **4h** | `4h_text_embeddings.py` | 15 | TF-IDF (500 terms, unigrams+bigrams) + SVD to 15 dense dimensions. Captures term frequency patterns. |
| **4i** | `4i_bert_embeddings.py` | 15 | PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO) sentence embeddings → PCA to 15 dimensions. Captures semantic meaning. |

**Text coverage:** ~76% of patients have some text, but only ~8% of cancer patients have genuinely useful clinical content (imaging findings, symptom details). The ASSOCIATED_TEXT field is ~83% lab boilerplate, ~9% admin, ~8% useful clinical content. Useful text is 100% redundant with structured SNOMED codes for patient identification — it only adds detail (e.g., cyst size, laterality).

**Leakage exclusions:** Referral language (2-week-wait, urgent, referred, specialist), CA-125 mentions, "suspicious" imaging text, and all lab text features were excluded due to leakage or inverted signal.

### Data Files

```
data/
├── 3mo/
│   ├── FE_ovarian_dropped_patients_clinical_windowed_3m.csv  # Clinical events
│   └── FE_ovarian_dropped_patients_med_windowed_3m.csv       # Medication events
├── 6mo/  (same structure)
└── 12mo/ (same structure)
```

**Columns in clinical CSV:** PATIENT_GUID, SEX, CANCER_ID, DATE_OF_DIAGNOSIS, INDEX_DATE, AGE_AT_DIAGNOSIS, AGE_AT_INDEX, EVENT_DATE, EVENT_AGE, MONTHS_BEFORE_INDEX, TIME_WINDOW (A=early, B=recent), EVENT_TYPE, CATEGORY, SNOMED_ID, TERM, ASSOCIATED_TEXT, VALUE, LABEL, TIME_WINDOW_NUM

**Columns in med CSV:** Same structure but EVENT_TYPE = MEDICATION, VALUE = quantity, DURATION_IN_DAYS

### Results Files

```
results/
├── FE/{window}/
│   ├── feature_matrix_{window}.csv           # After step 4a
│   ├── feature_matrix_enhanced_{window}.csv  # After step 4b
│   ├── feature_matrix_mega_{window}.csv      # After step 4c
│   └── feature_matrix_final_{window}.csv     # After step 4f (final)
├── Cleanup_Finalresults/{window}/
│   └── feature_matrix_clean_{window}.csv     # After step 5 (ready for modeling)
├── Text_Features/{window}/
│   └── text_features_{window}.csv            # 10 imaging keyword features
├── Text_Embeddings/{window}/
│   └── text_embeddings_{window}.csv          # 15 TF-IDF+SVD dimensions
└── BERT_Embeddings/{window}/
    └── bert_embeddings_{window}.csv          # 15 PubMedBERT dimensions
```

---

## 3_Modeling/

### Scripts

| Script | Features Used | Seeds | Results Folder | Description |
|--------|-------------|-------|----------------|-------------|
| `1_run_modeling.py` | All 1,080 features | [44] | `results_run2/` | All features including generic utilisation, no calibration |
| `2_run_modeling_balanced.py` | All features | [42,43,44,45] | `balanced_results/` | Same but with undersampled training (1:1 neg:pos) |
| `3_run_modeling_clinical.py` | ~855 clinical-only | [44] | `results_clinical/` | Removes generic healthcare utilisation features |
| `4_run_modeling_text.py` | ~895 clinical + text + embeddings | [44] | `results_text_emb/` | Clinical + 10 text keywords + 15 TF-IDF + 15 BERT |
| `5_run_modeling_text_4seeds.py` | ~895 clinical + text + embeddings | [42,43,44,45] | `results_text_emb_4seeds/` | **Final run** — same as above with 4 seeds + Spec@80%Sens reporting |

### What each modeling script does:

1. **Loads** `feature_matrix_clean_{window}.csv` (from cleanup)
2. **Filters** clinical-only (removes generic utilisation features)
3. **Merges** text keywords + TF-IDF embeddings + BERT embeddings (text scripts only)
4. **Splits** data: 75% train / 10% val / 15% test (stratified)
5. **Selects** top 265 features via XGBoost importance + correlation filter (225 structured + 40 text/embedding)
6. **Tunes** 3 models with Optuna (75 trials each): XGBoost, LightGBM, CatBoost
7. **Creates** Ensemble (weighted average of top 2 by val AUC)
8. **Finds** threshold on full validation set (tiered: 80/70 → 75/65 → 70/65 → 70/60 → balanced)
9. **Reports** forced operating points: what specificity each model achieves at 80% and 75% sensitivity
10. **Evaluates** on test set — reports AUC, sensitivity, specificity
11. **Saves** predictions, feature importances, threshold plots

### Results Folders

| Folder | Config | Seeds | Best 3mo AUC | Status |
|--------|--------|-------|-------------|--------|
| `results_run1/` | All features, old calibration | 4 seeds | 0.813 (Ensemble mean) | Complete |
| `results_run2/` | All features + trends, no calibration | seed 44 | 0.826 (Ensemble) | Complete |
| `results_clinical/` | Clinical-only | seed 44 | 0.830 (CatBoost) | Complete |
| `results_text/` | Clinical + 10 text keywords | seed 44 | 0.829 (Ensemble) | Complete |
| `results_text_emb/` | Clinical + text + TF-IDF + BERT | seed 44 | 0.831 (CatBoost) | Complete |
| **`results_text_emb_4seeds/`** | **Clinical + text + TF-IDF + BERT** | **4 seeds** | **0.814 (CatBoost mean)** | **Complete (final)** |

### Results Files (per window, per run)

```
{results_folder}/{window}/
├── feature_importances_{window}.csv    # All features ranked by XGBoost importance
├── selected_features_{window}.csv      # Top N features used for modeling
├── predictions_{window}.csv            # Patient-level predictions (GUID, label, model probabilities)
├── threshold_analysis_{window}.png     # ROC, PR, Sens/Spec vs threshold plots
├── feature_importance_{window}.png     # Top 30 features bar chart
└── shap_{window}.png                   # SHAP summary plot (if available)

{results_folder}/
├── final_threshold_results.csv         # All seeds/models/windows aggregated results
└── nohup_*.log                         # Full run log
```

### Other Files

| File | Description |
|------|-------------|
| `OVARIAN_CANCER_METHODOLOGY_AND_RESULTS.md` | Methodology write-up and results summary |
| `PROJECT_FILE_GUIDE.md` | This file |
| `lung_v2_ensemble_reference/` | Reference files from AI-UK lung cancer pipeline (training_v1.py, predict_unseen_input.py, sample data) |

---

## Key Results Summary

### Final 4-Seed Results (Clinical + Text + TF-IDF + BERT)

| Window | Best Model | AUC (mean±SD) | Sensitivity | Specificity | Spec@80%Sens | 70/65 Hit Rate |
|--------|-----------|---------------|-------------|-------------|-------------|----------------|
| **3mo** | **CatBoost** | **0.814±0.013** | **75.1%** | **72.0%** | **63.5%** | **4/4** |
| 6mo | Ensemble | 0.780±0.011 | 73.9% | 67.2% | 60.2% | 3/4 |
| **12mo** | **Ensemble** | **0.780±0.012** | **74.7%** | **68.0%** | **57.6%** | **4/4** |

### Single-Seed Comparison (Seed 44, CatBoost)

| Window | Clinical Only | Clinical + Text + Embeddings |
|--------|-------------|------------------------------|
| 3mo | AUC 0.830, Sens 80.8%, Spec 67.4% | AUC 0.831, Sens 78.7%, Spec 71.3% |
| 6mo | AUC 0.777, Sens 78.6%, Spec 62.2% | AUC 0.779, Sens 76.0%, Spec 65.3% |
| 12mo | AUC 0.767, Sens 74.9%, Spec 65.4% | AUC 0.767, Sens 81.4%, Spec 61.1% |

### Feature Contribution (3mo)

| Feature Group | Selected | % of Model Importance |
|--------------|----------|----------------------|
| Structured clinical | 244 | 94.2% |
| TF-IDF embeddings | 15 | 2.7% |
| BERT embeddings | 15 | 2.2% |
| Text keywords | 5 | 0.8% |

### Key Findings:
- **Structured clinical features carry 94% of the model's predictive power.**
- Text features add marginal value (~0.001 AUC improvement) because useful text is 100% redundant with structured SNOMED codes.
- At 80% sensitivity, specificity drops to 55-64% — the 80/70 target is not achievable with this data.
- The 75/65 target is achievable for 3mo (2/4 seeds) and 70/65 is reliable across all windows (3-4/4 seeds).
- 12mo is the most clinically valuable window — detecting cancer 1 year before diagnosis with AUC 0.780 and 70/65 in 4/4 seeds.
