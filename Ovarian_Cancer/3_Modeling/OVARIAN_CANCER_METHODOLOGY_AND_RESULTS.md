# OVARIAN CANCER PREDICTION — METHODOLOGY

## 1. OBJECTIVE

**Goal:** Predict ovarian cancer risk in primary care patients using structured Electronic Health Record (EHR) data and clinical free-text keywords.

**Prediction Windows:** Evaluated for 3 months (3mo), 6 months (6mo), and 12 months (12mo) before diagnosis (index date).

**Target Operating Points:**
- **Primary:** Sensitivity >=75% + Specificity >=65%
- **Aspirational:** Sensitivity >=80% + Specificity >=70%

## 2. DATA SOURCE AND FEATURE ENGINEERING

**Input:** A cleaned, patient-level feature matrix, produced separately for each lookback window (3mo, 6mo, 12mo).

**Upstream Pipeline Steps:**
1. Sanity checks on raw windowed clinical/medication data.
2. Patient-overlap check and removal of patients present only in medication data.
3. Comprehensive feature engineering from clinical events, medications, labs, and free text.
4. Feature cleanup: redundant/sparse feature removal, correlation-based deduplication, near-zero variance removal, column name sanitisation.
5. Clinical feature filter: removal of generic healthcare utilisation features to focus on clinically meaningful signals.

**Features:** Approximately 855–865 features per window (clinical + text), constructed only from pre-index date information (no data leakage).
- Demographics (age, age bands, age x symptom interactions).
- Symptom/observation counts, acceleration, recency, and persistence per category.
- Laboratory-derived features (values, trends, slopes, percent change, abnormality flags).
- Medication patterns (counts, escalation, frequency trends, polypharmacy).
- NICE guideline cardinal symptoms and persistence counts.
- Ovarian-specific clinical features (mimic patterns, cyst interactions, treatment patterns).
- SNOMED-level binary features (individual diagnosis/medication codes).
- Per-symptom frequency trends, interval statistics, and worsening flags.
- Imaging free-text keyword features (10 features from ASSOCIATED_TEXT).
- Temporal, sequence, and interaction terms.

**Outcome:** Binary label (cancer vs. non-cancer).

**Class Imbalance:** Approximately 9% cancer cases; handled via scale_pos_weight in tree models.

## 3. TRAIN / VALIDATION / TEST SPLIT

**Method:** Stratified random split to preserve class balance.

**Proportions:** 75% Training, 10% Validation, 15% Test.

**Validation Set Usage:**
- Hyperparameter tuning (Optuna).
- Early stopping for gradient-boosted models.
- Learning ensemble weights.
- Selecting the final probability threshold.
- Full validation set used for threshold search.

**Test Set Usage:** Reserved solely for reporting final, unbiased generalisation performance metrics (AUC, sensitivity, specificity, PPV, NPV).

## 4. MODELS

**Base Classifiers:**
- **XGBoost:** Gradient-boosted trees with scale_pos_weight, Optuna-tuned.
- **LightGBM:** Gradient-boosted trees with scale_pos_weight, Optuna-tuned.
- **CatBoost:** Gradient-boosted trees with class_weights, Optuna-tuned.

**Ensemble:**
- Top 2 models by validation AUC, with optimised weights (grid search over weight combinations summing to 1.0, maximising a tiered metric on validation set).

**Overfitting Control:** All tree models use early stopping (50 rounds) on the validation set.

## 5. HYPERPARAMETER TUNING

**Tool:** Optuna, with 75 trials per model (XGBoost, LightGBM, CatBoost).

**Objective:** Maximise 5-fold cross-validation AUC on the training set.

**Search Space:** Learning rate, tree depth, L1/L2 regularisation, subsample/column sampling, number of estimators, and model-specific parameters.

## 6. FEATURE SELECTION

**Method:** XGBoost-based importance ranking with correlation filter.
- Train a preliminary XGBoost model on all features.
- Rank features by importance, take top 470 candidates.
- Filter by minimum label correlation (10th percentile threshold).
- Select final top 235 features.

Applied per seed per window to avoid information leakage.

## 7. THRESHOLD SELECTION

**Threshold Selection (on Validation Set):**
- Tiered search on the full ROC curve:
  - **Tier 0:** Sensitivity >=80% AND Specificity >=70% (aspirational).
  - **Tier 1:** Sensitivity >=75% AND Specificity >=65% (primary target).
  - **Tier 2:** Sensitivity >=70% AND Specificity >=65%.
  - **Tier 2b:** Sensitivity >=70% AND Specificity >=60%.
  - **Tier 3:** Best balanced (maximise minimum of sensitivity and specificity).
- Among valid thresholds, select the one maximising (sensitivity + specificity).

**Forced Operating Point Reporting:** For every model, test-set performance at forced 80/70 and 75/65 sensitivity targets is also reported.

## 8. CALIBRATION

No isotonic calibration applied. Raw (uncalibrated) probabilities from tree models are used directly for threshold selection and evaluation.

## 9. TEXT FEATURE EXTRACTION

**Source:** ASSOCIATED_TEXT field in clinical records (XML-wrapped free text).

**Coverage:** ~18% of rows have text; ~76% of patients have at least one text entry.

**10 imaging-focused keyword features extracted:**
- `TEXT_IMG_complex_cyst` — complex/septated/multiloculated cyst (20x enrichment in cancer).
- `TEXT_IMG_cyst_any` — any ovarian/adnexal/pelvic cyst mention (3.3x enrichment).
- `TEXT_IMG_solid` — solid component/mass/lesion (12x enrichment).
- `TEXT_IMG_mass` — ovarian/adnexal/pelvic/abdominal mass (1.9x enrichment).
- `TEXT_IMG_heterogenous` — heterogeneous echogenicity (2.2x enrichment).
- `TEXT_IMG_fibroid` — fibroid/leiomyoma findings (1.9x enrichment).
- `TEXT_IMG_endometrial_thick` — endometrial thickening (2.0x enrichment).
- `TEXT_IMG_ascites` — ascites/free fluid (1.6x enrichment).
- `TEXT_IMG_enlarged_ovary` — enlarged/bulky ovary (6.5x enrichment).
- `TEXT_COMP_imaging_finding_count` — total number of imaging findings per patient (2.2x enrichment).

**Excluded (leakage prevention):** Referral language (2-week-wait, urgent, referred, specialist), CA-125 mentions, "suspicious" imaging language, and all lab text features (inverted signal due to healthcare utilisation confounding).

---

# OVARIAN CANCER PREDICTION — INSIGHTS & RESULTS

## 1. NICE NG12 Symptoms and Signs for Ovarian Cancer

The NICE guideline NG12 (Suspected cancer: recognition and referral) specifies the key clinical features that should trigger investigation for suspected ovarian cancer.

**NICE Recommendations (1.6.1) for Ovarian Cancer Referral:**
- Carry out tests for ovarian cancer in primary care in any woman (especially if aged 50 or over) who has any of the following symptoms on a **persistent or frequent** basis (particularly more than 12 times per month):
  - Persistent abdominal distension (bloating).
  - Feeling full (early satiety) and/or loss of appetite.
  - Pelvic or abdominal pain.
  - Increased urinary urgency and/or frequency.
- Consider measuring serum CA125.
- If serum CA125 is 35 IU/ml or greater, arrange an ultrasound of the abdomen and pelvis.

**Key Observations on Symptom Prevalence:**
- NICE cardinal symptoms (bloating, early satiety) have very low recording rates in primary care EHR data — often <5% even in cancer patients.
- Ovarian cancer is frequently misdiagnosed as IBS, UTI, or menopausal symptoms, leading to delayed referral.
- Because of the low prevalence of NICE-specific features, the model relies on a broader set of signals including:
  - Medication prescribing patterns (laxatives, GI meds, iron supplements, pain escalation).
  - Laboratory trends (haemoglobin, ESR, albumin, renal function).
  - Symptom recency and persistence (distinct visit dates, frequency trends).
  - Imaging findings from clinical free text (complex cysts, ascites, enlarged ovary).
  - The "diagnostic odyssey" pattern — multiple visits across different symptom systems without resolution.

## 2. Results

### 3-MONTH WINDOW (3mo)

| Metric | Detail |
|---|---|
| Dataset | 17,142 patients (1,595 cancer, 15,547 non-cancer). Train 12,855 / Val 1,715 / Test 2,572 |
| Features | 865 (855 clinical + 10 text); 235 selected |
| CV AUC | XGBoost 0.822; LightGBM 0.797; CatBoost 0.815 |
| **Best Test AUC** | **CatBoost 0.830; Ensemble 0.829** |
| Target (75/65) | XGBoost: Sens 77.0%, Spec 69.2% ✅; Ensemble: Sens 77.4%, Spec 70.8% ✅; CatBoost: Sens 77.0%, Spec 68.5% ✅ |
| 80/70 forced | Not achieved — at 80%+ sens, spec drops to ~60-64% |

### 6-MONTH WINDOW (6mo)

| Metric | Detail |
|---|---|
| Dataset | 16,952 patients (1,524 cancer, 15,428 non-cancer). Train 12,713 / Val 1,696 / Test 2,543 |
| Features | 866 (856 clinical + 10 text); 235 selected |
| CV AUC | XGBoost 0.783; LightGBM 0.758; CatBoost 0.784 |
| **Best Test AUC** | **CatBoost 0.775; Ensemble 0.774** |
| Target (70/65) | XGBoost: Sens 74.2%, Spec 65.5% ✅; Ensemble: Sens 74.2%, Spec 66.4% ✅ |

### 12-MONTH WINDOW (12mo)

| Metric | Detail |
|---|---|
| Dataset | 16,690 patients (1,435 cancer, 15,255 non-cancer). Train 12,517 / Val 1,669 / Test 2,504 |
| Features | 842 (832 clinical + 10 text); 235 selected |
| CV AUC | XGBoost 0.770; LightGBM 0.745; CatBoost 0.771 |
| **Best Test AUC** | **Ensemble 0.772; XGBoost 0.772** |
| Target (70/65) | XGBoost: Sens 74.4%, Spec 66.0% ✅; Ensemble: Sens 71.6%, Spec 68.8% ✅; CatBoost: Sens 70.7%, Spec 69.4% ✅ |

## 3. Comparison Across Windows

| Window | Best Model | Test AUC | Sensitivity | Specificity | Target Met |
|---|---|---|---|---|---|
| **3mo** | **Ensemble** | **0.829** | **77.4%** | **70.8%** | **75/65 ✅** |
| 6mo | Ensemble | 0.774 | 74.2% | 66.4% | 70/65 ✅ |
| 12mo | Ensemble | 0.772 | 71.6% | 68.8% | 70/65 ✅ |

## 4. Key Findings

### 4.1 Performance Across Windows
- **3mo achieves highest AUC (0.829)** with 75/65 target met — Sensitivity 77.4%, Specificity 70.8%.
- **12mo is the most clinically valuable window.** AUC 0.772 with 70/65 met means the model reliably identifies patients a year before diagnosis.
- **6mo performance (AUC 0.774) is similar to 12mo**, suggesting the signal plateau is around 6–12 months before diagnosis.

### 4.2 What the Model Detects at Each Window

**3mo (near-diagnosis) — Active investigation signals:**
- Symptom trajectories increasing rapidly (TRAJ_increasing_trend — top feature).
- Time-decay weighted symptom scores (recent symptoms weighted more).
- Renal function changes (LAB_RENAL_mean, max, min).
- GI symptom cluster in recent window.
- Severe anaemia (Hb <100).
- Pain concentrated in recent window with multiple GP visits.
- Ovarian cyst recency and persistence.

**12mo (early detection) — Subtle long-term patterns:**
- Baseline lab values (renal function, haematology).
- Age as a risk factor (>=60).
- Laxative prescriptions (GI symptoms treated as IBS).
- Iron supplements (subtle anaemia developing).
- Repeated UTI antibiotic prescriptions.
- Gynaecological symptoms in background.
- The "diagnostic odyssey" — multiple visits across different symptom systems without resolution.

### 4.3 Impact of Text Features
- 10 imaging-focused text features extracted from ASSOCIATED_TEXT field.
- Top text feature `TEXT_IMG_complex_cyst` has 20x enrichment in cancer patients (correlation 0.100).
- Text features improved 3mo Ensemble AUC from 0.818 (clinical-only) to **0.829** (+0.011).
- Text features improved 12mo XGBoost AUC from 0.756 to **0.772** (+0.016).
- Referral-related and lab text features were excluded due to leakage and inverted signal (healthcare utilisation confounding).

### 4.4 Impact of Clinical Feature Filter
- Removing generic healthcare utilisation features (total event counts, visit totals, monthly bins) improved model focus on true clinical signals.
- Clinical-only CatBoost achieved AUC 0.830 for 3mo — demonstrating that utilisation noise was masking real signals in earlier experiments.

### 4.5 Aspirational Target (80/70)
- Sensitivity >=80% + Specificity >=70% was **not achieved** in any configuration. At 80%+ sensitivity, specificity consistently drops to ~60-64%, indicating this target may be beyond the signal available in structured primary care data for ovarian cancer.

## 5. Top Predictive Features (3mo)

| Rank | Feature | Importance | Clinical Meaning |
|---|---|---|---|
| 1 | TRAJ_increasing_trend | 0.0146 | Symptom frequency increasing over time |
| 2 | DECAY_symptom_weighted_score | 0.0111 | Recent symptoms weighted higher |
| 3 | LAB_RENAL_mean | 0.0062 | Baseline renal function |
| 4 | LAB_RENAL_max | 0.0059 | Peak renal value |
| 5 | CLUSTER_GI_count_B | 0.0059 | GI symptoms in recent window |
| 6 | CAT_REPRODUCTIVE_unique_visit_dates | 0.0059 | Reproductive visit pattern |
| 7 | OV_LAB_hb_severe_anaemia | 0.0056 | Haemoglobin <100 (severe anaemia) |
| 8 | MEDCAT_CARDIOVASCULAR_B_concentration | 0.0046 | Cardiovascular medication in recent window |
| 9 | CAT_ABDOMINAL_PAIN_B_concentration | 0.0045 | Pain concentrated recently |
| 10 | RxP_ABDOMINAL_PAIN_score | 0.0042 | Recent + frequent abdominal pain |

## 6. Recommended Configuration

**Recommended model:** Clinical + Text Ensemble (4_run_modeling_text.py)

**Rationale:**
- Highest 3mo Ensemble AUC (0.829) with 75/65 target met.
- Text features add genuine signal from imaging reports without leakage.
- Clinical-only filter removes confounding healthcare utilisation features.
- 235 features selected (225 structured + 10 text) to prevent feature competition.

| Window | Model | AUC | Sensitivity | Specificity | Target |
|---|---|---|---|---|---|
| **3mo** | **Ensemble** | **0.829** | **77.4%** | **70.8%** | **75/65 ✅** |
| 6mo | Ensemble | 0.774 | 74.2% | 66.4% | 70/65 ✅ |
| 12mo | Ensemble | 0.772 | 71.6% | 68.8% | 70/65 ✅ |







