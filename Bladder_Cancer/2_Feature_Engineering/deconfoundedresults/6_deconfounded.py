#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  BLADDER CANCER — FEATURE DECONFOUNDING (v5 fix)
  
  Problem: Top SHAP features are alcohol recording & eGFR —
           these are GP-engagement proxies, not cancer signals.
           Comorbid patients get LOWER scores because their
           illness generates many recordings that look "normal".
  
  Fix:  
    1. Remove pure recording-count confounders
    2. Normalize clinical features by GP engagement
    3. Add comorbidity-ADJUSTED cancer signals
    4. Retrain
    
  Run AFTER feature engineering, BEFORE model training.
  Input:  cleanupfeatures/bladder_feature_matrix_v2_cleaned.csv
  Output: deconfoundedresults/bladder_feature_matrix_v2_deconfounded.csv
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANUP_DIR = os.path.join(SCRIPT_DIR, 'cleanupfeatures')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'deconfoundedresults')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(CLEANUP_DIR, 'bladder_feature_matrix_v2_cleaned.csv')
if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Input not found: {INPUT_FILE}. Run 5_feature_cleanup.py first.")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'bladder_feature_matrix_v2_deconfounded.csv')

print("=" * 70)
print("FEATURE DECONFOUNDING v5")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Input: {df.shape}")
print(f"Cancer: {(df['LABEL']==1).sum():,}  Non-cancer: {(df['LABEL']==0).sum():,}")

y = df['LABEL']
pid = df['PATIENT_GUID']
fm = df.drop(columns=['LABEL', 'PATIENT_GUID'])
feature_names = fm.columns.tolist()


# ══════════════════════════════════════════════════════
# FIX 1: DROP CONFOUNDING FEATURES
# These are GP-engagement proxies, not cancer signals
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FIX 1: REMOVE CONFOUNDING FEATURES")
print("=" * 70)

confounders_to_drop = []

# 1a. Alcohol RECORDING features (not alcohol VALUE features)
# "RF_ALCOHOL_UNITS_RECORDED" means "GP asked about alcohol" = GP engagement
# Keep: alcohol VALUES (RF_ALCOHOL_MEAN_VALUE, RF_HEAVY_DRINKER_FLAG, etc.)
# Drop: alcohol RECORDING counts and per-window recording means
alcohol_recording_patterns = [
    'RF_ALCOHOL_UNITS_RECORDED',
    'RF_ALCOHOL_AUDIT_RECORDED', 
    'RF_ALCOHOL_RECORDING_COUNT',
    'RF_ALCOHOL_UNIQUE_DATES',
    'RF_ALCOHOL_UNIQUE_TERMS',
]
for col in feature_names:
    if col in alcohol_recording_patterns:
        confounders_to_drop.append(col)

# 1b. Alcohol per-window MEANS (these are the top SHAP confounders)
# These capture "GP recorded alcohol in window X" not "patient drinks more"
# Keep: RF_ALCOHOL_LAST_VALUE, RF_ALCOHOL_MAX_VALUE, RF_HEAVY_DRINKER_FLAG
for col in feature_names:
    if col.startswith('RF_ALCOHOL_W') and col.endswith('_MEAN'):
        confounders_to_drop.append(col)

# 1c. Pure VISIT COUNT features (GP engagement, not cancer)
visit_count_confounders = [
    'TEMP_GP_VISIT_DAYS',
    'TEMP_GP_VISIT_DAYS_ALL',
    'TEMP_GP_VISITS_WA',
    'TEMP_GP_VISITS_WB',
    'TEMP_GP_VISIT_ACCELERATION',
    'TEMP_EVENTS_PER_MONTH',
    'TEMP_MONTHS_WITH_EVENTS',
    'CONSULT_MEAN_EVENTS_PER_VISIT',
    'CONSULT_MAX_EVENTS_PER_VISIT',
    'CONSULT_STD_EVENTS_PER_VISIT',
]
for col in feature_names:
    if col in visit_count_confounders:
        confounders_to_drop.append(col)

# 1d. BMI RECORDING counts (not BMI values)
# "LAB_BMI_COUNT" = "GP measured BMI" = engagement
# Keep: LAB_BMI_LAST, LAB_BMI_OBESE, LAB_BMI_UNDERWEIGHT
for col in feature_names:
    if 'BMI' in col and any(p in col for p in ['_COUNT', '_WA_COUNT', '_WB_COUNT']):
        confounders_to_drop.append(col)

# 1e. Smoking RECORDING counts (per-window)
# Keep: RF_CURRENT_SMOKER_FLAG, RF_EVER_SMOKER, RF_HEAVY_SMOKER_FLAG
# Drop: RF_SMOKE_W1_COUNT etc. (= GP asked about smoking in window X)
for col in feature_names:
    if col.startswith('RF_SMOKE_W') and col.endswith('_COUNT'):
        confounders_to_drop.append(col)

# Deduplicate and filter to existing columns
confounders_to_drop = list(set(c for c in confounders_to_drop if c in fm.columns))
print(f"  Dropping {len(confounders_to_drop)} confounder features:")
for c in sorted(confounders_to_drop):
    print(f"    - {c}")

fm.drop(columns=confounders_to_drop, inplace=True)


# ══════════════════════════════════════════════════════
# FIX 2: NORMALIZE COUNTS BY GP ENGAGEMENT
# Cancer signals should be RELATIVE to how much data exists
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FIX 2: ENGAGEMENT-NORMALIZED FEATURES")
print("=" * 70)

# Total clinical events = proxy for GP engagement
total_events = fm.get('OBS_TOTAL_COUNT', pd.Series(0, index=fm.index)).fillna(0)
total_events_safe = total_events.clip(lower=1)  # avoid division by zero

# Lab test count = proxy for investigation intensity
lab_count = fm.get('LAB_TOTAL_TESTS', pd.Series(0, index=fm.index)).fillna(0)
lab_count_safe = lab_count.clip(lower=1)

norm_count = 0

# Normalize key cancer signals by total events
for col_name, norm_name in [
    ('HAEM_TOTAL_COUNT', 'NORM_HAEM_PER_EVENT'),
    ('LUTS_TOTAL_COUNT', 'NORM_LUTS_PER_EVENT'),
    ('URINE_LAB_ABNORM_COUNT', 'NORM_URINE_ABNORM_PER_EVENT'),
    ('URINE_INV_COUNT', 'NORM_URINE_INV_PER_EVENT'),
    ('CATH_PROC_COUNT', 'NORM_CATH_PER_EVENT'),
    ('IMG_COUNT', 'NORM_IMG_PER_EVENT'),
    ('URO_COUNT', 'NORM_URO_PER_EVENT'),
    ('NOBS_WEIGHT_LOSS_COUNT', 'NORM_WEIGHT_LOSS_PER_EVENT'),
    ('NOBS_FATIGUE_COUNT', 'NORM_FATIGUE_PER_EVENT'),
    ('NOBS_ANAEMIA_DX_COUNT', 'NORM_ANAEMIA_DX_PER_EVENT'),
]:
    if col_name in fm.columns:
        fm[norm_name] = fm[col_name].fillna(0) / total_events_safe
        norm_count += 1

# Normalize UTI antibiotics by total meds
total_meds = fm.get('MED_TOTAL_COUNT', pd.Series(0, index=fm.index)).fillna(0)
total_meds_safe = total_meds.clip(lower=1)

for col_name, norm_name in [
    ('MED_UTI_ANTIBIOTICS_COUNT', 'NORM_UTI_AB_PER_MED'),
    ('MED_IRON_SUPPLEMENTS_COUNT', 'NORM_IRON_PER_MED'),
    ('MED_OPIOID_ANALGESICS_COUNT', 'NORM_OPIOID_PER_MED'),
]:
    if col_name in fm.columns:
        fm[norm_name] = fm[col_name].fillna(0) / total_meds_safe
        norm_count += 1

# Normalize lab abnormalities by number of tests
for col_name, norm_name in [
    ('LAB_ANAEMIA_MILD', 'NORM_ANAEMIA_PER_TEST'),
    ('LAB_CRP_HIGH', 'NORM_CRP_HIGH_PER_TEST'),
]:
    if col_name in fm.columns:
        fm[norm_name] = fm[col_name].fillna(0)  # binary flags, keep as is
        norm_count += 1

print(f"  Added {norm_count} engagement-normalized features")


# ══════════════════════════════════════════════════════
# FIX 3: COMORBIDITY-ADJUSTED CANCER SIGNALS
# The model penalizes comorbid patients → fix it
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FIX 3: COMORBIDITY-ADJUSTED CANCER SIGNALS")
print("=" * 70)

comorb_burden = fm.get('COMORB_BURDEN_SCORE', pd.Series(0, index=fm.index)).fillna(0)
comorb_count = fm.get('COMORB_TOTAL_CONDITIONS', pd.Series(0, index=fm.index)).fillna(0)
has_comorbidity = (comorb_count > 0).astype(int)

adj_count = 0

# Cancer signal DESPITE comorbidity (these should INCREASE suspicion)
# If you have cancer signals AND comorbidities, that's MORE suspicious
for signal_col, adj_name in [
    ('HAEM_ANY_FLAG', 'ADJ_HAEM_DESPITE_COMORB'),
    ('LUTS_ANY_FLAG', 'ADJ_LUTS_DESPITE_COMORB'),
    ('SYN_BLEEDING_SCORE', 'ADJ_BLEEDING_DESPITE_COMORB'),
    ('NOBS_WEIGHT_LOSS_FLAG', 'ADJ_WEIGHT_LOSS_DESPITE_COMORB'),
    ('NOBS_FATIGUE_FLAG', 'ADJ_FATIGUE_DESPITE_COMORB'),
    ('NOBS_ANAEMIA_DX_FLAG', 'ADJ_ANAEMIA_DX_DESPITE_COMORB'),
    ('LAB_HB_DECLINING', 'ADJ_HB_DECLINING_DESPITE_COMORB'),
    ('RF_EVER_SMOKER', 'ADJ_SMOKER_DESPITE_COMORB'),
]:
    if signal_col in fm.columns:
        fm[adj_name] = fm[signal_col].fillna(0) * has_comorbidity
        adj_count += 1

# Comorbidity-adjusted acceleration
# If events are accelerating AND patient has comorbidities → suspicious
if 'TEMP_CLINICAL_ACCELERATION' in fm.columns:
    fm['ADJ_ACCELERATION_COMORB'] = fm['TEMP_CLINICAL_ACCELERATION'].fillna(0) * has_comorbidity
    fm['ADJ_ACCELERATION_NO_COMORB'] = fm['TEMP_CLINICAL_ACCELERATION'].fillna(0) * (1 - has_comorbidity)
    adj_count += 2

# "Unexplained" signals: cancer signal WITHOUT known cause
# Weight loss without diabetes → more suspicious for cancer
if 'NOBS_WEIGHT_LOSS_FLAG' in fm.columns and 'COMORB_DIABETES_FLAG' in fm.columns:
    fm['ADJ_UNEXPLAINED_WEIGHT_LOSS'] = (
        (fm['NOBS_WEIGHT_LOSS_FLAG'].fillna(0) == 1) &
        (fm['COMORB_DIABETES_FLAG'].fillna(0) == 0)
    ).astype(int)
    adj_count += 1

# Anaemia without CKD → more suspicious for bleeding/cancer
if 'NOBS_ANAEMIA_DX_FLAG' in fm.columns and 'COMORB_CKD_FLAG' in fm.columns:
    fm['ADJ_UNEXPLAINED_ANAEMIA'] = (
        (fm['NOBS_ANAEMIA_DX_FLAG'].fillna(0) == 1) &
        (fm['COMORB_CKD_FLAG'].fillna(0) == 0)
    ).astype(int)
    adj_count += 1

# HB declining without CKD
if 'LAB_HB_DECLINING' in fm.columns and 'COMORB_CKD_FLAG' in fm.columns:
    fm['ADJ_HB_DECLINING_NO_CKD'] = (
        (fm['LAB_HB_DECLINING'].fillna(0) == 1) &
        (fm['COMORB_CKD_FLAG'].fillna(0) == 0)
    ).astype(int)
    adj_count += 1

# Fatigue without heart failure or COPD → more cancer-suspicious
if 'NOBS_FATIGUE_FLAG' in fm.columns:
    hf = fm.get('COMORB_HEART_FAILURE_FLAG', pd.Series(0, index=fm.index)).fillna(0)
    copd = fm.get('COMORB_COPD_FLAG', pd.Series(0, index=fm.index)).fillna(0)
    fm['ADJ_UNEXPLAINED_FATIGUE'] = (
        (fm['NOBS_FATIGUE_FLAG'].fillna(0) == 1) &
        (hf == 0) & (copd == 0)
    ).astype(int)
    adj_count += 1

print(f"  Added {adj_count} comorbidity-adjusted features")


# ══════════════════════════════════════════════════════
# FIX 4: CANCER-SPECIFIC SIGNAL STRENGTH INDEX
# Aggregate score that ONLY counts cancer-relevant signals
# (not confounded by general illness)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FIX 4: CANCER SIGNAL STRENGTH INDEX")
print("=" * 70)

def safe_get(col, default=0):
    if col in fm.columns:
        return fm[col].fillna(default)
    return pd.Series(default, index=fm.index)

# Pure cancer signals only (not confounded by comorbidities)
fm['CANCER_SIGNAL_INDEX'] = (
    safe_get('HAEM_ANY_FLAG') * 5 +              # haematuria = strong signal
    safe_get('NOBS_WEIGHT_LOSS_FLAG') * 3 +       # unexplained weight loss
    safe_get('NOBS_NIGHT_SWEATS_FLAG') * 3 +      # B-symptom
    safe_get('NOBS_APPETITE_LOSS_FLAG') * 3 +      # cancer cachexia
    safe_get('NOBS_DVT_FLAG') * 4 +               # paraneoplastic
    safe_get('NOBS_PULMONARY_EMBOLISM_FLAG') * 4 + # paraneoplastic
    safe_get('RF_EVER_SMOKER') * 3 +              # major risk factor
    safe_get('RF_HEAVY_SMOKER_FLAG') * 2 +        # dose-response
    safe_get('COMORB_PREVIOUS_CANCER_FLAG') * 4 + # cancer history
    safe_get('LAB_HB_DECLINING') * 2 +            # chronic bleed
    safe_get('LAB_WEIGHT_LOSS_5PCT') * 3 +        # measured weight loss
    safe_get('MED_OPIOID_ESCALATION', 0) * 2 +   # pain escalation
    safe_get('MED_IRON_AND_HAEMOSTATIC', 0) * 3 + # bleeding management
    safe_get('LAB_CALCIUM_HIGH', 0) * 3 +         # paraneoplastic
    (safe_get('AGE_AT_INDEX') >= 70).astype(int) * 2 +  # age risk
    (safe_get('SEX_MALE') == 1).astype(int) * 1   # sex risk
)

# Cancer signal PER comorbidity level
# (sicker patients need HIGHER signals for same suspicion)
fm['CANCER_SIGNAL_PER_COMORB'] = fm['CANCER_SIGNAL_INDEX'] / (comorb_burden.clip(lower=1))

# Cancer signal above EXPECTED for comorbidity level
# Expected signal = mean signal for that comorbidity burden level
comorb_bins = pd.cut(comorb_burden, bins=[-1, 0, 2, 4, 100], labels=['0','1-2','3-4','5+'])
comorb_mean_signal = fm.groupby(comorb_bins)['CANCER_SIGNAL_INDEX'].transform('mean')
fm['CANCER_SIGNAL_RESIDUAL'] = fm['CANCER_SIGNAL_INDEX'] - comorb_mean_signal.fillna(0)

# Binary: signal above expected for comorbidity level
fm['CANCER_SIGNAL_ABOVE_EXPECTED'] = (fm['CANCER_SIGNAL_RESIDUAL'] > 0).astype(int)

print(f"  Added: CANCER_SIGNAL_INDEX, CANCER_SIGNAL_PER_COMORB,")
print(f"         CANCER_SIGNAL_RESIDUAL, CANCER_SIGNAL_ABOVE_EXPECTED")


# ══════════════════════════════════════════════════════
# FIX 5: AGE-ADJUSTED FEATURES
# Older patients have more of everything → normalize
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FIX 5: AGE-ADJUSTED FEATURES")
print("=" * 70)

age = safe_get('AGE_AT_INDEX').clip(lower=30)
age_decade = (age / 10).astype(int)

age_adj_count = 0
for col_name, adj_name in [
    ('OBS_TOTAL_COUNT', 'AGE_ADJ_OBS_COUNT'),
    ('MED_TOTAL_COUNT', 'AGE_ADJ_MED_COUNT'),
    ('COMORB_TOTAL_CONDITIONS', 'AGE_ADJ_COMORB_COUNT'),
    ('LAB_TOTAL_TESTS', 'AGE_ADJ_LAB_TESTS'),
]:
    if col_name in fm.columns:
        col_mean_by_age = fm.groupby(age_decade)[col_name].transform('mean')
        fm[adj_name] = fm[col_name].fillna(0) - col_mean_by_age.fillna(0)
        age_adj_count += 1

print(f"  Added {age_adj_count} age-adjusted features")


# ══════════════════════════════════════════════════════
# CLEAN AND SAVE
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

# Fill new features
new_cols = [c for c in fm.columns if c.startswith(('NORM_','ADJ_','CANCER_SIGNAL','AGE_ADJ'))]
for c in new_cols:
    fm[c] = fm[c].fillna(0)

fm.replace([np.inf, -np.inf], np.nan, inplace=True)

# Reassemble
output = pd.DataFrame({'PATIENT_GUID': pid, 'LABEL': y})
output = pd.concat([output, fm], axis=1)
output.to_csv(OUTPUT_FILE, index=False)

print(f"  Input features:  {len(feature_names)}")
print(f"  Dropped:         {len(confounders_to_drop)}")
print(f"  Added:           {len(new_cols)}")
print(f"  Output features: {fm.shape[1]}")
print(f"  Saved to: {OUTPUT_FILE}")

print(f"""
═══════════════════════════════════════════════════════════
NEXT: Update your model training script to use:
  bladder_feature_matrix_v2_deconfounded.csv

EXPECTED IMPROVEMENT:
  • Top SHAP features should shift from alcohol/eGFR
    to haematuria/smoking/cancer signals
  • FN patients (comorbid) should score HIGHER
  • AUC may stay same or dip slightly
  • BUT sensitivity for comorbid patients should IMPROVE
  • PPV should improve (fewer GP-engagement false alarms)
═══════════════════════════════════════════════════════════
""")