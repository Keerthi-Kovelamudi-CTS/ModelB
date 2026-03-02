"""
FEATURE ENGINEERING v4 FINAL COMPLETE ✅
  64,204 patients × 1674 features
  YOUR ORIGINAL v2: Levels 1-12 (all preserved)
  NEW v4 ADDITIONS: Levels 13-18
======================================================================
"""


import pandas as pd
import numpy as np
import re
from scipy import stats
from pathlib import Path
from datetime import datetime
import sys
import argparse
import hashlib
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Feature engineering.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo'], default='12mo', help='3mo, 6mo, or 12mo window (default: 12mo)')
args = parser.parse_args()
WINDOW = args.window
RAW_SUFFIX = None if WINDOW == '12mo' else ('3m' if WINDOW == '3mo' else '6m')

DATA_DIR = SCRIPT_DIR / 'data' / WINDOW
FE_RESULTS_DIR = SCRIPT_DIR / 'FE_Results' / WINDOW
FE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Log file in FE_Results (timestamped) — all print() is also written here
_log_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = FE_RESULTS_DIR / f'feature_engineering_{_log_ts}.log'

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

_log_f = open(LOG_FILE, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, _log_f)

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

if WINDOW == '12mo':
    clinical = pd.read_csv(DATA_DIR / 'FE_bladder_dropped_patients_clinical_windowed.csv')
    meds = pd.read_csv(DATA_DIR / 'FE_bladder_dropped_patients_med_windowed.csv')
else:
    clinical = pd.read_csv(DATA_DIR / f'FE_bladder_dropped_patients_clinical_windowed_{RAW_SUFFIX}.csv')
    meds = pd.read_csv(DATA_DIR / f'FE_bladder_dropped_patients_med_windowed_{RAW_SUFFIX}.csv')

clinical.columns = [c.upper() for c in clinical.columns]
meds.columns = [c.upper() for c in meds.columns]

clinical['EVENT_DATE'] = pd.to_datetime(clinical['EVENT_DATE'], errors='coerce')
clinical['INDEX_DATE'] = pd.to_datetime(clinical['INDEX_DATE'], errors='coerce')
meds['EVENT_DATE'] = pd.to_datetime(meds['EVENT_DATE'], errors='coerce')
meds['INDEX_DATE'] = pd.to_datetime(meds['INDEX_DATE'], errors='coerce')

print(f"Clinical: {len(clinical):,} rows")
print(f"Meds:     {len(meds):,} rows")

# ══════════════════════════════════════════════════════════════
# 2. DROP MEDS-ONLY PATIENTS
# ══════════════════════════════════════════════════════════════
clinical_patients = set(clinical['PATIENT_GUID'].unique())
before = len(meds)
meds = meds[meds['PATIENT_GUID'].isin(clinical_patients)]
print(f"Dropped {before - len(meds)} meds-only rows")

# ══════════════════════════════════════════════════════════════
# 3. MASTER PATIENT TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BUILDING MASTER PATIENT TABLE")
print("=" * 70)

master = clinical.groupby('PATIENT_GUID').agg(
    SEX=('SEX', 'first'),
    LABEL=('LABEL', 'first'),
    AGE_AT_INDEX=('AGE_AT_INDEX', 'first'),
    INDEX_DATE=('INDEX_DATE', 'first'),
    CANCER_ID=('CANCER_ID', 'first')
).reset_index()

print(f"Master: {len(master):,} patients")
print(f"  Cancer:     {(master['LABEL']==1).sum():,}")
print(f"  Non-cancer: {(master['LABEL']==0).sum():,}")

# ══════════════════════════════════════════════════════════════
# 3b. POST-DIAGNOSIS LEAKAGE CHECK (BUG 11): drop meds/clinical after index for cancer patients
# ══════════════════════════════════════════════════════════════
cancer_guids = set(master[master['LABEL'] == 1]['PATIENT_GUID'])
if len(cancer_guids) > 0:
    cancer_meds = meds[meds['PATIENT_GUID'].isin(cancer_guids)]
    post_index_meds = cancer_meds[cancer_meds['EVENT_DATE'] > cancer_meds['INDEX_DATE']]
    if len(post_index_meds) > 0:
        print(f"WARNING: {len(post_index_meds):,} medication records AFTER index date for cancer patients (dropping)")
        print(f"   Patients affected: {post_index_meds['PATIENT_GUID'].nunique()}")
        meds = meds[~((meds['PATIENT_GUID'].isin(cancer_guids)) & (meds['EVENT_DATE'] > meds['INDEX_DATE']))]
        print(f"   Meds remaining: {len(meds):,}")
    cancer_clinical = clinical[clinical['PATIENT_GUID'].isin(cancer_guids)]
    post_index_clin = cancer_clinical[cancer_clinical['EVENT_DATE'] > cancer_clinical['INDEX_DATE']]
    if len(post_index_clin) > 0:
        print(f"WARNING: {len(post_index_clin):,} clinical records AFTER index date for cancer patients (dropping)")
        clinical = clinical[~((clinical['PATIENT_GUID'].isin(cancer_guids)) & (clinical['EVENT_DATE'] > clinical['INDEX_DATE']))]
        print(f"   Clinical remaining: {len(clinical):,}")

# BUG 25: Drop rows with negative MONTHS_BEFORE_INDEX (would indicate event after index = leakage)
if 'MONTHS_BEFORE_INDEX' in clinical.columns:
    neg_clin = clinical['MONTHS_BEFORE_INDEX'] < 0
    if neg_clin.any():
        n_neg = neg_clin.sum()
        print(f"WARNING: {n_neg:,} clinical records with negative MONTHS_BEFORE_INDEX (dropping)")
        clinical = clinical[~neg_clin]
if len(meds) > 0 and 'MONTHS_BEFORE_INDEX' in meds.columns:
    neg_meds = meds['MONTHS_BEFORE_INDEX'] < 0
    if neg_meds.any():
        print(f"WARNING: {neg_meds.sum():,} med records with negative MONTHS_BEFORE_INDEX (dropping)")
        meds = meds[~neg_meds]

# ══════════════════════════════════════════════════════════════
# 4. SPLIT CLINICAL BY EVENT TYPE
# ══════════════════════════════════════════════════════════════
obs_data    = clinical[clinical['EVENT_TYPE'] == 'OBSERVATION'].copy()
lab_data    = clinical[clinical['EVENT_TYPE'] == 'LAB VALUE'].copy()
rf_data     = clinical[clinical['EVENT_TYPE'] == 'RISK FACTOR'].copy()
comorb_data = clinical[clinical['EVENT_TYPE'] == 'COMORBIDITY'].copy()

lab_data['VALUE'] = pd.to_numeric(lab_data['VALUE'], errors='coerce')
rf_data['VALUE'] = pd.to_numeric(rf_data['VALUE'], errors='coerce')

obs_lab = clinical[clinical['EVENT_TYPE'].isin(['OBSERVATION', 'LAB VALUE'])].copy()
haem = obs_data[obs_data['CATEGORY'] == 'HAEMATURIA'].copy()

print(f"Observations:  {len(obs_data):,}")
print(f"Lab values:    {len(lab_data):,}")
print(f"Risk factors:  {len(rf_data):,}")
print(f"Comorbidities: {len(comorb_data):,}")

# Counts for per-level summary (patients analyzed, records)
n_master = len(master)
n_obs_records, n_obs_patients = len(obs_data), obs_data['PATIENT_GUID'].nunique()
n_haem_records, n_haem_patients = len(haem), haem['PATIENT_GUID'].nunique()
n_rf_records, n_rf_patients = len(rf_data), rf_data['PATIENT_GUID'].nunique()
n_comorb_records, n_comorb_patients = len(comorb_data), comorb_data['PATIENT_GUID'].nunique()
n_lab_records, n_lab_patients = len(lab_data), lab_data['PATIENT_GUID'].nunique()
n_med_records, n_med_patients = len(meds), meds['PATIENT_GUID'].nunique()
n_obs_lab_records, n_obs_lab_patients = len(obs_lab), obs_lab['PATIENT_GUID'].nunique()

# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════
def safe_col(name):
    """BUG 17/37: Alphanumeric + underscore only; 60-char limit + hash for long names."""
    cleaned = re.sub(r'[^A-Z0-9_]', '_', name.upper())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if len(cleaned) > 60:
        suffix = hashlib.md5(name.encode()).hexdigest()[:4]
        return (cleaned[:55] + '_' + suffix) if len(cleaned) > 55 else cleaned[:60]
    return cleaned[:60] if cleaned else 'EMPTY'

all_features = {}
windows = {'W1': (12,17), 'W2': (18,23), 'W3': (24,29), 'W4': (30,36)}
MIN_PATIENTS_FOR_TERM = 50  # BUG 15/16: drop rare obs/med term columns to reduce sparse noise

# Key labs dict (original 13)
key_labs = {
    'EGFR':       ('RENAL',        'eGFR using creatinine'),
    'CREATININE': ('RENAL',        'Serum creatinine level'),
    'HB':         ('HAEMATOLOGY',  'Haemoglobin concentration'),
    'PLATELETS':  ('HAEMATOLOGY',  'Platelet count'),
    'WBC':        ('HAEMATOLOGY',  'White blood cell count'),
    'CRP':        ('INFLAMMATORY', 'C-reactive protein level'),
    'ESR':        ('INFLAMMATORY', 'Erythrocyte sedimentation rate'),
    'PSA':        ('PSA',          'Prostate specific antigen level'),
    'ALT':        ('LIVER',        'Alanine aminotransferase level'),
    'ALP':        ('LIVER',        'Alkaline phosphatase level'),
    'ALBUMIN':    ('METABOLIC',    'Serum albumin level'),
    'GLUCOSE':    ('METABOLIC',    'Plasma glucose level'),
    'HBA1C':      ('METABOLIC',    'HbA1c level'),
}

# New labs dict (v4 additions)
new_lab_terms = {
    'MCV':        ('HAEMATOLOGY',   'MCV - Mean corpuscular volume'),
    'FERRITIN':   ('HAEMATOLOGY',   'Serum ferritin level'),
    'IRON':       ('HAEMATOLOGY',   'Serum iron level'),
    'RBC':        ('HAEMATOLOGY',   'Red blood cell count'),
    'HAEMATOCRIT':('HAEMATOLOGY',   'Haematocrit'),
    # ALP2 merged into ALP (BUG 13: same test, different TERM string)
    'WEIGHT':     ('VITALS',        'Body weight'),
    'BMI':        ('VITALS',        'Body mass index'),
    'CALCIUM':    ('ELECTROLYTES',  'Serum calcium level'),
    'ADJ_CALCIUM':('ELECTROLYTES',  'Serum adjusted calcium concentration'),
}


# ══════════════════════════════════════════════════════════════
# LEVEL 1: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 1: DEMOGRAPHICS")
print("=" * 70)

demo = master[['PATIENT_GUID', 'AGE_AT_INDEX']].set_index('PATIENT_GUID').copy()
demo['SEX_MALE'] = master.set_index('PATIENT_GUID')['SEX'].map({'M': 1, 'F': 0, 'I': 0.5}).fillna(0.5)
demo['SEX_FEMALE'] = 1 - demo['SEX_MALE']
demo['AGE_BAND_UNDER50'] = (demo['AGE_AT_INDEX'] < 50).astype(int)
demo['AGE_BAND_50_59'] = ((demo['AGE_AT_INDEX'] >= 50) & (demo['AGE_AT_INDEX'] < 60)).astype(int)
demo['AGE_BAND_60_69'] = ((demo['AGE_AT_INDEX'] >= 60) & (demo['AGE_AT_INDEX'] < 70)).astype(int)
demo['AGE_BAND_70_79'] = ((demo['AGE_AT_INDEX'] >= 70) & (demo['AGE_AT_INDEX'] < 80)).astype(int)
demo['AGE_BAND_80PLUS'] = (demo['AGE_AT_INDEX'] >= 80).astype(int)
demo['AGE_SQUARED'] = demo['AGE_AT_INDEX'] ** 2

all_features['demo'] = demo
print(f"  Demographics: {demo.shape[1]} features")
print(f"  Patients: {n_master:,}  |  Records: — (demographics)")


# ══════════════════════════════════════════════════════════════
# LEVEL 2: OBSERVATION FEATURES — ALL GRANULARITIES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 2: OBSERVATION FEATURES")
print("=" * 70)

obs_feats = pd.DataFrame(index=master['PATIENT_GUID'])

# 2a. Count per CATEGORY (total)
cat_counts = obs_data.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
cat_counts.columns = ['OBS_' + safe_col(c) + '_COUNT' for c in cat_counts.columns]
obs_feats = obs_feats.join(cat_counts, how='left')

# 2b. Count per CATEGORY × WINDOW
for window in ['A', 'B']:
    w = obs_data[obs_data['TIME_WINDOW'] == window]
    wc = w.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    wc.columns = ['OBS_' + safe_col(c) + f'_W{window}' for c in wc.columns]
    obs_feats = obs_feats.join(wc, how='left')

# 2c. Count per individual TERM (BUG 15: keep only terms with >= MIN_PATIENTS_FOR_TERM)
term_counts = obs_data.groupby(['PATIENT_GUID', 'TERM']).size().unstack(fill_value=0)
term_patient_counts = (term_counts > 0).sum()
keep_terms = term_patient_counts[term_patient_counts >= MIN_PATIENTS_FOR_TERM].index
term_counts = term_counts[keep_terms] if len(keep_terms) > 0 else term_counts.iloc[:, :0]
term_counts.columns = ['OBS_TERM_' + safe_col(c) + '_COUNT' for c in term_counts.columns]
obs_feats = obs_feats.join(term_counts, how='left')
if len(term_patient_counts) > 0:
    print(f"  OBS terms: kept {len(keep_terms)}, dropped {len(term_patient_counts) - len(keep_terms)} rare")

# 2d. Count per TERM × WINDOW (BUG 16: same filter)
for window in ['A', 'B']:
    w = obs_data[obs_data['TIME_WINDOW'] == window]
    wc = w.groupby(['PATIENT_GUID', 'TERM']).size().unstack(fill_value=0)
    term_pts = (wc > 0).sum()
    keep = term_pts[term_pts >= MIN_PATIENTS_FOR_TERM].index
    wc = wc[keep] if len(keep) > 0 else wc.iloc[:, :0]
    wc.columns = ['OBS_TERM_' + safe_col(c) + f'_W{window}' for c in wc.columns]
    obs_feats = obs_feats.join(wc, how='left')

# 2e. Totals and unique counts
obs_feats['OBS_TOTAL_COUNT'] = obs_data.groupby('PATIENT_GUID').size()
obs_feats['OBS_UNIQUE_CATEGORIES'] = obs_data.groupby('PATIENT_GUID')['CATEGORY'].nunique()
obs_feats['OBS_UNIQUE_TERMS'] = obs_data.groupby('PATIENT_GUID')['TERM'].nunique()
obs_feats['OBS_UNIQUE_SNOMED'] = obs_data.groupby('PATIENT_GUID')['SNOMED_ID'].nunique()

# 2f. Window totals
for window in ['A', 'B']:
    w = obs_data[obs_data['TIME_WINDOW'] == window]
    obs_feats[f'OBS_TOTAL_W{window}'] = w.groupby('PATIENT_GUID').size()

# 2g. Binary flags per category
for cat in obs_data['CATEGORY'].unique():
    patients_with = set(obs_data[obs_data['CATEGORY'] == cat]['PATIENT_GUID'].unique())
    obs_feats['OBS_' + safe_col(cat) + '_FLAG'] = obs_feats.index.isin(patients_with).astype(int)

all_features['obs'] = obs_feats
print(f"  Observation features: {obs_feats.shape[1]} features")
print(f"  Patients: {n_obs_patients:,}  |  Records: {n_obs_records:,}")


# ══��═══════════════════════════════════════════════════════════
# LEVEL 3: HAEMATURIA DEEP FEATURES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 3: HAEMATURIA DEEP FEATURES")
print("=" * 70)

haem_feats = pd.DataFrame(index=master['PATIENT_GUID'])
if len(haem) > 0:
    haem_feats['HAEM_TOTAL_COUNT'] = haem.groupby('PATIENT_GUID').size()
    haem_feats['HAEM_UNIQUE_TERMS'] = haem.groupby('PATIENT_GUID')['TERM'].nunique()
    for window in ['A', 'B']:
        w = haem[haem['TIME_WINDOW'] == window]
        haem_feats[f'HAEM_W{window}_COUNT'] = w.groupby('PATIENT_GUID').size()
    for term_pattern, col_name in [
        ('frank|Frank', 'HAEM_FRANK_COUNT'), ('painless|Painless', 'HAEM_PAINLESS_COUNT'),
        ('painful|Painful', 'HAEM_PAINFUL_COUNT'), ('microscopic|Microscopic', 'HAEM_MICROSCOPIC_COUNT'),
        ('history|History|H/O', 'HAEM_HISTORY_COUNT')]:
        # BUG 26: Exclude "family history" (personal history only)
        if col_name == 'HAEM_HISTORY_COUNT':
            matched = haem[haem['TERM'].str.contains(term_pattern, case=False, na=False) & ~haem['TERM'].str.contains('family', case=False, na=False)]
        else:
            matched = haem[haem['TERM'].str.contains(term_pattern, case=False, na=False)]
        haem_feats[col_name] = matched.groupby('PATIENT_GUID').size()
    haem_feats['HAEM_FIRST_MONTHS_BEFORE'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()
    haem_feats['HAEM_LAST_MONTHS_BEFORE'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
    haem_feats['HAEM_SPAN_MONTHS'] = haem_feats['HAEM_FIRST_MONTHS_BEFORE'] - haem_feats['HAEM_LAST_MONTHS_BEFORE']
    haem_feats['HAEM_MEAN_MONTHS_BEFORE'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].mean()
    haem_feats['HAEM_RECURRENT'] = (haem_feats['HAEM_TOTAL_COUNT'] > 1).astype(float)
    haem_feats['HAEM_FREQUENT'] = (haem_feats['HAEM_TOTAL_COUNT'] >= 3).astype(float)
    denom = haem_feats['HAEM_WA_COUNT'].fillna(1)
    haem_feats['HAEM_ACCELERATION'] = np.where(
        haem_feats['HAEM_WA_COUNT'].fillna(0) > 0,
        haem_feats['HAEM_WB_COUNT'].fillna(0) / denom,
        np.where(haem_feats['HAEM_WB_COUNT'].fillna(0) > 0, 2.0, 0.0))
    haem_feats['HAEM_UNIQUE_DATES'] = haem.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    haem_feats['HAEM_ANY_FLAG'] = haem_feats.index.isin(set(haem['PATIENT_GUID'].unique())).astype(int)
    # BUG 32: HAEM_FIRST_OCCURRENCE_MONTHS removed (duplicate of HAEM_FIRST_MONTHS_BEFORE)

all_features['haem'] = haem_feats
print(f"  Haematuria features: {haem_feats.shape[1]} features")
print(f"  Patients: {n_haem_patients:,}  |  Records: {n_haem_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 4: URINE FEATURES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 4: URINE FEATURES")
print("=" * 70)

urine_feats = pd.DataFrame(index=master['PATIENT_GUID'])
ui = obs_data[obs_data['CATEGORY'] == 'URINE INVESTIGATIONS']
urine_feats['URINE_INV_COUNT'] = ui.groupby('PATIENT_GUID').size()
urine_feats['URINE_INV_UNIQUE_TERMS'] = ui.groupby('PATIENT_GUID')['TERM'].nunique()
for window in ['A', 'B']:
    w = ui[ui['TIME_WINDOW'] == window]
    urine_feats[f'URINE_INV_W{window}'] = w.groupby('PATIENT_GUID').size()
for term_pattern, col_name in [
    ('culture|Culture', 'URINE_CULTURE_COUNT'), ('microscopy|Microscopy', 'URINE_MICROSCOPY_COUNT'),
    ('MSU|Mid-stream', 'URINE_MSU_COUNT'), ('cytolog|Cytolog', 'URINE_CYTOLOGY_COUNT'),
    ('antibacterial|Antibacterial', 'URINE_ANTIBACTERIAL_COUNT')]:
    matched = ui[ui['TERM'].str.contains(term_pattern, case=False, na=False)]
    urine_feats[col_name] = matched.groupby('PATIENT_GUID').size()

ul = obs_data[obs_data['CATEGORY'] == 'URINE LAB ABNORMALITIES']
urine_feats['URINE_LAB_ABNORM_COUNT'] = ul.groupby('PATIENT_GUID').size()
urine_feats['URINE_LAB_UNIQUE_TERMS'] = ul.groupby('PATIENT_GUID')['TERM'].nunique()
for window in ['A', 'B']:
    w = ul[ul['TIME_WINDOW'] == window]
    urine_feats[f'URINE_LAB_ABNORM_W{window}'] = w.groupby('PATIENT_GUID').size()
for term_pattern, col_name in [
    ('blood test', 'URINE_BLOOD_TEST_COUNT'), ('red', 'URINE_RED_COUNT'),
    ('protein', 'URINE_PROTEIN_COUNT'), ('dark|concentrated', 'URINE_DARK_COUNT'),
    ('leucocyte', 'URINE_LEUCOCYTE_COUNT'), ('red cells', 'URINE_RED_CELLS_COUNT'),
    ('cytolog', 'URINE_LAB_CYTOLOGY_COUNT'), ('specific gravity', 'URINE_SPECIFIC_GRAVITY_COUNT')]:
    matched = ul[ul['TERM'].str.contains(term_pattern, case=False, na=False)]
    urine_feats[col_name] = matched.groupby('PATIENT_GUID').size()

urine_feats['URINE_ANY_INV_FLAG'] = urine_feats.index.isin(set(ui['PATIENT_GUID'].unique())).astype(int)
urine_feats['URINE_ANY_ABNORM_FLAG'] = urine_feats.index.isin(set(ul['PATIENT_GUID'].unique())).astype(int)
urine_feats['URINE_TOTAL_ACTIVITY'] = urine_feats.get('URINE_INV_COUNT', 0).fillna(0) + urine_feats.get('URINE_LAB_ABNORM_COUNT', 0).fillna(0)

# BUG 19: Urine microscopy RBC/WBC numeric values (direct haematuria quantification)
_cols = ['PATIENT_GUID', 'TERM', 'VALUE', 'EVENT_DATE', 'MONTHS_BEFORE_INDEX']
urine_obs = ui[_cols].copy() if len(ui) > 0 else pd.DataFrame()
if len(ul) > 0:
    urine_obs = pd.concat([urine_obs, ul[_cols]], axis=0, ignore_index=True) if len(urine_obs) > 0 else ul[_cols].copy()
if len(urine_obs) > 0 and 'VALUE' in urine_obs.columns:
    urine_obs = urine_obs.copy()
    urine_obs['VALUE_NUM'] = pd.to_numeric(urine_obs['VALUE'], errors='coerce')
    urine_obs = urine_obs[urine_obs['VALUE_NUM'].notna()]
    if len(urine_obs) > 0:
        rbc_terms = urine_obs['TERM'].str.contains('red cell|rbc|erythrocyte|blood cell', case=False, na=False)
        wbc_terms = urine_obs['TERM'].str.contains('white cell|wbc|leucocyte|leukocyte', case=False, na=False)
        if rbc_terms.any():
            rbc_df = urine_obs.loc[rbc_terms].sort_values('MONTHS_BEFORE_INDEX')
            urine_feats['URINE_MICROSCOPY_RBC_LAST'] = rbc_df.groupby('PATIENT_GUID')['VALUE_NUM'].last()
            urine_feats['URINE_MICROSCOPY_RBC_MAX'] = urine_obs.loc[rbc_terms].groupby('PATIENT_GUID')['VALUE_NUM'].max()
        if wbc_terms.any():
            wbc_df = urine_obs.loc[wbc_terms].sort_values('MONTHS_BEFORE_INDEX')
            urine_feats['URINE_MICROSCOPY_WBC_LAST'] = wbc_df.groupby('PATIENT_GUID')['VALUE_NUM'].last()
            urine_feats['URINE_MICROSCOPY_WBC_MAX'] = urine_obs.loc[wbc_terms].groupby('PATIENT_GUID')['VALUE_NUM'].max()

# 6-month velocity for urine abnormalities
for wname, (lo, hi) in windows.items():
    w = ul[(ul['MONTHS_BEFORE_INDEX'] >= lo) & (ul['MONTHS_BEFORE_INDEX'] <= hi)]
    urine_feats[f'URINE_LAB_ABNORM_6M_VEL_{wname}_COUNT'] = w.groupby('PATIENT_GUID').size()

all_features['urine'] = urine_feats
n_urine_records = len(ui) + len(ul)
n_urine_patients = len(set(ui['PATIENT_GUID'].unique()) | set(ul['PATIENT_GUID'].unique()))
print(f"  Urine features: {urine_feats.shape[1]} features")
print(f"  Patients: {n_urine_patients:,}  |  Records: {n_urine_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 5: LUTS FEATURES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 5: LUTS FEATURES")
print("=" * 70)

luts = obs_data[obs_data['CATEGORY'] == 'LUTS'].copy()
luts_feats = pd.DataFrame(index=master['PATIENT_GUID'])
luts_feats['LUTS_TOTAL_COUNT'] = luts.groupby('PATIENT_GUID').size()
luts_feats['LUTS_UNIQUE_SYMPTOMS'] = luts.groupby('PATIENT_GUID')['TERM'].nunique()
for window in ['A', 'B']:
    w = luts[luts['TIME_WINDOW'] == window]
    luts_feats[f'LUTS_W{window}_COUNT'] = w.groupby('PATIENT_GUID').size()
for term_pattern, col_name in [
    ('bladder pain|Bladder pain', 'LUTS_BLADDER_PAIN_COUNT'),
    ('difficulty|Difficulty', 'LUTS_DIFFICULTY_COUNT'),
    ('delay|Delay|hesitancy', 'LUTS_HESITANCY_COUNT'),
    ('dribbling|Dribbling', 'LUTS_DRIBBLING_COUNT'),
    ('frequency|Frequency|polyuria', 'LUTS_FREQUENCY_COUNT'),
    ('retention|Retention', 'LUTS_RETENTION_COUNT'),
    ('urinary tract infect', 'LUTS_UTI_COUNT')]:
    matched = luts[luts['TERM'].str.contains(term_pattern, case=False, na=False)]
    luts_feats[col_name] = matched.groupby('PATIENT_GUID').size()
luts_feats['LUTS_ANY_FLAG'] = luts_feats.index.isin(set(luts['PATIENT_GUID'].unique())).astype(int)
luts_feats['LUTS_RECURRENT'] = (luts_feats.get('LUTS_TOTAL_COUNT', 0).fillna(0) > 1).astype(int)
luts_feats['LUTS_MULTIPLE_SYMPTOMS'] = (luts_feats.get('LUTS_UNIQUE_SYMPTOMS', 0).fillna(0) > 1).astype(int)

all_features['luts'] = luts_feats
n_luts_records, n_luts_patients = len(luts), luts['PATIENT_GUID'].nunique()
print(f"  LUTS features: {luts_feats.shape[1]} features")
print(f"  Patients: {n_luts_patients:,}  |  Records: {n_luts_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 6: CATHETER / IMAGING / UROLOGICAL / GYNAECOLOGICAL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 6: CATHETER / IMAGING / UROLOGICAL / GYNAECOLOGICAL")
print("=" * 70)

other_obs_feats = pd.DataFrame(index=master['PATIENT_GUID'])

# Catheter/Procedures
cath = obs_data[obs_data['CATEGORY'] == 'CATHETER/PROCEDURES']
other_obs_feats['CATH_PROC_COUNT'] = cath.groupby('PATIENT_GUID').size()
other_obs_feats['CATH_UNIQUE_PROCS'] = cath.groupby('PATIENT_GUID')['TERM'].nunique()
other_obs_feats['CATH_ANY_FLAG'] = other_obs_feats.index.isin(set(cath['PATIENT_GUID'].unique())).astype(int)
for window in ['A', 'B']:
    w = cath[cath['TIME_WINDOW'] == window]
    other_obs_feats[f'CATH_PROC_W{window}'] = w.groupby('PATIENT_GUID').size()
for term_pattern, col_name in [
    ('insertion|Insertion', 'CATH_INSERTION_COUNT'), ('in situ', 'CATH_IN_SITU_COUNT'),
    ('complication', 'CATH_COMPLICATION_COUNT'), ('removal|Removal', 'CATH_REMOVAL_COUNT'),
    ('dilatation|Dilatation', 'CATH_DILATATION_COUNT'), ('prostatitis|Prostatitis', 'CATH_PROSTATITIS_COUNT'),
    ('trial without', 'CATH_TRIAL_WITHOUT_COUNT')]:
    matched = cath[cath['TERM'].str.contains(term_pattern, case=False, na=False)]
    other_obs_feats[col_name] = matched.groupby('PATIENT_GUID').size()

# Imaging
img = obs_data[obs_data['CATEGORY'] == 'IMAGING']
other_obs_feats['IMG_COUNT'] = img.groupby('PATIENT_GUID').size()
other_obs_feats['IMG_UNIQUE_TYPES'] = img.groupby('PATIENT_GUID')['TERM'].nunique()
other_obs_feats['IMG_ANY_FLAG'] = other_obs_feats.index.isin(set(img['PATIENT_GUID'].unique())).astype(int)
for window in ['A', 'B']:
    w = img[img['TIME_WINDOW'] == window]
    other_obs_feats[f'IMG_W{window}'] = w.groupby('PATIENT_GUID').size()
for term_pattern, col_name in [
    ('abnormal', 'IMG_ABNORMAL_COUNT'), ('bladder', 'IMG_BLADDER_COUNT'),
    ('kidney|renal', 'IMG_KIDNEY_COUNT'), ('urinary tract', 'IMG_URINARY_TRACT_COUNT')]:
    matched = img[img['TERM'].str.contains(term_pattern, case=False, na=False)]
    other_obs_feats[col_name] = matched.groupby('PATIENT_GUID').size()

# Urological conditions
uro = obs_data[obs_data['CATEGORY'] == 'UROLOGICAL CONDITIONS']
other_obs_feats['URO_COUNT'] = uro.groupby('PATIENT_GUID').size()
other_obs_feats['URO_UNIQUE_CONDITIONS'] = uro.groupby('PATIENT_GUID')['TERM'].nunique()
other_obs_feats['URO_ANY_FLAG'] = other_obs_feats.index.isin(set(uro['PATIENT_GUID'].unique())).astype(int)
for window in ['A', 'B']:
    w = uro[uro['TIME_WINDOW'] == window]
    other_obs_feats[f'URO_W{window}'] = w.groupby('PATIENT_GUID').size()
for term_pattern, col_name in [
    ('bladder', 'URO_BLADDER_COUNT'), ('stone', 'URO_STONE_COUNT'),
    ('hydronephrosis|Hydronephrosis', 'URO_HYDRONEPHROSIS_COUNT'),
    ('pyuria|Pyuria', 'URO_PYURIA_COUNT'), ('atrophy', 'URO_ATROPHY_COUNT'),
    ('trabeculation', 'URO_TRABECULATION_COUNT')]:
    matched = uro[uro['TERM'].str.contains(term_pattern, case=False, na=False)]
    other_obs_feats[col_name] = matched.groupby('PATIENT_GUID').size()

# Gynaecological
gynae = obs_data[obs_data['CATEGORY'] == 'GYNAECOLOGICAL/BLEEDING']
other_obs_feats['GYNAE_COUNT'] = gynae.groupby('PATIENT_GUID').size()
other_obs_feats['GYNAE_ANY_FLAG'] = other_obs_feats.index.isin(set(gynae['PATIENT_GUID'].unique())).astype(int)
for term_pattern, col_name in [
    ('intermenstrual', 'GYNAE_INTERMENSTRUAL_COUNT'),
    ('menorrhagia|Menorrhagia', 'GYNAE_MENORRHAGIA_COUNT'),
    ('vaginal', 'GYNAE_VAGINAL_BLEEDING_COUNT')]:
    matched = gynae[gynae['TERM'].str.contains(term_pattern, case=False, na=False)]
    other_obs_feats[col_name] = matched.groupby('PATIENT_GUID').size()

# Others
others = obs_data[obs_data['CATEGORY'] == 'OTHERS']
other_obs_feats['OTHER_OBS_COUNT'] = others.groupby('PATIENT_GUID').size()
other_obs_feats['OTHER_ABDOMINAL_PELVIC_COUNT'] = others[
    others['TERM'].str.contains('abdominal|pelvic', case=False, na=False)
].groupby('PATIENT_GUID').size()
other_obs_feats['OTHER_AKI_COUNT'] = others[
    others['TERM'].str.contains('kidney injury|AKI', case=False, na=False)
].groupby('PATIENT_GUID').size()

all_features['other_obs'] = other_obs_feats
n_other_obs_records = len(cath) + len(img) + len(uro) + len(gynae) + len(others)
n_other_obs_patients = len(set(cath['PATIENT_GUID']) | set(img['PATIENT_GUID']) | set(uro['PATIENT_GUID']) | set(gynae['PATIENT_GUID']) | set(others['PATIENT_GUID']))
print(f"  Other obs features: {other_obs_feats.shape[1]} features")
print(f"  Patients: {n_other_obs_patients:,}  |  Records: {n_other_obs_records:,}")

# ══════════════════════════════════════════════════════════════
# LEVEL 7: RISK FACTORS — FULL DEPTH
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 7: RISK FACTORS")
print("=" * 70)

rf_feats = pd.DataFrame(index=master['PATIENT_GUID'])

rf_categories = rf_data['CATEGORY'].unique()
for cat in rf_categories:
    cat_d = rf_data[rf_data['CATEGORY'] == cat]
    col = 'RF_' + safe_col(cat)
    rf_feats[col + '_FLAG'] = rf_feats.index.isin(set(cat_d['PATIENT_GUID'].unique())).astype(int)
    rf_feats[col + '_COUNT'] = cat_d.groupby('PATIENT_GUID').size()
    rf_feats[col + '_UNIQUE_TERMS'] = cat_d.groupby('PATIENT_GUID')['TERM'].nunique()

rf_feats['RF_EVER_SMOKER'] = (
    (rf_feats.get('RF_CURRENT_SMOKER_FLAG', 0) == 1) |
    (rf_feats.get('RF_EX_SMOKER_FLAG', 0) == 1)
).astype(int)

heavy = rf_data[rf_data['TERM'].str.contains('heavy|Heavy|Very heavy', case=False, na=False)]
rf_feats['RF_HEAVY_SMOKER_FLAG'] = rf_feats.index.isin(set(heavy['PATIENT_GUID'].unique())).astype(int)
light = rf_data[rf_data['TERM'].str.contains('light|Light|trivial', case=False, na=False)]
rf_feats['RF_LIGHT_SMOKER_FLAG'] = rf_feats.index.isin(set(light['PATIENT_GUID'].unique())).astype(int)
passive = rf_data[rf_data['TERM'].str.contains('passive|Passive', case=False, na=False)]
rf_feats['RF_PASSIVE_SMOKING_FLAG'] = rf_feats.index.isin(set(passive['PATIENT_GUID'].unique())).astype(int)
stopped = rf_data[rf_data['TERM'].str.contains('stopped|Stopped|ceased|Recently stopped', case=False, na=False)]
rf_feats['RF_STOPPED_SMOKING_FLAG'] = rf_feats.index.isin(set(stopped['PATIENT_GUID'].unique())).astype(int)
rf_feats['RF_CESSATION_REFUSED_FLAG'] = rf_feats.get('RF_SMOKING_CESSATION_REFUSED_FLAG', 0)

smoke_all = rf_data[rf_data['CATEGORY'].isin(['CURRENT SMOKER', 'EX SMOKER', 'SMOKING CESSATION REFUSED'])]
rf_feats['RF_ALL_SMOKING_COUNT'] = smoke_all.groupby('PATIENT_GUID').size()

# Alcohol features
alc = rf_data[rf_data['CATEGORY'] == 'ALCOHOL'].copy()
alc_with_val = alc[alc['VALUE'].notna()]
if len(alc_with_val) > 0:
    rf_feats['RF_ALCOHOL_LAST_VALUE'] = alc_with_val.sort_values('MONTHS_BEFORE_INDEX').groupby('PATIENT_GUID')['VALUE'].first()
    rf_feats['RF_ALCOHOL_MEAN_VALUE'] = alc_with_val.groupby('PATIENT_GUID')['VALUE'].mean()
    rf_feats['RF_ALCOHOL_MAX_VALUE'] = alc_with_val.groupby('PATIENT_GUID')['VALUE'].max()
    rf_feats['RF_ALCOHOL_MIN_VALUE'] = alc_with_val.groupby('PATIENT_GUID')['VALUE'].min()
    alc_first = alc_with_val.sort_values('MONTHS_BEFORE_INDEX', ascending=False).groupby('PATIENT_GUID')['VALUE'].first()
    alc_last = alc_with_val.sort_values('MONTHS_BEFORE_INDEX').groupby('PATIENT_GUID')['VALUE'].first()
    rf_feats['RF_ALCOHOL_TREND'] = alc_last - alc_first
    rf_feats['RF_HEAVY_DRINKER_FLAG'] = (rf_feats.get('RF_ALCOHOL_MAX_VALUE', 0).fillna(0) > 14).astype(int)
    audit = alc[alc['TERM'].str.contains('AUDIT', case=False, na=False)]
    audit_val = audit[audit['VALUE'].notna()]
    if len(audit_val) > 0:
        rf_feats['RF_AUDIT_LAST_SCORE'] = audit_val.sort_values('MONTHS_BEFORE_INDEX').groupby('PATIENT_GUID')['VALUE'].first()
        rf_feats['RF_AUDIT_HIGH_FLAG'] = (rf_feats.get('RF_AUDIT_LAST_SCORE', 0).fillna(0) >= 8).astype(int)

# Alcohol per-window values + slope + deep features
units = alc[alc['TERM'].str.contains('units', case=False, na=False)]
rf_feats['RF_ALCOHOL_UNITS_RECORDED'] = rf_feats.index.isin(set(units['PATIENT_GUID'].unique())).astype(int)
rf_feats['RF_ALCOHOL_AUDIT_RECORDED'] = rf_feats.index.isin(set(audit['PATIENT_GUID'].unique())).astype(int) if len(audit) > 0 else 0
rf_feats['RF_ALCOHOL_UNIQUE_DATES'] = alc.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
rf_feats['RF_ALCOHOL_RECORDING_COUNT'] = alc.groupby('PATIENT_GUID').size()

for wname, (lo, hi) in windows.items():
    w = alc_with_val[(alc_with_val['MONTHS_BEFORE_INDEX'] >= lo) & (alc_with_val['MONTHS_BEFORE_INDEX'] <= hi)]
    rf_feats[f'RF_ALCOHOL_{wname}_MEAN'] = w.groupby('PATIENT_GUID')['VALUE'].mean()

if len(alc_with_val) > 0:
    multi_alc = alc_with_val.groupby('PATIENT_GUID').filter(lambda x: len(x) >= 2)
    if len(multi_alc) > 0:
        alc_slopes = {}
        for pid, group in multi_alc.groupby('PATIENT_GUID'):
            x = -group['MONTHS_BEFORE_INDEX'].values.astype(float)
            y = group['VALUE'].values.astype(float)
            if np.std(x) > 0 and np.std(y) > 0:
                slope, _, _, _, _ = stats.linregress(x, y)
                alc_slopes[pid] = slope
            elif np.std(x) > 0:
                alc_slopes[pid] = 0.0
        rf_feats['RF_ALCOHOL_VALUE_SLOPE'] = pd.Series(alc_slopes)

# Alcohol deep: non-drinker, excess, counselling, type
for term_pattern, col_name in [
    ('non.drink|Non.drink|never|Never|teetotal', 'RF_ALC_NON_DRINKER'),
    ('excess|Excess|hazard|Hazard|increas|Increas', 'RF_ALC_EXCESS_FLAG'),
    ('counsel|Counsel|advice|Advice', 'RF_ALC_COUNSELLING'),
    ('spirit|Spirit|wine|Wine|beer|Beer', 'RF_ALC_TYPE_RECORDED')]:
    matched = alc[alc['TERM'].str.contains(term_pattern, case=False, na=False)]
    rf_feats[col_name] = rf_feats.index.isin(set(matched['PATIENT_GUID'].unique())).astype(int)

# Smoking per 6-month window
smoke_data = rf_data[rf_data['CATEGORY'].isin(['CURRENT SMOKER', 'EX SMOKER', 'SMOKING CESSATION REFUSED'])]
for wname, (lo, hi) in windows.items():
    w = smoke_data[(smoke_data['MONTHS_BEFORE_INDEX'] >= lo) & (smoke_data['MONTHS_BEFORE_INDEX'] <= hi)]
    rf_feats[f'RF_SMOKE_{wname}_COUNT'] = w.groupby('PATIENT_GUID').size()

# Recently stopped smoking
current_smoke = rf_data[rf_data['CATEGORY'] == 'CURRENT SMOKER']
ex_smoke = rf_data[rf_data['CATEGORY'] == 'EX SMOKER']
early_current = current_smoke[current_smoke['MONTHS_BEFORE_INDEX'] >= 24]
late_ex = ex_smoke[ex_smoke['MONTHS_BEFORE_INDEX'] < 24]
stopped_recently = set(early_current['PATIENT_GUID'].unique()) & set(late_ex['PATIENT_GUID'].unique())
rf_feats['RF_STOPPED_RECENTLY'] = rf_feats.index.isin(stopped_recently).astype(int)

all_features['rf'] = rf_feats
print(f"  Risk factor features: {rf_feats.shape[1]} features")
print(f"  Patients: {n_rf_patients:,}  |  Records: {n_rf_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 8: COMORBIDITIES — FULL DEPTH
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 8: COMORBIDITIES")
print("=" * 70)

comorb_feats = pd.DataFrame(index=master['PATIENT_GUID'])
comorb_categories = ['DIABETES','CKD','ANAEMIA','HYPERTENSION','OBESITY','COPD',
                     'HEART FAILURE','ATRIAL FIBRILLATION','RECURRENT UTI','BPH','PREVIOUS CANCER']
for cat in comorb_categories:
    cat_d = comorb_data[comorb_data['CATEGORY'] == cat]
    col = 'COMORB_' + safe_col(cat)
    patients_with = set(cat_d['PATIENT_GUID'].unique())
    comorb_feats[col + '_FLAG'] = comorb_feats.index.isin(patients_with).astype(int)
    comorb_feats[col + '_COUNT'] = cat_d.groupby('PATIENT_GUID').size()
    comorb_feats[col + '_FIRST_MONTHS'] = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()
    comorb_feats[col + '_LAST_MONTHS'] = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()

# CKD stages (BUG 28: word boundaries to avoid stage 30 / 3AB etc.)
for stage, pattern in [('3A', r'\bstage 3A\b|3A\b'), ('3B', r'\bstage 3B\b|3B\b'), ('3', r'\bstage 3\b(?!\s*[AB])'), ('4', r'\bstage 4\b'), ('5', r'\bstage 5\b')]:
    matched = comorb_data[(comorb_data['CATEGORY']=='CKD') & (comorb_data['TERM'].str.contains(pattern, case=False, na=False, regex=True))]
    comorb_feats[f'COMORB_CKD_STAGE{stage}_FLAG'] = comorb_feats.index.isin(set(matched['PATIENT_GUID'].unique())).astype(int)
comorb_feats['COMORB_CKD_SEVERITY'] = (
    comorb_feats.get('COMORB_CKD_STAGE3A_FLAG',0).fillna(0)*1 + comorb_feats.get('COMORB_CKD_STAGE3B_FLAG',0).fillna(0)*2 +
    comorb_feats.get('COMORB_CKD_STAGE4_FLAG',0).fillna(0)*3 + comorb_feats.get('COMORB_CKD_STAGE5_FLAG',0).fillna(0)*4)

# Diabetes type
for dtype, pattern in [('T1','type 1|Type 1'),('T2','type 2|Type 2')]:
    matched = comorb_data[(comorb_data['CATEGORY']=='DIABETES') & (comorb_data['TERM'].str.contains(pattern, case=False, na=False))]
    comorb_feats[f'COMORB_DIABETES_{dtype}_FLAG'] = comorb_feats.index.isin(set(matched['PATIENT_GUID'].unique())).astype(int)

comorb_feats['COMORB_TOTAL_CONDITIONS'] = comorb_data.groupby('PATIENT_GUID')['CATEGORY'].nunique()
comorb_feats['COMORB_TOTAL_EVENTS'] = comorb_data.groupby('PATIENT_GUID').size()
comorb_feats['COMORB_BURDEN_SCORE'] = (
    comorb_feats.get('COMORB_DIABETES_FLAG',0).fillna(0)*2 + comorb_feats.get('COMORB_CKD_FLAG',0).fillna(0)*2 +
    comorb_feats.get('COMORB_ANAEMIA_FLAG',0).fillna(0)*1 + comorb_feats.get('COMORB_HYPERTENSION_FLAG',0).fillna(0)*1 +
    comorb_feats.get('COMORB_OBESITY_FLAG',0).fillna(0)*1 + comorb_feats.get('COMORB_COPD_FLAG',0).fillna(0)*2 +
    comorb_feats.get('COMORB_HEART_FAILURE_FLAG',0).fillna(0)*3 + comorb_feats.get('COMORB_ATRIAL_FIBRILLATION_FLAG',0).fillna(0)*2 +
    comorb_feats.get('COMORB_RECURRENT_UTI_FLAG',0).fillna(0)*3 + comorb_feats.get('COMORB_BPH_FLAG',0).fillna(0)*2 +
    comorb_feats.get('COMORB_PREVIOUS_CANCER_FLAG',0).fillna(0)*3)
comorb_feats['COMORB_MULTIMORBID'] = (comorb_feats.get('COMORB_TOTAL_CONDITIONS',0).fillna(0) >= 3).astype(int)

# Comorbidity NEW in late window
for cat in ['DIABETES','CKD','ANAEMIA','HYPERTENSION','RECURRENT UTI','BPH']:
    cat_d = comorb_data[comorb_data['CATEGORY'] == cat]
    early = cat_d[cat_d['MONTHS_BEFORE_INDEX'] >= 24]
    late = cat_d[cat_d['MONTHS_BEFORE_INDEX'] < 24]
    new_in_late = set(late['PATIENT_GUID'].unique()) - set(early['PATIENT_GUID'].unique())
    comorb_feats['COMORB_' + cat.replace(' ','_') + '_NEW_LATE'] = comorb_feats.index.isin(new_in_late).astype(int)
new_late_cols = [c for c in comorb_feats.columns if c.endswith('_NEW_LATE')]
comorb_feats['COMORB_NEW_LATE_TOTAL'] = comorb_feats[new_late_cols].sum(axis=1) if new_late_cols else 0

all_features['comorb'] = comorb_feats
print(f"  Comorbidity features: {comorb_feats.shape[1]} features")
print(f"  Patients: {n_comorb_patients:,}  |  Records: {n_comorb_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 9: LAB VALUES — ORIGINAL 13 + NEW 10
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 9: LAB VALUES")
print("=" * 70)

lab_feats = pd.DataFrame(index=master['PATIENT_GUID'])

# Combine original + new labs
all_lab_terms = {**key_labs, **new_lab_terms}

# BUG 12: Include OBSERVATION rows that match lab (CATEGORY, TERM) so Body weight, BMI, etc. are in lab_data
lab_value_terms = set(all_lab_terms.values())
obs_for_lab = clinical[(clinical['EVENT_TYPE'] == 'OBSERVATION')].copy()
obs_for_lab['_ct'] = list(zip(obs_for_lab['CATEGORY'], obs_for_lab['TERM']))
obs_for_lab = obs_for_lab[obs_for_lab['_ct'].isin(lab_value_terms)].drop(columns=['_ct'])
obs_for_lab['VALUE'] = pd.to_numeric(obs_for_lab['VALUE'], errors='coerce')
obs_for_lab = obs_for_lab[obs_for_lab['VALUE'].notna()]
if len(obs_for_lab) > 0:
    lab_data = pd.concat([lab_data, obs_for_lab], ignore_index=True)
    print(f"  Lab data extended with {len(obs_for_lab):,} OBSERVATION rows matching lab terms (BUG 12 fix)")

for short, (category, term) in all_lab_terms.items():
    # BUG 13: ALP merges both Alkaline phosphatase and Serum alkaline phosphatase terms
    if short == 'ALP':
        t_data = lab_data[(lab_data['CATEGORY'] == 'LIVER') & (lab_data['TERM'].isin(['Alkaline phosphatase level', 'Serum alkaline phosphatase level'])) & (lab_data['VALUE'].notna())]
    else:
        t_data = lab_data[(lab_data['CATEGORY']==category) & (lab_data['TERM']==term) & (lab_data['VALUE'].notna())]
    if len(t_data) == 0:
        continue
    prefix = f'LAB_{short}'
    
    lab_feats[f'{prefix}_LAST'] = t_data.sort_values('MONTHS_BEFORE_INDEX').groupby('PATIENT_GUID')['VALUE'].first()
    lab_feats[f'{prefix}_FIRST'] = t_data.sort_values('MONTHS_BEFORE_INDEX', ascending=False).groupby('PATIENT_GUID')['VALUE'].first()
    lab_feats[f'{prefix}_MEAN'] = t_data.groupby('PATIENT_GUID')['VALUE'].mean()
    lab_feats[f'{prefix}_MEDIAN'] = t_data.groupby('PATIENT_GUID')['VALUE'].median()
    lab_feats[f'{prefix}_MIN'] = t_data.groupby('PATIENT_GUID')['VALUE'].min()
    lab_feats[f'{prefix}_MAX'] = t_data.groupby('PATIENT_GUID')['VALUE'].max()
    lab_feats[f'{prefix}_STD'] = t_data.groupby('PATIENT_GUID')['VALUE'].std()
    lab_feats[f'{prefix}_RANGE'] = lab_feats[f'{prefix}_MAX'] - lab_feats[f'{prefix}_MIN']
    lab_feats[f'{prefix}_COUNT'] = t_data.groupby('PATIENT_GUID').size()
    lab_feats[f'{prefix}_TEST_FLAG'] = lab_feats.index.isin(set(t_data['PATIENT_GUID'].unique())).astype(int)
    lab_feats[f'{prefix}_TREND'] = lab_feats[f'{prefix}_LAST'] - lab_feats[f'{prefix}_FIRST']
    lab_feats[f'{prefix}_PCT_CHANGE'] = np.where(
        lab_feats[f'{prefix}_FIRST'].notna() & (lab_feats[f'{prefix}_FIRST'] != 0),
        (lab_feats[f'{prefix}_LAST'] - lab_feats[f'{prefix}_FIRST']) / lab_feats[f'{prefix}_FIRST'].abs() * 100, np.nan)
    
    for window in ['A','B']:
        w_data = t_data[t_data['TIME_WINDOW']==window]
        if len(w_data) > 0:
            lab_feats[f'{prefix}_W{window}_MEAN'] = w_data.groupby('PATIENT_GUID')['VALUE'].mean()
            lab_feats[f'{prefix}_W{window}_COUNT'] = w_data.groupby('PATIENT_GUID').size()
    
    # 6-month window means
    for wname, (lo, hi) in windows.items():
        w = t_data[(t_data['MONTHS_BEFORE_INDEX']>=lo) & (t_data['MONTHS_BEFORE_INDEX']<=hi)]
        if len(w) > 0:
            lab_feats[f'{prefix}_{wname}_MEAN'] = w.groupby('PATIENT_GUID')['VALUE'].mean()
    
    # Slope
    multi = t_data.groupby('PATIENT_GUID').filter(lambda x: len(x) >= 2)
    if len(multi) > 0:
        slopes, r_vals = {}, {}
        for pid, group in multi.groupby('PATIENT_GUID'):
            x = -group['MONTHS_BEFORE_INDEX'].values.astype(float)
            y = group['VALUE'].values.astype(float)
            if np.std(x) > 0 and np.std(y) > 0:
                slope, _, r_val, _, _ = stats.linregress(x, y)
                slopes[pid] = slope
                r_vals[pid] = r_val
            elif np.std(x) > 0:
                slopes[pid] = 0.0
                r_vals[pid] = 0.0
        lab_feats[f'{prefix}_SLOPE'] = pd.Series(slopes)
        lab_feats[f'{prefix}_R_VALUE'] = pd.Series(r_vals)
    
    # Volatility (CV, IQR, max step)
    multi3 = t_data.groupby('PATIENT_GUID').filter(lambda x: len(x) >= 3)
    if len(multi3) > 0:
        lab_feats[f'{prefix}_CV'] = multi3.groupby('PATIENT_GUID')['VALUE'].agg(
            lambda x: x.std()/x.mean() if x.mean()!=0 else 0)
        lab_feats[f'{prefix}_IQR'] = multi3.groupby('PATIENT_GUID')['VALUE'].agg(
            lambda x: x.quantile(0.75)-x.quantile(0.25))
        def max_step(group):
            # BUG 40: sort by EVENT_DATE for precise chronological order
            vals = group.sort_values('EVENT_DATE')['VALUE'].values
            return np.max(np.abs(np.diff(vals))) if len(vals) >= 2 else 0.0
        step_ser = multi3.groupby('PATIENT_GUID').apply(max_step)
        # BUG 41: reindex so result aligns with lab_feats (handles single-group edge case)
        lab_feats[f'{prefix}_MAX_STEP'] = step_ser.reindex(lab_feats.index).values

# Clinical thresholds — original labs
if 'LAB_EGFR_TREND' in lab_feats.columns:
    lab_feats['LAB_EGFR_DECLINING_5'] = (lab_feats['LAB_EGFR_TREND']<-5).astype(float)
    lab_feats['LAB_EGFR_DECLINING_10'] = (lab_feats['LAB_EGFR_TREND']<-10).astype(float)
    lab_feats['LAB_EGFR_LOW'] = (lab_feats['LAB_EGFR_LAST']<60).astype(float)
    lab_feats['LAB_EGFR_VERY_LOW'] = (lab_feats['LAB_EGFR_LAST']<30).astype(float)
    lab_feats['LAB_EGFR_EVER_BELOW_45'] = (lab_feats['LAB_EGFR_MIN']<45).astype(float)
    lab_feats['LAB_EGFR_EVER_BELOW_30'] = (lab_feats['LAB_EGFR_MIN']<30).astype(float)
    lab_feats['LAB_EGFR_SLOPE_RAPIDLY_DECLINING'] = (lab_feats.get('LAB_EGFR_SLOPE',0).fillna(0)<-1).astype(int)
    first_e = lab_feats.get('LAB_EGFR_FIRST',0).fillna(0)
    last_e = lab_feats.get('LAB_EGFR_LAST',0).fillna(0)
    lab_feats['LAB_EGFR_DROP_GT_20PCT'] = np.where(first_e>0, ((first_e-last_e)/first_e*100)>20, False).astype(float)
    lab_feats['LAB_EGFR_DROP_GT_10PCT'] = np.where(first_e>0, ((first_e-last_e)/first_e*100)>10, False).astype(float)

if 'LAB_CREATININE_LAST' in lab_feats.columns:
    lab_feats['LAB_CREATININE_HIGH'] = (lab_feats['LAB_CREATININE_LAST']>120).astype(float)
    lab_feats['LAB_CREATININE_RISING'] = (lab_feats['LAB_CREATININE_TREND']>10).astype(float)

if 'LAB_HB_LAST' in lab_feats.columns:
    lab_feats['LAB_ANAEMIA_MILD'] = (lab_feats['LAB_HB_LAST']<130).astype(float)
    lab_feats['LAB_ANAEMIA_MODERATE'] = (lab_feats['LAB_HB_LAST']<110).astype(float)
    lab_feats['LAB_ANAEMIA_SEVERE'] = (lab_feats['LAB_HB_LAST']<80).astype(float)
    lab_feats['LAB_HB_DECLINING'] = (lab_feats['LAB_HB_TREND']<-10).astype(float)
    lab_feats['LAB_HB_EVER_BELOW_100'] = (lab_feats['LAB_HB_MIN']<100).astype(float)
    lab_feats['LAB_HB_EVER_BELOW_80'] = (lab_feats['LAB_HB_MIN']<80).astype(float)

if 'LAB_CRP_LAST' in lab_feats.columns:
    lab_feats['LAB_CRP_ELEVATED'] = (lab_feats['LAB_CRP_LAST']>5).astype(float)
    lab_feats['LAB_CRP_HIGH'] = (lab_feats['LAB_CRP_LAST']>10).astype(float)
    lab_feats['LAB_CRP_VERY_HIGH'] = (lab_feats['LAB_CRP_LAST']>50).astype(float)
    lab_feats['LAB_CRP_EVER_ABOVE_50'] = (lab_feats['LAB_CRP_MAX']>50).astype(float)
    lab_feats['LAB_CRP_EVER_ABOVE_100'] = (lab_feats['LAB_CRP_MAX']>100).astype(float)

if 'LAB_ESR_LAST' in lab_feats.columns:
    lab_feats['LAB_ESR_ELEVATED'] = (lab_feats['LAB_ESR_LAST']>20).astype(float)
    lab_feats['LAB_ESR_HIGH'] = (lab_feats['LAB_ESR_LAST']>40).astype(float)

if 'LAB_PSA_LAST' in lab_feats.columns:
    lab_feats['LAB_PSA_ELEVATED'] = (lab_feats['LAB_PSA_LAST']>4).astype(float)
    lab_feats['LAB_PSA_HIGH'] = (lab_feats['LAB_PSA_LAST']>10).astype(float)
    lab_feats['LAB_PSA_RISING'] = (lab_feats['LAB_PSA_TREND']>1).astype(float)

if 'LAB_ALBUMIN_LAST' in lab_feats.columns:
    lab_feats['LAB_ALBUMIN_LOW'] = (lab_feats['LAB_ALBUMIN_LAST']<35).astype(float)
    lab_feats['LAB_ALBUMIN_VERY_LOW'] = (lab_feats['LAB_ALBUMIN_LAST']<30).astype(float)

if 'LAB_ALP_LAST' in lab_feats.columns:
    lab_feats['LAB_ALP_ELEVATED'] = (lab_feats['LAB_ALP_LAST']>130).astype(float)

if 'LAB_WBC_LAST' in lab_feats.columns:
    lab_feats['LAB_WBC_HIGH'] = (lab_feats['LAB_WBC_LAST']>11).astype(float)
    lab_feats['LAB_WBC_LOW'] = (lab_feats['LAB_WBC_LAST']<4).astype(float)

if 'LAB_PLATELETS_LAST' in lab_feats.columns:
    lab_feats['LAB_PLATELETS_HIGH'] = (lab_feats['LAB_PLATELETS_LAST']>400).astype(float)
    lab_feats['LAB_PLATELETS_LOW'] = (lab_feats['LAB_PLATELETS_LAST']<150).astype(float)

if 'LAB_HBA1C_LAST' in lab_feats.columns:
    lab_feats['LAB_HBA1C_PREDIABETIC'] = (lab_feats['LAB_HBA1C_LAST']>=42).astype(float)
    lab_feats['LAB_HBA1C_DIABETIC'] = (lab_feats['LAB_HBA1C_LAST']>=48).astype(float)
    lab_feats['LAB_HBA1C_POOR_CONTROL'] = (lab_feats['LAB_HBA1C_LAST']>=64).astype(float)

# NEW lab thresholds
if 'LAB_MCV_MIN' in lab_feats.columns:
    lab_feats['LAB_MCV_LOW'] = (lab_feats['LAB_MCV_MIN']<80).astype(float)
    lab_feats['LAB_MCV_HIGH'] = (lab_feats['LAB_MCV_MAX']>100).astype(float)
if 'LAB_FERRITIN_MIN' in lab_feats.columns:
    lab_feats['LAB_FERRITIN_LOW'] = (lab_feats['LAB_FERRITIN_MIN']<30).astype(float)
    lab_feats['LAB_FERRITIN_VERY_LOW'] = (lab_feats['LAB_FERRITIN_MIN']<15).astype(float)
if 'LAB_IRON_MIN' in lab_feats.columns:
    lab_feats['LAB_IRON_LOW'] = (lab_feats['LAB_IRON_MIN']<10).astype(float)
if 'LAB_RBC_MIN' in lab_feats.columns:
    lab_feats['LAB_RBC_LOW'] = (lab_feats['LAB_RBC_MIN']<4.0).astype(float)
if 'LAB_HAEMATOCRIT_MIN' in lab_feats.columns:
    lab_feats['LAB_HAEMATOCRIT_LOW'] = (lab_feats['LAB_HAEMATOCRIT_MIN']<0.36).astype(float)
if 'LAB_CALCIUM_MAX' in lab_feats.columns:
    lab_feats['LAB_CALCIUM_HIGH'] = (lab_feats['LAB_CALCIUM_MAX']>2.6).astype(float)
    lab_feats['LAB_CALCIUM_VERY_HIGH'] = (lab_feats['LAB_CALCIUM_MAX']>2.8).astype(float)
if 'LAB_WEIGHT_FIRST' in lab_feats.columns:
    lab_feats['LAB_WEIGHT_CHANGE_KG'] = (
        lab_feats.get('LAB_WEIGHT_LAST', pd.Series(0, index=lab_feats.index)).fillna(0) -
        lab_feats.get('LAB_WEIGHT_FIRST', pd.Series(0, index=lab_feats.index)).fillna(0)
    )
    lab_feats['LAB_WEIGHT_CHANGE_PCT'] = np.where(
        lab_feats.get('LAB_WEIGHT_FIRST',0).fillna(0) > 30,
        (lab_feats['LAB_WEIGHT_CHANGE_KG'] / lab_feats['LAB_WEIGHT_FIRST']) * 100, 0)
    lab_feats['LAB_WEIGHT_LOSS_5PCT'] = (lab_feats['LAB_WEIGHT_CHANGE_PCT'] < -5).astype(float)
    lab_feats['LAB_WEIGHT_LOSS_10PCT'] = (lab_feats['LAB_WEIGHT_CHANGE_PCT'] < -10).astype(float)
if 'LAB_BMI_LAST' in lab_feats.columns:
    lab_feats['LAB_BMI_OBESE'] = (lab_feats['LAB_BMI_LAST']>=30).astype(float)
    lab_feats['LAB_BMI_UNDERWEIGHT'] = (lab_feats['LAB_BMI_LAST']<18.5).astype(float)

# Iron deficiency patterns
lab_feats['LAB_IRON_DEFICIENCY_PATTERN'] = 0
if 'LAB_FERRITIN_LOW' in lab_feats.columns and 'LAB_IRON_LOW' in lab_feats.columns:
    lab_feats['LAB_IRON_DEFICIENCY_PATTERN'] = (
        (lab_feats['LAB_FERRITIN_LOW']==1) & (lab_feats['LAB_IRON_LOW']==1)).astype(int)
lab_feats['LAB_MICROCYTIC_ANAEMIA_PATTERN'] = 0
if 'LAB_MCV_LOW' in lab_feats.columns and 'LAB_FERRITIN_LOW' in lab_feats.columns:
    lab_feats['LAB_MICROCYTIC_ANAEMIA_PATTERN'] = (
        (lab_feats['LAB_MCV_LOW']==1) & (lab_feats['LAB_FERRITIN_LOW']==1)).astype(int)

# Lab monitoring intensity
lab_data_valid = lab_data[lab_data['VALUE'].notna()]
lab_feats['LAB_TOTAL_TESTS'] = lab_data_valid.groupby('PATIENT_GUID').size()
lab_feats['LAB_UNIQUE_TEST_TYPES'] = lab_data_valid.groupby('PATIENT_GUID')['TERM'].nunique()
lab_feats['LAB_UNIQUE_DATES'] = lab_data_valid.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()

all_features['lab'] = lab_feats
print(f"  Lab features: {lab_feats.shape[1]} features")
print(f"  Patients: {n_lab_patients:,}  |  Records: {n_lab_records:,}")


# ═════════���════════════════════════════════════════════════════
# LEVEL 10: MEDICATION FEATURES — FULL DEPTH
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 10: MEDICATION FEATURES")
print("=" * 70)

med_feats = pd.DataFrame(index=master['PATIENT_GUID'])

meds['VALUE_NUM'] = pd.to_numeric(meds['VALUE'], errors='coerce')
if 'DURATION_IN_DAYS' in meds.columns:
    meds['DURATION_NUM'] = pd.to_numeric(meds['DURATION_IN_DAYS'], errors='coerce')
else:
    meds['DURATION_NUM'] = np.nan
    print("WARNING: DURATION_IN_DAYS column not found in meds data")

# Category-level counts
med_categories = meds['CATEGORY'].unique()
for cat in med_categories:
    cat_d = meds[meds['CATEGORY'] == cat]
    col = 'MED_' + safe_col(cat)
    med_feats[col + '_COUNT'] = cat_d.groupby('PATIENT_GUID').size()
    med_feats[col + '_FLAG'] = med_feats.index.isin(set(cat_d['PATIENT_GUID'].unique())).astype(int)
    for window in ['A', 'B']:
        w = cat_d[cat_d['TIME_WINDOW'] == window]
        med_feats[col + f'_W{window}'] = w.groupby('PATIENT_GUID').size()
    med_feats[col + '_TOTAL_QTY'] = cat_d.groupby('PATIENT_GUID')['VALUE_NUM'].sum()
    med_feats[col + '_TOTAL_DURATION'] = cat_d.groupby('PATIENT_GUID')['DURATION_NUM'].sum()

# Totals
med_feats['MED_TOTAL_COUNT'] = meds.groupby('PATIENT_GUID').size()
med_feats['MED_UNIQUE_CATEGORIES'] = meds.groupby('PATIENT_GUID')['CATEGORY'].nunique()
med_feats['MED_UNIQUE_TERMS'] = meds.groupby('PATIENT_GUID')['TERM'].nunique()
if 'DMD_CODE' in meds.columns:
    med_feats['MED_UNIQUE_DMD'] = meds.groupby('PATIENT_GUID')['DMD_CODE'].nunique()
for window in ['A', 'B']:
    w = meds[meds['TIME_WINDOW'] == window]
    med_feats[f'MED_TOTAL_W{window}'] = w.groupby('PATIENT_GUID').size()
med_feats['MED_TOTAL_QUANTITY'] = meds.groupby('PATIENT_GUID')['VALUE_NUM'].sum()
med_feats['MED_MEAN_QUANTITY'] = meds.groupby('PATIENT_GUID')['VALUE_NUM'].mean()
med_feats['MED_TOTAL_DURATION'] = meds.groupby('PATIENT_GUID')['DURATION_NUM'].sum()
med_feats['MED_MEAN_DURATION'] = meds.groupby('PATIENT_GUID')['DURATION_NUM'].mean()

# UTI antibiotic deep features
uti_meds = meds[meds['CATEGORY'] == 'UTI ANTIBIOTICS']
if len(uti_meds) > 0:
    med_feats['MED_UTI_AB_COUNT'] = uti_meds.groupby('PATIENT_GUID').size()
    med_feats['MED_UTI_AB_UNIQUE_DRUGS'] = uti_meds.groupby('PATIENT_GUID')['TERM'].nunique()
    med_feats['MED_UTI_AB_TOTAL_QTY'] = uti_meds.groupby('PATIENT_GUID')['VALUE_NUM'].sum()
    med_feats['MED_UTI_AB_TOTAL_DURATION'] = uti_meds.groupby('PATIENT_GUID')['DURATION_NUM'].sum()
    med_feats['MED_UTI_AB_RECURRENT_2'] = (med_feats.get('MED_UTI_AB_COUNT',0).fillna(0)>2).astype(int)
    med_feats['MED_UTI_AB_RECURRENT_3'] = (med_feats.get('MED_UTI_AB_COUNT',0).fillna(0)>3).astype(int)
    med_feats['MED_UTI_AB_FREQUENT'] = (med_feats.get('MED_UTI_AB_COUNT',0).fillna(0)>=5).astype(int)
    for window in ['A','B']:
        w = uti_meds[uti_meds['TIME_WINDOW']==window]
        med_feats[f'MED_UTI_AB_W{window}'] = w.groupby('PATIENT_GUID').size()
    med_feats['MED_UTI_AB_ACCELERATION'] = np.where(
        med_feats.get('MED_UTI_AB_WA',0).fillna(0) > 0,
        med_feats.get('MED_UTI_AB_WB',0).fillna(0) / med_feats.get('MED_UTI_AB_WA',0).fillna(1),
        np.where(med_feats.get('MED_UTI_AB_WB',0).fillna(0) > 0, 2.0, 0.0))
    med_feats['MED_UTI_AB_UNIQUE_DATES'] = uti_meds.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    # UTI ab span
    uti_span = uti_meds.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min','max'])
    med_feats['MED_UTI_AB_SPAN_OCCURRENCE_MONTHS'] = uti_span['max'] - uti_span['min']

# Catheter supply composite
cath_meds = meds[meds['CATEGORY'].str.contains('CATHETER', case=False, na=False)]
med_feats['MED_ANY_CATHETER_FLAG'] = med_feats.index.isin(set(cath_meds['PATIENT_GUID'].unique())).astype(int)
med_feats['MED_CATHETER_TOTAL_COUNT'] = cath_meds.groupby('PATIENT_GUID').size()
med_feats['MED_CATHETER_UNIQUE_TYPES'] = cath_meds.groupby('PATIENT_GUID')['CATEGORY'].nunique()

# BUG 36: Deduplicate by (PATIENT_GUID, drug_key) so DMD-coded + pattern match don't double-count
meds_for_terms = meds.copy()
if 'DMD_CODE' in meds_for_terms.columns:
    valid_dmd = meds_for_terms['DMD_CODE'].notna() & (meds_for_terms['DMD_CODE'].astype(str) != '-1')
    meds_for_terms['_drug_key'] = np.where(valid_dmd, meds_for_terms['DMD_CODE'].astype(str), meds_for_terms['TERM'].astype(str))
else:
    meds_for_terms['_drug_key'] = meds_for_terms['TERM'].astype(str)
meds_for_terms = meds_for_terms.drop_duplicates(subset=['PATIENT_GUID', '_drug_key'], keep='first')
# One column per drug (key); name columns by first TERM seen for that key
term_counts = meds_for_terms.groupby(['PATIENT_GUID', '_drug_key']).size().unstack(fill_value=0)
term_names = meds_for_terms.groupby('_drug_key')['TERM'].first()
term_counts.columns = [term_names.get(c, c) for c in term_counts.columns]
term_patient_counts = (term_counts > 0).sum()
keep_med_terms = term_patient_counts[term_patient_counts >= MIN_PATIENTS_FOR_TERM].index
term_counts = term_counts[keep_med_terms] if len(keep_med_terms) > 0 else term_counts.iloc[:, :0]
term_counts.columns = ['MED_TERM_' + safe_col(str(c)) + '_COUNT' for c in term_counts.columns]
med_feats = med_feats.join(term_counts, how='left')
if len(term_patient_counts) > 0:
    print(f"  MED terms: kept {len(keep_med_terms)}, dropped {len(term_patient_counts) - len(keep_med_terms)} rare")

# NEW: Opioid escalation
opioid = meds[meds['CATEGORY'] == 'OPIOID ANALGESICS']
if len(opioid) > 0:
    weak = opioid[opioid['TERM'].str.contains('codamol|codeine|dihydrocodeine|tramadol', case=False, na=False)]
    strong = opioid[opioid['TERM'].str.contains('morphine|oxycodone|fentanyl', case=False, na=False)]
    has_weak = set(weak['PATIENT_GUID'].unique())
    has_strong = set(strong['PATIENT_GUID'].unique())
    med_feats['MED_OPIOID_ESCALATION'] = (med_feats.index.isin(has_weak) & med_feats.index.isin(has_strong)).astype(int)
    med_feats['MED_STRONG_OPIOID_FLAG'] = med_feats.index.isin(has_strong).astype(int)

# NEW: UTI antibiotic escalation (first-line → cipro)
if len(uti_meds) > 0:
    cipro_pats = set(uti_meds[uti_meds['TERM'].str.contains('ciprofloxacin|ofloxacin', case=False, na=False)]['PATIENT_GUID'])
    first_line = set(uti_meds[uti_meds['TERM'].str.contains('nitrofurantoin|trimethoprim', case=False, na=False)]['PATIENT_GUID'])
    med_feats['MED_UTI_ESCALATED_TO_CIPRO'] = (med_feats.index.isin(cipro_pats) & med_feats.index.isin(first_line)).astype(int)

# NEW: Opioid + Laxative combo
has_lax = set(meds[meds['CATEGORY']=='LAXATIVES']['PATIENT_GUID'].unique())
has_opi = set(meds[meds['CATEGORY']=='OPIOID ANALGESICS']['PATIENT_GUID'].unique())
med_feats['MED_OPIOID_WITH_LAXATIVE'] = (med_feats.index.isin(has_lax) & med_feats.index.isin(has_opi)).astype(int)

# NEW: Iron + Haemostatic combo
has_iron = set(meds[meds['CATEGORY']=='IRON SUPPLEMENTS']['PATIENT_GUID'].unique())
has_haemo = set(meds[meds['CATEGORY']=='HAEMOSTATIC']['PATIENT_GUID'].unique())
med_feats['MED_IRON_AND_HAEMOSTATIC'] = (med_feats.index.isin(has_iron) & med_feats.index.isin(has_haemo)).astype(int)
all_features['med'] = med_feats
print(f"  Medication features: {med_feats.shape[1]} features")
print(f"  Patients: {n_med_patients:,}  |  Records: {n_med_records:,}")  

# ══════════════════════════════════════════════════════════════
# LEVEL 11: TEMPORAL PATTERN FEATURES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 11: TEMPORAL PATTERNS")
print("=" * 70)

temp_feats = pd.DataFrame(index=master['PATIENT_GUID'])

for window in ['A', 'B']:
    w = obs_lab[obs_lab['TIME_WINDOW'] == window]
    temp_feats[f'TEMP_CLINICAL_W{window}'] = w.groupby('PATIENT_GUID').size()
temp_feats.fillna({'TEMP_CLINICAL_WA': 0, 'TEMP_CLINICAL_WB': 0}, inplace=True)

temp_feats['TEMP_CLINICAL_ACCELERATION'] = np.where(
    temp_feats['TEMP_CLINICAL_WA'] > 0,
    temp_feats['TEMP_CLINICAL_WB'] / temp_feats['TEMP_CLINICAL_WA'],
    np.where(temp_feats['TEMP_CLINICAL_WB'] > 0, 2.0, 0.0))
temp_feats['TEMP_ACCELERATION_HIGH'] = (temp_feats['TEMP_CLINICAL_ACCELERATION'] > 1.5).astype(int)
temp_feats['TEMP_ACCELERATION_VERY_HIGH'] = (temp_feats['TEMP_CLINICAL_ACCELERATION'] > 2.0).astype(int)
temp_feats['TEMP_DECELERATION'] = (temp_feats['TEMP_CLINICAL_ACCELERATION'] < 0.5).astype(int)

# BUG 9: Check TIME_WINDOW so TEMP_ALL_WRF/WCOMORB are meaningful
tw_counts = clinical['TIME_WINDOW'].value_counts()
print("TIME_WINDOW value_counts:", tw_counts.to_dict())
if 'RF' not in tw_counts.index or 'COMORB' not in tw_counts.index:
    print("  (RF or COMORB missing — TEMP_ALL_WRF/WCOMORB and related ratios may be all NaN)")

all_events = clinical.copy()
for window in ['A', 'B', 'RF', 'COMORB']:
    w = all_events[all_events['TIME_WINDOW'] == window]
    temp_feats[f'TEMP_ALL_W{window}'] = w.groupby('PATIENT_GUID').size()

temp_feats['TEMP_GP_VISIT_DAYS'] = obs_lab.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
temp_feats['TEMP_GP_VISIT_DAYS_ALL'] = clinical.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
for window in ['A', 'B']:
    w = obs_lab[obs_lab['TIME_WINDOW'] == window]
    temp_feats[f'TEMP_GP_VISITS_W{window}'] = w.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
_denom = temp_feats['TEMP_GP_VISITS_WA'].fillna(1)
temp_feats['TEMP_GP_VISIT_ACCELERATION'] = np.where(
    temp_feats['TEMP_GP_VISITS_WA'].fillna(0) > 0,
    temp_feats['TEMP_GP_VISITS_WB'].fillna(0) / _denom,
    np.where(temp_feats['TEMP_GP_VISITS_WB'].fillna(0) > 0, 2.0, 0.0))

span = obs_lab.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min', 'max'])
temp_feats['TEMP_EVENT_SPAN_MONTHS'] = span['max'] - span['min']
temp_feats['TEMP_FIRST_EVENT_MONTHS'] = span['max']
temp_feats['TEMP_LAST_EVENT_MONTHS'] = span['min']

temp_feats['TEMP_EVENTS_PER_MONTH'] = np.where(
    temp_feats.get('TEMP_EVENT_SPAN_MONTHS', 0).fillna(0) > 0,
    (temp_feats['TEMP_CLINICAL_WA'].fillna(0) + temp_feats['TEMP_CLINICAL_WB'].fillna(0)) / temp_feats['TEMP_EVENT_SPAN_MONTHS'].fillna(1), 0)

for window in ['A', 'B']:
    w = obs_lab[obs_lab['TIME_WINDOW'] == window]
    temp_feats[f'TEMP_UNIQUE_CATS_W{window}'] = w.groupby('PATIENT_GUID')['CATEGORY'].nunique()
temp_feats['TEMP_CATEGORY_ACCELERATION'] = np.where(
    temp_feats.get('TEMP_UNIQUE_CATS_WA', 0).fillna(0) > 0,
    temp_feats.get('TEMP_UNIQUE_CATS_WB', 0).fillna(0) / temp_feats.get('TEMP_UNIQUE_CATS_WA', 0).fillna(1),
    np.where(temp_feats.get('TEMP_UNIQUE_CATS_WB', 0).fillna(0) > 0, 2.0, 0.0))

events_per_day = obs_lab.groupby(['PATIENT_GUID', 'EVENT_DATE']).size().reset_index(name='events')
temp_feats['TEMP_MAX_EVENTS_PER_DAY'] = events_per_day.groupby('PATIENT_GUID')['events'].max()
temp_feats['TEMP_MEAN_EVENTS_PER_DAY'] = events_per_day.groupby('PATIENT_GUID')['events'].mean()
temp_feats['TEMP_DAYS_WITH_MULTIPLE_EVENTS'] = events_per_day[events_per_day['events'] > 1].groupby('PATIENT_GUID').size()

obs_lab_sorted = obs_lab.sort_values(['PATIENT_GUID', 'EVENT_DATE'])
obs_lab_sorted['PREV_DATE'] = obs_lab_sorted.groupby('PATIENT_GUID')['EVENT_DATE'].shift(1)
obs_lab_sorted['GAP_DAYS'] = (obs_lab_sorted['EVENT_DATE'] - obs_lab_sorted['PREV_DATE']).dt.days
gap_stats = obs_lab_sorted.groupby('PATIENT_GUID')['GAP_DAYS'].agg(['max', 'mean', 'median'])
gap_stats.columns = ['TEMP_MAX_GAP_DAYS', 'TEMP_MEAN_GAP_DAYS', 'TEMP_MEDIAN_GAP_DAYS']
temp_feats = temp_feats.join(gap_stats, how='left')

# Med temporal
if len(meds) > 0:
    for window in ['A', 'B']:
        w = meds[meds['TIME_WINDOW'] == window]
        temp_feats[f'TEMP_MED_W{window}'] = w.groupby('PATIENT_GUID').size()
    temp_feats['TEMP_MED_ACCELERATION'] = np.where(
        temp_feats.get('TEMP_MED_WA', 0).fillna(0) > 0,
        temp_feats.get('TEMP_MED_WB', 0).fillna(0) / temp_feats.get('TEMP_MED_WA', 0).fillna(1),
        np.where(temp_feats.get('TEMP_MED_WB', 0).fillna(0) > 0, 2.0, 0.0))
    med_span = meds.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min', 'max'])
    temp_feats['TEMP_MED_SPAN_MONTHS'] = med_span['max'] - med_span['min']
    temp_feats['TEMP_MED_FIRST_MONTHS'] = med_span['max']
    temp_feats['TEMP_MED_LAST_MONTHS'] = med_span['min']

temp_feats['TEMP_COMBINED_WA'] = temp_feats.get('TEMP_CLINICAL_WA', 0).fillna(0) + temp_feats.get('TEMP_MED_WA', 0).fillna(0)
temp_feats['TEMP_COMBINED_WB'] = temp_feats.get('TEMP_CLINICAL_WB', 0).fillna(0) + temp_feats.get('TEMP_MED_WB', 0).fillna(0)
temp_feats['TEMP_COMBINED_ACCELERATION'] = np.where(
    temp_feats['TEMP_COMBINED_WA'] > 0,
    temp_feats['TEMP_COMBINED_WB'] / temp_feats['TEMP_COMBINED_WA'],
    np.where(temp_feats['TEMP_COMBINED_WB'] > 0, 2.0, 0.0))

# Burstiness
obs_lab_monthly = obs_lab.copy()
obs_lab_monthly['MONTH_BUCKET'] = obs_lab_monthly['MONTHS_BEFORE_INDEX']
monthly_counts = obs_lab_monthly.groupby(['PATIENT_GUID', 'MONTH_BUCKET']).size().reset_index(name='monthly_count')
temp_feats['TEMP_MONTHLY_EVENT_STD'] = monthly_counts.groupby('PATIENT_GUID')['monthly_count'].std()
temp_feats['TEMP_MONTHLY_EVENT_MAX'] = monthly_counts.groupby('PATIENT_GUID')['monthly_count'].max()
temp_feats['TEMP_MONTHS_WITH_EVENTS'] = monthly_counts.groupby('PATIENT_GUID').size()

# 6-month window counts
for wname, (lo, hi) in windows.items():
    w = obs_lab[(obs_lab['MONTHS_BEFORE_INDEX']>=lo)&(obs_lab['MONTHS_BEFORE_INDEX']<=hi)]
    temp_feats[f'OBS_6M_{wname}_COUNT'] = w.groupby('PATIENT_GUID').size()
for w1, w2 in [('W1','W2'),('W2','W3'),('W3','W4')]:
    c1, c2 = f'OBS_6M_{w1}_COUNT', f'OBS_6M_{w2}_COUNT'
    if c1 in temp_feats.columns and c2 in temp_feats.columns:
        temp_feats[f'OBS_6M_VEL_{w2}v{w1}_COUNT'] = temp_feats[c2].fillna(0) - temp_feats[c1].fillna(0)
if 'OBS_6M_W1_COUNT' in temp_feats.columns and 'OBS_6M_W4_COUNT' in temp_feats.columns:
    temp_feats['OBS_6M_SLOPE_COUNT'] = temp_feats['OBS_6M_W1_COUNT'].fillna(0) - temp_feats['OBS_6M_W4_COUNT'].fillna(0)

all_features['temp'] = temp_feats
print(f"  Temporal features: {temp_feats.shape[1]} features")
print(f"  Patients: {n_obs_lab_patients:,}  |  Records: {n_obs_lab_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 12: EVENT CLUSTERING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 12: EVENT CLUSTERING")
print("=" * 70)

cluster_feats = pd.DataFrame(index=master['PATIENT_GUID'])

obs_lab_cl = obs_lab.sort_values(['PATIENT_GUID','EVENT_DATE']).copy()
# BUG 38: Drop NaT so isocalendar().week.astype(int) does not raise
obs_lab_cl = obs_lab_cl[obs_lab_cl['EVENT_DATE'].notna()].copy()
obs_lab_cl['WEEK'] = obs_lab_cl['EVENT_DATE'].dt.isocalendar().week.astype(int)
obs_lab_cl['YEAR'] = obs_lab_cl['EVENT_DATE'].dt.year

weekly = obs_lab_cl.groupby(['PATIENT_GUID','YEAR','WEEK']).agg(
    events=('EVENT_DATE','count'), unique_cats=('CATEGORY','nunique')).reset_index()

cluster_feats['OBS_CLUSTER_BURST_WEEKS'] = weekly[weekly['events']>=3].groupby('PATIENT_GUID').size()
cluster_feats['OBS_CLUSTER_DIVERSE_WEEKS'] = weekly[weekly['unique_cats']>=2].groupby('PATIENT_GUID').size()
cluster_feats['OBS_CLUSTER_MAX_EVENTS_PER_WEEK'] = weekly.groupby('PATIENT_GUID')['events'].max()

# Haem with other events same week (BUG 23: merge-based instead of tuple .isin)
if len(haem) > 0:
    haem_key = haem[['PATIENT_GUID', 'EVENT_DATE']].drop_duplicates()
    haem_key['IS_HAEM'] = 1
    obs_lab_cl = obs_lab_cl.merge(haem_key, on=['PATIENT_GUID', 'EVENT_DATE'], how='left')
    obs_lab_cl['HAS_HAEM'] = obs_lab_cl['IS_HAEM'].fillna(0).astype(int)
    obs_lab_cl.drop(columns=['IS_HAEM'], inplace=True)
    haem_week = obs_lab_cl.groupby(['PATIENT_GUID','YEAR','WEEK']).agg(
        has_haem=('HAS_HAEM','max'), other_events=('EVENT_DATE','count')).reset_index()
    haem_with_other = haem_week[(haem_week['has_haem']==1)&(haem_week['other_events']>1)]
    cluster_feats['OBS_CLUSTER_HAEM_WITH_OTHER_EVENTS'] = haem_with_other.groupby('PATIENT_GUID').size()

all_features['cluster'] = cluster_feats
print(f"  Clustering features: {cluster_feats.shape[1]} features")
print(f"  Patients: {n_obs_lab_patients:,}  |  Records: {n_obs_lab_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 13: NEW OBS CATEGORIES DEEP FEATURES (v4)
# ═══════════════════════════════════════════════════════���══════
print("\n" + "=" * 70)
print("LEVEL 13: NEW OBS CATEGORIES (v4)")
print("=" * 70)

new_obs_feats = pd.DataFrame(index=master['PATIENT_GUID'])

new_obs_cats = ['WEIGHT_LOSS','FATIGUE','APPETITE_LOSS','DYSURIA','SUPRAPUBIC_PAIN',
                'BACK_PAIN','LOIN_PAIN','DVT','PULMONARY_EMBOLISM','ANAEMIA_DX','NIGHT_SWEATS','FRAILTY']

for cat in new_obs_cats:
    cat_d = obs_data[obs_data['CATEGORY']==cat]
    if len(cat_d)==0:
        continue
    col = safe_col(cat)
    new_obs_feats[f'NOBS_{col}_COUNT'] = cat_d.groupby('PATIENT_GUID').size()
    new_obs_feats[f'NOBS_{col}_FLAG'] = new_obs_feats.index.isin(set(cat_d['PATIENT_GUID'].unique())).astype(int)
    new_obs_feats[f'NOBS_{col}_UNIQUE_TERMS'] = cat_d.groupby('PATIENT_GUID')['TERM'].nunique()
    for window in ['A','B']:
        w = cat_d[cat_d['TIME_WINDOW']==window]
        new_obs_feats[f'NOBS_{col}_W{window}'] = w.groupby('PATIENT_GUID').size()
    new_obs_feats[f'NOBS_{col}_FIRST_MONTHS'] = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()
    new_obs_feats[f'NOBS_{col}_LAST_MONTHS'] = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
    new_obs_feats[f'NOBS_{col}_RECURRENT'] = (new_obs_feats.get(f'NOBS_{col}_COUNT',0).fillna(0)>1).astype(int)

# Frailty severity
frailty = obs_data[obs_data['CATEGORY']=='FRAILTY'].copy()
if len(frailty) > 0:
    for pat, cname in [('Mild','NOBS_FRAILTY_MILD'),('Moderate','NOBS_FRAILTY_MODERATE'),('Severe','NOBS_FRAILTY_SEVERE')]:
        matched = frailty[frailty['TERM'].str.contains(pat, case=False, na=False)]
        new_obs_feats[cname] = new_obs_feats.index.isin(set(matched['PATIENT_GUID'].unique())).astype(int)
    new_obs_feats['NOBS_FRAILTY_SEVERITY'] = (
        new_obs_feats.get('NOBS_FRAILTY_MILD',0).fillna(0)*1 +
        new_obs_feats.get('NOBS_FRAILTY_MODERATE',0).fillna(0)*2 +
        new_obs_feats.get('NOBS_FRAILTY_SEVERE',0).fillna(0)*3)
    # BUG 20: Use numeric frailty index from VALUE if present
    if 'VALUE' in frailty.columns:
        frailty['VALUE_NUM'] = pd.to_numeric(frailty['VALUE'], errors='coerce')
        if frailty['VALUE_NUM'].notna().any():
            _fn = frailty[frailty['VALUE_NUM'].notna()].sort_values('MONTHS_BEFORE_INDEX').groupby('PATIENT_GUID')['VALUE_NUM'].last()
            new_obs_feats['NOBS_FRAILTY_INDEX_NUMERIC'] = _fn

# Anaemia subtype
anaemia_dx = obs_data[obs_data['CATEGORY']=='ANAEMIA_DX']
if len(anaemia_dx) > 0:
    for pat, cname in [('Normocytic','NOBS_ANAEMIA_NORMOCYTIC'),('Macrocytic','NOBS_ANAEMIA_MACROCYTIC'),('Chronic','NOBS_ANAEMIA_CHRONIC')]:
        matched = anaemia_dx[anaemia_dx['TERM'].str.contains(pat, case=False, na=False)]
        new_obs_feats[cname] = new_obs_feats.index.isin(set(matched['PATIENT_GUID'].unique())).astype(int)

all_features['new_obs'] = new_obs_feats
new_obs_subset = obs_data[obs_data['CATEGORY'].isin(new_obs_cats)]
n_new_obs_records, n_new_obs_patients = len(new_obs_subset), new_obs_subset['PATIENT_GUID'].nunique()
print(f"  New obs features: {new_obs_feats.shape[1]} features")
print(f"  Patients: {n_new_obs_patients:,}  |  Records: {n_new_obs_records:,}")

# ══════════════════════════════════════════════════════════════
# LEVEL 14: TEMPORAL SEQUENCE FEATURES (v4 NEW)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 14: TEMPORAL SEQUENCES (v4 NEW)")
print("=" * 70)

seq_feats = pd.DataFrame(index=master['PATIENT_GUID'])

obs_windowed = obs_data[obs_data['TIME_WINDOW'].isin(['A','B'])].copy()

# Time between event pairs (BUG 31: first cat2 event ON OR AFTER first cat1 event per patient)
def time_between(df, cat1, cat2, name):
    c1 = df[df['CATEGORY']==cat1][['PATIENT_GUID','EVENT_DATE']].rename(columns={'EVENT_DATE':'D1'})
    c2 = df[df['CATEGORY']==cat2][['PATIENT_GUID','EVENT_DATE']].rename(columns={'EVENT_DATE':'D2'})
    merged = c1.merge(c2, on='PATIENT_GUID')
    merged = merged[merged['D2'] >= merged['D1']]
    if len(merged) == 0:
        return pd.DataFrame({name: np.nan}, index=master['PATIENT_GUID'])
    merged['GAP'] = (merged['D2'] - merged['D1']).dt.days
    result = merged.groupby('PATIENT_GUID')['GAP'].min()
    return pd.DataFrame({name: result})

for cat1, cat2, name in [
    ('HAEMATURIA','URINE INVESTIGATIONS','SEQ_HAEM_TO_URINE_INV_DAYS'),
    ('HAEMATURIA','IMAGING','SEQ_HAEM_TO_IMAGING_DAYS'),
    ('LUTS','URINE INVESTIGATIONS','SEQ_LUTS_TO_URINE_INV_DAYS'),
    ('LUTS','IMAGING','SEQ_LUTS_TO_IMAGING_DAYS'),
    ('DYSURIA','URINE INVESTIGATIONS','SEQ_DYSURIA_TO_URINE_INV_DAYS')]:
    result = time_between(obs_windowed, cat1, cat2, name)
    seq_feats = seq_feats.join(result, how='left')

# Anaemia → Iron supplement timing
if len(meds) > 0:
    an_first = obs_windowed[obs_windowed['CATEGORY']=='ANAEMIA_DX'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    ir_first = meds[meds['CATEGORY']=='IRON SUPPLEMENTS'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    an_ir = pd.DataFrame({'an':an_first,'ir':ir_first}).dropna()
    an_ir['SEQ_ANAEMIA_TO_IRON_DAYS'] = (an_ir['ir']-an_ir['an']).dt.days
    seq_feats = seq_feats.join(an_ir[['SEQ_ANAEMIA_TO_IRON_DAYS']], how='left')

# Event acceleration (near vs far)
near = obs_windowed[obs_windowed['MONTHS_BEFORE_INDEX'].between(12,18)].groupby('PATIENT_GUID').size()
far = obs_windowed[obs_windowed['MONTHS_BEFORE_INDEX'].between(30,36)].groupby('PATIENT_GUID').size()
accel = pd.DataFrame({'near':near,'far':far}).fillna(0)
accel['SEQ_ACCELERATION_RATIO'] = (accel['near']+1)/(accel['far']+1)
accel['SEQ_ACCELERATION_DIFF'] = accel['near']-accel['far']
seq_feats = seq_feats.join(accel[['SEQ_ACCELERATION_RATIO','SEQ_ACCELERATION_DIFF']], how='left')

# Recurrence in distinct months
obs_windowed_c = obs_windowed.copy()
obs_windowed_c['EVENT_MONTH'] = obs_windowed_c['EVENT_DATE'].dt.to_period('M')
for cat in ['HAEMATURIA','LUTS','DYSURIA','BACK_PAIN']:
    col = safe_col(cat)
    months_with = obs_windowed_c[obs_windowed_c['CATEGORY']==cat].groupby('PATIENT_GUID')['EVENT_MONTH'].nunique()
    seq_feats[f'SEQ_{col}_UNIQUE_MONTHS'] = months_with
    seq_feats[f'SEQ_{col}_RECURRING'] = (seq_feats[f'SEQ_{col}_UNIQUE_MONTHS'].fillna(0)>=3).astype(int)

all_features['seq'] = seq_feats
n_seq_records, n_seq_patients = len(obs_windowed), obs_windowed['PATIENT_GUID'].nunique()
print(f"  Sequence features: {seq_feats.shape[1]} features")
print(f"  Patients: {n_seq_patients:,}  |  Records: {n_seq_records:,}")


# ══════════════════════════════════════════════════════════════
# LEVEL 15: SYNDROME / COMPOSITE SCORES (v4 NEW)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 15: SYNDROME SCORES (v4 NEW)")
print("=" * 70)

syn_feats = pd.DataFrame(index=master['PATIENT_GUID'])

# Build patient sets
def pset(df, cat): return set(df[df['CATEGORY']==cat]['PATIENT_GUID'].unique())

s_haem=pset(obs_data,'HAEMATURIA'); s_wl=pset(obs_data,'WEIGHT_LOSS'); s_fat=pset(obs_data,'FATIGUE')
s_app=pset(obs_data,'APPETITE_LOSS'); s_ns=pset(obs_data,'NIGHT_SWEATS'); s_dys=pset(obs_data,'DYSURIA')
s_bp=pset(obs_data,'BACK_PAIN'); s_lp=pset(obs_data,'LOIN_PAIN'); s_sp=pset(obs_data,'SUPRAPUBIC_PAIN')
s_dvt=pset(obs_data,'DVT'); s_pe=pset(obs_data,'PULMONARY_EMBOLISM'); s_andx=pset(obs_data,'ANAEMIA_DX')
s_frail=pset(obs_data,'FRAILTY'); s_luts=pset(obs_data,'LUTS')
s_ui=pset(obs_data,'URINE INVESTIGATIONS'); s_img=pset(obs_data,'IMAGING'); s_cath=pset(obs_data,'CATHETER/PROCEDURES')
s_smoke=set(rf_data[rf_data['CATEGORY'].isin(['CURRENT SMOKER','EX SMOKER'])]['PATIENT_GUID'])
s_csm=set(rf_data[rf_data['CATEGORY']=='CURRENT SMOKER']['PATIENT_GUID'])

s_iron_rx=set(meds[meds['CATEGORY']=='IRON SUPPLEMENTS']['PATIENT_GUID']) if len(meds)>0 else set()
s_opioid=set(meds[meds['CATEGORY']=='OPIOID ANALGESICS']['PATIENT_GUID']) if len(meds)>0 else set()
s_tranex=set(meds[meds['CATEGORY']=='HAEMOSTATIC']['PATIENT_GUID']) if len(meds)>0 else set()
s_utiab=set(meds[meds['CATEGORY']=='UTI ANTIBIOTICS']['PATIENT_GUID']) if len(meds)>0 else set()

# Low HB
low_hb = set(lab_data[(lab_data['TERM'].str.contains('Haemoglobin',case=False,na=False))&(lab_data['VALUE']<100)]['PATIENT_GUID'])
low_fer = set(lab_data[(lab_data['TERM'].str.contains('ferritin',case=False,na=False))&(lab_data['VALUE']<30)]['PATIENT_GUID'])

# BUG 30: Vectorized syndrome scores (no .map(lambda))
syn_feats['SYN_BLEEDING_SCORE'] = (
    syn_feats.index.isin(s_haem).astype(int) + syn_feats.index.isin(low_hb).astype(int) +
    syn_feats.index.isin(low_fer).astype(int) + syn_feats.index.isin(s_andx).astype(int) +
    syn_feats.index.isin(s_iron_rx).astype(int) + syn_feats.index.isin(s_tranex).astype(int))
syn_feats['SYN_BLEEDING_HIGH'] = (syn_feats['SYN_BLEEDING_SCORE']>=3).astype(int)
syn_feats['SYN_CONSTITUTIONAL_SCORE'] = (
    syn_feats.index.isin(s_wl).astype(int) + syn_feats.index.isin(s_fat).astype(int) +
    syn_feats.index.isin(s_app).astype(int) + syn_feats.index.isin(s_ns).astype(int))
syn_feats['SYN_CONSTITUTIONAL_HIGH'] = (syn_feats['SYN_CONSTITUTIONAL_SCORE']>=2).astype(int)
syn_feats['SYN_PATHWAY_SCORE'] = (
    syn_feats.index.isin(s_haem).astype(int) + syn_feats.index.isin(s_ui).astype(int) +
    syn_feats.index.isin(s_img).astype(int) + syn_feats.index.isin(s_cath).astype(int) +
    syn_feats.index.isin(s_luts).astype(int))
syn_feats['SYN_PAIN_SCORE'] = (
    syn_feats.index.isin(s_bp).astype(int) + syn_feats.index.isin(s_lp).astype(int) +
    syn_feats.index.isin(s_sp).astype(int) + syn_feats.index.isin(s_opioid).astype(int))
syn_feats['SYN_VTE_SCORE'] = (
    syn_feats.index.isin(s_dvt).astype(int) + syn_feats.index.isin(s_pe).astype(int))
syn_feats['SYN_VTE_ANY'] = (syn_feats['SYN_VTE_SCORE']>0).astype(int)

# UTI treatment failure
uti_cts = meds[meds['CATEGORY']=='UTI ANTIBIOTICS'].groupby('PATIENT_GUID').size() if len(meds)>0 else pd.Series(dtype=int)
uti_3p = set(uti_cts[uti_cts>=3].index)
syn_feats['SYN_UTI_TREATMENT_FAILURE'] = (syn_feats.index.isin(uti_3p)&syn_feats.index.isin(s_haem)).astype(int)
uti_4p = set(uti_cts[uti_cts>=4].index)
syn_feats['SYN_UTI_MANY_COURSES'] = syn_feats.index.isin(uti_4p).astype(int)
uti_2p = set(uti_cts[uti_cts>=2].index)
syn_feats['SYN_UTI_RECURRENT_WITH_SYMPTOMS'] = (syn_feats.index.isin(uti_2p)&syn_feats.index.isin(s_dys|s_luts)).astype(int)

# Combinations
syn_feats['SYN_HAEM_AND_SMOKING'] = (syn_feats.index.isin(s_haem)&syn_feats.index.isin(s_smoke)).astype(int)
syn_feats['SYN_HAEM_AND_ANAEMIA'] = (syn_feats.index.isin(s_haem)&(syn_feats.index.isin(low_hb)|syn_feats.index.isin(s_andx))).astype(int)
syn_feats['SYN_HAEM_AND_WEIGHT_LOSS'] = (syn_feats.index.isin(s_haem)&syn_feats.index.isin(s_wl)).astype(int)
syn_feats['SYN_HAEM_AND_FATIGUE'] = (syn_feats.index.isin(s_haem)&syn_feats.index.isin(s_fat)).astype(int)
syn_feats['SYN_HAEM_AND_LUTS'] = (syn_feats.index.isin(s_haem)&syn_feats.index.isin(s_luts)).astype(int)
syn_feats['SYN_ANAEMIA_AND_IRON_RX'] = ((syn_feats.index.isin(s_andx)|syn_feats.index.isin(low_hb))&syn_feats.index.isin(s_iron_rx)).astype(int)
syn_feats['SYN_FATIGUE_AND_WEIGHT_LOSS'] = (syn_feats.index.isin(s_fat)&syn_feats.index.isin(s_wl)).astype(int)
syn_feats['SYN_PAIN_AND_OPIOID'] = ((syn_feats.index.isin(s_bp)|syn_feats.index.isin(s_lp))&syn_feats.index.isin(s_opioid)).astype(int)
syn_feats['SYN_FRAILTY_AND_CONSTITUTIONAL'] = (
    syn_feats.index.isin(s_frail) &
    (syn_feats.index.isin(s_wl) | syn_feats.index.isin(s_fat) |
     syn_feats.index.isin(s_app) | syn_feats.index.isin(s_ns))
).astype(int)

male_o65 = set(master[(master['SEX']=='M')&(master['AGE_AT_INDEX']>=65)]['PATIENT_GUID'])
syn_feats['SYN_HAEM_MALE_OVER65'] = (syn_feats.index.isin(s_haem)&syn_feats.index.isin(male_o65)).astype(int)
syn_feats['SYN_SMOKER_MALE_OVER65'] = (syn_feats.index.isin(s_csm)&syn_feats.index.isin(male_o65)).astype(int)

syn_feats['SYN_MASTER_SUSPICION'] = (
    syn_feats['SYN_BLEEDING_SCORE']*2 + syn_feats['SYN_CONSTITUTIONAL_SCORE'] +
    syn_feats['SYN_PATHWAY_SCORE'] + syn_feats['SYN_PAIN_SCORE'] +
    syn_feats['SYN_VTE_SCORE']*2 + syn_feats['SYN_UTI_TREATMENT_FAILURE']*2 +
    syn_feats['SYN_HAEM_AND_ANAEMIA']*3 + syn_feats['SYN_HAEM_AND_SMOKING'] + syn_feats['SYN_HAEM_MALE_OVER65'])

all_features['syn'] = syn_feats
print(f"  Syndrome features: {syn_feats.shape[1]} features")
print(f"  Patients: {n_master:,}  |  Records: — (derived)")


# ══════════════════════════════════════════════════════════════
# LEVEL 16: CLINICAL PATHWAY FEATURES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 16: CLINICAL PATHWAYS")
print("=" * 70)

path_feats = pd.DataFrame(index=master['PATIENT_GUID'])
path_feats['PATH_HAEM_THEN_URINE_INV'] = (path_feats.index.isin(s_haem)&path_feats.index.isin(s_ui)).astype(int)
path_feats['PATH_HAEM_THEN_URINE_ABNORM'] = (path_feats.index.isin(s_haem)&path_feats.index.isin(
    set(obs_data[obs_data['CATEGORY']=='URINE LAB ABNORMALITIES']['PATIENT_GUID']))).astype(int)
path_feats['PATH_UTI_AB_AND_URINE_TEST'] = (path_feats.index.isin(s_utiab)&path_feats.index.isin(s_ui)).astype(int)
s_uabn = set(obs_data[obs_data['CATEGORY']=='URINE LAB ABNORMALITIES']['PATIENT_GUID'])
path_feats['PATH_MULTI_INVESTIGATION'] = (path_feats.index.isin(s_haem)&path_feats.index.isin(s_ui)&path_feats.index.isin(s_uabn)).astype(int)
path_feats['PATH_HAEM_LUTS_URINE'] = (path_feats.index.isin(s_haem)&path_feats.index.isin(s_luts)&path_feats.index.isin(s_uabn)).astype(int)
path_feats['PATH_SMOKER_HAEM_URINE'] = (path_feats.index.isin(s_smoke)&path_feats.index.isin(s_haem)&path_feats.index.isin(s_uabn)).astype(int)
path_feats['PATH_INVESTIGATION_DEPTH'] = (
    path_feats.index.isin(s_haem).astype(int)+path_feats.index.isin(s_ui).astype(int)+
    path_feats.index.isin(s_uabn).astype(int)+path_feats.index.isin(s_luts).astype(int)+path_feats.index.isin(s_utiab).astype(int))

all_features['path'] = path_feats
print(f"  Pathway features: {path_feats.shape[1]} features")
print(f"  Patients: {n_master:,}  |  Records: — (derived)")


# ══════════════════════════════════════════════════════════════
# MERGE ALL FEATURES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MERGING ALL FEATURES")
print("=" * 70)

feature_matrix = master[['PATIENT_GUID', 'LABEL', 'CANCER_ID', 'INDEX_DATE']].set_index('PATIENT_GUID')
for name, fg in all_features.items():
    print(f"  Joining {name}: {fg.shape[1]} cols")
    feature_matrix = feature_matrix.join(fg, how='left')
print(f"\nMerged shape: {feature_matrix.shape}")

fm = feature_matrix
def get_col(name, default=0):
    if name in fm.columns: return fm[name].fillna(default)
    return pd.Series(default, index=fm.index)


# ══════════════════════════════════════════════════════════════
# LEVEL 17: INTERACTION FEATURES (POST-MERGE)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 17: INTERACTION FEATURES")
print("=" * 70)

# Haem combos
fm['INT_HAEM_AND_UTI_AB'] = (get_col('HAEM_TOTAL_COUNT')>0).astype(int)*(get_col('MED_UTI_ANTIBIOTICS_COUNT')>0).astype(int)
fm['INT_HAEM_AND_IMAGING'] = (get_col('HAEM_TOTAL_COUNT')>0).astype(int)*(get_col('IMG_COUNT')>0).astype(int)
fm['INT_HAEM_AND_LUTS'] = (get_col('HAEM_TOTAL_COUNT')>0).astype(int)*(get_col('LUTS_TOTAL_COUNT')>0).astype(int)
fm['INT_HAEM_AND_CATHETER'] = (get_col('HAEM_TOTAL_COUNT')>0).astype(int)*(get_col('CATH_PROC_COUNT')>0).astype(int)
fm['INT_HAEM_AND_URINE_ABNORM'] = (get_col('HAEM_TOTAL_COUNT')>0).astype(int)*(get_col('URINE_LAB_ABNORM_COUNT')>0).astype(int)
fm['INT_HAEM_AND_URINE_INV'] = (get_col('HAEM_TOTAL_COUNT')>0).astype(int)*(get_col('URINE_INV_COUNT')>0).astype(int)

# Smoking combos
fm['INT_SMOKER_AND_HAEM'] = (get_col('RF_EVER_SMOKER')==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_CURRENT_SMOKER_AND_HAEM'] = (get_col('RF_CURRENT_SMOKER_FLAG')==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_HEAVY_SMOKER_AND_HAEM'] = (get_col('RF_HEAVY_SMOKER_FLAG')==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_SMOKER_AND_LUTS'] = (get_col('RF_EVER_SMOKER')==1).astype(int)*(get_col('LUTS_TOTAL_COUNT')>0).astype(int)
fm['INT_SMOKER_AND_UTI_AB'] = (get_col('RF_EVER_SMOKER')==1).astype(int)*(get_col('MED_UTI_ANTIBIOTICS_COUNT')>0).astype(int)

# Age interactions
fm['INT_AGE_X_HAEM'] = get_col('AGE_AT_INDEX')*get_col('HAEM_TOTAL_COUNT')
fm['INT_AGE_X_LUTS'] = get_col('AGE_AT_INDEX')*get_col('LUTS_TOTAL_COUNT')
fm['INT_AGE_X_UTI_AB'] = get_col('AGE_AT_INDEX')*get_col('MED_UTI_ANTIBIOTICS_COUNT')
fm['INT_AGE_X_SMOKING'] = get_col('AGE_AT_INDEX')*get_col('RF_EVER_SMOKER')
fm['INT_AGE_X_COMORB_BURDEN'] = get_col('AGE_AT_INDEX')*get_col('COMORB_BURDEN_SCORE')
fm['INT_AGE_X_ACCELERATION'] = get_col('AGE_AT_INDEX')*get_col('TEMP_CLINICAL_ACCELERATION')

# Sex interactions
fm['INT_MALE_AND_HAEM'] = (get_col('SEX_MALE')==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_MALE_AND_LUTS'] = (get_col('SEX_MALE')==1).astype(int)*(get_col('LUTS_TOTAL_COUNT')>0).astype(int)
fm['INT_MALE_AND_BPH'] = (get_col('SEX_MALE')==1).astype(int)*(get_col('COMORB_BPH_FLAG')==1).astype(int)
fm['INT_MALE_LUTS_BPH'] = (get_col('SEX_MALE')==1).astype(int)*(get_col('LUTS_TOTAL_COUNT')>0).astype(int)*(get_col('COMORB_BPH_FLAG')==1).astype(int)
fm['INT_FEMALE_AND_GYNAE'] = (get_col('SEX_FEMALE')==1).astype(int)*(get_col('GYNAE_COUNT')>0).astype(int)
fm['INT_FEMALE_AND_HAEM'] = (get_col('SEX_FEMALE')==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)

# Comorbidity interactions
fm['INT_RECURRENT_UTI_AND_UTI_AB'] = (get_col('COMORB_RECURRENT_UTI_FLAG')==1).astype(int)*(get_col('MED_UTI_ANTIBIOTICS_COUNT')>0).astype(int)
fm['INT_CKD_AND_EGFR_LOW'] = (get_col('COMORB_CKD_FLAG')==1).astype(int)*(get_col('LAB_EGFR_LOW',0)==1).astype(int)
fm['INT_ANAEMIA_AND_HB_LOW'] = (get_col('COMORB_ANAEMIA_FLAG')==1).astype(int)*(get_col('LAB_ANAEMIA_MODERATE',0)==1).astype(int)
fm['INT_DIABETES_AND_HBA1C_HIGH'] = (get_col('COMORB_DIABETES_FLAG')==1).astype(int)*(get_col('LAB_HBA1C_DIABETIC',0)==1).astype(int)
fm['INT_PREV_CANCER_AND_HAEM'] = (get_col('COMORB_PREVIOUS_CANCER_FLAG')==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)

# Lab + clinical
fm['INT_CRP_HIGH_AND_HAEM'] = (get_col('LAB_CRP_HIGH',0)==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_ANAEMIA_AND_HAEM'] = (get_col('LAB_ANAEMIA_MILD',0)==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_HB_DECLINING_AND_HAEM'] = (get_col('LAB_HB_DECLINING',0)==1).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_PSA_HIGH_AND_LUTS'] = (get_col('LAB_PSA_ELEVATED',0)==1).astype(int)*(get_col('LUTS_TOTAL_COUNT')>0).astype(int)

# Acceleration combos
fm['INT_ACCEL_AND_HAEM'] = (get_col('TEMP_CLINICAL_ACCELERATION')>1.5).astype(int)*(get_col('HAEM_TOTAL_COUNT')>0).astype(int)
fm['INT_ACCEL_AND_LUTS'] = (get_col('TEMP_CLINICAL_ACCELERATION')>1.5).astype(int)*(get_col('LUTS_TOTAL_COUNT')>0).astype(int)
fm['INT_ACCEL_AND_UTI_AB'] = (get_col('TEMP_CLINICAL_ACCELERATION')>1.5).astype(int)*(get_col('MED_UTI_ANTIBIOTICS_COUNT')>0).astype(int)
fm['INT_ACCEL_AND_SMOKING'] = (get_col('TEMP_CLINICAL_ACCELERATION')>1.5).astype(int)*(get_col('RF_EVER_SMOKER')==1).astype(int)

# Composite scores
fm['INT_BLADDER_SYMPTOM_SCORE'] = (
    (get_col('HAEM_TOTAL_COUNT')>0).astype(int)*3 + (get_col('LUTS_TOTAL_COUNT')>0).astype(int)*2 +
    (get_col('URINE_LAB_ABNORM_COUNT')>0).astype(int)*2 + (get_col('URINE_INV_COUNT')>0).astype(int)*1 +
    (get_col('CATH_PROC_COUNT')>0).astype(int)*2 + (get_col('IMG_COUNT')>0).astype(int)*1 +
    (get_col('URO_COUNT')>0).astype(int)*2)
fm['INT_RISK_COMPOSITE_SCORE'] = (
    (get_col('RF_EVER_SMOKER')==1).astype(int)*3 + (get_col('RF_HEAVY_SMOKER_FLAG')==1).astype(int)*2 +
    (get_col('SEX_MALE')==1).astype(int)*1 + (get_col('AGE_AT_INDEX')>=65).astype(int)*2 +
    (get_col('COMORB_PREVIOUS_CANCER_FLAG')==1).astype(int)*3 + (get_col('COMORB_RECURRENT_UTI_FLAG')==1).astype(int)*2)
fm['INT_OVERALL_SUSPICION_SCORE'] = fm['INT_BLADDER_SYMPTOM_SCORE'] + fm['INT_RISK_COMPOSITE_SCORE']

# BUG 50: Smoking cessation meds + haematuria (confirms smoking; interaction with haem)
if 'MED_SMOKING_CESSATION_MEDS_FLAG' in fm.columns or 'MED_SMOKING_CESSATION_MEDS_COUNT' in fm.columns:
    cess = ((get_col('MED_SMOKING_CESSATION_MEDS_FLAG', 0) == 1) | (get_col('MED_SMOKING_CESSATION_MEDS_COUNT', 0) > 0))
    fm['INT_CESSATION_MEDS_AND_HAEM'] = (cess.astype(int) * (get_col('HAEM_TOTAL_COUNT') > 0).astype(int))
    fm['RF_SMOKING_FROM_MEDS'] = (cess | (get_col('RF_EVER_SMOKER') == 1)).astype(int)

# BUG 51: Anticoagulant + haematuria (haem on anticoag less specific for cancer)
if 'MED_ANTICOAGULANTS_FLAG' in fm.columns or 'MED_ANTICOAGULANTS_COUNT' in fm.columns:
    anticoag = (get_col('MED_ANTICOAGULANTS_FLAG', 0) == 1) | (get_col('MED_ANTICOAGULANTS_COUNT', 0) > 0)
    fm['INT_ANTICOAG_AND_HAEM'] = (anticoag.astype(int) * (get_col('HAEM_TOTAL_COUNT') > 0).astype(int))
    fm['INT_HAEM_WITHOUT_ANTICOAG'] = (get_col('HAEM_TOTAL_COUNT') > 0).astype(int) * (1 - anticoag.astype(int))

# BUG 52: Norethisterone (female + haem) — differential for gynaecological cause
if 'MED_NORETHISTERONE_FLAG' in fm.columns or 'MED_NORETHISTERONE_COUNT' in fm.columns:
    noreth = (get_col('MED_NORETHISTERONE_FLAG', 0) == 1) | (get_col('MED_NORETHISTERONE_COUNT', 0) > 0)
    fm['INT_NORETHISTERONE_AND_HAEM'] = (noreth.astype(int) * (get_col('HAEM_TOTAL_COUNT') > 0).astype(int) * (get_col('SEX_FEMALE') == 1).astype(int))

# BUG 53: COPD meds + smoking / haem
if 'MED_COPD_RESPIRATORY_FLAG' in fm.columns or 'MED_COPD_RESPIRATORY_COUNT' in fm.columns:
    copd_med = (get_col('MED_COPD_RESPIRATORY_FLAG', 0) == 1) | (get_col('MED_COPD_RESPIRATORY_COUNT', 0) > 0)
    fm['INT_COPD_MEDS_AND_SMOKING'] = (copd_med.astype(int) * (get_col('RF_EVER_SMOKER') == 1).astype(int))
    fm['INT_COPD_MEDS_AND_HAEM'] = (copd_med.astype(int) * (get_col('HAEM_TOTAL_COUNT') > 0).astype(int))

# BUG 54: Haemostatic (tranexamic etc) + haem / low Hb — urgent bleeding signal
if 'MED_HAEMOSTATIC_FLAG' in fm.columns or 'MED_HAEMOSTATIC_COUNT' in fm.columns:
    haemostatic = (get_col('MED_HAEMOSTATIC_FLAG', 0) == 1) | (get_col('MED_HAEMOSTATIC_COUNT', 0) > 0)
    fm['INT_TRANEXAMIC_AND_HAEM'] = (haemostatic.astype(int) * (get_col('HAEM_TOTAL_COUNT') > 0).astype(int))
    fm['INT_TRANEXAMIC_AND_LOW_HB'] = (haemostatic.astype(int) * (get_col('LAB_ANAEMIA_MILD', 0) == 1).astype(int))
    fm['INT_TRANEXAMIC_HAEM_LOW_HB'] = (haemostatic.astype(int) * (get_col('HAEM_TOTAL_COUNT') > 0).astype(int) * (get_col('LAB_ANAEMIA_MILD', 0) == 1).astype(int))

int_count = len([c for c in fm.columns if c.startswith('INT_')])
print(f"  Interaction features: {int_count} features")
print(f"  Patients: {n_master:,}  |  Records: — (from merged matrix)")


# ══════════════════════════════════════════════════════════════
# LEVEL 18: RATIO FEATURES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LEVEL 18: RATIO FEATURES")
print("=" * 70)

fm['RATIO_OBS_TO_LAB'] = np.where(
    get_col('OBS_TOTAL_COUNT')+get_col('LAB_TOTAL_TESTS')>0,
    get_col('OBS_TOTAL_COUNT')/(get_col('OBS_TOTAL_COUNT')+get_col('LAB_TOTAL_TESTS')),0)
fm['RATIO_HAEM_TO_ALL_OBS'] = np.where(get_col('OBS_TOTAL_COUNT')>0, get_col('HAEM_TOTAL_COUNT')/get_col('OBS_TOTAL_COUNT'),0)
fm['RATIO_UTI_AB_TO_ALL_MED'] = np.where(get_col('MED_TOTAL_COUNT')>0, get_col('MED_UTI_ANTIBIOTICS_COUNT')/get_col('MED_TOTAL_COUNT'),0)
fm['RATIO_CATHETER_TO_ALL_MED'] = np.where(get_col('MED_TOTAL_COUNT')>0, get_col('MED_CATHETER_TOTAL_COUNT')/get_col('MED_TOTAL_COUNT'),0)
fm['RATIO_MED_TO_CLINICAL'] = np.where(
    get_col('OBS_TOTAL_COUNT')+get_col('MED_TOTAL_COUNT')>0,
    get_col('MED_TOTAL_COUNT')/(get_col('OBS_TOTAL_COUNT')+get_col('MED_TOTAL_COUNT')),0)
total_ab = get_col('TEMP_CLINICAL_WA')+get_col('TEMP_CLINICAL_WB')
fm['RATIO_WB_PROPORTION'] = np.where(total_ab>0, get_col('TEMP_CLINICAL_WB')/total_ab, 0.5)
total_all = get_col('TEMP_ALL_WA',0)+get_col('TEMP_ALL_WB',0)+get_col('TEMP_ALL_WCOMORB',0)+get_col('TEMP_ALL_WRF',0)
fm['RATIO_COMORB_TO_ALL'] = np.where(total_all>0, get_col('TEMP_ALL_WCOMORB',0)/total_all, 0)

ratio_count = len([c for c in fm.columns if c.startswith('RATIO_')])
print(f"  Ratio features: {ratio_count} features")
print(f"  Patients: {n_master:,}  |  Records: — (from merged matrix)")


# ══════════════════════════════════════════════════════════════
# EXTRA 1: RECENCY FEATURES
# How recently before index did key events happen?
# Cancer patients have MORE RECENT events
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXTRA 1: Recency features...")
print("=" * 70)

# BUG 24: Reuse obs_windowed from Level 14 (no redefinition)
recency_cats = [
    'HAEMATURIA', 'LUTS', 'URINE LAB ABNORMALITIES', 'URINE INVESTIGATIONS',
    'CATHETER/PROCEDURES', 'IMAGING', 'UROLOGICAL CONDITIONS',
    'WEIGHT_LOSS', 'FATIGUE', 'DYSURIA', 'BACK_PAIN', 'ANAEMIA_DX', 'FRAILTY'
]

for cat in recency_cats:
    cat_d = obs_windowed[obs_windowed['CATEGORY'] == cat]
    if len(cat_d) == 0:
        continue
    col = 'RECENCY_' + safe_col(cat)
    fm[col + '_MONTHS'] = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
    fm[col + '_IN_WB'] = (fm[col + '_MONTHS'].fillna(99) <= 23).astype(int)
    fm[col + '_VERY_RECENT'] = (fm[col + '_MONTHS'].fillna(99) <= 15).astype(int)

if len(meds) > 0:
    for cat in ['UTI ANTIBIOTICS', 'IRON SUPPLEMENTS', 'OPIOID ANALGESICS', 'HAEMOSTATIC']:
        cat_d = meds[meds['CATEGORY'] == cat]
        if len(cat_d) == 0:
            continue
        col = 'RECENCY_MED_' + safe_col(cat)
        fm[col + '_MONTHS'] = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        fm[col + '_IN_WB'] = (fm[col + '_MONTHS'].fillna(99) <= 23).astype(int)

recency_count = len([c for c in fm.columns if c.startswith('RECENCY_')])
print(f"  Recency features: {recency_count}")


# ══════════════════════════════════════════════════════════════
# EXTRA 2: FIRST OCCURRENCE + SPAN FEATURES
# When did symptoms FIRST appear? How long did they persist?
# ══════════════════════════════════════════════════════════════
print("\nEXTRA 2: First occurrence + span features...")

for cat in recency_cats:
    cat_d = obs_windowed[obs_windowed['CATEGORY'] == cat]
    if len(cat_d) == 0:
        continue
    col = safe_col(cat)
    first = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()
    last = cat_d.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
    fm[f'FIRST_{col}_MONTHS'] = first
    fm[f'SPAN_{col}_MONTHS'] = first - last
    fm[f'SPAN_{col}_LONGSTANDING'] = (fm[f'SPAN_{col}_MONTHS'].fillna(0) > 12).astype(int)

span_count = len([c for c in fm.columns if c.startswith(('FIRST_','SPAN_'))])
print(f"  First/Span features: {span_count}")


# ══════════════════════════════════════════════════════════════
# EXTRA 3: CROSS-LAB INTERACTION FEATURES
# Combinations of lab abnormalities that signal cancer
# ══════════════════════════════════════════════════════════════
print("\nEXTRA 3: Cross-lab interaction features...")

fm['XLAB_ANAEMIA_AND_ALP_HIGH'] = (
    (get_col('LAB_HB_EVER_BELOW_100', 0) == 1) &
    (get_col('LAB_ALP_ELEVATED', 0) == 1)
).astype(int)

fm['XLAB_LOW_ALBUMIN_HIGH_CRP'] = (
    (get_col('LAB_ALBUMIN_LOW', 0) == 1) &
    (get_col('LAB_CRP_HIGH', 0) == 1)
).astype(int)

fm['XLAB_IDA_FULL_PATTERN'] = (
    (get_col('LAB_HB_EVER_BELOW_100', 0) == 1) &
    (get_col('LAB_IRON_DEFICIENCY_PATTERN', 0) == 1)
).astype(int)

fm['XLAB_MICROCYTIC_AND_HAEM'] = (
    (get_col('LAB_MICROCYTIC_ANAEMIA_PATTERN', 0) == 1) &
    (get_col('HAEM_ANY_FLAG', 0) == 1)
).astype(int)

fm['XLAB_HB_AND_EGFR_BOTH_DECLINING'] = (
    (get_col('LAB_HB_DECLINING', 0) == 1) &
    (get_col('LAB_EGFR_DECLINING_5', 0) == 1)
).astype(int)

fm['XLAB_CRP_AND_ESR_BOTH_HIGH'] = (
    (get_col('LAB_CRP_HIGH', 0) == 1) &
    (get_col('LAB_ESR_HIGH', 0) == 1)
).astype(int)

fm['XLAB_CALCIUM_HIGH_AND_WEIGHT_LOSS'] = (
    (get_col('LAB_CALCIUM_HIGH', 0) == 1) &
    (get_col('NOBS_WEIGHT_LOSS_FLAG', 0) == 1)
).astype(int)

fm['XLAB_PLATELETS_HIGH_AND_HAEM'] = (
    (get_col('LAB_PLATELETS_HIGH', 0) == 1) &
    (get_col('HAEM_ANY_FLAG', 0) == 1)
).astype(int)

fm['XLAB_LOW_ALBUMIN_WEIGHT_LOSS'] = (
    (get_col('LAB_ALBUMIN_LOW', 0) == 1) &
    (get_col('NOBS_WEIGHT_LOSS_FLAG', 0) == 1)
).astype(int)

fm['XLAB_PSA_HIGH_AND_HAEM'] = (
    (get_col('LAB_PSA_HIGH', 0) == 1) &
    (get_col('HAEM_ANY_FLAG', 0) == 1)
).astype(int)

abnormal_labs = []
for col_name in ['LAB_ANAEMIA_MILD', 'LAB_CRP_ELEVATED', 'LAB_ESR_ELEVATED',
                 'LAB_ALBUMIN_LOW', 'LAB_EGFR_LOW', 'LAB_CREATININE_HIGH',
                 'LAB_ALP_ELEVATED', 'LAB_WBC_HIGH', 'LAB_PLATELETS_HIGH',
                 'LAB_FERRITIN_LOW', 'LAB_IRON_LOW', 'LAB_MCV_LOW',
                 'LAB_CALCIUM_HIGH', 'LAB_HB_DECLINING']:
    if col_name in fm.columns:
        abnormal_labs.append(get_col(col_name, 0))
if abnormal_labs:
    fm['XLAB_ABNORMAL_COUNT'] = sum(abnormal_labs)
    fm['XLAB_MULTI_ABNORMAL'] = (fm['XLAB_ABNORMAL_COUNT'] >= 3).astype(int)

xlab_count = len([c for c in fm.columns if c.startswith('XLAB_')])
print(f"  Cross-lab features: {xlab_count}")


# ══════════════════════════════════════════════════════════════
# EXTRA 4: SYMPTOM DIVERSITY OVER TIME
# How many DIFFERENT symptom categories appear across windows?
# Cancer patients accumulate more diverse symptoms
# ══════════════════════════════════════════════════════════════
print("\nEXTRA 4: Symptom diversity features...")

for wname, (lo, hi) in windows.items():
    w = obs_windowed[(obs_windowed['MONTHS_BEFORE_INDEX'] >= lo) & (obs_windowed['MONTHS_BEFORE_INDEX'] <= hi)]
    fm[f'DIV_UNIQUE_CATS_{wname}'] = w.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    fm[f'DIV_UNIQUE_TERMS_{wname}'] = w.groupby('PATIENT_GUID')['TERM'].nunique()

fm['DIV_CAT_ACCELERATION'] = (
    (get_col('DIV_UNIQUE_CATS_W1', 0) + get_col('DIV_UNIQUE_CATS_W2', 0))
    - (get_col('DIV_UNIQUE_CATS_W3', 0) + get_col('DIV_UNIQUE_CATS_W4', 0))
)

early_cats = obs_windowed[obs_windowed['MONTHS_BEFORE_INDEX'] >= 24].groupby('PATIENT_GUID')['CATEGORY'].apply(set)
late_cats = obs_windowed[obs_windowed['MONTHS_BEFORE_INDEX'] < 24].groupby('PATIENT_GUID')['CATEGORY'].apply(set)
new_cats_df = pd.DataFrame({'early': early_cats, 'late': late_cats})
new_cats_df['early'] = new_cats_df['early'].apply(lambda x: x if isinstance(x, set) else set())
new_cats_df['late'] = new_cats_df['late'].apply(lambda x: x if isinstance(x, set) else set())
new_cats_df['DIV_NEW_CATS_IN_LATE'] = new_cats_df.apply(lambda r: len(r['late'] - r['early']), axis=1)
# BUG 47: Safer assignment — reindex to fm.index to avoid join duplicating rows
fm['DIV_NEW_CATS_IN_LATE'] = new_cats_df['DIV_NEW_CATS_IN_LATE'].reindex(fm.index).values

fm['DIV_TOTAL_UNIQUE_EVENT_TYPES'] = clinical[
    clinical['TIME_WINDOW'].isin(['A','B'])
].groupby('PATIENT_GUID')['EVENT_TYPE'].nunique()

cancer_cats = ['WEIGHT_LOSS','FATIGUE','APPETITE_LOSS','DYSURIA','SUPRAPUBIC_PAIN',
               'BACK_PAIN','LOIN_PAIN','DVT','PULMONARY_EMBOLISM','ANAEMIA_DX','NIGHT_SWEATS','FRAILTY']
cancer_cat_flags = []
for cat in cancer_cats:
    flag_col = f'NOBS_{safe_col(cat)}_FLAG'
    if flag_col in fm.columns:
        cancer_cat_flags.append(get_col(flag_col, 0))
if cancer_cat_flags:
    fm['DIV_CANCER_SPECIFIC_CAT_COUNT'] = sum(cancer_cat_flags)
    fm['DIV_CANCER_SPECIFIC_HIGH'] = (fm['DIV_CANCER_SPECIFIC_CAT_COUNT'] >= 3).astype(int)

div_count = len([c for c in fm.columns if c.startswith('DIV_')])
print(f"  Diversity features: {div_count}")


# ══════════════════════════════════════════════════════════════
# EXTRA 5: LAB-TO-EVENT TIMING
# When did abnormal labs happen relative to symptoms?
# ══════════════════════════════════════════════════════════════
print("\nEXTRA 5: Lab-to-event timing features...")

lab_windowed = lab_data[lab_data['TIME_WINDOW'].isin(['A','B'])].copy()

hb_low = lab_windowed[
    (lab_windowed['TERM'].str.contains('Haemoglobin', case=False, na=False)) &
    (lab_windowed['VALUE'] < 120)
]
haem_first = haem.groupby('PATIENT_GUID')['EVENT_DATE'].min()
if len(hb_low) > 0:
    hb_low_first = hb_low.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    timing = pd.DataFrame({'hb_low': hb_low_first, 'haem': haem_first}).dropna()
    timing['LTIMING_HB_LOW_TO_HAEM_DAYS'] = (timing['haem'] - timing['hb_low']).dt.days
    fm = fm.join(timing[['LTIMING_HB_LOW_TO_HAEM_DAYS']], how='left')

crp_high = lab_windowed[
    (lab_windowed['TERM'].str.contains('C-reactive', case=False, na=False)) &
    (lab_windowed['VALUE'] > 10)
]
if len(crp_high) > 0:
    crp_high_first = crp_high.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    timing2 = pd.DataFrame({'crp_high': crp_high_first, 'haem': haem_first}).dropna()
    timing2['LTIMING_CRP_HIGH_TO_HAEM_DAYS'] = (timing2['haem'] - timing2['crp_high']).dt.days
    fm = fm.join(timing2[['LTIMING_CRP_HIGH_TO_HAEM_DAYS']], how='left')

first_symptom = obs_windowed.groupby('PATIENT_GUID')['EVENT_DATE'].min()
if len(lab_windowed) > 0:
    merged_lt = lab_windowed.merge(
        first_symptom.reset_index().rename(columns={'EVENT_DATE': 'first_symptom'}),
        on='PATIENT_GUID', how='inner'
    )
    labs_before_symptom = merged_lt[merged_lt['EVENT_DATE'] < merged_lt['first_symptom']]
    fm['LTIMING_LABS_BEFORE_FIRST_SYMPTOM'] = labs_before_symptom.groupby('PATIENT_GUID').size()

ltiming_count = len([c for c in fm.columns if c.startswith('LTIMING_')])
print(f"  Lab-timing features: {ltiming_count}")


# ══════════════════════════════════════════════════════════════
# EXTRA 6: CONSULTATION PATTERN FEATURES
# How the patient interacts with the GP system
# ══════════════════════════════════════════════════════════════
print("\nEXTRA 6: Consultation pattern features...")

all_windowed = clinical[clinical['TIME_WINDOW'].isin(['A','B'])].copy()

visit_events = all_windowed.groupby(['PATIENT_GUID','EVENT_DATE']).size().reset_index(name='events_per_visit')
fm['CONSULT_MEAN_EVENTS_PER_VISIT'] = visit_events.groupby('PATIENT_GUID')['events_per_visit'].mean()
fm['CONSULT_MAX_EVENTS_PER_VISIT'] = visit_events.groupby('PATIENT_GUID')['events_per_visit'].max()
fm['CONSULT_STD_EVENTS_PER_VISIT'] = visit_events.groupby('PATIENT_GUID')['events_per_visit'].std()

visit_types = all_windowed.groupby(['PATIENT_GUID','EVENT_DATE'])['EVENT_TYPE'].nunique().reset_index(name='n_types')
fm['CONSULT_MULTI_TYPE_VISITS'] = visit_types[visit_types['n_types'] >= 2].groupby('PATIENT_GUID').size()
fm['CONSULT_MULTI_TYPE_RATIO'] = np.where(
    get_col('TEMP_GP_VISIT_DAYS', 1) > 0,
    get_col('CONSULT_MULTI_TYPE_VISITS', 0) / get_col('TEMP_GP_VISIT_DAYS', 1), 0)

visit_cats = all_windowed.groupby(['PATIENT_GUID','EVENT_DATE'])['CATEGORY'].nunique().reset_index(name='n_cats')
fm['CONSULT_COMPLEX_VISITS'] = visit_cats[visit_cats['n_cats'] >= 3].groupby('PATIENT_GUID').size()

all_sorted = all_windowed.sort_values(['PATIENT_GUID','EVENT_DATE']).drop_duplicates(['PATIENT_GUID','EVENT_DATE'])
all_sorted['PREV'] = all_sorted.groupby('PATIENT_GUID')['EVENT_DATE'].shift(1)
all_sorted['GAP'] = (all_sorted['EVENT_DATE'] - all_sorted['PREV']).dt.days
# BUG 44: Require 3+ gaps (4+ visits) for meaningful CV; dropna and guard std/mean
def _gap_cv_agg(x):
    g = x.dropna()
    if len(g) < 3 or g.mean() <= 0:
        return 0.0
    return g.std() / g.mean() if g.std() > 0 else 0.0
gap_cv = all_sorted.groupby('PATIENT_GUID')['GAP'].agg(_gap_cv_agg)
fm['CONSULT_GAP_CV'] = gap_cv

short_returns = all_sorted[(all_sorted['GAP'] > 0) & (all_sorted['GAP'] <= 14)]
fm['CONSULT_SHORT_RETURN_COUNT'] = short_returns.groupby('PATIENT_GUID').size()

very_short = all_sorted[(all_sorted['GAP'] > 0) & (all_sorted['GAP'] <= 7)]
fm['CONSULT_VERY_SHORT_RETURN_COUNT'] = very_short.groupby('PATIENT_GUID').size()

consult_count = len([c for c in fm.columns if c.startswith('CONSULT_')])
print(f"  Consultation features: {consult_count}")

extra_total = (
    len([c for c in fm.columns if c.startswith('RECENCY_')]) +
    len([c for c in fm.columns if c.startswith('FIRST_')]) + len([c for c in fm.columns if c.startswith('SPAN_')]) +
    len([c for c in fm.columns if c.startswith('XLAB_')]) +
    len([c for c in fm.columns if c.startswith('DIV_')]) +
    len([c for c in fm.columns if c.startswith('LTIMING_')]) +
    len([c for c in fm.columns if c.startswith('CONSULT_')]))
print(f"\n  TOTAL EXTRA FEATURES: {extra_total}")


# ══════════════════════════════════════════════════════════════
# FILL NEW EXTRA FEATURES (non-timing keep NaN for interpretability)
# ══════════════════════════════════════════════════════════════
new_patterns = ['RECENCY_', 'FIRST_', 'SPAN_', 'XLAB_', 'DIV_', 'LTIMING_', 'CONSULT_']
for col in fm.columns:
    if col == 'LABEL':
        continue
    if any(col.startswith(p) for p in new_patterns):
        if '_MONTHS' in col or '_DAYS' in col:
            pass
        else:
            fm[col] = fm[col].fillna(0)

fm.replace([np.inf, -np.inf], np.nan, inplace=True)


# ══════════════════════════════════════════════════════════════
# V5: FEATURE ENGINEERING IMPROVEMENTS (12mo: includes lab acceleration)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("V5-1: LAB CLINICAL RATIOS")
print("=" * 70)

if 'LAB_WBC_LAST' in fm.columns and 'LAB_HB_LAST' in fm.columns:
    hb = fm['LAB_HB_LAST'].fillna(0)
    fm['V5_WBC_HB_RATIO'] = np.where(hb > 0, fm['LAB_WBC_LAST'].fillna(0) / np.where(hb > 0, hb, 1) * 100, np.nan)
if 'LAB_PLATELETS_LAST' in fm.columns and 'LAB_WBC_LAST' in fm.columns:
    wbc = fm['LAB_WBC_LAST'].fillna(0)
    fm['V5_PLT_WBC_RATIO'] = np.where(wbc > 0, fm['LAB_PLATELETS_LAST'].fillna(0) / np.where(wbc > 0, wbc, 1), np.nan)
if 'LAB_CRP_LAST' in fm.columns and 'LAB_ALBUMIN_LAST' in fm.columns:
    alb = fm['LAB_ALBUMIN_LAST'].fillna(0)
    fm['V5_CRP_ALBUMIN_RATIO'] = np.where(alb > 0, fm['LAB_CRP_LAST'].fillna(0) / np.where(alb > 0, alb, 1), np.nan)
    fm['V5_GLASGOW_PROGNOSTIC_PROXY'] = (
        (fm['LAB_CRP_LAST'].fillna(0) > 10).astype(int) +
        (fm['LAB_ALBUMIN_LAST'].fillna(99) < 35).astype(int)
    )
if 'LAB_EGFR_LAST' in fm.columns and 'LAB_CREATININE_LAST' in fm.columns:
    fm['V5_EGFR_CREATININE_PRODUCT'] = (
        fm['LAB_EGFR_LAST'].fillna(0) * fm['LAB_CREATININE_LAST'].fillna(0) / 1000)
if 'LAB_HB_LAST' in fm.columns and 'LAB_MCV_LAST' in fm.columns:
    mcv = fm['LAB_MCV_LAST'].fillna(0)
    fm['V5_HB_MCV_RATIO'] = np.where(mcv > 0, fm['LAB_HB_LAST'].fillna(0) / np.where(mcv > 0, mcv, 1), np.nan)
if 'LAB_FERRITIN_LAST' in fm.columns and 'LAB_CRP_LAST' in fm.columns:
    crp = fm['LAB_CRP_LAST'].fillna(0)
    fm['V5_FERRITIN_CRP_RATIO'] = np.where(crp > 0, fm['LAB_FERRITIN_LAST'].fillna(0) / np.where(crp > 0, crp, 1), np.nan)
    fm['V5_TRUE_IRON_DEFICIENCY'] = (
        (fm['LAB_FERRITIN_LAST'].fillna(999) < 30) &
        (fm['LAB_CRP_LAST'].fillna(0) > 5)
    ).astype(int)
if 'LAB_CALCIUM_LAST' in fm.columns and 'LAB_ALBUMIN_LAST' in fm.columns:
    fm['V5_CORRECTED_CALCIUM'] = (
        fm['LAB_CALCIUM_LAST'].fillna(0) +
        0.02 * (40 - fm['LAB_ALBUMIN_LAST'].fillna(40)))
    fm['V5_HYPERCALCAEMIA_CORRECTED'] = (fm['V5_CORRECTED_CALCIUM'] > 2.6).astype(int)

v5_1_count = len([c for c in fm.columns if c.startswith('V5_')])
print(f"  Lab ratio features: {v5_1_count}")


# V5-2: Lab trajectory acceleration (12mo only — needs 24+ month lookback)
print("\n" + "=" * 70)
print("V5-2: LAB TRAJECTORY ACCELERATION (12mo only)")
print("=" * 70)
if WINDOW == '12mo':
    key_accel_labs = ['HB', 'EGFR', 'ALBUMIN', 'FERRITIN', 'WEIGHT', 'PLATELETS', 'CRP']
    lab_windowed = lab_data[lab_data['TIME_WINDOW'].isin(['A', 'B'])].copy()
    for short in key_accel_labs:
        if short not in all_lab_terms:
            continue
        category, term = all_lab_terms[short]
        t_data = lab_windowed[
            (lab_windowed['CATEGORY'] == category) &
            (lab_windowed['TERM'] == term) &
            (lab_windowed['VALUE'].notna())
        ]
        if len(t_data) < 10:
            continue
        prefix = f'V5_ACCEL_{short}'
        early_slopes, late_slopes = {}, {}
        for pid, group in t_data.groupby('PATIENT_GUID'):
            early = group[group['MONTHS_BEFORE_INDEX'] >= 24]
            late = group[group['MONTHS_BEFORE_INDEX'] < 24]
            if len(early) >= 2:
                x = -early['MONTHS_BEFORE_INDEX'].values.astype(float)
                y = early['VALUE'].values.astype(float)
                if np.std(x) > 0 and np.std(y) > 0:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    early_slopes[pid] = slope
                elif np.std(x) > 0:
                    early_slopes[pid] = 0.0
            if len(late) >= 2:
                x = -late['MONTHS_BEFORE_INDEX'].values.astype(float)
                y = late['VALUE'].values.astype(float)
                if np.std(x) > 0 and np.std(y) > 0:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    late_slopes[pid] = slope
                elif np.std(x) > 0:
                    late_slopes[pid] = 0.0
        early_s = pd.Series(early_slopes)
        late_s = pd.Series(late_slopes)
        fm[f'{prefix}_EARLY_SLOPE'] = early_s
        fm[f'{prefix}_LATE_SLOPE'] = late_s
        both = pd.DataFrame({'early': early_s, 'late': late_s}).dropna()
        if len(both) > 0:
            fm[f'{prefix}_SLOPE_CHANGE'] = late_s - early_s
        if short in ['HB', 'EGFR', 'ALBUMIN', 'FERRITIN', 'WEIGHT']:
            if f'{prefix}_SLOPE_CHANGE' in fm.columns:
                fm[f'{prefix}_WORSENING'] = (fm[f'{prefix}_SLOPE_CHANGE'].fillna(0) < -0.5).astype(int)
        elif short in ['CRP', 'PLATELETS']:
            if f'{prefix}_SLOPE_CHANGE' in fm.columns:
                fm[f'{prefix}_WORSENING'] = (fm[f'{prefix}_SLOPE_CHANGE'].fillna(0) > 0.5).astype(int)
    v5_2_count = len([c for c in fm.columns if c.startswith('V5_ACCEL_')])
    print(f"  Lab acceleration features: {v5_2_count}")
else:
    print("  Skipped (requires 12mo window)")


# V5-3: Medication temporal patterns
print("\n" + "=" * 70)
print("V5-3: MEDICATION TEMPORAL PATTERNS")
print("=" * 70)
if len(meds) > 0:
    meds_windowed = meds[meds['TIME_WINDOW'].isin(['A', 'B'])].copy()
    for cat in ['UTI ANTIBIOTICS', 'IRON SUPPLEMENTS', 'OPIOID ANALGESICS',
                'BLADDER ANTISPASMODICS', 'LUTS MEDICATIONS', 'CATHETER MAINTENANCE',
                'CATHETER SUPPLIES - CATHETERS', 'ANTICOAGULANTS', 'HAEMOSTATIC',
                'URINARY RETENTION DRUGS']:
        cat_d = meds_windowed[meds_windowed['CATEGORY'] == cat]
        if len(cat_d) == 0:
            continue
        col = 'V5_MED_' + safe_col(cat)
        early = cat_d[cat_d['MONTHS_BEFORE_INDEX'] >= 24]
        late = cat_d[cat_d['MONTHS_BEFORE_INDEX'] < 24]
        early_pts = set(early['PATIENT_GUID'].unique())
        late_pts = set(late['PATIENT_GUID'].unique())
        new_in_late = late_pts - early_pts
        fm[f'{col}_NEW_START'] = fm.index.isin(new_in_late).astype(int)
        stopped = early_pts - late_pts
        fm[f'{col}_STOPPED'] = fm.index.isin(stopped).astype(int)
        early_counts = early.groupby('PATIENT_GUID').size()
        late_counts = late.groupby('PATIENT_GUID').size()
        both_counts = pd.DataFrame({'early': early_counts, 'late': late_counts}).fillna(0)
        increasing = set(both_counts[both_counts['late'] > both_counts['early'] * 1.5].index)
        fm[f'{col}_INCREASING'] = fm.index.isin(increasing).astype(int)
    uti_meds_w = meds_windowed[meds_windowed['CATEGORY'] == 'UTI ANTIBIOTICS']
    if len(uti_meds_w) > 0:
        early_uti = uti_meds_w[uti_meds_w['MONTHS_BEFORE_INDEX'] >= 24]
        late_uti = uti_meds_w[uti_meds_w['MONTHS_BEFORE_INDEX'] < 24]
        early_diversity = early_uti.groupby('PATIENT_GUID')['TERM'].nunique()
        late_diversity = late_uti.groupby('PATIENT_GUID')['TERM'].nunique()
        fm['V5_UTI_AB_EARLY_DIVERSITY'] = early_diversity
        fm['V5_UTI_AB_LATE_DIVERSITY'] = late_diversity
        early_drugs = early_uti.groupby('PATIENT_GUID')['TERM'].apply(set)
        late_drugs = late_uti.groupby('PATIENT_GUID')['TERM'].apply(set)
        drug_df = pd.DataFrame({'early': early_drugs, 'late': late_drugs})
        drug_df['early'] = drug_df['early'].apply(lambda x: x if isinstance(x, set) else set())
        drug_df['late'] = drug_df['late'].apply(lambda x: x if isinstance(x, set) else set())
        drug_df['V5_UTI_AB_NEW_DRUGS_TRIED'] = drug_df.apply(lambda r: len(r['late'] - r['early']), axis=1)
        fm = fm.join(drug_df[['V5_UTI_AB_NEW_DRUGS_TRIED']], how='left')
    cath_supply = meds_windowed[meds_windowed['CATEGORY'].str.contains('CATHETER', case=False, na=False)]
    if len(cath_supply) > 0:
        fm['V5_CATHETER_FIRST_MONTHS'] = cath_supply.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()
        fm['V5_CATHETER_LAST_MONTHS'] = cath_supply.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        fm['V5_CATHETER_DURATION_MONTHS'] = (
            fm['V5_CATHETER_FIRST_MONTHS'].fillna(0) - fm['V5_CATHETER_LAST_MONTHS'].fillna(0))
    meds_late = meds_windowed[meds_windowed['MONTHS_BEFORE_INDEX'] < 24]
    if len(meds_late) > 0:
        fm['V5_POLYPHARMACY_LATE_CATS'] = meds_late.groupby('PATIENT_GUID')['CATEGORY'].nunique()
        fm['V5_POLYPHARMACY_LATE_TERMS'] = meds_late.groupby('PATIENT_GUID')['TERM'].nunique()
        fm['V5_HIGH_POLYPHARMACY'] = (fm['V5_POLYPHARMACY_LATE_CATS'].fillna(0) >= 5).astype(int)
v5_3_count = len([c for c in fm.columns if c.startswith('V5_MED_') or c.startswith('V5_UTI_') or
                  c.startswith('V5_CATHETER_') or c.startswith('V5_POLY') or c.startswith('V5_HIGH')])
print(f"  Medication temporal features: {v5_3_count}")


# V5-4: Symptom-to-investigation gap
print("\n" + "=" * 70)
print("V5-4: INVESTIGATION GAP PATTERNS")
print("=" * 70)
obs_windowed_v5 = obs_data[obs_data['TIME_WINDOW'].isin(['A', 'B'])].copy()
if len(haem) > 0 and len(obs_windowed_v5[obs_windowed_v5['CATEGORY'] == 'URINE INVESTIGATIONS']) > 0:
    # BUG 18: Vectorized merge instead of O(n²) nested loop
    haem_ev = haem[['PATIENT_GUID', 'EVENT_DATE']].rename(columns={'EVENT_DATE': 'HAEM_DATE'})
    ui_ev = obs_windowed_v5[obs_windowed_v5['CATEGORY'] == 'URINE INVESTIGATIONS'][['PATIENT_GUID', 'EVENT_DATE']].rename(columns={'EVENT_DATE': 'UI_DATE'})
    merged_inv = haem_ev.merge(ui_ev, on='PATIENT_GUID', how='inner')
    merged_inv['GAP_DAYS'] = (merged_inv['UI_DATE'] - merged_inv['HAEM_DATE']).dt.total_seconds() / 86400
    after_haem = merged_inv[merged_inv['GAP_DAYS'] >= 0]
    min_gap = after_haem.groupby('PATIENT_GUID')['GAP_DAYS'].min()
    fm['V5_HAEM_MIN_INVESTIGATION_GAP_DAYS'] = min_gap
    min_gap_aligned = min_gap.reindex(fm.index).fillna(999)
    fm['V5_HAEM_INVESTIGATED_14D'] = (min_gap_aligned <= 14).astype(int)
    fm['V5_HAEM_INVESTIGATED_30D'] = (min_gap_aligned <= 30).astype(int)
    haem_pats = set(haem['PATIENT_GUID'].unique())
    investigated_pats = set(min_gap.index)
    not_investigated = haem_pats - investigated_pats
    fm['V5_HAEM_NOT_INVESTIGATED'] = fm.index.isin(not_investigated).astype(int)
luts_data = obs_windowed_v5[obs_windowed_v5['CATEGORY'] == 'LUTS']
if len(luts_data) > 0:
    luts_first = luts_data.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    ui_events_df = obs_windowed_v5[obs_windowed_v5['CATEGORY'] == 'URINE INVESTIGATIONS']
    if len(ui_events_df) > 0:
        ui_first_after = {}
        for pid in luts_first.index:
            luts_date = luts_first[pid]
            pat_ui = ui_events_df[
                (ui_events_df['PATIENT_GUID'] == pid) &
                (ui_events_df['EVENT_DATE'] >= luts_date)
            ]
            if len(pat_ui) > 0:
                gap = (pat_ui['EVENT_DATE'].min() - luts_date).days
                ui_first_after[pid] = gap
        if ui_first_after:
            fm['V5_LUTS_TO_INVESTIGATION_DAYS'] = pd.Series(ui_first_after)
            fm['V5_LUTS_INVESTIGATED_30D'] = (fm['V5_LUTS_TO_INVESTIGATION_DAYS'].fillna(999) <= 30).astype(int)
v5_4_count = len([c for c in fm.columns if c.startswith('V5_HAEM_INV') or c.startswith('V5_HAEM_NOT') or c.startswith('V5_LUTS_')])
print(f"  Investigation gap features: {v5_4_count}")


# V5-5: Frailty-adjusted features
print("\n" + "=" * 70)
print("V5-5: FRAILTY-ADJUSTED FEATURES")
print("=" * 70)
frailty_proxy = (
    (fm['AGE_AT_INDEX'].fillna(0) >= 75).astype(float) * 1 +
    (fm['AGE_AT_INDEX'].fillna(0) >= 85).astype(float) * 1 +
    fm.get('COMORB_BURDEN_SCORE', pd.Series(0, index=fm.index)).fillna(0) / 5 +
    fm.get('MED_UNIQUE_CATEGORIES', pd.Series(0, index=fm.index)).fillna(0) / 5 +
    fm.get('NOBS_FRAILTY_SEVERITY', pd.Series(0, index=fm.index)).fillna(0) +
    fm.get('NOBS_WEIGHT_LOSS_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 0.5 +
    fm.get('NOBS_FATIGUE_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 0.5
)
fm['V5_FRAILTY_PROXY'] = frailty_proxy
fm['V5_HAEM_IN_FRAIL'] = (
    (fm.get('HAEM_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1) &
    (frailty_proxy >= 2)
).astype(int)
fm['V5_HAEM_IN_FIT'] = (
    (fm.get('HAEM_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1) &
    (frailty_proxy < 1)
).astype(int)
fm['V5_WL_IN_FIT_PATIENT'] = (
    (fm.get('NOBS_WEIGHT_LOSS_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1) &
    (fm['AGE_AT_INDEX'].fillna(0) < 70) &
    (fm.get('COMORB_BURDEN_SCORE', pd.Series(0, index=fm.index)).fillna(0) < 4)
).astype(int)
fm['V5_YOUNG_WITH_ANAEMIA'] = (
    (fm.get('LAB_ANAEMIA_MODERATE', pd.Series(0, index=fm.index)).fillna(0) == 1) &
    (fm['AGE_AT_INDEX'].fillna(0) < 65)
).astype(int)
v5_5_count = len([c for c in fm.columns if c.startswith('V5_FRAILTY') or c.startswith('V5_HAEM_IN_F') or c.startswith('V5_WL_IN') or c.startswith('V5_YOUNG')])
print(f"  Frailty-adjusted features: {v5_5_count}")


# V5-6: Multi-system involvement
print("\n" + "=" * 70)
print("V5-6: MULTI-SYSTEM INVOLVEMENT")
print("=" * 70)
system_flags = {
    'UROLOGICAL':     ['HAEM_ANY_FLAG', 'LUTS_ANY_FLAG', 'URO_ANY_FLAG', 'URINE_ANY_ABNORM_FLAG', 'CATH_ANY_FLAG'],
    'HAEMATOLOGICAL': ['LAB_ANAEMIA_MILD', 'LAB_FERRITIN_LOW', 'LAB_IRON_LOW', 'LAB_MCV_LOW', 'LAB_PLATELETS_HIGH', 'NOBS_ANAEMIA_DX_FLAG', 'COMORB_ANAEMIA_FLAG'],
    'RENAL':          ['LAB_EGFR_LOW', 'LAB_CREATININE_HIGH', 'COMORB_CKD_FLAG'],
    'INFLAMMATORY':   ['LAB_CRP_HIGH', 'LAB_ESR_HIGH', 'LAB_WBC_HIGH'],
    'NUTRITIONAL':    ['LAB_ALBUMIN_LOW', 'LAB_WEIGHT_LOSS_5PCT', 'LAB_BMI_UNDERWEIGHT', 'NOBS_WEIGHT_LOSS_FLAG', 'NOBS_APPETITE_LOSS_FLAG'],
    'CONSTITUTIONAL': ['NOBS_FATIGUE_FLAG', 'NOBS_NIGHT_SWEATS_FLAG', 'NOBS_FRAILTY_FLAG'],
    'HEPATIC':        ['LAB_ALP_ELEVATED', 'LAB_CALCIUM_HIGH'],
    'THROMBOTIC':     ['NOBS_DVT_FLAG', 'NOBS_PULMONARY_EMBOLISM_FLAG'],
    'PAIN':           ['NOBS_BACK_PAIN_FLAG', 'NOBS_LOIN_PAIN_FLAG', 'NOBS_SUPRAPUBIC_PAIN_FLAG', 'NOBS_DYSURIA_FLAG'],
}
for system, flags in system_flags.items():
    existing_flags = [f for f in flags if f in fm.columns]
    if existing_flags:
        fm[f'V5_SYSTEM_{system}_SCORE'] = sum(fm[f].fillna(0).clip(0, 1) for f in existing_flags)
        fm[f'V5_SYSTEM_{system}_ANY'] = (fm[f'V5_SYSTEM_{system}_SCORE'] > 0).astype(int)
system_any_cols = [c for c in fm.columns if c.startswith('V5_SYSTEM_') and c.endswith('_ANY')]
if system_any_cols:
    fm['V5_SYSTEMS_INVOLVED_COUNT'] = sum(fm[c].fillna(0) for c in system_any_cols)
    fm['V5_MULTI_SYSTEM_2'] = (fm['V5_SYSTEMS_INVOLVED_COUNT'] >= 2).astype(int)
    fm['V5_MULTI_SYSTEM_3'] = (fm['V5_SYSTEMS_INVOLVED_COUNT'] >= 3).astype(int)
    fm['V5_MULTI_SYSTEM_4'] = (fm['V5_SYSTEMS_INVOLVED_COUNT'] >= 4).astype(int)
if 'V5_SYSTEM_UROLOGICAL_SCORE' in fm.columns and 'V5_SYSTEM_HAEMATOLOGICAL_SCORE' in fm.columns:
    fm['V5_URO_HAEM_COMBINED'] = fm['V5_SYSTEM_UROLOGICAL_SCORE'].fillna(0) + fm['V5_SYSTEM_HAEMATOLOGICAL_SCORE'].fillna(0)
    fm['V5_URO_HAEM_BOTH_ACTIVE'] = (
        (fm['V5_SYSTEM_UROLOGICAL_ANY'].fillna(0) == 1) & (fm['V5_SYSTEM_HAEMATOLOGICAL_ANY'].fillna(0) == 1)
    ).astype(int)
v5_6_count = len([c for c in fm.columns if c.startswith('V5_SYSTEM') or c.startswith('V5_MULTI') or c.startswith('V5_URO_HAEM')])
print(f"  Multi-system features: {v5_6_count}")


# V5-7: Unexplained findings
print("\n" + "=" * 70)
print("V5-7: UNEXPLAINED FINDINGS")
print("=" * 70)
has_uti_dx = fm.get('COMORB_RECURRENT_UTI_FLAG', pd.Series(0, index=fm.index)).fillna(0)
has_haem = fm.get('HAEM_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0)
has_uti_ab = (fm.get('MED_UTI_ANTIBIOTICS_COUNT', pd.Series(0, index=fm.index)).fillna(0) > 0).astype(int)
fm['V5_UNEXPLAINED_HAEMATURIA'] = ((has_haem == 1) & (has_uti_dx == 0) & (has_uti_ab == 0)).astype(int)
fm['V5_PERSISTENT_HAEM_POST_AB'] = (
    (has_haem == 1) & (has_uti_ab == 1) &
    (fm.get('HAEM_TOTAL_COUNT', pd.Series(0, index=fm.index)).fillna(0) >= 2)
).astype(int)
has_anaemia_lab = fm.get('LAB_ANAEMIA_MILD', pd.Series(0, index=fm.index)).fillna(0)
has_iron_rx = (fm.get('MED_IRON_SUPPLEMENTS_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1)
fm['V5_UNEXPLAINED_ANAEMIA'] = (
    (has_anaemia_lab == 1) & (~has_iron_rx) &
    (fm.get('COMORB_ANAEMIA_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 0)
).astype(int)
has_wl = fm.get('NOBS_WEIGHT_LOSS_FLAG', pd.Series(0, index=fm.index)).fillna(0)
has_diabetes = fm.get('COMORB_DIABETES_FLAG', pd.Series(0, index=fm.index)).fillna(0)
fm['V5_UNEXPLAINED_WEIGHT_LOSS'] = ((has_wl == 1) & (has_diabetes == 0)).astype(int)
fm['V5_UNEXPLAINED_RAISED_CRP'] = (
    (fm.get('LAB_CRP_HIGH', pd.Series(0, index=fm.index)).fillna(0) == 1) &
    (fm.get('MED_UTI_ANTIBIOTICS_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 0) &
    (fm.get('COMORB_COPD_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 0)
).astype(int)
fm['V5_UNEXPLAINED_RAISED_ALP'] = (fm.get('LAB_ALP_ELEVATED', pd.Series(0, index=fm.index)).fillna(0) == 1).astype(int)
fm['V5_UNEXPLAINED_LOW_ALBUMIN'] = (
    (fm.get('LAB_ALBUMIN_LOW', pd.Series(0, index=fm.index)).fillna(0) == 1) &
    (fm.get('COMORB_CKD_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 0) &
    (fm.get('COMORB_HEART_FAILURE_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 0)
).astype(int)
unexplained_cols = [c for c in fm.columns if c.startswith('V5_UNEXPLAINED')]
if unexplained_cols:
    fm['V5_UNEXPLAINED_TOTAL'] = sum(fm[c].fillna(0) for c in unexplained_cols)
    fm['V5_MULTI_UNEXPLAINED'] = (fm['V5_UNEXPLAINED_TOTAL'] >= 2).astype(int)
v5_7_count = len([c for c in fm.columns if c.startswith('V5_UNEXPLAINED') or c.startswith('V5_PERSISTENT') or c.startswith('V5_MULTI_UNEX')])
print(f"  Unexplained findings features: {v5_7_count}")


# V5-8: NICE NG12 proxy
print("\n" + "=" * 70)
print("V5-8: NICE NG12 PROXY FEATURES")
print("=" * 70)
age = fm['AGE_AT_INDEX'].fillna(0)
male = fm.get('SEX_MALE', pd.Series(0, index=fm.index)).fillna(0)
haem_flag = fm.get('HAEM_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0)
frank_haem = (fm.get('HAEM_FRANK_COUNT', pd.Series(0, index=fm.index)).fillna(0) > 0).astype(int)
microscopic_haem = (fm.get('HAEM_MICROSCOPIC_COUNT', pd.Series(0, index=fm.index)).fillna(0) > 0).astype(int)
dysuria_flag = fm.get('NOBS_DYSURIA_FLAG', pd.Series(0, index=fm.index)).fillna(0)
wbc_high = fm.get('LAB_WBC_HIGH', pd.Series(0, index=fm.index)).fillna(0)
fm['V5_NICE_VH_CRITERIA'] = ((age >= 45) & (frank_haem == 1)).astype(int)
fm['V5_NICE_NVH_CRITERIA'] = (
    (age >= 60) & (microscopic_haem == 1) & ((dysuria_flag == 1) | (wbc_high == 1))
).astype(int)
fm['V5_NICE_ANY_CRITERIA'] = ((fm['V5_NICE_VH_CRITERIA'] == 1) | (fm['V5_NICE_NVH_CRITERIA'] == 1)).astype(int)
fm['V5_NICE_PLUS_SMOKING'] = (
    (fm['V5_NICE_ANY_CRITERIA'] == 1) & (fm.get('RF_EVER_SMOKER', pd.Series(0, index=fm.index)).fillna(0) == 1)
).astype(int)
fm['V5_NICE_PLUS_ANAEMIA'] = (
    (fm['V5_NICE_ANY_CRITERIA'] == 1) & (fm.get('LAB_ANAEMIA_MILD', pd.Series(0, index=fm.index)).fillna(0) == 1)
).astype(int)
fm['V5_NICE_PLUS_WEIGHT_LOSS'] = (
    (fm['V5_NICE_ANY_CRITERIA'] == 1) & (fm.get('NOBS_WEIGHT_LOSS_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1)
).astype(int)
fm['V5_QCANCER_PROXY'] = (
    (age / 10).clip(0, 10) + (male * 1.5) + (haem_flag * 5) + (frank_haem * 3) +
    (fm.get('RF_EVER_SMOKER', pd.Series(0, index=fm.index)).fillna(0) * 3) +
    (fm.get('RF_HEAVY_SMOKER_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 2) +
    (fm.get('COMORB_RECURRENT_UTI_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 2) +
    (fm.get('NOBS_WEIGHT_LOSS_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 2) +
    (fm.get('LAB_ANAEMIA_MILD', pd.Series(0, index=fm.index)).fillna(0) * 1.5) + (dysuria_flag * 1)
)
v5_8_count = len([c for c in fm.columns if c.startswith('V5_NICE') or c.startswith('V5_QCANCER')])
print(f"  NICE/QCancer proxy features: {v5_8_count}")


# V5-9: Smoking quantification
print("\n" + "=" * 70)
print("V5-9: SMOKING QUANTIFICATION")
print("=" * 70)
smoke_data_v5 = rf_data[rf_data['CATEGORY'].isin(['CURRENT SMOKER', 'EX SMOKER'])].copy()
smoke_with_val = smoke_data_v5[smoke_data_v5['VALUE'].notna()].copy()
if len(smoke_with_val) > 0:
    consumption = smoke_with_val[smoke_with_val['TERM'].str.contains(
        'consumption|Cigarette smoker|cigarette smoker', case=False, na=False)]
    if len(consumption) > 0:
        fm['V5_SMOKE_CIGS_PER_DAY_LAST'] = consumption.sort_values(
            'MONTHS_BEFORE_INDEX').groupby('PATIENT_GUID')['VALUE'].first()
        fm['V5_SMOKE_CIGS_PER_DAY_MAX'] = consumption.groupby('PATIENT_GUID')['VALUE'].max()
        fm['V5_SMOKE_HEAVY_GT20'] = (fm['V5_SMOKE_CIGS_PER_DAY_MAX'].fillna(0) >= 20).astype(int)
fm['V5_SMOKE_DURATION_PROXY'] = np.where(
    fm.get('RF_CURRENT_SMOKER_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1,
    (fm['AGE_AT_INDEX'].fillna(0) - 18).clip(0, None),
    np.where(
        fm.get('RF_EX_SMOKER_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1,
        (fm['AGE_AT_INDEX'].fillna(0) - 18).clip(0, None) * 0.5, 0))
if 'V5_SMOKE_CIGS_PER_DAY_MAX' in fm.columns:
    fm['V5_PACK_YEARS_PROXY'] = (
        fm['V5_SMOKE_CIGS_PER_DAY_MAX'].fillna(0) * fm['V5_SMOKE_DURATION_PROXY'].fillna(0) / 20)
    fm['V5_HIGH_PACK_YEARS'] = (fm['V5_PACK_YEARS_PROXY'] >= 20).astype(int)
v5_9_count = len([c for c in fm.columns if c.startswith('V5_SMOKE') or c.startswith('V5_PACK')])
print(f"  Smoking quantification features: {v5_9_count}")


# V5-10: Master cancer suspicion score v2
print("\n" + "=" * 70)
print("V5-10: MASTER SUSPICION SCORE v2")
print("=" * 70)
fm['V5_MASTER_SUSPICION_V2'] = (
    fm.get('HAEM_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 5 +
    fm.get('HAEM_RECURRENT', pd.Series(0, index=fm.index)).fillna(0) * 3 +
    fm.get('V5_UNEXPLAINED_HAEMATURIA', pd.Series(0, index=fm.index)).fillna(0) * 4 +
    fm.get('V5_PERSISTENT_HAEM_POST_AB', pd.Series(0, index=fm.index)).fillna(0) * 5 +
    fm.get('LUTS_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 2 +
    fm.get('LAB_ANAEMIA_MODERATE', pd.Series(0, index=fm.index)).fillna(0) * 3 +
    fm.get('LAB_HB_DECLINING', pd.Series(0, index=fm.index)).fillna(0) * 3 +
    fm.get('V5_TRUE_IRON_DEFICIENCY', pd.Series(0, index=fm.index)).fillna(0) * 3 +
    fm.get('V5_UNEXPLAINED_ANAEMIA', pd.Series(0, index=fm.index)).fillna(0) * 2 +
    fm.get('NOBS_WEIGHT_LOSS_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 2 +
    fm.get('NOBS_FATIGUE_FLAG', pd.Series(0, index=fm.index)).fillna(0) * 1 +
    fm.get('V5_UNEXPLAINED_WEIGHT_LOSS', pd.Series(0, index=fm.index)).fillna(0) * 2 +
    fm.get('V5_GLASGOW_PROGNOSTIC_PROXY', pd.Series(0, index=fm.index)).fillna(0) * 2 +
    fm.get('RF_EVER_SMOKER', pd.Series(0, index=fm.index)).fillna(0) * 2 +
    fm.get('V5_HIGH_PACK_YEARS', pd.Series(0, index=fm.index)).fillna(0) * 2 +
    (fm['AGE_AT_INDEX'].fillna(0) >= 65).astype(int) * 1 +
    fm.get('SEX_MALE', pd.Series(0, index=fm.index)).fillna(0) * 1 +
    fm.get('V5_MULTI_SYSTEM_3', pd.Series(0, index=fm.index)).fillna(0) * 3 +
    fm.get('SYN_UTI_TREATMENT_FAILURE', pd.Series(0, index=fm.index)).fillna(0) * 3 +
    (fm.get('TEMP_CLINICAL_ACCELERATION', pd.Series(0, index=fm.index)).fillna(0) > 1.5).astype(int) * 2
)
print("  Master suspicion v2 score created")


# FILL V5 FEATURES
print("\n" + "=" * 70)
print("FILLING V5 FEATURES")
print("=" * 70)
for col in fm.columns:
    if col.startswith('V5_') and col != 'LABEL':
        if '_MONTHS' in col or '_DAYS' in col or 'RATIO' in col or 'SLOPE' in col or 'PROXY' in col:
            pass
        else:
            fm[col] = fm[col].fillna(0)
fm.replace([np.inf, -np.inf], np.nan, inplace=True)
v5_total = len([c for c in fm.columns if c.startswith('V5_')])
print(f"  TOTAL V5 NEW FEATURES: {v5_total}")
print(f"  Total features now: {fm.shape[1] - 1}")

# ══════════════════════════════════════════════════════════════
# V6: TIER 1–3 FEATURES (after V5)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("V6-1: GP CONTACT INTENSITY CHANGE")
print("=" * 70)
obs_windowed_ab = obs_windowed
for q_name, (lo, hi) in [('Q1', (12, 14)), ('Q2', (15, 17)), ('Q3', (18, 20)), ('Q4', (21, 23)),
                         ('Q5', (24, 26)), ('Q6', (27, 29)), ('Q7', (30, 32)), ('Q8', (33, 36))]:
    q = obs_windowed_ab[obs_windowed_ab['MONTHS_BEFORE_INDEX'].between(lo, hi)]
    s = q.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    fm[f'V6_VISIT_DENSITY_{q_name}'] = s.reindex(fm.index)
visit_quarter_cols = [f'V6_VISIT_DENSITY_Q{i}' for i in range(1, 9) if f'V6_VISIT_DENSITY_Q{i}' in fm.columns]
if len(visit_quarter_cols) >= 4:
    q_df = fm[visit_quarter_cols].fillna(0)
    x = np.arange(len(visit_quarter_cols), dtype=float)
    slopes = np.apply_along_axis(
        lambda y: stats.linregress(x, y)[0] if np.std(y) > 0 else np.nan,
        1, q_df.values)
    fm['V6_VISIT_DENSITY_SLOPE'] = pd.Series(slopes, index=fm.index)
    fm['V6_VISIT_ACCELERATING'] = (fm['V6_VISIT_DENSITY_SLOPE'].fillna(0) > 0.3).astype(int)
    fm['V6_VISIT_RAPIDLY_ACCELERATING'] = (fm['V6_VISIT_DENSITY_SLOPE'].fillna(0) > 0.8).astype(int)

print("\n" + "=" * 70)
print("V6-2: DIAGNOSTIC ODYSSEY FEATURES")
print("=" * 70)
obs_windowed_monthly = obs_windowed_ab.copy()
obs_windowed_monthly['MONTH'] = obs_windowed_monthly['EVENT_DATE'].dt.to_period('M')
for cat in ['HAEMATURIA', 'LUTS', 'DYSURIA', 'BACK_PAIN', 'FATIGUE', 'WEIGHT_LOSS', 'UROLOGICAL CONDITIONS']:
    cat_d = obs_windowed_monthly[obs_windowed_monthly['CATEGORY'] == cat]
    if len(cat_d) == 0:
        continue
    col = safe_col(cat)
    months_with = cat_d.groupby('PATIENT_GUID')['MONTH'].nunique()
    fm[f'V6_ODYSSEY_{col}_MONTHS'] = months_with.reindex(fm.index)
    fm[f'V6_ODYSSEY_{col}_CHRONIC'] = (fm[f'V6_ODYSSEY_{col}_MONTHS'].fillna(0) >= 3).astype(int)
    fm[f'V6_ODYSSEY_{col}_PERSISTENT'] = (fm[f'V6_ODYSSEY_{col}_MONTHS'].fillna(0) >= 5).astype(int)
odyssey_chronic_cols = [c for c in fm.columns if c.startswith('V6_ODYSSEY_') and c.endswith('_CHRONIC')]
if odyssey_chronic_cols:
    fm['V6_ODYSSEY_TOTAL_CHRONIC'] = sum(fm[c].fillna(0) for c in odyssey_chronic_cols)
    fm['V6_ODYSSEY_MULTI_CHRONIC'] = (fm['V6_ODYSSEY_TOTAL_CHRONIC'] >= 2).astype(int)

print("\n" + "=" * 70)
print("V6-3: LAB TEST ORDERING PATTERNS")
print("=" * 70)
lab_windowed_v6 = lab_data[lab_data['TIME_WINDOW'].isin(['A', 'B'])].copy()
if len(lab_windowed_v6) > 0:
    lab_per_date = lab_windowed_v6.groupby(['PATIENT_GUID', 'EVENT_DATE']).agg(
        n_tests=('TERM', 'nunique'),
        n_categories=('CATEGORY', 'nunique')
    ).reset_index()
    fm['V6_LAB_MAX_TESTS_PER_VISIT'] = lab_per_date.groupby('PATIENT_GUID')['n_tests'].max().reindex(fm.index)
    fm['V6_LAB_MEAN_TESTS_PER_VISIT'] = lab_per_date.groupby('PATIENT_GUID')['n_tests'].mean().reindex(fm.index)
    comp_visits = (lab_per_date['n_tests'] >= 5).groupby(lab_per_date['PATIENT_GUID']).sum()
    fm['V6_LAB_COMPREHENSIVE_VISITS'] = comp_visits.reindex(fm.index).fillna(0)
    fbc_dates = lab_windowed_v6[lab_windowed_v6['CATEGORY'] == 'HAEMATOLOGY'].groupby('PATIENT_GUID')['EVENT_DATE'].apply(set)
    renal_dates = lab_windowed_v6[lab_windowed_v6['CATEGORY'] == 'RENAL'].groupby('PATIENT_GUID')['EVENT_DATE'].apply(set)
    inflam_dates = lab_windowed_v6[lab_windowed_v6['CATEGORY'] == 'INFLAMMATORY'].groupby('PATIENT_GUID')['EVENT_DATE'].apply(set)
    liver_dates = lab_windowed_v6[lab_windowed_v6['CATEGORY'] == 'LIVER'].groupby('PATIENT_GUID')['EVENT_DATE'].apply(set) if 'LIVER' in lab_windowed_v6['CATEGORY'].unique() else pd.Series(dtype=object)
    screening_df = pd.DataFrame({'fbc': fbc_dates, 'renal': renal_dates, 'inflam': inflam_dates, 'liver': liver_dates})
    screening_df = screening_df.map(lambda x: x if isinstance(x, set) else set())
    def count_screening_dates(row):
        all_dates = row['fbc'] | row['renal'] | row['inflam'] | row['liver']
        return sum(1 for d in all_dates if sum(1 for s in [row['fbc'], row['renal'], row['inflam'], row['liver']] if d in s) >= 3)
    fm['V6_LAB_SCREENING_PATTERN_COUNT'] = screening_df.apply(count_screening_dates, axis=1).reindex(fm.index)
    fm['V6_LAB_SCREENING_PATTERN_FLAG'] = (fm['V6_LAB_SCREENING_PATTERN_COUNT'].fillna(0) > 0).astype(int)
    early_lab = lab_windowed_v6[lab_windowed_v6['MONTHS_BEFORE_INDEX'] >= 24]
    late_lab = lab_windowed_v6[lab_windowed_v6['MONTHS_BEFORE_INDEX'] < 24]
    fm['V6_LAB_EARLY_COUNT'] = early_lab.groupby('PATIENT_GUID').size().reindex(fm.index)
    fm['V6_LAB_LATE_COUNT'] = late_lab.groupby('PATIENT_GUID').size().reindex(fm.index)
    fm['V6_LAB_COUNT_ACCELERATION'] = (fm['V6_LAB_LATE_COUNT'].fillna(0) - fm['V6_LAB_EARLY_COUNT'].fillna(0))
    fm['V6_LAB_TESTS_INCREASING'] = (fm['V6_LAB_COUNT_ACCELERATION'] > 3).astype(int)
    early_lab_types = early_lab.groupby('PATIENT_GUID')['TERM'].apply(set)
    late_lab_types = late_lab.groupby('PATIENT_GUID')['TERM'].apply(set)
    lab_types_df = pd.DataFrame({'early': early_lab_types, 'late': late_lab_types})
    lab_types_df = lab_types_df.fillna(set()).map(lambda x: x if isinstance(x, set) else set())
    lab_types_df['V6_NEW_LAB_TYPES'] = lab_types_df.apply(lambda r: len(r['late'] - r['early']), axis=1)
    fm['V6_NEW_LAB_TYPES_IN_LATE'] = lab_types_df['V6_NEW_LAB_TYPES'].reindex(fm.index)
    fm['V6_NEW_LAB_TYPES_FLAG'] = (fm['V6_NEW_LAB_TYPES_IN_LATE'].fillna(0) > 0).astype(int)

def cooccurrence_within_days(df, cat1, cat2, max_days, feature_name):
    c1 = df[df['CATEGORY'] == cat1][['PATIENT_GUID', 'EVENT_DATE']].rename(columns={'EVENT_DATE': 'D1'})
    c2 = df[df['CATEGORY'] == cat2][['PATIENT_GUID', 'EVENT_DATE']].rename(columns={'EVENT_DATE': 'D2'})
    if len(c1) == 0 or len(c2) == 0:
        return
    merged = c1.merge(c2, on='PATIENT_GUID')
    merged['GAP'] = (merged['D2'] - merged['D1']).dt.days.abs()
    close = merged[merged['GAP'] <= max_days]
    close_pats = set(close['PATIENT_GUID'].unique())
    fm[feature_name] = fm.index.isin(close_pats).astype(int)

print("\n" + "=" * 70)
print("V6-4: SYMPTOM CO-OCCURRENCE TIMING")
print("=" * 70)
cooccurrence_within_days(obs_windowed_ab, 'HAEMATURIA', 'LUTS', 30, 'V6_HAEM_LUTS_WITHIN_30D')
cooccurrence_within_days(obs_windowed_ab, 'HAEMATURIA', 'LUTS', 90, 'V6_HAEM_LUTS_WITHIN_90D')
cooccurrence_within_days(obs_windowed_ab, 'HAEMATURIA', 'WEIGHT_LOSS', 90, 'V6_HAEM_WL_WITHIN_90D')
cooccurrence_within_days(obs_windowed_ab, 'HAEMATURIA', 'FATIGUE', 90, 'V6_HAEM_FATIGUE_WITHIN_90D')
cooccurrence_within_days(obs_windowed_ab, 'HAEMATURIA', 'BACK_PAIN', 60, 'V6_HAEM_BACKPAIN_WITHIN_60D')
cooccurrence_within_days(obs_windowed_ab, 'HAEMATURIA', 'DYSURIA', 30, 'V6_HAEM_DYSURIA_WITHIN_30D')
cooccurrence_within_days(obs_windowed_ab, 'LUTS', 'DYSURIA', 30, 'V6_LUTS_DYSURIA_WITHIN_30D')
cooccurrence_within_days(obs_windowed_ab, 'HAEMATURIA', 'ANAEMIA_DX', 60, 'V6_HAEM_ANAEMIA_WITHIN_60D')

print("\n" + "=" * 70)
print("V6-5: TREATMENT RESPONSE FEATURES")
print("=" * 70)
if len(meds) > 0:
    meds_windowed_v6 = meds[meds['TIME_WINDOW'].isin(['A', 'B'])].copy()
    uti_ab_dates = meds_windowed_v6[meds_windowed_v6['CATEGORY'] == 'UTI ANTIBIOTICS']
    if len(uti_ab_dates) > 0 and len(haem) > 0:
        first_ab = uti_ab_dates.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        haem_ab = haem.merge(first_ab.reset_index().rename(columns={'EVENT_DATE': 'FIRST_AB_DATE'}), on='PATIENT_GUID', how='inner')
        haem_after_ab = haem_ab[haem_ab['EVENT_DATE'] > haem_ab['FIRST_AB_DATE']]
        fm['V6_HAEM_PERSISTS_AFTER_AB'] = fm.index.isin(set(haem_after_ab['PATIENT_GUID'].unique())).astype(int)
        fm['V6_HAEM_COUNT_AFTER_AB'] = haem_after_ab.groupby('PATIENT_GUID').size().reindex(fm.index)
        haem_gap_after_ab = haem_after_ab.groupby('PATIENT_GUID').apply(
            lambda g: (g['EVENT_DATE'].min() - g['FIRST_AB_DATE'].iloc[0]).days)
        fm['V6_DAYS_AB_TO_NEXT_HAEM'] = haem_gap_after_ab.reindex(fm.index)
    antispas = meds_windowed_v6[meds_windowed_v6['CATEGORY'] == 'BLADDER ANTISPASMODICS']
    luts_windowed_v6 = obs_windowed_ab[obs_windowed_ab['CATEGORY'] == 'LUTS']
    if len(antispas) > 0 and len(luts_windowed_v6) > 0:
        first_antispas = antispas.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        luts_w = luts_windowed_v6.merge(first_antispas.reset_index().rename(columns={'EVENT_DATE': 'FIRST_AS_DATE'}), on='PATIENT_GUID', how='inner')
        luts_after = luts_w[luts_w['EVENT_DATE'] > luts_w['FIRST_AS_DATE']]
        fm['V6_LUTS_PERSISTS_AFTER_ANTISPAS'] = fm.index.isin(set(luts_after['PATIENT_GUID'].unique())).astype(int)
    iron_meds = meds_windowed_v6[meds_windowed_v6['CATEGORY'] == 'IRON SUPPLEMENTS']
    if len(iron_meds) > 0 and 'LAB_HB_DECLINING' in fm.columns:
        iron_pats = set(iron_meds['PATIENT_GUID'].unique())
        fm['V6_IRON_BUT_HB_STILL_DROPPING'] = (
            fm.index.isin(iron_pats).astype(int) * fm.get('LAB_HB_DECLINING', pd.Series(0, index=fm.index)).fillna(0).astype(int))

print("\n" + "=" * 70)
print("V6-6: AGE-SEX-SYMPTOM RISK STRATIFICATION")
print("=" * 70)
age_v6 = fm['AGE_AT_INDEX'].fillna(0)
male_v6 = fm.get('SEX_MALE', pd.Series(0, index=fm.index)).fillna(0)
female_v6 = fm.get('SEX_FEMALE', pd.Series(0, index=fm.index)).fillna(0)
smoker_v6 = fm.get('RF_EVER_SMOKER', pd.Series(0, index=fm.index)).fillna(0)
haem_flag_v6 = fm.get('HAEM_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0)
luts_flag_v6 = fm.get('LUTS_ANY_FLAG', pd.Series(0, index=fm.index)).fillna(0)
fm['V6_PROFILE_CLASSIC_BC'] = ((age_v6 >= 60) & (male_v6 == 1) & (smoker_v6 == 1) & (haem_flag_v6 == 1)).astype(int)
fm['V6_PROFILE_MALE_HAEM_OVER50'] = ((age_v6 >= 50) & (male_v6 == 1) & (haem_flag_v6 == 1)).astype(int)
fm['V6_PROFILE_FEMALE_HAEM_OVER60'] = ((age_v6 >= 60) & (female_v6 == 1) & (haem_flag_v6 == 1)).astype(int)
fm['V6_PROFILE_SMOKER_LUTS_OVER60'] = ((age_v6 >= 60) & (smoker_v6 == 1) & (luts_flag_v6 == 1)).astype(int)
fm['V6_PROFILE_YOUNG_HAEM_SMOKER'] = (age_v6.between(40, 60) & (smoker_v6 == 1) & (haem_flag_v6 == 1)).astype(int)
fm['V6_RISK_TIER'] = 0
fm.loc[(age_v6 >= 45) & (haem_flag_v6 == 1), 'V6_RISK_TIER'] = 1
fm.loc[(age_v6 >= 60) & (haem_flag_v6 == 1), 'V6_RISK_TIER'] = 2
fm.loc[(age_v6 >= 60) & (haem_flag_v6 == 1) & (smoker_v6 == 1), 'V6_RISK_TIER'] = 3
fm.loc[(age_v6 >= 60) & (haem_flag_v6 == 1) & (smoker_v6 == 1) & (male_v6 == 1), 'V6_RISK_TIER'] = 4
age_decade = (age_v6 / 10).astype(int).clip(3, 9)
fm['V6_AGE_DECADE_X_HAEM'] = age_decade * haem_flag_v6
fm['V6_AGE_DECADE_X_LUTS'] = age_decade * luts_flag_v6
fm['V6_AGE_DECADE_X_SMOKING'] = age_decade * smoker_v6

print("\n" + "=" * 70)
print("V6-7: EVENT SEQUENCE PATTERNS (BIGRAMS)")
print("=" * 70)
event_seq = obs_windowed_ab.sort_values(['PATIENT_GUID', 'EVENT_DATE'])[['PATIENT_GUID', 'EVENT_DATE', 'CATEGORY']].copy()
event_seq['NEXT_CAT'] = event_seq.groupby('PATIENT_GUID')['CATEGORY'].shift(-1)
event_seq['BIGRAM'] = event_seq['CATEGORY'] + ' → ' + event_seq['NEXT_CAT'].fillna('')
event_seq = event_seq[event_seq['NEXT_CAT'].notna()]
bigram_features = {
    'V6_SEQ_HAEM_THEN_URINE': ('HAEMATURIA', 'URINE INVESTIGATIONS'),
    'V6_SEQ_HAEM_THEN_IMAGING': ('HAEMATURIA', 'IMAGING'),
    'V6_SEQ_HAEM_THEN_LUTS': ('HAEMATURIA', 'LUTS'),
    'V6_SEQ_LUTS_THEN_HAEM': ('LUTS', 'HAEMATURIA'),
    'V6_SEQ_DYSURIA_THEN_HAEM': ('DYSURIA', 'HAEMATURIA'),
    'V6_SEQ_URINE_THEN_IMAGING': ('URINE INVESTIGATIONS', 'IMAGING'),
    'V6_SEQ_FATIGUE_THEN_HAEM': ('FATIGUE', 'HAEMATURIA'),
}
for feat_name, (cat1, cat2) in bigram_features.items():
    matched = event_seq[(event_seq['CATEGORY'] == cat1) & (event_seq['NEXT_CAT'] == cat2)]
    fm[feat_name] = fm.index.isin(set(matched['PATIENT_GUID'].unique())).astype(int)
bigram_diversity = event_seq.groupby('PATIENT_GUID')['BIGRAM'].nunique()
fm['V6_EVENT_BIGRAM_DIVERSITY'] = bigram_diversity.reindex(fm.index)

print("\n" + "=" * 70)
print("V6-8: MEDICATION COMBINATION SIGNALS")
print("=" * 70)
if len(meds) > 0:
    meds_w_v6 = meds[meds['TIME_WINDOW'].isin(['A', 'B'])].copy()
    pat_med_cats = meds_w_v6.groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    combos = {
        'V6_COMBO_UTI_AB_AND_ANTISPAS': ('UTI ANTIBIOTICS', 'BLADDER ANTISPASMODICS'),
        'V6_COMBO_UTI_AB_AND_RETENTION': ('UTI ANTIBIOTICS', 'URINARY RETENTION DRUGS'),
        'V6_COMBO_IRON_AND_LAXATIVES': ('IRON SUPPLEMENTS', 'LAXATIVES'),
        'V6_COMBO_OPIOID_AND_ANTISPAS': ('OPIOID ANALGESICS', 'BLADDER ANTISPASMODICS'),
        'V6_COMBO_IRON_AND_OPIOID': ('IRON SUPPLEMENTS', 'OPIOID ANALGESICS'),
        'V6_COMBO_CATHETER_AND_UTI_AB': ('CATHETER SUPPLIES - CATHETERS', 'UTI ANTIBIOTICS'),
    }
    for feat_name, (cat1, cat2) in combos.items():
        pats_with_both = set(pid for pid, cats in pat_med_cats.items() if cat1 in cats and cat2 in cats)
        fm[feat_name] = fm.index.isin(pats_with_both).astype(int)
    fm['V6_MED_CATEGORY_DIVERSITY'] = pat_med_cats.reindex(fm.index)
    fm['V6_HIGH_MED_DIVERSITY'] = (fm['V6_MED_CATEGORY_DIVERSITY'].fillna(0) >= 6).astype(int)

print("\n" + "=" * 70)
print("V6-9: LAB VALUE RELATIVE TO BASELINE")
print("=" * 70)
for short in ['HB', 'EGFR', 'ALBUMIN', 'CRP', 'PLATELETS', 'WBC']:
    first_col = f'LAB_{short}_FIRST'
    last_col = f'LAB_{short}_LAST'
    mean_col = f'LAB_{short}_MEAN'
    if first_col in fm.columns and last_col in fm.columns:
        fm[f'V6_{short}_DEVIATION_FROM_BASELINE'] = fm[last_col].fillna(0) - fm[first_col].fillna(0)
        fm[f'V6_{short}_RELATIVE_CHANGE'] = np.where(
            fm[first_col].fillna(0).abs() > 0,
            (fm[last_col].fillna(0) - fm[first_col].fillna(0)) / fm[first_col].fillna(1).abs() * 100, 0)
    if mean_col in fm.columns and last_col in fm.columns:
        fm[f'V6_{short}_LATEST_VS_MEAN'] = fm[last_col].fillna(0) - fm[mean_col].fillna(0)
    if last_col in fm.columns:
        pop_mean = fm[last_col].mean()
        pop_std = fm[last_col].std()
        if pop_std > 0:
            fm[f'V6_{short}_POPULATION_ZSCORE'] = (fm[last_col].fillna(pop_mean) - pop_mean) / pop_std

print("\n" + "=" * 70)
print("V6-10: TIME-SINCE-LAST-EVENT (DAYS)")
print("=" * 70)
index_dates = clinical.groupby('PATIENT_GUID')['INDEX_DATE'].first()
for cat in ['HAEMATURIA', 'LUTS', 'DYSURIA', 'URINE INVESTIGATIONS', 'URINE LAB ABNORMALITIES', 'IMAGING', 'WEIGHT_LOSS', 'FATIGUE']:
    cat_d = obs_windowed_ab[obs_windowed_ab['CATEGORY'] == cat]
    if len(cat_d) == 0:
        continue
    col = safe_col(cat)
    last_date = cat_d.groupby('PATIENT_GUID')['EVENT_DATE'].max()
    timing = pd.DataFrame({'last': last_date, 'index': index_dates}).dropna()
    timing['days'] = (timing['index'] - timing['last']).dt.days
    fm[f'V6_DAYS_SINCE_LAST_{col}'] = timing['days'].reindex(fm.index)
    fm[f'V6_{col}_WITHIN_60D'] = (fm[f'V6_DAYS_SINCE_LAST_{col}'].fillna(9999) <= 60).astype(int)
    fm[f'V6_{col}_WITHIN_30D'] = (fm[f'V6_DAYS_SINCE_LAST_{col}'].fillna(9999) <= 30).astype(int)

print("\n" + "=" * 70)
print("V6-11: RED FLAG COMBINATION FEATURES")
print("=" * 70)
fm['V6_RED_FLAG_1'] = ((age_v6 >= 45) & (fm.get('HAEM_FRANK_COUNT', pd.Series(0, index=fm.index)).fillna(0) > 0)).astype(int)
fm['V6_RED_FLAG_2'] = ((age_v6 >= 60) & (fm.get('HAEM_MICROSCOPIC_COUNT', pd.Series(0, index=fm.index)).fillna(0) > 0) & (fm.get('NOBS_DYSURIA_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1)).astype(int)
fm['V6_RED_FLAG_3'] = ((age_v6 >= 60) & (fm.get('COMORB_RECURRENT_UTI_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1)).astype(int)
fm['V6_RED_FLAG_4'] = ((fm.get('V5_UNEXPLAINED_ANAEMIA', pd.Series(0, index=fm.index)).fillna(0) == 1) & (haem_flag_v6 == 1)).astype(int)
fm['V6_RED_FLAG_5'] = ((age_v6 >= 60) & (fm.get('NOBS_WEIGHT_LOSS_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1) & (fm.get('NOBS_FATIGUE_FLAG', pd.Series(0, index=fm.index)).fillna(0) == 1)).astype(int)
fm['V6_RED_FLAG_6'] = ((fm.get('LAB_PLATELETS_HIGH', pd.Series(0, index=fm.index)).fillna(0) == 1) & (haem_flag_v6 == 1)).astype(int)
red_flag_cols = [c for c in fm.columns if c.startswith('V6_RED_FLAG_') and not c.endswith('_COUNT') and c != 'V6_RED_FLAG_COUNT']
fm['V6_RED_FLAG_COUNT'] = sum(fm[c].fillna(0) for c in red_flag_cols)
fm['V6_MULTI_RED_FLAGS'] = (fm['V6_RED_FLAG_COUNT'] >= 2).astype(int)

print("\n" + "=" * 70)
print("V6 FILLNA")
print("=" * 70)
for col in fm.columns:
    if col.startswith('V6_') and col != 'LABEL':
        if any(x in col for x in ['_DAYS', '_MONTHS', 'DEVIATION', 'RELATIVE', 'ZSCORE', 'SLOPE', 'DENSITY_Q', 'DENSITY_']):
            pass
        else:
            fm[col] = fm[col].fillna(0)
fm.replace([np.inf, -np.inf], np.nan, inplace=True)
v6_total = len([c for c in fm.columns if c.startswith('V6_')])
print(f"  TOTAL V6 NEW FEATURES: {v6_total}")

meta_cols = ['CANCER_ID', 'INDEX_DATE']
meta = fm[meta_cols].copy()
fm.drop(columns=meta_cols, inplace=True)
assert 'CANCER_ID' not in fm.columns, "CANCER_ID still in feature matrix — DATA LEAKAGE!"
assert 'INDEX_DATE' not in fm.columns, "INDEX_DATE still in feature matrix — DATA LEAKAGE!"

count_flag_patterns = [
    '_COUNT', '_FLAG', '_TOTAL', '_WA', '_WB', '_RECURRENT', '_FREQUENT',
    'BURDEN', 'ACCELERATION', 'ANY_', 'SEX_MALE', 'SEX_FEMALE',
    'UNIQUE_', 'SPAN_', 'EVENTS_PER', 'GP_VISIT', 'AGE_BAND_',
    'AGE_SQUARED', 'INT_', 'RATIO_', 'MULTIMORBID', 'SEVERITY',
    'DECELERATION', '_HIGH', '_LOW', '_ELEVATED', '_DECLINING',
    '_VERY_', '_POOR_', '_MODERATE', '_MILD', '_SEVERE',
    'PREDIABETIC', 'DIABETIC', 'RISING', 'CATHETER_TOTAL',
    'COMPOSITE', 'SUSPICION', 'SYMPTOM_SCORE', 'DAYS_WITH',
    'MONTHS_WITH', 'MONTHLY_EVENT', 'SYN_', 'PATH_', 'SEQ_',
    'NOBS_', 'OBS_CLUSTER', 'OBS_6M_', 'BURST', 'DIVERSE',
    'NEW_LATE', 'OPIOID_', 'IRON_', 'HAEMOSTATIC', 'ESCALAT',
    'STRONG_OPIOID', 'LAXATIVE', 'MICROCYTIC', 'DEFICIENCY',
    'WEIGHT_LOSS', 'WEIGHT_CHANGE', 'BMI_', 'CALCIUM_',
    'FERRITIN_', 'HAEMATOCRIT_', 'RBC_', 'MCV_', 'IRON_LOW',
    'STOPPED_RECENTLY', 'NON_DRINKER', 'EXCESS', 'COUNSELLING',
    'TYPE_RECORDED', 'UNITS_RECORDED', 'AUDIT_RECORDED',
    'RECORDING_COUNT', 'EVER_BELOW', 'EVER_ABOVE', 'DROP_GT',
    'RAPIDLY_DECLINING',
    # BUG 33: V5 binary/count columns that should get fillna(0)
    'V5_SYSTEM_', 'V5_MULTI_', 'V5_URO_HAEM', 'V5_UNEXPLAINED', 'V5_PERSISTENT',
    'V5_NICE_', 'V5_HAEM_IN', 'V5_WL_IN', 'V5_YOUNG', 'V5_GLASGOW', 'V5_TRUE_IRON',
    'V5_MED_', 'V5_UTI_AB_', 'V5_HAEM_INV', 'V5_HAEM_NOT', 'V5_LUTS_INV',
    'V5_SMOKE_HEAVY', 'V5_HIGH_PACK', 'V5_MASTER', 'V5_HYPERCALC',
]

for col in fm.columns:
    if col == 'LABEL':
        continue
    if '_MONTHS' in col or '_DAYS' in col:
        continue
    if any(p in col for p in count_flag_patterns):
        fm[col] = fm[col].fillna(0)

fm.replace([np.inf, -np.inf], np.nan, inplace=True)

# IMP 3 + BUG 35: Data integrity checks before final summary
print("\n" + "=" * 70)
print("DATA INTEGRITY CHECKS")
print("=" * 70)
assert fm.index.is_unique, f"Duplicate patient_guids: {fm.index.duplicated().sum()}"
print("✅ No duplicate patients")
assert fm['LABEL'].isna().sum() == 0, f"LABEL has {fm['LABEL'].isna().sum()} NaN values!"
assert set(fm['LABEL'].dropna().unique()).issubset({0, 1}), f"LABEL has unexpected values: {fm['LABEL'].unique()}"
print(f"✅ LABEL integrity: {(fm['LABEL']==1).sum():,} cancer, {(fm['LABEL']==0).sum():,} non-cancer")
leakage_cols = [c for c in fm.columns if any(x in c.upper() for x in [
    'CANCER_ID', 'DATE_OF_DIAGNOSIS', 'DIAGNOSIS_DATE', 'INDEX_DATE',
    'REFERRAL', 'BIOPSY', 'CYSTOSCOPY', 'TWO_WEEK_WAIT', '2WW'])]
if leakage_cols:
    print(f"⚠️ POTENTIAL LEAKAGE COLUMNS: {leakage_cols}")
else:
    print("✅ No obvious leakage columns")
numeric_cols = fm.select_dtypes(include=[np.number]).columns.drop('LABEL', errors='ignore')
inf_cols = []
for c in numeric_cols:
    if c not in fm.columns:
        continue
    try:
        arr = np.asarray(fm[c].dropna(), dtype=float)
        if arr.size > 0 and np.isinf(arr).any():
            inf_cols.append(c)
    except (TypeError, ValueError):
        pass
if inf_cols:
    print(f"⚠️ Columns with infinity: {inf_cols[:10]}")
else:
    print("✅ No infinity values")

# BUG 34: Optional high-correlation (near-duplicate) feature check
try:
    from itertools import combinations
    features_only = fm.drop(columns=['LABEL'], errors='ignore')
    sample = features_only.sample(n=min(5000, len(features_only)), random_state=42)
    corr = sample.corr()
    high_corr_pairs = []
    for i, j in combinations(range(len(corr.columns)), 2):
        if abs(corr.iloc[i, j]) > 0.99:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    if high_corr_pairs:
        print(f"\n⚠️ BUG 34: {len(high_corr_pairs)} near-duplicate feature pairs (|r|>0.99), first 15:")
        for c1, c2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:15]:
            print(f"   {c1[:50]:50s} ↔ {c2[:50]:50s} r={r:.4f}")
    else:
        print("✅ No highly correlated feature pairs (|r|>0.99) in sample")
except Exception as e:
    print(f"  (Dedup check skipped: {e})")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL FEATURE MATRIX SUMMARY")
print("=" * 70)

label = fm['LABEL']
features = fm.drop(columns=['LABEL'])

print(f"Patients:             {len(fm):,}")
print(f"Total features:       {features.shape[1]}")
print(f"Cancer (label=1):     {(label==1).sum():,}")
print(f"Non-cancer (label=0): {(label==0).sum():,}")
print(f"Ratio:                1:{(label==0).sum()/(label==1).sum():.1f}")

groups = {
    'Demographics':          [c for c in features.columns if c.startswith(('AGE_', 'SEX_'))],
    'Observations':          [c for c in features.columns if c.startswith('OBS_') and not c.startswith('OBS_CLUSTER') and not c.startswith('OBS_6M')],
    'Haematuria':            [c for c in features.columns if c.startswith('HAEM_')],
    'Urine':                 [c for c in features.columns if c.startswith('URINE_')],
    'LUTS':                  [c for c in features.columns if c.startswith('LUTS_')],
    'Cath/Img/Uro/Gynae/Oth':[c for c in features.columns if c.startswith(('CATH_','IMG_','URO_','GYNAE_','OTHER_'))],
    'Risk Factors':          [c for c in features.columns if c.startswith('RF_')],
    'Comorbidities':         [c for c in features.columns if c.startswith('COMORB_')],
    'Lab Values':            [c for c in features.columns if c.startswith('LAB_')],
    'Medications':           [c for c in features.columns if c.startswith('MED_')],
    'Temporal':              [c for c in features.columns if c.startswith(('TEMP_','OBS_6M'))],
    'Clustering':            [c for c in features.columns if c.startswith('OBS_CLUSTER')],
    'New Obs (v4)':          [c for c in features.columns if c.startswith('NOBS_')],
    'Sequences (v4)':        [c for c in features.columns if c.startswith('SEQ_')],
    'Syndromes (v4)':        [c for c in features.columns if c.startswith('SYN_')],
    'Pathways':              [c for c in features.columns if c.startswith('PATH_')],
    'Interactions':          [c for c in features.columns if c.startswith('INT_')],
    'Ratios':                [c for c in features.columns if c.startswith('RATIO_')],
    'Recency (extra)':       [c for c in features.columns if c.startswith('RECENCY_')],
    'First/Span (extra)':    [c for c in features.columns if c.startswith(('FIRST_','SPAN_'))],
    'Cross-lab XLAB (extra)':[c for c in features.columns if c.startswith('XLAB_')],
    'Diversity (extra)':     [c for c in features.columns if c.startswith('DIV_')],
    'Lab-timing (extra)':   [c for c in features.columns if c.startswith('LTIMING_')],
    'Consultation (extra)':  [c for c in features.columns if c.startswith('CONSULT_')],
    'V5 (extra)':           [c for c in features.columns if c.startswith('V5_')],
    'V6 (extra)':           [c for c in features.columns if c.startswith('V6_')],
}

print(f"\nFeature groups:")
total_grouped = 0
for group, cols in groups.items():
    print(f"  {group:30s}: {len(cols):4d} features")
    total_grouped += len(cols)
ungrouped = features.shape[1] - total_grouped
if ungrouped > 0:
    print(f"  {'Other':30s}: {ungrouped:4d} features")

# NaN summary
nan_pct = (features.isna().sum() / len(features) * 100).sort_values(ascending=False)
print(f"\nTop 15 features with most NaN (%):")
print(nan_pct[nan_pct > 0].head(15).round(1))

zero_pct = ((features == 0).sum() / len(features) * 100).sort_values(ascending=False)
print(f"\nTop 15 most-zero features (%):")
print(zero_pct.head(15).round(1))


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

output_path = FE_RESULTS_DIR / f'bladder_feature_matrix_{WINDOW}.csv'
fm.reset_index().to_csv(output_path, index=False)
print(f"Feature matrix saved to: {output_path}")
print(f"  Shape: {fm.shape}")

meta_path = FE_RESULTS_DIR / f'bladder_feature_meta_{WINDOW}.csv'
meta.reset_index().to_csv(meta_path, index=False)
print(f"Meta saved to: {meta_path}")

with open(FE_RESULTS_DIR / f'bladder_feature_list_{WINDOW}.txt', 'w') as f:
    f.write(f"Total features: {features.shape[1]}\n\n")
    for group, cols in groups.items():
        f.write(f"\n{'='*60}\n{group} ({len(cols)} features)\n{'='*60}\n")
        for col in sorted(cols):
            nan_p = nan_pct.get(col, 0)
            if hasattr(nan_p, 'iloc'):
                nan_p = float(nan_p.iloc[0]) if len(nan_p) else 0.0
            else:
                nan_p = float(nan_p)
            f.write(f"  {col:60s} NaN: {nan_p:.1f}%\n")
print(f"Feature list saved to: {FE_RESULTS_DIR / f'bladder_feature_list_{WINDOW}.txt'}")
print(f"Log saved to: {LOG_FILE}")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING v4 FINAL COMPLETE ✅")
print(f"  {len(fm):,} patients × {features.shape[1]} features")
print("  YOUR ORIGINAL v2: Levels 1-12 (all preserved)")
print("  NEW v4 ADDITIONS: Levels 13-18")
print("=" * 70)
_log_f.close()
sys.stdout = sys.__stdout__        
