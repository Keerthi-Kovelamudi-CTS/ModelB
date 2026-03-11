#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  BLADDER CANCER — FEATURE ENGINEERING v4 FINAL
  Input:  data/{3mo|6mo|12mo|12mo_250k}/ — 12mo: no suffix; 12mo_250k: _12m_250k; 3mo/6mo: _3m|6m
  Output: FE_Results/{window}/ (bladder_feature_matrix_{window}.csv + meta, feature list, log)
  Usage: python 4_feature_engineering.py [--window 3mo|6mo|12mo|12mo_250k]  (default: 12mo_250k)
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Feature engineering.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
args = parser.parse_args()
WINDOW = args.window
RAW_SUFFIX = None if WINDOW == '12mo' else ('3m' if WINDOW == '3mo' else ('6m' if WINDOW == '6mo' else '12m_250k'))

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
elif WINDOW == '12mo_250k':
    clinical = pd.read_csv(DATA_DIR / 'FE_bladder_dropped_patients_clinical_windowed_12m_250k.csv')
    meds = pd.read_csv(DATA_DIR / 'FE_bladder_dropped_patients_med_windowed_12m_250k.csv')
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
    return (name.upper()
            .replace(' ', '_').replace('/', '_').replace('-', '_')
            .replace('(', '').replace(')', '').replace('+', 'PLUS')
            .replace('^', '').replace('.', '').replace(',', '')
            .replace(':', '_').replace('=', '_').replace('[', '').replace(']', '')
            .replace('__', '_').strip('_')[:50])

all_features = {}
windows = {'W1': (12,17), 'W2': (18,23), 'W3': (24,29), 'W4': (30,36)}

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
    'ALP2':       ('LIVER',         'Serum alkaline phosphatase level'),
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

# 2c. Count per individual TERM
term_counts = obs_data.groupby(['PATIENT_GUID', 'TERM']).size().unstack(fill_value=0)
term_counts.columns = ['OBS_TERM_' + safe_col(c) + '_COUNT' for c in term_counts.columns]
obs_feats = obs_feats.join(term_counts, how='left')

# 2d. Count per TERM × WINDOW
for window in ['A', 'B']:
    w = obs_data[obs_data['TIME_WINDOW'] == window]
    wc = w.groupby(['PATIENT_GUID', 'TERM']).size().unstack(fill_value=0)
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
        matched = haem[haem['TERM'].str.contains(term_pattern, case=False, na=False)]
        haem_feats[col_name] = matched.groupby('PATIENT_GUID').size()
    haem_feats['HAEM_FIRST_MONTHS_BEFORE'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()
    haem_feats['HAEM_LAST_MONTHS_BEFORE'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
    haem_feats['HAEM_SPAN_MONTHS'] = haem_feats['HAEM_FIRST_MONTHS_BEFORE'] - haem_feats['HAEM_LAST_MONTHS_BEFORE']
    haem_feats['HAEM_MEAN_MONTHS_BEFORE'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].mean()
    haem_feats['HAEM_RECURRENT'] = (haem_feats['HAEM_TOTAL_COUNT'] > 1).astype(float)
    haem_feats['HAEM_FREQUENT'] = (haem_feats['HAEM_TOTAL_COUNT'] >= 3).astype(float)
    haem_feats['HAEM_ACCELERATION'] = np.where(
        haem_feats.get('HAEM_WA_COUNT', 0).fillna(0) > 0,
        haem_feats.get('HAEM_WB_COUNT', 0).fillna(0) / haem_feats.get('HAEM_WA_COUNT', 0).fillna(1),
        np.where(haem_feats.get('HAEM_WB_COUNT', 0).fillna(0) > 0, 2.0, 0.0))
    haem_feats['HAEM_UNIQUE_DATES'] = haem.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    haem_feats['HAEM_ANY_FLAG'] = haem_feats.index.isin(set(haem['PATIENT_GUID'].unique())).astype(int)
    haem_feats['HAEM_FIRST_OCCURRENCE_MONTHS'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()

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
            if np.std(x) > 0:
                slope, _, _, _, _ = stats.linregress(x, y)
                alc_slopes[pid] = slope
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

# CKD stages
for stage, pattern in [('3A','stage 3A|3A'),('3B','stage 3B|3B'),('3','stage 3[^AB]|stage 3$'),('4','stage 4'),('5','stage 5')]:
    matched = comorb_data[(comorb_data['CATEGORY']=='CKD') & (comorb_data['TERM'].str.contains(pattern, case=False, na=False))]
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

for short, (category, term) in all_lab_terms.items():
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
            if np.std(x) > 0:
                slope, _, r_val, _, _ = stats.linregress(x, y)
                slopes[pid] = slope
                r_vals[pid] = r_val
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
            vals = group.sort_values('MONTHS_BEFORE_INDEX', ascending=False)['VALUE'].values
            return np.max(np.abs(np.diff(vals))) if len(vals)>=2 else 0
        lab_feats[f'{prefix}_MAX_STEP'] = multi3.groupby('PATIENT_GUID').apply(max_step)

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
    lab_feats['LAB_WEIGHT_CHANGE_KG'] = lab_feats.get('LAB_WEIGHT_LAST',0# (continuing from LAB_WEIGHT_CHANGE_KG)
    ).fillna(0) - lab_feats.get('LAB_WEIGHT_FIRST',0).fillna(0)
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
meds['DURATION_NUM'] = pd.to_numeric(meds.get('DURATION_IN_DAYS'), errors='coerce')

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
    dur = pd.to_numeric(cat_d['DURATION_IN_DAYS'], errors='coerce')
    med_feats[col + '_TOTAL_DURATION'] = cat_d.assign(DUR=dur).groupby('PATIENT_GUID')['DUR'].sum()

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
    med_feats['MED_UTI_AB_TOTAL_DURATION'] = uti_meds.assign(DUR=pd.to_numeric(uti_meds['DURATION_IN_DAYS'],errors='coerce')).groupby('PATIENT_GUID')['DUR'].sum()
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

# Specific drug term counts
term_counts = meds.groupby(['PATIENT_GUID', 'TERM']).size().unstack(fill_value=0)
term_counts.columns = ['MED_TERM_' + safe_col(c) + '_COUNT' for c in term_counts.columns]
med_feats = med_feats.join(term_counts, how='left')

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

all_events = clinical.copy()
for window in ['A', 'B', 'RF', 'COMORB']:
    w = all_events[all_events['TIME_WINDOW'] == window]
    temp_feats[f'TEMP_ALL_W{window}'] = w.groupby('PATIENT_GUID').size()

temp_feats['TEMP_GP_VISIT_DAYS'] = obs_lab.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
temp_feats['TEMP_GP_VISIT_DAYS_ALL'] = clinical.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
for window in ['A', 'B']:
    w = obs_lab[obs_lab['TIME_WINDOW'] == window]
    temp_feats[f'TEMP_GP_VISITS_W{window}'] = w.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
temp_feats['TEMP_GP_VISIT_ACCELERATION'] = np.where(
    temp_feats.get('TEMP_GP_VISITS_WA', 0).fillna(0) > 0,
    temp_feats.get('TEMP_GP_VISITS_WB', 0).fillna(0) / temp_feats.get('TEMP_GP_VISITS_WA', 0).fillna(1),
    np.where(temp_feats.get('TEMP_GP_VISITS_WB', 0).fillna(0) > 0, 2.0, 0.0))

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
obs_lab_cl['WEEK'] = obs_lab_cl['EVENT_DATE'].dt.isocalendar().week.astype(int)
obs_lab_cl['YEAR'] = obs_lab_cl['EVENT_DATE'].dt.year

weekly = obs_lab_cl.groupby(['PATIENT_GUID','YEAR','WEEK']).agg(
    events=('EVENT_DATE','count'), unique_cats=('CATEGORY','nunique')).reset_index()

cluster_feats['OBS_CLUSTER_BURST_WEEKS'] = weekly[weekly['events']>=3].groupby('PATIENT_GUID').size()
cluster_feats['OBS_CLUSTER_DIVERSE_WEEKS'] = weekly[weekly['unique_cats']>=2].groupby('PATIENT_GUID').size()
cluster_feats['OBS_CLUSTER_MAX_EVENTS_PER_WEEK'] = weekly.groupby('PATIENT_GUID')['events'].max()

# Haem with other events same week
if len(haem) > 0:
    haem_dates = set(zip(haem['PATIENT_GUID'], haem['EVENT_DATE']))
    obs_lab_cl['HAS_HAEM'] = list(zip(obs_lab_cl['PATIENT_GUID'], obs_lab_cl['EVENT_DATE']))
    obs_lab_cl['HAS_HAEM'] = obs_lab_cl['HAS_HAEM'].isin(haem_dates).astype(int)
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
frailty = obs_data[obs_data['CATEGORY']=='FRAILTY']
if len(frailty) > 0:
    for pat, cname in [('Mild','NOBS_FRAILTY_MILD'),('Moderate','NOBS_FRAILTY_MODERATE'),('Severe','NOBS_FRAILTY_SEVERE')]:
        matched = frailty[frailty['TERM'].str.contains(pat, case=False, na=False)]
        new_obs_feats[cname] = new_obs_feats.index.isin(set(matched['PATIENT_GUID'].unique())).astype(int)
    new_obs_feats['NOBS_FRAILTY_SEVERITY'] = (
        new_obs_feats.get('NOBS_FRAILTY_MILD',0).fillna(0)*1 +
        new_obs_feats.get('NOBS_FRAILTY_MODERATE',0).fillna(0)*2 +
        new_obs_feats.get('NOBS_FRAILTY_SEVERE',0).fillna(0)*3)

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

# Time between event pairs
def time_between(df, cat1, cat2, name):
    d1 = df[df['CATEGORY']==cat1].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    d2 = df[df['CATEGORY']==cat2].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    m = pd.DataFrame({'d1':d1,'d2':d2}).dropna()
    m[name] = (m['d2']-m['d1']).dt.days
    return m[[name]]

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

def sc(pid, sets): return sum(1 for s in sets if pid in s)

syn_feats['SYN_BLEEDING_SCORE'] = syn_feats.index.map(lambda x: sc(x,[s_haem,low_hb,low_fer,s_andx,s_iron_rx,s_tranex]))
syn_feats['SYN_BLEEDING_HIGH'] = (syn_feats['SYN_BLEEDING_SCORE']>=3).astype(int)
syn_feats['SYN_CONSTITUTIONAL_SCORE'] = syn_feats.index.map(lambda x: sc(x,[s_wl,s_fat,s_app,s_ns]))
syn_feats['SYN_CONSTITUTIONAL_HIGH'] = (syn_feats['SYN_CONSTITUTIONAL_SCORE']>=2).astype(int)
syn_feats['SYN_PATHWAY_SCORE'] = syn_feats.index.map(lambda x: sc(x,[s_haem,s_ui,s_img,s_cath,s_luts]))
syn_feats['SYN_PAIN_SCORE'] = syn_feats.index.map(lambda x: sc(x,[s_bp,s_lp,s_sp,s_opioid]))
syn_feats['SYN_VTE_SCORE'] = syn_feats.index.map(lambda x: sc(x,[s_dvt,s_pe]))
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
syn_feats['SYN_FRAILTY_AND_CONSTITUTIONAL'] = (syn_feats.index.isin(s_frail)&(syn_feats.index.map(lambda x:sc(x,[s_wl,s_fat,s_app,s_ns]))>=1)).astype(int)

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

obs_windowed = obs_data[obs_data['TIME_WINDOW'].isin(['A','B'])].copy()

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
fm = fm.join(new_cats_df[['DIV_NEW_CATS_IN_LATE']], how='left')

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
gap_cv = all_sorted.groupby('PATIENT_GUID')['GAP'].agg(
    lambda x: x.std()/x.mean() if x.mean() > 0 and len(x) >= 2 else 0)
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
# FILL NaN AND CLEAN UP
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CLEANING UP")
print("=" * 70)

meta_cols = ['CANCER_ID', 'INDEX_DATE']
meta = fm[meta_cols].copy()
fm.drop(columns=meta_cols, inplace=True)

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
    'RAPIDLY_DECLINING'
]

for col in fm.columns:
    if col == 'LABEL':
        continue
    if '_MONTHS' in col or '_DAYS' in col:
        continue
    if any(p in col for p in count_flag_patterns):
        fm[col] = fm[col].fillna(0)

fm.replace([np.inf, -np.inf], np.nan, inplace=True)


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════���════════════════════════════════════════════════════
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