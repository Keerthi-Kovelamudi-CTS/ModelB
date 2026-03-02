"""
Drop meds-only patients. Inputs from data/{3mo|6mo|12mo}/; outputs to same data/{window}/.
12mo: FE_bladder_clinical_windowed.csv, FE_bladder_med_windowed.csv → FE_bladder_dropped_patients_*.csv (no suffix)
Usage: python 3_droppatients.py [--window 3mo|6mo|12mo]  (default: 12mo)
"""
import os
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Drop meds-only patients.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo'], default='12mo', help='3mo, 6mo, or 12mo window (default: 12mo)')
args = parser.parse_args()
WINDOW = args.window

DATA_DIR = os.path.join(SCRIPT_DIR, 'data', WINDOW)
if WINDOW == '12mo':
    CLINICAL_FILE = os.path.join(DATA_DIR, 'FE_bladder_clinical_windowed.csv')
    MEDS_FILE = os.path.join(DATA_DIR, 'FE_bladder_med_windowed.csv')
    OUT_CLINICAL = os.path.join(DATA_DIR, 'FE_bladder_dropped_patients_clinical_windowed.csv')
    OUT_MEDS = os.path.join(DATA_DIR, 'FE_bladder_dropped_patients_med_windowed.csv')
else:
    RAW_SUFFIX = '3m' if WINDOW == '3mo' else '6m'
    CLINICAL_FILE = os.path.join(DATA_DIR, f'FE_bladder_clinical_windowed_{RAW_SUFFIX}.csv')
    MEDS_FILE = os.path.join(DATA_DIR, f'FE_bladder_med_windowed_{RAW_SUFFIX}.csv')
    OUT_CLINICAL = os.path.join(DATA_DIR, f'FE_bladder_dropped_patients_clinical_windowed_{RAW_SUFFIX}.csv')
    OUT_MEDS = os.path.join(DATA_DIR, f'FE_bladder_dropped_patients_med_windowed_{RAW_SUFFIX}.csv')

clinical = pd.read_csv(CLINICAL_FILE, low_memory=False)
meds = pd.read_csv(MEDS_FILE, low_memory=False)

clinical_patients = set(clinical['PATIENT_GUID'].unique())
meds_only = set(meds['PATIENT_GUID'].unique()) - clinical_patients
print(f"Dropping {len(meds_only)} meds-only patients")
meds = meds[meds['PATIENT_GUID'].isin(clinical_patients)]

print(f"Clinical: {clinical['PATIENT_GUID'].nunique()} patients")
print(f"Meds:     {meds['PATIENT_GUID'].nunique()} patients")
print(f"Meds ⊂ Clinical? {set(meds['PATIENT_GUID'].unique()).issubset(clinical_patients)}")

clinical.to_csv(OUT_CLINICAL, index=False)
meds.to_csv(OUT_MEDS, index=False)
print(f"\nSaved to {DATA_DIR}:")
print(f"  {os.path.basename(OUT_CLINICAL)}")
print(f"  {os.path.basename(OUT_MEDS)}")
