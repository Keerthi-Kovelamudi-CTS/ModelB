import os
import pandas as pd

# Paths: same data/ folder as other FE scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
CLINICAL_FILE = os.path.join(DATA_DIR, 'FE_bladder_clinical_windowed.csv')
MEDS_FILE = os.path.join(DATA_DIR, 'FE_bladder_med_windowed.csv')
OUT_CLINICAL = os.path.join(DATA_DIR, 'FE_bladder_dropped_patients_clinical_windowed.csv')
OUT_MEDS = os.path.join(DATA_DIR, 'FE_bladder_dropped_patients_med_windowed.csv')

clinical = pd.read_csv(CLINICAL_FILE, low_memory=False)
meds = pd.read_csv(MEDS_FILE, low_memory=False)

# Drop 109 meds-only patients
clinical_patients = set(clinical['PATIENT_GUID'].unique())
meds_only = set(meds['PATIENT_GUID'].unique()) - clinical_patients
print(f"Dropping {len(meds_only)} meds-only patients")
meds = meds[meds['PATIENT_GUID'].isin(clinical_patients)]

print(f"Clinical: {clinical['PATIENT_GUID'].nunique()} patients")
print(f"Meds:     {meds['PATIENT_GUID'].nunique()} patients")
print(f"Meds ⊂ Clinical? {set(meds['PATIENT_GUID'].unique()).issubset(clinical_patients)}")

# Write back to data/ with dropped-patient names
clinical.to_csv(OUT_CLINICAL, index=False)
meds.to_csv(OUT_MEDS, index=False)
print(f"\nSaved to {DATA_DIR}:")
print(f"  {os.path.basename(OUT_CLINICAL)}")
print(f"  {os.path.basename(OUT_MEDS)}")