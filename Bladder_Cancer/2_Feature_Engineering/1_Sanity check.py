"""
SANITY CHECKS — Run BEFORE feature engineering
Inputs: data/{3mo|6mo|12mo|12mo_250k}/ (3mo/6mo: _3m/_6m; 12mo: no suffix; 12mo_250k: _12m_250k)
Outputs: sanity_check_results/{window}/
Usage: python "1_Sanity check.py" [--window 3mo|6mo|12mo|12mo_250k]  (default: 12mo_250k)
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Sanity checks for FE inputs.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
args = parser.parse_args()
WINDOW = args.window

DATA_DIR = os.path.join(SCRIPT_DIR, 'data', WINDOW)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'sanity_check_results', WINDOW)
os.makedirs(RESULTS_DIR, exist_ok=True)

if WINDOW == '12mo':
    CLINICAL_FILE = os.path.join(DATA_DIR, 'FE_bladder_clinical_windowed.csv')
    MEDS_FILE = os.path.join(DATA_DIR, 'FE_bladder_med_windowed.csv')
elif WINDOW == '12mo_250k':
    CLINICAL_FILE = os.path.join(DATA_DIR, 'FE_bladder_clinical_windowed_12m_250k.csv')
    MEDS_FILE = os.path.join(DATA_DIR, 'FE_bladder_med_windowed_12m_250k.csv')
else:
    RAW_SUFFIX = '3m' if WINDOW == '3mo' else '6m'
    CLINICAL_FILE = os.path.join(DATA_DIR, f'FE_bladder_clinical_windowed_{RAW_SUFFIX}.csv')
    MEDS_FILE = os.path.join(DATA_DIR, f'FE_bladder_med_windowed_{RAW_SUFFIX}.csv')


class Tee:
    """Write all print output to both console and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


def main():
    log_path = os.path.join(RESULTS_DIR, f'sanity_check_{WINDOW}.log')
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        run_checks(WINDOW)
    finally:
        sys.stdout = tee.terminal
        tee.close()
    print(f"\nResults written to: {RESULTS_DIR}")
    print(f"Log file: sanity_check_{WINDOW}.log")


def run_checks(window):
    # Load
    if not os.path.isfile(CLINICAL_FILE):
        raise FileNotFoundError(f"Clinical file not found: {CLINICAL_FILE}")
    if not os.path.isfile(MEDS_FILE):
        raise FileNotFoundError(f"Meds file not found: {MEDS_FILE}")

    clinical = pd.read_csv(CLINICAL_FILE, low_memory=False)
    meds = pd.read_csv(MEDS_FILE, low_memory=False)

    print("=" * 60)
    print("SANITY CHECKS — Feature Engineering Inputs")
    print("=" * 60)
    print(f"Input folder: {DATA_DIR}")
    print(f"Results folder: {RESULTS_DIR}")
    print(f"Clinical: {os.path.basename(CLINICAL_FILE)}")
    print(f"Meds:    {os.path.basename(MEDS_FILE)}")
    print()
    print("=" * 60)
    print("FILE SIZES")
    print("=" * 60)
    print(f"Clinical: {len(clinical):,} rows, {clinical.shape[1]} columns")
    print(f"Meds:     {len(meds):,} rows, {meds.shape[1]} columns")

    # Normalize column names to uppercase for checks
    clinical.columns = [str(c).strip().upper() for c in clinical.columns]
    meds.columns = [str(c).strip().upper() for c in meds.columns]

    # CHECK 1: Columns present
    print("\n" + "=" * 60)
    print("CHECK 1: COLUMNS")
    print("=" * 60)
    print(f"Clinical columns: {list(clinical.columns)}")
    print(f"Meds columns:     {list(meds.columns)}")

    expected_clinical = ['PATIENT_GUID', 'SEX', 'CANCER_ID', 'DATE_OF_DIAGNOSIS',
        'INDEX_DATE', 'AGE_AT_DIAGNOSIS', 'AGE_AT_INDEX', 'EVENT_DATE',
        'EVENT_AGE', 'MONTHS_BEFORE_INDEX', 'TIME_WINDOW', 'EVENT_TYPE',
        'CATEGORY', 'SNOMED_ID', 'TERM', 'ASSOCIATED_TEXT', 'VALUE', 'LABEL']
    expected_meds = ['PATIENT_GUID', 'SEX', 'CANCER_ID', 'DATE_OF_DIAGNOSIS',
        'INDEX_DATE', 'AGE_AT_DIAGNOSIS', 'AGE_AT_INDEX', 'EVENT_DATE',
        'EVENT_AGE', 'MONTHS_BEFORE_INDEX', 'TIME_WINDOW', 'EVENT_TYPE',
        'CATEGORY', 'DMD_CODE', 'TERM', 'ASSOCIATED_TEXT', 'VALUE',
        'DURATION_IN_DAYS', 'LABEL']

    clin_cols = set(clinical.columns)
    med_cols = set(meds.columns)
    missing_clin = [c for c in expected_clinical if c not in clin_cols]
    missing_meds = [c for c in expected_meds if c not in med_cols]
    check1_ok = not missing_clin and not missing_meds
    print(f"\nMissing clinical cols: {missing_clin if missing_clin else 'NONE ✅'}")
    print(f"Missing meds cols:     {missing_meds if missing_meds else 'NONE ✅'}")

    # CHECK 2: Patient counts per label
    print("\n" + "=" * 60)
    print("CHECK 2: PATIENT COUNTS PER LABEL")
    print("=" * 60)
    for name, df in [('Clinical', clinical), ('Meds', meds)]:
        summary = df.groupby('LABEL').agg(
            unique_patients=('PATIENT_GUID', 'nunique'),
            total_rows=('PATIENT_GUID', 'count')
        )
        print(f"\n{name}:")
        print(summary)

    # CHECK 3: Ratio check
    print("\n" + "=" * 60)
    print("CHECK 3: CANCER TO NON-CANCER RATIO")
    print("=" * 60)
    clin_cancer = clinical[clinical['LABEL'] == 1]['PATIENT_GUID'].nunique()
    clin_non = clinical[clinical['LABEL'] == 0]['PATIENT_GUID'].nunique()
    med_cancer = meds[meds['LABEL'] == 1]['PATIENT_GUID'].nunique()
    med_non = meds[meds['LABEL'] == 0]['PATIENT_GUID'].nunique()
    print(f"Clinical ratio: 1:{clin_non/max(clin_cancer,1):.1f}  (cancer={clin_cancer}, non-cancer={clin_non})")
    print(f"Meds ratio:     1:{med_non/max(med_cancer,1):.1f}  (cancer={med_cancer}, non-cancer={med_non})")

    # CHECK 4: Patient overlap
    print("\n" + "=" * 60)
    print("CHECK 4: PATIENT OVERLAP BETWEEN QUERIES")
    print("=" * 60)
    clin_neg_patients = set(clinical[clinical['LABEL'] == 0]['PATIENT_GUID'].unique())
    med_neg_patients = set(meds[meds['LABEL'] == 0]['PATIENT_GUID'].unique())
    clin_pos_patients = set(clinical[clinical['LABEL'] == 1]['PATIENT_GUID'].unique())
    med_pos_patients = set(meds[meds['LABEL'] == 1]['PATIENT_GUID'].unique())
    med_neg_in_clin = med_neg_patients.issubset(clin_neg_patients)
    med_pos_in_clin = med_pos_patients.issubset(clin_pos_patients)
    check4_ok = med_neg_in_clin and med_pos_in_clin
    print(f"Non-cancer: Meds patients ⊂ Clinical patients? {med_neg_in_clin}")
    print(f"Cancer:     Meds patients ⊂ Clinical patients? {med_pos_in_clin}")
    print(f"\nNon-cancer patients ONLY in meds (not clinical): {len(med_neg_patients - clin_neg_patients)}")
    print(f"Cancer patients ONLY in meds (not clinical):     {len(med_pos_patients - clin_pos_patients)}")

    # CHECK 5: Event types
    print("\n" + "=" * 60)
    print("CHECK 5: EVENT TYPES BREAKDOWN")
    print("=" * 60)
    if 'EVENT_TYPE' in clinical.columns:
        print("\nClinical event types:")
        print(clinical.groupby(['LABEL', 'EVENT_TYPE']).size().unstack(fill_value=0))
    if 'EVENT_TYPE' in meds.columns:
        print("\nMeds event types:")
        print(meds.groupby(['LABEL', 'EVENT_TYPE']).size().unstack(fill_value=0))

    # CHECK 6: Time windows
    print("\n" + "=" * 60)
    print("CHECK 6: TIME WINDOWS")
    print("=" * 60)
    if 'TIME_WINDOW' in clinical.columns:
        print("\nClinical time windows:")
        print(clinical.groupby(['LABEL', 'TIME_WINDOW']).size().unstack(fill_value=0))
    if 'TIME_WINDOW' in meds.columns:
        print("\nMeds time windows:")
        print(meds.groupby(['LABEL', 'TIME_WINDOW']).size().unstack(fill_value=0))

    # CHECK 7: MONTHS_BEFORE_INDEX range
    print("\n" + "=" * 60)
    print("CHECK 7: MONTHS_BEFORE_INDEX RANGE")
    print("=" * 60)
    for name, df in [('Clinical', clinical), ('Meds', meds)]:
        if 'MONTHS_BEFORE_INDEX' not in df.columns:
            continue
        print(f"\n{name}:")
        for lbl in [0, 1]:
            subset = df[df['LABEL'] == lbl]['MONTHS_BEFORE_INDEX']
            if len(subset) > 0:
                print(f"  Label {lbl}: min={subset.min()}, max={subset.max()}, mean={subset.mean():.1f}")

    # CHECK 8: Sex distribution
    print("\n" + "=" * 60)
    print("CHECK 8: SEX DISTRIBUTION (patients)")
    print("=" * 60)
    for name, df in [('Clinical', clinical), ('Meds', meds)]:
        print(f"\n{name}:")
        sex_dist = df.groupby(['LABEL', 'SEX'])['PATIENT_GUID'].nunique().unstack(fill_value=0)
        print(sex_dist)

    # CHECK 9: Age distribution
    print("\n" + "=" * 60)
    print("CHECK 9: AGE AT INDEX DISTRIBUTION")
    print("=" * 60)
    for name, df in [('Clinical', clinical), ('Meds', meds)]:
        if 'AGE_AT_INDEX' not in df.columns:
            continue
        print(f"\n{name}:")
        for lbl in [0, 1]:
            sub = df[df['LABEL'] == lbl].drop_duplicates('PATIENT_GUID')['AGE_AT_INDEX']
            if len(sub) > 0:
                print(f"  Label {lbl}: min={sub.min()}, max={sub.max()}, mean={sub.mean():.1f}, median={sub.median():.1f}")

    # CHECK 10: Top categories
    print("\n" + "=" * 60)
    print("CHECK 10: TOP 10 CATEGORIES BY ROWS")
    print("=" * 60)
    if 'CATEGORY' in clinical.columns:
        print("\nClinical — Cancer (label=1):")
        print(clinical[clinical['LABEL'] == 1]['CATEGORY'].value_counts().head(10))
        print("\nClinical — Non-cancer (label=0):")
        print(clinical[clinical['LABEL'] == 0]['CATEGORY'].value_counts().head(10))
    if 'CATEGORY' in meds.columns:
        print("\nMeds — Cancer (label=1):")
        print(meds[meds['LABEL'] == 1]['CATEGORY'].value_counts().head(10))
        print("\nMeds — Non-cancer (label=0):")
        print(meds[meds['LABEL'] == 0]['CATEGORY'].value_counts().head(10))

    # SUMMARY
    print("\n" + "=" * 60)
    print("SUMMARY — WHAT TO LOOK FOR")
    print("=" * 60)
    print("""
✅ GOOD if:
  - Expected columns present (no missing)
  - Non-cancer ratio in reasonable range (e.g. ~1:10)
  - Meds patients are SUBSET of clinical patients
  - MONTHS_BEFORE_INDEX in expected range (e.g. 0–60)
  - Age/sex distribution similar between cancer and non-cancer
  - No excessive NULL time_windows

⚠️ INVESTIGATE if:
  - Missing required columns
  - Ratio much higher or lower than expected
  - Meds patients NOT subset of clinical
  - Large age/sex mismatch between cohorts
  - MONTHS_BEFORE_INDEX outside expected range
""")

    # Write summary CSV to sanity_check_results
    summary_rows = [
        {'check': '1_columns_clinical', 'passed': check1_ok, 'detail': 'NONE' if not missing_clin else str(missing_clin)},
        {'check': '1_columns_meds', 'passed': not missing_meds, 'detail': 'NONE' if not missing_meds else str(missing_meds)},
        {'check': '2_file_sizes', 'passed': True, 'detail': f"clinical={len(clinical):,} rows, meds={len(meds):,} rows"},
        {'check': '3_ratio_clinical', 'passed': True, 'detail': f"1:{clin_non/max(clin_cancer,1):.1f}"},
        {'check': '3_ratio_meds', 'passed': True, 'detail': f"1:{med_non/max(med_cancer,1):.1f}"},
        {'check': '4_meds_subset_of_clinical', 'passed': check4_ok, 'detail': f"neg_ok={med_neg_in_clin}, pos_ok={med_pos_in_clin}"},
    ]
    summary_path = os.path.join(RESULTS_DIR, f'sanity_check_summary_{window}.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary CSV: {os.path.basename(summary_path)}")


if __name__ == '__main__':
    main()
