"""
Patient overlap check: clinical vs meds cohorts.
Inputs: data/{3mo|6mo|12mo|12mo_250k}/ (3mo/6mo: _3m/_6m; 12mo: no suffix; 12mo_250k: _12m_250k)
Outputs: patientoverlapresults/{window}/
Usage: python 2_Patientoverlapcheck.py [--window 3mo|6mo|12mo|12mo_250k]  (default: 12mo_250k)
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Patient overlap check.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
args = parser.parse_args()
WINDOW = args.window

DATA_DIR = os.path.join(SCRIPT_DIR, 'data', WINDOW)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'patientoverlapresults', WINDOW)
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
    if not os.path.isfile(CLINICAL_FILE):
        raise FileNotFoundError(f"Clinical file not found: {CLINICAL_FILE}")
    if not os.path.isfile(MEDS_FILE):
        raise FileNotFoundError(f"Meds file not found: {MEDS_FILE}")

    clinical = pd.read_csv(CLINICAL_FILE, low_memory=False)
    meds = pd.read_csv(MEDS_FILE, low_memory=False)

    clinical.columns = [str(c).strip().upper() for c in clinical.columns]
    meds.columns = [str(c).strip().upper() for c in meds.columns]

    log_path = os.path.join(RESULTS_DIR, f'patientoverlap_{WINDOW}.log')
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        run_checks(clinical, meds, WINDOW)
    finally:
        sys.stdout = tee.terminal
        tee.close()
    print(f"\nResults written to: {RESULTS_DIR}")
    print(f"Log file: patientoverlap_{WINDOW}.log")


def run_checks(clinical, meds, window):
    # CANCER PATIENTS (LABEL=1)
    clin_pos = set(clinical[clinical['LABEL'] == 1]['PATIENT_GUID'].unique())
    med_pos = set(meds[meds['LABEL'] == 1]['PATIENT_GUID'].unique())

    print("=" * 60)
    print("CANCER PATIENTS (LABEL=1)")
    print("=" * 60)
    print(f"In clinical:        {len(clin_pos)}")
    print(f"In meds:            {len(med_pos)}")
    print(f"In BOTH:            {len(clin_pos & med_pos)}")
    print(f"Only in clinical:   {len(clin_pos - med_pos)}")
    print(f"Only in meds:       {len(med_pos - clin_pos)}")
    print(f"Meds ⊂ Clinical?    {med_pos.issubset(clin_pos)}")

    # NON-CANCER PATIENTS (LABEL=0)
    clin_neg = set(clinical[clinical['LABEL'] == 0]['PATIENT_GUID'].unique())
    med_neg = set(meds[meds['LABEL'] == 0]['PATIENT_GUID'].unique())

    print("\n" + "=" * 60)
    print("NON-CANCER PATIENTS (LABEL=0)")
    print("=" * 60)
    print(f"In clinical:        {len(clin_neg)}")
    print(f"In meds:            {len(med_neg)}")
    print(f"In BOTH:            {len(clin_neg & med_neg)}")
    print(f"Only in clinical:   {len(clin_neg - med_neg)}")
    print(f"Only in meds:       {len(med_neg - clin_neg)}")
    print(f"Meds ⊂ Clinical?    {med_neg.issubset(clin_neg)}")

    # MASTER PATIENT LIST
    all_clin = clin_pos | clin_neg
    all_med = med_pos | med_neg

    print("\n" + "=" * 60)
    print("ALL PATIENTS (BOTH LABELS)")
    print("=" * 60)
    print(f"In clinical:        {len(all_clin)}")
    print(f"In meds:            {len(all_med)}")
    print(f"In BOTH:            {len(all_clin & all_med)}")
    print(f"Only in clinical:   {len(all_clin - all_med)}")
    print(f"Only in meds:       {len(all_med - all_clin)}")
    print(f"Meds ⊂ Clinical?    {all_med.issubset(all_clin)}")

    # WHAT TO LOOK FOR
    print("\n" + "=" * 60)
    print("WHAT TO LOOK FOR")
    print("=" * 60)
    print("""
✅ GOOD if:
  - "In BOTH" is high for meds patients
  - "Only in clinical" is large (most patients don't have these meds)
  - "Only in meds" is small (< 100)
  - Meds ⊂ Clinical ideally True, but False is OK if "only in meds" is small

⚠️ INVESTIGATE if:
  - "Only in meds" is large (> 500)
  - That would mean the sampling selected different patients
""")

    # Summary CSV
    summary = pd.DataFrame([
        {'cohort': 'cancer', 'in_clinical': len(clin_pos), 'in_meds': len(med_pos), 'in_both': len(clin_pos & med_pos), 'only_in_meds': len(med_pos - clin_pos), 'meds_subset_of_clinical': med_pos.issubset(clin_pos)},
        {'cohort': 'non_cancer', 'in_clinical': len(clin_neg), 'in_meds': len(med_neg), 'in_both': len(clin_neg & med_neg), 'only_in_meds': len(med_neg - clin_neg), 'meds_subset_of_clinical': med_neg.issubset(clin_neg)},
        {'cohort': 'all', 'in_clinical': len(all_clin), 'in_meds': len(all_med), 'in_both': len(all_clin & all_med), 'only_in_meds': len(all_med - all_clin), 'meds_subset_of_clinical': all_med.issubset(all_clin)},
    ])
    summary_path = os.path.join(RESULTS_DIR, f'patientoverlap_summary_{window}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary CSV: {os.path.basename(summary_path)}")


if __name__ == '__main__':
    main()
