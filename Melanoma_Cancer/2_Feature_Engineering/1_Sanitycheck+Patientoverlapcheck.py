# ══════════════════════════════════════════════════════════════
# MELANOMA CANCER — FEATURE ENGINEERING PIPELINE
# STEP 0: Load Data
# STEP 1: Sanity Checks
# STEP 2: Patient-Level Fixes
# ═══════════════════════════════════════════════════════════════

import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Output: same prints to console and to results/sanity_check/sanity_check_melanoma.log
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'sanity_check'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / 'sanity_check_melanoma.log'


class Tee:
    """Write prints to both console and log file."""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, 'w', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()


# ═══════════════════════════════════════════════════════════════
# STEP 0: LOAD DATA
# ═══════════════════════════════════════════════════════════════

BASE_PATH = SCRIPT_DIR / 'data'
BASE_PATH = str(BASE_PATH)

# File paths
files = {
    '3mo': {
        'clinical': f"{BASE_PATH}/3mo/FE_mel_obs_windowed_3m.csv",
        'med':      f"{BASE_PATH}/3mo/FE_mel_med_windowed_3m.csv",
    },
    '6mo': {
        'clinical': f"{BASE_PATH}/6mo/FE_mel_obs_windowed_6m.csv",
        'med':      f"{BASE_PATH}/6mo/FE_mel_med_windowed_6m.csv",
    },
    '12mo': {
        'clinical': f"{BASE_PATH}/12mo/FE_mel_obs_windowed_12m.csv",
        'med':      f"{BASE_PATH}/12mo/FE_mel_med_windowed_12m.csv",
    },
}

# Load all 6 files (all prints below also go to OUTPUT_FILE)
sys.stdout = Tee(OUTPUT_FILE)
try:
    print(f"Sanity check + Patient overlap check — Melanoma pipeline")
    print(f"Log: {OUTPUT_FILE}\n")

    data = {}
    for window, paths in files.items():
        data[window] = {}
        for dtype, path in paths.items():
            try:
                df = pd.read_csv(path, low_memory=False)
                data[window][dtype] = df
                print(f"✅ {window} {dtype}: {df.shape[0]:,} rows × {df.shape[1]} cols")
            except FileNotFoundError:
                print(f"❌ FILE NOT FOUND: {path}")
                data[window][dtype] = None

    print("\n" + "═"*70)

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: SANITY CHECKS — Run for each window
    # ═══════════════════════════════════════════════════════════════

    def sanity_check(clin_df, med_df, window_name):
        """Complete sanity checks for one window."""

        print(f"\n{'═'*70}")
        print(f"  SANITY CHECKS — {window_name} WINDOW")
        print(f"{'═'*70}")

        # ── 1a. Row counts per label ──
        print(f"\n── 1a. ROW COUNTS PER LABEL ──")
        if clin_df is not None:
            print(f"\nClinical:")
            print(clin_df['LABEL'].value_counts().to_string())
        if med_df is not None:
            print(f"\nMedication:")
            print(med_df['LABEL'].value_counts().to_string())

        # ── 1b. Unique patient counts per label ──
        print(f"\n── 1b. UNIQUE PATIENTS PER LABEL ──")
        if clin_df is not None:
            clin_patients = clin_df.groupby('LABEL')['PATIENT_GUID'].nunique()
            print(f"\nClinical:")
            print(clin_patients.to_string())
            print(f"  Ratio (neg:pos): {clin_patients.get(0,0) / max(clin_patients.get(1,0),1):.1f}:1")
        if med_df is not None:
            med_patients = med_df.groupby('LABEL')['PATIENT_GUID'].nunique()
            print(f"\nMedication:")
            print(med_patients.to_string())
            print(f"  Ratio (neg:pos): {med_patients.get(0,0) / max(med_patients.get(1,0),1):.1f}:1")

        # ── 1c. Date range checks ──
        print(f"\n── 1c. DATE RANGES ──")
        if clin_df is not None:
            clin_df['EVENT_DATE'] = pd.to_datetime(clin_df['EVENT_DATE'], errors='coerce')
            clin_df['INDEX_DATE'] = pd.to_datetime(clin_df['INDEX_DATE'], errors='coerce')
            print(f"\nClinical event dates: {clin_df['EVENT_DATE'].min()} → {clin_df['EVENT_DATE'].max()}")
            print(f"Clinical index dates: {clin_df['INDEX_DATE'].min()} → {clin_df['INDEX_DATE'].max()}")
        if med_df is not None:
            med_df['EVENT_DATE'] = pd.to_datetime(med_df['EVENT_DATE'], errors='coerce')
            med_df['INDEX_DATE'] = pd.to_datetime(med_df['INDEX_DATE'], errors='coerce')
            print(f"Med event dates:      {med_df['EVENT_DATE'].min()} → {med_df['EVENT_DATE'].max()}")
            print(f"Med index dates:      {med_df['INDEX_DATE'].min()} → {med_df['INDEX_DATE'].max()}")

        # ── 1d. Event type / category distribution ──
        print(f"\n── 1d. CATEGORY DISTRIBUTION ──")
        if clin_df is not None:
            print(f"\nClinical — EVENT_TYPE counts:")
            print(clin_df['EVENT_TYPE'].value_counts().to_string())
            print(f"\nClinical — Top 20 CATEGORY counts:")
            print(clin_df['CATEGORY'].value_counts().head(20).to_string())
        if med_df is not None:
            print(f"\nMedication — Top 20 CATEGORY counts:")
            print(med_df['CATEGORY'].value_counts().head(20).to_string())

        # ── 1e. Null/missing value audit ──
        print(f"\n── 1e. NULL VALUE AUDIT ──")
        if clin_df is not None:
            clin_nulls = clin_df.isnull().sum()
            clin_nulls = clin_nulls[clin_nulls > 0]
            if len(clin_nulls) > 0:
                print(f"\nClinical nulls:")
                for col, cnt in clin_nulls.items():
                    print(f"  {col}: {cnt:,} ({cnt/len(clin_df)*100:.1f}%)")
            else:
                print(f"\nClinical: No nulls ✅")
        if med_df is not None:
            med_nulls = med_df.isnull().sum()
            med_nulls = med_nulls[med_nulls > 0]
            if len(med_nulls) > 0:
                print(f"\nMedication nulls:")
                for col, cnt in med_nulls.items():
                    print(f"  {col}: {cnt:,} ({cnt/len(med_df)*100:.1f}%)")
            else:
                print(f"\nMedication: No nulls ✅")

        # ── 1f. Time window distribution ──
        print(f"\n── 1f. TIME WINDOW DISTRIBUTION ──")
        if clin_df is not None:
            print(f"\nClinical TIME_WINDOW:")
            print(clin_df.groupby(['LABEL','TIME_WINDOW']).size().unstack(fill_value=0).to_string())
        if med_df is not None:
            print(f"\nMedication TIME_WINDOW:")
            print(med_df.groupby(['LABEL','TIME_WINDOW']).size().unstack(fill_value=0).to_string())

        # ── 1g. Months before index distribution ──
        print(f"\n── 1g. MONTHS BEFORE INDEX STATS ──")
        if clin_df is not None:
            print(f"\nClinical MONTHS_BEFORE_INDEX:")
            print(clin_df['MONTHS_BEFORE_INDEX'].describe().to_string())
            out_of_range = clin_df[
                (clin_df['MONTHS_BEFORE_INDEX'] < 0) |
                (clin_df['MONTHS_BEFORE_INDEX'] > 40)
            ]
            if len(out_of_range) > 0:
                print(f"  ⚠️ {len(out_of_range)} rows outside expected range!")
            else:
                print(f"  ✅ All within expected range")

        return clin_df, med_df


    # Run sanity checks for all 3 windows
    for window in ['3mo', '6mo', '12mo']:
        clin = data[window]['clinical']
        med = data[window]['med']
        if clin is not None and med is not None:
            data[window]['clinical'], data[window]['med'] = sanity_check(clin, med, window)


    # ═══════════════════════════════════════════════════════════════
    # STEP 2: PATIENT-LEVEL FIXES — Run for each window
    # ═══════════════════════════════════════════════════════════════

    def patient_fixes(clin_df, med_df, window_name):
        """Patient-level checks and fixes for one window."""

        print(f"\n{'═'*70}")
        print(f"  PATIENT-LEVEL FIXES — {window_name} WINDOW")
        print(f"{'═'*70}")

        # ── 2a. Patient overlap check ──
        print(f"\n── 2a. PATIENT OVERLAP CHECK ──")
        if clin_df is not None:
            pos_patients_clin = set(clin_df[clin_df['LABEL']==1]['PATIENT_GUID'].unique())
            neg_patients_clin = set(clin_df[clin_df['LABEL']==0]['PATIENT_GUID'].unique())
            overlap_clin = pos_patients_clin & neg_patients_clin
            print(f"  Clinical — Positive patients: {len(pos_patients_clin):,}")
            print(f"  Clinical — Negative patients: {len(neg_patients_clin):,}")
            print(f"  Clinical — Overlap: {len(overlap_clin)}")
            if len(overlap_clin) > 0:
                print(f"  ❌ OVERLAP FOUND! Removing overlapping patients from negative cohort...")
                clin_df = clin_df[~((clin_df['LABEL']==0) & (clin_df['PATIENT_GUID'].isin(overlap_clin)))]
                print(f"  ✅ Removed. New clinical rows: {len(clin_df):,}")
            else:
                print(f"  ✅ No overlap")

        if med_df is not None:
            pos_patients_med = set(med_df[med_df['LABEL']==1]['PATIENT_GUID'].unique())
            neg_patients_med = set(med_df[med_df['LABEL']==0]['PATIENT_GUID'].unique())
            overlap_med = pos_patients_med & neg_patients_med
            print(f"\n  Medication — Positive patients: {len(pos_patients_med):,}")
            print(f"  Medication — Negative patients: {len(neg_patients_med):,}")
            print(f"  Medication — Overlap: {len(overlap_med)}")
            if len(overlap_med) > 0:
                print(f"  ❌ OVERLAP FOUND! Removing overlapping patients from negative cohort...")
                med_df = med_df[~((med_df['LABEL']==0) & (med_df['PATIENT_GUID'].isin(overlap_med)))]
                print(f"  ✅ Removed. New med rows: {len(med_df):,}")
            else:
                print(f"  ✅ No overlap")

        # ── 2b. Cross-file patient check ──
        print(f"\n── 2b. CROSS-FILE PATIENT CHECK ──")
        if clin_df is not None and med_df is not None:
            all_clin_patients = set(clin_df['PATIENT_GUID'].unique())
            all_med_patients = set(med_df['PATIENT_GUID'].unique())

            # Patients in clinical only (no meds) — FINE, keep them
            clin_only = all_clin_patients - all_med_patients
            # Patients in meds only (no clinical) — REMOVE these
            med_only = all_med_patients - all_clin_patients
            # Patients in both — GOOD
            in_both = all_clin_patients & all_med_patients

            print(f"  Patients in clinical only (no meds): {len(clin_only):,} — ✅ KEEP (have clinical signal)")
            print(f"  Patients in meds only (no clinical): {len(med_only):,} — ❌ REMOVE (no clinical signal)")
            print(f"  Patients in both files:              {len(in_both):,} — ✅ KEEP")

            if len(med_only) > 0:
                # Check label distribution of med-only patients
                med_only_labels = med_df[med_df['PATIENT_GUID'].isin(med_only)].groupby('LABEL')['PATIENT_GUID'].nunique()
                print(f"\n  Med-only patients by label:")
                print(f"    Positive (label=1): {med_only_labels.get(1,0)}")
                print(f"    Negative (label=0): {med_only_labels.get(0,0)}")

                # Remove med-only patients from med_df
                med_df = med_df[~med_df['PATIENT_GUID'].isin(med_only)]
                print(f"\n  ✅ Removed med-only patients. New med rows: {len(med_df):,}")

        # ── 2c. Dedup rows ──
        print(f"\n── 2c. DEDUP ROWS ──")
        if clin_df is not None:
            before = len(clin_df)
            clin_df = clin_df.drop_duplicates(
                subset=['PATIENT_GUID', 'EVENT_DATE', 'CODE_ID', 'CATEGORY', 'LABEL']
            )
            after = len(clin_df)
            print(f"  Clinical: {before:,} → {after:,} ({before-after:,} duplicates removed)")

        if med_df is not None:
            before = len(med_df)
            med_df = med_df.drop_duplicates(
                subset=['PATIENT_GUID', 'EVENT_DATE', 'CODE_ID', 'CATEGORY', 'LABEL']
            )
            after = len(med_df)
            print(f"  Medication: {before:,} → {after:,} ({before-after:,} duplicates removed)")

        # ── 2d. Verify age/sex ──
        print(f"\n── 2d. AGE/SEX VERIFICATION ──")
        if clin_df is not None:
            sex_dist = clin_df.groupby('LABEL')['PATIENT_GUID'].apply(
                lambda x: clin_df.loc[x.index, 'SEX'].value_counts().to_dict()
            )
            print(f"  Clinical SEX distribution:")
            print(f"  {clin_df['SEX'].value_counts().to_string()}")
            print(f"  (Both sexes expected for melanoma — no sex filter applied)")

            age_stats = clin_df['AGE_AT_INDEX'].describe()
            print(f"\n  Clinical AGE_AT_INDEX:")
            print(f"  {age_stats.to_string()}")
            under_18 = clin_df[clin_df['AGE_AT_INDEX'] < 18]['PATIENT_GUID'].nunique()
            if under_18 > 0:
                print(f"  ⚠️ {under_18} patients under 18 — removing...")
                remove_guids = set(clin_df[clin_df['AGE_AT_INDEX'] < 18]['PATIENT_GUID'].unique())
                clin_df = clin_df[~clin_df['PATIENT_GUID'].isin(remove_guids)]
                med_df = med_df[~med_df['PATIENT_GUID'].isin(remove_guids)]
            else:
                print(f"  ✅ All 18+")

        # ── 2e. Final patient counts ──
        print(f"\n── 2e. FINAL PATIENT COUNTS ──")
        if clin_df is not None:
            final_clin = clin_df.groupby('LABEL')['PATIENT_GUID'].nunique()
            print(f"  Clinical: Pos={final_clin.get(1,0):,} | Neg={final_clin.get(0,0):,} | Ratio={final_clin.get(0,0)/max(final_clin.get(1,0),1):.1f}:1")
        if med_df is not None:
            final_med = med_df.groupby('LABEL')['PATIENT_GUID'].nunique()
            print(f"  Medication: Pos={final_med.get(1,0):,} | Neg={final_med.get(0,0):,} | Ratio={final_med.get(0,0)/max(final_med.get(1,0),1):.1f}:1")

        # ── 2f. Master patient list ──
        print(f"\n── 2f. MASTER PATIENT LIST ──")
        if clin_df is not None:
            master_patients = clin_df.sort_values('AGE_AT_INDEX', ascending=False).drop_duplicates(subset=['PATIENT_GUID'], keep='first')[['PATIENT_GUID','LABEL','SEX','AGE_AT_INDEX','INDEX_DATE']]
            print(f"  Master patient list: {len(master_patients):,} patients")
            print(f"    Positive: {len(master_patients[master_patients['LABEL']==1]):,}")
            print(f"    Negative: {len(master_patients[master_patients['LABEL']==0]):,}")
        else:
            master_patients = None

        return clin_df, med_df, master_patients


    # Run patient fixes for all 3 windows
    master_patients = {}
    for window in ['3mo', '6mo', '12mo']:
        clin = data[window]['clinical']
        med = data[window]['med']
        if clin is not None and med is not None:
            data[window]['clinical'], data[window]['med'], master_patients[window] = patient_fixes(clin, med, window)


    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL SUMMARY — ALL WINDOWS")
    print(f"{'═'*70}")

    for window in ['3mo', '6mo', '12mo']:
        clin = data[window]['clinical']
        med = data[window]['med']
        mp = master_patients.get(window)

        if clin is not None and med is not None and mp is not None:
            print(f"\n  {window}:")
            print(f"    Clinical rows:  {len(clin):,}")
            print(f"    Med rows:       {len(med):,}")
            print(f"    Total patients: {len(mp):,}")
            print(f"    Positive:       {len(mp[mp['LABEL']==1]):,}")
            print(f"    Negative:       {len(mp[mp['LABEL']==0]):,}")
            print(f"    Ratio:          {len(mp[mp['LABEL']==0])/max(len(mp[mp['LABEL']==1]),1):.1f}:1")

    print(f"\n{'═'*70}")
    print(f"  ✅ STEP 0+1+2 COMPLETE — Ready for STEP 3 (Data Cleaning)")
    print(f"{'═'*70}")

    # ═══════════════════════════════════════════════════════════════
    # SAVE CLEANED DATA TO DISK
    # ═══════════════════════════════════════════════════════════════
    suffix_map = {'3mo': '3m', '6mo': '6m', '12mo': '12m'}

    print(f"\n{'═'*70}")
    print(f"  SAVING CLEANED DATA")
    print(f"{'═'*70}")

    for window in ['3mo', '6mo', '12mo']:
        suffix = suffix_map[window]
        clin = data[window]['clinical']
        med = data[window]['med']
        mp = master_patients.get(window)

        if clin is not None:
            clin_path = f"{BASE_PATH}/{window}/FE_mel_dropped_patients_obs_windowed_{suffix}.csv"
            clin.to_csv(clin_path, index=False)
            print(f"\n  ✅ Saved: {clin_path} ({len(clin):,} rows)")

        if med is not None:
            med_path = f"{BASE_PATH}/{window}/FE_mel_dropped_patients_med_windowed_{suffix}.csv"
            med.to_csv(med_path, index=False)
            print(f"  ✅ Saved: {med_path} ({len(med):,} rows)")

        if mp is not None:
            mp_path = str(OUTPUT_DIR / f"master_patients_{window}.csv")
            mp.to_csv(mp_path, index=False)
            print(f"  ✅ Saved: {mp_path} ({len(mp):,} patients)")

    print(f"\n{'═'*70}")
    print(f"  ✅ ALL FILES SAVED — Pipeline complete")
    print(f"{'═'*70}")

finally:
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = sys.__stdout__

print(f"Output saved to: {OUTPUT_FILE}")
