# ═══════════════════════════════════════════════════════════════
# MELANOMA — PREPROCESS RAW 300K DATA TO FE FORMAT
# Transforms raw Snowflake extract into the format expected by
# the Feature Engineering pipeline (4_Feature_engineering.py)
#
# Input:  300K_NonCancer_Patients.csv (raw SQL extract)
#         (same file as Leukaemia — shared 300K non-cancer patients)
# Output: data/{3mo,6mo,12mo}/FE_mel_obs_windowed_{suffix}.csv
#         data/{3mo,6mo,12mo}/FE_mel_med_windowed_{suffix}.csv
#
# Excludes patients already used in Melanoma training/test
# (from unique_patient_guids_all_data.csv)
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent

# Use same 300K file as Leukaemia
RAW_FILE = Path("/Users/keerthikovelamudi/Documents/Model B/Bladder_3/Leukaemia_Cancer/4_ExpandedData_Test/300K_NonCancer_Patients.csv")
MAPPING_FILE = SCRIPT_DIR / 'code_category_mapping.json'
EXCLUDE_FILE = SCRIPT_DIR.parent / '2_Feature_Engineering' / 'data' / 'unique_patient_guids_all_data.csv'
OUTPUT_DIR = SCRIPT_DIR / 'data'

WINDOW_DEFS = {
    '3mo': {'B': (3, 14), 'A': (15, 27)},
    '6mo': {'B': (6, 17), 'A': (18, 30)},
    '12mo': {'B': (12, 23), 'A': (24, 36)},
}

INDEX_DATE = pd.Timestamp('2026-02-25')
CHUNK_SIZE = 500_000


def load_mapping():
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    obs_map = {str(k): v for k, v in mapping['obs'].items()}
    med_map = {str(k): v for k, v in mapping['med'].items()}
    print(f"  Loaded mapping: {len(obs_map)} obs codes, {len(med_map)} med codes")
    return obs_map, med_map


def load_exclude_patients():
    if EXCLUDE_FILE.exists():
        df = pd.read_csv(EXCLUDE_FILE)
        pids = set(df['PATIENT_GUID'].unique())
        print(f"  Loaded {len(pids):,} patients to EXCLUDE (training/test set)")
        return pids
    else:
        print(f"  WARNING: Exclude file not found: {EXCLUDE_FILE}")
        return set()


def process_chunk(chunk, obs_codes, med_codes, obs_map, med_map, exclude_pids, all_patient_info):
    df = chunk.copy()
    df.columns = df.columns.str.strip()

    # Exclude training patients
    df = df[~df['PATIENT_GUID'].isin(exclude_pids)]
    if len(df) == 0:
        return df.head(0)

    # Parse dates
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['DATE_OF_BIRTH'] = pd.to_datetime(df['DATE_OF_BIRTH'], errors='coerce')
    df = df[df['EVENT_DATE'].notna()]

    # Standardize EVENT_TYPE
    df['EVENT_TYPE'] = df['EVENT_TYPE'].str.strip().str.upper()

    # Unify CODE_ID
    obs_mask = df['EVENT_TYPE'] == 'OBSERVATION'
    med_mask = df['EVENT_TYPE'] == 'MEDICATION'

    df['CODE_ID'] = None
    df.loc[obs_mask, 'CODE_ID'] = df.loc[obs_mask, 'SNOMED_C_T_CONCEPT_ID'].astype(str).str.strip().str.replace('.0', '', regex=False)
    df.loc[med_mask, 'CODE_ID'] = df.loc[med_mask, 'MED_CODE_ID'].astype(str).str.strip().str.replace('.0', '', regex=False)
    df.loc[med_mask, 'TERM'] = df.loc[med_mask, 'DRUG_TERM']

    # Collect patient demographics (ALL patients, before code filtering)
    patient_chunk = df[['PATIENT_GUID', 'SEX', 'DATE_OF_BIRTH']].drop_duplicates('PATIENT_GUID')
    for _, row in patient_chunk.iterrows():
        pid = row['PATIENT_GUID']
        if pid not in all_patient_info:
            all_patient_info[pid] = {
                'SEX': row['SEX'],
                'DATE_OF_BIRTH': row['DATE_OF_BIRTH'],
            }

    # Filter to approved codes only
    df_obs = df[obs_mask & df['CODE_ID'].isin(obs_codes)]
    df_med = df[med_mask & df['CODE_ID'].isin(med_codes)]
    df = pd.concat([df_obs, df_med], ignore_index=True)

    if len(df) == 0:
        return df.head(0)

    # Map CATEGORY
    obs_in = df['EVENT_TYPE'] == 'OBSERVATION'
    med_in = df['EVENT_TYPE'] == 'MEDICATION'
    df.loc[obs_in, 'CATEGORY'] = df.loc[obs_in, 'CODE_ID'].map(obs_map)
    df.loc[med_in, 'CATEGORY'] = df.loc[med_in, 'CODE_ID'].map(med_map)

    # Time fields
    df['INDEX_DATE'] = INDEX_DATE
    df['AGE_AT_INDEX'] = ((INDEX_DATE - df['DATE_OF_BIRTH']).dt.days / 365.25).astype(int)
    df['MONTHS_BEFORE_INDEX'] = (
        (INDEX_DATE.year - df['EVENT_DATE'].dt.year) * 12 +
        (INDEX_DATE.month - df['EVENT_DATE'].dt.month)
    ).astype(int)

    df = df[(df['MONTHS_BEFORE_INDEX'] >= 3) & (df['MONTHS_BEFORE_INDEX'] <= 36)]

    df['CANCER_ID'] = None
    df['DATE_OF_DIAGNOSIS'] = None
    df['AGE_AT_DIAGNOSIS'] = None
    df['LABEL'] = 0
    df['TIME_WINDOW'] = None
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

    output_cols = [
        'PATIENT_GUID', 'SEX', 'CANCER_ID', 'DATE_OF_DIAGNOSIS', 'INDEX_DATE',
        'AGE_AT_DIAGNOSIS', 'AGE_AT_INDEX', 'EVENT_DATE', 'EVENT_AGE',
        'MONTHS_BEFORE_INDEX', 'TIME_WINDOW', 'EVENT_TYPE', 'CATEGORY',
        'CODE_ID', 'TERM', 'ASSOCIATED_TEXT', 'VALUE', 'LABEL'
    ]

    return df[output_cols]


def window_and_save(all_data, all_patient_info):
    for window_name, window_def in WINDOW_DEFS.items():
        suffix = {'3mo': '3m', '6mo': '6m', '12mo': '12m'}[window_name]
        out_dir = OUTPUT_DIR / window_name
        out_dir.mkdir(parents=True, exist_ok=True)

        b_min, b_max = window_def['B']
        a_min, a_max = window_def['A']

        window_data = all_data[
            (all_data['MONTHS_BEFORE_INDEX'] >= b_min) &
            (all_data['MONTHS_BEFORE_INDEX'] <= a_max)
        ].copy()

        window_data.loc[
            (window_data['MONTHS_BEFORE_INDEX'] >= b_min) &
            (window_data['MONTHS_BEFORE_INDEX'] <= b_max), 'TIME_WINDOW'
        ] = 'B'
        window_data.loc[
            (window_data['MONTHS_BEFORE_INDEX'] >= a_min) &
            (window_data['MONTHS_BEFORE_INDEX'] <= a_max), 'TIME_WINDOW'
        ] = 'A'

        window_data = window_data[window_data['TIME_WINDOW'].isin(['A', 'B'])]

        # Add placeholder rows for patients with 0 approved codes
        patients_in_window = set(window_data['PATIENT_GUID'].unique())
        all_pids = set(all_patient_info.keys())
        missing_pids = all_pids - patients_in_window

        if len(missing_pids) > 0:
            placeholders = []
            mid_month = (b_min + b_max) // 2
            for pid in missing_pids:
                info = all_patient_info[pid]
                dob = info['DATE_OF_BIRTH']
                age = int((INDEX_DATE - dob).days / 365.25) if pd.notna(dob) else 65
                placeholders.append({
                    'PATIENT_GUID': pid, 'SEX': info['SEX'],
                    'CANCER_ID': None, 'DATE_OF_DIAGNOSIS': None,
                    'INDEX_DATE': INDEX_DATE, 'AGE_AT_DIAGNOSIS': None,
                    'AGE_AT_INDEX': age,
                    'EVENT_DATE': INDEX_DATE - pd.DateOffset(months=mid_month),
                    'EVENT_AGE': age, 'MONTHS_BEFORE_INDEX': mid_month,
                    'TIME_WINDOW': 'B', 'EVENT_TYPE': 'OBSERVATION',
                    'CATEGORY': 'PLACEHOLDER', 'CODE_ID': '0',
                    'TERM': 'No approved codes found',
                    'ASSOCIATED_TEXT': None, 'VALUE': None, 'LABEL': 0,
                })
            window_data = pd.concat([window_data, pd.DataFrame(placeholders)], ignore_index=True)

        # Split obs + med (Melanoma uses "obs" not "clinical")
        obs_data = window_data[window_data['EVENT_TYPE'] == 'OBSERVATION']
        med_data = window_data[window_data['EVENT_TYPE'] == 'MEDICATION']

        obs_path = out_dir / f'FE_mel_obs_windowed_{suffix}.csv'
        med_path = out_dir / f'FE_mel_med_windowed_{suffix}.csv'

        obs_data.to_csv(obs_path, index=False)
        med_data.to_csv(med_path, index=False)

        n_real = len(patients_in_window)
        n_placeholder = len(missing_pids)
        n_total = window_data['PATIENT_GUID'].nunique()

        print(f"  {window_name}:")
        print(f"    Obs: {len(obs_data):,} rows | Med: {len(med_data):,} rows")
        print(f"    Patients with events: {n_real:,} | Placeholder: {n_placeholder:,} | Total: {n_total:,}")
        print(f"    Saved: {obs_path.name}, {med_path.name}")


if __name__ == '__main__':
    print(f"{'═'*60}")
    print(f"  MELANOMA — PREPROCESS RAW DATA → FE FORMAT")
    print(f"  Input: {RAW_FILE.name}")
    print(f"  Index date: {INDEX_DATE.date()}")
    print(f"{'═'*60}")

    print(f"\n  Loading category mapping...")
    obs_map, med_map = load_mapping()
    obs_codes = set(obs_map.keys())
    med_codes = set(med_map.keys())

    print(f"\n  Loading patients to exclude...")
    exclude_pids = load_exclude_patients()

    print(f"\n  Processing {RAW_FILE.name} in chunks of {CHUNK_SIZE:,}...")
    print(f"  Filtering to {len(obs_codes)} obs + {len(med_codes)} med approved codes")
    print(f"  Excluding {len(exclude_pids):,} training/test patients")

    all_chunks = []
    all_patient_info = {}
    total_rows = 0
    kept_rows = 0

    for i, chunk in enumerate(pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
        total_rows += len(chunk)
        processed = process_chunk(chunk, obs_codes, med_codes, obs_map, med_map, exclude_pids, all_patient_info)
        kept_rows += len(processed)
        if len(processed) > 0:
            all_chunks.append(processed)

        if (i + 1) % 20 == 0:
            print(f"    Chunk {i+1}: {total_rows:,} rows → {kept_rows:,} kept | {len(all_patient_info):,} patients")

    print(f"\n  Total: {total_rows:,} raw rows → {kept_rows:,} kept")
    print(f"  Total patients (after excluding training): {len(all_patient_info):,}")

    if all_chunks:
        all_data = pd.concat(all_chunks, ignore_index=True)
    else:
        all_data = pd.DataFrame()

    patients_with_events = set(all_data['PATIENT_GUID'].unique()) if len(all_data) > 0 else set()
    patients_without = set(all_patient_info.keys()) - patients_with_events

    print(f"  With approved codes: {len(patients_with_events):,}")
    print(f"  Zero codes (placeholder): {len(patients_without):,}")

    if len(all_data) > 0:
        print(f"\n  Categories:")
        print(all_data['CATEGORY'].value_counts().to_string())

    print(f"\n  Windowing and saving...")
    window_and_save(all_data, all_patient_info)

    print(f"\n{'═'*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Total patients: {len(all_patient_info):,}")
    print(f"    With approved codes: {len(patients_with_events):,}")
    print(f"    Placeholder: {len(patients_without):,}")
    print(f"{'═'*60}")
