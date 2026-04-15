# ═══════════════════════════════════════════════════════════════
# LYMPHOMA — PREPROCESS RAW 300K DATA TO FE FORMAT
# Transforms raw Snowflake extract into the format expected by
# the Lymphoma Feature Engineering pipeline (4_Feature_engineering_holdout.py)
#
# Input:  300K_NonCancer_Patients.csv (shared with Leukaemia — raw SQL extract)
# Output: fe_holdout_workspace/fe_input/{3mo,6mo,12mo}/lymphoma_{window}_obs_dropped.csv
#         fe_holdout_workspace/fe_input/{3mo,6mo,12mo}/lymphoma_{window}_med_dropped.csv
#
# Alignment with training:
#   - Filters to ONLY the 315 Lymphoma approved codes (204 obs + 111 med)
#   - Maps CODE_ID → CATEGORY using Lymphoma training mapping
#   - Windows into A/B using same month ranges as training
#   - Patients with 0 approved codes get placeholder rows
#     (demographics only → all-zero features → low risk prediction)
#   - ALL 300K patients get a prediction score
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent

# Auto-detect 300K non-cancer dump (shared across cancers)
_RAW_CANDIDATES = [
    Path.home() / "pipeline" / "300K_raw.csv",                   # gcloud VM
    SCRIPT_DIR / "300K_NonCancer_Patients.csv",                   # local copy in same dir
    Path("/Users/keerthikovelamudi/Documents/Model B/Bladder_3/Leukaemia_Cancer/4_ExpandedData_Test/300K_NonCancer_Patients.csv"),  # Mac shared
]
RAW_FILE = next((p for p in _RAW_CANDIDATES if p.exists()), _RAW_CANDIDATES[-1])
MAPPING_FILE = SCRIPT_DIR / 'code_category_mapping.json'
EXCLUDE_FILE = SCRIPT_DIR.parent / '2_Feature_Engineering' / 'data' / 'unique_patient_guids_all_data.csv'
OUTPUT_DIR = SCRIPT_DIR / 'fe_holdout_workspace' / 'fe_input'

# Time window definitions (from training data)
WINDOW_DEFS = {
    '3mo': {'B': (3, 14), 'A': (15, 27)},
    '6mo': {'B': (6, 17), 'A': (18, 30)},
    '12mo': {'B': (12, 23), 'A': (24, 36)},
}

INDEX_DATE = pd.Timestamp('2026-02-25')  # Same as training longterm_mh_end
CHUNK_SIZE = 500_000


def load_mapping():
    """Load CODE_ID → CATEGORY mapping from training data."""
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
    """
    Process a chunk of raw data:
    1. Exclude training patients
    2. Unify CODE_ID
    3. Filter to approved codes ONLY
    4. Map CATEGORY
    5. Calculate time fields
    """
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

    # For medications: use DRUG_TERM as TERM
    df.loc[med_mask, 'TERM'] = df.loc[med_mask, 'DRUG_TERM']

    # Collect patient demographics (for ALL patients, before filtering)
    patient_chunk = df[['PATIENT_GUID', 'SEX', 'DATE_OF_BIRTH']].drop_duplicates('PATIENT_GUID')
    for _, row in patient_chunk.iterrows():
        pid = row['PATIENT_GUID']
        if pid not in all_patient_info:
            all_patient_info[pid] = {
                'SEX': row['SEX'],
                'DATE_OF_BIRTH': row['DATE_OF_BIRTH'],
            }

    # FILTER: Keep only approved codes (matches training exactly)
    df_obs = df[obs_mask & df['CODE_ID'].isin(obs_codes)]
    df_med = df[med_mask & df['CODE_ID'].isin(med_codes)]
    df = pd.concat([df_obs, df_med], ignore_index=True)

    if len(df) == 0:
        return df.head(0)  # Return empty with correct schema

    # Map CATEGORY
    obs_in = df['EVENT_TYPE'] == 'OBSERVATION'
    med_in = df['EVENT_TYPE'] == 'MEDICATION'
    df.loc[obs_in, 'CATEGORY'] = df.loc[obs_in, 'CODE_ID'].map(obs_map)
    df.loc[med_in, 'CATEGORY'] = df.loc[med_in, 'CODE_ID'].map(med_map)

    # Assign INDEX_DATE
    df['INDEX_DATE'] = INDEX_DATE

    # Calculate AGE_AT_INDEX
    df['AGE_AT_INDEX'] = ((INDEX_DATE - df['DATE_OF_BIRTH']).dt.days / 365.25).astype(int)

    # Calculate MONTHS_BEFORE_INDEX
    df['MONTHS_BEFORE_INDEX'] = (
        (INDEX_DATE.year - df['EVENT_DATE'].dt.year) * 12 +
        (INDEX_DATE.month - df['EVENT_DATE'].dt.month)
    ).astype(int)

    # Keep only events within valid window range (3-36 months)
    df = df[(df['MONTHS_BEFORE_INDEX'] >= 3) & (df['MONTHS_BEFORE_INDEX'] <= 36)]

    # Add missing columns
    df['CANCER_ID'] = None
    df['DATE_OF_DIAGNOSIS'] = None
    df['AGE_AT_DIAGNOSIS'] = None
    df['LABEL'] = 0
    df['TIME_WINDOW'] = None

    # Clean VALUE
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

    output_cols = [
        'PATIENT_GUID', 'SEX', 'CANCER_ID', 'DATE_OF_DIAGNOSIS', 'INDEX_DATE',
        'AGE_AT_DIAGNOSIS', 'AGE_AT_INDEX', 'EVENT_DATE', 'EVENT_AGE',
        'MONTHS_BEFORE_INDEX', 'TIME_WINDOW', 'EVENT_TYPE', 'CATEGORY',
        'CODE_ID', 'TERM', 'ASSOCIATED_TEXT', 'VALUE', 'LABEL'
    ]

    return df[output_cols]


def create_placeholder_rows(patients_with_events, all_patient_info):
    """
    Create placeholder rows for patients with ZERO approved codes.
    These patients need to exist in the feature matrix (all-zero features)
    so they get a prediction score (low probability).
    """
    patients_without = set(all_patient_info.keys()) - patients_with_events

    if len(patients_without) == 0:
        return pd.DataFrame()

    placeholder_rows = []
    for pid in patients_without:
        info = all_patient_info[pid]
        dob = info['DATE_OF_BIRTH']
        age = int((INDEX_DATE - dob).days / 365.25) if pd.notna(dob) else 65

        placeholder_rows.append({
            'PATIENT_GUID': pid,
            'SEX': info['SEX'],
            'CANCER_ID': None,
            'DATE_OF_DIAGNOSIS': None,
            'INDEX_DATE': INDEX_DATE,
            'AGE_AT_DIAGNOSIS': None,
            'AGE_AT_INDEX': age,
            'EVENT_DATE': INDEX_DATE - pd.DateOffset(months=12),  # Dummy event 12 months before
            'EVENT_AGE': age - 1,
            'MONTHS_BEFORE_INDEX': 12,
            'TIME_WINDOW': None,  # Will be assigned in windowing
            'EVENT_TYPE': 'OBSERVATION',
            'CATEGORY': 'PLACEHOLDER',  # FE will ignore unknown categories
            'CODE_ID': '0',
            'TERM': 'No approved codes found',
            'ASSOCIATED_TEXT': None,
            'VALUE': None,
            'LABEL': 0,
        })

    return pd.DataFrame(placeholder_rows)


def window_and_save(all_data, all_patient_info):
    """Split data into 3mo/6mo/12mo windows with A/B time windows and save."""

    for window_name, window_def in WINDOW_DEFS.items():
        suffix = {'3mo': '3m', '6mo': '6m', '12mo': '12m'}[window_name]
        out_dir = OUTPUT_DIR / window_name
        out_dir.mkdir(parents=True, exist_ok=True)

        b_min, b_max = window_def['B']
        a_min, a_max = window_def['A']

        # Filter to this window's range
        window_data = all_data[
            (all_data['MONTHS_BEFORE_INDEX'] >= b_min) &
            (all_data['MONTHS_BEFORE_INDEX'] <= a_max)
        ].copy()

        # Assign TIME_WINDOW
        window_data.loc[
            (window_data['MONTHS_BEFORE_INDEX'] >= b_min) &
            (window_data['MONTHS_BEFORE_INDEX'] <= b_max), 'TIME_WINDOW'
        ] = 'B'
        window_data.loc[
            (window_data['MONTHS_BEFORE_INDEX'] >= a_min) &
            (window_data['MONTHS_BEFORE_INDEX'] <= a_max), 'TIME_WINDOW'
        ] = 'A'

        window_data = window_data[window_data['TIME_WINDOW'].isin(['A', 'B'])]

        # Add placeholder rows for patients missing from this window
        patients_in_window = set(window_data['PATIENT_GUID'].unique())
        all_pids = set(all_patient_info.keys())
        missing_pids = all_pids - patients_in_window

        if len(missing_pids) > 0:
            placeholders = []
            for pid in missing_pids:
                info = all_patient_info[pid]
                dob = info['DATE_OF_BIRTH']
                age = int((INDEX_DATE - dob).days / 365.25) if pd.notna(dob) else 65
                mid_month = (b_min + b_max) // 2  # Place in middle of B window

                placeholders.append({
                    'PATIENT_GUID': pid,
                    'SEX': info['SEX'],
                    'CANCER_ID': None,
                    'DATE_OF_DIAGNOSIS': None,
                    'INDEX_DATE': INDEX_DATE,
                    'AGE_AT_DIAGNOSIS': None,
                    'AGE_AT_INDEX': age,
                    'EVENT_DATE': INDEX_DATE - pd.DateOffset(months=mid_month),
                    'EVENT_AGE': age,
                    'MONTHS_BEFORE_INDEX': mid_month,
                    'TIME_WINDOW': 'B',
                    'EVENT_TYPE': 'OBSERVATION',
                    'CATEGORY': 'PLACEHOLDER',
                    'CODE_ID': '0',
                    'TERM': 'No approved codes found',
                    'ASSOCIATED_TEXT': None,
                    'VALUE': None,
                    'LABEL': 0,
                })

            placeholder_df = pd.DataFrame(placeholders)
            window_data = pd.concat([window_data, placeholder_df], ignore_index=True)

        # Split into clinical and medication
        clin_data = window_data[window_data['EVENT_TYPE'] == 'OBSERVATION']
        med_data = window_data[window_data['EVENT_TYPE'] == 'MEDICATION']

        # Save (names match Lymphoma training FE inputs)
        clin_path = out_dir / f'lymphoma_{window_name}_obs_dropped.csv'
        med_path = out_dir / f'lymphoma_{window_name}_med_dropped.csv'

        clin_data.to_csv(clin_path, index=False)
        med_data.to_csv(med_path, index=False)

        n_real = len(patients_in_window)
        n_placeholder = len(missing_pids)
        n_total = window_data['PATIENT_GUID'].nunique()

        print(f"  {window_name}:")
        print(f"    Clinical: {len(clin_data):,} rows | Medication: {len(med_data):,} rows")
        print(f"    Patients with events: {n_real:,} | Placeholder (zero features): {n_placeholder:,} | Total: {n_total:,}")
        print(f"    Saved: {clin_path.name}, {med_path.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'═'*60}")
    print(f"  PREPROCESS RAW DATA → FE FORMAT")
    print(f"  Input: {RAW_FILE.name}")
    print(f"  Index date: {INDEX_DATE.date()}")
    print(f"{'═'*60}")

    # Load mapping
    print(f"\n  Loading category mapping...")
    obs_map, med_map = load_mapping()
    obs_codes = set(obs_map.keys())
    med_codes = set(med_map.keys())

    # Load patients to exclude
    print(f"\n  Loading patients to exclude...")
    exclude_pids = load_exclude_patients()

    # Process in chunks
    print(f"\n  Processing {RAW_FILE.name} in chunks of {CHUNK_SIZE:,}...")
    print(f"  Filtering to {len(obs_codes)} obs + {len(med_codes)} med approved codes")
    print(f"  Excluding {len(exclude_pids):,} training/test patients")

    all_chunks = []
    all_patient_info = {}  # pid → {SEX, DATE_OF_BIRTH}
    total_rows = 0
    kept_rows = 0

    for i, chunk in enumerate(pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
        total_rows += len(chunk)
        processed = process_chunk(chunk, obs_codes, med_codes, obs_map, med_map, exclude_pids, all_patient_info)
        kept_rows += len(processed)
        if len(processed) > 0:
            all_chunks.append(processed)

        if (i + 1) % 20 == 0:
            print(f"    Chunk {i+1}: {total_rows:,} rows processed → {kept_rows:,} kept | {len(all_patient_info):,} patients seen")

    print(f"\n  Processing complete:")
    print(f"    Total raw rows: {total_rows:,}")
    print(f"    Kept (approved codes): {kept_rows:,} ({kept_rows/total_rows*100:.1f}%)")
    print(f"    Total patients: {len(all_patient_info):,}")

    # Combine
    print(f"\n  Combining chunks...")
    if all_chunks:
        all_data = pd.concat(all_chunks, ignore_index=True)
    else:
        all_data = pd.DataFrame()

    patients_with_events = set(all_data['PATIENT_GUID'].unique()) if len(all_data) > 0 else set()
    patients_without = set(all_patient_info.keys()) - patients_with_events

    print(f"  Events data: {len(all_data):,} rows, {len(patients_with_events):,} patients")
    print(f"  Patients with 0 approved codes: {len(patients_without):,} (will get all-zero features)")

    # Save zero-code patient list — predict_unseen.py uses this to suppress predictions
    zero_code_path = OUTPUT_DIR / 'zero_code_patients.csv'
    pd.DataFrame({'PATIENT_GUID': sorted(patients_without)}).to_csv(zero_code_path, index=False)
    print(f"  Saved zero-code patient list: {zero_code_path} ({len(patients_without):,} patients)")

    print(f"\n  Category distribution:")
    if len(all_data) > 0:
        print(all_data['CATEGORY'].value_counts().to_string())

    # Window and save (includes placeholder rows for zero-code patients)
    print(f"\n  Windowing and saving...")
    window_and_save(all_data, all_patient_info)

    print(f"\n{'═'*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'═'*60}")
    print(f"  Total patients: {len(all_patient_info):,}")
    print(f"    With approved codes: {len(patients_with_events):,} ({len(patients_with_events)/len(all_patient_info)*100:.1f}%)")
    print(f"    Zero codes (placeholder): {len(patients_without):,} ({len(patients_without)/len(all_patient_info)*100:.1f}%)")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. python3 4_Feature_engineering_holdout.py")
    print(f"       → reads fe_holdout_workspace/fe_input/ and writes fe_holdout_workspace/fe_results/FE/")
    print(f"    2. python3 ../3_Modeling/predict_unseen.py --window 3mo \\")
    print(f"         --feature_matrix fe_holdout_workspace/fe_results/FE/3mo/feature_matrix_final_3mo.csv")
    print(f"    3. python3 ../3_Modeling/predict_combined_evaluation.py --window 3mo \\")
    print(f"         --unseen_predictions predictions_unseen.csv")
    print(f"{'═'*60}")
