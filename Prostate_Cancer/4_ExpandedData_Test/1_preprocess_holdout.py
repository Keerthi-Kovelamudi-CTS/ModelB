# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PREPROCESS RAW HOLDOUT DATA TO FE FORMAT
# Transforms raw Snowflake/BigQuery extract into the format
# expected by the FE pipeline.
#
# Input:  data/raw/<raw_holdout_file>.csv
# Output: data/fe_input/{3mo,6mo,12mo}/prostate_{window}_{obs,med}_dropped.csv
#
# Uses code_category_mapping.json (from training) to align
# CODE_ID → CATEGORY mapping with training data.
# ═══════════════════════════════════════════════════════════════

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '2_Feature_Engineering'))
import config as fe_config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MAPPING_FILE = SCRIPT_DIR / 'code_category_mapping.json'
EXCLUDE_FILE = fe_config.BASE_PATH / 'unique_patient_guids_all_data.csv'
RAW_DIR = SCRIPT_DIR / 'data' / 'raw'
OUTPUT_DIR = SCRIPT_DIR / 'data' / 'fe_input'

# Time window definitions (must match training SQL)
WINDOW_DEFS = {
    '3mo': {'B': (3, 14), 'A': (15, 27)},
    '6mo': {'B': (6, 17), 'A': (18, 30)},
    '12mo': {'B': (12, 23), 'A': (24, 36)},
}

# UPDATE this to match your training index date
INDEX_DATE = pd.Timestamp('2026-02-25')


def load_mapping():
    """Load CODE_ID -> CATEGORY mapping from training data."""
    if not MAPPING_FILE.exists():
        logger.error(f"Missing {MAPPING_FILE} — generate from training data first")
        return {}, {}
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    obs_map = {str(k): v for k, v in mapping.get('obs', {}).items()}
    med_map = {str(k): v for k, v in mapping.get('med', {}).items()}
    logger.info(f"  Loaded mapping: {len(obs_map)} obs codes, {len(med_map)} med codes")
    return obs_map, med_map


def load_exclude_patients():
    """Load patient GUIDs from training data to exclude from holdout."""
    if not EXCLUDE_FILE.exists():
        logger.warning(f"  No exclusion file at {EXCLUDE_FILE}")
        return set()
    df = pd.read_csv(EXCLUDE_FILE)
    col = df.columns[0]
    guids = set(df[col].unique())
    logger.info(f"  Loaded {len(guids):,} training patients to exclude")
    return guids


def run_preprocess():
    """Preprocess raw holdout data into FE-ready format."""

    logger.info(f"{'='*70}")
    logger.info(f"  PREPROCESS HOLDOUT DATA - {fe_config.CANCER_NAME.upper()}")
    logger.info(f"{'='*70}")

    # Find raw file
    raw_files = list(RAW_DIR.glob('*.csv'))
    if not raw_files:
        logger.error(f"No CSV files found in {RAW_DIR}")
        return
    raw_file = raw_files[0]
    logger.info(f"  Raw file: {raw_file.name}")

    obs_map, med_map = load_mapping()
    exclude = load_exclude_patients()

    # Load raw data
    df = pd.read_csv(raw_file, low_memory=False)
    df.columns = df.columns.str.upper()
    logger.info(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Exclude training patients
    if exclude:
        before = df['PATIENT_GUID'].nunique()
        df = df[~df['PATIENT_GUID'].isin(exclude)]
        after = df['PATIENT_GUID'].nunique()
        logger.info(f"  Excluded training patients: {before:,} -> {after:,}")

    # Map CODE_ID to CATEGORY
    df['CODE_ID_STR'] = df['CODE_ID'].astype(str)
    df['CATEGORY'] = None
    df['EVENT_TYPE'] = None

    obs_mask = df['CODE_ID_STR'].isin(obs_map)
    med_mask = df['CODE_ID_STR'].isin(med_map)

    df.loc[obs_mask, 'CATEGORY'] = df.loc[obs_mask, 'CODE_ID_STR'].map(obs_map)
    df.loc[obs_mask, 'EVENT_TYPE'] = 'observation'
    df.loc[med_mask, 'CATEGORY'] = df.loc[med_mask, 'CODE_ID_STR'].map(med_map)
    df.loc[med_mask, 'EVENT_TYPE'] = 'medication'

    # Keep only mapped rows
    df = df[df['CATEGORY'].notna()].copy()
    logger.info(f"  After mapping: {df.shape[0]:,} rows ({df['PATIENT_GUID'].nunique():,} patients)")

    # Set index date and calculate months
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = INDEX_DATE
    df['MONTHS_BEFORE_INDEX'] = ((INDEX_DATE - df['EVENT_DATE']).dt.days / 30.44).round(1)

    # Set label = 0 for all holdout (non-cancer)
    df['LABEL'] = 0

    # Window and save
    for window, ranges in WINDOW_DEFS.items():
        out_dir = OUTPUT_DIR / window
        out_dir.mkdir(parents=True, exist_ok=True)

        window_df = df.copy()
        b_range = ranges['B']
        a_range = ranges['A']

        window_df['TIME_WINDOW'] = None
        b_mask = (window_df['MONTHS_BEFORE_INDEX'] >= b_range[0]) & (window_df['MONTHS_BEFORE_INDEX'] < b_range[1])
        a_mask = (window_df['MONTHS_BEFORE_INDEX'] >= a_range[0]) & (window_df['MONTHS_BEFORE_INDEX'] < a_range[1])
        window_df.loc[b_mask, 'TIME_WINDOW'] = 'B'
        window_df.loc[a_mask, 'TIME_WINDOW'] = 'A'
        window_df = window_df[window_df['TIME_WINDOW'].notna()]

        obs_df = window_df[window_df['EVENT_TYPE'] == 'observation']
        med_df = window_df[window_df['EVENT_TYPE'] == 'medication']

        obs_path = out_dir / f'{fe_config.DATA_PREFIX}_{window}_obs_dropped.csv'
        med_path = out_dir / f'{fe_config.DATA_PREFIX}_{window}_med_dropped.csv'
        obs_df.to_csv(obs_path, index=False)
        med_df.to_csv(med_path, index=False)
        logger.info(f"  {window}: obs={len(obs_df):,} med={len(med_df):,} patients={window_df['PATIENT_GUID'].nunique():,}")

    logger.info(f"\n  PREPROCESS COMPLETE")
    logger.info(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_preprocess()
