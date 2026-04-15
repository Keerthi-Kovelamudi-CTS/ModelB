# ═══════════════════════════════════════════════════════════════
# DATA CLEANING — Config-driven lab range clamping & date parsing
# Reads *_dropped.csv, cleans, overwrites in place.
# ═══════════════════════════════════════════════════════════════

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def clean_obs(df, cfg):
    """Clean observations dataframe: dates, numeric VALUE, lab outlier clamping."""
    logger.info(f"  Cleaning observations... ({len(df):,} rows)")

    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')

    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    outliers_removed = 0
    for category, terms in cfg.LAB_RANGES.items():
        for term, (low, high) in terms.items():
            mask = (df['CATEGORY'] == category) & (df['TERM'] == term) & df['VALUE'].notna()
            oor = mask & ((df['VALUE'] < low) | (df['VALUE'] > high))
            outliers_removed += int(oor.sum())
            df.loc[oor, 'VALUE'] = np.nan
    logger.info(f"  Lab outliers set to NaN: {outliers_removed:,}")

    df['TIME_WINDOW_NUM'] = df['TIME_WINDOW'].map({'A': 0, 'B': 1})
    df['MONTHS_BEFORE_INDEX'] = pd.to_numeric(df['MONTHS_BEFORE_INDEX'], errors='coerce')

    logger.info(f"  Final rows: {len(df):,}")
    return df


def clean_med(df, cfg):
    """Clean medications dataframe: dates, impossible quantity clamping."""
    logger.info(f"  Cleaning medications... ({len(df):,} rows)")

    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')

    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    impossible = df['VALUE'].notna() & ((df['VALUE'] < 0) | (df['VALUE'] > 10000))
    logger.info(f"  Impossible quantities set to NaN: {int(impossible.sum()):,}")
    df.loc[impossible, 'VALUE'] = np.nan

    if 'DURATION_IN_DAYS' in df.columns:
        null_pct = df['DURATION_IN_DAYS'].isnull().mean()
        if null_pct > 0.95:
            df = df.drop(columns=['DURATION_IN_DAYS'])
            logger.info(f"  Dropped DURATION_IN_DAYS ({null_pct*100:.0f}% null)")

    df['TIME_WINDOW_NUM'] = df['TIME_WINDOW'].map({'A': 0, 'B': 1})
    df['MONTHS_BEFORE_INDEX'] = pd.to_numeric(df['MONTHS_BEFORE_INDEX'], errors='coerce')

    logger.info(f"  Final rows: {len(df):,}")
    return df


def run_clean_data(cfg=None):
    """Run data cleaning for all windows."""
    if cfg is None:
        cfg = config

    for window in cfg.WINDOWS:
        data_dir = cfg.BASE_PATH / window
        obs_path = data_dir / f'{cfg.DATA_PREFIX}_{window}_obs_dropped.csv'
        med_path = data_dir / f'{cfg.DATA_PREFIX}_{window}_med_dropped.csv'

        if not obs_path.exists() or not med_path.exists():
            logger.warning(f"Skipping {window}: missing {obs_path.name} or {med_path.name}")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"  CLEAN DATA - {window}")
        logger.info(f"{'='*60}")

        obs = pd.read_csv(obs_path, low_memory=False)
        med = pd.read_csv(med_path, low_memory=False)

        obs = clean_obs(obs, cfg)
        med = clean_med(med, cfg)

        obs.to_csv(obs_path, index=False)
        med.to_csv(med_path, index=False)
        logger.info(f"  Saved: {obs_path.name}, {med_path.name}")

    logger.info(f"  CLEAN DATA COMPLETE")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_clean_data()
