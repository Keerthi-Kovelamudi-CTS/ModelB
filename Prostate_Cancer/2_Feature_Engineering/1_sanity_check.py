# ═══════════════════════════════════════════════════════════════
# SANITY CHECK + PATIENT OVERLAP FIX — Config-driven
# Step 0-2: Load → validate → fix overlaps → dedup → save
#
# Must run BEFORE clean_data.py and pipeline.py
# Reads raw windowed CSVs, writes *_dropped.csv (cleaned)
# ═══════════════════════════════════════════════════════════════

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# STEP 1: SANITY CHECKS
# ═══════════════════════════════════════════════════════════════

def sanity_check(clin_df, med_df, window_name):
    """Run data quality checks for one window. Read-only — no mutations."""

    logger.info(f"\n{'='*70}")
    logger.info(f"  SANITY CHECKS - {window_name}")
    logger.info(f"{'='*70}")

    # 1a. Row counts per label
    logger.info(f"\n-- 1a. ROW COUNTS PER LABEL --")
    if clin_df is not None:
        logger.info(f"Clinical:\n{clin_df['LABEL'].value_counts().to_string()}")
    if med_df is not None:
        logger.info(f"Medication:\n{med_df['LABEL'].value_counts().to_string()}")

    # 1b. Unique patient counts
    logger.info(f"\n-- 1b. UNIQUE PATIENTS PER LABEL --")
    if clin_df is not None:
        cp = clin_df.groupby('LABEL')['PATIENT_GUID'].nunique()
        logger.info(f"Clinical: Pos={cp.get(1,0):,} | Neg={cp.get(0,0):,} | "
                     f"Ratio={cp.get(0,0)/max(cp.get(1,0),1):.1f}:1")
    if med_df is not None:
        mp = med_df.groupby('LABEL')['PATIENT_GUID'].nunique()
        logger.info(f"Medication: Pos={mp.get(1,0):,} | Neg={mp.get(0,0):,} | "
                     f"Ratio={mp.get(0,0)/max(mp.get(1,0),1):.1f}:1")

    # 1c. Date ranges
    logger.info(f"\n-- 1c. DATE RANGES --")
    if clin_df is not None:
        clin_df['EVENT_DATE'] = pd.to_datetime(clin_df['EVENT_DATE'], errors='coerce')
        clin_df['INDEX_DATE'] = pd.to_datetime(clin_df['INDEX_DATE'], errors='coerce')
        logger.info(f"Clinical events: {clin_df['EVENT_DATE'].min()} -> {clin_df['EVENT_DATE'].max()}")
        logger.info(f"Clinical index:  {clin_df['INDEX_DATE'].min()} -> {clin_df['INDEX_DATE'].max()}")
    if med_df is not None:
        med_df['EVENT_DATE'] = pd.to_datetime(med_df['EVENT_DATE'], errors='coerce')
        med_df['INDEX_DATE'] = pd.to_datetime(med_df['INDEX_DATE'], errors='coerce')
        logger.info(f"Med events:      {med_df['EVENT_DATE'].min()} -> {med_df['EVENT_DATE'].max()}")

    # 1d. Category distribution
    logger.info(f"\n-- 1d. CATEGORY DISTRIBUTION --")
    if clin_df is not None:
        logger.info(f"Clinical top 20 categories:\n{clin_df['CATEGORY'].value_counts().head(20).to_string()}")
    if med_df is not None:
        logger.info(f"Medication top 20 categories:\n{med_df['CATEGORY'].value_counts().head(20).to_string()}")

    # 1e. Null audit
    logger.info(f"\n-- 1e. NULL AUDIT --")
    if clin_df is not None:
        nulls = clin_df.isnull().sum()
        nulls = nulls[nulls > 0]
        if len(nulls) > 0:
            for col, cnt in nulls.items():
                logger.info(f"  Clinical {col}: {cnt:,} ({cnt/len(clin_df)*100:.1f}%)")
        else:
            logger.info(f"  Clinical: No nulls")

    # 1f. Time window distribution
    logger.info(f"\n-- 1f. TIME WINDOW DISTRIBUTION --")
    if clin_df is not None:
        logger.info(f"Clinical:\n{clin_df.groupby(['LABEL','TIME_WINDOW']).size().unstack(fill_value=0).to_string()}")

    # 1g. Months before index
    logger.info(f"\n-- 1g. MONTHS BEFORE INDEX --")
    if clin_df is not None and 'MONTHS_BEFORE_INDEX' in clin_df.columns:
        logger.info(f"Clinical:\n{clin_df['MONTHS_BEFORE_INDEX'].describe().to_string()}")
        oor = clin_df[(clin_df['MONTHS_BEFORE_INDEX'] < 0) | (clin_df['MONTHS_BEFORE_INDEX'] > 40)]
        if len(oor) > 0:
            logger.warning(f"  {len(oor)} rows outside expected range!")
        else:
            logger.info(f"  All within expected range")


# ═══════════════════════════════════════════════════════════════
# STEP 2: PATIENT-LEVEL FIXES
# ═══════════════════════════════════════════════════════════════

def patient_fixes(clin_df, med_df, window_name, cfg):
    """Fix overlaps, cross-file issues, dedup, sex/age filter."""

    logger.info(f"\n{'='*70}")
    logger.info(f"  PATIENT-LEVEL FIXES - {window_name}")
    logger.info(f"{'='*70}")

    # ── 2a. Patient overlap check (same patient in pos AND neg) ──
    logger.info(f"\n-- 2a. PATIENT OVERLAP CHECK --")
    if clin_df is not None:
        pos_clin = set(clin_df[clin_df['LABEL'] == 1]['PATIENT_GUID'].unique())
        neg_clin = set(clin_df[clin_df['LABEL'] == 0]['PATIENT_GUID'].unique())
        overlap_clin = pos_clin & neg_clin
        logger.info(f"  Clinical: Pos={len(pos_clin):,} | Neg={len(neg_clin):,} | Overlap={len(overlap_clin)}")
        if overlap_clin:
            logger.warning(f"  OVERLAP FOUND! Removing {len(overlap_clin)} patients from negative cohort...")
            clin_df = clin_df[~((clin_df['LABEL'] == 0) & (clin_df['PATIENT_GUID'].isin(overlap_clin)))]
            logger.info(f"  Removed. New clinical rows: {len(clin_df):,}")

    if med_df is not None:
        pos_med = set(med_df[med_df['LABEL'] == 1]['PATIENT_GUID'].unique())
        neg_med = set(med_df[med_df['LABEL'] == 0]['PATIENT_GUID'].unique())
        overlap_med = pos_med & neg_med
        logger.info(f"  Medication: Pos={len(pos_med):,} | Neg={len(neg_med):,} | Overlap={len(overlap_med)}")
        if overlap_med:
            logger.warning(f"  OVERLAP FOUND! Removing {len(overlap_med)} patients from negative cohort...")
            med_df = med_df[~((med_df['LABEL'] == 0) & (med_df['PATIENT_GUID'].isin(overlap_med)))]
            logger.info(f"  Removed. New med rows: {len(med_df):,}")

    # ── 2b. Cross-file patient check ──
    logger.info(f"\n-- 2b. CROSS-FILE PATIENT CHECK --")
    if clin_df is not None and med_df is not None:
        all_clin = set(clin_df['PATIENT_GUID'].unique())
        all_med = set(med_df['PATIENT_GUID'].unique())
        clin_only = all_clin - all_med
        med_only = all_med - all_clin
        in_both = all_clin & all_med

        logger.info(f"  Clinical only (no meds): {len(clin_only):,} - KEEP")
        logger.info(f"  Meds only (no clinical): {len(med_only):,} - REMOVE")
        logger.info(f"  In both:                 {len(in_both):,} - KEEP")

        if med_only:
            med_df = med_df[~med_df['PATIENT_GUID'].isin(med_only)]
            logger.info(f"  Removed med-only patients. New med rows: {len(med_df):,}")

    # ── 2c. Dedup rows ──
    logger.info(f"\n-- 2c. DEDUP ROWS --")
    dedup_cols = ['PATIENT_GUID', 'EVENT_DATE', 'CODE_ID', 'CATEGORY', 'LABEL']
    if clin_df is not None:
        before = len(clin_df)
        available_cols = [c for c in dedup_cols if c in clin_df.columns]
        clin_df = clin_df.drop_duplicates(subset=available_cols)
        logger.info(f"  Clinical: {before:,} -> {len(clin_df):,} ({before - len(clin_df):,} dupes removed)")

    if med_df is not None:
        before = len(med_df)
        available_cols = [c for c in dedup_cols if c in med_df.columns]
        med_df = med_df.drop_duplicates(subset=available_cols)
        logger.info(f"  Medication: {before:,} -> {len(med_df):,} ({before - len(med_df):,} dupes removed)")

    # ── 2d. Sex/age verification ──
    logger.info(f"\n-- 2d. SEX/AGE VERIFICATION --")
    if clin_df is not None:
        sex_dist = clin_df['SEX'].value_counts()
        logger.info(f"  SEX distribution: {sex_dist.to_dict()}")

        # Prostate = male only
        if cfg.CANCER_NAME == 'prostate':
            non_male = clin_df[clin_df['SEX'] != 'M']
            if len(non_male) > 0:
                n_patients = non_male['PATIENT_GUID'].nunique()
                logger.warning(f"  Removing {n_patients} non-male patients (prostate cancer)")
                remove_guids = non_male['PATIENT_GUID'].unique()
                clin_df = clin_df[~clin_df['PATIENT_GUID'].isin(remove_guids)]
                if med_df is not None:
                    med_df = med_df[~med_df['PATIENT_GUID'].isin(remove_guids)]
            else:
                logger.info(f"  All male")

        # Remove under-18
        under18 = clin_df[clin_df['AGE_AT_INDEX'] < 18]['PATIENT_GUID'].unique()
        if len(under18) > 0:
            logger.warning(f"  Removing {len(under18)} patients under 18")
            clin_df = clin_df[~clin_df['PATIENT_GUID'].isin(under18)]
            if med_df is not None:
                med_df = med_df[~med_df['PATIENT_GUID'].isin(under18)]
        else:
            logger.info(f"  All 18+")

    # ── 2e. Final counts ──
    logger.info(f"\n-- 2e. FINAL PATIENT COUNTS --")
    master_patients = None
    if clin_df is not None:
        final = clin_df.groupby('LABEL')['PATIENT_GUID'].nunique()
        logger.info(f"  Clinical: Pos={final.get(1,0):,} | Neg={final.get(0,0):,} | "
                     f"Ratio={final.get(0,0)/max(final.get(1,0),1):.1f}:1")

        # Master patient list
        master_patients = (clin_df.sort_values('AGE_AT_INDEX', ascending=False)
                          .drop_duplicates(subset=['PATIENT_GUID'], keep='first')
                          [['PATIENT_GUID', 'LABEL', 'SEX', 'AGE_AT_INDEX', 'INDEX_DATE']])
        logger.info(f"  Master patient list: {len(master_patients):,} patients")

    return clin_df, med_df, master_patients


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_sanity_check(cfg=None):
    """Run sanity check + patient fixes for all windows, save cleaned data."""
    if cfg is None:
        cfg = config

    cfg.SANITY_RESULTS.mkdir(parents=True, exist_ok=True)

    # Load raw data
    # NOTE: Update these filename patterns to match your raw CSV naming convention
    data = {}
    for window in cfg.WINDOWS:
        data[window] = {}
        data_dir = cfg.BASE_PATH / window
        # Try common naming patterns
        for obs_pattern in [
            f'{cfg.DATA_PREFIX}_{window}_obs.csv',
            f'FE_{cfg.DATA_PREFIX}_clinical_windowed_{window.replace("mo","m")}.csv',
        ]:
            obs_path = data_dir / obs_pattern
            if obs_path.exists():
                data[window]['obs'] = pd.read_csv(obs_path, low_memory=False)
                data[window]['obs'].columns = data[window]['obs'].columns.str.upper()
                logger.info(f"Loaded {window} obs: {data[window]['obs'].shape[0]:,} rows from {obs_path.name}")
                break
        else:
            logger.warning(f"  {window} obs: NOT FOUND in {data_dir}")
            data[window]['obs'] = None

        for med_pattern in [
            f'{cfg.DATA_PREFIX}_{window}_med.csv',
            f'FE_{cfg.DATA_PREFIX}_med_windowed_{window.replace("mo","m")}.csv',
        ]:
            med_path = data_dir / med_pattern
            if med_path.exists():
                data[window]['med'] = pd.read_csv(med_path, low_memory=False)
                data[window]['med'].columns = data[window]['med'].columns.str.upper()
                logger.info(f"Loaded {window} med: {data[window]['med'].shape[0]:,} rows from {med_path.name}")
                break
        else:
            logger.warning(f"  {window} med: NOT FOUND in {data_dir}")
            data[window]['med'] = None

    # Run checks + fixes per window
    for window in cfg.WINDOWS:
        clin = data[window].get('obs')
        med = data[window].get('med')
        if clin is None and med is None:
            continue

        sanity_check(clin, med, window.upper())
        clin, med, master = patient_fixes(clin, med, window.upper(), cfg)
        data[window]['obs'] = clin
        data[window]['med'] = med

        # Save cleaned data as *_dropped.csv (input for clean_data.py)
        data_dir = cfg.BASE_PATH / window
        if clin is not None:
            out = data_dir / f'{cfg.DATA_PREFIX}_{window}_obs_dropped.csv'
            clin.to_csv(out, index=False)
            logger.info(f"  Saved: {out.name} ({len(clin):,} rows)")
        if med is not None:
            out = data_dir / f'{cfg.DATA_PREFIX}_{window}_med_dropped.csv'
            med.to_csv(out, index=False)
            logger.info(f"  Saved: {out.name} ({len(med):,} rows)")
        if master is not None:
            out = cfg.SANITY_RESULTS / f'master_patients_{window}.csv'
            master.to_csv(out, index=False)
            logger.info(f"  Saved: {out.name} ({len(master):,} patients)")

    logger.info(f"\n  SANITY CHECK + PATIENT FIXES COMPLETE")
    logger.info(f"  Output: *_dropped.csv ready for clean_data.py")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_sanity_check()
