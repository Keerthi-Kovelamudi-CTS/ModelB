# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PIPELINE ORCHESTRATOR
# Runs the full FE pipeline: sanity → clean → base → advanced →
#   mega → cancer-specific → trend → signal → cleanup
#
# Usage:
#   python 0_run_pipeline.py                  # run everything
#   python 0_run_pipeline.py --step sanity    # run one step
#   python 0_run_pipeline.py --step fe        # feature engineering only
# ═══════════════════════════════════════════════════════════════

import argparse
import logging
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

import config
from io_utils import read_table, write_table

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def load_data(cfg):
    """Load obs + med CSVs for all windows."""
    data = {}
    for window in cfg.WINDOWS:
        data[window] = {}
        try:
            obs_path = cfg.BASE_PATH / window / f'{cfg.DATA_PREFIX}_{window}_obs_dropped.parquet'
            med_path = cfg.BASE_PATH / window / f'{cfg.DATA_PREFIX}_{window}_med_dropped.parquet'
            obs = read_table(obs_path, low_memory=False)
            obs.columns = obs.columns.str.upper()
            med = read_table(med_path, low_memory=False)
            med.columns = med.columns.str.upper()
            if 'VALUE' in obs.columns:
                obs['VALUE'] = pd.to_numeric(obs['VALUE'], errors='coerce')
            if 'VALUE' in med.columns:
                med['VALUE'] = pd.to_numeric(med['VALUE'], errors='coerce')
            data[window]['obs'] = obs
            data[window]['med'] = med
            logger.info(f"Loaded {window}: obs {obs.shape[0]:,} rows, med {med.shape[0]:,} rows")
        except FileNotFoundError as e:
            logger.warning(f"Warning {window}: {e}")
            data[window]['obs'] = None
            data[window]['med'] = None
    return data


def run_feature_engineering(cfg):
    """Run the full FE pipeline (steps 4a through 4f) — PER WINDOW.

    Each window: load _dropped → build features through all 6 sub-steps in
    one in-memory pipeline → save final matrix → free everything → next.
    Avoids the OOM that comes from holding 6 windows' DataFrames at once.
    Always re-runs all windows; existing outputs are overwritten.
    """
    import gc
    _pipeline = import_module('3_pipeline')
    _cancer = import_module('4_cancer_features')

    build_clinical_features = _pipeline.build_clinical_features
    build_medication_features = _pipeline.build_medication_features
    build_interaction_features = _pipeline.build_interaction_features
    build_advanced_features = _pipeline.build_advanced_features
    extract_maximum_features = _pipeline.extract_maximum_features
    build_new_signal_features = _pipeline.build_new_signal_features
    build_trend_features = _pipeline.build_trend_features
    build_acceleration_features = _pipeline.build_acceleration_features
    build_cancer_specific_features = _cancer.build_cancer_specific_features

    saved = {}
    for window in cfg.WINDOWS:
        (cfg.FE_RESULTS / window).mkdir(parents=True, exist_ok=True)
        out_path = cfg.FE_RESULTS / window / f'feature_matrix_final_{window}.parquet'

        # ── Load this window's _dropped data (parquet preferred) ─
        obs_path = cfg.BASE_PATH / window / f'{cfg.DATA_PREFIX}_{window}_obs_dropped.parquet'
        med_path = cfg.BASE_PATH / window / f'{cfg.DATA_PREFIX}_{window}_med_dropped.parquet'
        try:
            clin = read_table(obs_path, low_memory=False)
            clin.columns = clin.columns.str.upper()
            if 'VALUE' in clin.columns:
                clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')
            med = read_table(med_path, low_memory=False)
            med.columns = med.columns.str.upper()
            if 'VALUE' in med.columns:
                med['VALUE'] = pd.to_numeric(med['VALUE'], errors='coerce')
        except FileNotFoundError as e:
            logger.warning(f"[{window}] Missing data: {e}")
            continue

        logger.info(f"\n{'#'*70}\n  FEATURE ENGINEERING — {window.upper()}\n{'#'*70}")
        logger.info(f"  Loaded: obs {len(clin):,} rows | med {len(med):,} rows")

        # ── 4a: Base features ─────────────────────────────────
        logger.info(f"  4a: Base features...")
        clin_features = build_clinical_features(clin, cfg)
        med_features = build_medication_features(med, cfg)
        fm = clin_features.join(med_features, how='left')
        med_cols = [c for c in fm.columns if c.startswith('MED_')]
        fm[med_cols] = fm[med_cols].fillna(0)
        fm = build_interaction_features(fm, cfg)
        del clin_features, med_features
        if cfg.SAVE_INTERMEDIATES:
            write_table(fm, cfg.FE_RESULTS / window / f'feature_matrix_{window}.parquet', index=True)
        logger.info(f"    Base: {fm.shape[0]:,} patients × {fm.shape[1]} features")

        # ── 4b: Advanced features ────────────────────────────
        logger.info(f"  4b: Advanced features...")
        prev_cols = fm.shape[1]
        adv = build_advanced_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(adv, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]
        del adv
        if cfg.SAVE_INTERMEDIATES:
            write_table(fm, cfg.FE_RESULTS / window / f'feature_matrix_enhanced_{window}.parquet', index=True)
        logger.info(f"    +adv: {fm.shape[1]} total (+{fm.shape[1]-prev_cols})")

        # ── 4c: Maximum features ─────────────────────────────
        logger.info(f"  4c: Maximum features...")
        prev_cols = fm.shape[1]
        mega = extract_maximum_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(mega, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]
        del mega
        if cfg.SAVE_INTERMEDIATES:
            write_table(fm, cfg.FE_RESULTS / window / f'feature_matrix_mega_{window}.parquet', index=True)
        logger.info(f"    +mega: {fm.shape[1]} total (+{fm.shape[1]-prev_cols})")

        # ── 4d: Cancer-specific features ─────────────────────
        logger.info(f"  4d: {cfg.CANCER_NAME}-specific features...")
        prev_cols = fm.shape[1]
        cancer_feats = build_cancer_specific_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(cancer_feats, how='left', rsuffix='_cancer')
        fm = fm.fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]
        # Remove near-zero-variance cancer-specific features
        cancer_cols = [c for c in fm.columns if c.startswith(cfg.PREFIX)]
        nzv = [c for c in cancer_cols if fm[c].value_counts(normalize=True).iloc[0] > 0.995]
        if nzv:
            fm = fm.drop(columns=nzv)
            logger.info(f"    Removed {len(nzv)} near-zero-var {cfg.CANCER_NAME} features")
        del cancer_feats
        logger.info(f"    +cancer: {fm.shape[1]} total (+{fm.shape[1]-prev_cols})")

        # ── 4e: New signal features ──────────────────────────
        logger.info(f"  4e: New signal features...")
        prev_cols = fm.shape[1]
        new_feats = build_new_signal_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(new_feats, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]
        del new_feats
        logger.info(f"    +new: {fm.shape[1]} total (+{fm.shape[1]-prev_cols})")

        # ── 4f: Trend features ───────────────────────────────
        logger.info(f"  4f: Trend features...")
        prev_cols = fm.shape[1]
        trend_feats = build_trend_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(trend_feats, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]
        del trend_feats
        logger.info(f"    +trend: {fm.shape[1]} total (+{fm.shape[1]-prev_cols})")

        # ── 4g: Acceleration features (fine-grained ramp-up signals) ──
        logger.info(f"  4g: Acceleration features...")
        prev_cols = fm.shape[1]
        accel_feats = build_acceleration_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(accel_feats, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]
        del accel_feats
        logger.info(f"    +accel: {fm.shape[1]} total (+{fm.shape[1]-prev_cols})")

        # ── Save final matrix for this window ────────────────
        write_table(fm, out_path, index=True)
        saved[window] = (fm.shape[0], fm.shape[1])
        logger.info(f"  Saved [{window}]: {fm.shape[0]:,} × {fm.shape[1]} | "
                     f"Pos: {(fm[cfg.LABEL_COL]==1).sum():,} | Neg: {(fm[cfg.LABEL_COL]==0).sum():,}")

        # ── Free everything for this window before the next ──
        del fm, clin, med
        gc.collect()

    logger.info(f"\n  FEATURE ENGINEERING COMPLETE — {len(saved)} windows saved")
    return saved


def run_cleanup(cfg):
    """Run feature cleanup (step 5)."""
    import_module('5_cleanup').run_cleanup(cfg)


def main():
    parser = argparse.ArgumentParser(description=f'{config.CANCER_NAME.title()} Cancer FE Pipeline')
    parser.add_argument('--step', choices=['sanity', 'clean', 'fe', 'cleanup', 'all'],
                        default='all', help='Which step to run (default: all)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.SCRIPT_DIR / 'pipeline.log', mode='w'),
        ]
    )

    cfg = config

    logger.info(f"{'='*70}")
    logger.info(f"  {cfg.CANCER_NAME.upper()} CANCER - FEATURE ENGINEERING PIPELINE")
    logger.info(f"{'='*70}")

    if args.step in ('sanity', 'all'):
        import_module('1_sanity_check').run_sanity_check(cfg)

    if args.step in ('clean', 'all'):
        import_module('2_clean_data').run_clean_data(cfg)

    if args.step in ('fe', 'all'):
        run_feature_engineering(cfg)

    if args.step in ('cleanup', 'all'):
        run_cleanup(cfg)

    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
