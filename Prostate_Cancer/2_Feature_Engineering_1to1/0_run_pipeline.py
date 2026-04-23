# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PIPELINE ORCHESTRATOR
# Runs the full FE pipeline: sanity → clean → base → advanced →
#   mega → cancer-specific → trend → signal → cleanup → text
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

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def load_data(cfg):
    """Load obs + med CSVs for all windows."""
    data = {}
    for window in cfg.WINDOWS:
        data[window] = {}
        try:
            obs_path = cfg.BASE_PATH / window / f'{cfg.DATA_PREFIX}_{window}_obs_dropped.csv'
            med_path = cfg.BASE_PATH / window / f'{cfg.DATA_PREFIX}_{window}_med_dropped.csv'
            obs = pd.read_csv(obs_path, low_memory=False)
            obs.columns = obs.columns.str.upper()
            med = pd.read_csv(med_path, low_memory=False)
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
    """Run the full FE pipeline (steps 4a through 4f)."""
    _pipeline = import_module('3_pipeline')
    _cancer = import_module('4_cancer_features')

    build_clinical_features = _pipeline.build_clinical_features
    build_medication_features = _pipeline.build_medication_features
    build_interaction_features = _pipeline.build_interaction_features
    build_advanced_features = _pipeline.build_advanced_features
    extract_maximum_features = _pipeline.extract_maximum_features
    build_new_signal_features = _pipeline.build_new_signal_features
    build_trend_features = _pipeline.build_trend_features
    build_cancer_specific_features = _cancer.build_cancer_specific_features

    # Ensure output dirs exist
    for window in cfg.WINDOWS:
        (cfg.FE_RESULTS / window).mkdir(parents=True, exist_ok=True)

    data = load_data(cfg)

    # ── Step 4a: Base features ────────────────────────────────
    logger.info(f"\n{'#'*70}")
    logger.info(f"  STEP 4a: BASE FEATURES")
    logger.info(f"{'#'*70}")

    feature_matrices = {}
    for window in cfg.WINDOWS:
        clin = data[window]['obs']
        med = data[window]['med']
        if clin is None or med is None:
            logger.warning(f"  Missing data for {window}")
            continue

        logger.info(f"\n  --- {window.upper()} ---")
        clin_features = build_clinical_features(clin, cfg)
        med_features = build_medication_features(med, cfg)

        fm = clin_features.join(med_features, how='left')
        med_cols = [c for c in fm.columns if c.startswith('MED_')]
        fm[med_cols] = fm[med_cols].fillna(0)

        fm = build_interaction_features(fm, cfg)
        feature_matrices[window] = fm
        logger.info(f"  Base: {fm.shape[0]} patients x {fm.shape[1]} features")

        if cfg.SAVE_INTERMEDIATES:
            fm.to_csv(cfg.FE_RESULTS / window / f'feature_matrix_{window}.csv')

    # ── Step 4b: Advanced features ────────────────────────────
    logger.info(f"\n{'#'*70}")
    logger.info(f"  STEP 4b: ADVANCED FEATURES")
    logger.info(f"{'#'*70}")

    enhanced_matrices = {}
    for window in cfg.WINDOWS:
        clin = data[window]['obs']
        med = data[window]['med']
        if clin is None or med is None:
            continue

        adv = build_advanced_features(clin, med, feature_matrices[window], window.upper(), cfg)
        enhanced = feature_matrices[window].join(adv, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        enhanced = enhanced.loc[:, ~enhanced.columns.duplicated()]
        enhanced_matrices[window] = enhanced
        logger.info(f"  {window}: {enhanced.shape[1]} features (+{enhanced.shape[1] - feature_matrices[window].shape[1]})")

        if cfg.SAVE_INTERMEDIATES:
            enhanced.to_csv(cfg.FE_RESULTS / window / f'feature_matrix_enhanced_{window}.csv')

    # ── Step 4c: Maximum features ─────────────────────────────
    logger.info(f"\n{'#'*70}")
    logger.info(f"  STEP 4c: MAXIMUM FEATURES")
    logger.info(f"{'#'*70}")

    mega_matrices = {}
    for window in cfg.WINDOWS:
        clin = data[window]['obs']
        med = data[window]['med']
        if clin is None or med is None:
            continue

        mega = extract_maximum_features(clin, med, enhanced_matrices[window], window.upper(), cfg)
        final = enhanced_matrices[window].join(mega, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        final = final.loc[:, ~final.columns.duplicated()]
        mega_matrices[window] = final
        logger.info(f"  {window}: {final.shape[1]} features (+{final.shape[1] - enhanced_matrices[window].shape[1]})")

        if cfg.SAVE_INTERMEDIATES:
            final.to_csv(cfg.FE_RESULTS / window / f'feature_matrix_mega_{window}.csv')

    # ── Step 4d: Cancer-specific features ─────────────────────
    logger.info(f"\n{'#'*70}")
    logger.info(f"  STEP 4d: {cfg.CANCER_NAME.upper()}-SPECIFIC FEATURES")
    logger.info(f"{'#'*70}")

    final_matrices = {}
    for window in cfg.WINDOWS:
        clin = data[window]['obs']
        med = data[window]['med']
        if clin is None or med is None:
            continue

        cancer_feats = build_cancer_specific_features(clin, med, mega_matrices[window], window.upper(), cfg)
        final = mega_matrices[window].join(cancer_feats, how='left', rsuffix='_cancer')
        final = final.fillna(0).replace([np.inf, -np.inf], 0)
        final = final.loc[:, ~final.columns.duplicated()]

        # Remove near-zero-variance cancer-specific features
        cancer_cols = [c for c in final.columns if c.startswith(cfg.PREFIX)]
        nzv = [c for c in cancer_cols if final[c].value_counts(normalize=True).iloc[0] > 0.995]
        if nzv:
            final = final.drop(columns=nzv)
            logger.info(f"  Removed {len(nzv)} near-zero-var {cfg.CANCER_NAME} features")

        final_matrices[window] = final
        logger.info(f"  {window}: {final.shape[1]} features (+{len(cancer_feats.columns)} {cfg.CANCER_NAME}-specific)")

    # ── Step 4e: New signal features ──────────────────────────
    logger.info(f"\n{'#'*70}")
    logger.info(f"  STEP 4e: NEW SIGNAL FEATURES")
    logger.info(f"{'#'*70}")

    for window in cfg.WINDOWS:
        clin = data[window]['obs']
        med = data[window]['med']
        if clin is None or med is None:
            continue

        new_feats = build_new_signal_features(clin, med, final_matrices[window], window.upper(), cfg)
        final = final_matrices[window].join(new_feats, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        final = final.loc[:, ~final.columns.duplicated()]
        final_matrices[window] = final
        logger.info(f"  {window}: {final.shape[1]} features (+{new_feats.shape[1]})")

    # ── Step 4f: Trend features ───────────────────────────────
    logger.info(f"\n{'#'*70}")
    logger.info(f"  STEP 4f: TREND FEATURES")
    logger.info(f"{'#'*70}")

    for window in cfg.WINDOWS:
        clin = data[window]['obs']
        med = data[window]['med']
        if clin is None or med is None:
            continue

        trend_feats = build_trend_features(clin, med, final_matrices[window], window.upper(), cfg)
        final = final_matrices[window].join(trend_feats, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        final = final.loc[:, ~final.columns.duplicated()]
        final_matrices[window] = final
        logger.info(f"  {window}: {final.shape[1]} features (+{trend_feats.shape[1]})")

    # ── Save final matrices ───────────────────────────────────
    for window in cfg.WINDOWS:
        if window in final_matrices:
            out_path = cfg.FE_RESULTS / window / f'feature_matrix_final_{window}.csv'
            final_matrices[window].to_csv(out_path)
            fm = final_matrices[window]
            logger.info(f"  Saved {window}: {fm.shape[0]} x {fm.shape[1]} | "
                        f"Pos: {(fm[cfg.LABEL_COL]==1).sum()} | Neg: {(fm[cfg.LABEL_COL]==0).sum()}")

    logger.info(f"\n  FEATURE ENGINEERING COMPLETE")
    return final_matrices


def run_cleanup(cfg):
    """Run feature cleanup (step 5)."""
    import_module('5_cleanup').run_cleanup(cfg)


def run_text(cfg):
    """Run text feature pipeline (step 6)."""
    import_module('6_text_features').run_text_features(cfg)


def main():
    parser = argparse.ArgumentParser(description=f'{config.CANCER_NAME.title()} Cancer FE Pipeline')
    parser.add_argument('--step', choices=['sanity', 'clean', 'fe', 'cleanup', 'text', 'all'],
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

    if args.step in ('text', 'all'):
        run_text(cfg)

    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
