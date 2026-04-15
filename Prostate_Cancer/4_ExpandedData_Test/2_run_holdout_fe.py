# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — RUN FE ON HOLDOUT DATA
# Imports the SAME pipeline functions as training — no copy-paste.
# Reads from: data/fe_input/{3mo,6mo,12mo}/
# Writes to:  results/{3mo,6mo,12mo}/holdout_features_{window}.csv
# ═══════════════════════════════════════════════════════════════

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Import FE modules from 2_Feature_Engineering (same code as training)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '2_Feature_Engineering'))
import config as fe_config
from importlib import import_module

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

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / 'data' / 'fe_input'
RESULTS_DIR = SCRIPT_DIR / 'results'


def run_holdout_fe():
    """Run the full FE pipeline on holdout data — identical to training FE."""

    cfg = fe_config

    for window in cfg.WINDOWS:
        obs_path = INPUT_DIR / window / f'{cfg.DATA_PREFIX}_{window}_obs_dropped.csv'
        med_path = INPUT_DIR / window / f'{cfg.DATA_PREFIX}_{window}_med_dropped.csv'

        if not obs_path.exists() or not med_path.exists():
            logger.warning(f"  Skipping {window}: missing input files")
            continue

        out_dir = RESULTS_DIR / window
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'#'*70}")
        logger.info(f"  HOLDOUT FE - {window.upper()}")
        logger.info(f"{'#'*70}")

        clin = pd.read_csv(obs_path, low_memory=False)
        clin.columns = clin.columns.str.upper()
        med = pd.read_csv(med_path, low_memory=False)
        med.columns = med.columns.str.upper()
        if 'VALUE' in clin.columns:
            clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

        logger.info(f"  Loaded: obs={clin.shape[0]:,} med={med.shape[0]:,}")

        # Step 4a: Base
        clin_feat = build_clinical_features(clin, cfg)
        med_feat = build_medication_features(med, cfg)
        fm = clin_feat.join(med_feat, how='left')
        med_cols = [c for c in fm.columns if c.startswith('MED_')]
        fm[med_cols] = fm[med_cols].fillna(0)
        fm = build_interaction_features(fm, cfg)
        logger.info(f"  Base: {fm.shape[1]} features")

        # Step 4b: Advanced
        adv = build_advanced_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(adv, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]

        # Step 4c: Maximum
        mega = extract_maximum_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(mega, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]

        # Step 4d: Cancer-specific
        cancer = build_cancer_specific_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(cancer, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]

        # Step 4e: New signal
        sig = build_new_signal_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(sig, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]

        # Step 4f: Trends
        trend = build_trend_features(clin, med, fm, window.upper(), cfg)
        fm = fm.join(trend, how='left').fillna(0).replace([np.inf, -np.inf], 0)
        fm = fm.loc[:, ~fm.columns.duplicated()]

        # Save
        out_path = out_dir / f'holdout_features_{window}.csv'
        fm.to_csv(out_path)
        logger.info(f"  SAVED: {out_path.name} ({fm.shape[0]} patients x {fm.shape[1]} features)")

    logger.info(f"\n  HOLDOUT FE COMPLETE")
    logger.info(f"  Next: Run 3_Modeling/2_predict_unseen.py")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_holdout_fe()
