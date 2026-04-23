# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PREDICT ON UNSEEN / HOLDOUT DATA
# Loads saved models + threshold, applies to new feature matrix.
# If labels present, calculates evaluation metrics.
#
# Usage:
#   python 2_predict_unseen.py --window 12mo --data /path/to/holdout_features.csv
#   python 2_predict_unseen.py --window 12mo  (uses default holdout path)
# ═══════════════════════════════════════════════════════════════

import argparse
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report,
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR.parent / '2_Feature_Engineering_1to1'))
import config as fe_config


def predict_unseen(window, data_path=None):
    """Load saved models, predict on unseen data, evaluate if labels present."""

    model_dir = SCRIPT_DIR / 'results' / '1_training' / window / 'saved_models'
    if not model_dir.exists():
        logger.error(f"No saved models for {window} at {model_dir}")
        return

    # Load config
    config_path = model_dir / 'config.json'
    model_config = joblib.load(config_path)
    selected_features = model_config['selected_features']
    threshold = model_config['threshold']
    ensemble_models = model_config['ensemble_models']
    ensemble_weights = model_config['ensemble_weights']

    logger.info(f"  Loaded model config: {len(selected_features)} features, "
                 f"threshold={threshold:.4f}, ensemble={ensemble_models}")

    # Load data
    if data_path is None:
        # Default: look in 4_ExpandedData_Test
        data_path = SCRIPT_DIR.parent / '4_ExpandedData_Test' / 'results' / f'holdout_features_{window}.csv'

    df = pd.read_csv(data_path, index_col=0)
    logger.info(f"  Loaded holdout data: {df.shape[0]} patients x {df.shape[1]} features")

    has_labels = fe_config.LABEL_COL in df.columns
    y_true = None
    if has_labels:
        y_true = df[fe_config.LABEL_COL].values
        logger.info(f"  Labels found: Pos={y_true.sum()} Neg={(y_true==0).sum()}")

    # Align features
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        logger.warning(f"  {len(missing)} missing features (will be 0): {missing[:10]}")
        for f in missing:
            df[f] = 0

    X = df[selected_features].fillna(0)

    # Load models and predict
    preds = {}
    for name in ensemble_models:
        model_path = model_dir / f'{name}_model.pkl'
        model = joblib.load(model_path)
        preds[name] = model.predict_proba(X)[:, 1]
        logger.info(f"  {name}: predicted")

    # Ensemble (N-model weighted sum)
    ens = sum(w * preds[n] for w, n in zip(ensemble_weights, ensemble_models))

    # Apply calibration if present
    calibrator = model_config.get('calibrator')
    if calibrator is not None:
        ens = calibrator.transform(ens)

    # Force zero-risk on patients with no discriminative clinical features.
    # Training only saw patients with >=1 clinical event → boosted trees extrapolate
    # all-zero feature vectors unstably (often scoring 0.25-0.50 due to scale_pos_weight +
    # calibrator OOD behaviour). Clinically, zero relevant codes = no evidence = no risk.
    # Exclude demographics-only columns from the zero-check (age/age_band are always set).
    clinical_cols = [c for c in selected_features if c not in ('AGE_AT_INDEX', 'AGE_BAND')]
    if clinical_cols:
        all_zero = (X[clinical_cols].abs().sum(axis=1) == 0)
        n_zero = int(all_zero.sum())
        if n_zero > 0:
            logger.info(f"  Zero-feature patients → forced to risk=0: {n_zero:,}")
            ens[all_zero.values] = 0.0

    y_pred = (ens >= threshold).astype(int)

    # Save predictions
    out_dir = SCRIPT_DIR / 'results' / '2_predictions' / window
    pred_df = pd.DataFrame({
        'PATIENT_GUID': X.index,
        'y_pred_proba': ens,
        'y_pred': y_pred,
    })
    if has_labels:
        pred_df['y_true'] = y_true

    pred_path = out_dir / f'predictions_unseen_{window}.csv'
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"  Saved predictions: {pred_path}")

    # Evaluate if labels present
    if has_labels:
        auc = roc_auc_score(y_true, ens)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        logger.info(f"\n  HOLDOUT EVALUATION ({window}):")
        logger.info(f"    AUC: {auc:.4f}")
        logger.info(f"    Sensitivity: {sens:.4f}")
        logger.info(f"    Specificity: {spec:.4f}")
        logger.info(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
        logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Non-cancer', 'Cancer'])}")

        metrics_path = out_dir / f'predictions_unseen_{window}_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Window: {window}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"Sensitivity: {sens:.4f}\n")
            f.write(f"Specificity: {spec:.4f}\n")
            f.write(f"Threshold: {threshold:.4f}\n")
            f.write(f"TP={tp} FP={fp} FN={fn} TN={tn}\n")
            f.write(f"\n{classification_report(y_true, y_pred, target_names=['Non-cancer', 'Cancer'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', required=True, choices=['3mo', '6mo', '12mo'])
    parser.add_argument('--data', default=None, help='Path to holdout feature CSV')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    predict_unseen(args.window, args.data)
