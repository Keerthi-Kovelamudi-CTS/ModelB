# ═══════════════════════════════════════════════════════════════
# BREAST CANCER — PREDICT ON UNSEEN / HOLDOUT DATA
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

# Required for joblib unpickle of saved calibrator artifacts
from _calibrator import CalibratorWrapper  # noqa: F401

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)




def _apply_encoder_mappings_if_needed(df, window):
    """Apply training-time encoder mapping if raw demo columns present but encoded missing.

    Reads encoder_mappings.json (written by 3_pipeline.py during the
    training FE run). Safe no-op if:
      - the file isn't found (logs a warning), or
      - the encoded columns are already in df.
    """
    import json as _json
    encoded = ['SEX_MALE', 'ETHNICITY_16_ENC', 'ETHNICITY_6_ENC']
    if all(c in df.columns for c in encoded):
        return  # already encoded by FE — common path

    # Locate the sidecar produced by 3_pipeline.py
    try:
        import config as fe_cfg
        enc_path = fe_cfg.FE_RESULTS / window.lower() / 'encoder_mappings.json'
    except Exception:
        # fallback path relative to this script
        enc_path = SCRIPT_DIR.parent / '2_Feature_Engineering' / 'results' / '3_feature_engineering' / window.lower() / 'encoder_mappings.json'

    if not enc_path.exists():
        logger.warning(
            f"  encoder_mappings.json not found at {enc_path} — "
            f"ethnicity may be encoded inconsistently with training"
        )
        return

    with open(enc_path, encoding='utf-8') as f:
        mappings = _json.load(f)
    logger.info(f"  Loaded encoder mappings from {enc_path}")

    if 'SEX_MALE' not in df.columns and 'SEX' in df.columns:
        df['SEX_MALE'] = df['SEX'].astype(str).str.upper().map(
            mappings.get('sex_male', {'M': 1, 'F': 0})
        ).fillna(-1).astype(int)
    if 'ETHNICITY_16_ENC' not in df.columns and 'PATIENT_ETHNICITY_16' in df.columns:
        df['ETHNICITY_16_ENC'] = df['PATIENT_ETHNICITY_16'].astype(str).map(
            mappings.get('ethnicity_16_enc', {})
        ).fillna(-1).astype(int)
    if 'ETHNICITY_6_ENC' not in df.columns and 'PATIENT_ETHNICITY_6' in df.columns:
        df['ETHNICITY_6_ENC'] = df['PATIENT_ETHNICITY_6'].astype(str).map(
            mappings.get('ethnicity_6_enc', {})
        ).fillna(-1).astype(int)


def compute_shap_top_factors(model_dir, seeds, ensemble_models, X, top_n=5):
    """Per-patient SHAP top-N factor extraction.

    Averages TreeExplainer SHAP values across all base models AND seeds. SHAP values are
    on the model's logit scale (not the calibrated probability), so magnitudes are useful
    for ranking features but not directly interpretable as "this raised the probability by X".

    Returns: list of length n_rows; each element is a list of (feature_name, value, shap)
    tuples sorted by |shap| descending. Returns None if SHAP unavailable or all models fail.
    """
    if not HAS_SHAP:
        return None
    n_rows, n_feats = X.shape
    feature_names = list(X.columns)
    agg = np.zeros((n_rows, n_feats), dtype=float)
    n_used = 0
    for seed in seeds:
        seed_dir = Path(model_dir) / f'seed_{seed}'
        if not seed_dir.exists():
            continue
        for name in ensemble_models:
            mp = seed_dir / f'{name}_model.pkl'
            if not mp.exists():
                continue
            try:
                m = joblib.load(mp)
                explainer = shap.TreeExplainer(m)
                sv = explainer.shap_values(X)
                # LightGBM (some versions) returns [neg, pos]; XGB/CatBoost return one array.
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) >= 2 else sv[0]
                sv = np.asarray(sv)
                if sv.shape == agg.shape:
                    agg += sv
                    n_used += 1
            except Exception as e:
                logger.warning(f"  SHAP failed for {name} seed={seed}: {e}")
    if n_used == 0:
        logger.warning("  SHAP: no base models contributed")
        return None
    agg /= n_used
    Xv = X.values
    out = []
    for i in range(n_rows):
        idx = np.argsort(-np.abs(agg[i]))[:top_n]
        out.append([
            (feature_names[j], float(Xv[i, j]), float(agg[i, j])) for j in idx
        ])
    return out


def format_top_factors_text(top_factors, max_chars=300):
    """Compact 'name=value(±shap)' joined string for clinical review."""
    if top_factors is None:
        return ''
    parts = [f"{n}={v:.3g}({s:+.3f})" for n, v, s in top_factors]
    s = '; '.join(parts)
    return s if len(s) <= max_chars else s[:max_chars - 1] + '…'

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR.parent / '2_Feature_Engineering'))
import config as fe_config


def predict_unseen(window, data_path=None, explain=False, top_n=5):
    """Load saved models, predict on unseen data, evaluate if labels present."""

    model_dir = SCRIPT_DIR / 'results' / '1_training' / window / 'saved_models'
    if not model_dir.exists():
        logger.error(f"No saved models for {window} at {model_dir}")
        return

    # Load config (multi-seed + per-band format; falls back to legacy single-seed)
    config_path = model_dir / 'config.json'
    model_config = joblib.load(config_path)
    selected_features = model_config['selected_features']
    ensemble_models = model_config['ensemble_models']
    seeds = model_config.get('seeds', [42])
    is_multi_seed = 'seeds' in model_config and (model_dir / f'seed_{seeds[0]}').exists()

    if is_multi_seed:
        logger.info(f"  Loaded multi-seed config: {len(selected_features)} features, "
                     f"seeds={seeds}, ensemble={ensemble_models}")
    else:
        # Legacy single-seed
        threshold = model_config['threshold']
        logger.info(f"  Loaded legacy model config: {len(selected_features)} features, "
                     f"threshold={threshold:.4f}, ensemble={ensemble_models}")

    # Load data
    if data_path is None:
        # Default: Truveta holdout FE output → 2_Feature_Engineering/data/features/holdout_{window}/
        data_path = (SCRIPT_DIR.parent / '2_Feature_Engineering' / 'data' / 'features'
                     / f'holdout_{window}' / 'breast_feature_matrix.parquet')

    df = (pd.read_parquet(data_path) if str(data_path).endswith('.parquet') else pd.read_csv(data_path, index_col=0))
    logger.info(f"  Loaded holdout data: {df.shape[0]} patients x {df.shape[1]} features")

    # ─── Apply training-time encoder mappings if raw demographics are present ───
    # Reads encoder_mappings.json saved by 3_pipeline.py and applies the same
    # integer→category mapping. No-op if the encoded columns are already in df
    # (the normal path when this script runs on FE-pipeline output).
    _apply_encoder_mappings_if_needed(df, window)

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

    if is_multi_seed:
        # Multi-seed: load models per seed, predict, stack via meta-learner, average across seeds
        per_seed_ens = []
        for seed in seeds:
            seed_dir = model_dir / f'seed_{seed}'
            preds = {}
            for name in ensemble_models:
                m = joblib.load(seed_dir / f'{name}_model.pkl')
                preds[name] = m.predict_proba(X)[:, 1]
            meta = joblib.load(seed_dir / 'meta_learner.pkl')
            stack_X = np.column_stack([preds[n] for n in ensemble_models])
            per_seed_ens.append(meta.predict_proba(stack_X)[:, 1])
            logger.info(f"  Seed {seed}: predicted")
        ens_raw = np.mean(per_seed_ens, axis=0)

        # Per-band calibration
        calibrators_per_band = model_config['calibrators_per_band']
        thresholds_per_band = model_config['thresholds_per_band']
        global_cal = model_config['global_calibrator']
        global_threshold = model_config['global_threshold']
        BAND_BINS = model_config['band_bins']
        BAND_LABELS = model_config['band_labels']
        # Source age from the FULL df (not selected X) so per-band calibration works
        # even if patient_age gets dropped from selected_features (defensive).
        ages = (df['patient_age'].values if 'patient_age' in df.columns
                else np.full(len(X), 65.0))
        bands = np.asarray(pd.cut(ages, bins=BAND_BINS, labels=BAND_LABELS, right=False).astype(str))
        ens = np.zeros(len(ens_raw))
        y_pred = np.zeros(len(ens_raw), dtype=int)
        for band in BAND_LABELS:
            mask = (bands == band)
            if mask.sum() == 0:
                continue
            cal = calibrators_per_band.get(band, global_cal)
            thr = thresholds_per_band.get(band, global_threshold)
            ens[mask] = cal.transform(ens_raw[mask])
            y_pred[mask] = (ens[mask] >= thr).astype(int)
        threshold = global_threshold  # for downstream output
    else:
        # Legacy single-seed path
        preds = {}
        for name in ensemble_models:
            m = joblib.load(model_dir / f'{name}_model.pkl')
            preds[name] = m.predict_proba(X)[:, 1]
            logger.info(f"  {name}: predicted")
        meta = model_config.get('meta_learner')
        if meta is not None:
            stack_X = np.column_stack([preds[n] for n in ensemble_models])
            ens = meta.predict_proba(stack_X)[:, 1]
        else:
            ens = sum(w * preds[n] for w, n in zip(model_config['ensemble_weights'], ensemble_models))
        calibrator = model_config.get('calibrator')
        if calibrator is not None:
            ens = calibrator.transform(ens)
        y_pred = (ens >= threshold).astype(int)

    # Force zero-risk on patients with no discriminative clinical features
    clinical_cols = [c for c in selected_features if c not in ('patient_age', 'AGE_BAND')]
    if clinical_cols:
        all_zero = (X[clinical_cols].abs().sum(axis=1) == 0)
        n_zero = int(all_zero.sum())
        if n_zero > 0:
            logger.info(f"  Zero-feature patients → forced to risk=0: {n_zero:,}")
            ens[all_zero.values] = 0.0
            y_pred[all_zero.values] = 0

    # Save predictions
    out_dir = SCRIPT_DIR / 'results' / '2_predictions' / window
    pred_df = pd.DataFrame({
        'patient_id': X.index,
        'y_pred_proba': ens,
        'y_pred': y_pred,
    })
    if has_labels:
        pred_df['y_true'] = y_true

    # Per-patient SHAP top factors (opt-in via --explain). Multi-seed inference only.
    if explain:
        if not HAS_SHAP:
            logger.warning("  --explain requested but `shap` not installed; skipping")
        elif not is_multi_seed:
            logger.warning("  --explain only supported on multi-seed models; skipping")
        else:
            logger.info(f"  Computing per-patient SHAP across {len(seeds)} seeds × {len(ensemble_models)} base models...")
            top = compute_shap_top_factors(model_dir, seeds, ensemble_models, X, top_n=top_n)
            if top is not None:
                pred_df['top_factors_text'] = [format_top_factors_text(t) for t in top]
                # Structured columns for the top-3 factors (filterable downstream).
                for k in range(min(3, top_n)):
                    pred_df[f'top{k+1}_feature'] = [t[k][0] if k < len(t) else '' for t in top]
                    pred_df[f'top{k+1}_value']   = [t[k][1] if k < len(t) else 0.0 for t in top]
                    pred_df[f'top{k+1}_shap']    = [t[k][2] if k < len(t) else 0.0 for t in top]
                logger.info(f"  SHAP done: top_factors_text + top1..top{min(3, top_n)} structured columns added")

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
    parser.add_argument('--window', required=True, choices=list(fe_config.WINDOWS))
    parser.add_argument('--data', default=None, help='Path to holdout feature CSV')
    parser.add_argument('--explain', action='store_true',
                        help='Add per-patient SHAP top-factor columns (slower; multi-seed only)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top SHAP factors per patient when --explain is on')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    predict_unseen(args.window, args.data, explain=args.explain, top_n=args.top_n)
