# ═══════════════════════════════════════════════════════════════
# EXPLAIN PREDICTIONS — SHAP + Clinical Language
# Generates clinician-friendly explanations for model predictions.
#
# For each patient:
#   1. Compute SHAP values (why did the model predict this?)
#   2. Map top SHAP features to clinical descriptions
#   3. Generate a readable report
#
# Usage:
#   python 1_explain_predictions.py --window 12mo
#   python 1_explain_predictions.py --window 12mo --top 15
#   python 1_explain_predictions.py --window 12mo --patient GUID123
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELING_DIR = PROJECT_DIR / '3_Modeling_Ethnicity'
FE_DIR = PROJECT_DIR / '2_Feature_Engineering_Ethnicity'
RESULTS_DIR = SCRIPT_DIR / 'results'

sys.path.insert(0, str(FE_DIR))
import config as fe_config

from feature_dictionary import (
    get_explainability_class,
    get_clinical_description,
    classify_all_features,
)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install: pip install shap")


def load_model_and_data(window):
    """Load trained model, feature matrix, and predictions."""
    model_dir = MODELING_DIR / 'results' / '1_training' / window / 'saved_models'
    config = joblib.load(model_dir / 'config.json')
    selected = config['selected_features']
    threshold = config['threshold']

    # Load the highest-weight ensemble model for SHAP (most representative of the ensemble)
    ens_models = config['ensemble_models']
    ens_weights = config['ensemble_weights']
    model_name = ens_models[int(np.argmax(ens_weights))]
    model = joblib.load(model_dir / f'{model_name}_model.pkl')

    # Load feature matrix
    cleanup_path = fe_config.CLEANUP_RESULTS / window / f'feature_matrix_clean_{window}.csv'
    fm = pd.read_csv(cleanup_path, index_col=0)

    # Load text features and merge
    for name, path_fn in [
        ('text', fe_config.TEXT_RESULTS),
        ('tfidf', fe_config.EMB_RESULTS),
        ('bert', fe_config.BERT_RESULTS),
    ]:
        fpath = path_fn / window / f'{"text_features" if name == "text" else "text_embeddings" if name == "tfidf" else "bert_embeddings"}_{window}.csv'
        if fpath.exists():
            tdf = pd.read_csv(fpath, index_col=0)
            fm = fm.join(tdf, how='left')

    fm = fm.fillna(0).loc[:, ~fm.columns.duplicated()]

    # Align to selected features
    missing = [f for f in selected if f not in fm.columns]
    for f in missing:
        fm[f] = 0

    X = fm[selected]
    y = fm['LABEL'] if 'LABEL' in fm.columns else None

    return model, model_name, X, y, selected, threshold, config


def compute_shap_values(model, X, model_name):
    """Compute SHAP values for the model."""
    if not HAS_SHAP:
        # Fallback: use feature importance * value as proxy
        logger.info("  Using feature importance proxy (SHAP not available)")
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            shap_values = X.values * imp[np.newaxis, :]
            return pd.DataFrame(shap_values, index=X.index, columns=X.columns)
        return None

    logger.info(f"  Computing SHAP values for {model_name}...")
    try:
        if 'xgboost' in model_name or 'lightgbm' in model_name or 'catboost' in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        logger.warning(f"  SHAP failed ({type(e).__name__}: {e}); "
                       f"falling back to feature_importances_ as an approximation")
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            shap_values = X.values * imp[np.newaxis, :]
            return pd.DataFrame(shap_values, index=X.index, columns=X.columns)
        return None

    # Handle binary classification (some return list of 2 arrays)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # positive class

    return pd.DataFrame(shap_values, index=X.index, columns=X.columns)


def explain_patient(patient_guid, shap_df, X, threshold, top_n=10):
    """Generate clinical explanation for one patient."""
    if patient_guid not in shap_df.index:
        return None

    shap_row = shap_df.loc[patient_guid]
    value_row = X.loc[patient_guid]

    # Sort by absolute SHAP value (impact on prediction)
    abs_shap = shap_row.abs().sort_values(ascending=False)
    top_features = abs_shap.head(top_n * 2).index.tolist()  # get extra, filter later

    explanations = []
    for feat in top_features:
        shap_val = shap_row[feat]
        feat_val = value_row[feat]
        direction = 'increases' if shap_val > 0 else 'decreases'
        exp_class = get_explainability_class(feat)
        description = get_clinical_description(feat, feat_val)

        explanations.append({
            'feature': feat,
            'value': round(float(feat_val), 4),
            'shap_value': round(float(shap_val), 6),
            'abs_shap': round(float(abs(shap_val)), 6),
            'direction': direction,
            'explainability': exp_class,
            'clinical_description': description,
        })

    return explanations


def generate_clinician_report(patient_guid, explanations, risk_score, threshold, top_n=10):
    """Generate a readable clinical report."""
    prediction = 'HIGH RISK' if risk_score >= threshold else 'LOW RISK'

    lines = []
    lines.append(f"Patient: {patient_guid}")
    lines.append(f"Risk Score: {risk_score:.2%}  |  Threshold: {threshold:.2%}  |  Prediction: {prediction}")
    lines.append("")

    # Only show direct and indirect features to clinicians
    explainable = [e for e in explanations if e['explainability'] in ('direct', 'indirect')]
    risk_factors = [e for e in explainable if e['direction'] == 'increases'][:top_n]
    protective = [e for e in explainable if e['direction'] == 'decreases'][:5]

    if risk_factors:
        lines.append("WHY THIS PATIENT IS FLAGGED:")
        lines.append("-" * 50)
        for i, e in enumerate(risk_factors, 1):
            if e['value'] != 0:
                lines.append(f"  {i}. {e['clinical_description']}")

    if protective:
        lines.append("")
        lines.append("PROTECTIVE FACTORS:")
        lines.append("-" * 50)
        for i, e in enumerate(protective, 1):
            if e['value'] != 0:
                lines.append(f"  {i}. {e['clinical_description']}")

    # Flag opaque features if they're in top 5
    opaque_in_top5 = [e for e in explanations[:5] if e['explainability'] == 'opaque']
    if opaque_in_top5:
        lines.append("")
        lines.append(f"NOTE: {len(opaque_in_top5)} of the top 5 model drivers are statistical features")
        lines.append("that cannot be explained clinically. Consider running the enhancement loop.")

    return '\n'.join(lines)


def run_explain(window, top_n=10, patient_guid=None):
    """Run SHAP explainability for a window."""
    out_dir = RESULTS_DIR / window
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPLAINABILITY ANALYSIS - {window.upper()}")
    logger.info(f"{'='*60}")

    model, model_name, X, y, selected, threshold, config = load_model_and_data(window)
    logger.info(f"  Model: {model_name} | Features: {len(selected)} | Patients: {X.shape[0]}")

    # Classify features
    classification = classify_all_features(selected)
    logger.info(f"  Feature classes: {len(classification['direct'])} direct, "
                f"{len(classification['indirect'])} indirect, "
                f"{len(classification['opaque'])} opaque")

    # Compute SHAP
    shap_df = compute_shap_values(model, X, model_name)
    if shap_df is None:
        logger.error("  Cannot compute explanations")
        return

    # Global feature importance (mean |SHAP|)
    global_importance = shap_df.abs().mean().sort_values(ascending=False)

    # Save global importance with explainability class
    global_df = pd.DataFrame({
        'feature': global_importance.index,
        'mean_abs_shap': global_importance.values,
        'explainability': [get_explainability_class(f) for f in global_importance.index],
        'clinical_description': [get_clinical_description(f) for f in global_importance.index],
    })
    global_df.to_csv(out_dir / f'global_feature_importance_{window}.csv', index=False)
    logger.info(f"  Saved global importance: {out_dir / f'global_feature_importance_{window}.csv'}")

    # Show top 20 globally
    logger.info(f"\n  TOP 20 FEATURES BY MEAN |SHAP|:")
    logger.info(f"  {'─'*70}")
    for i, (_, row) in enumerate(global_df.head(20).iterrows(), 1):
        cls_marker = {'direct': 'D', 'indirect': 'I', 'opaque': 'X'}[row['explainability']]
        logger.info(f"  [{cls_marker}] {i:2d}. {row['feature']:45s} SHAP={row['mean_abs_shap']:.6f}")
    logger.info(f"  D=direct, I=indirect, X=opaque")

    # Flag problem: opaque features in top 10
    top10_opaque = global_df.head(10)[global_df.head(10)['explainability'] == 'opaque']
    if len(top10_opaque) > 0:
        logger.warning(f"\n  WARNING: {len(top10_opaque)} OPAQUE features in top 10:")
        for _, row in top10_opaque.iterrows():
            logger.warning(f"    {row['feature']} (SHAP={row['mean_abs_shap']:.6f})")
        logger.warning(f"  → Run 3_enhancement_loop.py to remove and retrain")

    # Per-patient explanations
    if patient_guid:
        patients = [patient_guid] if patient_guid in X.index else []
        if not patients:
            logger.warning(f"  Patient {patient_guid} not found")
    else:
        # Explain all positive patients + sample of negatives
        if y is not None:
            pos_patients = X.index[y == 1].tolist()
            neg_sample = X.index[y == 0].tolist()[:50]
            patients = pos_patients + neg_sample
        else:
            patients = X.index.tolist()[:100]

    logger.info(f"\n  Generating explanations for {len(patients)} patients...")

    all_explanations = []
    all_reports = []
    for pid in patients:
        explanations = explain_patient(pid, shap_df, X, threshold, top_n=top_n)
        if explanations is None:
            continue

        # Get risk score (use model predict_proba as proxy)
        risk_score = float(model.predict_proba(X.loc[[pid]])[0, 1])
        report = generate_clinician_report(pid, explanations, risk_score, threshold, top_n=top_n)

        all_explanations.append({
            'patient_guid': pid,
            'risk_score': round(risk_score, 4),
            'prediction': 'HIGH RISK' if risk_score >= threshold else 'LOW RISK',
            'top_factors': explanations[:top_n],
        })
        all_reports.append(report)

    # Save
    with open(out_dir / f'patient_explanations_{window}.json', 'w') as f:
        json.dump(all_explanations, f, indent=2)

    with open(out_dir / f'clinician_reports_{window}.txt', 'w') as f:
        f.write(f"\n{'='*60}\n".join(all_reports))

    logger.info(f"  Saved {len(all_explanations)} patient explanations")
    logger.info(f"  Saved clinician reports: clinician_reports_{window}.txt")

    # Sample report
    if all_reports:
        logger.info(f"\n  SAMPLE REPORT:")
        logger.info(f"  {'─'*60}")
        logger.info(f"  {all_reports[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='12mo', choices=['3mo', '6mo', '12mo'])
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--patient', default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_explain(args.window, top_n=args.top, patient_guid=args.patient)
