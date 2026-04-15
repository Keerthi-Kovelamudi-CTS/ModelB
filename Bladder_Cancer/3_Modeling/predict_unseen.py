# ═══════════════════════════════════════════════════════════════
# BLADDER CANCER — PREDICT ON UNSEEN DATA
# Loads saved .joblib models and runs predictions on new patient data
#
# Usage:
#   python3 predict_unseen.py --window 3mo \
#     --feature_matrix ../2_Feature_Engineering/results/Cleanup_Finalresults/3mo/feature_matrix_clean_3mo.csv
#
# If LABEL column exists → computes full metrics
# Output: predictions_unseen.csv + metrics_report.txt
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent

# Bladder models: 7 individual + ensemble/meta models
MODEL_FILES = {
    'xgboost':              'model_xgboost.joblib',
    'lightgbm':             'model_lightgbm.joblib',
    'catboost':             'model_catboost.joblib',
    'random_forest':        'model_random_forest.joblib',
    'logistic_regression':  'model_logistic_regression.joblib',
    'lightgbm_dart':        'model_lightgbm_(dart).joblib',
    'xgboost_deep':         'model_xgboost_(deep).joblib',
}

META_FILES = {
    'stacking_meta_lr':     'stacking_meta_lr.joblib',
    'deployment_meta_v4':   'deployment_meta_v4.joblib',
}


def load_models_and_config(window):
    """Load saved .joblib models and model_summary_v4.json config."""
    models_dir = SCRIPT_DIR / 'models' / window
    results_dir = SCRIPT_DIR / 'results' / window

    if not models_dir.exists():
        print(f"  ERROR: No models directory at {models_dir}")
        sys.exit(1)

    # Load config
    config_path = results_dir / 'model_summary_v4.json'
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Loaded config: {config_path.name}")
    else:
        print(f"  WARNING: No model_summary_v4.json found at {config_path}")

    # Load individual models
    models = {}
    for name, fname in MODEL_FILES.items():
        path = models_dir / fname
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {fname} not found, skipping")

    # Load meta/ensemble models
    meta_models = {}
    for name, fname in META_FILES.items():
        path = models_dir / fname
        if path.exists():
            meta_models[name] = joblib.load(path)
            print(f"  Loaded meta: {name}")

    # Extract feature list from config
    features = []
    if 'dataset' in config and 'features' in config['dataset']:
        features = config['dataset']['features']
    elif 'features' in config:
        features = config['features']

    print(f"  Individual models: {len(models)}")
    print(f"  Meta models: {len(meta_models)}")
    if features:
        print(f"  Features from config: {len(features)}")
    else:
        print(f"  No feature list in config — will use all feature columns from input")

    return models, meta_models, config, features


def predict(models, meta_models, config, features, feature_matrix):
    """Run predictions using all loaded models."""

    # Determine feature columns
    non_feature_cols = {'PATIENT_GUID', 'LABEL', 'ACTUAL_LABEL', 'patient_guid'}
    if features:
        available = [f for f in features if f in feature_matrix.columns]
        missing = [f for f in features if f not in feature_matrix.columns]

        if missing:
            print(f"\n  WARNING: {len(missing)} features missing from input (filled with 0):")
            for f in missing[:10]:
                print(f"    {f}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

        X = pd.DataFrame(0, index=feature_matrix.index, columns=features)
        for f in available:
            X[f] = feature_matrix[f]
    else:
        # No feature list — use all non-label columns
        feat_cols = [c for c in feature_matrix.columns if c not in non_feature_cols]
        X = feature_matrix[feat_cols].copy()
        print(f"  Using {len(feat_cols)} feature columns from input matrix")

    # Smart NaN fill (same logic as cleanup)
    for col in X.columns:
        if X[col].isna().any():
            if any(x in col for x in ['_mean', '_min', '_max', '_first', '_last', '_delta', '_slope', '_pct_change']):
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
            elif any(x in col for x in ['_days_', 'SEQ_', '_to_index']):
                X[col] = X[col].fillna(-1)
            else:
                X[col] = X[col].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    predictions = pd.DataFrame({'PATIENT_GUID': feature_matrix.index})

    # Preserve label if present (check both LABEL and ACTUAL_LABEL)
    if 'LABEL' in feature_matrix.columns:
        predictions['LABEL'] = feature_matrix['LABEL'].values
    elif 'ACTUAL_LABEL' in feature_matrix.columns:
        predictions['LABEL'] = feature_matrix['ACTUAL_LABEL'].values

    # ── CONFIDENCE SCORE ──
    demo_cols = [c for c in X.columns if c in ['AGE_AT_INDEX', 'AGE_BAND', 'AGE_AT_DIAGNOSIS']]
    clinical_non_zero = (X.drop(columns=demo_cols, errors='ignore') != 0).sum(axis=1)
    nz_values = clinical_non_zero.values

    predictions['non_zero_features'] = nz_values
    predictions['confidence'] = 'HIGH'
    predictions.loc[nz_values < 5, 'confidence'] = 'LOW'
    predictions.loc[(nz_values >= 5) & (nz_values < 15), 'confidence'] = 'MEDIUM'

    n_low = (predictions['confidence'] == 'LOW').sum()
    n_med = (predictions['confidence'] == 'MEDIUM').sum()
    n_high = (predictions['confidence'] == 'HIGH').sum()
    print(f"\n  Confidence: HIGH={n_high:,} ({n_high/len(predictions)*100:.1f}%) | MEDIUM={n_med:,} ({n_med/len(predictions)*100:.1f}%) | LOW={n_low:,} ({n_low/len(predictions)*100:.1f}%)")
    print(f"  LOW = <5 non-zero features (insufficient data)")

    # Extract target threshold from config
    target_threshold = 0.5
    if 'target' in config:
        target_threshold = config['target'].get('threshold', 0.5)
        target_model = config['target'].get('model', 'unknown')
        print(f"\n  Target model: {target_model}")
        print(f"  Target threshold: {target_threshold}")

    # Individual model predictions
    print(f"\n  Predicting with individual models...")
    individual_probs = {}
    for name, model in models.items():
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"  ERROR predicting with {name}: {e}")
            continue
        predictions[f'{name}_prob'] = proba
        individual_probs[name] = proba
        print(f"  {name}: mean={proba.mean():.4f}, median={np.median(proba):.4f}")

    # Ensemble computations (matching Bladder pipeline column names)
    if len(individual_probs) >= 2:
        print(f"\n  Computing ensemble predictions...")

        # Top 2 by AUC from config
        if 'test' in config:
            test_metrics = config['test']
            auc_scores = {}
            name_map = {
                'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost',
                'random_forest': 'Random Forest', 'logistic_regression': 'Logistic Regression',
                'lightgbm_dart': 'LightGBM (DART)', 'xgboost_deep': 'XGBoost (Deep)'
            }
            for short_name, display_name in name_map.items():
                if display_name in test_metrics and short_name in individual_probs:
                    auc_scores[short_name] = test_metrics[display_name].get('auc', 0)
            sorted_by_auc = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_by_auc) >= 2:
                top1, top2 = sorted_by_auc[0][0], sorted_by_auc[1][0]
                predictions['ensemble_top_2_prob'] = 0.5 * individual_probs[top1] + 0.5 * individual_probs[top2]
                print(f"    Top 2 ensemble: {top1} + {top2}")

        # Top 2 recall
        if 'test' in config:
            recall_scores = {}
            for short_name, display_name in name_map.items():
                if display_name in test_metrics and short_name in individual_probs:
                    recall_scores[short_name] = test_metrics[display_name].get('recall', 0)
            sorted_by_recall = sorted(recall_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_by_recall) >= 2:
                r1, r2 = sorted_by_recall[0][0], sorted_by_recall[1][0]
                predictions['ensemble_top_2_recall_prob'] = 0.5 * individual_probs[r1] + 0.5 * individual_probs[r2]

        # XGB+CB 40/60
        if 'xgboost' in individual_probs and 'catboost' in individual_probs:
            predictions['ensemble_xgb+cb_40/60_prob'] = (
                0.4 * individual_probs['xgboost'] + 0.6 * individual_probs['catboost']
            )

        # Avg Trees (tree-based models average)
        tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest',
                       'lightgbm_dart', 'xgboost_deep']
        tree_probs = [individual_probs[m] for m in tree_models if m in individual_probs]
        if tree_probs:
            predictions['ensemble_avg_trees_prob'] = np.mean(tree_probs, axis=0)

        # Optuna weighted (approximate from config if weights stored, else equal)
        if 'optuna_weights' in config:
            w = config['optuna_weights']
            ens = np.zeros(len(X))
            for m_name, weight in w.items():
                if m_name in individual_probs:
                    ens += weight * individual_probs[m_name]
            predictions['ensemble_optuna_wt_prob'] = ens
        else:
            # Fallback: use AUC-weighted average of all models
            if auc_scores:
                total_auc = sum(auc_scores.values())
                ens = np.zeros(len(X))
                for m_name, auc_val in auc_scores.items():
                    if m_name in individual_probs:
                        ens += (auc_val / total_auc) * individual_probs[m_name]
                predictions['ensemble_optuna_wt_prob'] = ens

    # Stacking meta-learner
    if 'stacking_meta_lr' in meta_models:
        print(f"  Computing stacking ensemble...")
        # Stack individual model probabilities as features for the meta-learner
        stack_cols = [f'{name}_prob' for name in models.keys() if f'{name}_prob' in predictions.columns]
        if stack_cols:
            meta_X = predictions[stack_cols].values
            try:
                stacking_prob = meta_models['stacking_meta_lr'].predict_proba(meta_X)[:, 1]
                predictions['ensemble_stacking_prob'] = stacking_prob
                print(f"    Stacking: mean={stacking_prob.mean():.4f}")
            except Exception as e:
                print(f"    Stacking failed: {e}")

    # Deployment meta v4 (primary ensemble)
    if 'deployment_meta_v4' in meta_models:
        print(f"  Computing deployment meta v4 ensemble...")
        stack_cols = [f'{name}_prob' for name in models.keys() if f'{name}_prob' in predictions.columns]
        if stack_cols:
            meta_X = predictions[stack_cols].values
            try:
                deploy_prob = meta_models['deployment_meta_v4'].predict_proba(meta_X)[:, 1]
                predictions['deployment_meta_v4_prob'] = deploy_prob
                print(f"    Deployment meta v4: mean={deploy_prob.mean():.4f}")
            except Exception as e:
                print(f"    Deployment meta v4 failed: {e}")

    # Flag columns using target threshold (LOW confidence → never flagged)
    prob_cols = [c for c in predictions.columns if c.endswith('_prob')]
    for col in prob_cols:
        name = col.replace('_prob', '')
        flag_col = f'{name}_flag'
        predictions[flag_col] = ((predictions[col] >= target_threshold) & (predictions['confidence'].values != 'LOW')).astype(int)

    return predictions


def compute_metrics(predictions, config):
    """Compute comprehensive metrics (same structure as Leukaemia version)."""

    if 'LABEL' not in predictions.columns:
        return None

    labels = predictions['LABEL'].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    n_total = len(labels)
    has_both = n_pos > 0 and n_neg > 0

    # Get target threshold
    target_threshold = 0.5
    if 'target' in config:
        target_threshold = config['target'].get('threshold', 0.5)

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  BLADDER CANCER — METRICS REPORT")
    lines.append(f"{'='*70}")
    lines.append(f"  Total patients: {n_total:,}")
    lines.append(f"  Cancer (LABEL=1): {n_pos:,}")
    lines.append(f"  Non-cancer (LABEL=0): {n_neg:,}")
    if 'target' in config:
        lines.append(f"  Target model: {config['target'].get('model', 'unknown')}")
        lines.append(f"  Target threshold: {target_threshold}")

    for col in predictions.columns:
        if not col.endswith('_prob'):
            continue
        name = col.replace('_prob', '')
        proba = predictions[col].values
        threshold = target_threshold
        preds = (proba >= threshold).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        lines.append(f"\n{'─'*70}")
        lines.append(f"  {name} (threshold={threshold:.4f})")
        lines.append(f"{'─'*70}")
        lines.append(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

        # Full metrics (both classes)
        if has_both:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc_roc = roc_auc_score(labels, proba)
            auc_pr = average_precision_score(labels, proba)
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)
            npv = tn / max(tn + fn, 1)
            f1 = 2 * ppv * sens / max(ppv + sens, 0.001)
            accuracy = (tp + tn) / n_total

            lines.append(f"\n  PERFORMANCE METRICS:")
            lines.append(f"    AUC-ROC:     {auc_roc:.4f}")
            lines.append(f"    AUC-PR:      {auc_pr:.4f}")
            lines.append(f"    Accuracy:    {accuracy:.1%}")
            lines.append(f"    Sensitivity: {sens:.1%}")
            lines.append(f"    Specificity: {spec:.1%}")
            lines.append(f"    PPV:         {ppv:.1%}")
            lines.append(f"    NPV:         {npv:.1%}")
            lines.append(f"    F1:          {f1:.4f}")

            lines.append(f"\n  OPERATING POINTS:")
            for target_sens in [0.85, 0.80, 0.75, 0.70]:
                best_spec = 0
                best_sens = 0
                for t in np.arange(0.01, 0.99, 0.003):
                    p = (proba >= t).astype(int)
                    s = ((p == 1) & (labels == 1)).sum() / max(n_pos, 1)
                    sp = ((p == 0) & (labels == 0)).sum() / max(n_neg, 1)
                    if s >= target_sens and sp > best_spec:
                        best_spec = sp
                        best_sens = s
                if best_spec > 0:
                    lines.append(f"    Sens>={target_sens:.0%}: Sens={best_sens:.1%} Spec={best_spec:.1%}")
                else:
                    lines.append(f"    Sens>={target_sens:.0%}: Not achievable")

        # Accuracy at threshold
        correct = (preds == labels).sum()
        lines.append(f"\n  ACCURACY AT THRESHOLD {threshold:.4f}:")
        lines.append(f"    Correctly predicted: {correct:,} / {n_total:,} ({correct/n_total*100:.1f}%)")
        if n_neg > 0:
            lines.append(f"    Non-cancer correctly cleared: {tn:,} / {n_neg:,} ({tn/n_neg*100:.1f}%)")
            lines.append(f"    Non-cancer incorrectly flagged: {fp:,} / {n_neg:,} ({fp/n_neg*100:.2f}%)")
        if n_pos > 0:
            lines.append(f"    Cancer correctly detected: {tp:,} / {n_pos:,} ({tp/n_pos*100:.1f}%)")
            lines.append(f"    Cancer missed: {fn:,} / {n_pos:,} ({fn/n_pos*100:.1f}%)")

        # Probability distribution
        lines.append(f"\n  PROBABILITY DISTRIBUTION:")
        lines.append(f"    Mean:   {proba.mean():.4f}")
        lines.append(f"    Median: {np.median(proba):.4f}")
        lines.append(f"    Std:    {proba.std():.4f}")
        for pct in [25, 50, 75, 90, 95, 99]:
            lines.append(f"    {pct}th percentile: {np.percentile(proba, pct):.4f}")
        lines.append(f"    Max: {proba.max():.4f}")

        if n_neg > 0:
            non_cancer_probs = proba[labels == 0]
            lines.append(f"\n  NON-CANCER PROBABILITY DISTRIBUTION:")
            lines.append(f"    Mean:   {non_cancer_probs.mean():.4f}  (ideal: close to 0)")
            lines.append(f"    Median: {np.median(non_cancer_probs):.4f}")
            for pct in [90, 95, 99]:
                lines.append(f"    {pct}th percentile: {np.percentile(non_cancer_probs, pct):.4f}")
            lines.append(f"    Max: {non_cancer_probs.max():.4f}")

        if n_pos > 0:
            cancer_probs = proba[labels == 1]
            lines.append(f"\n  CANCER PROBABILITY DISTRIBUTION:")
            lines.append(f"    Mean:   {cancer_probs.mean():.4f}  (ideal: close to 1)")
            lines.append(f"    Median: {np.median(cancer_probs):.4f}")
            for pct in [10, 25, 50]:
                lines.append(f"    {pct}th percentile: {np.percentile(cancer_probs, pct):.4f}")
            lines.append(f"    Min: {cancer_probs.min():.4f}")

        # Risk stratification
        lines.append(f"\n  RISK STRATIFICATION:")
        buckets = [
            ("Very Low (0-5%)", 0.00, 0.05),
            ("Low (5-10%)", 0.05, 0.10),
            ("Moderate (10-20%)", 0.10, 0.20),
            ("Elevated (20-30%)", 0.20, 0.30),
            ("High (30-50%)", 0.30, 0.50),
            ("Very High (50%+)", 0.50, 1.01),
        ]
        for label, lo, hi in buckets:
            bucket_mask = (proba >= lo) & (proba < hi)
            count = bucket_mask.sum()
            pct = count / n_total * 100
            if has_both or n_pos > 0 or n_neg > 0:
                cancer_in = (bucket_mask & (labels == 1)).sum()
                non_cancer_in = (bucket_mask & (labels == 0)).sum()
                lines.append(f"    {label:25s}: {count:>8,} ({pct:5.1f}%)  [Cancer: {cancer_in:,} | Non-cancer: {non_cancer_in:,}]")
            else:
                lines.append(f"    {label:25s}: {count:>8,} ({pct:5.1f}%)")

        # False positive rate at thresholds
        if n_neg > 0:
            lines.append(f"\n  FALSE POSITIVE RATE AT VARIOUS THRESHOLDS:")
            for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                flagged = ((proba >= t) & (labels == 0)).sum()
                cleared = n_neg - flagged
                lines.append(f"    Threshold {t:.2f}: {flagged:>8,} false positives ({flagged/n_neg*100:.2f}%) | {cleared:,} correctly cleared")

        # Confidence breakdown
        if 'confidence' in predictions.columns:
            lines.append(f"\n  RESULTS BY CONFIDENCE LEVEL:")
            for conf in ['HIGH', 'MEDIUM', 'LOW']:
                conf_mask = predictions['confidence'].values == conf
                conf_count = conf_mask.sum()
                if conf_count == 0:
                    continue
                conf_probs = proba[conf_mask]
                conf_labels = labels[conf_mask]
                conf_neg = (conf_labels == 0).sum()
                conf_pos = (conf_labels == 1).sum()
                lines.append(f"\n    {conf} CONFIDENCE ({conf_count:,} patients, {conf_pos:,} cancer, {conf_neg:,} non-cancer):")
                lines.append(f"      Avg probability: {conf_probs.mean():.4f}")
                lines.append(f"      Median probability: {np.median(conf_probs):.4f}")
                if conf_neg > 0:
                    conf_nc = conf_probs[conf_labels == 0]
                    lines.append(f"      Non-cancer avg: {conf_nc.mean():.4f}")
                    flagged_conf = ((conf_probs >= threshold) & (conf_labels == 0)).sum()
                    lines.append(f"      Flagged at threshold: {flagged_conf:,} ({flagged_conf/conf_count*100:.1f}%)")

        # Model quality summary
        lines.append(f"\n  MODEL QUALITY SUMMARY:")
        if n_neg > 0:
            non_cancer_probs = proba[labels == 0]
            avg_nc = non_cancer_probs.mean()
            pct_under_5 = (non_cancer_probs < 0.05).sum() / n_neg * 100
            pct_under_10 = (non_cancer_probs < 0.10).sum() / n_neg * 100
            pct_under_20 = (non_cancer_probs < 0.20).sum() / n_neg * 100
            lines.append(f"    Avg probability for non-cancer (ALL): {avg_nc:.4f}")
            lines.append(f"    Non-cancer with prob < 5%:  {pct_under_5:.1f}%")
            lines.append(f"    Non-cancer with prob < 10%: {pct_under_10:.1f}%")
            lines.append(f"    Non-cancer with prob < 20%: {pct_under_20:.1f}%")

            if 'confidence' in predictions.columns:
                high_mask = predictions['confidence'].values == 'HIGH'
                if high_mask.sum() > 0:
                    high_nc = proba[(labels == 0) & high_mask]
                    if len(high_nc) > 0:
                        lines.append(f"\n    HIGH CONFIDENCE non-cancer only ({len(high_nc):,} patients):")
                        lines.append(f"      Avg probability: {high_nc.mean():.4f} (ideal: close to 0)")
                        lines.append(f"      Prob < 10%: {(high_nc < 0.10).sum()/len(high_nc)*100:.1f}%")
                        lines.append(f"      Prob < 20%: {(high_nc < 0.20).sum()/len(high_nc)*100:.1f}%")
                        lines.append(f"      Prob < 30%: {(high_nc < 0.30).sum()/len(high_nc)*100:.1f}%")

        if n_pos > 0:
            cancer_probs = proba[labels == 1]
            avg_c = cancer_probs.mean()
            pct_over_50 = (cancer_probs >= 0.50).sum() / n_pos * 100
            pct_over_30 = (cancer_probs >= 0.30).sum() / n_pos * 100
            lines.append(f"    Avg probability for cancer: {avg_c:.4f} (ideal: close to 1)")
            lines.append(f"    Cancer with prob >= 50%: {pct_over_50:.1f}%")
            lines.append(f"    Cancer with prob >= 30%: {pct_over_30:.1f}%")

    lines.append(f"\n{'='*70}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Bladder Cancer — Predict on unseen patient data')
    parser.add_argument('--window', required=True, choices=['3mo', '6mo', '12mo'],
                        help='Prediction window (3mo, 6mo, or 12mo)')
    parser.add_argument('--feature_matrix', required=True,
                        help='Pre-built feature matrix CSV from FE pipeline')
    parser.add_argument('--output', default='predictions_unseen.csv',
                        help='Output filename (default: predictions_unseen.csv)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  BLADDER CANCER — PREDICT ON UNSEEN DATA")
    print(f"  Window: {args.window}")
    print(f"{'='*60}")

    print(f"\n  Loading models from: models/{args.window}/")
    models, meta_models, config, features = load_models_and_config(args.window)

    # Load feature matrix
    print(f"\n  Loading feature matrix: {args.feature_matrix}")
    fm = pd.read_csv(args.feature_matrix, index_col=0)

    # Check for label columns
    has_labels = 'LABEL' in fm.columns or 'ACTUAL_LABEL' in fm.columns
    label_col = 'LABEL' if 'LABEL' in fm.columns else ('ACTUAL_LABEL' if 'ACTUAL_LABEL' in fm.columns else None)

    print(f"  {fm.shape[0]:,} patients x {fm.shape[1]} features")
    if has_labels:
        labels = fm[label_col]
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        print(f"  Labels ({label_col}): {n_pos:,} cancer, {n_neg:,} non-cancer")
    else:
        print(f"  No LABEL column — predictions only (no metrics)")

    # Run predictions
    print(f"\n  Running predictions...")
    predictions = predict(models, meta_models, config, features, fm)

    # Save predictions
    output_path = SCRIPT_DIR / args.output
    predictions.to_csv(output_path, index=False)

    # Summary
    n_patients = len(predictions)
    print(f"\n{'='*60}")
    print(f"  PREDICTIONS SAVED")
    print(f"{'='*60}")
    print(f"  Patients: {n_patients:,}")

    for col in predictions.columns:
        if col.endswith('_flag'):
            model = col.replace('_flag', '')
            n_flagged = int(predictions[col].sum())
            print(f"  {model}: {n_flagged:,} flagged as at-risk ({n_flagged/n_patients*100:.1f}%)")

    print(f"  Saved to: {output_path}")

    # Compute and save metrics
    if has_labels:
        print(f"\n  Computing metrics...")
        report = compute_metrics(predictions, config)
        if report:
            print(report)
            report_path = SCRIPT_DIR / args.output.replace('.csv', '_metrics.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\n  Metrics saved to: {report_path}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
