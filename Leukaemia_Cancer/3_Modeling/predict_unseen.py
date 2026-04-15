# ═══════════════════════════════════════════════════════════════
# LEUKAEMIA CANCER — PREDICT ON UNSEEN DATA
# Loads saved models and runs predictions on new patient data
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
SUFFIX_MAP = {'3mo': '3m', '6mo': '6m', '12mo': '12m'}


def load_models_and_config(results_dir, window):
    """Load saved models and config from a modeling run."""
    models_dir = results_dir / window / 'saved_models'

    if not models_dir.exists():
        print(f"  ERROR: No saved models found at {models_dir}")
        print(f"  Run the modeling script first to train and save models.")
        sys.exit(1)

    with open(models_dir / 'config.json') as f:
        config = json.load(f)

    models = {}
    for name, fname in [('XGBoost', 'xgboost_model.pkl'),
                         ('LightGBM', 'lightgbm_model.pkl'),
                         ('CatBoost', 'catboost_model.pkl')]:
        path = models_dir / fname
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  Loaded {name}")

    print(f"  Features: {len(config['features'])}")
    print(f"  Ensemble: {config['ensemble_top2'][0]} ({config['ensemble_weights'][0]:.2f}) + {config['ensemble_top2'][1]} ({config['ensemble_weights'][1]:.2f})")

    return models, config


def merge_text_features(feature_matrix, fe_results_dir, window):
    """Auto-merge text/TF-IDF/BERT features if they exist alongside the clean matrix."""
    fm = feature_matrix.copy()
    text_dirs = {
        'Text_Features': f'text_features_{window}.csv',
        'Text_Embeddings': f'text_embeddings_{window}.csv',
        'BERT_Embeddings': f'bert_embeddings_{window}.csv',
    }
    for subdir, fname in text_dirs.items():
        path = fe_results_dir / subdir / window / fname
        if path.exists():
            tf = pd.read_csv(path, index_col=0)
            # Only add columns not already in fm
            new_cols = [c for c in tf.columns if c not in fm.columns]
            if new_cols:
                fm = fm.join(tf[new_cols], how='left')
                print(f"  Merged {len(new_cols)} features from {subdir}")
    fm = fm.fillna(0)
    return fm


def predict(models, config, feature_matrix, zero_code_pids=None):
    """Run predictions using saved models."""

    features = config['features']

    available = [f for f in features if f in feature_matrix.columns]
    missing = [f for f in features if f not in feature_matrix.columns]

    # Categorize missing features
    text_missing = [f for f in missing if f.startswith(('TEXT_', 'EMB_', 'BERT_'))]
    clinical_missing = [f for f in missing if not f.startswith(('TEXT_', 'EMB_', 'BERT_'))]

    if missing:
        print(f"\n  Feature alignment: {len(available)}/{len(features)} available, {len(missing)} filled with 0")
        if text_missing:
            print(f"    Text/EMB/BERT missing: {len(text_missing)} (check if 6_text_features.py ran)")
        if clinical_missing:
            print(f"    Clinical missing: {len(clinical_missing)} (normal for non-cancer patients — these codes don't appear)")
            if len(clinical_missing) <= 20:
                for f in clinical_missing:
                    print(f"      {f}")

    X = pd.DataFrame(0, index=feature_matrix.index, columns=features)
    for f in available:
        X[f] = feature_matrix[f]

    # Smart NaN fill (same logic as 5_feature_cleanup.py)
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

    if 'LABEL' in feature_matrix.columns:
        predictions['LABEL'] = feature_matrix['LABEL'].values

    # ── ZERO-CODE PATIENT SUPPRESSION ──
    # Patients with zero approved codes have no clinical basis for prediction.
    # Force their probability to 0 regardless of model output.
    zero_mask = np.zeros(len(feature_matrix), dtype=bool)
    if zero_code_pids is not None and len(zero_code_pids) > 0:
        zero_mask = np.array(feature_matrix.index.astype(str).isin(zero_code_pids))
        n_zero = int(zero_mask.sum())
        n_has_codes = len(feature_matrix) - n_zero
        print(f"\n  Zero-code suppression: {n_zero:,} patients ({n_zero/len(feature_matrix)*100:.1f}%) have zero approved codes → prob forced to 0")
        print(f"  Patients with codes: {n_has_codes:,} ({n_has_codes/len(feature_matrix)*100:.1f}%)")

    # ── CONFIDENCE SCORE ──
    demo_cols = [c for c in X.columns if c in ['AGE_AT_INDEX', 'AGE_BAND']]
    clinical_non_zero = (X.drop(columns=demo_cols, errors='ignore') != 0).sum(axis=1).values

    predictions['non_zero_features'] = clinical_non_zero
    predictions['has_approved_codes'] = ~zero_mask

    predictions['confidence'] = 'HIGH'
    predictions.loc[zero_mask, 'confidence'] = 'NO_SIGNAL'
    predictions.loc[(~zero_mask) & (clinical_non_zero < 5), 'confidence'] = 'LOW'
    predictions.loc[(~zero_mask) & (clinical_non_zero >= 5) & (clinical_non_zero < 15), 'confidence'] = 'MEDIUM'

    n_nosig = (predictions['confidence'] == 'NO_SIGNAL').sum()
    n_low = (predictions['confidence'] == 'LOW').sum()
    n_med = (predictions['confidence'] == 'MEDIUM').sum()
    n_high = (predictions['confidence'] == 'HIGH').sum()
    print(f"\n  Confidence breakdown:")
    print(f"    NO_SIGNAL: {n_nosig:,} ({n_nosig/len(predictions)*100:.1f}%) — zero approved codes, forced to prob=0")
    print(f"    LOW:       {n_low:,} ({n_low/len(predictions)*100:.1f}%) — <5 non-zero features")
    print(f"    MEDIUM:    {n_med:,} ({n_med/len(predictions)*100:.1f}%) — 5-14 features")
    print(f"    HIGH:      {n_high:,} ({n_high/len(predictions)*100:.1f}%) — 15+ features")

    for name, model in models.items():
        raw_proba = model.predict_proba(X)[:, 1]
        proba = np.where(zero_mask, 0.0, raw_proba)
        predictions[f'{name}_prob'] = proba
        threshold = config.get(f'threshold_{name}', 0.5)
        predictions[f'{name}_flag'] = ((proba >= threshold) & (~zero_mask)).astype(int)
        tier = config.get(f'tier_{name}', 'unknown')
        print(f"  {name}: threshold={threshold:.4f} ({tier})")

    # Ensemble
    top2 = config['ensemble_top2']
    w1, w2 = config['ensemble_weights']
    if top2[0] in models and top2[1] in models:
        ens_prob = w1 * predictions[f'{top2[0]}_prob'] + w2 * predictions[f'{top2[1]}_prob']
        predictions['Ensemble_prob'] = ens_prob
        ens_threshold = config.get('threshold_Ensemble', 0.5)
        predictions['Ensemble_flag'] = ((ens_prob >= ens_threshold) & (~zero_mask)).astype(int)

    return predictions


def compute_metrics(predictions, config):
    """Compute comprehensive metrics."""

    if 'LABEL' not in predictions.columns:
        return None

    labels = predictions['LABEL'].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    n_total = len(labels)
    has_both = n_pos > 0 and n_neg > 0

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  METRICS REPORT")
    lines.append(f"{'='*70}")
    lines.append(f"  Total patients: {n_total:,}")
    lines.append(f"  Cancer (LABEL=1): {n_pos:,}")
    lines.append(f"  Non-cancer (LABEL=0): {n_neg:,}")

    for col in predictions.columns:
        if not col.endswith('_prob'):
            continue
        name = col.replace('_prob', '')
        proba = predictions[col].values
        threshold = config.get(f'threshold_{name}', 0.5)
        preds = (proba >= threshold).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        lines.append(f"\n{'─'*70}")
        lines.append(f"  {name} (threshold={threshold:.4f})")
        lines.append(f"{'─'*70}")
        lines.append(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

        # ── FULL METRICS (both classes) ──
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

        # ── ACCURACY (at threshold) ──
        correct = (preds == labels).sum()
        lines.append(f"\n  ACCURACY AT THRESHOLD {threshold:.4f}:")
        lines.append(f"    Correctly predicted: {correct:,} / {n_total:,} ({correct/n_total*100:.1f}%)")
        if n_neg > 0:
            lines.append(f"    Non-cancer correctly cleared: {tn:,} / {n_neg:,} ({tn/n_neg*100:.1f}%)")
            lines.append(f"    Non-cancer incorrectly flagged: {fp:,} / {n_neg:,} ({fp/n_neg*100:.2f}%)")
        if n_pos > 0:
            lines.append(f"    Cancer correctly detected: {tp:,} / {n_pos:,} ({tp/n_pos*100:.1f}%)")
            lines.append(f"    Cancer missed: {fn:,} / {n_pos:,} ({fn/n_pos*100:.1f}%)")

        # ── PROBABILITY DISTRIBUTION ──
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

        # ── RISK STRATIFICATION ──
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
            count = ((proba >= lo) & (proba < hi)).sum()
            pct = count / n_total * 100
            # If we have labels, show how many cancer vs non-cancer in each bucket
            if has_both or n_pos > 0 or n_neg > 0:
                bucket_mask = (proba >= lo) & (proba < hi)
                cancer_in = (bucket_mask & (labels == 1)).sum()
                non_cancer_in = (bucket_mask & (labels == 0)).sum()
                lines.append(f"    {label:25s}: {count:>8,} ({pct:5.1f}%)  [Cancer: {cancer_in:,} | Non-cancer: {non_cancer_in:,}]")
            else:
                lines.append(f"    {label:25s}: {count:>8,} ({pct:5.1f}%)")

        # ── FALSE POSITIVE RATE AT THRESHOLDS ──
        if n_neg > 0:
            lines.append(f"\n  FALSE POSITIVE RATE AT VARIOUS THRESHOLDS:")
            for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                flagged = ((proba >= t) & (labels == 0)).sum()
                cleared = n_neg - flagged
                lines.append(f"    Threshold {t:.2f}: {flagged:>8,} false positives ({flagged/n_neg*100:.2f}%) | {cleared:,} correctly cleared")

        # ── CONFIDENCE BREAKDOWN ──
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
                    conf_nc_probs = conf_probs[conf_labels == 0]
                    lines.append(f"      Non-cancer avg: {conf_nc_probs.mean():.4f}")
                    flagged_conf = (conf_probs >= threshold).sum()
                    lines.append(f"      Flagged at threshold: {flagged_conf:,} ({flagged_conf/conf_count*100:.1f}%)")

        # ── MODEL QUALITY SUMMARY ──
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

            # HIGH confidence only (reliable predictions)
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
    parser = argparse.ArgumentParser(description='Predict cancer risk on unseen patient data')
    parser.add_argument('--window', required=True, choices=['3mo', '6mo', '12mo'],
                        help='Prediction window (3mo, 6mo, or 12mo)')
    parser.add_argument('--feature_matrix', required=True,
                        help='Feature matrix CSV — use FE/feature_matrix_final_{window}.csv (NOT cleanup)')
    parser.add_argument('--results_dir', default=None,
                        help='Results directory with saved models (default: auto-detect)')
    parser.add_argument('--output', default='predictions_unseen.csv',
                        help='Output filename (default: predictions_unseen.csv)')
    parser.add_argument('--zero_code_file', default=None,
                        help='CSV of zero-code patient GUIDs (from preprocess). Patients listed here get prob=0.')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  LEUKAEMIA CANCER — PREDICT ON UNSEEN DATA")
    print(f"  Window: {args.window}")
    print(f"{'='*60}")

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        for rdir in ['results_text_emb_4seeds', 'results_clinical']:
            candidate = SCRIPT_DIR / rdir
            if (candidate / args.window / 'saved_models' / 'config.json').exists():
                results_dir = candidate
                print(f"  Auto-detected models: {rdir}")
                break
        else:
            print("  ERROR: No saved models found. Run modeling first.")
            sys.exit(1)

    print(f"\n  Loading models from: {results_dir}")
    models, config = load_models_and_config(results_dir, args.window)

    # Load feature matrix
    print(f"\n  Loading feature matrix: {args.feature_matrix}")
    fm = pd.read_csv(args.feature_matrix, index_col=0)
    print(f"  {fm.shape[0]:,} patients x {fm.shape[1]} features")

    # Sanitize column names (same as cleanup — XGBoost doesn't allow special chars)
    import re as _re
    fm.columns = [_re.sub(r'[\[\]<>{}/\\]', '_', str(c)) for c in fm.columns]
    fm = fm.loc[:, ~fm.columns.duplicated()]

    # Auto-merge text features if available
    fm_path = Path(args.feature_matrix)
    fe_results_dir = fm_path.parent.parent  # e.g., .../results/FE/3mo/ → .../results/
    print(f"\n  Merging text features from: {fe_results_dir}")
    fm = merge_text_features(fm, fe_results_dir, args.window)
    print(f"  After merge: {fm.shape[0]:,} patients x {fm.shape[1]} features")

    has_labels = 'LABEL' in fm.columns
    if has_labels:
        n_pos = (fm['LABEL'] == 1).sum()
        n_neg = (fm['LABEL'] == 0).sum()
        print(f"  Labels: {n_pos:,} cancer, {n_neg:,} non-cancer")
    else:
        print(f"  No LABEL column — predictions only")

    # Load zero-code patient list — patients with no approved codes get prob=0
    # (no clinical basis for prediction; only demographics → unreliable)
    zero_code_pids = set()
    if args.zero_code_file:
        zc_path = Path(args.zero_code_file)
    else:
        # Search common locations for zero_code_patients.csv
        candidates = [
            SCRIPT_DIR.parent / '4_ExpandedData_Test' / 'zero_code_patients.csv',
            SCRIPT_DIR.parent / '4_ExpandedData_Test' / 'fe_holdout_workspace' / 'fe_input' / 'zero_code_patients.csv',
        ]
        zc_path = next((p for p in candidates if p.exists()), candidates[0])
    if zc_path.exists():
        zero_code_pids = set(pd.read_csv(zc_path)['PATIENT_GUID'].astype(str))
        print(f"\n  Loaded zero-code list: {len(zero_code_pids):,} patients from {zc_path}")
    else:
        print(f"\n  No zero-code file found at {zc_path} — all patients scored normally")

    # Run predictions
    print(f"\n  Running predictions...")
    predictions = predict(models, config, fm, zero_code_pids=zero_code_pids)

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
