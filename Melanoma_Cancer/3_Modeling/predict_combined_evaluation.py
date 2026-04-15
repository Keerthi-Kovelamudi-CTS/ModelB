# ═══════════════════════════════════════════════════════════════
# MELANOMA — COMBINED EVALUATION
# Run A: 277K non-cancer only → specificity, FPR, risk stratification
# Run B: Held-out test cancer + 277K non-cancer → full metrics (AUC, Sens, Spec)
#
# Usage:
#   python3 predict_combined_evaluation.py --window 3mo \
#     --unseen_predictions predictions_unseen.csv
#
# Takes the unseen predictions (from predict_unseen.py) and combines
# with held-out test cancer patients from training for Run B.
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(results_dir, window):
    """Load thresholds from saved config."""
    config_path = results_dir / window / 'saved_models' / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def compute_full_report(predictions, config, run_name):
    """Compute comprehensive metrics for a prediction set."""

    labels = predictions['LABEL'].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    n_total = len(labels)
    has_both = n_pos > 0 and n_neg > 0

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {run_name}")
    lines.append(f"{'='*70}")
    lines.append(f"  Total patients: {n_total:,}")
    lines.append(f"  Cancer (LABEL=1): {n_pos:,}")
    lines.append(f"  Non-cancer (LABEL=0): {n_neg:,}")
    if n_pos > 0 and n_neg > 0:
        lines.append(f"  Ratio: 1:{n_neg // max(n_pos, 1)}")

    prob_cols = [c for c in predictions.columns if c.endswith('_prob')]

    for col in prob_cols:
        name = col.replace('_prob', '')
        proba = predictions[col].values
        threshold = config.get(f'threshold_{name}', 0.5)
        preds = (proba >= threshold).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        lines.append(f"\n  {'─'*65}")
        lines.append(f"  {name} (threshold={threshold:.4f})")
        lines.append(f"  {'─'*65}")
        lines.append(f"  Confusion: TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

        # Accuracy
        correct = tp + tn
        lines.append(f"\n  ACCURACY: {correct:,} / {n_total:,} ({correct/n_total*100:.2f}%)")

        if has_both:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc_roc = roc_auc_score(labels, proba)
            auc_pr = average_precision_score(labels, proba)
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)
            npv = tn / max(tn + fn, 1)
            f1 = 2 * ppv * sens / max(ppv + sens, 0.001)

            lines.append(f"\n  FULL METRICS:")
            lines.append(f"    AUC-ROC:     {auc_roc:.4f}")
            lines.append(f"    AUC-PR:      {auc_pr:.4f}")
            lines.append(f"    Sensitivity: {sens:.1%} ({tp:,} / {n_pos:,} cancer detected)")
            lines.append(f"    Specificity: {spec:.1%} ({tn:,} / {n_neg:,} non-cancer cleared)")
            lines.append(f"    PPV:         {ppv:.1%} (of those flagged, {ppv*100:.1f}% actually have cancer)")
            lines.append(f"    NPV:         {npv:.1%} (of those cleared, {npv*100:.2f}% truly non-cancer)")
            lines.append(f"    F1:          {f1:.4f}")

            # Operating points
            lines.append(f"\n  OPERATING POINTS:")
            for target_sens in [0.85, 0.80, 0.75, 0.70]:
                best_spec, best_sens, best_t = 0, 0, 0
                best_ppv, best_npv = 0, 0
                for t in np.arange(0.01, 0.99, 0.003):
                    p = (proba >= t).astype(int)
                    _tp = ((p == 1) & (labels == 1)).sum()
                    _tn = ((p == 0) & (labels == 0)).sum()
                    _fp = ((p == 1) & (labels == 0)).sum()
                    _fn = ((p == 0) & (labels == 1)).sum()
                    s = _tp / max(_tp + _fn, 1)
                    sp = _tn / max(_tn + _fp, 1)
                    if s >= target_sens and sp > best_spec:
                        best_spec, best_sens, best_t = sp, s, t
                        best_ppv = _tp / max(_tp + _fp, 1)
                        best_npv = _tn / max(_tn + _fn, 1)
                if best_spec > 0:
                    lines.append(f"    Sens>={target_sens:.0%}: Sens={best_sens:.1%} Spec={best_spec:.1%} PPV={best_ppv:.2%} NPV={best_npv:.4%} (thresh={best_t:.3f})")
                else:
                    lines.append(f"    Sens>={target_sens:.0%}: Not achievable")

        elif n_neg > 0:
            spec = tn / max(tn + fp, 1)
            lines.append(f"\n  NON-CANCER METRICS:")
            lines.append(f"    Specificity: {spec:.1%}")
            lines.append(f"    False positive rate: {fp/n_neg*100:.2f}%")
            lines.append(f"    Flagged as at-risk: {fp:,} / {n_neg:,}")

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
            cancer_in = (bucket_mask & (labels == 1)).sum()
            non_cancer_in = (bucket_mask & (labels == 0)).sum()
            pct = count / n_total * 100
            lines.append(f"    {label:25s}: {count:>8,} ({pct:5.1f}%)  [Cancer: {cancer_in:,} | Non-cancer: {non_cancer_in:,}]")

        # Model quality
        if n_neg > 0:
            nc_probs = proba[labels == 0]
            lines.append(f"\n  MODEL QUALITY:")
            lines.append(f"    Non-cancer avg prob: {nc_probs.mean():.4f} (ideal: 0)")
            lines.append(f"    Non-cancer < 5%:    {(nc_probs < 0.05).sum()/n_neg*100:.1f}%")
            lines.append(f"    Non-cancer < 10%:   {(nc_probs < 0.10).sum()/n_neg*100:.1f}%")
            lines.append(f"    Non-cancer < 20%:   {(nc_probs < 0.20).sum()/n_neg*100:.1f}%")
        if n_pos > 0:
            c_probs = proba[labels == 1]
            lines.append(f"    Cancer avg prob:     {c_probs.mean():.4f} (ideal: 1)")
            lines.append(f"    Cancer >= 30%:      {(c_probs >= 0.30).sum()/n_pos*100:.1f}%")
            lines.append(f"    Cancer >= 50%:      {(c_probs >= 0.50).sum()/n_pos*100:.1f}%")

        # False positive table
        if n_neg > 0:
            lines.append(f"\n  FALSE POSITIVES AT THRESHOLDS:")
            for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                fp_t = ((proba >= t) & (labels == 0)).sum()
                if n_pos > 0:
                    tp_t = ((proba >= t) & (labels == 1)).sum()
                    lines.append(f"    Threshold {t:.2f}: FP={fp_t:>8,} ({fp_t/n_neg*100:.2f}%) | TP={tp_t:,}/{n_pos:,} ({tp_t/n_pos*100:.1f}%)")
                else:
                    lines.append(f"    Threshold {t:.2f}: FP={fp_t:>8,} ({fp_t/n_neg*100:.2f}%)")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Combined evaluation: non-cancer only + cancer+non-cancer')
    parser.add_argument('--window', required=True, choices=['3mo', '6mo', '12mo'])
    parser.add_argument('--unseen_predictions', required=True,
                        help='Predictions CSV from predict_unseen.py (277K non-cancer)')
    parser.add_argument('--results_dir', default=None,
                        help='Results directory with saved models and training predictions')
    parser.add_argument('--output', default='combined_evaluation.txt')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  MELANOMA — COMBINED EVALUATION")
    print(f"  Window: {args.window}")
    print(f"{'='*60}")

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        for rdir in ['results_text_emb_4seeds', 'results_clinical']:
            candidate = SCRIPT_DIR / rdir
            if (candidate / args.window / 'saved_models' / 'config.json').exists():
                results_dir = candidate
                break
        else:
            print("  ERROR: No saved models found.")
            sys.exit(1)

    config = load_config(results_dir, args.window)
    print(f"  Models from: {results_dir}")

    # Load unseen predictions (277K non-cancer)
    print(f"\n  Loading unseen predictions: {args.unseen_predictions}")
    unseen = pd.read_csv(args.unseen_predictions)
    print(f"  Unseen: {len(unseen):,} patients (Cancer: {(unseen['LABEL']==1).sum():,}, Non-cancer: {(unseen['LABEL']==0).sum():,})")

    # Load training test predictions (held-out cancer + non-cancer)
    training_pred_path = results_dir / args.window / f'predictions_{args.window}.csv'
    print(f"\n  Loading training test predictions: {training_pred_path}")
    training_pred = pd.read_csv(training_pred_path)
    training_cancer = training_pred[training_pred['LABEL'] == 1]
    print(f"  Training test set: {len(training_pred):,} total, {len(training_cancer):,} cancer (held-out, NOT used for training)")

    # Align columns — both need same probability columns
    prob_cols = [c for c in unseen.columns if c.endswith('_prob')]
    common_cols = ['PATIENT_GUID', 'LABEL'] + prob_cols
    flag_cols = [c for c in unseen.columns if c.endswith('_flag')]
    common_cols += flag_cols

    report_lines = []

    # ═══════════════════════════════════════════════════
    # RUN A: Non-cancer only (277K)
    # ═══════════════════════════════════════════════════
    print(f"\n  Running evaluation A: Non-cancer only...")
    run_a = compute_full_report(unseen, config, f"RUN A: NON-CANCER ONLY ({len(unseen):,} patients)")
    report_lines.append(run_a)
    print(run_a)

    # ═══════════════════════════════════════════════════
    # RUN B: Held-out cancer + 277K non-cancer
    # ═══════════════════════════════════════════════════
    # Only keep non-cancer from unseen (in case any have LABEL=1)
    unseen_non_cancer = unseen[unseen['LABEL'] == 0]

    # Combine: held-out cancer patients + 277K non-cancer
    # Ensure same columns — only use columns that exist in BOTH
    actual_common = [c for c in common_cols if c in training_cancer.columns and c in unseen_non_cancer.columns]
    training_cancer_aligned = training_cancer[actual_common].copy()
    unseen_aligned = unseen_non_cancer[actual_common].copy()

    combined = pd.concat([training_cancer_aligned, unseen_aligned], ignore_index=True)
    n_cancer = (combined['LABEL'] == 1).sum()
    n_non_cancer = (combined['LABEL'] == 0).sum()

    print(f"\n  Running evaluation B: Cancer + Non-cancer...")
    print(f"  Combined: {n_cancer:,} cancer + {n_non_cancer:,} non-cancer = {len(combined):,} total (ratio 1:{n_non_cancer//max(n_cancer,1)})")

    run_b = compute_full_report(combined, config,
        f"RUN B: HELD-OUT CANCER ({n_cancer:,}) + NON-CANCER ({n_non_cancer:,}) — REAL-WORLD RATIO 1:{n_non_cancer//max(n_cancer,1)}")
    report_lines.append(run_b)
    print(run_b)

    # ═══════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════
    output_path = SCRIPT_DIR / args.output
    full_report = '\n'.join(report_lines)
    with open(output_path, 'w') as f:
        f.write(full_report)

    # Also save combined predictions
    combined_pred_path = SCRIPT_DIR / args.output.replace('.txt', '_predictions.csv')
    combined.to_csv(combined_pred_path, index=False)

    print(f"\n{'='*60}")
    print(f"  SAVED:")
    print(f"    Report: {output_path}")
    print(f"    Combined predictions: {combined_pred_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
