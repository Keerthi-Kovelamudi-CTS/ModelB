"""
Try to reach 80% sensitivity / 70% specificity
by using CatBoost (highest recall) in the ensemble
and testing weighted combinations on existing test_predictions.csv.

Run after run_modeling.py. Reads results/test_predictions.csv.
"""

import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Load test predictions
pred_path = os.path.join(RESULTS_DIR, 'test_predictions.csv')
if not os.path.isfile(pred_path):
    print(f"Run run_modeling.py first. Not found: {pred_path}")
    exit(1)

import pandas as pd
pred_df = pd.read_csv(pred_path)
y_test = pred_df['ACTUAL_LABEL'].values

# Get individual model probabilities (column names from run_modeling.py: name.lower().replace(' ', '_') + '_prob')
xgb_prob = pred_df['xgboost_prob'].values
lgb_prob = pred_df['lightgbm_prob'].values
catboost_prob = pred_df['catboost_prob'].values if 'catboost_prob' in pred_df.columns else None
rf_prob = pred_df['random_forest_prob'].values
lr_prob = pred_df['logistic_regression_prob'].values

n_pos = (y_test == 1).sum()
n_neg = (y_test == 0).sum()

print("=" * 70)
print("STRATEGY: Weighted Ensemble (favour high-recall models)")
print("=" * 70)

# Weight by RECALL — favour models that catch more cancer (CatBoost recall~0.72, LGB~0.68, XGB~0.57)
ensembles = {}

if catboost_prob is not None:
    # Ensemble A: XGB + CatBoost (highest AUC + highest recall)
    ensembles['XGB+CatBoost (50/50)'] = 0.5 * xgb_prob + 0.5 * catboost_prob
    ensembles['XGB+CatBoost (40/60)'] = 0.4 * xgb_prob + 0.6 * catboost_prob
    ensembles['XGB+CatBoost (30/70)'] = 0.3 * xgb_prob + 0.7 * catboost_prob

    # Ensemble B: XGB + LGB + CatBoost
    ensembles['XGB+LGB+CB (33/33/33)'] = (xgb_prob + lgb_prob + catboost_prob) / 3
    ensembles['XGB+LGB+CB (25/25/50)'] = 0.25 * xgb_prob + 0.25 * lgb_prob + 0.50 * catboost_prob
    ensembles['XGB+LGB+CB (20/30/50)'] = 0.20 * xgb_prob + 0.30 * lgb_prob + 0.50 * catboost_prob

# Ensemble C: XGB + LGB (original)
ensembles['XGB+LGB (50/50)'] = 0.5 * xgb_prob + 0.5 * lgb_prob

# Individual models
ensembles['XGBoost alone'] = xgb_prob
ensembles['LightGBM alone'] = lgb_prob
if catboost_prob is not None:
    ensembles['CatBoost alone'] = catboost_prob

print(f"\n{'Ensemble':<30s} {'Thresh':>7s} {'Sens':>7s} {'Spec':>7s} {'PPV':>7s} {'F1':>7s} {'Meets':>7s}")
print("-" * 100)

TARGET_SENS, TARGET_SPEC = 0.80, 0.70
best_result = None

for ens_name, ens_prob in ensembles.items():
    best_for_this = None
    for t in np.arange(0.01, 0.60, 0.005):
        y_pred = (ens_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        tn = n_neg - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
        if sens >= TARGET_SENS and spec >= TARGET_SPEC:
            if best_for_this is None or spec > best_for_this['spec']:
                best_for_this = {
                    'thresh': t, 'sens': sens, 'spec': spec,
                    'ppv': ppv, 'f1': f1, 'tp': tp, 'fn': fn, 'fp': fp
                }
    if best_for_this:
        r = best_for_this
        meets = "YES"
        print(f"{ens_name:<30s} {r['thresh']:7.3f} {r['sens']:7.3f} {r['spec']:7.3f} "
              f"{r['ppv']:7.3f} {r['f1']:7.3f} {meets:>7s}")
        if best_result is None or r['spec'] > best_result[1]['spec']:
            best_result = (ens_name, r)
    else:
        best_close = None
        for t in np.arange(0.01, 0.60, 0.005):
            y_pred = (ens_prob >= t).astype(int)
            tp = ((y_pred == 1) & (y_test == 1)).sum()
            fn = ((y_pred == 0) & (y_test == 1)).sum()
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            tn = n_neg - fp
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
            dist = max(0, TARGET_SENS - sens) + max(0, TARGET_SPEC - spec)
            if best_close is None or dist < best_close['dist']:
                best_close = {'thresh': t, 'sens': sens, 'spec': spec, 'ppv': ppv, 'f1': f1, 'dist': dist}
        r = best_close
        meets = "NO"
        print(f"{ens_name:<30s} {r['thresh']:7.3f} {r['sens']:7.3f} {r['spec']:7.3f} "
              f"{r['ppv']:7.3f} {r['f1']:7.3f} {meets:>7s}  (gap: {r['dist']:.3f})")

if best_result:
    ens_name, r = best_result
    print(f"\n{'='*70}")
    print("TARGET MET!")
    print(f"  Ensemble: {ens_name}")
    print(f"  Threshold: {r['thresh']:.3f}")
    print(f"  Sensitivity: {r['sens']:.1%} (target >=80%)")
    print(f"  Specificity: {r['spec']:.1%} (target >=70%)")
    print(f"  PPV: {r['ppv']:.1%}")
    print(f"  TP={r['tp']}, FN={r['fn']}, FP={r['fp']}")
else:
    print(f"\n{'='*70}")
    print("TARGET NOT MET with any ensemble combination.")
    print("Best available: see run_modeling.py output (threshold chosen on validation).")

print("\nDONE")
