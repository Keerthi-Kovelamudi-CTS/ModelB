# ══════════════════════════════════════════════════════════════
# RUN THIS DIAGNOSTIC SCRIPT — SHARE ALL OUTPUT WITH ME
# Uses same input as 1_run_modeling.py (cleaned CSV from cleanupfeatures).
# Output: prints to console; optional log in diagnosticresults/{window}/
# Usage: python 6_diagnostics.py [--window 3mo|6mo|12mo|12mo_250k]  (default: 12mo_250k)
# ══════════════════════════════════════════════════════════════

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLADDER_ROOT = os.path.dirname(SCRIPT_DIR)
parser = argparse.ArgumentParser(description='Run diagnostics on cleaned feature matrix (same input as 1_run_modeling).')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
args = parser.parse_args()
WINDOW = args.window

# Same input as 1_run_modeling.py
INPUT_PATH = os.path.join(BLADDER_ROOT, '2_Feature_Engineering', 'cleanupfeatures', WINDOW, f'bladder_feature_matrix_{WINDOW}_cleaned.csv')
if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(f"Cleaned matrix not found: {INPUT_PATH}. Run 5_feature_cleanup.py --window {WINDOW} first.")

# Results go to 3_Modeling/diagnosticresults/{window}/
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'diagnosticresults', WINDOW)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load feature matrix (same file as modeling)
df = pd.read_csv(INPUT_PATH)
print(f"Loaded: {INPUT_PATH} (window={WINDOW})")
print(f"Results dir: {RESULTS_DIR}")
y = df['LABEL'].values
X = df.drop(columns=['LABEL', 'PATIENT_GUID'])
feature_names = X.columns.tolist()

print("=" * 70)
print("DIAGNOSTIC 1: BASIC STATS")
print("=" * 70)
print(f"Total patients:  {len(df):,}")
print(f"Cancer:          {(y==1).sum():,}")
print(f"Non-cancer:      {(y==0).sum():,}")
print(f"Cancer rate:     {(y==1).mean()*100:.2f}%")
print(f"Total features:  {len(feature_names)}")
print(f"NaN total:       {X.isna().sum().sum():,}")
print(f"Inf total:       {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")

# Features that are constant (zero variance)
constant = [c for c in feature_names if X[c].nunique() <= 1]
print(f"Constant features (useless): {len(constant)}")

# Features that are >99% zero
mostly_zero = [c for c in feature_names if (X[c].fillna(0) == 0).mean() > 0.99]
print(f"Features >99% zero:          {len(mostly_zero)}")

# Features that are >95% NaN
mostly_nan = [c for c in feature_names if X[c].isna().mean() > 0.95]
print(f"Features >95% NaN:           {len(mostly_nan)}")


print("\n" + "=" * 70)
print("DIAGNOSTIC 2: UNIVARIATE AUC — TOP 50 FEATURES")
print("=" * 70)

aucs = {}
for c in feature_names:
    vals = X[c].fillna(0)
    if vals.nunique() <= 1:
        continue
    try:
        auc = roc_auc_score(y, vals)
        aucs[c] = max(auc, 1 - auc)
    except:
        pass

sorted_aucs = sorted(aucs.items(), key=lambda x: -x[1])
print(f"\n{'Rank':<6} {'Feature':<60} {'AUC':>8}")
print("-" * 76)
for rank, (feat, auc) in enumerate(sorted_aucs[:50], 1):
    print(f"{rank:<6} {feat:<60} {auc:>8.4f}")

# How many features have AUC > 0.55? (meaningful signal)
signal_features = [f for f, a in aucs.items() if a > 0.55]
weak_features = [f for f, a in aucs.items() if a <= 0.52]
print(f"\nFeatures with AUC > 0.55 (signal):  {len(signal_features)}")
print(f"Features with AUC ≤ 0.52 (noise):   {len(weak_features)}")
print(f"Features with AUC 0.52-0.55 (weak):  {len(aucs) - len(signal_features) - len(weak_features)}")


print("\n" + "=" * 70)
print("DIAGNOSTIC 3: TRAIN vs TEST GAP (OVERFITTING CHECK)")
print("=" * 70)

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.133, random_state=42, stratify=y_tr)

neg = (y_tr2 == 0).sum()
pos = (y_tr2 == 1).sum()
spw = neg / max(pos, 1)

model = xgb.XGBClassifier(
    n_estimators=2000, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    scale_pos_weight=spw, eval_metric='auc', random_state=42,
    tree_method='hist', early_stopping_rounds=50)
model.fit(X_tr2, y_tr2, eval_set=[(X_va, y_va)], verbose=False)

train_auc = roc_auc_score(y_tr2, model.predict_proba(X_tr2)[:, 1])
val_auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

print(f"Train AUC: {train_auc:.4f}")
print(f"Val AUC:   {val_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")
print(f"Gap (Train-Val):  {train_auc - val_auc:+.4f}")
print(f"Gap (Train-Test): {train_auc - test_auc:+.4f}")

if train_auc - test_auc > 0.05:
    print("���️ OVERFITTING — model memorizes train, fails on test")
    print("   → Need more regularization or fewer features")
elif train_auc - test_auc < 0.02:
    print("⚠️ UNDERFITTING — model can't learn enough from features")
    print("   → Data ceiling or features lack signal")
else:
    print("✅ Moderate gap — some room for improvement")


print("\n" + "=" * 70)
print("DIAGNOSTIC 4: FEATURE COUNT EXPERIMENT")
print("=" * 70)

# Train with ONLY top N features by univariate AUC
for top_n in [20, 50, 100, 150, 200, 300, 500, len(feature_names)]:
    top_n = min(top_n, len(sorted_aucs))
    top_feats = [f for f, _ in sorted_aucs[:top_n]]
    
    m = xgb.XGBClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, eval_metric='auc', random_state=42,
        tree_method='hist', early_stopping_rounds=50)
    m.fit(X_tr2[top_feats], y_tr2, eval_set=[(X_va[top_feats], y_va)], verbose=False)
    
    tr_auc = roc_auc_score(y_tr2, m.predict_proba(X_tr2[top_feats])[:, 1])
    te_auc = roc_auc_score(y_te, m.predict_proba(X_te[top_feats])[:, 1])
    
    print(f"  Top {top_n:4d} features: Train={tr_auc:.4f} Test={te_auc:.4f} Gap={tr_auc-te_auc:+.4f}")


print("\n" + "=" * 70)
print("DIAGNOSTIC 5: CANCER PATIENT ANALYSIS")
print("=" * 70)

# What do MISSED cancers look like?
y_pred_prob = model.predict_proba(X_te)[:, 1]
cancer_mask = y_te == 1
cancer_probs = y_pred_prob[cancer_mask]

print(f"Cancer patients in test: {cancer_mask.sum()}")
print(f"Cancer probability distribution:")
print(f"  Min:    {cancer_probs.min():.4f}")
print(f"  25th:   {np.percentile(cancer_probs, 25):.4f}")
print(f"  Median: {np.median(cancer_probs):.4f}")
print(f"  75th:   {np.percentile(cancer_probs, 75):.4f}")
print(f"  Max:    {cancer_probs.max():.4f}")

# At threshold 0.5
missed = cancer_probs < 0.5
caught = cancer_probs >= 0.5
print(f"\nAt threshold 0.5:")
print(f"  Caught: {caught.sum()} ({caught.mean()*100:.1f}%)")
print(f"  Missed: {missed.sum()} ({missed.mean()*100:.1f}%)")

# What scores do missed cancers get?
print(f"\nMissed cancer probability distribution:")
if missed.sum() > 0:
    missed_probs = cancer_probs[missed]
    print(f"  Min:    {missed_probs.min():.4f}")
    print(f"  25th:   {np.percentile(missed_probs, 25):.4f}")
    print(f"  Median: {np.median(missed_probs):.4f}")
    print(f"  75th:   {np.percentile(missed_probs, 75):.4f}")
    print(f"  Max:    {missed_probs.max():.4f}")

# How many missed cancers have haematuria?
X_te_df = pd.DataFrame(X_te, columns=feature_names)
cancer_test = X_te_df[cancer_mask].copy()
cancer_test['PRED_PROB'] = cancer_probs
cancer_test['MISSED'] = missed

haem_col = 'HAEM_ANY_FLAG' if 'HAEM_ANY_FLAG' in cancer_test.columns else None
if haem_col:
    print(f"\nMissed cancers WITH haematuria:    {cancer_test[cancer_test['MISSED']==True][haem_col].sum()}")
    print(f"Missed cancers WITHOUT haematuria: {(cancer_test[cancer_test['MISSED']==True][haem_col]==0).sum()}")
    print(f"Caught cancers WITH haematuria:    {cancer_test[cancer_test['MISSED']==False][haem_col].sum()}")
    print(f"Caught cancers WITHOUT haematuria: {(cancer_test[cancer_test['MISSED']==False][haem_col]==0).sum()}")


print("\n" + "=" * 70)
print("DIAGNOSTIC 6: FEATURE GROUP CONTRIBUTION")
print("=" * 70)

# Train with ONLY each feature group to see which groups matter
groups = {
    'Demographics': [c for c in feature_names if c.startswith(('AGE_', 'SEX_'))],
    'Haematuria': [c for c in feature_names if c.startswith('HAEM_')],
    'LUTS': [c for c in feature_names if c.startswith('LUTS_')],
    'Urine': [c for c in feature_names if c.startswith('URINE_')],
    'Lab Values': [c for c in feature_names if c.startswith('LAB_')],
    'Medications': [c for c in feature_names if c.startswith('MED_')],
    'Risk Factors': [c for c in feature_names if c.startswith('RF_')],
    'Comorbidities': [c for c in feature_names if c.startswith('COMORB_')],
    'Temporal': [c for c in feature_names if c.startswith(('TEMP_', 'OBS_6M'))],
    'Observations': [c for c in feature_names if c.startswith('OBS_') and not c.startswith('OBS_6M') and not c.startswith('OBS_CLUSTER')],
    'Syndromes': [c for c in feature_names if c.startswith('SYN_')],
    'Interactions': [c for c in feature_names if c.startswith('INT_')],
    'V5 features': [c for c in feature_names if c.startswith('V5_')],
    'V6 features': [c for c in feature_names if c.startswith('V6_')],
    'New Obs': [c for c in feature_names if c.startswith('NOBS_')],
    'Sequences': [c for c in feature_names if c.startswith('SEQ_')],
    'Pathways': [c for c in feature_names if c.startswith('PATH_')],
    'Recency': [c for c in feature_names if c.startswith('RECENCY_')],
    'Cross-lab': [c for c in feature_names if c.startswith('XLAB_')],
    'Diversity': [c for c in feature_names if c.startswith('DIV_')],
    'Ratios': [c for c in feature_names if c.startswith('RATIO_')],
    'Consultation': [c for c in feature_names if c.startswith('CONSULT_')],
}

print(f"\n{'Group':<25} {'N_feats':>8} {'Solo AUC':>10} {'Contribution':>14}")
print("-" * 60)

group_aucs = {}
for gname, gcols in groups.items():
    gcols = [c for c in gcols if c in feature_names]
    if len(gcols) == 0:
        continue
    
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, eval_metric='auc', random_state=42,
        tree_method='hist', early_stopping_rounds=30)
    
    try:
        m.fit(X_tr2[gcols], y_tr2, eval_set=[(X_va[gcols], y_va)], verbose=False)
        te_auc = roc_auc_score(y_te, m.predict_proba(X_te[gcols])[:, 1])
        group_aucs[gname] = te_auc
        contrib = "🔴 HIGH" if te_auc > 0.75 else "🟡 MED" if te_auc > 0.65 else "⚪ LOW"
        print(f"{gname:<25} {len(gcols):>8} {te_auc:>10.4f} {contrib:>14}")
    except Exception as e:
        print(f"{gname:<25} {len(gcols):>8} {'ERROR':>10} {str(e)[:30]}")


print("\n" + "=" * 70)
print("DIAGNOSTIC 7: HAEMATURIA SUBGROUP ANALYSIS")
print("=" * 70)

# The BIGGEST question: does the model work for patients WITHOUT haematuria?
if haem_col and haem_col in X_te_df.columns:
    for haem_val, label in [(1, 'WITH haematuria'), (0, 'WITHOUT haematuria')]:
        mask = X_te_df[haem_col] == haem_val
        if mask.sum() < 20:
            continue
        sub_y = y_te[mask.values]
        sub_p = y_pred_prob[mask.values]
        
        if len(np.unique(sub_y)) < 2:
            print(f"\n{label}: Only one class — can't compute AUC")
            continue
        
        sub_auc = roc_auc_score(sub_y, sub_p)
        sub_rec = recall_score(sub_y, (sub_p >= 0.5).astype(int))
        sub_prec = precision_score(sub_y, (sub_p >= 0.5).astype(int), zero_division=0)
        
        print(f"\n{label}:")
        print(f"  Patients: {mask.sum():,} (cancer: {sub_y.sum():,})")
        print(f"  AUC:       {sub_auc:.4f}")
        print(f"  Recall:    {sub_rec:.4f}")
        print(f"  Precision: {sub_prec:.4f}")


print("\n" + "=" * 70)
print("DIAGNOSTIC 8: DATA CEILING TEST")
print("=" * 70)

# If we add RANDOM NOISE features, does AUC change?
# If yes → model is overfitting to noise
np.random.seed(42)
noise_cols = pd.DataFrame(
    np.random.randn(len(X), 50),
    columns=[f'NOISE_{i}' for i in range(50)],
    index=X.index
)
X_with_noise = pd.concat([X, noise_cols], axis=1)

X_tr_n, X_te_n = X_with_noise.iloc[X_tr2.index], X_with_noise.iloc[X_te.index]
X_va_n = X_with_noise.iloc[X_va.index]

m_noise = xgb.XGBClassifier(
    n_estimators=2000, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    scale_pos_weight=spw, eval_metric='auc', random_state=42,
    tree_method='hist', early_stopping_rounds=50)
m_noise.fit(X_tr_n, y_tr2, eval_set=[(X_va_n, y_va)], verbose=False)
noise_auc = roc_auc_score(y_te, m_noise.predict_proba(X_te_n)[:, 1])

print(f"AUC without noise: {test_auc:.4f}")
print(f"AUC with noise:    {noise_auc:.4f}")
print(f"Difference:        {noise_auc - test_auc:+.4f}")

if noise_auc < test_auc - 0.005:
    print("⚠️ Noise HURTS — model is sensitive to noise features")
    print("   → You have TOO MANY features. REDUCE them.")
elif noise_auc > test_auc + 0.005:
    print("⚠️ Noise HELPS — model is severely underfitting")
    print("   → Very unusual. Check data integrity.")
else:
    print("✅ Noise doesn't matter — model is robust")
    print("   → Data has a natural ceiling around this AUC")


print("\n" + "=" * 70)
print("SHARE ALL THIS OUTPUT WITH ME")
print("=" * 70)

# Write summary to diagnosticresults/{window}/
summary_path = os.path.join(RESULTS_DIR, f'diagnostics_summary_{WINDOW}.txt')
with open(summary_path, 'w') as f:
    f.write(f"Diagnostics (window={WINDOW})\n")
    f.write(f"Input: {INPUT_PATH}\n")
    f.write(f"Patients: {len(df):,}  Cancer: {(y==1).sum():,}  Features: {len(feature_names)}\n")
    f.write(f"Train AUC: {train_auc:.4f}  Val AUC: {val_auc:.4f}  Test AUC: {test_auc:.4f}\n")
    f.write(f"Gap (Train-Test): {train_auc - test_auc:+.4f}\n")
    f.write(f"Constant features: {len(constant)}  >99%% zero: {len(mostly_zero)}  >95%% NaN: {len(mostly_nan)}\n")
    if group_aucs:
        f.write("\nGroup solo AUC:\n")
        for gname, a in sorted(group_aucs.items(), key=lambda x: -x[1]):
            f.write(f"  {gname}: {a:.4f}\n")
print(f"Summary saved to: {summary_path}")