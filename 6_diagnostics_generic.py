# ══════════════════════════════════════════════════════════════
# GENERIC DIAGNOSTICS SCRIPT — Works for any cancer type
# Auto-detects feature groups from column prefixes.
#
# Two modes:
#   --mode full    : Run on ALL cleaned features (default)
#   --mode model   : Run on top N features only (matches actual modeling)
#
# Usage:
#   python 6_diagnostics_generic.py --cancer leukaemia --window 3mo
#   python 6_diagnostics_generic.py --cancer leukaemia --window 3mo --mode model
#   python 6_diagnostics_generic.py --cancer melanoma --window 12mo --top_n 200
    #  cd "/Users/keerthikovelamudi/Documents/Model B/Bladder_3"                                                                                                      
    # for w in 3mo 6mo 12mo; do                                              
    # python3 6_diagnostics_generic.py --cancer lymphoma --window $w --mode full                                                                                   
    # python3 6_diagnostics_generic.py --cancer lymphoma --window $w --mode model --top_n 225
    # done  
# ══════════════════════════════════════════════════════════════

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ── Config ──────────────────────────────────────────────────
CANCER_PATHS = {
    'bladder': {
        'root': 'Bladder_Cancer',
        'input': '2_Feature_Engineering/cleanupfeatures/{window}/bladder_feature_matrix_{window}_cleaned.csv',
    },
    'leukaemia': {
        'root': 'Leukaemia_Cancer',
        'input': '2_Feature_Engineering/results/Cleanup_Finalresults/{window}/feature_matrix_clean_{window}.csv',
    },
    'melanoma': {
        'root': 'Melanoma_Cancer',
        'input': '2_Feature_Engineering/results/Cleanup_Finalresults/{window}/feature_matrix_clean_{window}.csv',
    },
    'ovarian': {
        'root': 'Ovarian_Cancer',
        'input': '2_Feature_Engineering/results/Cleanup_Finalresults/{window}/feature_matrix_clean_{window}.csv',
    },
    'lymphoma': {
        'root': 'Lymphoma_Cancer',
        'input': '2_Feature_Engineering/results/Cleanup_Finalresults/{window}/feature_matrix_clean_{window}.csv',
    },
}

# Generic utilization prefixes to remove in "model" mode
# (matches is_clinical_feature() filter in modeling scripts)
GENERIC_PREFIXES = ['MONTHLY', 'ROLLING3M', 'RATE', 'VISIT', 'TEMP']
GENERIC_AGG = ['AGG_total_events', 'AGG_unique_categories', 'AGG_unique_code_ids',
               'AGG_unique_snomed_codes', 'AGG_events_A', 'AGG_events_B']
GENERIC_MED_AGG = ['MED_AGG_total_prescriptions', 'MED_AGG_unique_categories',
                   'MED_AGG_unique_drugs', 'MED_AGG_count_A', 'MED_AGG_count_B',
                   'MED_AGG_acceleration', 'MED_AGG_polypharmacy']

WINDOWS = ['3mo', '6mo', '12mo']

# ── Args ────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Run diagnostics on any cancer cleaned feature matrix.')
parser.add_argument('--cancer', required=True, choices=list(CANCER_PATHS.keys()), help='Cancer type')
parser.add_argument('--window', required=True, choices=WINDOWS, help='Time window')
parser.add_argument('--mode', choices=['full', 'model'], default='full',
                    help='full = all features; model = clinical filter + top N (default: full)')
parser.add_argument('--top_n', type=int, default=225,
                    help='Number of top features to select in model mode (default: 225)')
args = parser.parse_args()

CANCER = args.cancer
WINDOW = args.window
MODE = args.mode
TOP_N = args.top_n

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANCER_ROOT = os.path.join(SCRIPT_DIR, CANCER_PATHS[CANCER]['root'])
INPUT_PATH = os.path.join(SCRIPT_DIR, CANCER_PATHS[CANCER]['root'],
                          CANCER_PATHS[CANCER]['input'].format(window=WINDOW))

if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(f"Cleaned matrix not found: {INPUT_PATH}")

mode_label = f"model_top{TOP_N}" if MODE == 'model' else 'full'
RESULTS_DIR = os.path.join(CANCER_ROOT, 'diagnostics_results', f'{WINDOW}_{mode_label}')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Cancer: {CANCER.upper()}")
print(f"Window: {WINDOW}")
print(f"Mode:   {MODE.upper()}" + (f" (top {TOP_N})" if MODE == 'model' else " (all features)"))
print(f"Loaded: {INPUT_PATH}")

y = df['LABEL'].values
drop_cols = [c for c in ['LABEL', 'PATIENT_GUID'] if c in df.columns]
X = df.drop(columns=drop_cols)

# ── Model mode: apply clinical filter + top N ───────────────
if MODE == 'model':
    all_count = X.shape[1]

    # Step 1: Remove generic utilization features (matches modeling script)
    def is_generic(col):
        # Generic prefixes (MONTHLY_, ROLLING3M_, RATE_, VISIT_, TEMP_)
        # Note: VISITS_ (per-category) is kept, only VISIT_ (generic) removed
        if col.startswith('VISIT_'):  # generic visits
            return True
        for prefix in ['MONTHLY', 'ROLLING3M', 'RATE', 'TEMP']:
            if col.startswith(prefix + '_'):
                return True
        if col in GENERIC_AGG or col in GENERIC_MED_AGG:
            return True
        # Remove LAB _has_ever (just means had a blood test)
        if col.startswith('LAB_') and '_has_ever' in col:
            return True
        return False

    clinical_cols = [c for c in X.columns if not is_generic(c)]
    removed = all_count - len(clinical_cols)
    X = X[clinical_cols]
    print(f"\n  Clinical filter: {all_count} -> {len(clinical_cols)} features (removed {removed} generic)")

    # Step 2: Select top N by univariate AUC
    if len(clinical_cols) > TOP_N:
        uni_aucs = {}
        for c in clinical_cols:
            vals = X[c].fillna(0)
            if vals.nunique() <= 1:
                continue
            try:
                auc = roc_auc_score(y, vals)
                uni_aucs[c] = max(auc, 1 - auc)
            except:
                pass
        top_features = [f for f, _ in sorted(uni_aucs.items(), key=lambda x: -x[1])[:TOP_N]]
        X = X[top_features]
        print(f"  Top N selection: {len(clinical_cols)} -> {len(top_features)} features")

print(f"\nFinal features: {X.shape[1]}")
print(f"Results: {RESULTS_DIR}")

feature_names = X.columns.tolist()


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 1: BASIC STATS")
print("=" * 70)

print(f"Total patients:  {len(df):,}")
print(f"Cancer:          {(y==1).sum():,}")
print(f"Non-cancer:      {(y==0).sum():,}")
print(f"Cancer rate:     {(y==1).mean()*100:.2f}%")
print(f"Total features:  {len(feature_names)}")
print(f"NaN total:       {X.isna().sum().sum():,}")
print(f"Inf total:       {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")

constant = [c for c in feature_names if X[c].nunique() <= 1]
mostly_zero = [c for c in feature_names if (X[c].fillna(0) == 0).mean() > 0.99]
mostly_nan = [c for c in feature_names if X[c].isna().mean() > 0.95]

print(f"Constant features (useless): {len(constant)}")
print(f"Features >99% zero:          {len(mostly_zero)}")
print(f"Features >95% NaN:           {len(mostly_nan)}")

if constant:
    print(f"\n  Constant features: {constant[:20]}{'...' if len(constant) > 20 else ''}")


# ══════════════════════════════════════════════════════════════
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

signal_features = [f for f, a in aucs.items() if a > 0.55]
weak_features = [f for f, a in aucs.items() if a <= 0.52]
print(f"\nFeatures with AUC > 0.55 (signal):  {len(signal_features)}")
print(f"Features with AUC <= 0.52 (noise):  {len(weak_features)}")
print(f"Features with AUC 0.52-0.55 (weak): {len(aucs) - len(signal_features) - len(weak_features)}")


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 3: TRAIN vs TEST GAP (OVERFITTING CHECK)")
print("=" * 70)

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
    print("WARNING: OVERFITTING — model memorizes train, fails on test")
    print("   -> Need more regularization or fewer features")
elif train_auc - test_auc < 0.02:
    print("WARNING: UNDERFITTING — model can't learn enough from features")
    print("   -> Data ceiling or features lack signal")
else:
    print("OK: Moderate gap — some room for improvement")


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 4: FEATURE COUNT EXPERIMENT")
print("=" * 70)

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


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 5: CANCER PATIENT ANALYSIS")
print("=" * 70)

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

for thresh in [0.3, 0.4, 0.5]:
    caught = cancer_probs >= thresh
    missed = cancer_probs < thresh
    print(f"\nAt threshold {thresh}:")
    print(f"  Caught: {caught.sum()} ({caught.mean()*100:.1f}%)")
    print(f"  Missed: {missed.sum()} ({missed.mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 6: AUTO-DETECTED FEATURE GROUP CONTRIBUTION")
print("=" * 70)

# Auto-detect groups from column prefixes
prefix_groups = {}
for col in feature_names:
    parts = col.split('_')
    if len(parts) >= 2:
        prefix = parts[0]
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(col)

# Filter to groups with at least 3 features
groups = {k: v for k, v in prefix_groups.items() if len(v) >= 3}

print(f"\nAuto-detected {len(groups)} feature groups:\n")
print(f"{'Group':<25} {'N_feats':>8} {'Solo AUC':>10} {'Signal':>14}")
print("-" * 60)

group_aucs = {}
for gname in sorted(groups.keys()):
    gcols = groups[gname]

    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, eval_metric='auc', random_state=42,
        tree_method='hist', early_stopping_rounds=30)

    try:
        m.fit(X_tr2[gcols], y_tr2, eval_set=[(X_va[gcols], y_va)], verbose=False)
        te_auc = roc_auc_score(y_te, m.predict_proba(X_te[gcols])[:, 1])
        group_aucs[gname] = te_auc
        signal = "HIGH" if te_auc > 0.75 else "MEDIUM" if te_auc > 0.65 else "LOW"
        print(f"{gname:<25} {len(gcols):>8} {te_auc:>10.4f} {signal:>14}")
    except Exception as e:
        print(f"{gname:<25} {len(gcols):>8} {'ERROR':>10} {str(e)[:30]}")


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 7: DATA CEILING TEST (NOISE INJECTION)")
print("=" * 70)

np.random.seed(42)
noise_cols = pd.DataFrame(
    np.random.randn(len(X), 50),
    columns=[f'NOISE_{i}' for i in range(50)],
    index=X.index
)
X_with_noise = pd.concat([X, noise_cols], axis=1)

X_tr_n = X_with_noise.iloc[X_tr2.index]
X_te_n = X_with_noise.iloc[X_te.index]
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
    print("WARNING: Noise HURTS — model is sensitive to noise features")
    print("   -> You have TOO MANY features. REDUCE them.")
elif noise_auc > test_auc + 0.005:
    print("WARNING: Noise HELPS — model is severely underfitting")
    print("   -> Very unusual. Check data integrity.")
else:
    print("OK: Noise doesn't matter — model is robust")
    print("   -> Data has a natural ceiling around this AUC")


# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 8: CORRELATION CHECK — REDUNDANT FEATURES")
print("=" * 70)

# Find highly correlated feature pairs (>0.95)
numeric_X = X.select_dtypes(include=[np.number])
corr_matrix = numeric_X.corr().abs()

# Get upper triangle
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []
for col in upper.columns:
    for idx in upper.index:
        if upper.loc[idx, col] > 0.95:
            high_corr_pairs.append((idx, col, upper.loc[idx, col]))

high_corr_pairs.sort(key=lambda x: -x[2])
print(f"Feature pairs with correlation > 0.95: {len(high_corr_pairs)}")
if high_corr_pairs:
    print(f"\nTop 20 correlated pairs:")
    print(f"{'Feature 1':<40} {'Feature 2':<40} {'Corr':>6}")
    print("-" * 88)
    for f1, f2, corr in high_corr_pairs[:20]:
        print(f"{f1:<40} {f2:<40} {corr:>6.3f}")

# Count features that could be dropped (redundant)
to_drop = set()
for f1, f2, corr in high_corr_pairs:
    # Drop the one with lower univariate AUC
    auc1 = aucs.get(f1, 0.5)
    auc2 = aucs.get(f2, 0.5)
    to_drop.add(f2 if auc1 >= auc2 else f1)
print(f"\nRedundant features (could drop): {len(to_drop)}")


# ══════════════════════════════════════════════════════════════
# Save summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING SUMMARY")
print("=" * 70)

summary_path = os.path.join(RESULTS_DIR, f'diagnostics_summary_{CANCER}_{WINDOW}_{mode_label}.txt')
with open(summary_path, 'w') as f:
    f.write(f"Diagnostics: {CANCER.upper()} (window={WINDOW})\n")
    f.write(f"Input: {INPUT_PATH}\n")
    f.write(f"Patients: {len(df):,}  Cancer: {(y==1).sum():,}  Features: {len(feature_names)}\n")
    f.write(f"Cancer rate: {(y==1).mean()*100:.2f}%\n\n")

    f.write(f"Train AUC: {train_auc:.4f}  Val AUC: {val_auc:.4f}  Test AUC: {test_auc:.4f}\n")
    f.write(f"Gap (Train-Test): {train_auc - test_auc:+.4f}\n\n")

    f.write(f"Constant features: {len(constant)}\n")
    f.write(f">99% zero: {len(mostly_zero)}\n")
    f.write(f">95% NaN: {len(mostly_nan)}\n")
    f.write(f"Redundant (corr>0.95): {len(to_drop)}\n\n")

    f.write(f"Signal features (AUC>0.55): {len(signal_features)}\n")
    f.write(f"Noise features (AUC<=0.52): {len(weak_features)}\n\n")

    f.write(f"AUC without noise: {test_auc:.4f}\n")
    f.write(f"AUC with noise:    {noise_auc:.4f}\n\n")

    if group_aucs:
        f.write("Feature group solo AUC:\n")
        for gname, a in sorted(group_aucs.items(), key=lambda x: -x[1]):
            f.write(f"  {gname}: {a:.4f}\n")

    f.write(f"\nTop 30 features by univariate AUC:\n")
    for rank, (feat, auc) in enumerate(sorted_aucs[:30], 1):
        f.write(f"  {rank}. {feat}: {auc:.4f}\n")

print(f"Summary saved to: {summary_path}")
print("\n" + "=" * 70)
print(f"DONE — {CANCER.upper()} {WINDOW}")
print("=" * 70)
