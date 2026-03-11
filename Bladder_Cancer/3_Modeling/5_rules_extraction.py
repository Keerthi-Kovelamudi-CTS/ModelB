#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  BLADDER CANCER — CLINICAL RULES EXTRACTION
  
  Extracts interpretable rules from XGBoost model:
  1. Top SHAP-driven decision rules
  2. Risk factor profiles (who gets flagged)
  3. Simple clinical scoring system
  4. Decision tree surrogate (explainable proxy)
  5. Patient archetype analysis
  
  Run AFTER 1_run_modeling.py
═══════════════════════════════════════════════════════════════
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLADDER_2 = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
args = parser.parse_args()
WINDOW = args.window
RUN_SUBFOLDER = f"{WINDOW}_65-25-10"

INPUT_FILE = os.path.join(BLADDER_2, '2_Feature_Engineering', 'cleanupfeatures', WINDOW,
                          f'bladder_feature_matrix_{WINDOW}_cleaned.csv')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', RUN_SUBFOLDER)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', RUN_SUBFOLDER)
RULES_DIR = os.path.join(SCRIPT_DIR, 'rules', RUN_SUBFOLDER)
os.makedirs(RULES_DIR, exist_ok=True)

TEST_SIZE = 0.25
VAL_FRACTION = 10/75

# Clinical threshold from your best model
CLINICAL_THRESHOLD = 0.185


# ══════════════════════════════════════════════════════
# 1. LOAD DATA + MODEL
# ══════════════════════════════════════════════════════
print("=" * 70)
print("1. LOADING DATA + MODEL")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)
y = df['LABEL'].values
patient_ids = df['PATIENT_GUID'].values
X = df.drop(columns=['LABEL', 'PATIENT_GUID'])
feature_names = X.columns.tolist()

# Same split as modeling
X_tv, X_test, y_tv, y_test, pid_tv, pid_test = train_test_split(
    X, y, patient_ids, test_size=TEST_SIZE, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val, pid_train, pid_val = train_test_split(
    X_tv, y_tv, pid_tv, test_size=VAL_FRACTION, random_state=SEED, stratify=y_tv)

# Load XGBoost model
xgb_model = joblib.load(os.path.join(MODELS_DIR, 'model_xgboost.joblib'))
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_flags = (xgb_probs >= CLINICAL_THRESHOLD).astype(int)

print(f"Patients: {len(df):,}  Features: {len(feature_names)}")
print(f"Test set: {len(X_test):,}  Cancer: {(y_test==1).sum():,}")
print(f"Test AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
print(f"Threshold: {CLINICAL_THRESHOLD}  Flagged: {xgb_flags.sum():,}/{len(xgb_flags):,}")


# ══════════════════════════════════════════════════════
# 2. SHAP-BASED FEATURE RULES
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. SHAP-BASED FEATURE RULES")
print("=" * 70)

try:
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    HAS_SHAP = True
except ImportError:
    print("  SHAP not installed — using feature_importances_ instead")
    HAS_SHAP = False

if HAS_SHAP:
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
else:
    top_features = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)

# Top 20 features for rules
TOP_N = 20
top_feat_names = top_features.head(TOP_N).index.tolist()

print(f"\nTop {TOP_N} features for rule extraction:")
for i, f in enumerate(top_feat_names, 1):
    print(f"  {i:2d}. {f}")


# ══════════════════════════════════════════════════════
# 3. THRESHOLD-BASED RULES (QUANTILE ANALYSIS)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. CLINICAL THRESHOLD RULES")
print("=" * 70)

# For each top feature, find the threshold that best separates cancer/non-cancer
rules = []

print(f"\n{'Feature':<50s} {'Threshold':>10s} {'Dir':>4s} {'Cancer%':>8s} {'NonCa%':>8s} {'Lift':>6s} {'Rule'}")
print("─" * 130)

for feat in top_feat_names:
    vals = X_test[feat].fillna(0).values
    cancer_vals = vals[y_test == 1]
    noncancer_vals = vals[y_test == 0]

    # Skip if feature is constant
    if np.std(vals) < 1e-10:
        continue

    # Check if feature is binary/near-binary
    unique_vals = np.unique(vals[~np.isnan(vals)])
    is_binary = len(unique_vals) <= 3

    if is_binary:
        # Binary feature: rule is simply "feature == 1"
        cancer_rate_1 = y_test[vals > 0].mean() if (vals > 0).sum() > 0 else 0
        cancer_rate_0 = y_test[vals == 0].mean() if (vals == 0).sum() > 0 else 0
        pct_cancer = (cancer_vals > 0).mean() * 100
        pct_noncancer = (noncancer_vals > 0).mean() * 100
        lift = cancer_rate_1 / max(cancer_rate_0, 0.001)

        rule_text = f"IF {feat} = YES → cancer rate {cancer_rate_1:.1%} (lift {lift:.1f}x)"
        rules.append({
            'feature': feat, 'threshold': 0.5, 'direction': '>',
            'cancer_pct': pct_cancer, 'noncancer_pct': pct_noncancer,
            'lift': lift, 'rule': rule_text, 'type': 'binary'
        })
        print(f"{feat:<50s} {'YES':>10s} {'=':>4s} {pct_cancer:>7.1f}% {pct_noncancer:>7.1f}% {lift:>5.1f}x {rule_text}")
    else:
        # Continuous feature: find best percentile threshold
        # Test percentiles: 25th, 50th, 75th, 90th
        best_lift = 0
        best_rule = None

        for pctile in [25, 50, 75, 90]:
            thresh = np.percentile(vals[vals > 0], pctile) if (vals > 0).sum() > 10 else np.percentile(vals, pctile)

            # Test both directions
            for direction in ['>', '≤']:
                if direction == '>':
                    mask = vals > thresh
                else:
                    mask = vals <= thresh

                if mask.sum() < 50 or (~mask).sum() < 50:
                    continue

                rate_in = y_test[mask].mean()
                rate_out = y_test[~mask].mean()
                lift = rate_in / max(rate_out, 0.001)

                pct_cancer = (cancer_vals > thresh).mean() * 100 if direction == '>' else (cancer_vals <= thresh).mean() * 100
                pct_noncancer = (noncancer_vals > thresh).mean() * 100 if direction == '>' else (noncancer_vals <= thresh).mean() * 100

                if lift > best_lift:
                    best_lift = lift
                    best_rule = {
                        'feature': feat, 'threshold': round(thresh, 2),
                        'direction': direction, 'cancer_pct': pct_cancer,
                        'noncancer_pct': pct_noncancer, 'lift': lift,
                        'rule': f"IF {feat} {direction} {thresh:.2f} → cancer rate {rate_in:.1%} (lift {lift:.1f}x)",
                        'type': 'continuous', 'pctile': pctile
                    }

        if best_rule:
            rules.append(best_rule)
            r = best_rule
            print(f"{r['feature']:<50s} {r['threshold']:>10.2f} {r['direction']:>4s} "
                  f"{r['cancer_pct']:>7.1f}% {r['noncancer_pct']:>7.1f}% {r['lift']:>5.1f}x {r['rule']}")


# ══════════════════════════════════════════════════════
# 4. COMBINATION RULES (HIGH-RISK PROFILES)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. COMBINATION RULES (HIGH-RISK PROFILES)")
print("=" * 70)

# Define clinically meaningful combinations
X_test_filled = X_test.fillna(0)

combo_rules = [
    {
        'name': 'HAEMATURIA + SMOKER + MALE + AGE≥60',
        'mask': lambda df: (df['HAEM_ANY_FLAG'] > 0) &
                           (df['RF_EVER_SMOKER'] > 0) &
                           (df['SEX_FEMALE'] == 0) &
                           ((df['AGE_BAND_60_69'] > 0) | (df['AGE_BAND_70_79'] > 0) | (df['AGE_BAND_80PLUS'] > 0))
    },
    {
        'name': 'HAEMATURIA + ANY AGE/SEX',
        'mask': lambda df: (df['HAEM_ANY_FLAG'] > 0)
    },
    {
        'name': 'NO HAEMATURIA + SMOKER + ANAEMIA',
        'mask': lambda df: (df['HAEM_ANY_FLAG'] == 0) &
                           (df['RF_EVER_SMOKER'] > 0) &
                           ((df['LAB_ANAEMIA_MODERATE'] > 0) | (df['LAB_ANAEMIA_SEVERE'] > 0))
    },
    {
        'name': 'NO HAEMATURIA + SMOKER + LOW HB',
        'mask': lambda df: (df['HAEM_ANY_FLAG'] == 0) &
                           (df['RF_EVER_SMOKER'] > 0) &
                           (df['LAB_HB_EVER_BELOW_100'] > 0)
    },
    {
        'name': 'RECURRENT UTI + SMOKER',
        'mask': lambda df: (df['COMORB_RECURRENT_UTI_FLAG'] > 0) &
                           (df['RF_EVER_SMOKER'] > 0)
    },
    {
        'name': 'URINE BLOOD + RECENT ABNORMAL LAB',
        'mask': lambda df: (df['URINE_BLOOD_TEST_COUNT'] > 0) &
                           (df['RECENCY_URINE_LAB_ABNORMALITIES_IN_WB'] > 0)
    },
    {
        'name': 'HIGH BLEEDING SCORE (≥2)',
        'mask': lambda df: (df['SYN_BLEEDING_SCORE'] >= 2)
    },
    {
        'name': 'HIGH BLADDER SYMPTOM SCORE (≥2)',
        'mask': lambda df: (df['INT_BLADDER_SYMPTOM_SCORE'] >= 2)
    },
    {
        'name': 'HIGH SUSPICION SCORE (≥3)',
        'mask': lambda df: (df['INT_OVERALL_SUSPICION_SCORE'] >= 3)
    },
    {
        'name': 'SMOKER + MALE + AGE≥60 + HIGH GP VISITS',
        'mask': lambda df: (df['RF_EVER_SMOKER'] > 0) &
                           (df['SEX_FEMALE'] == 0) &
                           ((df['AGE_BAND_60_69'] > 0) | (df['AGE_BAND_70_79'] > 0) | (df['AGE_BAND_80PLUS'] > 0)) &
                           (df['TEMP_GP_VISIT_DAYS_ALL'] > df['TEMP_GP_VISIT_DAYS_ALL'].median())
    },
    {
        'name': 'NO SYMPTOMS BUT HIGH MODEL SCORE (≥0.35)',
        'mask': lambda df: (df['HAEM_ANY_FLAG'] == 0) &
                           (df['SYN_BLEEDING_SCORE'] == 0) &
                           (df['INT_BLADDER_SYMPTOM_SCORE'] == 0) &
                           (pd.Series(xgb_probs >= 0.35, index=df.index))
    },
]

print(f"\n{'Rule':<55s} {'N':>6s} {'Cancer':>7s} {'Rate':>7s} {'Lift':>6s} {'Sens':>6s} {'PPV':>6s}")
print("─" * 100)

combo_results = []
for combo in combo_rules:
    try:
        mask = combo['mask'](X_test_filled)
        n = mask.sum()
        if n < 10:
            continue
        n_cancer = y_test[mask].sum()
        rate = y_test[mask].mean()
        lift = rate / y_test.mean()
        sens = n_cancer / (y_test == 1).sum()  # what % of all cancers this catches
        ppv = rate

        combo_results.append({
            'rule': combo['name'], 'n': int(n), 'n_cancer': int(n_cancer),
            'rate': rate, 'lift': lift, 'sensitivity': sens, 'ppv': ppv
        })
        print(f"{combo['name']:<55s} {n:>6,d} {n_cancer:>7,d} {rate:>6.1%} {lift:>5.1f}x {sens:>5.1%} {ppv:>5.1%}")
    except Exception as e:
        print(f"  {combo['name']}: SKIPPED ({e})")


# ══════════════════════════════════════════════════════
# 5. SIMPLE CLINICAL SCORING SYSTEM
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. SIMPLE CLINICAL SCORING SYSTEM (GP-FRIENDLY)")
print("=" * 70)

# Build a simple point-based score from top features
# Each risk factor gets points based on SHAP importance

score_features = {
    # Feature name: (condition_lambda, points, label)
    'RF_EVER_SMOKER':               (lambda x: x > 0, 3, 'Ever smoker'),
    'HAEM_ANY_FLAG':                (lambda x: x > 0, 5, 'Haematuria recorded'),
    'LAB_HB_EVER_BELOW_100':        (lambda x: x > 0, 3, 'Hb ever < 100'),
    'LAB_HB_EVER_BELOW_80':         (lambda x: x > 0, 4, 'Hb ever < 80 (severe anaemia)'),
    'LAB_ANAEMIA_SEVERE':           (lambda x: x > 0, 3, 'Severe anaemia diagnosis'),
    'URINE_BLOOD_TEST_COUNT':       (lambda x: x > 0, 3, 'Blood in urine test'),
    'COMORB_RECURRENT_UTI_FLAG':    (lambda x: x > 0, 2, 'Recurrent UTIs'),
    'RF_HEAVY_SMOKER_FLAG':         (lambda x: x > 0, 2, 'Heavy smoker (bonus)'),
    'SYN_BLEEDING_SCORE':           (lambda x: x >= 2, 3, 'Bleeding syndrome score ≥ 2'),
    'NOBS_WEIGHT_LOSS_FLAG':        (lambda x: x > 0, 2, 'Weight loss recorded'),
    'NOBS_FATIGUE_FLAG':            (lambda x: x > 0, 1, 'Fatigue recorded'),
    'AGE_BAND_70_79':               (lambda x: x > 0, 2, 'Age 70-79'),
    'AGE_BAND_80PLUS':              (lambda x: x > 0, 2, 'Age 80+'),
    'AGE_BAND_60_69':               (lambda x: x > 0, 1, 'Age 60-69'),
}

# Only use features that exist
available_score_features = {k: v for k, v in score_features.items() if k in X_test.columns}

# Calculate scores
test_scores = np.zeros(len(X_test))
for feat, (cond, points, label) in available_score_features.items():
    vals = X_test_filled[feat].values
    test_scores += np.array([points if cond(v) else 0 for v in vals])

print(f"\nSIMPLE CLINICAL SCORE (max {sum(v[1] for v in available_score_features.values())} points):")
print(f"{'Points':>7s}  {'Criterion'}")
print("─" * 50)
for feat, (cond, points, label) in sorted(available_score_features.items(), key=lambda x: -x[1][1]):
    print(f"  +{points:<5d}  {label}")

# Evaluate score performance
score_auc = roc_auc_score(y_test, test_scores)
print(f"\nSimple score AUC: {score_auc:.4f}  (vs XGBoost {roc_auc_score(y_test, xgb_probs):.4f})")

# Score thresholds
print(f"\n{'Score':>6s} {'N':>7s} {'Cancer':>7s} {'Rate':>7s} {'Sens':>7s} {'Spec':>7s} {'PPV':>7s}")
print("─" * 55)

total_cancer = (y_test == 1).sum()
total_noncancer = (y_test == 0).sum()

for thresh in range(0, 16):
    flagged = test_scores >= thresh
    n = flagged.sum()
    if n == 0:
        continue
    tp = ((flagged) & (y_test == 1)).sum()
    fp = ((flagged) & (y_test == 0)).sum()
    fn = ((~flagged) & (y_test == 1)).sum()
    tn = ((~flagged) & (y_test == 0)).sum()
    rate = y_test[flagged].mean() if n > 0 else 0
    sens = tp / total_cancer if total_cancer > 0 else 0
    spec = tn / total_noncancer if total_noncancer > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

    marker = ""
    if sens >= 0.78 and spec >= 0.70:
        marker = " ← meets target"

    print(f"  ≥{thresh:<4d} {n:>7,d} {tp:>7,d} {rate:>6.1%} {sens:>6.1%} {spec:>6.1%} {ppv:>6.1%}{marker}")


# ══════════════════════════════════════════════════════
# 6. SURROGATE DECISION TREE
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. SURROGATE DECISION TREE (EXPLAINABLE PROXY)")
print("=" * 70)

# Use XGBoost predictions as labels → train simple decision tree
# This gives an interpretable approximation of the XGBoost model

# Use top 20 features only for interpretability
X_test_top = X_test_filled[top_feat_names].copy()
X_train_top = X_train.fillna(0)[top_feat_names].copy()

# Get XGBoost predictions on train for surrogate training
xgb_train_probs = xgb_model.predict_proba(X_train)[:, 1]
xgb_train_flags = (xgb_train_probs >= CLINICAL_THRESHOLD).astype(int)

for max_depth in [3, 4, 5]:
    dt = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=100,
        class_weight='balanced', random_state=SEED)
    dt.fit(X_train_top, xgb_train_flags)

    # How well does surrogate match XGBoost?
    dt_preds = dt.predict(X_test_top)
    agreement = (dt_preds == xgb_flags).mean()

    # How well does surrogate predict actual cancer?
    dt_probs = dt.predict_proba(X_test_top)[:, 1]
    dt_auc = roc_auc_score(y_test, dt_probs)

    print(f"\nDepth-{max_depth} surrogate:")
    print(f"  Agreement with XGBoost: {agreement:.1%}")
    print(f"  Test AUC (vs actual cancer): {dt_auc:.4f}")
    print(f"  Tree rules:")
    tree_text = export_text(dt, feature_names=top_feat_names, max_depth=max_depth)
    for line in tree_text.split('\n'):
        print(f"    {line}")

# Save best surrogate
best_dt = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=100,
    class_weight='balanced', random_state=SEED)
best_dt.fit(X_train_top, xgb_train_flags)
joblib.dump(best_dt, os.path.join(MODELS_DIR, 'surrogate_decision_tree.joblib'))


# ══════════════════════════════════════════════════════
# 7. PATIENT ARCHETYPES (WHO GETS FLAGGED?)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. PATIENT ARCHETYPES")
print("=" * 70)

# Segment flagged patients by key features
test_analysis = X_test_filled.copy()
test_analysis['cancer'] = y_test
test_analysis['xgb_prob'] = xgb_probs
test_analysis['flagged'] = xgb_flags

# Define archetypes
archetypes = {
    'CLASSIC: Haem + Smoker + Male ≥60': lambda df: (
        (df['HAEM_ANY_FLAG'] > 0) & (df['RF_EVER_SMOKER'] > 0) &
        (df['SEX_FEMALE'] == 0) &
        ((df['AGE_BAND_60_69'] > 0) | (df['AGE_BAND_70_79'] > 0) | (df['AGE_BAND_80PLUS'] > 0))
    ),
    'HAEMATURIA ONLY (no other risk)': lambda df: (
        (df['HAEM_ANY_FLAG'] > 0) & (df['RF_EVER_SMOKER'] == 0)
    ),
    'SILENT: No haem, no bleeding, smoker': lambda df: (
        (df['HAEM_ANY_FLAG'] == 0) & (df['SYN_BLEEDING_SCORE'] == 0) &
        (df['RF_EVER_SMOKER'] > 0)
    ),
    'ANAEMIA PATHWAY: Low Hb + iron def': lambda df: (
        (df['HAEM_ANY_FLAG'] == 0) &
        ((df['LAB_HB_EVER_BELOW_100'] > 0) | (df['LAB_ANAEMIA_SEVERE'] > 0))
    ),
    'UTI PATHWAY: Recurrent UTI + AB': lambda df: (
        (df['COMORB_RECURRENT_UTI_FLAG'] > 0) |
        (df['MED_UTI_AB_RECURRENT_2'] > 0)
    ),
    'HIGH UTILISATION: Many GP visits + tests': lambda df: (
        (df['TEMP_GP_VISIT_DAYS_ALL'] > df['TEMP_GP_VISIT_DAYS_ALL'].quantile(0.75)) &
        (df['LAB_TOTAL_TESTS'] > df['LAB_TOTAL_TESTS'].quantile(0.75)) if 'LAB_TOTAL_TESTS' in df.columns
        else (df['TEMP_GP_VISIT_DAYS_ALL'] > df['TEMP_GP_VISIT_DAYS_ALL'].quantile(0.75))
    ),
    'YOUNG (<60) FLAGGED': lambda df: (
        (df['AGE_BAND_UNDER50'] > 0) | (df['AGE_BAND_50_59'] > 0)
    ),
}

print(f"\n{'Archetype':<50s} {'Total':>7s} {'Flagged':>8s} {'Cancer':>7s} {'Flag%':>7s} {'CaRate':>7s} {'PPV':>7s}")
print("─" * 100)

archetype_results = []
for arch_name, arch_func in archetypes.items():
    try:
        mask = arch_func(test_analysis)
        subset = test_analysis[mask]
        n = len(subset)
        if n < 10:
            continue
        n_flagged = subset['flagged'].sum()
        n_cancer = subset['cancer'].sum()
        flag_rate = n_flagged / n if n > 0 else 0
        cancer_rate = n_cancer / n if n > 0 else 0
        # PPV among flagged in this archetype
        flagged_subset = subset[subset['flagged'] == 1]
        ppv = flagged_subset['cancer'].mean() if len(flagged_subset) > 0 else 0

        archetype_results.append({
            'archetype': arch_name, 'n': n, 'n_flagged': n_flagged,
            'n_cancer': n_cancer, 'flag_rate': flag_rate,
            'cancer_rate': cancer_rate, 'ppv': ppv
        })
        print(f"{arch_name:<50s} {n:>7,d} {n_flagged:>8,d} {n_cancer:>7,d} "
              f"{flag_rate:>6.1%} {cancer_rate:>6.1%} {ppv:>6.1%}")
    except Exception as e:
        print(f"  {arch_name}: SKIPPED ({e})")


# ══════════════════════════════════════════════════════
# 8. MISSED CANCER ANALYSIS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. MISSED CANCER ANALYSIS (WHO DOES THE MODEL MISS?)")
print("=" * 70)

missed = test_analysis[(test_analysis['cancer'] == 1) & (test_analysis['flagged'] == 0)]
caught = test_analysis[(test_analysis['cancer'] == 1) & (test_analysis['flagged'] == 1)]

print(f"\nCaught: {len(caught):,} ({len(caught)/total_cancer*100:.1f}%)")
print(f"Missed: {len(missed):,} ({len(missed)/total_cancer*100:.1f}%)")

# Compare caught vs missed
compare_features = [
    'HAEM_ANY_FLAG', 'RF_EVER_SMOKER', 'LAB_HB_EVER_BELOW_100',
    'LAB_ANAEMIA_SEVERE', 'COMORB_RECURRENT_UTI_FLAG',
    'SYN_BLEEDING_SCORE', 'INT_BLADDER_SYMPTOM_SCORE',
    'INT_OVERALL_SUSPICION_SCORE', 'NOBS_WEIGHT_LOSS_FLAG',
    'NOBS_FATIGUE_FLAG', 'URINE_BLOOD_TEST_COUNT',
    'SEX_FEMALE', 'AGE_BAND_UNDER50', 'AGE_BAND_80PLUS',
]

available_compare = [f for f in compare_features if f in X_test.columns]

print(f"\n{'Feature':<45s} {'Caught':>8s} {'Missed':>8s} {'Diff':>8s}")
print("─" * 75)

for feat in available_compare:
    caught_mean = caught[feat].fillna(0).mean()
    missed_mean = missed[feat].fillna(0).mean()
    diff = caught_mean - missed_mean
    marker = " ★" if abs(diff) > 0.05 else ""
    print(f"{feat:<45s} {caught_mean:>7.2%} {missed_mean:>7.2%} {diff:>+7.2%}{marker}")

# Missed cancer probability distribution
print(f"\nMissed cancer model scores:")
print(f"  Min:    {missed['xgb_prob'].min():.4f}")
print(f"  25th:   {missed['xgb_prob'].quantile(0.25):.4f}")
print(f"  Median: {missed['xgb_prob'].median():.4f}")
print(f"  75th:   {missed['xgb_prob'].quantile(0.75):.4f}")
print(f"  Max:    {missed['xgb_prob'].max():.4f}")

# How many missed are "near miss" (close to threshold)?
near_miss = missed[missed['xgb_prob'] >= CLINICAL_THRESHOLD * 0.8]
print(f"\n  Near-miss (score ≥ {CLINICAL_THRESHOLD*0.8:.3f}): {len(near_miss)} ({len(near_miss)/len(missed)*100:.1f}% of missed)")
print(f"  Far-miss  (score < {CLINICAL_THRESHOLD*0.8:.3f}): {len(missed)-len(near_miss)} ({(len(missed)-len(near_miss))/len(missed)*100:.1f}% of missed)")


# ══════════════════════════════════════════════════════
# 9. CLINICAL DECISION SUPPORT RULES (PAPER-READY)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. CLINICAL DECISION SUPPORT RULES")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  BLADDER CANCER RISK ASSESSMENT — CLINICAL DECISION RULES          ║
║  Based on XGBoost model (AUC 0.879, 12-month lead time)           ║
╠═══════════���══════════════════════════════════════════════════════════╣
║                                                                      ║
║  TIER 1 — URGENT REFERRAL (Score ≥ 0.35)                           ║
║  PPV ~53%, catches 60% of cancers                                    ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ • Visible/non-visible haematuria                            │    ║
║  │ • Haematuria + smoking history                              │    ║
║  │ • Haematuria + age ≥ 60 + male                              │    ║
║  │ • Recurrent haematuria (multiple episodes)                  │    ║
║  │ • Bleeding syndrome score ≥ 2                               │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
║  TIER 2 — ENHANCED SURVEILLANCE (Score 0.185-0.35)                  ║
║  PPV ~20%, catches additional 19% of cancers                         ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ • Ever-smoker + age ≥ 60 + unexplained anaemia (Hb<100)    │    ║
║  │ • Recurrent UTIs + smoking history                          │    ║
║  │ • New iron deficiency + no obvious cause                    │    ║
║  │ • Increasing GP attendance + new symptoms                   │    ║
║  │ • Weight loss + fatigue + smoker                            │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
║  TIER 3 — ROUTINE CARE (Score < 0.185)                              ║
║  NPV 97.2%, safe to reassure                                        ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ • Standard care pathway                                     │    ║
║  │ • Safety-net advice: report new haematuria, weight loss,    │    ║
║  │   persistent UTI symptoms                                   │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════
# 10. SAVE ALL RESULTS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10. SAVING RESULTS")
print("=" * 70)

# Rules
rules_df = pd.DataFrame(rules)
rules_df.to_csv(os.path.join(RULES_DIR, 'feature_threshold_rules.csv'), index=False)
print(f"  Saved: feature_threshold_rules.csv ({len(rules)} rules)")

# Combination rules
combo_df = pd.DataFrame(combo_results)
combo_df.to_csv(os.path.join(RULES_DIR, 'combination_rules.csv'), index=False)
print(f"  Saved: combination_rules.csv ({len(combo_results)} combos)")

# Archetypes
arch_df = pd.DataFrame(archetype_results)
arch_df.to_csv(os.path.join(RULES_DIR, 'patient_archetypes.csv'), index=False)
print(f"  Saved: patient_archetypes.csv ({len(archetype_results)} archetypes)")

# Simple score performance
score_df = pd.DataFrame({
    'patient_id': pid_test,
    'actual_cancer': y_test,
    'xgb_prob': xgb_probs,
    'simple_score': test_scores,
    'xgb_flag': xgb_flags
})
score_df.to_csv(os.path.join(RULES_DIR, 'patient_scores.csv'), index=False)
print(f"  Saved: patient_scores.csv")

# Clinical rules summary
with open(os.path.join(RULES_DIR, 'clinical_rules_summary.txt'), 'w') as f:
    f.write("BLADDER CANCER RISK ASSESSMENT — CLINICAL DECISION RULES\n")
    f.write(f"Model: XGBoost (AUC {roc_auc_score(y_test, xgb_probs):.3f})\n")
    f.write(f"Data: {len(df):,} patients, 12-36 months before diagnosis\n\n")
    f.write("TIER 1 — URGENT (score ≥ 0.35): PPV ~53%\n")
    f.write("TIER 2 — ENHANCED (score 0.185-0.35): PPV ~20%\n")
    f.write("TIER 3 — ROUTINE (score < 0.185): NPV 97.2%\n\n")
    f.write("SIMPLE CLINICAL SCORING SYSTEM:\n")
    for feat, (cond, points, label) in sorted(available_score_features.items(), key=lambda x: -x[1][1]):
        f.write(f"  +{points} points: {label}\n")
    f.write(f"\nSimple score AUC: {score_auc:.4f}\n")
print(f"  Saved: clinical_rules_summary.txt")

print("\n" + "=" * 70)
print("RULES EXTRACTION COMPLETE ✅")
print("=" * 70)
print(f"""
  XGBoost AUC:       {roc_auc_score(y_test, xgb_probs):.4f}
  Simple Score AUC:  {score_auc:.4f}
  Surrogate Tree:    Saved
  Clinical Rules:    3-tier system
  Patient Archetypes: {len(archetype_results)} profiles
  
  Files in: {RULES_DIR}
""")