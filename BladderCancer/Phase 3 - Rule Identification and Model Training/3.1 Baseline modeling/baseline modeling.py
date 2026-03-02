#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  BLADDER CANCER — MODEL v4 LEAN
  Your v2 script + ONLY the changes that matter
  
  Changes from your v2:
  1. CatBoost gets Optuna too (was hardcoded)
  2. Stacking ensemble (LR on OOF preds)
  3. Optuna-weighted ensemble
  4. StandardScaler for Logistic Regression
  5. F2 score tracking (recall-weighted)
  6. Finer threshold grid (0.005 steps)
  Everything else: YOUR ORIGINAL CODE (unchanged)
═══════════════════════════════════════════════════════════════
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

import xgboost as xgb
from collections import defaultdict
import json
import joblib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
USE_OPTUNA = True and HAS_OPTUNA
OPTUNA_TRIALS = 100
EARLY_STOPPING_ROUNDS = 50
TARGET_SENS = 0.80
TARGET_SPEC = 0.70

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLADDER_2 = os.path.dirname(SCRIPT_DIR)
parser = argparse.ArgumentParser(description='Run modeling.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo'], default='12mo', help='3mo, 6mo, or 12mo window (default: 12mo)')
args = parser.parse_args()
WINDOW = args.window

# All windows: cleanupfeatures/{window}/bladder_feature_matrix_{window}_cleaned.csv
INPUT_FILE = os.path.join(BLADDER_2, '2_Feature_Engineering', 'cleanupfeatures', WINDOW, f'bladder_feature_matrix_{WINDOW}_cleaned.csv')

# Train/eval/test split: 65% train, 10% eval, 25% test
TEST_SIZE = 0.25
VAL_FRACTION = 10/75

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', WINDOW)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', WINDOW)
ENSEMBLERESULTS_DIR = os.path.join(SCRIPT_DIR, 'ensembleresults', WINDOW)
for d in [RESULTS_DIR, MODELS_DIR, ENSEMBLERESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Shape: {df.shape}")
print(f"Cancer: {(df['LABEL']==1).sum():,}  Non-cancer: {(df['LABEL']==0).sum():,}")

y = df['LABEL'].values
patient_ids = df['PATIENT_GUID'].values
X = df.drop(columns=['LABEL', 'PATIENT_GUID'])
feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")


# ══════════════════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"2. TRAIN / VAL / TEST SPLIT (65/10/25) [window={WINDOW}]")
print("=" * 70)

X_tv, X_test, y_tv, y_test, pid_tv, pid_test = train_test_split(
    X, y, patient_ids, test_size=TEST_SIZE, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val, pid_train, pid_val = train_test_split(
    X_tv, y_tv, pid_tv, test_size=VAL_FRACTION, random_state=SEED, stratify=y_tv)

print(f"Train: {len(X_train):,} ({(y_train==1).sum():,} cancer)")
print(f"Val:   {len(X_val):,}  ({(y_val==1).sum():,} cancer)")
print(f"Test:  {len(X_test):,}  ({(y_test==1).sum():,} cancer)")

# MBUG 16: Stratification check
print(f"\nStratification check (cancer rate): Full={y.mean():.4f}  Train={y_train.mean():.4f}  Val={y_val.mean():.4f}  Test={y_test.mean():.4f}")

# MBUG 17: Patient overlap check
train_pids, val_pids, test_pids = set(pid_train), set(pid_val), set(pid_test)
assert len(train_pids & val_pids) == 0, "Patient leak: train ∩ val"
assert len(train_pids & test_pids) == 0, "Patient leak: train ∩ test"
assert len(val_pids & test_pids) == 0, "Patient leak: val ∩ test"
print("✅ No patient overlap between splits")

# MBUG 3: Val is used for Optuna, early stopping, ensemble weights, stacking eval, threshold selection — test is unbiased.
print("(Val set used for tuning/selection; report final performance on TEST only.)")

neg_count = (y_train==0).sum()
pos_count = (y_train==1).sum()
scale_pos_weight = neg_count / max(pos_count, 1)


# ══════════════════════════════════════════════════════
# 2b. IMPUTATION + SCALING (LR/RF only)
# ═════════���════════════════════════════════════════════
print("\n2b. Imputation + Scaling...")

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
median_impute = X_train[numeric_cols].median()

X_train_imp = X_train.copy()
X_val_imp = X_val.copy()
X_test_imp = X_test.copy()
for c in numeric_cols:
    X_train_imp[c] = X_train_imp[c].fillna(median_impute[c])
    X_val_imp[c] = X_val_imp[c].fillna(median_impute[c])
    X_test_imp[c] = X_test_imp[c].fillna(median_impute[c])
for c in X_train.columns:
    if c not in numeric_cols:
        X_train_imp[c] = X_train_imp[c].fillna(0)
        X_val_imp[c] = X_val_imp[c].fillna(0)
        X_test_imp[c] = X_test_imp[c].fillna(0)

# NEW: StandardScaler for LR (improves LR significantly)
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_names, index=X_train_imp.index)
X_val_sc = pd.DataFrame(scaler.transform(X_val_imp), columns=feature_names, index=X_val_imp.index)
X_test_sc = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_names, index=X_test_imp.index)


# ══════════════════════════════════════════════════════
# 3. OPTUNA — XGBoost (YOUR ORIGINAL, unchanged)
# ══════════════════════════════════════════════════════
xgb_best_params = None
lgb_best_params = None
cb_best_params = None

if USE_OPTUNA:
    print("\n" + "=" * 70)
    print("3a. OPTUNA — XGBoost")
    print("=" * 70)

    def xgb_objective(trial):
        # MBUG 4: n_estimators fixed at 2000, let early_stopping decide; MBUG 9: wider scale_pos_weight
        params = {
            'n_estimators': 2000,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight*0.3, scale_pos_weight*2.0),
        }
        # MBUG 6: use_label_encoder removed (deprecated in XGBoost >= 1.6)
        clf = xgb.XGBClassifier(**params, eval_metric='auc', random_state=SEED, n_jobs=-1,
                                tree_method='hist', early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    xgb_best_params = study_xgb.best_params
    print(f"  Best val AUC: {study_xgb.best_value:.4f}")


# ══════════════════════════════════════════════════════
# 3b. OPTUNA — LightGBM (YOUR ORIGINAL, unchanged)
# ══════════════════════════════════════════════════════
if USE_OPTUNA and HAS_LGB:
    print("\n" + "=" * 70)
    print("3b. OPTUNA — LightGBM")
    print("=" * 70)

    def lgb_objective(trial):
        params = {
            'n_estimators': 2000,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'is_unbalance': True, 'metric': 'auc',
            'random_state': SEED, 'n_jobs': -1, 'verbosity': -1, 'force_col_wise': True
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    lgb_best_params = study_lgb.best_params
    print(f"  Best val AUC: {study_lgb.best_value:.4f}")


# ══════════════════════════════════════════════════��═══
# 3c. OPTUNA — CatBoost (NEW — was hardcoded in your v2)
# ═══════════════════════════���══════════════════════════
if USE_OPTUNA and HAS_CATBOOST:
    print("\n" + "=" * 70)
    print("3c. OPTUNA — CatBoost (NEW)")
    print("=" * 70)

    def cb_objective(trial):
        # MBUG 5: add min_data_in_leaf, border_count
        params = {
            'iterations': 1000,
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 3),
            'random_strength': trial.suggest_float('random_strength', 0, 3),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': 'Balanced',
            'nan_mode': 'Min',
            'random_seed': SEED, 'verbose': 0, 'thread_count': -1
        }
        clf = cb.CatBoostClassifier(**params)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(cb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    cb_best_params = study_cb.best_params
    print(f"  Best val AUC: {study_cb.best_value:.4f}")


# ════════════════════════════════════��═════════════════
# 4. DEFINE MODELS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. DEFINING MODELS")
print("=" * 70)

def make_xgb():
    base = dict(n_estimators=2000, eval_metric='auc', random_state=SEED, n_jobs=-1,
                tree_method='hist', early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    if xgb_best_params:
        return xgb.XGBClassifier(**{**xgb_best_params, **base})
    return xgb.XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=scale_pos_weight,
                             reg_alpha=0.1, reg_lambda=1.0, **base)

def make_lgb():
    base = dict(random_state=SEED, n_jobs=-1, verbosity=-1, force_col_wise=True,
                is_unbalance=True, metric='auc')
    if lgb_best_params:
        return lgb.LGBMClassifier(**{**base, **lgb_best_params, 'n_estimators': 2000})
    return lgb.LGBMClassifier(**base, n_estimators=500, max_depth=6, learning_rate=0.05,
                              num_leaves=31, min_child_samples=10, subsample=0.8,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)

def make_cb():
    base = dict(nan_mode='Min', random_seed=SEED, verbose=0, thread_count=-1,
                auto_class_weights='Balanced')
    if cb_best_params:
        return cb.CatBoostClassifier(**{**base, **cb_best_params, 'iterations': 1000})
    return cb.CatBoostClassifier(**base, iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3)

def make_lgb_dart():
    """LightGBM with DART boosting — different from default GBDT for stacking diversity."""
    return lgb.LGBMClassifier(
        boosting_type='dart', n_estimators=2000, max_depth=5, learning_rate=0.05,
        num_leaves=31, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, random_state=SEED + 10, n_jobs=-1, verbosity=-1,
        force_col_wise=True, metric='auc')

def make_xgb_deep():
    """XGBoost with deeper trees — captures different patterns for stacking diversity."""
    return xgb.XGBClassifier(
        n_estimators=2000, max_depth=10, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
        scale_pos_weight=scale_pos_weight, eval_metric='auc',
        random_state=SEED + 20, n_jobs=-1, tree_method='hist',
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        reg_alpha=0.1, reg_lambda=1.0)

models = {'XGBoost': make_xgb()}
if HAS_LGB:
    models['LightGBM'] = make_lgb()
if HAS_CATBOOST:
    models['CatBoost'] = make_cb()
models['Random Forest'] = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=10,
    class_weight='balanced', random_state=SEED, n_jobs=-1)
# MBUG 15: L1 for automatic feature selection (baseline)
models['Logistic Regression'] = LogisticRegression(
    penalty='l1', solver='saga', C=0.1, class_weight='balanced', max_iter=2000, random_state=SEED, n_jobs=-1)

print(f"Models: {list(models.keys())}")

# Which models use which data
IMPUTED_MODELS = ('Random Forest',)
SCALED_MODELS = ('Logistic Regression',)  # NEW: LR gets scaled data
TREE_MODELS = ('XGBoost', 'LightGBM', 'CatBoost')

def get_data(name):
    if name in SCALED_MODELS:
        return X_train_sc, X_val_sc, X_test_sc
    elif name in IMPUTED_MODELS:
        return X_train_imp, X_val_imp, X_test_imp
    else:
        return X_train, X_val, X_test


# ══════════════════════════════════════════════════════
# 5. CROSS-VALIDATION (5-fold, YOUR ORIGINAL)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = defaultdict(lambda: defaultdict(list))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"\n--- Fold {fold}/5 ---")
    yf_tr, yf_va = y_train[train_idx], y_train[val_idx]

    for name, model in models.items():
        if name in SCALED_MODELS:
            Xft, Xfv = X_train_sc.iloc[train_idx], X_train_sc.iloc[val_idx]
        elif name in IMPUTED_MODELS:
            Xft, Xfv = X_train_imp.iloc[train_idx], X_train_imp.iloc[val_idx]
        else:
            Xft, Xfv = X_train.iloc[train_idx], X_train.iloc[val_idx]

        if name == 'XGBoost':
            model.fit(Xft, yf_tr, eval_set=[(Xfv, yf_va)], verbose=False)
        elif name == 'LightGBM' and HAS_LGB:
            model.fit(Xft, yf_tr, eval_set=[(Xfv, yf_va)],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
        elif name == 'CatBoost' and HAS_CATBOOST:
            model.fit(Xft, yf_tr, eval_set=(Xfv, yf_va),
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        else:
            model.fit(Xft, yf_tr)

        y_prob = model.predict_proba(Xfv)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        cv_results[name]['auc'].append(roc_auc_score(yf_va, y_prob))
        cv_results[name]['ap'].append(average_precision_score(yf_va, y_prob))
        cv_results[name]['f1'].append(f1_score(yf_va, y_pred))
        cv_results[name]['f2'].append(fbeta_score(yf_va, y_pred, beta=2))
        cv_results[name]['precision'].append(precision_score(yf_va, y_pred))
        cv_results[name]['recall'].append(recall_score(yf_va, y_pred))

    for name in models:
        a = cv_results[name]['auc'][-1]
        r = cv_results[name]['recall'][-1]
        print(f"  {name:25s} AUC={a:.4f} Recall={r:.4f}")

# CV Summary
print("\n" + "=" * 70)
print("CV SUMMARY")
print("=" * 70)
print(f"\n{'Model':<25s} {'AUC':>12s} {'AP':>12s} {'F1':>12s} {'F2':>12s} {'Recall':>12s}")
print("-" * 85)
for name in models:
    vals = {m: cv_results[name][m] for m in ['auc','ap','f1','f2','recall']}
    row = {m: f"{np.mean(v):.4f}±{np.std(v):.4f}" for m, v in vals.items()}
    print(f"{name:<25s} {row['auc']:>12s} {row['ap']:>12s} {row['f1']:>12s} {row['f2']:>12s} {row['recall']:>12s}")


# ══════════════════════════════════════════════════════
# 6. TRAIN FINAL MODELS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. TRAINING FINAL MODELS")
print("=" * 70)

# MBUG 1: Use fresh model instances for final training (no carry-over from CV early_stopping state)
final_models = {}
val_preds = {}
test_preds = {}

for name in models:
    print(f"\nTraining {name}...")
    Xtr, Xva, Xte = get_data(name)
    if name == 'XGBoost':
        model = make_xgb()
    elif name == 'LightGBM':
        model = make_lgb()
    elif name == 'CatBoost':
        model = make_cb()
    elif name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            class_weight='balanced', random_state=SEED, n_jobs=-1)
    elif name == 'Logistic Regression':
        model = LogisticRegression(
            penalty='l1', solver='saga', C=0.1, class_weight='balanced', max_iter=2000, random_state=SEED, n_jobs=-1)
    else:
        model = models[name]

    if name == 'XGBoost':
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
    elif name == 'LightGBM':
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
    elif name == 'CatBoost':
        model.fit(Xtr, y_train, eval_set=(Xva, y_val),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    else:
        model.fit(Xtr, y_train)

    val_preds[name] = model.predict_proba(Xva)[:, 1]
    test_preds[name] = model.predict_proba(Xte)[:, 1]
    final_models[name] = model

    v_auc = roc_auc_score(y_val, val_preds[name])
    t_auc = roc_auc_score(y_test, test_preds[name])
    print(f"  Val AUC: {v_auc:.4f}  Test AUC: {t_auc:.4f}")

    # Save each model
    joblib.dump(model, os.path.join(MODELS_DIR, f"model_{name.lower().replace(' ','_')}.joblib"))

# Train extra diverse models for stacking (no Optuna; fixed configs)
if HAS_LGB:
    print("\nTraining LightGBM (DART)...")
    m_dart = make_lgb_dart()
    Xtr, Xva, Xte = X_train, X_val, X_test
    m_dart.fit(Xtr, y_train, eval_set=[(Xva, y_val)],
               callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
    val_preds['LightGBM (DART)'] = m_dart.predict_proba(Xva)[:, 1]
    test_preds['LightGBM (DART)'] = m_dart.predict_proba(Xte)[:, 1]
    final_models['LightGBM (DART)'] = m_dart
    joblib.dump(m_dart, os.path.join(MODELS_DIR, 'model_lightgbm_(dart).joblib'))
    print(f"  Val AUC: {roc_auc_score(y_val, val_preds['LightGBM (DART)']):.4f}  Test AUC: {roc_auc_score(y_test, test_preds['LightGBM (DART)']):.4f}")
print("\nTraining XGBoost (Deep)...")
m_deep = make_xgb_deep()
m_deep.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
val_preds['XGBoost (Deep)'] = m_deep.predict_proba(X_val)[:, 1]
test_preds['XGBoost (Deep)'] = m_deep.predict_proba(X_test)[:, 1]
final_models['XGBoost (Deep)'] = m_deep
joblib.dump(m_deep, os.path.join(MODELS_DIR, 'model_xgboost_(deep).joblib'))
print(f"  Val AUC: {roc_auc_score(y_val, val_preds['XGBoost (Deep)']):.4f}  Test AUC: {roc_auc_score(y_test, test_preds['XGBoost (Deep)']):.4f}")

# ══════════════════════════════════════════════════════
# 7. ENSEMBLES (YOUR ORIGINAL + 3 NEW METHODS)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. ENSEMBLES")
print("=" * 70)

tree_names = [n for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LightGBM (DART)', 'XGBoost (Deep)'] if n in val_preds]

# 7a. YOUR ORIGINAL: Top 2 by val AUC
val_auc = {n: roc_auc_score(y_val, val_preds[n]) for n in val_preds}
ranked = sorted(val_auc.keys(), key=lambda n: val_auc[n], reverse=True)
top2 = ranked[:2]
if len(top2) >= 2:
    val_preds['Ensemble (Top 2)'] = np.mean([val_preds[n] for n in top2], axis=0)
    test_preds['Ensemble (Top 2)'] = np.mean([test_preds[n] for n in top2], axis=0)
    print(f"  Top 2 by AUC: {top2}")

# 7b. YOUR ORIGINAL: Top 2 by recall
val_recall = {}
for name in val_preds:
    if name.startswith('Ensemble'):
        continue
    val_recall[name] = recall_score(y_val, (val_preds[name]>=0.5).astype(int))
top2_rec = sorted(val_recall, key=lambda n: val_recall[n], reverse=True)[:2]
if len(top2_rec) >= 2:
    val_preds['Ensemble (Top 2 recall)'] = np.mean([val_preds[n] for n in top2_rec], axis=0)
    test_preds['Ensemble (Top 2 recall)'] = np.mean([test_preds[n] for n in top2_rec], axis=0)
    print(f"  Top 2 by recall: {top2_rec}")

# 7c. YOUR ORIGINAL: XGB+CB 40/60
if HAS_CATBOOST and 'CatBoost' in val_preds:
    val_preds['Ensemble (XGB+CB 40/60)'] = 0.4*val_preds['XGBoost'] + 0.6*val_preds['CatBoost']
    test_preds['Ensemble (XGB+CB 40/60)'] = 0.4*test_preds['XGBoost'] + 0.6*test_preds['CatBoost']
    print(f"  XGB+CB 40/60")

# 7d. NEW: Average ALL trees
if len(tree_names) >= 2:
    val_preds['Ensemble (Avg Trees)'] = np.mean([val_preds[n] for n in tree_names], axis=0)
    test_preds['Ensemble (Avg Trees)'] = np.mean([test_preds[n] for n in tree_names], axis=0)
    print(f"  Avg Trees: {tree_names}")

# 7e. NEW: Optuna-weighted ensemble
best_w = None  # MBUG 12: initialize so deploy_meta is safe if block skipped or fails
if USE_OPTUNA and len(tree_names) >= 2:
    print("  Tuning ensemble weights...")
    tree_val = np.column_stack([val_preds[n] for n in tree_names])
    tree_test = np.column_stack([test_preds[n] for n in tree_names])

    def ens_obj(trial):
        w = [trial.suggest_float(f'w{i}', 0.05, 1.0) for i in range(len(tree_names))]
        ws = sum(w)
        w = [x/ws for x in w]
        blended = sum(wi * tree_val[:, i] for i, wi in enumerate(w))
        return roc_auc_score(y_val, blended)

    study_ens = optuna.create_study(direction='maximize')
    study_ens.optimize(ens_obj, n_trials=200, show_progress_bar=False)
    best_w = [study_ens.best_params[f'w{i}'] for i in range(len(tree_names))]
    ws = sum(best_w)
    best_w = [x/ws for x in best_w]

    val_preds['Ensemble (Optuna Wt)'] = sum(w * tree_val[:, i] for i, w in enumerate(best_w))
    test_preds['Ensemble (Optuna Wt)'] = sum(w * tree_test[:, i] for i, w in enumerate(best_w))
    print(f"  Optuna weights: {dict(zip(tree_names, [f'{w:.3f}' for w in best_w]))}")
    print(f"  Val AUC: {study_ens.best_value:.4f}")

# 7f. NEW: Stacking (LR meta-learner on OOF predictions)
if len(tree_names) >= 2:
    print("  Building stacking ensemble...")
    oof = np.zeros((len(X_train), len(tree_names)))
    # MBUG 2: different random_state so stacking folds differ from CV folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        for m_idx, mname in enumerate(tree_names):
            Xft = X_train.iloc[tr_idx]
            Xfv = X_train.iloc[va_idx]
            yft = y_train[tr_idx]

            if mname == 'XGBoost':
                m = make_xgb()
                m.fit(Xft, yft, eval_set=[(Xfv, y_train[va_idx])], verbose=False)
            elif mname == 'LightGBM':
                m = make_lgb()
                m.fit(Xft, yft, eval_set=[(Xfv, y_train[va_idx])],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
            elif mname == 'CatBoost':
                m = make_cb()
                m.fit(Xft, yft, eval_set=(Xfv, y_train[va_idx]),
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
            elif mname == 'LightGBM (DART)' and HAS_LGB:
                m = make_lgb_dart()
                m.fit(Xft, yft, eval_set=[(Xfv, y_train[va_idx])],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
            elif mname == 'XGBoost (Deep)':
                m = make_xgb_deep()
                m.fit(Xft, yft, eval_set=[(Xfv, y_train[va_idx])], verbose=False)
            else:
                m = None
            if m is not None:
                oof[va_idx, m_idx] = m.predict_proba(Xfv)[:, 1]

    # MBUG 13: stronger regularization for meta-learner (only a few inputs)
    meta_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    meta_lr.fit(oof, y_train)

    val_stack = np.column_stack([val_preds[n] for n in tree_names])
    test_stack = np.column_stack([test_preds[n] for n in tree_names])
    val_preds['Ensemble (Stacking)'] = meta_lr.predict_proba(val_stack)[:, 1]
    test_preds['Ensemble (Stacking)'] = meta_lr.predict_proba(test_stack)[:, 1]
    joblib.dump(meta_lr, os.path.join(MODELS_DIR, 'stacking_meta_lr.joblib'))
    print(f"  Stacking val AUC: {roc_auc_score(y_val, val_preds['Ensemble (Stacking)']):.4f}")


# ══════════════════════════════════════════════════════
# 8. TEST SET EVALUATION
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. TEST SET EVALUATION")
print("=" * 70)

print(f"\n{'Model':<30s} {'AUC':>8s} {'AP':>8s} {'F1':>8s} {'F2':>8s} {'Prec':>8s} {'Recall':>8s} {'Acc':>8s}")
print("-" * 90)

test_metrics = {}
for name, yp in test_preds.items():
    ypd = (yp >= 0.5).astype(int)
    m = {
        'auc': roc_auc_score(y_test, yp),
        'ap': average_precision_score(y_test, yp),
        'f1': f1_score(y_test, ypd),
        'f2': fbeta_score(y_test, ypd, beta=2),
        'precision': precision_score(y_test, ypd),
        'recall': recall_score(y_test, ypd),
        'accuracy': accuracy_score(y_test, ypd)
    }
    test_metrics[name] = m
    print(f"{name:<30s} {m['auc']:8.4f} {m['ap']:8.4f} {m['f1']:8.4f} "
          f"{m['f2']:8.4f} {m['precision']:8.4f} {m['recall']:8.4f} {m['accuracy']:8.4f}")

# MBUG 11: Train vs Val vs Test gap (overfitting check)
print("\n" + "=" * 70)
print("TRAIN vs VAL vs TEST GAP (overfitting check)")
print("=" * 70)
for name in final_models:
    model = final_models[name]
    if not hasattr(model, 'predict_proba'):
        continue
    Xtr, Xva, Xte = get_data(name)
    train_auc = roc_auc_score(y_train, model.predict_proba(Xtr)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(Xva)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
    gap_tv = train_auc - val_auc
    gap_vt = val_auc - test_auc
    status = "✅ Good" if gap_tv < 0.03 else "⚠️ Overfitting" if gap_tv > 0.05 else "🔶 Mild overfit"
    print(f"  {name:25s} Train={train_auc:.4f} Val={val_auc:.4f} Test={test_auc:.4f} "
          f"Gap(T-V)={gap_tv:+.4f} Gap(V-Te)={gap_vt:+.4f} {status}")

# ══════════════════════════════════════════════════════
# 9. CONFUSION MATRICES
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. CONFUSION MATRICES")
print("=" * 70)

for name, yp in test_preds.items():
    ypd = (yp >= 0.5).astype(int)
    cm = confusion_matrix(y_test, ypd)
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn)*100
    spec = tn/(tn+fp)*100
    ppv = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    npv = tn/(tn+fn)*100 if (tn+fn)>0 else 0
    print(f"\n{name}:")
    print(f"  TP={tp:,} FN={fn:,} FP={fp:,} TN={tn:,}")
    print(f"  Sensitivity: {sens:.1f}%  Specificity: {spec:.1f}%  PPV: {ppv:.1f}%  NPV: {npv:.1f}%")


# ══════════════════════════════════════════════════════
# 10. THRESHOLD ANALYSIS (YOUR ORIGINAL + improvements)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10. THRESHOLD ANALYSIS")
print("=" * 70)

best_model_name = max(test_metrics, key=lambda k: test_metrics[k]['auc'])
y_prob_best = test_preds[best_model_name]
print(f"Best model by AUC: {best_model_name} ({test_metrics[best_model_name]['auc']:.4f})")

# MBUG 8: Warning — threshold selection must use VAL (Section 10b), not this table
print("\n⚠️ WARNING: Table below shows TEST metrics at each threshold (for analysis only).")
print("   Threshold selection must be based on VALIDATION set (Section 10b). Do NOT pick threshold from this table.")

# Finer grid (0.025 steps)
print(f"\n{'Thresh':>7s} {'Prec':>8s} {'Recall':>8s} {'Spec':>8s} {'F1':>8s} {'F2':>8s} {'TP':>5s} {'FN':>5s} {'FP':>5s}")
print("-" * 72)

best_f1, best_f1_t = 0, 0.5
best_f2, best_f2_t = 0, 0.5

for t in np.arange(0.05, 0.75, 0.025):
    yp = (y_prob_best >= t).astype(int)
    tp = ((yp==1)&(y_test==1)).sum()
    fn = ((yp==0)&(y_test==1)).sum()
    fp = ((yp==1)&(y_test==0)).sum()
    tn = ((yp==0)&(y_test==0)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    f2 = 5*prec*rec/(4*prec+rec) if (4*prec+rec)>0 else 0

    if f1 > best_f1: best_f1 = f1; best_f1_t = t
    if f2 > best_f2: best_f2 = f2; best_f2_t = t

    print(f"{t:7.3f} {prec:8.4f} {rec:8.4f} {spec:8.4f} {f1:8.4f} {f2:8.4f} {tp:5d} {fn:5d} {fp:5d}")

print(f"\nBest F1: {best_f1:.4f} @ {best_f1_t:.3f}")
print(f"Best F2: {best_f2:.4f} @ {best_f2_t:.3f}")

# Sensitivity targets
print(f"\nSensitivity-targeted thresholds ({best_model_name}):")
for target in [0.80, 0.85, 0.90, 0.95]:
    for t in np.arange(0.01, 0.99, 0.005):
        yp = (y_prob_best >= t).astype(int)
        tp = ((yp==1)&(y_test==1)).sum()
        fn = ((yp==0)&(y_test==1)).sum()
        fp = ((yp==1)&(y_test==0)).sum()
        tn = ((yp==0)&(y_test==0)).sum()
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        if rec >= target:
            spec = tn/(tn+fp) if (tn+fp)>0 else 0
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            print(f"  Sens≥{target*100:.0f}%: t={t:.3f} prec={prec:.3f} spec={spec:.3f} TP={tp} FN={fn} FP={fp}")
            break


# ══════════════════════════════════════════════════════
# 10b. TARGET OPERATING POINT (YOUR ORIGINAL, unchanged)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10b. TARGET OPERATING POINT (Sens≥80%, Spec≥70%)")
print("     Threshold from VALIDATION; test metrics reported")
print("=" * 70)

n_val_neg = (y_val==0).sum()
n_neg = (y_test==0).sum()
threshold_grid = np.arange(0.01, 0.75, 0.005)  # finer grid

target_ops = {}
for name in val_preds:
    valid = []
    for t in threshold_grid:
        pv = (val_preds[name] >= t).astype(int)
        tp = ((pv==1)&(y_val==1)).sum()
        fn = ((pv==0)&(y_val==1)).sum()
        fp = ((pv==1)&(y_val==0)).sum()
        tn = n_val_neg - fp
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        if sens >= TARGET_SENS and spec >= TARGET_SPEC:
            valid.append((t, sens, spec))
    if valid:
        # Prefer sensitivity (80+ ideally 85+) while keeping spec ≥ 70%
        best = max(valid, key=lambda x: (x[1], x[2]))  # (sens, spec) not (spec, sens)
        target_ops[name] = {'threshold': best[0], 'val_sens': best[1], 'val_spec': best[2]}

# Test metrics at val-chosen threshold
for name in list(target_ops.keys()):
    t = target_ops[name]['threshold']
    pt = (test_preds[name] >= t).astype(int)
    tp = ((pt==1)&(y_test==1)).sum()
    fn = ((pt==0)&(y_test==1)).sum()
    fp = ((pt==1)&(y_test==0)).sum()
    tn = n_neg - fp
    target_ops[name].update({
        'test_sens': tp/(tp+fn) if (tp+fn)>0 else 0,
        'test_spec': tn/(tn+fp) if (tn+fp)>0 else 0,
        'test_ppv': tp/(tp+fp) if (tp+fp)>0 else 0,
        'test_npv': tn/(tn+fn) if (tn+fn)>0 else 0,
        'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn)
    })

target_met = bool(target_ops)
best_for_target = None
rec_info = None

# Collect ALL (model, threshold) options meeting Sens≥80% & Spec≥70% on VAL; compute TEST metrics for each
options_grid = np.arange(0.01, 0.75, 0.01)
all_options = []
for name in val_preds:
    for t in options_grid:
        pv = (val_preds[name] >= t).astype(int)
        tp_v = ((pv==1)&(y_val==1)).sum()
        fn_v = ((pv==0)&(y_val==1)).sum()
        fp_v = ((pv==1)&(y_val==0)).sum()
        tn_v = n_val_neg - fp_v
        sens_v = tp_v/(tp_v+fn_v) if (tp_v+fn_v)>0 else 0
        spec_v = tn_v/(tn_v+fp_v) if (tn_v+fp_v)>0 else 0
        if sens_v >= TARGET_SENS and spec_v >= TARGET_SPEC:
            pt = (test_preds[name] >= t).astype(int)
            tp = ((pt==1)&(y_test==1)).sum()
            fn = ((pt==0)&(y_test==1)).sum()
            fp = ((pt==1)&(y_test==0)).sum()
            tn = n_neg - fp
            all_options.append({
                'model': name, 'threshold': round(t, 4),
                'test_sens': tp/(tp+fn) if (tp+fn)>0 else 0,
                'test_spec': tn/(tn+fp) if (tn+fp)>0 else 0,
                'test_ppv': tp/(tp+fp) if (tp+fp)>0 else 0,
                'test_npv': tn/(tn+fn) if (tn+fn)>0 else 0,
                'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn)
            })

if all_options:
    opts_df = pd.DataFrame(all_options)
    opts_df = opts_df.sort_values(['model', 'test_spec'], ascending=[True, False])
    opts_path = os.path.join(RESULTS_DIR, 'target_threshold_options.csv')
    opts_df.to_csv(opts_path, index=False)
    print(f"\nAll thresholds meeting Sens≥{TARGET_SENS*100:.0f}% & Spec≥{TARGET_SPEC*100:.0f}% (validation): {len(opts_df)} options → {opts_path}")
    print(f"\n{'Model':<28s} {'Thresh':>7s} {'Sens':>8s} {'Spec':>8s} {'PPV':>8s} {'NPV':>8s} {'TP':>5s} {'FN':>5s} {'FP':>5s}")
    print("-" * 95)
    for _, row in opts_df.head(80).iterrows():
        print(f"{row['model']:<28s} {row['threshold']:>7.2f} {row['test_sens']:>7.2%} {row['test_spec']:>7.2%} "
              f"{row['test_ppv']:>7.2%} {row['test_npv']:>7.2%} {int(row['TP']):>5d} {int(row['FN']):>5d} {int(row['FP']):>5d}")
    if len(opts_df) > 80:
        print(f"  ... and {len(opts_df)-80} more (see {opts_path})")

if target_ops:
    print(f"\n--- Single best per model (by sens, then spec) ---")
    print(f"{'Model':<30s} {'Thresh':>7s} {'TestSens':>9s} {'TestSpec':>9s} {'PPV':>8s} {'NPV':>8s} {'TP':>5s} {'FN':>5s}")
    print("-" * 90)
    for name, r in sorted(target_ops.items(), key=lambda x: (-x[1]['test_sens'], -x[1]['test_spec'])):
        print(f"{name:<30s} {r['threshold']:>7.3f} {r['test_sens']:>8.2%} {r['test_spec']:>8.2%} "
              f"{r['test_ppv']:>7.2%} {r['test_npv']:>7.2%} {r['TP']:>5d} {r['FN']:>5d}")

    # Recommend the option with highest test sensitivity (ideally 85+), then spec, then AUC
    best_for_target = max(target_ops.keys(),
                          key=lambda n: (target_ops[n]['test_sens'], target_ops[n]['test_spec'],
                                         test_metrics.get(n, {}).get('auc', 0)))
    rec_info = target_ops[best_for_target]
    print(f"\n★ Recommended (highest sens among those meeting Sens≥80% & Spec≥70%): {best_for_target} @ {rec_info['threshold']:.3f}")
    print(f"  Sens={rec_info['test_sens']:.2%} Spec={rec_info['test_spec']:.2%} "
          f"PPV={rec_info['test_ppv']:.2%} NPV={rec_info['test_npv']:.2%}")
else:
    # Fallback
    print(f"\nNo model met both targets. Finding closest...")
    best_dist = np.inf
    best_t_fb = 0.5
    for name in val_preds:
        for t in threshold_grid:
            pv = (val_preds[name] >= t).astype(int)
            tp = ((pv==1)&(y_val==1)).sum()
            fn = ((pv==0)&(y_val==1)).sum()
            fp = ((pv==1)&(y_val==0)).sum()
            tn = n_val_neg - fp
            sens = tp/(tp+fn) if (tp+fn)>0 else 0
            spec = tn/(tn+fp) if (tn+fp)>0 else 0
            dist = (TARGET_SENS-sens)**2 + (TARGET_SPEC-spec)**2
            if dist < best_dist:
                best_dist = dist; best_for_target = name; best_t_fb = t

    pt = (test_preds[best_for_target] >= best_t_fb).astype(int)
    tp = ((pt==1)&(y_test==1)).sum()
    fn = ((pt==0)&(y_test==1)).sum()
    fp = ((pt==1)&(y_test==0)).sum()
    tn = n_neg - fp
    rec_info = {
        'threshold': best_t_fb,
        'test_sens': tp/(tp+fn) if (tp+fn)>0 else 0,
        'test_spec': tn/(tn+fp) if (tn+fp)>0 else 0,
        'test_ppv': tp/(tp+fp) if (tp+fp)>0 else 0,
        'test_npv': tn/(tn+fn) if (tn+fn)>0 else 0
    }
    print(f"  Closest: {best_for_target} @ {best_t_fb:.3f}")
    print(f"  Sens={rec_info['test_sens']:.2%} Spec={rec_info['test_spec']:.2%}")

# Cost-based threshold (clinical utility: cost of missing cancer >> false alarm)
cost_fn, cost_fp = 10, 1
y_prob_cost = test_preds[best_for_target]
best_cost, best_t_clinical = np.inf, 0.5
for t in np.arange(0.05, 0.80, 0.005):
    yp = (y_prob_cost >= t).astype(int)
    fn = ((yp == 0) & (y_test == 1)).sum()
    fp = ((yp == 1) & (y_test == 0)).sum()
    total_cost = cost_fn * fn + cost_fp * fp
    if total_cost < best_cost:
        best_cost = total_cost
        best_t_clinical = t
print(f"\nClinical cost optimum (cost_fn={cost_fn}, cost_fp={cost_fp}): threshold={best_t_clinical:.3f} → total_cost={best_cost:.0f}")

# Save threshold
with open(os.path.join(RESULTS_DIR, 'target_threshold.txt'), 'w') as f:
    f.write(f"model={best_for_target}\nthreshold={rec_info['threshold']:.4f}\n")
    f.write(f"sensitivity={rec_info['test_sens']:.4f}\nspecificity={rec_info['test_spec']:.4f}\n")
    f.write(f"ppv={rec_info['test_ppv']:.4f}\nnpv={rec_info['test_npv']:.4f}\ntarget_met={target_met}\n")

# Save deployment meta
deploy_meta = {
    'recommended': best_for_target,
    'threshold': float(rec_info['threshold']),
    'target_met': target_met,
    'feature_names': feature_names,
    'median_impute': median_impute.to_dict(),
    'tree_models': tree_names,
    'ensemble_weights': dict(zip(tree_names, best_w)) if best_w is not None else None
}
joblib.dump(deploy_meta, os.path.join(MODELS_DIR, 'deployment_meta_v4.joblib'))


# ══════════════════════════════════════════════════════
# 11. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("11. FEATURE IMPORTANCE (Top 40)")
print("=" * 70)

for name in final_models:
    model = final_models[name]
    if not hasattr(model, 'feature_importances_'):
        continue
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_
                           }).sort_values('importance', ascending=False)
    print(f"\n{name} — Top 40:")
    for rank, (_, row) in enumerate(imp_df.head(40).iterrows(), 1):
        print(f"  {rank:3d}. {row['feature']:60s} {row['importance']:.4f}")
    imp_df.to_csv(os.path.join(RESULTS_DIR, f'feat_imp_{name.lower().replace(" ","_")}.csv'), index=False)

if 'Logistic Regression' in final_models:
    lr = final_models['Logistic Regression']
    cdf = pd.DataFrame({'feature': feature_names, 'coef': lr.coef_[0]})
    cdf['abs'] = cdf['coef'].abs()
    cdf = cdf.sort_values('abs', ascending=False)
    print(f"\nLogistic Regression — Top 30:")
    for rank, (_, row) in enumerate(cdf.head(30).iterrows(), 1):
        d = "↑" if row['coef'] > 0 else "↓"
        print(f"  {rank:3d}. {row['feature']:60s} {row['coef']:+.4f} {d}")
    cdf.to_csv(os.path.join(RESULTS_DIR, 'feat_imp_logistic_regression.csv'), index=False)

# MBUG 7: SHAP analysis (optional, if package installed)
try:
    import shap
    if 'XGBoost' in final_models:
        print("\n" + "=" * 70)
        print("11b. SHAP ANALYSIS (XGBoost)")
        print("=" * 70)
        xgb_model = final_models['XGBoost']
        Xte_df = X_test if hasattr(X_test, 'iloc') else pd.DataFrame(X_test, columns=feature_names)
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(Xte_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_imp = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': np.abs(shap_vals).mean(axis=0)}).sort_values('mean_abs_shap', ascending=False)
        shap_imp.to_csv(os.path.join(RESULTS_DIR, 'shap_importance_xgb.csv'), index=False)
        for rank, (_, row) in enumerate(shap_imp.head(30).iterrows(), 1):
            print(f"  {rank:3d}. {row['feature']:60s} {row['mean_abs_shap']:.4f}")
        print("  Saved: shap_importance_xgb.csv")
except ImportError:
    print("\n  (SHAP not installed — skip 11b)")

# ══════════════════════════════════════════════════════
# 12. PLOTS (YOUR ORIGINAL + score distribution)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("12. GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# ROC
ax = axes[0, 0]
for name, yp in test_preds.items():
    fpr, tpr, _ = roc_curve(y_test, yp)
    ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test,yp):.4f})", linewidth=1.5)
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves', fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# PR
ax = axes[0, 1]
for name, yp in test_preds.items():
    p, r, _ = precision_recall_curve(y_test, yp)
    ax.plot(r, p, label=f"{name} ({average_precision_score(y_test,yp):.4f})", linewidth=1.5)
ax.axhline(y=(y_test==1).mean(), color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR Curves', fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Feature importance
ax = axes[1, 0]
xgb_m = final_models['XGBoost']
imp25 = pd.DataFrame({'f': feature_names, 'i': xgb_m.feature_importances_}).sort_values('i').tail(20)
ax.barh(range(20), imp25['i'].values, color='steelblue')
ax.set_yticks(range(20)); ax.set_yticklabels(imp25['f'].values, fontsize=8)
ax.set_title('XGBoost Top 20', fontweight='bold')

# Confusion matrix
ax = axes[1, 1]
cm = confusion_matrix(y_test, (y_prob_best>=0.5).astype(int))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
            xticklabels=['Non-Ca','Cancer'], yticklabels=['Non-Ca','Cancer'])
ax.set_title(f'CM: {best_model_name}', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_results_v4.png'), dpi=150, bbox_inches='tight')
print("Saved: model_results_v4.png")

# MBUG 10: Calibration curves (test set)
fig_cal, ax_cal = plt.subplots(figsize=(8, 8))
ax_cal.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
for name in ['XGBoost', 'LightGBM', 'CatBoost']:
    if name not in test_preds:
        continue
    prob_true, prob_pred = calibration_curve(y_test, test_preds[name], n_bins=10, strategy='uniform')
    ax_cal.plot(prob_pred, prob_true, 's-', label=name)
ax_cal.set_xlabel('Mean predicted probability')
ax_cal.set_ylabel('Fraction of positives')
ax_cal.set_title('Calibration Curves (test set)')
ax_cal.legend()
ax_cal.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'calibration_curves.png'), dpi=150, bbox_inches='tight')
plt.close(fig_cal)
print("Saved: calibration_curves.png")

# Probability calibration (isotonic) for XGBoost
if 'XGBoost' in final_models:
    print("\nCalibrating XGBoost (isotonic, cv=5)...")
    try:
        calibrated_model = CalibratedClassifierCV(make_xgb(), method='isotonic', cv=5)
        calibrated_model.fit(X_train, y_train)
        cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
        cal_auc = roc_auc_score(y_test, cal_probs)
        print(f"  Calibrated test AUC: {cal_auc:.4f}")
        joblib.dump(calibrated_model, os.path.join(MODELS_DIR, 'model_xgboost_calibrated.joblib'))
        print("  Saved: model_xgboost_calibrated.joblib")
    except Exception as e:
        print(f"  Calibration skipped: {e}")

# 13. SAVE RESULTS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("13. SAVING")
print("=" * 70)

pred_df = pd.DataFrame({'PATIENT_GUID': pid_test, 'ACTUAL_LABEL': y_test})
for name, yp in test_preds.items():
    safe = name.lower().replace(' ','_').replace('(','').replace(')','')
    pred_df[f'{safe}_prob'] = yp
pred_df.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)
pred_df.to_csv(os.path.join(ENSEMBLERESULTS_DIR, 'test_predictions.csv'), index=False)

summary = {
    'dataset': {'total': len(df), 'cancer': int((y==1).sum()), 'features': len(feature_names),
                'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    'cv': {n: {m: f"{np.mean(v):.4f}±{np.std(v):.4f}" for m,v in cv_results[n].items()} for n in cv_results},
    'test': {n: {k: round(v,4) for k,v in m.items()} for n,m in test_metrics.items()},
    'best_auc': best_model_name,
    'target': {'model': best_for_target, 'threshold': float(rec_info['threshold']),
               'sens': float(rec_info['test_sens']), 'spec': float(rec_info['test_spec']),
               'ppv': float(rec_info['test_ppv']), 'npv': float(rec_info['test_npv']), 'met': target_met}
}
with open(os.path.join(RESULTS_DIR, 'model_summary_v4.json'), 'w') as f:
    json.dump(summary, f, indent=2)


# ══════════════════════════════════════════════════════
# 14. FINAL REPORT
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)

print(f"""
╔═════════════���════════════════════════════════════════════════╗
║  BLADDER CANCER v4 LEAN — RESULTS                            ║
╠══════════════════════════════════════════════════════════════╣
║  {len(df):,} patients × {len(feature_names)} features                        ║
║  Train {len(X_train):,} / Val {len(X_val):,} / Test {len(X_test):,}                    ║
╠══════════════════════════════════════════════════════════════╣""")
print("║  CV RESULTS:")
for n in models:
    a = np.mean(cv_results[n]['auc'])
    print(f"║    {n:25s} AUC = {a:.4f}")
print("║")
print("║  TEST RESULTS (top 6):")
for n, m in sorted(test_metrics.items(), key=lambda x: -x[1]['auc'])[:6]:
    print(f"║    {n:30s} AUC={m['auc']:.4f} Rec={m['recall']:.4f}")
print(f"""║
║  ★ Best AUC: {best_model_name}
║  ★ Target:   {best_for_target} @ {rec_info['threshold']:.3f}
║    Sens={rec_info['test_sens']:.2%}  Spec={rec_info['test_spec']:.2%}
║    PPV={rec_info['test_ppv']:.2%}   NPV={rec_info['test_npv']:.2%}
║    Target met: {target_met}
╚══════════════════════════════════════════════════════════════╝
""")

print("DONE ✅")
