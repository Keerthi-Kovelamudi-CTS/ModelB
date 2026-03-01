"""
╔══════════════════════════════════════════════════════════════════╗
║  BLADDER CANCER EARLY DETECTION — MODEL TRAINING                 ║
║  Input:  2_Feature_Engineering/cleanupfeatures/ (cleaned matrix) ║
║  Split:  60% train / 20% val / 20% test (stratified)             ║
║  Pipeline: median impute → optional SMOTE → optional feat sel   ║
║  Models:  XGB, LGB, CatBoost, RF, LogReg + ensembles (AUC, recall, XGB+CB 40/60) ║
║  Tuning:  Optuna 75 trials (XGB/LGB); early stopping on val       ║
║  Target:  sens ≥ 80%, spec ≥ 70%; threshold chosen on VAL, test reported ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
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
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

import xgboost as xgb

from collections import defaultdict
import json
import joblib

# Optional: SMOTE, Optuna, LightGBM, CatBoost
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
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

# Config: no SMOTE, XGBoost only
USE_SMOTE = False
USE_FEATURE_SELECTION = False  # tree models handle many features; avoid dropping signal with weak selector
USE_OPTUNA = True and HAS_OPTUNA
OPTUNA_TRIALS = 75  # increased to improve chance of 80% sens / 70% spec
EARLY_STOPPING_ROUNDS = 50
FEATURE_SELECTION_TOP_PCT = 0.90  # keep top 90% of importance (cumulative)

# Paths: input from 2_Feature_Engineering/cleanupfeatures/, outputs within 3_Modeling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLADDER_2 = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(BLADDER_2, '2_Feature_Engineering', 'cleanupfeatures', 'bladder_feature_matrix_v2_cleaned.csv')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
for d in [RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)
if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Input not found: {INPUT_FILE}")
print(f"  Input:  {INPUT_FILE}")
print(f"  Output: {RESULTS_DIR}, {MODELS_DIR}")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Shape: {df.shape}")
print(f"Cancer:     {(df['LABEL']==1).sum():,}")
print(f"Non-cancer: {(df['LABEL']==0).sum():,}")

# Separate features and label
y = df['LABEL'].values
patient_ids = df['PATIENT_GUID'].values
X = df.drop(columns=['LABEL', 'PATIENT_GUID'])

feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")

# ══════════════════════════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT (60/20/20 stratified)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. TRAIN / VAL / TEST SPLIT (60/20/20)")
print("=" * 70)

X_tv, X_test, y_tv, y_test, pid_tv, pid_test = train_test_split(
    X, y, patient_ids, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val, pid_train, pid_val = train_test_split(
    X_tv, y_tv, pid_tv, test_size=0.25, random_state=SEED, stratify=y_tv
)

print(f"Train: {len(X_train):,} ({(y_train==1).sum():,} cancer, {(y_train==0).sum():,} non-cancer)")
print(f"Val:   {len(X_val):,}  ({(y_val==1).sum():,} cancer, {(y_val==0).sum():,} non-cancer)")
print(f"Test:  {len(X_test):,}  ({(y_test==1).sum():,} cancer, {(y_test==0).sum():,} non-cancer)")
print(f"Train cancer %: {(y_train==1).mean()*100:.1f}%")
print(f"Val cancer %:   {(y_val==1).mean()*100:.1f}%")
print(f"Test cancer %:  {(y_test==1).mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 2b. IMPUTATION: only for LR/RF; tree models keep NaN (native support)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2b. IMPUTATION (train median — used only for Logistic Regression & Random Forest)")
print("=" * 70)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
median_impute = X_train[numeric_cols].median()
# Imputed copies for LR and RF only; X_train, X_val, X_test keep NaN for tree models
X_train_imputed = X_train.copy()
X_val_imputed = X_val.copy()
X_test_imputed = X_test.copy()
for c in numeric_cols:
    X_train_imputed[c] = X_train_imputed[c].fillna(median_impute[c])
    X_val_imputed[c] = X_val_imputed[c].fillna(median_impute[c])
    X_test_imputed[c] = X_test_imputed[c].fillna(median_impute[c])
for c in X_train.columns:
    if c not in numeric_cols:
        X_train_imputed[c] = X_train_imputed[c].fillna(0)
        X_val_imputed[c] = X_val_imputed[c].fillna(0)
        X_test_imputed[c] = X_test_imputed[c].fillna(0)
print(f"  Train/val/test keep NaN for XGB/LGB/CatBoost. Imputed copies for LR/RF: {len(numeric_cols)} numeric cols.")

# ══════════════════════════════════════════════════════════════
# 2c. SMOTE on training set only (optional)
# ══════════════════════════════════════════════════════════════
if USE_SMOTE:
    print("\n" + "=" * 70)
    print("2c. SMOTE (oversample minority on train only)")
    print("=" * 70)
    smote = SMOTE(random_state=SEED, k_neighbors=5, n_jobs=-1)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    pid_train = np.arange(len(X_train))  # placeholder; resampling loses original IDs
    print(f"  After SMOTE: train {len(X_train):,} ({(y_train==1).sum():,} cancer, {(y_train==0).sum():,} non-cancer)")
else:
    if not HAS_SMOTE and USE_SMOTE:
        print("  (SMOTE requested but imblearn not installed; skip. pip install imbalanced-learn)")

# ══════════════════════════════════════════════════════════════
# 2d. Feature selection (importance-based on train, optional)
# ══════════════════════════════════════════════════════════════
if USE_FEATURE_SELECTION and len(feature_names) > 50:
    print("\n" + "=" * 70)
    print("2d. FEATURE SELECTION (top importance)")
    print("=" * 70)
    _fs_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1, use_label_encoder=False, eval_metric='auc')
    _fs_model.fit(X_train, y_train)
    imp = pd.Series(_fs_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    cumsum = imp.cumsum()
    thresh = cumsum.max() * FEATURE_SELECTION_TOP_PCT
    n_keep = max(1, np.argmax(cumsum.values >= thresh) + 1)
    n_keep = min(n_keep, max(50, len(feature_names) // 2))
    keep = imp.index[:n_keep].tolist()
    feature_names = keep
    X_train = X_train[feature_names]
    X_val = X_val[feature_names]
    X_test = X_test[feature_names]
    X_train_imputed = X_train_imputed[feature_names]
    X_val_imputed = X_val_imputed[feature_names]
    X_test_imputed = X_test_imputed[feature_names]
    print(f"  Kept {len(feature_names)} features (top {FEATURE_SELECTION_TOP_PCT*100:.0f}% cumulative importance).")
else:
    X_train = X_train[feature_names]
    X_val = X_val[feature_names]
    X_test = X_test[feature_names]
    X_train_imputed = X_train_imputed[feature_names]
    X_val_imputed = X_val_imputed[feature_names]
    X_test_imputed = X_test_imputed[feature_names]

# ══════════════════════════════════════════════════════════════
# 3a. OPTUNA HYPERPARAMETER TUNING (XGBoost only, on val set)
# ══════════════════════════════════════════════════════════════
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / max(pos_count, 1)
xgb_best_params = None
lgb_best_params = None

if USE_OPTUNA:
    print("\n" + "=" * 70)
    print("3a. OPTUNA TUNING (XGBoost)")
    print("=" * 70)

    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight * 0.7, scale_pos_weight * 1.3),
        }
        clf = xgb.XGBClassifier(**params, eval_metric='auc', random_state=SEED, n_jobs=-1,
                                use_label_encoder=False, tree_method='hist', enable_categorical=False,
                                early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    xgb_best_params = study_xgb.best_params
    print(f"  XGBoost best val AUC: {study_xgb.best_value:.4f}")

if USE_OPTUNA and HAS_LGB:
    print("\n" + "=" * 70)
    print("3a. OPTUNA TUNING (LightGBM)")
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
            'is_unbalance': True,
            'metric': 'auc',
            'random_state': SEED, 'n_jobs': -1, 'verbosity': -1, 'force_col_wise': True
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    lgb_best_params = study_lgb.best_params
    print(f"  LightGBM best val AUC: {study_lgb.best_value:.4f}")

# ══════════════════════════════════════════════════════════════
# 3. DEFINE MODELS (params from Optuna if available)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. DEFINING MODELS")
print("=" * 70)
print(f"Scale pos weight: {scale_pos_weight:.2f}")

def _xgb_params():
    if xgb_best_params is not None:
        return {**xgb_best_params, 'eval_metric': 'auc', 'random_state': SEED, 'n_jobs': -1,
                'use_label_encoder': False, 'tree_method': 'hist', 'enable_categorical': False,
                'early_stopping_rounds': EARLY_STOPPING_ROUNDS}
    return dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=5, scale_pos_weight=scale_pos_weight, reg_alpha=0.1, reg_lambda=1.0,
                 eval_metric='auc', random_state=SEED, n_jobs=-1, use_label_encoder=False,
                 tree_method='hist', enable_categorical=False, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

models = {'XGBoost': xgb.XGBClassifier(**_xgb_params())}

# LightGBM: is_unbalance=True (not scale_pos_weight) + metric='auc' for early stopping
if HAS_LGB:
    def _lgb_params():
        base = dict(n_estimators=500, random_state=SEED, n_jobs=-1, verbosity=-1, force_col_wise=True,
                    is_unbalance=True, metric='auc')
        if lgb_best_params is not None:
            return {**base, **lgb_best_params, 'n_estimators': 500}
        return {**base, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31,
                'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.1, 'reg_lambda': 1.0}
    models['LightGBM'] = lgb.LGBMClassifier(**_lgb_params())
# CatBoost: auto_class_weights only (do not combine with scale_pos_weight)
if HAS_CATBOOST:
    models['CatBoost'] = cb.CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights='Balanced',
        l2_leaf_reg=3, nan_mode='Min',
        random_seed=SEED, verbose=0, thread_count=-1
    )
# Random Forest: class_weight for imbalance
models['Random Forest'] = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=10,
    class_weight='balanced', random_state=SEED, n_jobs=-1
)
# Logistic Regression: class_weight for imbalance
models['Logistic Regression'] = LogisticRegression(
    class_weight='balanced', max_iter=1000, random_state=SEED, n_jobs=-1
)

print(f"Models: {list(models.keys())}")
if not HAS_LGB:
    print("  (LightGBM not installed; pip install lightgbm for LGB + ensemble)")
if not HAS_CATBOOST:
    print("  (CatBoost not installed; pip install catboost for CatBoost + ensemble)")

# ══════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION (5-fold stratified)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results = defaultdict(lambda: defaultdict(list))

IMPUTED_MODELS = ('Logistic Regression', 'Random Forest')
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"\n--- Fold {fold}/5 ---")
    Xf_train, Xf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    yf_train, yf_val = y_train[train_idx], y_train[val_idx]
    Xf_train_imp = X_train_imputed.iloc[train_idx]
    Xf_val_imp = X_train_imputed.iloc[val_idx]
    for name, model in models.items():
        use_imp = name in IMPUTED_MODELS
        Xft, Xfv = (Xf_train_imp, Xf_val_imp) if use_imp else (Xf_train, Xf_val)
        if name == 'XGBoost':
            model.fit(Xft, yf_train, eval_set=[(Xfv, yf_val)])
            y_prob = model.predict_proba(Xfv)[:, 1]
        elif name == 'LightGBM' and HAS_LGB:
            model.fit(Xft, yf_train, eval_set=[(Xfv, yf_val)],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            y_prob = model.predict_proba(Xfv)[:, 1]
        elif name == 'CatBoost' and HAS_CATBOOST:
            model.fit(Xft, yf_train, eval_set=(Xfv, yf_val),
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            y_prob = model.predict_proba(Xfv)[:, 1]
        else:
            model.fit(Xft, yf_train)
            y_prob = model.predict_proba(Xfv)[:, 1]
        
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Metrics
        cv_results[name]['auc'].append(roc_auc_score(yf_val, y_prob))
        cv_results[name]['ap'].append(average_precision_score(yf_val, y_prob))
        cv_results[name]['f1'].append(f1_score(yf_val, y_pred))
        cv_results[name]['precision'].append(precision_score(yf_val, y_pred))
        cv_results[name]['recall'].append(recall_score(yf_val, y_pred))
        cv_results[name]['accuracy'].append(accuracy_score(yf_val, y_pred))
    
    # Print fold results
    for name in models:
        auc = cv_results[name]['auc'][-1]
        rec = cv_results[name]['recall'][-1]
        pre = cv_results[name]['precision'][-1]
        print(f"  {name:25s} AUC={auc:.4f}  Recall={rec:.4f}  Precision={pre:.4f}")

# ══════════════════════════════════════════════════════════════
# 5. CV SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. CROSS-VALIDATION SUMMARY")
print("=" * 70)

cv_summary = {}
print(f"\n{'Model':<25s} {'AUC':>10s} {'AP':>10s} {'F1':>10s} {'Prec':>10s} {'Recall':>10s} {'Acc':>10s}")
print("-" * 85)

for name in models:
    row = {}
    for metric in ['auc', 'ap', 'f1', 'precision', 'recall', 'accuracy']:
        vals = cv_results[name][metric]
        row[metric] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    
    cv_summary[name] = row
    print(f"{name:<25s} {row['auc']:>10s} {row['ap']:>10s} {row['f1']:>10s} "
          f"{row['precision']:>10s} {row['recall']:>10s} {row['accuracy']:>10s}")

# ══════════════════════════════════════════════════════════════
# 6. TRAIN FINAL MODELS ON FULL TRAINING SET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. TRAINING FINAL MODELS ON FULL TRAINING SET")
print("=" * 70)

final_models = {}
test_predictions = {}
val_predictions = {}  # for ranking models by validation AUC

for name, model in models.items():
    print(f"\nTraining {name}...")
    use_imp = name in IMPUTED_MODELS
    Xtr, Xva, Xte = (X_train_imputed, X_val_imputed, X_test_imputed) if use_imp else (X_train, X_val, X_test)
    if name == 'XGBoost':
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)])
        y_prob_test = model.predict_proba(Xte)[:, 1]
        y_prob_val = model.predict_proba(Xva)[:, 1]
        final_models[name] = {'model': model}
    elif name == 'LightGBM' and HAS_LGB:
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        y_prob_test = model.predict_proba(Xte)[:, 1]
        y_prob_val = model.predict_proba(Xva)[:, 1]
        final_models[name] = {'model': model}
    elif name == 'CatBoost' and HAS_CATBOOST:
        model.fit(Xtr, y_train, eval_set=(Xva, y_val),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        y_prob_test = model.predict_proba(Xte)[:, 1]
        y_prob_val = model.predict_proba(Xva)[:, 1]
        final_models[name] = {'model': model}
    else:
        model.fit(Xtr, y_train)
        y_prob_test = model.predict_proba(Xte)[:, 1]
        y_prob_val = model.predict_proba(Xva)[:, 1]
        final_models[name] = {'model': model}
    test_predictions[name] = y_prob_test
    val_predictions[name] = y_prob_val

# Ensemble: top 2 models by validation AUC (average their predicted probabilities)
val_auc = {name: roc_auc_score(y_val, val_predictions[name]) for name in val_predictions}
ranked = sorted(val_auc.keys(), key=lambda n: val_auc[n], reverse=True)
top2 = ranked[:2]
if len(top2) >= 2:
    ensemble_probs = np.mean([test_predictions[n] for n in top2], axis=0)
    test_predictions['Ensemble (Top 2)'] = ensemble_probs
    val_predictions['Ensemble (Top 2)'] = np.mean([val_predictions[n] for n in top2], axis=0)
    final_models['Ensemble (Top 2)'] = {'model': None, 'top_2_names': top2}
    print(f"\nEnsemble (Top 2 by val AUC): {top2[0]} (val AUC={val_auc[top2[0]]:.4f}), {top2[1]} (val AUC={val_auc[top2[1]]:.4f})")

# Ensemble by validation RECALL (favour high-sensitivity models for 80% sens target)
val_recall = {}
for name in val_predictions:
    y_pred_val = (val_predictions[name] >= 0.5).astype(int)
    val_recall[name] = recall_score(y_val, y_pred_val)
ranked_by_recall = sorted(val_recall.keys(), key=lambda n: val_recall[n], reverse=True)
top2_recall = [n for n in ranked_by_recall if not n.startswith('Ensemble')][:2]
if len(top2_recall) >= 2:
    test_predictions['Ensemble (Top 2 by recall)'] = np.mean([test_predictions[n] for n in top2_recall], axis=0)
    val_predictions['Ensemble (Top 2 by recall)'] = np.mean([val_predictions[n] for n in top2_recall], axis=0)
    final_models['Ensemble (Top 2 by recall)'] = {'model': None, 'top_2_names': top2_recall}
    print(f"Ensemble (Top 2 by val recall): {top2_recall[0]} (val rec={val_recall[top2_recall[0]]:.4f}), {top2_recall[1]} (val rec={val_recall[top2_recall[1]]:.4f})")

# CatBoost-focused weighted ensemble (highest recall model weighted more)
if HAS_CATBOOST and 'CatBoost' in val_predictions:
    w_xgb, w_cb = 0.4, 0.6
    test_predictions['Ensemble (XGB+CB 40/60)'] = w_xgb * test_predictions['XGBoost'] + w_cb * test_predictions['CatBoost']
    val_predictions['Ensemble (XGB+CB 40/60)'] = w_xgb * val_predictions['XGBoost'] + w_cb * val_predictions['CatBoost']
    final_models['Ensemble (XGB+CB 40/60)'] = {'model': None, 'top_2_names': ['XGBoost', 'CatBoost']}
    print(f"Ensemble (XGB+CatBoost 40/60): weighted toward CatBoost (high recall)")

# ══════════════════════════════════════════════════════════════
# 7. TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. TEST SET EVALUATION")
print("=" * 70)

print(f"\n{'Model':<25s} {'AUC':>8s} {'AP':>8s} {'F1':>8s} {'Prec':>8s} {'Recall':>8s} {'Acc':>8s}")
print("-" * 73)

test_metrics = {}
for name, y_prob in test_predictions.items():
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_test, y_prob),
        'ap': average_precision_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred)
    }
    test_metrics[name] = metrics
    
    print(f"{name:<25s} {metrics['auc']:8.4f} {metrics['ap']:8.4f} {metrics['f1']:8.4f} "
          f"{metrics['precision']:8.4f} {metrics['recall']:8.4f} {metrics['accuracy']:8.4f}")

# ══════════════════════════════════════════════════════════════
# 8. CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. CONFUSION MATRICES (Test Set)")
print("=" * 70)

for name, y_prob in test_predictions.items():
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{name}:")
    print(f"  TP={tp:,} (cancer correctly caught)")
    print(f"  FN={fn:,} (cancer MISSED)")
    print(f"  FP={fp:,} (false alarm)")
    print(f"  TN={tn:,} (non-cancer correct)")
    print(f"  Sensitivity: {tp/(tp+fn)*100:.1f}%")
    print(f"  Specificity: {tn/(tn+fp)*100:.1f}%")
    print(f"  PPV:         {tp/(tp+fp)*100:.1f}%")
    print(f"  NPV:         {tn/(tn+fn)*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 9. OPTIMAL THRESHOLD ANALYSIS (maximise F1)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. OPTIMAL THRESHOLD ANALYSIS")
print("=" * 70)

best_model_name = max(test_metrics, key=lambda k: test_metrics[k]['auc'])
print(f"Best model by AUC: {best_model_name}")

y_prob_best = test_predictions[best_model_name]

# Try thresholds
print(f"\n{'Threshold':>10s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'TP':>6s} {'FN':>6s} {'FP':>6s}")
print("-" * 58)

thresholds_to_try = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
best_f1_thresh = 0.5
best_f1 = 0

for t in thresholds_to_try:
    y_pred_t = (y_prob_best >= t).astype(int)
    
    tp = ((y_pred_t == 1) & (y_test == 1)).sum()
    fn = ((y_pred_t == 0) & (y_test == 1)).sum()
    fp = ((y_pred_t == 1) & (y_test == 0)).sum()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_f1_thresh = t
    
    print(f"{t:10.2f} {prec:8.4f} {rec:8.4f} {f1:8.4f} {tp:6d} {fn:6d} {fp:6d}")

print(f"\nBest F1 threshold: {best_f1_thresh:.2f} (F1={best_f1:.4f})")

# Also find threshold for 80% and 90% sensitivity
print(f"\nSensitivity-targeted thresholds ({best_model_name}):")
for target_sens in [0.80, 0.85, 0.90, 0.95]:
    for t in np.arange(0.01, 0.99, 0.01):
        y_pred_t = (y_prob_best >= t).astype(int)
        tp = ((y_pred_t == 1) & (y_test == 1)).sum()
        fn = ((y_pred_t == 0) & (y_test == 1)).sum()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        if rec >= target_sens:
            fp = ((y_pred_t == 1) & (y_test == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            spec = 1 - fp / (y_test == 0).sum()
            print(f"  Sensitivity≥{target_sens*100:.0f}%: threshold={t:.2f}, "
                  f"precision={prec:.3f}, specificity={spec:.3f}, "
                  f"TP={tp}, FN={fn}, FP={fp}")
            break

# ══════════════════════════════════════════════════════════════
# 9b. TARGET OPERATING POINT: sensitivity ≥ 80%, specificity ≥ 70%
#     Threshold chosen on VALIDATION set; TEST metrics reported at that threshold
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9b. TARGET OPERATING POINT (Sensitivity ≥ 80%, Specificity ≥ 70%)")
print("    Threshold optimized on VALIDATION; test metrics reported.")
print("=" * 70)

TARGET_SENS, TARGET_SPEC = 0.80, 0.70
n_val_pos = (y_val == 1).sum()
n_val_neg = (y_val == 0).sum()
n_pos = (y_test == 1).sum()
n_neg = (y_test == 0).sum()
threshold_grid = np.linspace(0.01, 0.75, 75)  # 0.01–0.75 to maximize chance of 80% sens at ≥70% spec

# Step 1: On VALIDATION set, find (model, threshold) that meet 80% sens & 70% spec; pick best by val specificity
target_operating_points = {}
for name in val_predictions:
    y_prob_val = val_predictions[name]
    valid_thresholds = []
    for t in threshold_grid:
        y_pred_val = (y_prob_val >= t).astype(int)
        tp = ((y_pred_val == 1) & (y_val == 1)).sum()
        fn = ((y_pred_val == 0) & (y_val == 1)).sum()
        fp = ((y_pred_val == 1) & (y_val == 0)).sum()
        tn = n_val_neg - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if sens >= TARGET_SENS and spec >= TARGET_SPEC:
            valid_thresholds.append((t, sens, spec))
    if valid_thresholds:
        best = max(valid_thresholds, key=lambda x: (x[2], x[1]))
        target_operating_points[name] = {'threshold': float(best[0]), 'val_sens': float(best[1]), 'val_spec': float(best[2])}

# Step 2: For each candidate that met target on val, compute TEST metrics at that threshold
for name in list(target_operating_points.keys()):
    t = target_operating_points[name]['threshold']
    y_prob_test = test_predictions[name]
    y_pred_test = (y_prob_test >= t).astype(int)
    tp = ((y_pred_test == 1) & (y_test == 1)).sum()
    fn = ((y_pred_test == 0) & (y_test == 1)).sum()
    fp = ((y_pred_test == 1) & (y_test == 0)).sum()
    tn = n_neg - fp
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    target_operating_points[name].update({
        'sensitivity': sens, 'specificity': spec, 'PPV': ppv, 'NPV': npv,
        'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn)
    })

# Fallback: if no (model, threshold) meets both on val, pick closest on VALIDATION
best_for_target = None
rec = None
target_met = bool(target_operating_points)

if target_operating_points:
    print(f"\nModels and thresholds meeting Sensitivity ≥ {TARGET_SENS*100:.0f}% and Specificity ≥ {TARGET_SPEC*100:.0f}%:\n")
    print(f"{'Model':<25} {'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} {'TP':>6} {'FN':>6} {'FP':>6} {'TN':>6}")
    print("-" * 95)
    for name in target_operating_points:
        r = target_operating_points[name]
        print(f"{name:<25} {r['threshold']:>10.2f} {r['sensitivity']:>11.2%} {r['specificity']:>11.2%} "
              f"{r['TP']:>6d} {r['FN']:>6d} {r['FP']:>6d} {r['TN']:>6d}")
    # Recommended: use best model by AUC if it meets target, else model with highest specificity
    if best_model_name in target_operating_points:
        best_for_target = best_model_name
    else:
        best_for_target = max(
            target_operating_points.keys(),
            key=lambda n: (target_operating_points[n]['specificity'], target_operating_points[n]['sensitivity'])
        )
    rec = target_operating_points[best_for_target]
    print(f"\nRecommended for deployment: {best_for_target} at threshold {rec['threshold']:.2f} (threshold from validation)")
    print(f"  → Test Sensitivity {rec['sensitivity']:.2%}, Specificity {rec['specificity']:.2%}, PPV {rec['PPV']:.2%}, NPV {rec['NPV']:.2%}")
    # Save for inference
    with open(os.path.join(RESULTS_DIR, 'target_threshold.txt'), 'w') as f:
        f.write(f"model={best_for_target}\nthreshold={rec['threshold']:.4f}\n")
        f.write(f"sensitivity={rec['sensitivity']:.4f}\nspecificity={rec['specificity']:.4f}\n")
        f.write(f"ppv={rec['PPV']:.4f}\nnpv={rec['NPV']:.4f}\n")
        top2 = final_models.get(best_for_target, {}).get('top_2_names', [])
        if top2:
            f.write(f"top_2_models={','.join(top2)}\n")
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'target_threshold.txt')}")
    # Save model(s) to models/
    top2 = final_models.get(best_for_target, {}).get('top_2_names', [])
    if top2:
        for mname in top2:
            if mname in final_models and final_models[mname].get('model') is not None:
                path = os.path.join(MODELS_DIR, f"model_{mname.lower().replace(' ', '_')}.joblib")
                joblib.dump(final_models[mname]['model'], path)
                print(f"  Saved: {path}")
        meta = {'recommended': best_for_target, 'top_2_models': top2, 'threshold': rec['threshold'], 'sensitivity': rec['sensitivity'], 'specificity': rec['specificity'], 'ppv': rec['PPV'], 'npv': rec['NPV'], 'feature_names': feature_names, 'median_impute': median_impute, 'target_met': True}
        joblib.dump(meta, os.path.join(MODELS_DIR, 'ensemble_meta.joblib'))
    elif best_for_target in final_models and final_models[best_for_target].get('model') is not None:
        path = os.path.join(MODELS_DIR, f"model_{best_for_target.lower().replace(' ', '_')}.joblib")
        joblib.dump(final_models[best_for_target]['model'], path)
        meta = {'recommended': best_for_target, 'threshold': rec['threshold'], 'sensitivity': rec['sensitivity'], 'specificity': rec['specificity'], 'ppv': rec['PPV'], 'npv': rec['NPV'], 'feature_names': feature_names, 'median_impute': median_impute, 'target_met': True}
        joblib.dump(meta, os.path.join(MODELS_DIR, 'deployment_meta.joblib'))
        print(f"  Saved: {path}")
else:
    # Fallback: on VALIDATION pick (model, threshold) closest to (80%, 70%); then report TEST at that threshold
    print(f"\nNo model achieved both Sensitivity ≥ {TARGET_SENS*100:.0f}% and Specificity ≥ {TARGET_SPEC*100:.0f}% on validation.")
    print("Recommending closest operating point on validation (target_met=False); reporting test metrics.")
    best_dist = np.inf
    best_name, best_t = None, 0.5
    for name in val_predictions:
        y_prob_val = val_predictions[name]
        for t in threshold_grid:
            y_pred_val = (y_prob_val >= t).astype(int)
            tp = ((y_pred_val == 1) & (y_val == 1)).sum()
            fn = ((y_pred_val == 0) & (y_val == 1)).sum()
            fp = ((y_pred_val == 1) & (y_val == 0)).sum()
            tn = n_val_neg - fp
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            dist = (TARGET_SENS - sens) ** 2 + (TARGET_SPEC - spec) ** 2
            if dist < best_dist:
                best_dist = dist
                best_for_target = name
                best_t = t
    # Compute TEST metrics at chosen threshold
    y_prob_test = test_predictions[best_for_target]
    y_pred_test = (y_prob_test >= best_t).astype(int)
    tp = ((y_pred_test == 1) & (y_test == 1)).sum()
    fn = ((y_pred_test == 0) & (y_test == 1)).sum()
    fp = ((y_pred_test == 1) & (y_test == 0)).sum()
    tn = n_neg - fp
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec = {'threshold': float(best_t), 'sensitivity': sens, 'specificity': spec, 'PPV': ppv, 'NPV': npv}
    print(f"\nRecommended for deployment: {best_for_target} at threshold {rec['threshold']:.2f}")
    print(f"  → Test Sensitivity {rec['sensitivity']:.2%}, Specificity {rec['specificity']:.2%}, PPV {rec['PPV']:.2%}, NPV {rec['NPV']:.2%} (target not met)")
    # Save for inference
    with open(os.path.join(RESULTS_DIR, 'target_threshold.txt'), 'w') as f:
        f.write(f"model={best_for_target}\nthreshold={rec['threshold']:.4f}\n")
        f.write(f"sensitivity={rec['sensitivity']:.4f}\nspecificity={rec['specificity']:.4f}\n")
        f.write(f"ppv={rec['PPV']:.4f}\nnpv={rec['NPV']:.4f}\n")
        f.write("target_met=False\n")
        top2 = final_models.get(best_for_target, {}).get('top_2_names', [])
        if top2:
            f.write(f"top_2_models={','.join(top2)}\n")
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'target_threshold.txt')}")
    # Save model(s) to models/
    top2 = final_models.get(best_for_target, {}).get('top_2_names', [])
    if top2:
        for mname in top2:
            if mname in final_models and final_models[mname].get('model') is not None:
                path = os.path.join(MODELS_DIR, f"model_{mname.lower().replace(' ', '_')}.joblib")
                joblib.dump(final_models[mname]['model'], path)
                print(f"  Saved: {path}")
        meta = {'recommended': best_for_target, 'top_2_models': top2, 'threshold': rec['threshold'], 'sensitivity': rec['sensitivity'], 'specificity': rec['specificity'], 'ppv': rec['PPV'], 'npv': rec['NPV'], 'feature_names': feature_names, 'median_impute': median_impute, 'target_met': False}
        joblib.dump(meta, os.path.join(MODELS_DIR, 'ensemble_meta.joblib'))
    elif best_for_target in final_models and final_models[best_for_target].get('model') is not None:
        path = os.path.join(MODELS_DIR, f"model_{best_for_target.lower().replace(' ', '_')}.joblib")
        joblib.dump(final_models[best_for_target]['model'], path)
        meta = {'recommended': best_for_target, 'threshold': rec['threshold'], 'sensitivity': rec['sensitivity'], 'specificity': rec['specificity'], 'ppv': rec['PPV'], 'npv': rec['NPV'], 'feature_names': feature_names, 'median_impute': median_impute, 'target_met': False}
        joblib.dump(meta, os.path.join(MODELS_DIR, 'deployment_meta.joblib'))
        print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════
# 10. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10. FEATURE IMPORTANCE (Top 40)")
print("=" * 70)

for name in list(final_models.keys()):
    if name.startswith('Ensemble'):
        continue
    model = final_models[name]['model']
    if not hasattr(model, 'feature_importances_'):
        continue
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    print(f"\n{name} — Top 40 features:")
    print(f"{'Rank':>4s}  {'Feature':60s} {'Importance':>12s}")
    print("-" * 80)
    for i, row in imp_df.head(40).iterrows():
        rank = imp_df.index.get_loc(i) + 1
        print(f"{rank:4d}  {row['feature']:60s} {row['importance']:12.4f}")
    safe_name = name.lower().replace(' ', '_')
    imp_df.to_csv(os.path.join(RESULTS_DIR, f'feature_importance_{safe_name}.csv'), index=False)

# Logistic Regression coefficients
if 'Logistic Regression' in final_models:
    lr_model = final_models['Logistic Regression']['model']
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr_model.coef_[0]
    })
    coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
    
    print(f"\nLogistic Regression — Top 30 features:")
    print(f"{'Rank':>4s}  {'Feature':60s} {'Coefficient':>12s}")
    print("-" * 80)
    for i, row in coef_df.head(30).iterrows():
        rank = coef_df.index.get_loc(i) + 1
        direction = "↑cancer" if row['coefficient'] > 0 else "↓cancer"
        print(f"{rank:4d}  {row['feature']:60s} {row['coefficient']:+12.4f} {direction}")
    
    coef_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_logistic_regression.csv'), index=False)

# ══════════════════════════════════════════════════════════════
# 11. PLOTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("11. GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# ── 11a. ROC Curves ──
ax = axes[0, 0]
for name, y_prob in test_predictions.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves — Test Set', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── 11b. Precision-Recall Curves ──
ax = axes[0, 1]
for name, y_prob in test_predictions.items():
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    ap_val = average_precision_score(y_test, y_prob)
    ax.plot(rec_arr, prec_arr, label=f"{name} (AP={ap_val:.4f})", linewidth=2)
baseline = (y_test == 1).mean()
ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
ax.set_ylabel('Precision (PPV)', fontsize=12)
ax.set_title('Precision-Recall Curves — Test Set', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── 11c. Feature Importance (XGBoost Top 20) ──
ax = axes[1, 0]
xgb_model = final_models['XGBoost']['model']
imp_df_xgb = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True).tail(20)

ax.barh(range(20), imp_df_xgb['importance'].values, color='steelblue')
ax.set_yticks(range(20))
ax.set_yticklabels(imp_df_xgb['feature'].values, fontsize=8)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('XGBoost — Top 20 Features', fontsize=14, fontweight='bold')

# ── 11d. Confusion Matrix (Best Model) ──
ax = axes[1, 1]
y_pred_best = (y_prob_best >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
            xticklabels=['Non-Cancer', 'Cancer'],
            yticklabels=['Non-Cancer', 'Cancer'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_results.png'), dpi=150, bbox_inches='tight')
print("Saved: model_results.png")

# ── 11e. Feature Importance (XGBoost Top 25) ──
fig2, ax2 = plt.subplots(figsize=(12, 10))
imp_df_25 = pd.DataFrame({'feature': feature_names, 'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=True).tail(25)
y_pos = range(len(imp_df_25))
ax2.barh(y_pos, imp_df_25['importance'].values, color='steelblue')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(imp_df_25['feature'].values, fontsize=8)
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('XGBoost — Top 25 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_comparison.png'), dpi=150, bbox_inches='tight')
print("Saved: feature_importance_comparison.png")

# ══════════════════════════════════════════════════════════════
# 12. SAVE RESULTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("12. SAVING RESULTS")
print("=" * 70)

# Test predictions with patient IDs
pred_df = pd.DataFrame({
    'PATIENT_GUID': pid_test,
    'ACTUAL_LABEL': y_test
})
for name, y_prob in test_predictions.items():
    safe_name = name.lower().replace(' ', '_')
    pred_df[f'{safe_name}_prob'] = y_prob
    pred_df[f'{safe_name}_pred'] = (y_prob >= 0.5).astype(int)

pred_df.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)
print("Saved: test_predictions.csv")

# Summary JSON
summary = {
    'dataset': {
        'total_patients': len(df),
        'cancer': int((y == 1).sum()),
        'non_cancer': int((y == 0).sum()),
        'features': len(feature_names),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    },
    'config': {
        'use_smote': USE_SMOTE,
        'use_feature_selection': USE_FEATURE_SELECTION,
        'use_optuna': USE_OPTUNA,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS
    },
    'cv_results': {
        name: {metric: f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
               for metric, vals in metrics.items()}
        for name, metrics in cv_results.items()
    },
    'test_results': {
        name: {k: round(v, 4) for k, v in metrics.items()}
        for name, metrics in test_metrics.items()
    },
    'best_model': best_model_name,
    'best_f1_threshold': best_f1_thresh,
    'ensemble_top_2': final_models.get('Ensemble (Top 2)', {}).get('top_2_names'),
    'target_operating_point': {
        'target_sensitivity': TARGET_SENS,
        'target_specificity': TARGET_SPEC,
        'target_met': target_met,
        'recommended_model': best_for_target,
        'ensemble_top_2': final_models.get('Ensemble (Top 2)', {}).get('top_2_names') if best_for_target == 'Ensemble (Top 2)' else None,
        'threshold': float(rec['threshold']),
        'sensitivity': float(rec['sensitivity']),
        'specificity': float(rec['specificity']),
        'ppv': float(rec['PPV']),
        'npv': float(rec['NPV'])
    } if (best_for_target and rec) else None
}

with open(os.path.join(RESULTS_DIR, 'model_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved: model_summary.json")

# ══════════════════════════════════════════════════════════════
# 13. FINAL REPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("13. FINAL REPORT")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  BLADDER CANCER EARLY DETECTION — RESULTS                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Dataset:  {len(df):,} patients × {len(feature_names)} features{' ' * (19 - len(str(len(feature_names))))}║
║  Cancer:   {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)                                    ║
║  Split:    Train {len(X_train):,} / Val {len(X_val):,} / Test {len(X_test):,}               ║
║                                                                  ║
║  CROSS-VALIDATION (5-fold):                                      ║""")

for name in models:
    auc_mean = np.mean(cv_results[name]['auc'])
    auc_std = np.std(cv_results[name]['auc'])
    print(f"║    {name:25s} AUC = {auc_mean:.4f} ± {auc_std:.4f}          ║")

print(f"""║                                                                  ║
║  TEST SET:                                                       ║""")

for name, metrics in test_metrics.items():
    print(f"║    {name:25s} AUC={metrics['auc']:.4f} Rec={metrics['recall']:.4f} Pre={metrics['precision']:.4f} ║")

print(f"""║                                                                  ║
║  Best Model: {best_model_name:48s} ║
║  Target (Sens≥80%, Spec≥70%): {str(best_for_target or 'N/A') + (' @ ' + f'{rec["threshold"]:.2f}' if rec else '')[:45]:45s} ║
║                                                                  ║
║  OUTPUT FILES:                                                   ║
║    results/  model_results.png, test_predictions.csv, etc.       ║
║    models/   saved model(s) + deployment_meta.joblib             ║
║    target_threshold.txt  (model, threshold, sens, spec, PPV, NPV) ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("DONE ✅")