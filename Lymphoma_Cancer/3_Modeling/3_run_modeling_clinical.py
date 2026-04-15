# ═══════════════════════════════════════════════════════════════
# LYMPHOMA CANCER — CLINICAL-ONLY MODELING
# Removes generic healthcare utilization features
# Keeps only clinically meaningful features
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score
)

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import catboost
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from pathlib import Path

# Train 75% / Val 10% / Test 15%; early-stop and threshold on val, report on test
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15
SEEDS = [45]  # Starting seed (matches Leukaemia pattern); sweep more seeds [13, 42, 45, 77] once first run completes
USE_ALL_FEATURES = False  # Use feature selection (top N)
N_SELECT_FEATURES = 225  # Select top 225 features
OPTUNA_TRIALS = 75

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR.parent / '2_Feature_Engineering' / 'results' / 'Cleanup_Finalresults'
RESULTS_PATH = SCRIPT_DIR / 'results_clinical'

for window in ['3mo', '6mo', '12mo']:
    (RESULTS_PATH / window).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# LOAD CLEAN MATRICES
# ═══════════════════════════════════════════════════════════════

clean_matrices = {}
for window in ['3mo', '6mo', '12mo']:
    path = BASE_PATH / window / f"feature_matrix_clean_{window}.csv"
    fm = pd.read_csv(path, index_col=0)
    clean_matrices[window] = fm
    print(f"✅ {window}: {fm.shape[0]} × {fm.shape[1]} | Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")


# ═══════════════════════════════════════════════════════════════
# CLINICAL FEATURE FILTER — Remove generic utilization features
# Keep only clinically meaningful features
# ═══════════════════════════════════════════════════════════════

def is_clinical_feature(col):
    """Return True if the feature is clinically meaningful, not generic utilization."""

    # ALWAYS KEEP: demographics
    if col in ['AGE_AT_INDEX', 'AGE_BAND', 'LABEL']:
        return True

    # KEEP: Age bands and age interactions with symptoms
    if col.startswith('AGE_') or col.startswith('AGEX_'):
        return True

    # KEEP: Specific symptom features (counts, acceleration, recency per category)
    # But REMOVE generic OBS totals
    if col.startswith('OBS_'):
        # Keep per-category symptom features
        return True

    # KEEP: Lab VALUES (actual test results, slopes, deltas, abnormality flags)
    if col.startswith('LABTERM_'):
        return True
    if col.startswith('LAB_'):
        # Remove generic "has_ever" (just means had a blood test)
        if '_has_ever' in col:
            return False
        # Keep actual lab values: mean, min, max, delta, slope, std, last, first
        return True

    # KEEP: Lymphoma-specific clinical features
    if col.startswith('LYM_'):
        return True

    # KEEP: Lymphoma sub-prefix features (redundant with LYM_ catch above, kept for clarity)
    if col.startswith('LYM_LAB_'):
        return True
    if col.startswith('LYM_RF_'):
        return True
    if col.startswith('LYM_IPI_'):
        return True
    if col.startswith('LYM_nodal_'):
        return True
    if col.startswith('LYM_bsymptoms_'):
        return True

    # KEEP: Specific symptom clusters
    if col.startswith('CLUSTER_'):
        return True

    # KEEP: Cross-domain clinical interactions
    if col.startswith('CROSS_'):
        return True

    # KEEP: Clinical interaction features
    if col.startswith('INT_'):
        return True

    # KEEP: CODE-level features (specific diagnoses/terms)
    if col.startswith('CODE_'):
        return True

    # KEEP: SNOMED-level features (specific diagnoses/terms)
    if col.startswith('SNOMED_'):
        return True

    # KEEP: Recency scores (symptom-specific timing)
    if col.startswith('RECENCY_'):
        return True

    # KEEP: Recency x Persistence (symptom-specific)
    if col.startswith('RxP_'):
        return True

    # KEEP: Per-category visit persistence
    if col.startswith('VISITS_'):
        return True

    # KEEP: Per-category granular features (days_to_index, duration, gap, B_concentration)
    if col.startswith('CAT_'):
        return True

    # KEEP: Medication features that are category-specific
    # But REMOVE generic medication aggregates
    if col.startswith('MED_'):
        # Remove generic aggregates
        generic_meds = [
            'MED_AGG_total_prescriptions', 'MED_AGG_unique_categories',
            'MED_AGG_unique_drugs', 'MED_AGG_count_A', 'MED_AGG_count_B',
            'MED_AGG_acceleration', 'MED_AGG_polypharmacy'
        ]
        if col in generic_meds:
            return False
        return True

    # KEEP: Per-medication-category granular features
    if col.startswith('MEDCAT_'):
        return True

    # KEEP: Medication recurrence (specific categories)
    if col.startswith('MEDREC_'):
        return True

    # KEEP: Medication escalation patterns (clinically specific)
    if col.startswith('MED_ESC_'):
        return True

    # KEEP: Sequence features (symptom→investigation timing)
    if col.startswith('SEQ_'):
        return True

    # KEEP: Investigation patterns (clinically relevant)
    if col.startswith('INV_PATTERN_'):
        return True
    if col.startswith('INV_') and 'haem' in col.lower():
        return True
    if col.startswith('INV_') and 'lymph' in col.lower():
        return True
    if col.startswith('INV_') and 'imaging' in col.lower():
        return True

    # KEEP: Symptom-specific aggregates
    if col.startswith('AGG_symptom_') or col == 'AGG_new_symptom_cats_in_B':
        return True

    # KEEP: Time-decay weighted symptom scores
    if col.startswith('DECAY_'):
        return True

    # KEEP: Trajectory features for symptoms/infection/bleeding/haem (clinically meaningful)
    if col.startswith('TRAJ_'):
        # Keep infection, bleeding, and haem trajectories, increasing trend
        if 'infection' in col.lower() or 'bleeding' in col.lower() or 'haem' in col.lower() or 'trend' in col.lower():
            return True
        # Keep quarter-level event trajectories
        if 'events_Q' in col:
            return True
        return False

    # KEEP: Clinical co-occurrence pairs
    if col.startswith('PAIR_') or col.startswith('CMPAIR_'):
        return True

    # KEEP: Specific recurrence features
    if col.startswith('RECUR_'):
        return True

    # ── REMOVE everything else (generic utilization) ──
    # AGG_total_events, AGG_unique_categories, AGG_unique_snomed_codes
    # AGG_events_A, AGG_events_B
    # MONTHLY_ (generic monthly event counts)
    # ROLLING3M_ (generic rolling window counts)
    # RATE_ (generic event rates)
    # VISIT_ (generic visit counts — different from VISITS_ per symptom)
    # INV_count_A, INV_count_B, INV_count_total, INV_unique_types, INV_acceleration
    # TEMP_ (generic temporal span)
    # ENTROPY_, GINI_ (generic diversity measures)
    return False


print(f"\n{'═'*70}")
print(f"  FILTERING TO CLINICAL-ONLY FEATURES")
print(f"{'═'*70}")

for window in ['3mo', '6mo', '12mo']:
    fm = clean_matrices[window]
    all_cols = [c for c in fm.columns if c != 'LABEL']
    keep_cols = [c for c in all_cols if is_clinical_feature(c)]
    remove_cols = [c for c in all_cols if not is_clinical_feature(c)]

    print(f"\n  {window}: {len(all_cols)} → {len(keep_cols)} features (removed {len(remove_cols)} generic)")

    # Show what's removed
    removed_prefixes = {}
    for c in remove_cols:
        p = c.split('_')[0]
        removed_prefixes[p] = removed_prefixes.get(p, 0) + 1
    print(f"  Removed groups: {dict(sorted(removed_prefixes.items(), key=lambda x: -x[1]))}")

    clean_matrices[window] = fm[['LABEL'] + keep_cols]
    print(f"  Final: {clean_matrices[window].shape[0]} × {clean_matrices[window].shape[1]}")


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def select_features(X_train, y_train, n_top=150):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=n_neg / n_pos,
        random_state=42, eval_metric='auc', verbosity=0
    )
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_xgb = importances.nlargest(n_top * 2).index.tolist()

    label_corr = X_train[top_xgb].corrwith(y_train).abs()
    min_corr = label_corr.quantile(0.1)
    filtered = label_corr[label_corr >= min_corr].index.tolist()

    final = [f for f in importances.nlargest(len(importances)).index if f in filtered][:n_top]
    return final, importances


def train_val_test_split(X, y, seed):
    """Split into train 75% / val 10% / test 15%. Stratified. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=seed, stratify=y
    )
    val_ratio = VAL_RATIO / (1 - TEST_RATIO)  # val share of rest after test
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio, random_state=seed, stratify=y_rest
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def tiered_metric(y_true, y_pred_proba):
    """
    Single score for Optuna: prefer 75/65, then 70/65, then 70/60, else balanced.
    Returns value to maximize (higher = better).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    specificity = 1 - fpr
    sensitivity = tpr
    # Tier 1: 75/65
    valid = (sensitivity >= 0.75) & (specificity >= 0.65)
    if valid.sum() > 0:
        return 2.0 + (sensitivity + specificity)[valid].max()  # e.g. 2.0–4.0
    # Tier 2: 70/65
    valid = (sensitivity >= 0.70) & (specificity >= 0.65)
    if valid.sum() > 0:
        return 1.0 + (sensitivity + specificity)[valid].max()  # 1.0–3.0
    # Tier 2b: 70/60
    valid = (sensitivity >= 0.70) & (specificity >= 0.60)
    if valid.sum() > 0:
        return 0.5 + (sensitivity + specificity)[valid].max()
    # Tier 3: best balanced
    balance = np.minimum(sensitivity, specificity)
    return np.max(balance)  # 0–1


def choose_tier(thresh_results):
    """Pick best achievable tier: 80/70 → 75/65 → 70/65 → 70/60 → balanced."""
    if thresh_results['tier0']['met'] and 'threshold' in thresh_results['tier0']:
        return thresh_results['tier0'], 'Tier0 (80/70)'
    if thresh_results['tier1']['met'] and 'threshold' in thresh_results['tier1']:
        return thresh_results['tier1'], 'Tier1 (75/65)'
    if thresh_results['tier2']['met'] and 'threshold' in thresh_results['tier2']:
        return thresh_results['tier2'], 'Tier2 (70/65)'
    if thresh_results['tier2b']['met'] and 'threshold' in thresh_results['tier2b']:
        return thresh_results['tier2b'], 'Tier2b (70/60)'
    return thresh_results['tier3'], 'Tier3 (balanced)'



def get_top2_models_by_auc(y_val, preds_val_dict):
    """Return list of 2 model names with highest validation AUC (for Ensemble)."""
    aucs = {name: roc_auc_score(y_val, p) for name, p in preds_val_dict.items()}
    top2 = sorted(aucs.keys(), key=lambda n: aucs[n], reverse=True)[:2]
    return top2


def optimize_ensemble_weights_two(y_val, preds_val_dict, tiered_score_fn=tiered_metric):
    """Grid search over 2 weights (sum=1) for exactly 2 models. preds_val_dict must have 2 keys."""
    names = list(preds_val_dict.keys())
    assert len(names) == 2, "optimize_ensemble_weights_two expects exactly 2 models"
    v1, v2 = preds_val_dict[names[0]], preds_val_dict[names[1]]
    best_score = -np.inf
    best_weights = (0.5, 0.5)
    for w1 in np.arange(0.05, 0.96, 0.05):
        w2 = 1.0 - w1
        ens = w1 * v1 + w2 * v2
        score = tiered_score_fn(y_val, ens)
        if score > best_score:
            best_score = score
            best_weights = (w1, w2)
    return best_weights


def find_best_threshold(y_true, y_pred_proba):
    """
    Tiered threshold search:
      Tier 1: Sens>=75% AND Spec>=65% → maximize (sens + spec)
      Tier 2: Sens>=70% AND Spec>=65% → maximize (sens + spec)
      Tier 3: Best balanced (max min(sens, spec))
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    specificity = 1 - fpr
    sensitivity = tpr

    results = {}

    # Tier 0: 80/70 (preferred)
    valid_t0 = (sensitivity >= 0.80) & (specificity >= 0.70)
    if valid_t0.sum() > 0:
        scores_t0 = (sensitivity + specificity) * valid_t0
        best_idx = np.argmax(scores_t0)
        results['tier0'] = {
            'threshold': thresholds[best_idx],
            'sensitivity': sensitivity[best_idx],
            'specificity': specificity[best_idx],
            'tier': 'Tier0 (Sens>=80% Spec>=70%)',
            'met': True
        }
    else:
        results['tier0'] = {'met': False, 'tier': 'Tier0 (Sens>=80% Spec>=70%)'}

    # Tier 1: 75/65
    valid_t1 = (sensitivity >= 0.75) & (specificity >= 0.65)
    if valid_t1.sum() > 0:
        scores_t1 = (sensitivity + specificity) * valid_t1
        best_idx = np.argmax(scores_t1)
        results['tier1'] = {
            'threshold': thresholds[best_idx],
            'sensitivity': sensitivity[best_idx],
            'specificity': specificity[best_idx],
            'tier': 'Tier1 (Sens>=75% Spec>=65%)',
            'met': True
        }
    else:
        results['tier1'] = {'met': False, 'tier': 'Tier1 (Sens>=75% Spec>=65%)'}

    # Tier 2: 70/65
    valid_t2 = (sensitivity >= 0.70) & (specificity >= 0.65)
    if valid_t2.sum() > 0:
        scores_t2 = (sensitivity + specificity) * valid_t2
        best_idx = np.argmax(scores_t2)
        results['tier2'] = {
            'threshold': thresholds[best_idx],
            'sensitivity': sensitivity[best_idx],
            'specificity': specificity[best_idx],
            'tier': 'Tier2 (Sens>=70% Spec>=65%)',
            'met': True
        }
    else:
        results['tier2'] = {'met': False, 'tier': 'Tier2 (Sens>=70% Spec>=65%)'}

    # Tier 2b: 70/60 (fallback when 65% spec not reachable)
    valid_2b = (sensitivity >= 0.70) & (specificity >= 0.60)
    if valid_2b.sum() > 0:
        scores_2b = (sensitivity + specificity) * valid_2b
        best_idx = np.argmax(scores_2b)
        results['tier2b'] = {
            'threshold': thresholds[best_idx],
            'sensitivity': sensitivity[best_idx],
            'specificity': specificity[best_idx],
            'tier': 'Tier2b (Sens>=70% Spec>=60%)',
            'met': True
        }
    else:
        results['tier2b'] = {'met': False, 'tier': 'Tier2b (Sens>=70% Spec>=60%)'}

    # Tier 3: Best balanced
    balance = np.minimum(sensitivity, specificity)
    best_idx = np.argmax(balance)
    results['tier3'] = {
        'threshold': thresholds[best_idx],
        'sensitivity': sensitivity[best_idx],
        'specificity': specificity[best_idx],
        'tier': 'Tier3 (Best balanced)',
        'met': True
    }

    # Youden's J for comparison
    j = sensitivity + specificity - 1
    best_j_idx = np.argmax(j)
    results['youden'] = {
        'threshold': thresholds[best_j_idx],
        'sensitivity': sensitivity[best_j_idx],
        'specificity': specificity[best_j_idx],
        'tier': "Youden's J",
        'met': True
    }

    # Full operating point table
    results['operating_points'] = []
    for target_spec in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        idx = np.argmin(np.abs(specificity - target_spec))
        t = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        results['operating_points'].append({
            'target_spec': target_spec,
            'threshold': t,
            'sensitivity': sensitivity[idx],
            'specificity': specificity[idx],
        })

    return results


def full_evaluation(y_true, y_pred_proba, threshold, model_name, window_name, tier_name):
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)

    meets_80_70 = '✅' if (sens >= 0.80 and spec >= 0.70) else '  '
    meets_75_65 = '✅' if (sens >= 0.75 and spec >= 0.65) else '  '
    meets_70_65 = '✅' if (sens >= 0.70 and spec >= 0.65) else '  '
    meets_70_60 = '✅' if (sens >= 0.70 and spec >= 0.60) else '  '

    print(f"\n  {model_name} — {window_name} [{tier_name}]")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  {'─'*60}")
    print(f"  AUC-ROC:       {auc_roc:.4f}")
    print(f"  AUC-PR:        {auc_pr:.4f}")
    print(f"  Sensitivity:   {sens:.4f}  {meets_80_70} 80%+ | {meets_75_65} 75%+ | {meets_70_65} 70%+")
    print(f"  Specificity:   {spec:.4f}  {meets_80_70} 70%+ | {meets_70_65} 65%+ | {meets_70_60} 60%+")
    print(f"  Precision:     {prec:.4f}")
    print(f"  NPV:           {npv:.4f}")
    print(f"  F1:            {f1:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {
        'window': window_name, 'model': model_name, 'tier': tier_name,
        'threshold': threshold, 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr,
        'sensitivity': sens, 'specificity': spec, 'precision': prec,
        'NPV': npv, 'F1': f1, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'meets_80_70': sens >= 0.80 and spec >= 0.70,
        'meets_75_65': sens >= 0.75 and spec >= 0.65,
        'meets_70_65': sens >= 0.70 and spec >= 0.65,
        'meets_70_60': sens >= 0.70 and spec >= 0.60,
    }


# ═══════════════════════════════════════════════════════════════
# OPTUNA — OPTIMIZED FOR 75/65 TARGET
# ═══════════════════════════════════════════════════════════════

def tune_xgb(X_train, y_train, spw, n_trials=100, use_smote=False):

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 10.0),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            m = xgb.XGBClassifier(
                **params, scale_pos_weight=spw,
                random_state=42, eval_metric='auc',
                early_stopping_rounds=50, verbosity=0
            )
            m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            pred = m.predict_proba(X_v)[:, 1]
            scores.append(roc_auc_score(y_v, pred))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  XGBoost best CV AUC: {study.best_value:.4f}")
    return study.best_params


def tune_lgb(X_train, y_train, spw, n_trials=100, use_smote=False):

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            m = lgb.LGBMClassifier(
                **params, scale_pos_weight=spw,
                random_state=42, verbose=-1
            )
            m.fit(X_t, y_t, eval_set=[(X_v, y_v)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            pred = m.predict_proba(X_v)[:, 1]
            scores.append(roc_auc_score(y_v, pred))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  LightGBM best CV AUC: {study.best_value:.4f}")
    return study.best_params


def tune_catboost(X_train, y_train, spw, n_trials=75, use_smote=False):
    """Optuna tune CatBoost by 5-fold CV AUC (same pattern as XGB/LGB)."""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.9),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 60),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            m = catboost.CatBoostClassifier(
                **params, class_weights=[1.0, spw],  # [neg, pos] for imbalance
                random_state=42, verbose=0
            )
            m.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=50, verbose=False)
            pred = m.predict_proba(X_v)[:, 1]
            scores.append(roc_auc_score(y_v, pred))
        return np.mean(scores)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  CatBoost best CV AUC: {study.best_value:.4f}")
    return study.best_params


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP — Multiple seeds, train/val/test, calibration, threshold on val
# ═══════════════════════════════════════════════════════════════

all_results = []

for seed in SEEDS:
    for window in ['3mo', '6mo', '12mo']:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed} — {window.upper()}  (Train/Val/Test 75/10/15)")
        print(f"  Target: Sens>=75% + Spec>=65% → 70/65 → 70/60")
        print(f"{'#'*70}")

        fm = clean_matrices[window]
        y = fm['LABEL']
        X = fm.drop(columns=['LABEL']).select_dtypes(include=[np.number])

        # 1. Train/val/test split (75/10/15) — full data, no balancing
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed)
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  Train: {X_train.shape[0]} ({n_pos} pos, {n_neg} neg) | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

        # 2. Feature selection or use all
        if USE_ALL_FEATURES:
            top_features = X_train.columns.tolist()
            X_train_sel = X_train
            X_val_sel = X_val
            X_test_sel = X_test
            print(f"\n  ── Using all features (no selection) ──")
            print(f"  Features: {len(top_features)}")
            if seed == SEEDS[0]:
                _model = xgb.XGBClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7,
                    scale_pos_weight=spw, random_state=42, eval_metric='auc', verbosity=0
                )
                _model.fit(X_train_sel, y_train)
                all_imp = pd.Series(_model.feature_importances_, index=X_train_sel.columns)
                all_imp.sort_values(ascending=False).to_csv(RESULTS_PATH / window / f"feature_importances_{window}.csv")
                pd.DataFrame({'feature': top_features}).to_csv(RESULTS_PATH / window / f"selected_features_{window}.csv", index=False)
        else:
            print(f"\n  ── Feature selection (top {N_SELECT_FEATURES}) ──")
            top_features, all_imp = select_features(X_train, y_train, n_top=N_SELECT_FEATURES)
            X_train_sel = X_train[top_features]
            X_val_sel = X_val[top_features]
            X_test_sel = X_test[top_features]
            print(f"  Selected {len(top_features)} features from {X_train.shape[1]}")
            if seed == SEEDS[0]:
                all_imp.sort_values(ascending=False).to_csv(RESULTS_PATH / window / f"feature_importances_{window}.csv")
                pd.DataFrame({'feature': top_features}).to_csv(RESULTS_PATH / window / f"selected_features_{window}.csv", index=False)

        # 3. Use full train data (no SMOTE / no balancing)
        X_train_res, y_train_res = X_train_sel, y_train
        train_spw = spw

        predictions_raw_val = {}
        predictions_raw_test = {}
        models = {}

        # ══════════════════════════════════════════════════════════
        # XGBoost — Optuna on train (tiered metric), early-stop on val
        # ══════════════════════════════════════════════════════════
        if HAS_OPTUNA:
            print(f"\n  ── Tuning XGBoost ({OPTUNA_TRIALS} trials, AUC) ──")
            best_xgb = tune_xgb(X_train_sel, y_train, spw, n_trials=OPTUNA_TRIALS, use_smote=False)
            xgb_model = xgb.XGBClassifier(
                **best_xgb, scale_pos_weight=train_spw,
                random_state=seed, eval_metric='auc',
                early_stopping_rounds=50, verbosity=0
            )
        else:
            best_xgb = {}
            xgb_model = xgb.XGBClassifier(
                n_estimators=1000, max_depth=6, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.5, min_child_weight=10,
                gamma=2.0, max_delta_step=2, scale_pos_weight=train_spw,
                reg_alpha=2.0, reg_lambda=5.0, random_state=seed,
                eval_metric='auc', early_stopping_rounds=50, verbosity=0
            )

        xgb_model.fit(X_train_res, y_train_res, eval_set=[(X_val_sel, y_val)], verbose=False)
        predictions_raw_val['XGBoost'] = xgb_model.predict_proba(X_val_sel)[:, 1]
        predictions_raw_test['XGBoost'] = xgb_model.predict_proba(X_test_sel)[:, 1]
        models['XGBoost'] = xgb_model

        # ══════════════════════════════════════════════════════════
        # LightGBM — Optuna tuned
        # ══════════════════════════════════════════════════════════
        if HAS_OPTUNA:
            print(f"\n  ── Tuning LightGBM ({OPTUNA_TRIALS} trials, AUC) ──")
            best_lgb = tune_lgb(X_train_sel, y_train, spw, n_trials=OPTUNA_TRIALS, use_smote=False)
            lgb_model = lgb.LGBMClassifier(
                **best_lgb, scale_pos_weight=train_spw,
                random_state=seed, verbose=-1
            )
        else:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=1000, max_depth=6, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.5, min_child_samples=30,
                num_leaves=40, scale_pos_weight=train_spw, reg_alpha=2.0,
                reg_lambda=5.0, random_state=seed, verbose=-1
            )

        lgb_model.fit(
            X_train_res, y_train_res, eval_set=[(X_val_sel, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        predictions_raw_val['LightGBM'] = lgb_model.predict_proba(X_val_sel)[:, 1]
        predictions_raw_test['LightGBM'] = lgb_model.predict_proba(X_test_sel)[:, 1]
        models['LightGBM'] = lgb_model

        # ══════════════════════════════════════════════════════════
        # CatBoost — Optuna on train (AUC), early-stop on val
        # ══════════════════════════════════════════════════════════
        if HAS_CATBOOST:
            if HAS_OPTUNA:
                print(f"\n  ── Tuning CatBoost ({OPTUNA_TRIALS} trials, AUC) ──")
                best_cat = tune_catboost(X_train_sel, y_train, spw, n_trials=OPTUNA_TRIALS, use_smote=False)
                cat_model = catboost.CatBoostClassifier(
                    **best_cat, class_weights=[1.0, train_spw],
                    random_state=seed, verbose=0
                )
            else:
                best_cat = {}
                cat_model = catboost.CatBoostClassifier(
                    iterations=1000, depth=6, learning_rate=0.03,
                    l2_leaf_reg=3.0, subsample=0.7, colsample_bylevel=0.5,
                    min_data_in_leaf=30, class_weights=[1.0, train_spw],
                    random_state=seed, verbose=0
                )
            cat_model.fit(
                X_train_res, y_train_res, eval_set=(X_val_sel, y_val),
                early_stopping_rounds=50, verbose=False
            )
            predictions_raw_val['CatBoost'] = cat_model.predict_proba(X_val_sel)[:, 1]
            predictions_raw_test['CatBoost'] = cat_model.predict_proba(X_test_sel)[:, 1]
            models['CatBoost'] = cat_model

        # 4. Use FULL val set for threshold search (no cal/thresh split)
        #    Raw probabilities from tree models are fine for threshold search
        xgb_val = predictions_raw_val['XGBoost']
        lgb_val = predictions_raw_val['LightGBM']
        xgb_pred = predictions_raw_test['XGBoost']
        lgb_pred = predictions_raw_test['LightGBM']
        if HAS_CATBOOST:
            cat_val = predictions_raw_val['CatBoost']
            cat_pred = predictions_raw_test['CatBoost']
            preds_val_dict = {'XGBoost': xgb_val, 'LightGBM': lgb_val, 'CatBoost': cat_val}
            preds_test_dict = {'XGBoost': xgb_pred, 'LightGBM': lgb_pred, 'CatBoost': cat_pred}
        else:
            preds_val_dict = {'XGBoost': xgb_val, 'LightGBM': lgb_val}
            preds_test_dict = {'XGBoost': xgb_pred, 'LightGBM': lgb_pred}

        # 5. Ensemble: use only top 2 models by validation AUC
        top2_names = get_top2_models_by_auc(y_val, preds_val_dict)
        preds_val_top2 = {n: preds_val_dict[n] for n in top2_names}
        preds_test_top2 = {n: preds_test_dict[n] for n in top2_names}
        weights_two = optimize_ensemble_weights_two(y_val, preds_val_top2)
        w1, w2 = weights_two
        ens_val = w1 * preds_val_top2[top2_names[0]] + w2 * preds_val_top2[top2_names[1]]
        ens_pred = w1 * preds_test_top2[top2_names[0]] + w2 * preds_test_top2[top2_names[1]]
        if seed == SEEDS[0]:
            print(f"  Top 2 for Ensemble: {top2_names[0]}, {top2_names[1]} (by val AUC)")
            print(f"  Optimized ensemble weights: {top2_names[0]}={w1:.2f} {top2_names[1]}={w2:.2f}")

        predictions = {
            'XGBoost': xgb_pred, 'LightGBM': lgb_pred,
            'Ensemble': ens_pred,
        }
        if HAS_CATBOOST:
            predictions['CatBoost'] = cat_pred

        # 6. Threshold search on FULL VAL; report on TEST (lock threshold from val)
        preds_val_for_thresh = {'XGBoost': xgb_val, 'LightGBM': lgb_val, 'Ensemble': ens_val}
        if HAS_CATBOOST:
            preds_val_for_thresh['CatBoost'] = cat_val

        print(f"\n{'═'*65}")
        print(f"  THRESHOLD ON VAL — REPORT ON TEST — {window.upper()} (seed {seed})")
        print(f"{'═'*65}")

        for model_name, pred_test in predictions.items():
            pred_val = preds_val_for_thresh[model_name]
            thresh_results = find_best_threshold(y_val, pred_val)
            chosen, chosen_tier = choose_tier(thresh_results)
            # Evaluate on TEST with val-chosen threshold
            result = full_evaluation(y_test, pred_test, chosen['threshold'], model_name, window, chosen_tier)
            result['seed'] = seed
            all_results.append(result)

            # Also report forced operating points: 80/70 and 75/65
            fpr_t, tpr_t, thresholds_t = roc_curve(y_val, pred_val)
            spec_t = 1 - fpr_t
            for target_sens, target_spec, label in [
                (0.80, 0.70, '80/70'),
                (0.75, 0.65, '75/65'),
            ]:
                # Find threshold where sens >= target on val
                valid = tpr_t >= target_sens
                if valid.sum() > 0:
                    # Among those, pick highest specificity
                    best_idx = np.argmax(spec_t * valid)
                    t = thresholds_t[best_idx] if best_idx < len(thresholds_t) else thresholds_t[-1]
                    val_sens = tpr_t[best_idx]
                    val_spec = spec_t[best_idx]
                    # Evaluate on test
                    y_pred_forced = (pred_test >= t).astype(int)
                    cm_f = confusion_matrix(y_test, y_pred_forced)
                    tn_f, fp_f, fn_f, tp_f = cm_f.ravel()
                    sens_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
                    spec_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
                    met = '✅' if (sens_f >= target_sens and spec_f >= target_spec) else '❌'
                    print(f"    → {label} target: thresh={t:.4f} → TEST Sens={sens_f:.4f} Spec={spec_f:.4f} {met}")
                else:
                    print(f"    → {label} target: not achievable on val")

        # Operating point table (Target: Sens>=75% + Spec>=65% → 70/65 → 70/60) for each model — first seed only
        if seed == SEEDS[0]:
            print(f"\n  ── Operating points (val) — Target: Sens>=75% + Spec>=65% → 70/65 → 70/60 ──")
            for model_name in preds_val_for_thresh:
                thresh_res = find_best_threshold(y_val, preds_val_for_thresh[model_name])
                t1, t2, t2b = thresh_res.get('tier1', {}), thresh_res.get('tier2', {}), thresh_res.get('tier2b', {})
                print(f"\n  {model_name}:")
                print(f"    {'Spec Target':>12s} {'Threshold':>10s} {'Sensitivity':>12s} {'Specificity':>12s}")
                for op in thresh_res['operating_points'][:6]:
                    print(f"    {op['target_spec']:12.0%} {op['threshold']:10.4f} {op['sensitivity']:12.4f} {op['specificity']:12.4f}")
                tier_met = []
                if t1.get('met'): tier_met.append(f"75/65 @ thresh={t1['threshold']:.4f} (Sens={t1['sensitivity']:.4f}, Spec={t1['specificity']:.4f})")
                if t2.get('met'): tier_met.append(f"70/65 @ thresh={t2['threshold']:.4f} (Sens={t2['sensitivity']:.4f}, Spec={t2['specificity']:.4f})")
                if t2b.get('met'): tier_met.append(f"70/60 @ thresh={t2b['threshold']:.4f} (Sens={t2b['sensitivity']:.4f}, Spec={t2b['specificity']:.4f})")
                if tier_met:
                    print(f"    Achieved: {' | '.join(tier_met)}")
                else:
                    print(f"    Achieved: Tier3 (balanced) only")

            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            for name, pred in predictions.items():
                fpr_p, tpr_p, _ = roc_curve(y_test, pred)
                auc = roc_auc_score(y_test, pred)
                axes[0].plot(fpr_p, tpr_p, label=f'{name} ({auc:.4f})', linewidth=2)
            axes[0].plot([0,1],[0,1],'k--',alpha=0.3)
            axes[0].axhline(y=0.75, color='green', linestyle=':', alpha=0.6); axes[0].axhline(y=0.70, color='orange', linestyle=':', alpha=0.6)
            axes[0].axvline(x=0.35, color='red', linestyle=':', alpha=0.6)
            axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title(f'ROC — {window}'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
            for name, pred in predictions.items():
                prec_arr, rec_arr, _ = precision_recall_curve(y_test, pred)
                axes[1].plot(rec_arr, prec_arr, label=f'{name} ({average_precision_score(y_test, pred):.4f})', linewidth=2)
            axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title(f'PR — {window}'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
            fpr_e, tpr_e, thresh_e = roc_curve(y_test, predictions['Ensemble'])
            spec_e = 1 - fpr_e
            min_len = min(len(thresh_e), len(tpr_e), len(spec_e))
            axes[2].plot(thresh_e[:min_len], tpr_e[:min_len], label='Sensitivity', color='blue'); axes[2].plot(thresh_e[:min_len], spec_e[:min_len], label='Specificity', color='red')
            axes[2].axhline(y=0.75, color='green', linestyle=':', alpha=0.6); axes[2].axhline(y=0.65, color='red', linestyle=':', alpha=0.6)
            axes[2].set_xlabel('Threshold'); axes[2].set_ylabel('Score'); axes[2].set_title(f'Sens/Spec — Ensemble — {window}'); axes[2].legend(); axes[2].grid(alpha=0.3); axes[2].set_xlim(0, 1)
            plt.tight_layout()
            plt.savefig(RESULTS_PATH / window / f"threshold_analysis_{window}.png", dpi=150, bbox_inches='tight')
            plt.close()

            feat_imp = pd.Series(models['XGBoost'].feature_importances_, index=top_features).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 10))
            feat_imp.head(30).plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title(f'Top 30 Features — {window}'); ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(RESULTS_PATH / window / f"feature_importance_{window}.png", dpi=150, bbox_inches='tight')
            plt.close()

            if HAS_SHAP:
                try:
                    explainer = shap.TreeExplainer(models['XGBoost'])
                    shap_values = explainer.shap_values(X_test_sel)
                    fig, ax = plt.subplots(figsize=(10, 10))
                    shap.summary_plot(shap_values, X_test_sel, max_display=25, show=False)
                    plt.title(f'SHAP — {window}')
                    plt.tight_layout()
                    plt.savefig(RESULTS_PATH / window / f"shap_{window}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"  ⚠️ SHAP skipped: {e}")

            pred_cols = {'PATIENT_GUID': X_test.index, 'LABEL': y_test.values,
                'XGBoost_prob': xgb_pred, 'LightGBM_prob': lgb_pred,
                'Ensemble_prob': ens_pred}
            if HAS_CATBOOST:
                pred_cols['CatBoost_prob'] = cat_pred
            pred_df = pd.DataFrame(pred_cols)
            pred_df.to_csv(RESULTS_PATH / window / f"predictions_{window}.csv", index=False)

            # ══════════════════════════════════════════════════════════
            # SAVE TRAINED MODELS + CONFIG FOR PREDICTION
            # ══════════════════════════════════════════════════════════
            import joblib, json

            models_dir = RESULTS_PATH / window / 'saved_models'
            models_dir.mkdir(parents=True, exist_ok=True)

            # Save models
            joblib.dump(xgb_model, models_dir / 'xgboost_model.pkl')
            joblib.dump(lgb_model, models_dir / 'lightgbm_model.pkl')
            if HAS_CATBOOST:
                joblib.dump(cat_model, models_dir / 'catboost_model.pkl')

            # Save config: feature list, ensemble weights, threshold
            config = {
                'features': top_features,
                'ensemble_top2': top2_names,
                'ensemble_weights': [float(w1), float(w2)],
                'seed': seed,
                'window': window,
            }

            # Save best threshold per model from val
            for mn in preds_val_for_thresh:
                tr = find_best_threshold(y_val, preds_val_for_thresh[mn])
                chosen_t, chosen_tier_t = choose_tier(tr)
                config[f'threshold_{mn}'] = float(chosen_t['threshold'])
                config[f'tier_{mn}'] = chosen_tier_t

            with open(models_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)

            print(f"  Saved models + config to: {models_dir}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY — Aggregate over seeds (mean +/- SD, 75/65 and 70/65 rates)
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═'*90}")
print(f"  FINAL RESULTS — ALL MODELS, ALL WINDOWS, {len(SEEDS)} SEEDS")
print(f"  (Threshold chosen on val; metrics on test)")
print(f"{'═'*90}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_PATH / "final_threshold_results.csv", index=False)

# Aggregate by (window, model): mean +/- SD, and how often 75/65 / 70/65 achieved
agg = results_df.groupby(['window', 'model']).agg({
    'AUC-ROC': ['mean', 'std'],
    'sensitivity': ['mean', 'std'],
    'specificity': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'F1': ['mean', 'std'],
    'meets_80_70': 'sum',
    'meets_75_65': 'sum',
    'meets_70_65': 'sum',
    'meets_70_60': 'sum',
}).reset_index()
agg.columns = ['window', 'model', 'AUC_mean', 'AUC_std', 'Sens_mean', 'Sens_std', 'Spec_mean', 'Spec_std',
               'Prec_mean', 'Prec_std', 'F1_mean', 'F1_std', 'n_80_70', 'n_75_65', 'n_70_65', 'n_70_60']
agg['n_seeds'] = len(SEEDS)

print(f"\n  MEAN +/- SD (over {len(SEEDS)} seeds) — threshold from val, metrics on test:")
print(f"  {'Window':>6s} {'Model':>10s} {'AUC':>12s} {'Sens':>12s} {'Spec':>12s} {'80/70':>8s} {'75/65':>8s} {'70/65':>8s} {'70/60':>8s}")
print(f"  {'─'*100}")

for _, r in agg.iterrows():
    auc_str = f"{r['AUC_mean']:.4f}±{r['AUC_std']:.4f}" if r['AUC_std'] == r['AUC_std'] else f"{r['AUC_mean']:.4f}"
    sens_str = f"{r['Sens_mean']:.4f}±{r['Sens_std']:.4f}" if r['Sens_std'] == r['Sens_std'] else f"{r['Sens_mean']:.4f}"
    spec_str = f"{r['Spec_mean']:.4f}±{r['Spec_std']:.4f}" if r['Spec_std'] == r['Spec_std'] else f"{r['Spec_mean']:.4f}"
    print(f"  {r['window']:>6s} {r['model']:>10s} {auc_str:>12s} {sens_str:>12s} {spec_str:>12s} "
          f"{int(r['n_80_70'])}/{int(r['n_seeds']):<6d} {int(r['n_75_65'])}/{int(r['n_seeds']):<6d} {int(r['n_70_65'])}/{int(r['n_seeds']):<6d} {int(r['n_70_60'])}/{int(r['n_seeds'])}")

# Best by mean AUC among those achieving 75/65 or 70/65
print(f"\n  ── BEST MODELS (by mean AUC over seeds) ──")
t0_agg = agg[agg['n_80_70'] > 0]
t1_agg = agg[agg['n_75_65'] > 0]
t2_agg = agg[agg['n_70_65'] > 0]
t2b_agg = agg[agg['n_70_60'] > 0]

if len(t0_agg) > 0:
    best0 = t0_agg.loc[t0_agg['AUC_mean'].idxmax()]
    print(f"\n  🏆 BEST 80/70 (Sens>=80% + Spec>=70%): {best0['window']} {best0['model']}")
    print(f"     AUC={best0['AUC_mean']:.4f}±{best0['AUC_std']:.4f} | Sens={best0['Sens_mean']:.4f}±{best0['Sens_std']:.4f} | Spec={best0['Spec_mean']:.4f}±{best0['Spec_std']:.4f}")
    print(f"     Achieved in {int(best0['n_80_70'])}/{int(best0['n_seeds'])} seeds")

if len(t1_agg) > 0:
    best = t1_agg.loc[t1_agg['AUC_mean'].idxmax()]
    print(f"\n  🥈 BEST 75/65 (Sens>=75% + Spec>=65%): {best['window']} {best['model']}")
    print(f"     AUC={best['AUC_mean']:.4f}±{best['AUC_std']:.4f} | Sens={best['Sens_mean']:.4f}±{best['Sens_std']:.4f} | Spec={best['Spec_mean']:.4f}±{best['Spec_std']:.4f}")
    print(f"     Achieved in {int(best['n_75_65'])}/{int(best['n_seeds'])} seeds")

if len(t2_agg) > 0:
    best2 = t2_agg.loc[t2_agg['AUC_mean'].idxmax()]
    print(f"\n  🥉 BEST 70/65 (Sens>=70% + Spec>=65%): {best2['window']} {best2['model']}")
    print(f"     AUC={best2['AUC_mean']:.4f}±{best2['AUC_std']:.4f} | Sens={best2['Sens_mean']:.4f}±{best2['Sens_std']:.4f} | Spec={best2['Spec_mean']:.4f}±{best2['Spec_std']:.4f}")
    print(f"     Achieved in {int(best2['n_70_65'])}/{int(best2['n_seeds'])} seeds")

print(f"\n{'═'*90}")
print(f"  ✅ COMPLETE — Results saved to {RESULTS_PATH}")
print(f"{'═'*90}")
