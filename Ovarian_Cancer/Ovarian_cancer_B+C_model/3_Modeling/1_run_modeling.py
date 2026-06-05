# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — MODELING PIPELINE  (1:1 BALANCED-COHORT VARIANT)
# Clinical + Text + TF-IDF + BERT embeddings
# Multi-seed evaluation with tiered sensitivity/specificity
#
# Cohort source: 1:1 ovarian cohort via SQL Queries/v4/*_v4.sql (non_cancer_ratio = 1).
# Train/val/test are all ~1:1 because the upstream cohort is balanced; no
# train-side downsampling needed. scale_pos_weight = n_neg/n_pos self-evaluates
# to ~1.0 on the balanced data (no double-correction). Per-band calibration runs
# on the val split as usual.
#
# Reads from: 2_Feature_Engineering/results/5_cleanup/
# Writes to:  3_Modeling/results/1_training/{window}/  (all 6 windows)
# ═══════════════════════════════════════════════════════════════

import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score,
)

# Import from shared module so joblib pickle/unpickle works regardless of entry point
from _calibrator import CalibratorWrapper

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
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ─── Import FE config for paths and params ───────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '2_Feature_Engineering'))
import config as fe_config

# ─── Modeling params (from FE config) ────────────────────────
SEEDS = fe_config.SEEDS
N_SELECT_FEATURES = fe_config.N_SELECT_FEATURES
N_SELECT_FEATURES_PER_WINDOW = getattr(fe_config, 'N_SELECT_FEATURES_PER_WINDOW', {})
OPTUNA_TRIALS = fe_config.OPTUNA_TRIALS
TRAIN_RATIO = fe_config.TRAIN_RATIO
VAL_RATIO = fe_config.VAL_RATIO
TEST_RATIO = fe_config.TEST_RATIO
PREFIX = fe_config.PREFIX
CANCER_NAME = fe_config.CANCER_NAME
WINDOWS = fe_config.WINDOWS

SCRIPT_DIR = Path(__file__).resolve().parent
CLEANUP_PATH = fe_config.CLEANUP_RESULTS
RESULTS_PATH = SCRIPT_DIR / 'results'


# ═══════════════════════════════════════════════════════════════
# CLINICAL FEATURE FILTER
# ═══════════════════════════════════════════════════════════════

def is_clinical_feature(col):
    """Return True if the feature is clinically meaningful.
    Aggressively removes utilization/generic features that confound the model."""

    # ALWAYS KEEP: demographics, cancer-specific
    if col in ['AGE_AT_INDEX', 'AGE_BAND', 'LABEL']:
        return True
    if col.startswith(PREFIX):
        return True

    # KEEP: Age features
    if col.startswith(('AGE_', 'AGEX_')):
        return True

    # KEEP: Per-category symptom features (counts, acceleration, recency)
    if col.startswith('OBS_'):
        return True

    # KEEP: Lab VALUES (actual results, slopes, deltas) but NOT has_ever/count
    if col.startswith(('LABTERM_', 'LAB_')):
        if any(x in col for x in ['_has_ever', '_count']):
            return False  # utilization: how many tests ordered
        return True

    # KEEP: Symptom clusters, cross-domain, interactions
    if col.startswith(('CLUSTER_', 'CROSS_', 'INT_')):
        return True

    # KEEP: Investigation patterns (clinically specific)
    if col.startswith('INV_PATTERN_'):
        return True

    # KEEP: Medication per-category (but NOT aggregates)
    if col.startswith('MED_ESC_'):
        return True
    if col.startswith('MED_'):
        # Remove ALL generic aggregates
        if col.startswith('MED_AGG_'):
            return False
        # Keep per-category counts
        return True

    # KEEP: Sequence features (symptom timing)
    if col.startswith('SEQ_'):
        return True

    # KEEP: Per-category recency
    if col.startswith('RECUR_'):
        if '_days_since_last' in col:
            return True
        return False

    # KEEP: Time-decay (symptom-specific weighted scores)
    if col.startswith('DECAY_'):
        # Remove generic total, keep symptom-specific
        if col == 'DECAY_total_weighted_score':
            return False
        return True

    # KEEP: Acceleration features (fine-grained ramp-up signals)
    if col.startswith('ACCEL_'):
        return True

    # ── REMOVE everything below (utilization confounds) ──
    # AGG_* (total events, unique categories, unique codes, events_A/B)
    # MONTHLY_* (raw monthly event bins — how active the patient is)
    # ROLLING3M_* (rolling window counts)
    # RATE_* (events per category)
    # VISIT_* (how many times patient visited GP)
    # TRAJ_* (generic temporal trajectory — events per quarter)
    # CAT_* (per-category raw totals)
    # MEDCAT_* (per-med-category raw totals)
    # MEDREC_* (medication recurrence counts)
    # PAIR_* (co-occurrence — statistical, not clinical)
    # CMPAIR_* (clinical-med pairs — statistical)
    # ENTROPY_* / GINI_* (diversity measures)
    # INV_count_*, INV_acceleration (generic investigation counts)
    # TEMP_* (generic event span)
    # AGG_symptom_* (keep only unique categories, not raw counts)
    if col.startswith('AGG_symptom_unique') or col == 'AGG_new_symptom_cats_in_B':
        return True
    if col.startswith('AGG_symptom_acceleration'):
        return True

    return False


# ═══════════════════════════════════════════════════════════════
# LOAD & MERGE DATA
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load clean matrices + text features for all windows."""
    matrices = {}
    for window in WINDOWS:
        # Prefer parquet, fall back to csv
        path = CLEANUP_PATH / window / f'feature_matrix_clean_{window}.parquet'
        if not path.exists():
            path = CLEANUP_PATH / window / f'feature_matrix_clean_{window}.csv'
        if not path.exists():
            logger.warning(f"Missing feature_matrix_clean_{window} (.parquet or .csv)")
            continue
        if path.suffix == '.parquet':
            fm = pd.read_parquet(path)
        else:
            fm = pd.read_csv(path, index_col=0)
        logger.info(f"Loaded {window}: {fm.shape[0]} x {fm.shape[1]}")

        # Filter to clinical features
        all_cols = [c for c in fm.columns if c != 'LABEL']
        keep = [c for c in all_cols if is_clinical_feature(c)]
        fm = fm[['LABEL'] + keep]
        logger.info(f"  Clinical filter: {len(all_cols)} -> {len(keep)} features")

        fm = fm.fillna(0).loc[:, ~fm.columns.duplicated()]
        matrices[window] = fm
        logger.info(f"  Final: {fm.shape[0]} x {fm.shape[1]}")

    return matrices


# ═══════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════

def select_features(X_train, y_train, n_top=None):
    """Select top features via averaged XGBoost + LightGBM importance.
    Force-include text/embedding features."""
    if n_top is None:
        n_top = N_SELECT_FEATURES

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=n_neg / n_pos,
        random_state=42, eval_metric='auc', verbosity=0,
    )
    xgb_m.fit(X_train, y_train)
    xgb_imp = pd.Series(xgb_m.feature_importances_, index=X_train.columns)

    lgb_m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, num_leaves=31,
        scale_pos_weight=n_neg / n_pos,
        random_state=42, verbose=-1,
    )
    lgb_m.fit(X_train, y_train)
    lgb_imp = pd.Series(lgb_m.feature_importances_, index=X_train.columns)

    # Rank-based fusion: each feature gets the sum of its (rank-normalized) importance in both models.
    def _rank_norm(s):
        r = s.rank(method='average')
        return r / r.max()

    importances = _rank_norm(xgb_imp) + _rank_norm(lgb_imp)

    clinical_feats = list(X_train.columns)
    clinical_imp = importances[clinical_feats]
    top_clinical = clinical_imp.nlargest(n_top * 2).index.tolist()

    label_corr = X_train[top_clinical].corrwith(pd.Series(y_train, index=X_train.index)).abs()
    min_corr = label_corr.quantile(0.1)
    filtered = label_corr[label_corr >= min_corr].index.tolist()
    final = [f for f in clinical_imp.nlargest(len(clinical_imp)).index if f in filtered][:n_top]

    # Force-include AGE_AT_INDEX (needed for per-band calibration; nearly always in
    # top selection anyway, but defensive — if it's somehow excluded, per-band would silently degrade)
    for must_have in ['AGE_AT_INDEX']:
        if must_have in X_train.columns and must_have not in final:
            final.append(must_have)

    logger.info(f"  Selected {len(final)} clinical features")
    return final, importances


# ═══════════════════════════════════════════════════════════════
# SPLITTING
# ═══════════════════════════════════════════════════════════════

def train_val_test_split(X, y, seed):
    """Balance cohort to 1:1, then stratified split into train/val/test.

    The cohort SQL oversamples non-cancer 5x (non_cancer_ratio=5) as a buffer;
    here we downsample the majority class to the minority count BEFORE splitting,
    so train, val, AND test all come out ~1:1 and positive cases are never
    discarded. Mirrors the Truveta/breast build_shared_split(balance_to_1to1=True).
    """
    y_arr = np.asarray(y)
    pos = np.where(y_arr == 1)[0]; neg = np.where(y_arr == 0)[0]
    n = min(len(pos), len(neg))
    rng = np.random.RandomState(seed)
    keep_pos = rng.choice(pos, n, replace=False) if len(pos) > n else pos
    keep_neg = rng.choice(neg, n, replace=False) if len(neg) > n else neg
    keep = np.sort(np.concatenate([keep_pos, keep_neg]))
    if len(keep) < len(y_arr):
        maj = 'cancer' if len(pos) > len(neg) else 'non-cancer'
        logger.info(f"  Balanced cohort to 1:1 — kept {n:,} cancer + {n:,} non-cancer; "
                    f"dropped {len(y_arr)-len(keep):,} surplus {maj} "
                    f"(was {len(pos):,} cancer / {len(neg):,} non-cancer)")
    X = X.iloc[keep] if hasattr(X, 'iloc') else X[keep]
    y = y.iloc[keep] if hasattr(y, 'iloc') else y_arr[keep]

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=seed, stratify=y
    )
    val_ratio = VAL_RATIO / (1 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio, random_state=seed, stratify=y_rest
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════
# THRESHOLD SEARCH
# ═══════════════════════════════════════════════════════════════

def tiered_metric(y_true, y_pred_proba):
    """Score for Optuna: prefer 75/65, then 70/65, then 70/60, else balanced."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    spec = 1 - fpr
    sens = tpr
    valid = (sens >= 0.75) & (spec >= 0.65)
    if valid.sum() > 0:
        return 2.0 + (sens + spec)[valid].max()
    valid = (sens >= 0.70) & (spec >= 0.65)
    if valid.sum() > 0:
        return 1.0 + (sens + spec)[valid].max()
    valid = (sens >= 0.70) & (spec >= 0.60)
    if valid.sum() > 0:
        return 0.5 + (sens + spec)[valid].max()
    return np.max(np.minimum(sens, spec))


def find_best_threshold(y_true, y_pred_proba):
    """Tiered threshold search. Also reports a high-sens operating point (tier90: sens>=90%).

    The high-sens tier is reported as an alternative operating point — not used as the
    primary threshold unless `choose_tier` is called with prefer_high_sens=True."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    spec = 1 - fpr
    sens = tpr
    results = {}

    for tier_name, s_min, sp_min in [
        ('tier0', 0.80, 0.70), ('tier1', 0.75, 0.65),
        ('tier2', 0.70, 0.65), ('tier2b', 0.70, 0.60),
    ]:
        valid = (sens >= s_min) & (spec >= sp_min)
        if valid.sum() > 0:
            scores = (sens + spec) * valid
            idx = np.argmax(scores)
            results[tier_name] = {
                'threshold': thresholds[idx], 'sensitivity': sens[idx],
                'specificity': spec[idx], 'met': True,
                'tier': f'{tier_name} (Sens>={s_min*100:.0f}% Spec>={sp_min*100:.0f}%)',
            }
        else:
            results[tier_name] = {'met': False, 'tier': tier_name}

    # Tier 3: balanced
    balance = np.minimum(sens, spec)
    idx = np.argmax(balance)
    results['tier3'] = {
        'threshold': thresholds[idx], 'sensitivity': sens[idx],
        'specificity': spec[idx], 'met': True, 'tier': 'Tier3 (balanced)',
    }

    # Tier 90: high-sensitivity operating point (sens>=90%, maximise spec)
    valid = sens >= 0.90
    if valid.sum() > 0:
        idx = np.argmax(spec * valid)
        results['tier90'] = {
            'threshold': thresholds[idx], 'sensitivity': sens[idx],
            'specificity': spec[idx], 'met': True,
            'tier': 'Tier90 (Sens>=90%, max Spec)',
        }
    else:
        results['tier90'] = {'met': False, 'tier': 'tier90'}

    return results


def choose_tier(thresh_results, prefer_high_sens=False):
    """Pick best achievable tier.
    If prefer_high_sens=True and tier90 is achievable, use it; otherwise fall back to the
    standard tier ladder (tier0 -> ... -> tier3)."""
    if prefer_high_sens and thresh_results.get('tier90', {}).get('met'):
        return thresh_results['tier90'], thresh_results['tier90']['tier']
    for t in ['tier0', 'tier1', 'tier2', 'tier2b', 'tier3']:
        if thresh_results[t].get('met') and 'threshold' in thresh_results[t]:
            return thresh_results[t], thresh_results[t]['tier']
    return thresh_results['tier3'], 'Tier3 (balanced)'


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════

def get_topk_models_by_auc(y_val, preds_dict, k=3):
    """Return up to k best models by val AUC."""
    aucs = {name: roc_auc_score(y_val, p) for name, p in preds_dict.items()}
    return sorted(aucs.keys(), key=lambda n: aucs[n], reverse=True)[:k]


def optimize_ensemble_weights(y_val, preds_dict):
    """Grid-search weights over N models (N can be 2 or 3). Weights sum to 1."""
    names = list(preds_dict.keys())
    preds = [preds_dict[n] for n in names]
    n = len(preds)
    best_score, best_w = -np.inf, tuple([1.0 / n] * n)

    if n == 2:
        for w1 in np.arange(0.05, 0.96, 0.05):
            w = (w1, 1.0 - w1)
            score = tiered_metric(y_val, w[0] * preds[0] + w[1] * preds[1])
            if score > best_score:
                best_score, best_w = score, w
    elif n == 3:
        step = 0.1
        for w1 in np.arange(0.0, 1.0 + step / 2, step):
            for w2 in np.arange(0.0, 1.0 - w1 + step / 2, step):
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w = (round(w1, 4), round(w2, 4), round(max(w3, 0.0), 4))
                score = tiered_metric(y_val, w[0] * preds[0] + w[1] * preds[1] + w[2] * preds[2])
                if score > best_score:
                    best_score, best_w = score, w
    else:
        # Fallback: equal weights
        best_w = tuple([1.0 / n] * n)

    return best_w


# ═══════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, seed=42):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'scale_pos_weight': n_neg / n_pos,
            'random_state': seed, 'eval_metric': 'auc', 'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict_proba(X_val)[:, 1]
        return tiered_metric(y_val, pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, seed=42):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
            'scale_pos_weight': n_neg / n_pos,
            'random_state': seed, 'verbose': -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = model.predict_proba(X_val)[:, 1]
        return tiered_metric(y_val, pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, seed=42):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 20, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10, log=True),
            'auto_class_weights': 'Balanced',
            'random_seed': seed, 'verbose': 0,
        }
        try:
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
            pred = model.predict_proba(X_val)[:, 1]
            return tiered_metric(y_val, pred)
        except Exception as e:
            logger.debug(f"  CatBoost trial failed: {e}")
            return 0.0  # failed trial gets worst score

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def run_modeling():
    matrices = load_data()
    all_results = []

    for window in WINDOWS:
        if window not in matrices:
            continue

        train_dir = RESULTS_PATH / '1_training' / window
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / 'saved_models').mkdir(exist_ok=True)

        fm = matrices[window]
        X = fm.drop(columns=['LABEL'])
        y = fm['LABEL'].to_numpy(dtype='int64')

        logger.info(f"\n{'#'*70}")
        logger.info(f"  MODELING - {window.upper()} | {X.shape[0]} patients x {X.shape[1]} features")
        logger.info(f"  Pos: {(y==1).sum()} | Neg: {(y==0).sum()}")
        logger.info(f"{'#'*70}")

        # ─── FIXED split (same train/val/test across seeds — enables averaging) ──
        SPLIT_SEED = 42
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, SPLIT_SEED)
        logger.info(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

        # Feature selection (window-specific n_top, on shared train split)
        n_top_window = N_SELECT_FEATURES_PER_WINDOW.get(window, N_SELECT_FEATURES)
        logger.info(f"  Selecting top {n_top_window} features for {window}")
        selected, importances = select_features(X_train, y_train, n_top=n_top_window)
        X_train_s = X_train[selected]
        X_val_s = X_val[selected]
        X_test_s = X_test[selected]
        pd.DataFrame({'feature': selected, 'importance': importances[selected].values}).to_csv(
            train_dir / f'selected_features_{window}.csv', index=False
        )

        # ─── Per-seed: tune + train + stack → val/test preds ──
        val_preds_per_seed, test_preds_per_seed = [], []
        all_seed_models = {}

        for seed in SEEDS:
            logger.info(f"\n  ━━━━ Seed {seed} ━━━━")
            preds_val, preds_test, models = {}, {}, {}

            logger.info(f"  Tuning XGBoost (seed={seed})...")
            xgb_params = tune_xgboost(X_train_s, y_train, X_val_s, y_val, seed=seed) if HAS_OPTUNA else {}
            xgb_params.update({'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
                               'random_state': seed, 'eval_metric': 'auc', 'verbosity': 0})
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            preds_val['xgboost'] = xgb_model.predict_proba(X_val_s)[:, 1]
            preds_test['xgboost'] = xgb_model.predict_proba(X_test_s)[:, 1]
            models['xgboost'] = xgb_model
            logger.info(f"    XGBoost val AUC: {roc_auc_score(y_val, preds_val['xgboost']):.4f}")

            logger.info(f"  Tuning LightGBM (seed={seed})...")
            lgb_params = tune_lightgbm(X_train_s, y_train, X_val_s, y_val, seed=seed) if HAS_OPTUNA else {}
            lgb_params.update({'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
                               'random_state': seed, 'verbose': -1})
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            preds_val['lightgbm'] = lgb_model.predict_proba(X_val_s)[:, 1]
            preds_test['lightgbm'] = lgb_model.predict_proba(X_test_s)[:, 1]
            models['lightgbm'] = lgb_model
            logger.info(f"    LightGBM val AUC: {roc_auc_score(y_val, preds_val['lightgbm']):.4f}")

            if HAS_CATBOOST:
                logger.info(f"  Tuning CatBoost (seed={seed})...")
                cb_params = tune_catboost(X_train_s, y_train, X_val_s, y_val, seed=seed) if HAS_OPTUNA else {}
                cb_params.update({'auto_class_weights': 'Balanced', 'random_seed': seed, 'verbose': 0})
                cb_model = CatBoostClassifier(**cb_params)
                cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                             early_stopping_rounds=50, verbose=0)
                preds_val['catboost'] = cb_model.predict_proba(X_val_s)[:, 1]
                preds_test['catboost'] = cb_model.predict_proba(X_test_s)[:, 1]
                models['catboost'] = cb_model
                logger.info(f"    CatBoost val AUC: {roc_auc_score(y_val, preds_val['catboost']):.4f}")

            # Per-seed stacking via LR meta-learner
            ens_models = list(preds_val.keys())
            stack_X_val = np.column_stack([preds_val[n] for n in ens_models])
            stack_X_test = np.column_stack([preds_test[n] for n in ens_models])
            meta = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
            meta.fit(stack_X_val, y_val)
            ens_val_seed = meta.predict_proba(stack_X_val)[:, 1]
            ens_test_seed = meta.predict_proba(stack_X_test)[:, 1]

            coefs = meta.coef_.flatten()
            seed_weights = list(coefs / max(abs(coefs).sum(), 1e-12))
            seed_desc = 'STACK(LR): ' + ' + '.join(f"{n}({w:+.2f})" for w, n in zip(seed_weights, ens_models))
            logger.info(f"  {seed_desc}")
            logger.info(f"    Seed {seed} stacked val AUC: {roc_auc_score(y_val, ens_val_seed):.4f}")
            logger.info(f"    Seed {seed} stacked test AUC: {roc_auc_score(y_test, ens_test_seed):.4f}")

            val_preds_per_seed.append(ens_val_seed)
            test_preds_per_seed.append(ens_test_seed)
            all_seed_models[seed] = {**models, '_meta': meta, '_ensemble_models': ens_models}

        # ─── Multi-seed ensemble: average predictions across SEEDS ──
        ens_val = np.mean(val_preds_per_seed, axis=0)
        ens_test = np.mean(test_preds_per_seed, axis=0)
        logger.info(f"\n  ━━━━ Multi-seed ensemble ({len(SEEDS)} seeds) ━━━━")
        logger.info(f"    Multi-seed val AUC: {roc_auc_score(y_val, ens_val):.4f}")
        logger.info(f"    Multi-seed test AUC: {roc_auc_score(y_test, ens_test):.4f}")

        # ─── Per-band calibration ──
        cal_idx, th_idx = train_test_split(
            np.arange(len(y_val)), test_size=0.30, random_state=SPLIT_SEED, stratify=y_val
        )

        BAND_BINS = [0, 55, 65, 75, 200]
        BAND_LABELS = ['<55', '55-65', '65-75', '75+']
        MIN_BAND_SIZE = 200

        val_age = (X_val['AGE_AT_INDEX'].values
                   if 'AGE_AT_INDEX' in X_val.columns else np.full(len(y_val), 65.0))
        test_age = (X_test['AGE_AT_INDEX'].values
                    if 'AGE_AT_INDEX' in X_test.columns else np.full(len(y_test), 65.0))
        val_bands = np.asarray(pd.cut(val_age, bins=BAND_BINS, labels=BAND_LABELS, right=False).astype(str))
        test_bands = np.asarray(pd.cut(test_age, bins=BAND_BINS, labels=BAND_LABELS, right=False).astype(str))

        # Global calibrator + threshold (fallback when band has too few patients)
        def _fit_pick_calibrator(x_cal, y_cal, x_th, y_th_local):
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso.fit(x_cal, y_cal)
            iso_pred = iso.transform(x_th)
            iso_a = roc_auc_score(y_th_local, iso_pred) if len(np.unique(y_th_local)) > 1 else 0.0
            platt = LogisticRegression(C=1.0, max_iter=1000)
            platt.fit(x_cal.reshape(-1, 1), y_cal)
            platt_pred = platt.predict_proba(x_th.reshape(-1, 1))[:, 1]
            platt_a = roc_auc_score(y_th_local, platt_pred) if len(np.unique(y_th_local)) > 1 else 0.0
            if iso_a >= platt_a:
                return CalibratorWrapper(iso, 'isotonic'), 'isotonic', iso_pred, iso_a, platt_a
            return CalibratorWrapper(platt, 'platt'), 'platt', platt_pred, iso_a, platt_a

        global_cal, global_choice, _, g_iso_a, g_platt_a = _fit_pick_calibrator(
            ens_val[cal_idx], y_val[cal_idx], ens_val[th_idx], y_val[th_idx])
        global_th_pred = global_cal.transform(ens_val[th_idx])
        global_th_results = find_best_threshold(y_val[th_idx], global_th_pred)
        global_best_tier, global_tier_name = choose_tier(global_th_results)
        global_threshold = global_best_tier['threshold']
        logger.info(f"    Global calibrator: {global_choice} (iso={g_iso_a:.4f}, platt={g_platt_a:.4f})  threshold={global_threshold:.4f}")

        calibrators_per_band = {}
        thresholds_per_band = {}
        tier90_thresholds_per_band = {}

        for band in BAND_LABELS:
            cal_band_mask = (val_bands[cal_idx] == band)
            th_band_mask = (val_bands[th_idx] == band)
            n_cal = int(cal_band_mask.sum())
            n_th = int(th_band_mask.sum())
            if n_cal < MIN_BAND_SIZE or n_th < MIN_BAND_SIZE:
                logger.info(f"    Band {band}: too few (cal={n_cal}, th={n_th}) — uses global")
                calibrators_per_band[band] = global_cal
                thresholds_per_band[band] = global_threshold
                tier90_thresholds_per_band[band] = global_th_results.get('tier90', {}).get('threshold', np.nan)
                continue
            try:
                cal_b, choice_b, _, iso_a_b, platt_a_b = _fit_pick_calibrator(
                    ens_val[cal_idx][cal_band_mask], y_val[cal_idx][cal_band_mask],
                    ens_val[th_idx][th_band_mask], y_val[th_idx][th_band_mask])
                cal_b_pred = cal_b.transform(ens_val[th_idx][th_band_mask])
                band_th_results = find_best_threshold(y_val[th_idx][th_band_mask], cal_b_pred)
                band_best_tier, _ = choose_tier(band_th_results)
                calibrators_per_band[band] = cal_b
                thresholds_per_band[band] = band_best_tier['threshold']
                tier90_thresholds_per_band[band] = band_th_results.get('tier90', {}).get('threshold', np.nan)
                logger.info(f"    Band {band} (cal={n_cal}, th={n_th}): {choice_b} "
                            f"(iso={iso_a_b:.4f}, platt={platt_a_b:.4f}) threshold={band_best_tier['threshold']:.4f}")
            except Exception as e:
                logger.warning(f"    Band {band}: per-band fit failed ({e}) — uses global")
                calibrators_per_band[band] = global_cal
                thresholds_per_band[band] = global_threshold
                tier90_thresholds_per_band[band] = global_th_results.get('tier90', {}).get('threshold', np.nan)

        # ─── Apply per-band calibration to test predictions ──
        ens_test_cal = np.zeros(len(ens_test))
        y_pred = np.zeros(len(ens_test), dtype=int)
        y_pred_t90 = np.zeros(len(ens_test), dtype=int)
        for band in BAND_LABELS:
            mask = (test_bands == band)
            if mask.sum() == 0:
                continue
            cal = calibrators_per_band.get(band, global_cal)
            thr = thresholds_per_band.get(band, global_threshold)
            t90 = tier90_thresholds_per_band.get(band, np.nan)
            ens_test_cal[mask] = cal.transform(ens_test[mask])
            y_pred[mask] = (ens_test_cal[mask] >= thr).astype(int)
            if pd.notna(t90):
                y_pred_t90[mask] = (ens_test_cal[mask] >= t90).astype(int)

        # ─── Final test metrics on multi-seed + per-band calibrated predictions ──
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        test_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        test_auc = roc_auc_score(y_test, ens_test_cal)

        logger.info(f"\n  TEST RESULTS (multi-seed + per-band calibration):")
        logger.info(f"    Sensitivity: {test_sens:.4f}")
        logger.info(f"    Specificity: {test_spec:.4f}")
        logger.info(f"    AUC: {test_auc:.4f}")
        logger.info(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

        # TIER90 metrics (high-sens) using band-specific tier90 thresholds
        tier90_sens = tier90_spec = np.nan
        if y_pred_t90.sum() > 0:
            cm90 = confusion_matrix(y_test, y_pred_t90)
            tn90, fp90, fn90, tp90 = cm90.ravel()
            tier90_sens = tp90 / (tp90 + fn90) if (tp90 + fn90) > 0 else 0
            tier90_spec = tn90 / (tn90 + fp90) if (tn90 + fp90) > 0 else 0
            logger.info(f"  TIER90 (per-band, Sens>=90%): Sens={tier90_sens:.4f} Spec={tier90_spec:.4f}")

        all_results.append({
            'window': window,
            'seeds': str(SEEDS),
            'tier': global_tier_name,
            'sensitivity': test_sens,
            'specificity': test_spec,
            'auc': test_auc,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'global_threshold': global_threshold,
            'thresholds_per_band': str(thresholds_per_band),
            'tier90_sensitivity': tier90_sens,
            'tier90_specificity': tier90_spec,
        })

        # ─── Save artifacts ──
        for seed_, mdict in all_seed_models.items():
            seed_dir = train_dir / 'saved_models' / f'seed_{seed_}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            for name in [k for k in mdict if not k.startswith('_')]:
                joblib.dump(mdict[name], seed_dir / f'{name}_model.pkl')
            joblib.dump(mdict['_meta'], seed_dir / 'meta_learner.pkl')

        joblib.dump({
            'selected_features': selected,
            'seeds': SEEDS,
            'ensemble_models': all_seed_models[SEEDS[0]]['_ensemble_models'],
            'calibrators_per_band': calibrators_per_band,
            'thresholds_per_band': thresholds_per_band,
            'tier90_thresholds_per_band': tier90_thresholds_per_band,
            'global_calibrator': global_cal,
            'global_threshold': global_threshold,
            'band_bins': BAND_BINS,
            'band_labels': BAND_LABELS,
            'window': window,
        }, train_dir / 'saved_models' / 'config.json')

        pred_df = pd.DataFrame({
            'PATIENT_GUID': X_test.index, 'y_true': y_test,
            'y_pred_proba_raw': ens_test,
            'y_pred_proba': ens_test_cal,
            'y_pred': y_pred,
            'y_pred_tier90': y_pred_t90,
            'age_band': test_bands,
        })
        pred_df.to_csv(train_dir / f'predictions_{window}.csv', index=False)

        imp_df = importances[selected].sort_values(ascending=False)
        imp_df.to_csv(train_dir / f'feature_importances_{window}.csv')

    # Final summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_PATH / 'final_results.csv', index=False)
        logger.info(f"\n{'='*70}")
        logger.info(f"  FINAL RESULTS")
        logger.info(f"{'='*70}")
        for _, row in results_df.iterrows():
            t90 = ''
            if pd.notna(row.get('tier90_sensitivity')):
                t90 = f" | TIER90: Sens={row['tier90_sensitivity']:.3f} Spec={row['tier90_specificity']:.3f}"
            logger.info(f"  {row['window']} seeds={row['seeds']}: "
                         f"Sens={row['sensitivity']:.3f} Spec={row['specificity']:.3f} "
                         f"AUC={row['auc']:.3f} [{row['tier']}]{t90}")


if __name__ == '__main__':
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)  # ensure log dir exists
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_PATH / 'modeling.log', mode='w'),
        ]
    )
    run_modeling()
