# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — MODELING PIPELINE
# Clinical + Text + TF-IDF + BERT embeddings
# Multi-seed evaluation with tiered sensitivity/specificity
#
# Reads from: 2_Feature_Engineering/results/5_cleanup/ + 6_text_features/
# Writes to:  3_Modeling/results/{3mo,6mo,12mo}/
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score,
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
OPTUNA_TRIALS = fe_config.OPTUNA_TRIALS
TRAIN_RATIO = fe_config.TRAIN_RATIO
VAL_RATIO = fe_config.VAL_RATIO
TEST_RATIO = fe_config.TEST_RATIO
PREFIX = fe_config.PREFIX
CANCER_NAME = fe_config.CANCER_NAME
WINDOWS = fe_config.WINDOWS

SCRIPT_DIR = Path(__file__).resolve().parent
CLEANUP_PATH = fe_config.CLEANUP_RESULTS
TEXT_PATH = fe_config.TEXT_RESULTS
EMB_PATH = fe_config.EMB_RESULTS
BERT_PATH = fe_config.BERT_RESULTS
RESULTS_PATH = SCRIPT_DIR / 'results'


# ═══════════════════════════════════════════════════════════════
# CLINICAL FEATURE FILTER
# ═══════════════════════════════════════════════════════════════

def is_clinical_feature(col):
    """Return True if the feature is clinically meaningful."""
    if col in ['AGE_AT_INDEX', 'AGE_BAND', 'LABEL']:
        return True
    if col.startswith(('AGE_', 'AGEX_')):
        return True
    if col.startswith('OBS_'):
        return True
    if col.startswith(('LABTERM_', 'LAB_')):
        if '_has_ever' in col:
            return False
        return True
    if col.startswith(PREFIX):
        return True
    if col.startswith(('CLUSTER_', 'CROSS_', 'INT_', 'CAT_', 'MEDCAT_',
                       'MEDREC_', 'MED_ESC_', 'SEQ_', 'INV_PATTERN_',
                       'PAIR_', 'CMPAIR_', 'RECUR_', 'DECAY_')):
        return True
    if col.startswith('MED_'):
        generic_meds = [
            'MED_AGG_total_prescriptions', 'MED_AGG_unique_categories',
            'MED_AGG_unique_drugs', 'MED_AGG_count_A', 'MED_AGG_count_B',
            'MED_AGG_acceleration', 'MED_AGG_polypharmacy',
        ]
        return col not in generic_meds
    if col.startswith('TRAJ_'):
        return any(x in col.lower() for x in ['trend', 'events_Q'])
    if col.startswith('AGG_symptom_') or col == 'AGG_new_symptom_cats_in_B':
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# LOAD & MERGE DATA
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load clean matrices + text features for all windows."""
    matrices = {}
    for window in WINDOWS:
        path = CLEANUP_PATH / window / f'feature_matrix_clean_{window}.csv'
        if not path.exists():
            logger.warning(f"Missing {path}")
            continue
        fm = pd.read_csv(path, index_col=0)
        logger.info(f"Loaded {window}: {fm.shape[0]} x {fm.shape[1]}")

        # Filter to clinical features
        all_cols = [c for c in fm.columns if c != 'LABEL']
        keep = [c for c in all_cols if is_clinical_feature(c)]
        fm = fm[['LABEL'] + keep]
        logger.info(f"  Clinical filter: {len(all_cols)} -> {len(keep)} features")

        # Merge text features
        for name, fpath in [
            ('text',  TEXT_PATH / window / f'text_features_{window}.csv'),
            ('tfidf', EMB_PATH / window / f'text_embeddings_{window}.csv'),
            ('bert',  BERT_PATH / window / f'bert_embeddings_{window}.csv'),
        ]:
            if fpath.exists():
                tdf = pd.read_csv(fpath, index_col=0)
                fm = fm.join(tdf, how='left')
                logger.info(f"  + {tdf.shape[1]} {name} features")

        fm = fm.fillna(0).loc[:, ~fm.columns.duplicated()]
        matrices[window] = fm
        logger.info(f"  Final: {fm.shape[0]} x {fm.shape[1]}")

    return matrices


# ═══════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════

def select_features(X_train, y_train, n_top=None):
    """Select top features using XGBoost importance. Force-include text/embedding features."""
    if n_top is None:
        n_top = N_SELECT_FEATURES

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=n_neg / n_pos,
        random_state=42, eval_metric='auc', verbosity=0,
    )
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)

    text_feats = [f for f in X_train.columns if f.startswith(('TEXT_', 'EMB_', 'BERT_'))]
    clinical_feats = [f for f in X_train.columns if not f.startswith(('TEXT_', 'EMB_', 'BERT_'))]

    n_clinical = n_top - len(text_feats)
    clinical_imp = importances[clinical_feats]
    top_clinical = clinical_imp.nlargest(n_clinical * 2).index.tolist()

    label_corr = X_train[top_clinical].corrwith(y_train).abs()
    min_corr = label_corr.quantile(0.1)
    filtered = label_corr[label_corr >= min_corr].index.tolist()
    selected_clinical = [f for f in clinical_imp.nlargest(len(clinical_imp)).index if f in filtered][:n_clinical]

    final = text_feats + selected_clinical
    logger.info(f"  Selected {len(text_feats)} text + {len(selected_clinical)} clinical = {len(final)} total")
    return final, importances


# ═══════════════════════════════════════════════════════════════
# SPLITTING
# ═══════════════════════════════════════════════════════════════

def train_val_test_split(X, y, seed):
    """Split into train/val/test. Stratified."""
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
    """Tiered threshold search: 80/70 -> 75/65 -> 70/65 -> 70/60 -> balanced."""
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
    return results


def choose_tier(thresh_results):
    """Pick best achievable tier."""
    for t in ['tier0', 'tier1', 'tier2', 'tier2b', 'tier3']:
        if thresh_results[t].get('met') and 'threshold' in thresh_results[t]:
            return thresh_results[t], thresh_results[t]['tier']
    return thresh_results['tier3'], 'Tier3 (balanced)'


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════

def get_top2_models_by_auc(y_val, preds_dict):
    aucs = {name: roc_auc_score(y_val, p) for name, p in preds_dict.items()}
    return sorted(aucs.keys(), key=lambda n: aucs[n], reverse=True)[:2]


def optimize_ensemble_weights(y_val, preds_dict):
    names = list(preds_dict.keys())
    v1, v2 = preds_dict[names[0]], preds_dict[names[1]]
    best_score, best_w = -np.inf, (0.5, 0.5)
    for w1 in np.arange(0.05, 0.96, 0.05):
        w2 = 1.0 - w1
        score = tiered_metric(y_val, w1 * v1 + w2 * v2)
        if score > best_score:
            best_score = score
            best_w = (w1, w2)
    return best_w


# ═══════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': n_neg / n_pos,
            'random_state': 42, 'eval_metric': 'auc', 'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict_proba(X_val)[:, 1]
        return tiered_metric(y_val, pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'scale_pos_weight': n_neg / n_pos,
            'random_state': 42, 'verbose': -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = model.predict_proba(X_val)[:, 1]
        return tiered_metric(y_val, pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': 'Balanced',
            'random_seed': 42, 'verbose': 0,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        pred = model.predict_proba(X_val)[:, 1]
        return tiered_metric(y_val, pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
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
        y = fm['LABEL'].values

        logger.info(f"\n{'#'*70}")
        logger.info(f"  MODELING - {window.upper()} | {X.shape[0]} patients x {X.shape[1]} features")
        logger.info(f"  Pos: {(y==1).sum()} | Neg: {(y==0).sum()}")
        logger.info(f"{'#'*70}")

        for seed in SEEDS:
            logger.info(f"\n  --- Seed {seed} ---")

            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed)
            logger.info(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

            # Feature selection
            selected, importances = select_features(X_train, y_train)
            X_train_s = X_train[selected]
            X_val_s = X_val[selected]
            X_test_s = X_test[selected]

            # Save selected features
            pd.DataFrame({'feature': selected, 'importance': importances[selected].values}).to_csv(
                train_dir / f'selected_features_{window}.csv', index=False
            )

            # Tune & train models
            preds_val, preds_test, models = {}, {}, {}

            logger.info(f"\n  Tuning XGBoost...")
            xgb_params = tune_xgboost(X_train_s, y_train, X_val_s, y_val) if HAS_OPTUNA else {}
            xgb_params.update({'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
                               'random_state': 42, 'eval_metric': 'auc', 'verbosity': 0})
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            preds_val['xgboost'] = xgb_model.predict_proba(X_val_s)[:, 1]
            preds_test['xgboost'] = xgb_model.predict_proba(X_test_s)[:, 1]
            models['xgboost'] = xgb_model
            logger.info(f"    XGBoost val AUC: {roc_auc_score(y_val, preds_val['xgboost']):.4f}")

            logger.info(f"  Tuning LightGBM...")
            lgb_params = tune_lightgbm(X_train_s, y_train, X_val_s, y_val) if HAS_OPTUNA else {}
            lgb_params.update({'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
                               'random_state': 42, 'verbose': -1})
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            preds_val['lightgbm'] = lgb_model.predict_proba(X_val_s)[:, 1]
            preds_test['lightgbm'] = lgb_model.predict_proba(X_test_s)[:, 1]
            models['lightgbm'] = lgb_model
            logger.info(f"    LightGBM val AUC: {roc_auc_score(y_val, preds_val['lightgbm']):.4f}")

            if HAS_CATBOOST:
                logger.info(f"  Tuning CatBoost...")
                cb_params = tune_catboost(X_train_s, y_train, X_val_s, y_val) if HAS_OPTUNA else {}
                cb_params.update({'auto_class_weights': 'Balanced', 'random_seed': 42, 'verbose': 0})
                cb_model = CatBoostClassifier(**cb_params)
                cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                             early_stopping_rounds=50, verbose=0)
                preds_val['catboost'] = cb_model.predict_proba(X_val_s)[:, 1]
                preds_test['catboost'] = cb_model.predict_proba(X_test_s)[:, 1]
                models['catboost'] = cb_model
                logger.info(f"    CatBoost val AUC: {roc_auc_score(y_val, preds_val['catboost']):.4f}")

            # Ensemble: top 2 by val AUC
            top2 = get_top2_models_by_auc(y_val, preds_val)
            top2_val = {n: preds_val[n] for n in top2}
            weights = optimize_ensemble_weights(y_val, top2_val)
            ens_val = weights[0] * preds_val[top2[0]] + weights[1] * preds_val[top2[1]]
            ens_test = weights[0] * preds_test[top2[0]] + weights[1] * preds_test[top2[1]]

            logger.info(f"\n  Ensemble: {top2[0]}({weights[0]:.2f}) + {top2[1]}({weights[1]:.2f})")
            logger.info(f"    Ensemble val AUC: {roc_auc_score(y_val, ens_val):.4f}")
            logger.info(f"    Ensemble test AUC: {roc_auc_score(y_test, ens_test):.4f}")

            # Threshold on val, report on test
            thresh_results = find_best_threshold(y_val, ens_val)
            best_tier, tier_name = choose_tier(thresh_results)
            threshold = best_tier['threshold']

            y_pred = (ens_test >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            test_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_auc = roc_auc_score(y_test, ens_test)

            logger.info(f"\n  TEST RESULTS ({tier_name}):")
            logger.info(f"    Threshold: {threshold:.4f}")
            logger.info(f"    Sensitivity: {test_sens:.4f}")
            logger.info(f"    Specificity: {test_spec:.4f}")
            logger.info(f"    AUC: {test_auc:.4f}")
            logger.info(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

            all_results.append({
                'window': window, 'seed': seed, 'tier': tier_name,
                'threshold': threshold, 'sensitivity': test_sens,
                'specificity': test_spec, 'auc': test_auc,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'ensemble': f"{top2[0]}({weights[0]:.2f})+{top2[1]}({weights[1]:.2f})",
            })

            # Save models
            for name, model in models.items():
                joblib.dump(model, train_dir / 'saved_models' / f'{name}_model.pkl')
            joblib.dump({
                'selected_features': selected, 'threshold': threshold,
                'ensemble_models': top2, 'ensemble_weights': weights,
                'seed': seed, 'window': window,
            }, train_dir / 'saved_models' / 'config.json')

            # Save predictions
            pred_df = pd.DataFrame({
                'PATIENT_GUID': X_test.index, 'y_true': y_test,
                'y_pred_proba': ens_test, 'y_pred': y_pred,
            })
            pred_df.to_csv(train_dir / f'predictions_{window}.csv', index=False)

            # Save feature importances
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
            logger.info(f"  {row['window']} seed={row['seed']}: "
                         f"Sens={row['sensitivity']:.3f} Spec={row['specificity']:.3f} "
                         f"AUC={row['auc']:.3f} [{row['tier']}]")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_PATH / 'modeling.log', mode='w'),
        ]
    )
    run_modeling()
