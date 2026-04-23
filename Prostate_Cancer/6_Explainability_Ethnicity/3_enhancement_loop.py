# ═══════════════════════════════════════════════════════════════
# ENHANCEMENT LOOP — Iteratively remove opaque features & retrain
#
# The loop:
#   1. Load current model + SHAP importance
#   2. Identify opaque features the model relies on
#   3. Remove them from the feature set
#   4. Retrain the model (same hyperparameters)
#   5. Compare performance: did we lose accuracy?
#   6. Recompute SHAP: are top features now explainable?
#   7. If not → repeat (up to max_iterations)
#
# Usage:
#   python 3_enhancement_loop.py --window 12mo
#   python 3_enhancement_loop.py --window 12mo --max-iterations 5
#   python 3_enhancement_loop.py --window 12mo --max-auc-drop 0.02
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import xgboost as xgb
import lightgbm as lgb

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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELING_DIR = PROJECT_DIR / '3_Modeling_Ethnicity'
FE_DIR = PROJECT_DIR / '2_Feature_Engineering_Ethnicity'
RESULTS_DIR = SCRIPT_DIR / 'results'

sys.path.insert(0, str(FE_DIR))
import config as fe_config

from feature_dictionary import get_explainability_class, classify_all_features


def load_data_and_config(window):
    """Load feature matrix, model config, and model."""
    model_dir = MODELING_DIR / 'results' / '1_training' / window / 'saved_models'
    config = joblib.load(model_dir / 'config.json')
    selected = config['selected_features']
    threshold = config['threshold']
    seed = config.get('seed', 42)

    # Load feature matrix
    cleanup_path = fe_config.CLEANUP_RESULTS / window / f'feature_matrix_clean_{window}.csv'
    fm = pd.read_csv(cleanup_path, index_col=0)

    # Merge text features
    for name, path_fn in [
        ('text', fe_config.TEXT_RESULTS),
        ('tfidf', fe_config.EMB_RESULTS),
        ('bert', fe_config.BERT_RESULTS),
    ]:
        fname = f'{"text_features" if name == "text" else "text_embeddings" if name == "tfidf" else "bert_embeddings"}_{window}.csv'
        fpath = path_fn / window / fname
        if fpath.exists():
            tdf = pd.read_csv(fpath, index_col=0)
            fm = fm.join(tdf, how='left')

    fm = fm.fillna(0).loc[:, ~fm.columns.duplicated()]

    missing = [f for f in selected if f not in fm.columns]
    for f in missing:
        fm[f] = 0

    X = fm[selected]
    y = fm['LABEL'].values

    return X, y, config, threshold, seed


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, features, seed=42):
    """Train XGBoost + LightGBM, ensemble, return metrics."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    spw = n_neg / max(n_pos, 1)

    X_tr = X_train[features]
    X_v = X_val[features]
    X_te = X_test[features]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=spw, random_state=seed,
        eval_metric='auc', verbosity=0,
    )
    xgb_model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
    xgb_val = xgb_model.predict_proba(X_v)[:, 1]
    xgb_test = xgb_model.predict_proba(X_te)[:, 1]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=spw, random_state=seed, verbose=-1,
    )
    lgb_model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_val = lgb_model.predict_proba(X_v)[:, 1]
    lgb_test = lgb_model.predict_proba(X_te)[:, 1]

    # Simple average ensemble
    ens_val = 0.5 * xgb_val + 0.5 * lgb_val
    ens_test = 0.5 * xgb_test + 0.5 * lgb_test

    # Find best threshold on val
    fpr, tpr, thresholds = roc_curve(y_val, ens_val)
    spec = 1 - fpr
    balance = np.minimum(tpr, spec)
    best_idx = np.argmax(balance)
    threshold = thresholds[best_idx]

    # Evaluate on test
    auc = roc_auc_score(y_test, ens_test)
    y_pred = (ens_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'auc': auc,
        'sensitivity': sens,
        'specificity': spec_val,
        'threshold': threshold,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
    }


def get_shap_importance(model, X, features):
    """Get mean |SHAP| per feature via SHAP TreeExplainer."""
    X_feat = X[features]
    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_feat)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        return pd.Series(np.abs(shap_values).mean(axis=0), index=features)
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=features)
    return pd.Series(0, index=features)


def run_enhancement_loop(window, max_iterations=3, max_auc_drop=0.015,
                         target_top10_explainable=0.80):
    """Run the iterative explainability enhancement loop."""

    out_dir = RESULTS_DIR / window
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"  EXPLAINABILITY ENHANCEMENT LOOP - {window.upper()}")
    logger.info(f"  Max iterations: {max_iterations}")
    logger.info(f"  Max AUC drop:   {max_auc_drop}")
    logger.info(f"  Target:         {target_top10_explainable*100:.0f}% of top 10 explainable")
    logger.info(f"{'='*70}")

    X, y, config, orig_threshold, seed = load_data_and_config(window)
    all_features = list(X.columns)

    # Split (same seed as training for comparability)
    from sklearn.model_selection import train_test_split
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=fe_config.TEST_RATIO, random_state=seed, stratify=y
    )
    val_ratio = fe_config.VAL_RATIO / (1 - fe_config.TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio, random_state=seed, stratify=y_rest
    )

    # Baseline: train with all features
    logger.info(f"\n  BASELINE (all {len(all_features)} features):")
    baseline = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, all_features, seed)
    logger.info(f"    AUC: {baseline['auc']:.4f} | Sens: {baseline['sensitivity']:.4f} | "
                f"Spec: {baseline['specificity']:.4f}")

    baseline_auc = baseline['auc']
    current_features = all_features.copy()
    # Initialise `result` to baseline so the final summary is defined even when the loop
    # exits with 0 iterations (target already met, no opaque features left, etc.).
    result = baseline
    iteration_log = [{
        'iteration': 0,
        'n_features': len(current_features),
        'n_opaque': len([f for f in current_features if get_explainability_class(f) == 'opaque']),
        'auc': baseline['auc'],
        'sensitivity': baseline['sensitivity'],
        'specificity': baseline['specificity'],
        'removed': [],
        'action': 'baseline',
    }]

    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'─'*70}")
        logger.info(f"  ITERATION {iteration}")
        logger.info(f"{'─'*70}")

        # Get SHAP importance from the LightGBM model (fully supported by SHAP TreeExplainer
        # regardless of xgboost version — avoids the xgboost-3.x / shap-0.49 incompatibility).
        importance = get_shap_importance(baseline['lgb_model'], X_train, current_features)
        importance = importance.sort_values(ascending=False)

        # Classify top features
        top10 = importance.head(10).index.tolist()
        top10_classes = {f: get_explainability_class(f) for f in top10}
        top10_opaque = [f for f, c in top10_classes.items() if c == 'opaque']

        top10_score = (10 - len(top10_opaque)) / 10
        logger.info(f"  Top-10 explainability: {top10_score*100:.0f}% ({len(top10_opaque)} opaque)")

        if top10_score >= target_top10_explainable:
            logger.info(f"  Target reached! Top-10 is {top10_score*100:.0f}% explainable.")
            break

        # Find all opaque features in the current set
        opaque_to_remove = [f for f in current_features if get_explainability_class(f) == 'opaque']

        if not opaque_to_remove:
            logger.info(f"  No opaque features left to remove.")
            break

        logger.info(f"  Removing {len(opaque_to_remove)} opaque features:")
        for f in opaque_to_remove[:10]:
            imp = importance.get(f, 0)
            logger.info(f"    - {f} (SHAP={imp:.6f})")
        if len(opaque_to_remove) > 10:
            logger.info(f"    ... and {len(opaque_to_remove) - 10} more")

        # Remove opaque features
        new_features = [f for f in current_features if f not in opaque_to_remove]
        logger.info(f"  Features: {len(current_features)} -> {len(new_features)}")

        # Retrain
        result = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, new_features, seed)
        auc_drop = baseline_auc - result['auc']

        logger.info(f"  Results after removal:")
        logger.info(f"    AUC:  {result['auc']:.4f} (delta: {-auc_drop:+.4f})")
        logger.info(f"    Sens: {result['sensitivity']:.4f}")
        logger.info(f"    Spec: {result['specificity']:.4f}")

        iteration_log.append({
            'iteration': iteration,
            'n_features': len(new_features),
            'n_opaque': len([f for f in new_features if get_explainability_class(f) == 'opaque']),
            'auc': result['auc'],
            'sensitivity': result['sensitivity'],
            'specificity': result['specificity'],
            'auc_drop': auc_drop,
            'removed': opaque_to_remove,
            'action': 'remove_opaque',
        })

        if auc_drop > max_auc_drop:
            logger.warning(f"  AUC drop {auc_drop:.4f} exceeds limit {max_auc_drop}.")
            logger.warning(f"  Some opaque features carry real signal. Options:")
            logger.warning(f"    1. Accept the drop (accuracy vs explainability tradeoff)")
            logger.warning(f"    2. Add them back selectively")
            logger.warning(f"    3. Replace with explainable alternatives")

            # Try adding back the most important opaque features one by one
            logger.info(f"\n  Trying selective add-back of top opaque features...")
            opaque_by_importance = importance[opaque_to_remove].sort_values(ascending=False)

            best_addback_features = new_features.copy()
            for add_feat in opaque_by_importance.index[:5]:
                test_features = best_addback_features + [add_feat]
                test_result = train_and_evaluate(
                    X_train, y_train, X_val, y_val, X_test, y_test, test_features, seed
                )
                test_drop = baseline_auc - test_result['auc']
                logger.info(f"    + {add_feat}: AUC={test_result['auc']:.4f} (drop={test_drop:+.4f})")

                if test_drop <= max_auc_drop:
                    logger.info(f"      Within limit. Keeping this feature for now.")
                    best_addback_features = test_features
                    result = test_result
                    break

            new_features = best_addback_features

        current_features = new_features
        baseline = result  # update for next iteration's SHAP

    # Final assessment
    final_classification = classify_all_features(current_features)
    final_top10_opaque = len([f for f in importance.head(10).index if get_explainability_class(f) == 'opaque'])

    logger.info(f"\n{'='*70}")
    logger.info(f"  ENHANCEMENT LOOP COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Iterations:           {len(iteration_log) - 1}")
    logger.info(f"  Features:             {len(all_features)} -> {len(current_features)}")
    logger.info(f"  Opaque removed:       {len(all_features) - len(current_features)}")
    logger.info(f"  Final AUC:            {result['auc']:.4f} (baseline: {baseline_auc:.4f}, "
                f"drop: {baseline_auc - result['auc']:.4f})")
    logger.info(f"  Final Sensitivity:    {result['sensitivity']:.4f}")
    logger.info(f"  Final Specificity:    {result['specificity']:.4f}")
    logger.info(f"  Explainable features: {len(final_classification['direct'])} direct + "
                f"{len(final_classification['indirect'])} indirect = "
                f"{len(final_classification['direct']) + len(final_classification['indirect'])} / {len(current_features)}")

    # Save iteration log
    log_df = pd.DataFrame([{k: v for k, v in entry.items() if k != 'removed'} for entry in iteration_log])
    log_df.to_csv(out_dir / f'enhancement_loop_log_{window}.csv', index=False)

    # Save final feature set
    final_features_df = pd.DataFrame({
        'feature': current_features,
        'explainability': [get_explainability_class(f) for f in current_features],
    })
    final_features_df.to_csv(out_dir / f'explainable_features_{window}.csv', index=False)

    # Save summary
    summary = {
        'window': window,
        'iterations': len(iteration_log) - 1,
        'baseline_features': len(all_features),
        'final_features': len(current_features),
        'features_removed': len(all_features) - len(current_features),
        'baseline_auc': round(baseline_auc, 4),
        'final_auc': round(result['auc'], 4),
        'auc_drop': round(baseline_auc - result['auc'], 4),
        'final_sensitivity': round(result['sensitivity'], 4),
        'final_specificity': round(result['specificity'], 4),
        'direct_features': len(final_classification['direct']),
        'indirect_features': len(final_classification['indirect']),
        'opaque_features': len(final_classification['opaque']),
    }
    with open(out_dir / f'enhancement_summary_{window}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  Saved:")
    logger.info(f"    enhancement_loop_log_{window}.csv")
    logger.info(f"    explainable_features_{window}.csv")
    logger.info(f"    enhancement_summary_{window}.json")
    logger.info(f"\n  Next: Re-run 1_explain_predictions.py with the refined model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='12mo', choices=['3mo', '6mo', '12mo'])
    parser.add_argument('--max-iterations', type=int, default=3)
    parser.add_argument('--max-auc-drop', type=float, default=0.015,
                        help='Max acceptable AUC drop from baseline (default: 0.015)')
    parser.add_argument('--target-top10', type=float, default=0.80,
                        help='Target: fraction of top 10 features that are explainable (default: 0.80)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_enhancement_loop(args.window, args.max_iterations, args.max_auc_drop, args.target_top10)
