"""
╔══════════════════════════════════════════════════════════════════╗
║  PROSTATE CANCER — Phase 1.3                                   ║
║  Combined Feature Ranking (Statistical + ML)                   ║
║                                                                ║
║  Input Files (from Data_Input/):                               ║
║    - Pro_positive_obs.csv, Pro_negative_obs.csv                  ║
║    - Pro_positive_med.csv, Pro_negative_med.csv                  ║
║                                                                ║
║  Output Files: output/Scores_prostate/                         ║
║      - prostate_obs_top150.csv                                 ║
║      - prostate_meds_top150.csv                                ║
║      - prostate_combined_top150.csv     ← FINAL DELIVERABLE   ║
║      - prostate_obs_all.csv                                    ║
║      - prostate_meds_all.csv                                   ║
║      - prostate_combined_all.csv                               ║
║      - feature_ranking_YYYYMMDD_HHMMSS.log                    ║
║                                                                ║
║  Method:                                                       ║
║    - Statistical:  prevalence + odds ratio + chi2              ║
║    - ML:           LogisticRegression + RandomForest + GBM     ║
║    - Combined:     weighted blend (default 30/20/50)           ║
║    - Filters to features more common in cases (OR > 1)         ║
║                                                                ║
║  Run: python Ranking.py                                        ║
║  Estimated time: 3-10 minutes                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import logging
import sys
import os
import time
from datetime import datetime


# ════════════════════════════════════════════════════════════════
# CONFIGURATION — PROSTATE
# ════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, 'Data_Input')

CONFIG = {
    'cancer_type': 'prostate',
    'random_seed': 42,
    'top_n_output': 150,
    'min_patients_per_feature': 3,

    'max_ml_features': 500,
    'cv_folds': 3,
    'neg_pos_ratio': 5,

    # Combined weights now include mutual_info as an independent ranking view.
    'combined_weights': {
        'stat_score':     0.25,
        'chi2_score':     0.15,
        'mi_score':       0.15,
        'ml_importance':  0.45,
    },
    'ml_weights': {
        'lr': 0.20,
        'rf': 0.25,
        'gb': 0.35,
        'mi': 0.20,
    },

    'data_dir': _DATA_DIR,
    'input_files': {
        'pos_obs':         'Pro_positive_obs.csv',
        'neg_obs':         'Pro_negative_obs.csv',
        'pos_meds':        'Pro_positive_med.csv',
        'neg_meds':        'Pro_negative_med.csv',
    },

    'code_columns': {
        'observation': 'snomed_id',
        'medication':  'med_code_id',
    },

    'output_dir': os.path.join(_SCRIPT_DIR, 'output', 'Scores_prostate'),
    'confidence_bins':   [-0.01, 0.2, 0.5, 0.8, 1.01],
    'confidence_labels': ['Low', 'Medium', 'High', 'Very High'],

    'min_positive_cohort_size': 50,

    # ── Significance & effect-size filter (replaces the loose OR>1 gate) ──
    # A feature must satisfy ALL of these to enter the ranking:
    'case_enriched_only':  True,
    'min_odds_ratio':      1.2,   # clinical effect size floor
    'max_p_bonferroni':    0.05,  # family-wise multiplicity control
    'min_case_count':      10,    # absolute count of positives with the feature

    # ── Stability selection (bootstrap) ──
    'stability_n_bootstrap':     20,
    'stability_top_k':           150,  # fraction of bootstraps in which the feature lands in top-K
}


# ════════════════════════════════════════════════════════════════
# EVERYTHING BELOW IS IDENTICAL TO MELANOMA/LEUKEMIA SCRIPT
# ════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(CONFIG['output_dir'], f'feature_ranking_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def normalise(arr):
    arr = np.array(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def _read_csv_safe(filepath, code_col):
    """Read CSV; dtype keys must match file headers (e.g. SNOMED_ID) before lowercasing."""
    raw_code_col = code_col.upper()
    df = pd.read_csv(filepath, dtype={raw_code_col: str}, keep_default_na=False)
    df.columns = df.columns.str.strip().str.lower()
    if code_col in df.columns:
        df[code_col] = df[code_col].astype(str).str.strip()
        df[code_col] = df[code_col].replace({'': pd.NA, 'nan': pd.NA})
        df.dropna(subset=[code_col], inplace=True)
    return df


def validate_csv(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Found: {list(df.columns)}")
    if len(df) == 0:
        raise ValueError(f"[{name}] Empty DataFrame!")
    logger.info(f"  ✅ {name}: {len(df):,} rows validated")
    return df


def _format_csv_no_trailing_zeros(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or not pd.api.types.is_float_dtype(out[col]):
            continue
        try:
            valid = out[col].dropna()
            if len(valid) == 0:
                continue
            if not np.isfinite(valid).all():
                continue
            whole = np.abs(valid - np.round(valid)) < 1e-9
            if whole.all():
                out[col] = out[col].astype('Int64')
        except (TypeError, ValueError):
            pass
    return out


CSV_COLUMN_RENAMES = {'med_code_id_pos': 'DMD Code', 'snomed_id_pos': 'SNOMED_ID'}


def save_csv(df, filename, top_n=None):
    filepath = os.path.join(CONFIG['output_dir'], filename)
    output = df.head(top_n).copy() if top_n else df.copy()
    output = output.rename(columns={k: v for k, v in CSV_COLUMN_RENAMES.items() if k in output.columns})
    output = _format_csv_no_trailing_zeros(output)
    output.to_csv(filepath, index=False)
    logger.info(f"  💾 Saved: {filepath} ({len(output):,} rows)")
    return output


def _detect_merge_key(pos_df, neg_df, feature_type):
    code_col = CONFIG['code_columns'].get(feature_type)
    if code_col and code_col in pos_df.columns and code_col in neg_df.columns:
        return code_col
    logger.warning(f"  ⚠️ No code ID column for {feature_type} — falling back to term merge")
    return 'term'


def compute_statistical_ranking(pos_df, neg_df, feature_type='observation'):
    logger.info(f"\n{'='*60}")
    logger.info(f"STATISTICAL RANKING: {feature_type}s")
    logger.info(f"{'='*60}")

    pos_df = pos_df.copy()
    neg_df = neg_df.copy()

    total_pos = int(pos_df['n_patient_count_total'].iloc[0])
    total_neg = int(neg_df['n_patient_count_total'].iloc[0])

    logger.info(f"  Positive: {total_pos:,} patients, {len(pos_df):,} terms")
    logger.info(f"  Negative: {total_neg:,} patients, {len(neg_df):,} terms")

    if total_pos == 0 or total_neg == 0:
        raise ValueError(f"Zero patients! pos={total_pos}, neg={total_neg}")

    if total_pos < CONFIG['min_positive_cohort_size']:
        logger.warning(f"  ⚠️ SMALL POSITIVE COHORT: {total_pos} patients. "
                       f"Results may be unstable.")

    merge_key = _detect_merge_key(pos_df, neg_df, feature_type)

    if merge_key == 'term':
        merged = pos_df.merge(neg_df, on='term', how='outer', suffixes=('_pos', '_neg'))
    else:
        merged = pos_df.merge(neg_df, on=merge_key, how='outer', suffixes=('_pos', '_neg'))
        if 'term_pos' in merged.columns:
            merged['term'] = merged['term_pos'].fillna(merged.get('term_neg', ''))
        elif 'term' not in merged.columns:
            merged['term'] = merged[merge_key].astype(str)

    merged['n_patient_count_pos'] = merged['n_patient_count_pos'].fillna(0).astype(int)
    merged['n_patient_count_neg'] = merged['n_patient_count_neg'].fillna(0).astype(int)

    code_col = CONFIG['code_columns'].get(feature_type)
    if code_col:
        pos_col = f'{code_col}_pos'
        neg_col = f'{code_col}_neg'
        if pos_col in merged.columns:
            merged['code_id'] = merged[pos_col].fillna(merged.get(neg_col, pd.NA))
        elif code_col in merged.columns:
            merged['code_id'] = merged[code_col]
        else:
            merged['code_id'] = pd.NA
    else:
        merged['code_id'] = pd.NA

    merged['code_type'] = 'SNOMED' if feature_type == 'observation' else 'DMD'

    merged['prevalence_pos'] = merged['n_patient_count_pos'] / total_pos
    merged['prevalence_neg'] = merged['n_patient_count_neg'] / total_neg
    merged['prevalence_diff'] = merged['prevalence_pos'] - merged['prevalence_neg']
    merged['prevalence_ratio'] = merged['prevalence_pos'] / merged['prevalence_neg'].replace(0, np.nan)

    a = merged['n_patient_count_pos'] + 0.5
    b = (total_pos - merged['n_patient_count_pos']) + 0.5
    c = merged['n_patient_count_neg'] + 0.5
    d = (total_neg - merged['n_patient_count_neg']) + 0.5
    merged['odds_ratio'] = (a * d) / (b * c)
    merged['log_odds_ratio'] = np.log(merged['odds_ratio'])

    merged['log_or_se'] = np.sqrt(1/a + 1/b + 1/c + 1/d)
    merged['or_ci_lower'] = np.exp(merged['log_odds_ratio'] - 1.96 * merged['log_or_se'])
    merged['or_ci_upper'] = np.exp(merged['log_odds_ratio'] + 1.96 * merged['log_or_se'])

    chi2_vals, p_vals = [], []
    for _, row in merged.iterrows():
        table = np.array([
            [row['n_patient_count_pos'], total_pos - row['n_patient_count_pos']],
            [row['n_patient_count_neg'], total_neg - row['n_patient_count_neg']]
        ])
        try:
            chi2, p, _, _ = chi2_contingency(table, correction=True)
            chi2_vals.append(round(chi2, 6))
            p_vals.append(p)
        except Exception:
            chi2_vals.append(np.nan)
            p_vals.append(np.nan)

    merged['chi_squared'] = chi2_vals
    merged['p_value'] = p_vals
    merged['feature_type'] = feature_type

    merged = merged[merged['n_patient_count_pos'] >= CONFIG['min_patients_per_feature']].copy()
    logger.info(f"  After min cases (>={CONFIG['min_patients_per_feature']}): {len(merged):,} terms")

    # Compute Bonferroni correction BEFORE filtering so the filter can use it.
    n_tests = max(len(merged), 1)
    merged['p_value_bonferroni'] = np.minimum(merged['p_value'] * n_tests, 1.0)

    if CONFIG.get('case_enriched_only', True):
        before = len(merged)
        min_or  = CONFIG.get('min_odds_ratio', 1.0)
        max_p   = CONFIG.get('max_p_bonferroni', 1.0)
        min_n   = CONFIG.get('min_case_count', 1)

        mask = (
            np.isfinite(merged['odds_ratio'])
            & (merged['odds_ratio'] > min_or)
            & (merged['p_value_bonferroni'] < max_p)
            & (merged['n_patient_count_pos'] >= min_n)
        )
        merged = merged[mask].copy()
        logger.info(
            f"  Significance + effect filter  "
            f"(OR>{min_or}, p_bonf<{max_p}, n_pos>={min_n}): "
            f"{len(merged):,} terms kept (dropped {before - len(merged):,})"
        )

    merged = merged.sort_values('odds_ratio', ascending=False).reset_index(drop=True)
    merged['stat_rank'] = range(1, len(merged) + 1)

    logger.info(f"  After all filters: {len(merged):,} terms")
    logger.info(f"  OR > 2.0: {(merged['odds_ratio'] > 2.0).sum()}")
    logger.info(f"  OR > 5.0: {(merged['odds_ratio'] > 5.0).sum()}")
    logger.info(f"  Significant (p<0.05): {(merged['p_value'] < 0.05).sum()}")
    logger.info(f"  Bonferroni sig (p<0.05): {(merged['p_value_bonferroni'] < 0.05).sum()}")

    return merged, total_pos, total_neg


def compute_ml_ranking(stat_df, total_pos, total_neg, feature_type='observation'):
    """ML ranking from aggregate prevalence inputs (no patient-level matrix needed).

    Simulates a binary patient-feature matrix using per-term prevalence in positives vs
    controls, trains LR/RF/GB/MI, and returns per-feature importance scores."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ML RANKING: {feature_type}s")
    logger.info(f"{'='*60}")

    t_start = time.time()
    np.random.seed(CONFIG['random_seed'])

    ml_terms = stat_df['term'].tolist()[:CONFIG['max_ml_features']]

    n_pos = min(int(total_pos), 5000)
    n_neg = min(int(total_neg), n_pos * CONFIG['neg_pos_ratio'])

    logger.info(f"  Simulating {n_pos:,} pos + {n_neg:,} neg, {len(ml_terms)} features")

    X_pos = np.zeros((n_pos, len(ml_terms)))
    X_neg = np.zeros((n_neg, len(ml_terms)))

    for i, term in enumerate(ml_terms):
        row = stat_df[stat_df['term'] == term]
        if len(row) == 0:
            continue
        prev_pos = np.clip(float(row['prevalence_pos'].values[0]), 0, 1)
        prev_neg = np.clip(float(row['prevalence_neg'].values[0]), 0, 1)
        X_pos[:, i] = np.random.binomial(1, prev_pos, n_pos)
        X_neg[:, i] = np.random.binomial(1, prev_neg, n_neg)

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    shuffle_idx = np.random.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]

    return _train_ml_models(X, y, ml_terms, feature_type, t_start)


def _train_ml_models(X, y, feature_list, feature_type, t_start):
    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_seed'])

    n_samples = len(y)
    n_pos = y.sum()
    n_neg = n_samples - n_pos
    sample_weights = np.where(
        y == 1,
        n_samples / (2.0 * n_pos),
        n_samples / (2.0 * n_neg)
    )

    # L1 Logistic Regression
    t0 = time.time()
    logger.info("  Training L1 Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(
        penalty='l1', solver='saga', C=1.0, max_iter=50000, tol=1e-3,
        random_state=CONFIG['random_seed'], class_weight='balanced'
    )
    lr_cv = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    lr.fit(X_scaled, y)
    lr_imp = np.abs(lr.coef_[0])
    logger.info(f"    AUROC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f} | "
                f"Non-zero: {(lr_imp > 0).sum()} | {time.time()-t0:.1f}s")

    # Random Forest
    t0 = time.time()
    logger.info("  Training Random Forest (200 trees)...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=30,
        random_state=CONFIG['random_seed'], class_weight='balanced', n_jobs=-1
    )
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    logger.info(f"    AUROC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f} | {time.time()-t0:.1f}s")

    # Gradient Boosting — manual CV with sample_weight
    t0 = time.time()
    logger.info("  Training Gradient Boosting (150 trees, class-balanced)...")
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        min_samples_leaf=30, random_state=CONFIG['random_seed'], subsample=0.7
    )

    gb_cv_scores = []
    for train_idx, val_idx in cv.split(X, y):
        gb_fold = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            min_samples_leaf=30, random_state=CONFIG['random_seed'], subsample=0.7
        )
        gb_fold.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
        y_pred = gb_fold.predict_proba(X[val_idx])[:, 1]
        gb_cv_scores.append(roc_auc_score(y[val_idx], y_pred))
    gb_cv = np.array(gb_cv_scores)

    gb.fit(X, y, sample_weight=sample_weights)
    gb_imp = gb.feature_importances_
    logger.info(f"    AUROC: {gb_cv.mean():.4f} ± {gb_cv.std():.4f} | {time.time()-t0:.1f}s")

    # Mutual Information — captures non-linear associations that OR/linear models miss.
    t0 = time.time()
    logger.info("  Computing Mutual Information...")
    mi_imp = mutual_info_classif(
        X, y, discrete_features=True, random_state=CONFIG['random_seed']
    )
    logger.info(f"    MI computed for {len(mi_imp)} features | {time.time()-t0:.1f}s")

    w = CONFIG['ml_weights']
    ml_scores = pd.DataFrame({
        'term': feature_list,
        'lr_importance': normalise(lr_imp),
        'rf_importance': normalise(rf_imp),
        'gb_importance': normalise(gb_imp),
        'mi_importance': normalise(mi_imp),
        'lr_cv_auroc': lr_cv.mean(),
        'rf_cv_auroc': rf_cv.mean(),
        'gb_cv_auroc': gb_cv.mean(),
    })
    ml_scores['ml_importance'] = (
        w.get('lr', 0) * ml_scores['lr_importance'] +
        w.get('rf', 0) * ml_scores['rf_importance'] +
        w.get('gb', 0) * ml_scores['gb_importance'] +
        w.get('mi', 0) * ml_scores['mi_importance']
    )
    ml_scores = ml_scores.sort_values('ml_importance', ascending=False).reset_index(drop=True)
    ml_scores['ml_rank'] = range(1, len(ml_scores) + 1)

    logger.info(f"  ✅ Complete: {time.time()-t_start:.1f}s total")
    return ml_scores


def compute_stability_scores(stat_df, total_pos, total_neg, feature_type,
                             n_bootstrap=None, top_k=None):
    """Bootstrap stability: fraction of bootstrap iterations in which each feature lands in top-K.

    For each bootstrap, the prevalence-simulated ML ranking is re-run with a different seed.
    A stable feature lands in the top-K every time; unstable features float in and out and
    should be treated with caution (likely capturing sample-specific noise).

    Returns: pd.Series indexed by feature term, value in [0, 1].
    """
    n_bootstrap = n_bootstrap or CONFIG['stability_n_bootstrap']
    top_k       = top_k       or CONFIG['stability_top_k']

    logger.info(f"\n{'='*60}")
    logger.info(f"STABILITY SELECTION: {feature_type}s ({n_bootstrap} bootstraps)")
    logger.info(f"{'='*60}")

    feature_list = stat_df['term'].tolist()[:CONFIG['max_ml_features']]
    # Dedupe to avoid duplicate-index issues in the returned Series.
    unique_features = list(dict.fromkeys(feature_list))
    appearances = {f: 0 for f in unique_features}

    saved_seed = CONFIG['random_seed']
    saved_cv = CONFIG['cv_folds']
    CONFIG['cv_folds'] = 2
    try:
        for b in range(n_bootstrap):
            logger.info(f"  Bootstrap {b+1}/{n_bootstrap}...")
            CONFIG['random_seed'] = saved_seed + b  # vary seed for simulation
            try:
                ml_b = compute_ml_ranking(stat_df, total_pos, total_neg, feature_type)
            except Exception as e:
                logger.warning(f"    Bootstrap {b+1} failed: {e}")
                continue
            top_terms = set(ml_b.head(top_k)['term'])
            for t in top_terms:
                if t in appearances:
                    appearances[t] += 1
    finally:
        CONFIG['random_seed'] = saved_seed
        CONFIG['cv_folds'] = saved_cv

    stability = pd.Series(appearances) / n_bootstrap
    n_stable = int((stability >= 0.8).sum())
    logger.info(f"  ✅ Stable features (≥80% of bootstraps): {n_stable} / {len(stability)}")
    return stability


def build_full_ranking(stat_df, ml_df, feature_type, stability_scores=None):
    logger.info(f"  Building combined ranking for {feature_type}s...")

    ml_terms_in_stat = set(stat_df['term']) & set(ml_df['term'])
    if len(ml_terms_in_stat) < len(ml_df):
        logger.warning(f"  ⚠️ {len(ml_df) - len(ml_terms_in_stat)} ML terms not found in stat ranking")

    ml_cols_to_pull = ['term', 'lr_importance', 'rf_importance', 'gb_importance',
                       'mi_importance', 'ml_importance', 'ml_rank',
                       'lr_cv_auroc', 'rf_cv_auroc', 'gb_cv_auroc']
    ml_cols_present = [c for c in ml_cols_to_pull if c in ml_df.columns]
    combined = stat_df.merge(ml_df[ml_cols_present], on='term', how='left')

    ml_fill_cols = [c for c in ml_cols_present if c != 'term']
    for col in ml_fill_cols:
        combined[col] = combined[col].fillna(0)

    combined['stat_score'] = normalise(combined['log_odds_ratio'].values)
    combined['chi2_score'] = normalise(combined['chi_squared'].values)
    # `mi_score` is the mutual-information contribution (already normalised at ml step).
    combined['mi_score'] = combined['mi_importance'] if 'mi_importance' in combined.columns else 0.0

    if stability_scores is not None:
        combined['stability_score'] = combined['term'].map(stability_scores).fillna(0.0)
    else:
        combined['stability_score'] = np.nan

    w = CONFIG['combined_weights']
    combined['combined_score'] = (
        w.get('stat_score',    0) * combined['stat_score'] +
        w.get('chi2_score',    0) * combined['chi2_score'] +
        w.get('mi_score',      0) * combined['mi_score'] +
        w.get('ml_importance', 0) * combined['ml_importance']
    )

    combined = combined.sort_values('combined_score', ascending=False).reset_index(drop=True)
    combined['combined_rank'] = range(1, len(combined) + 1)
    combined['confidence_tier'] = pd.cut(
        combined['combined_score'],
        bins=CONFIG['confidence_bins'],
        labels=CONFIG['confidence_labels']
    )
    combined['feature_type'] = feature_type

    top_n = CONFIG['top_n_output']
    combined['stat_top150'] = combined['stat_rank'] <= top_n
    combined['ml_top150'] = combined['ml_rank'].apply(lambda x: 0 < x <= top_n)
    combined['both_agree'] = combined['stat_top150'] & combined['ml_top150']

    n_stable = int((combined['stability_score'] >= 0.8).sum()) if combined['stability_score'].notna().any() else 0
    logger.info(f"  Total: {len(combined):,} | Both agree (top {top_n}): {combined['both_agree'].sum()} | "
                f"High-stability (≥80%): {n_stable}")

    return combined


def print_top_n(df, n):
    header = (f"  {'Rank':<6} {'Type':<12} {'Term':<42} {'OR':>8} {'CI_Lo':>7} "
              f"{'CI_Hi':>7} {'p-val':>10} {'StatR':>6} {'MLR':>6} {'Score':>7} "
              f"{'Tier':<10} {'Agree':<5}")
    logger.info(header)
    logger.info("  " + "-" * 130)
    for _, r in df.head(n).iterrows():
        agree = "✅" if r.get('both_agree', False) else ""
        logger.info(
            f"  {int(r['combined_rank']):<6} "
            f"{str(r.get('feature_type', '')):<12} "
            f"{str(r['term'])[:41]:<42} "
            f"{r['odds_ratio']:>8.2f} "
            f"{r.get('or_ci_lower', 0):>7.2f} "
            f"{r.get('or_ci_upper', 0):>7.2f} "
            f"{r['p_value']:>10.2e} "
            f"{int(r['stat_rank']):>6} "
            f"{int(r['ml_rank']):>6} "
            f"{r['combined_score']:>7.3f} "
            f"{str(r['confidence_tier']):<10} "
            f"{agree:<5}"
        )


def main():
    t_total = time.time()
    cancer = CONFIG['cancer_type'].upper()

    logger.info("=" * 80)
    logger.info(f"PRODUCTION — {cancer} — Phase 1.3: Feature Ranking FINAL v1.1")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: seed={CONFIG['random_seed']}, cv={CONFIG['cv_folds']}, "
                f"max_features={CONFIG['max_ml_features']}, neg:pos={CONFIG['neg_pos_ratio']}:1")
    logger.info("=" * 80)

    logger.info("\n📂 Loading input files...")
    data_dir = CONFIG['data_dir']
    files = CONFIG['input_files']

    def _data_path(name):
        return os.path.join(data_dir, files[name])

    try:
        obs_code = CONFIG['code_columns']['observation']
        med_code = CONFIG['code_columns']['medication']
        pos_obs  = _read_csv_safe(_data_path('pos_obs'), obs_code)
        neg_obs  = _read_csv_safe(_data_path('neg_obs'), obs_code)
        pos_meds = _read_csv_safe(_data_path('pos_meds'), med_code)
        neg_meds = _read_csv_safe(_data_path('neg_meds'), med_code)
    except FileNotFoundError as e:
        logger.error(f"❌ Missing file: {e}")
        logger.error(f"  Expected in: {os.path.abspath(data_dir)}")
        sys.exit(1)

    validate_csv(pos_obs,  ['term', obs_code, 'n_patient_count', 'n_patient_count_total'], 'pos_obs')
    validate_csv(neg_obs,  ['term', obs_code, 'n_patient_count', 'n_patient_count_total'], 'neg_obs')
    validate_csv(pos_meds, ['term', med_code, 'n_patient_count', 'n_patient_count_total'], 'pos_meds')
    validate_csv(neg_meds, ['term', med_code, 'n_patient_count', 'n_patient_count_total'], 'neg_meds')

    # PART 1: OBSERVATIONS
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: OBSERVATIONS")
    logger.info("=" * 80)

    obs_stat, obs_total_pos, obs_total_neg = compute_statistical_ranking(pos_obs, neg_obs, 'observation')
    obs_ml = compute_ml_ranking(obs_stat, obs_total_pos, obs_total_neg, 'observation')
    obs_stability = compute_stability_scores(obs_stat, obs_total_pos, obs_total_neg, 'observation')
    obs_full = build_full_ranking(obs_stat, obs_ml, 'observation', stability_scores=obs_stability)

    logger.info(f"\n📋 Top 20 Observations:")
    print_top_n(obs_full, 20)

    save_csv(obs_full, f'{CONFIG["cancer_type"]}_obs_all.csv')
    save_csv(obs_full, f'{CONFIG["cancer_type"]}_obs_top150.csv', top_n=CONFIG['top_n_output'])

    # PART 2: MEDICATIONS
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: MEDICATIONS")
    logger.info("=" * 80)

    meds_stat, meds_total_pos, meds_total_neg = compute_statistical_ranking(pos_meds, neg_meds, 'medication')
    meds_ml = compute_ml_ranking(meds_stat, meds_total_pos, meds_total_neg, 'medication')
    meds_stability = compute_stability_scores(meds_stat, meds_total_pos, meds_total_neg, 'medication')
    meds_full = build_full_ranking(meds_stat, meds_ml, 'medication', stability_scores=meds_stability)

    logger.info(f"\n📋 Top 20 Medications:")
    print_top_n(meds_full, 20)

    save_csv(meds_full, f'{CONFIG["cancer_type"]}_meds_all.csv')
    save_csv(meds_full, f'{CONFIG["cancer_type"]}_meds_top150.csv', top_n=CONFIG['top_n_output'])

    # PART 3: COMBINED
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: COMBINED (Observations + Medications)")
    logger.info("=" * 80)

    all_features = pd.concat([obs_full, meds_full], ignore_index=True)

    all_features['stat_score'] = normalise(all_features['log_odds_ratio'].values)
    all_features['chi2_score'] = normalise(all_features['chi_squared'].values)
    all_features['mi_score'] = normalise(all_features.get('mi_importance', pd.Series(0, index=all_features.index)).values)
    all_features['ml_importance_combined'] = normalise(all_features['ml_importance'].values)

    w = CONFIG['combined_weights']
    all_features['combined_score'] = (
        w.get('stat_score',    0) * all_features['stat_score'] +
        w.get('chi2_score',    0) * all_features['chi2_score'] +
        w.get('mi_score',      0) * all_features['mi_score'] +
        w.get('ml_importance', 0) * all_features['ml_importance_combined']
    )

    all_features = all_features.sort_values('combined_score', ascending=False).reset_index(drop=True)
    all_features['combined_rank'] = range(1, len(all_features) + 1)
    all_features['confidence_tier'] = pd.cut(
        all_features['combined_score'],
        bins=CONFIG['confidence_bins'],
        labels=CONFIG['confidence_labels']
    )

    logger.info(f"\n📋 Top 20 Combined Features:")
    print_top_n(all_features, 20)

    save_csv(all_features, f'{CONFIG["cancer_type"]}_combined_all.csv')
    save_csv(all_features, f'{CONFIG["cancer_type"]}_combined_top150.csv', top_n=CONFIG['top_n_output'])

    # SUMMARY
    combined_top150 = all_features.head(CONFIG['top_n_output'])
    n_obs  = (combined_top150['feature_type'] == 'observation').sum()
    n_meds = (combined_top150['feature_type'] == 'medication').sum()

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    logger.info(f"""
📂 Output Files ({CONFIG['output_dir']}/):
  ┌──────────────────────────────────────────────────────────────┐
  │ SEPARATE:                                                    │
  │   {CONFIG['cancer_type']}_obs_top150.csv       → {min(len(obs_full), 150):>4} observations          │
  │   {CONFIG['cancer_type']}_meds_top150.csv      → {min(len(meds_full), 150):>4} medications           │
  │                                                              │
  │ COMBINED:                                                    │
  │   {CONFIG['cancer_type']}_combined_top150.csv  → {n_obs:>4} obs + {n_meds:>4} meds          │
  │                                                              │
  │ FULL REFERENCE:                                              │
  │   {CONFIG['cancer_type']}_obs_all.csv          → {len(obs_full):>6} observations       │
  │   {CONFIG['cancer_type']}_meds_all.csv         → {len(meds_full):>6} medications        │
  │   {CONFIG['cancer_type']}_combined_all.csv     → {len(all_features):>6} total              │
  └──────────────────────────────────────────────────────────────┘

📊 Observations:  {len(obs_full):,} scored | OR>2: {(obs_full['odds_ratio']>2).sum()} | Sig: {(obs_full['p_value']<0.05).sum()}
📊 Medications:   {len(meds_full):,} scored | OR>2: {(meds_full['odds_ratio']>2).sum()} | Sig: {(meds_full['p_value']<0.05).sum()}

📊 Combined Top {CONFIG['top_n_output']}: {n_obs} obs ({n_obs/max(n_obs+n_meds,1)*100:.0f}%) + {n_meds} meds ({n_meds/max(n_obs+n_meds,1)*100:.0f}%)

📊 Confidence: VeryHigh={(combined_top150['confidence_tier']=='Very High').sum()} | High={(combined_top150['confidence_tier']=='High').sum()} | Medium={(combined_top150['confidence_tier']=='Medium').sum()} | Low={(combined_top150['confidence_tier']=='Low').sum()}""")

    logger.info(f"\n🏆 Top 5 Observations (combined ranking):")
    for _, r in combined_top150[combined_top150['feature_type'] == 'observation'].head(5).iterrows():
        logger.info(f"  #{int(r['combined_rank'])} {r['term']} "
                    f"(OR={r['odds_ratio']:.2f} [{r.get('or_ci_lower',0):.2f}-{r.get('or_ci_upper',0):.2f}], "
                    f"p={r['p_value']:.2e}, score={r['combined_score']:.3f})")

    logger.info(f"\n🏆 Top 5 Medications (combined ranking):")
    for _, r in combined_top150[combined_top150['feature_type'] == 'medication'].head(5).iterrows():
        logger.info(f"  #{int(r['combined_rank'])} {r['term']} "
                    f"(OR={r['odds_ratio']:.2f} [{r.get('or_ci_lower',0):.2f}-{r.get('or_ci_upper',0):.2f}], "
                    f"p={r['p_value']:.2e}, score={r['combined_score']:.3f})")

    total_time = time.time() - t_total
    logger.info(f"\n⏱️  Total execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("\n" + "=" * 80)
    logger.info("✅ PHASE 1.3 COMPLETE — Ready for Clinical Review (Phase 1.4)")
    logger.info("=" * 80)

    return all_features, obs_full, meds_full


if __name__ == '__main__':
    try:
        all_features, obs_full, meds_full = main()
    except Exception as e:
        logger.error(f"\n❌ FATAL: {e}", exc_info=True)
        sys.exit(1)