"""
NOTE: Phase-1 scoring is SKIPPED when CATEGORIZED=True (run_v3 reads the category map directly). This runs
only for the per-SNOMED path (V3_snomed); its output is unused by the categorized pipeline.

Lung Cancer - Phase 1.3: Combined Feature Ranking (Statistical + ML)
Version 1.1 (final)

Ranks candidate features (observations + medications) by combining a statistical
signal (odds ratio + chi-squared) with an ML feature-importance signal (LogReg +
RandomForest + GradientBoosting). The three signals are each rank-normalised to [0,1]
and weighted (see CONFIG['combined_weights']).

ML signal: learned on the REAL patient-level matrices (Lung_patient_*_matrix.csv, produced
by build_score_counts.py from the same cohort) - genuine per-patient co-occurrence. These
are REQUIRED: there is NO synthetic fallback (we never fabricate patient data). If a matrix
is missing, the scorer stops and tells you to run build_score_counts.py first.

Horizons: scores BOTH `12mo` and `1mo` in a single run. Each reads Data/{horizon}/
and writes output/{horizon}/Scores_lung/.

Inputs (from ./Data/{horizon}/):
    Lung_positive_obs.csv,  Lung_negative_obs.csv          (required - statistical counts)
    Lung_positive_meds.csv, Lung_negative_meds.csv         (required)
    Lung_patient_obs_matrix.csv,  Lung_patient_meds_matrix.csv   (required - real-data ML)

Outputs (to output/{horizon}/Scores_lung/):
    lung_obs_all.csv,  lung_meds_all.csv,  lung_combined_all.csv   <- deliverables
    feature_ranking_YYYYMMDD_HHMMSS.log

Run:  python "Combined Scoring (ML+Stat).py"      # scores both horizons
Estimated time: 3-10 minutes per horizon.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2 as chi2_dist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import logging
import sys
import os
import time
from datetime import datetime


# ------------------------------------------------------------------------
# CONFIGURATION - All tuneable parameters in one place
# ------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))   # V3 root, for the central config default
try:
    import config as _C
    _YEARS_DEFAULT = getattr(_C, "SCORING_YEARS_BEFORE", 5)
except ModuleNotFoundError:
    _C = None
    _YEARS_DEFAULT = 5
# e.g. `python "Combined Scoring (ML+Stat).py" 12mo 5` (horizon [years]); hard-errors on an unrecognized token.
HORIZONS = (_C.horizons_from_argv(sys.argv[1:]) if _C is not None
            else ([a for a in sys.argv[1:] if a in ("12mo", "1mo")] or ["12mo", "1mo"]))
# FE/scoring lookback in years -> window-tagged Data/ + output/ dirs (so each lookback is self-contained).
# Default = central config SCORING_YEARS_BEFORE (NOT a hardcoded 5) so this can't disagree with build_codelist.
YEARS = next((int(a) for a in sys.argv[1:] if a.isdigit()), _YEARS_DEFAULT)

# Tuning knobs are sourced from the central config.py (single source of truth; logged in the run
# manifest via config.summary()). getattr(_C, ...) falls back to the literal only if config didn't
# import (true standalone use) — _C is None in that case, so getattr returns the default.
CONFIG = {
    # General
    'cancer_type': 'lung',
    'random_seed': getattr(_C, "RANDOM_STATE", 42),
    'top_n_output': getattr(_C, "SCORING_TOP_N", 150),
    'min_patients_per_feature': getattr(_C, "MIN_PATIENTS_PER_FEATURE", 3),

    # ML optimization
    'max_ml_features': getattr(_C, "SCORING_MAX_ML_FEATURES", 500),
    'cv_folds': getattr(_C, "SCORING_CV_FOLDS", 3),
    'neg_pos_ratio': getattr(_C, "NEG_POS_RATIO", 5),

    # Scoring weights
    'combined_weights': getattr(_C, "SCORING_COMBINED_WEIGHTS",
                                {'stat_score': 0.30, 'chi2_score': 0.20, 'ml_importance': 0.50}),
    'ml_weights': getattr(_C, "SCORING_ML_WEIGHTS",
                          {'lr': 0.25, 'rf': 0.35, 'gb': 0.40}),

    # Input (files live in Data/<horizon>/) — set per-horizon by set_horizon()
    'data_dir': None,
    'input_files': {
        'pos_obs':         'Lung_positive_obs.csv',
        'neg_obs':         'Lung_negative_obs.csv',
        'pos_meds':        'Lung_positive_meds.csv',
        'neg_meds':        'Lung_negative_meds.csv',
        'obs_matrix':      'Lung_patient_obs_matrix.csv',
        'meds_matrix':     'Lung_patient_meds_matrix.csv',
    },

    # Output (output/<horizon>/Scores_lung/) — set per-horizon by set_horizon()
    'output_dir': None,
    'confidence_bins':   [-0.01, 0.2, 0.5, 0.8, 1.01],
    'confidence_labels': ['Low', 'Medium', 'High', 'Very High'],

    # Small cohort safety
    'min_positive_cohort_size': 50,
}


# ------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------

_LOG_FMT = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S')


def setup_logging():
    """Console logger, created once. Per-horizon log FILES are attached by set_horizon()."""
    lg = logging.getLogger("scoring")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(_LOG_FMT)
        lg.addHandler(sh)
    return lg


logger = setup_logging()


def set_horizon(horizon, years=5):
    """Point CONFIG at this horizon+lookback's Data/ + output/ dirs and attach a fresh log file."""
    CONFIG['data_dir'] = os.path.join(_SCRIPT_DIR, 'Data', horizon, f'{years}yr')
    CONFIG['output_dir'] = os.path.join(_SCRIPT_DIR, 'output', horizon, f'{years}yr', 'Scores_lung')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    for fh in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        logger.removeHandler(fh); fh.close()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(CONFIG['output_dir'], f'feature_ranking_{ts}.log'))
    fh.setFormatter(_LOG_FMT)
    logger.addHandler(fh)


# ------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------

def normalise(arr):
    arr = np.array(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def rank01(arr):
    """Percentile-rank to [0, 1]. Robust to outliers (unlike min-max), so each score
    component contributes on an equal footing instead of being compressed toward 0 by a
    few extreme values. Preserves order/direction (e.g. negative log-OR still ranks low)."""
    s = pd.Series(np.array(arr, dtype=float)).fillna(0.0)
    if s.max() == s.min():
        return np.zeros(len(s))
    return s.rank(pct=True, method="average").values


def validate_csv(df, required_cols, name):
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Found: {list(df.columns)}")
    if len(df) == 0:
        raise ValueError(f"[{name}] Empty DataFrame!")
    logger.info(f"  [OK] {name}: {len(df):,} rows validated")
    return df


def _format_csv_no_trailing_zeros(df):
    """Convert whole-number floats to int so CSV writes 74 not 74.0."""
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
    logger.info(f"  Saved: {filepath} ({len(output):,} rows)")
    return output


def _detect_merge_key(pos_df, neg_df, feature_type):
    """Detect the best merge key - prefer code ID over term."""
    if feature_type == 'observation':
        if 'snomed_id' in pos_df.columns and 'snomed_id' in neg_df.columns:
            return 'snomed_id'
    elif feature_type == 'medication':
        if 'med_code_id' in pos_df.columns and 'med_code_id' in neg_df.columns:
            return 'med_code_id'
    logger.warning(f"  [WARN] No code ID column for {feature_type} - falling back to term merge")
    return 'term'


# ------------------------------------------------------------------------
# STEP 1: STATISTICAL RANKING
# ------------------------------------------------------------------------

def compute_statistical_ranking(pos_df, neg_df, feature_type='observation'):
    logger.info(f"\n{'='*60}")
    logger.info(f"STATISTICAL RANKING: {feature_type}s")
    logger.info(f"{'='*60}")

    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    pos_df.columns = pos_df.columns.str.strip().str.lower()
    neg_df.columns = neg_df.columns.str.strip().str.lower()

    total_pos = int(pos_df['n_patient_count_total'].iloc[0])
    total_neg = int(neg_df['n_patient_count_total'].iloc[0])

    logger.info(f"  Positive: {total_pos:,} patients, {len(pos_df):,} terms")
    logger.info(f"  Negative: {total_neg:,} patients, {len(neg_df):,} terms")

    if total_pos == 0 or total_neg == 0:
        raise ValueError(f"Zero patients! pos={total_pos}, neg={total_neg}")

    if total_pos < CONFIG['min_positive_cohort_size']:
        logger.warning(f"  [WARN] SMALL POSITIVE COHORT: {total_pos} patients. "
                       f"Results may be unstable.")

    # Detect merge key
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

    # Resolve code ID for output
    if feature_type == 'observation':
        if 'snomed_id_pos' in merged.columns:
            merged['code_id'] = merged['snomed_id_pos'].fillna(merged.get('snomed_id_neg', np.nan))
        elif 'snomed_id' in merged.columns:
            merged['code_id'] = merged['snomed_id']
        else:
            merged['code_id'] = np.nan
        merged['code_type'] = 'SNOMED'
    elif feature_type == 'medication':
        if 'med_code_id_pos' in merged.columns:
            merged['code_id'] = merged['med_code_id_pos'].fillna(merged.get('med_code_id_neg', np.nan))
        elif 'med_code_id' in merged.columns:
            merged['code_id'] = merged['med_code_id']
        else:
            merged['code_id'] = np.nan
        merged['code_type'] = 'DMD'
    else:
        merged['code_id'] = np.nan
        merged['code_type'] = 'UNKNOWN'

    # Prevalence
    merged['prevalence_pos'] = merged['n_patient_count_pos'] / total_pos
    merged['prevalence_neg'] = merged['n_patient_count_neg'] / total_neg
    merged['prevalence_diff'] = merged['prevalence_pos'] - merged['prevalence_neg']
    merged['prevalence_ratio'] = merged['prevalence_pos'] / merged['prevalence_neg'].replace(0, np.nan)

    # Odds Ratio (Haldane 0.5 correction)
    a = merged['n_patient_count_pos'] + 0.5
    b = (total_pos - merged['n_patient_count_pos']) + 0.5
    c = merged['n_patient_count_neg'] + 0.5
    d = (total_neg - merged['n_patient_count_neg']) + 0.5
    merged['odds_ratio'] = (a * d) / (b * c)
    merged['log_odds_ratio'] = np.log(merged['odds_ratio'])

    # 95% CI
    merged['log_or_se'] = np.sqrt(1/a + 1/b + 1/c + 1/d)
    merged['or_ci_lower'] = np.exp(merged['log_odds_ratio'] - 1.96 * merged['log_or_se'])
    merged['or_ci_upper'] = np.exp(merged['log_odds_ratio'] + 1.96 * merged['log_or_se'])

    # Chi-Squared (Yates-corrected, 2x2) — VECTORIZED, exactly matches scipy.chi2_contingency(correction=True)
    # per feature but without the iterrows() loop. For a 2x2 table [[a,b],[c,d]] with row margins
    # r1=total_pos, r2=total_neg and column margins c1=a+c, c2=b+d, the continuity-corrected statistic is
    #   chi2 = N * max(|ad - bc| - N/2, 0)^2 / (r1 r2 c1 c2),   p = chi2_dist.sf(chi2, df=1)
    # (the max(.,0) is scipy's per-cell |O-E| clamp for 2x2). A zero column margin -> undefined -> NaN
    # (the same outcome as the old chi2_contingency ValueError -> except -> NaN).
    a = merged['n_patient_count_pos'].to_numpy(dtype=float)
    b = (total_pos - merged['n_patient_count_pos']).to_numpy(dtype=float)
    c = merged['n_patient_count_neg'].to_numpy(dtype=float)
    d = (total_neg - merged['n_patient_count_neg']).to_numpy(dtype=float)
    N = float(total_pos + total_neg)
    c1, c2 = a + c, b + d                                    # column margins (rows are total_pos / total_neg)
    den = float(total_pos) * float(total_neg) * c1 * c2
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_arr = N * np.maximum(np.abs(a * d - b * c) - N / 2.0, 0.0) ** 2 / den
    chi2_arr = np.where((c1 == 0) | (c2 == 0), np.nan, chi2_arr)     # zero column margin -> undefined (was except->NaN)
    merged['chi_squared'] = np.round(chi2_arr, 6)
    merged['p_value'] = chi2_dist.sf(chi2_arr, 1)                    # NaN propagates to NaN
    merged['p_value_bonferroni'] = np.minimum(merged['p_value'] * len(merged), 1.0)
    merged['feature_type'] = feature_type

    # Filter
    merged = merged[merged['n_patient_count_pos'] >= CONFIG['min_patients_per_feature']].copy()
    merged = merged.sort_values('odds_ratio', ascending=False).reset_index(drop=True)
    merged['stat_rank'] = range(1, len(merged) + 1)

    logger.info(f"  After filter (>={CONFIG['min_patients_per_feature']}): {len(merged):,} terms")
    logger.info(f"  OR > 2.0: {(merged['odds_ratio'] > 2.0).sum()}")
    logger.info(f"  OR > 5.0: {(merged['odds_ratio'] > 5.0).sum()}")
    logger.info(f"  Significant (p<0.05): {(merged['p_value'] < 0.05).sum()}")
    logger.info(f"  Bonferroni sig (p<0.05): {(merged['p_value_bonferroni'] < 0.05).sum()}")

    return merged, total_pos, total_neg


# ------------------------------------------------------------------------
# STEP 2: ML RANKING - real patient data (required; no synthetic fallback)
# ------------------------------------------------------------------------

def compute_ml_ranking_from_patient_matrix(patient_matrix_df, feature_list, feature_type='observation'):
    logger.info(f"\n{'='*60}")
    logger.info(f"ML RANKING (real patient data): {feature_type}s")
    logger.info(f"{'='*60}")

    t_start = time.time()
    np.random.seed(CONFIG['random_seed'])

    # Dedup feature_list while preserving order. Multiple SNOMED/DMD codes can share the same
    # term text, so the stat-ranking term list passed in here can contain duplicates. Without
    # dedup, reindex(columns=feature_list) produces duplicate column names, and downstream
    # data[feature_list] explodes (each duplicated name returns every matching column),
    # leaving X wider than feature_list and breaking the final DataFrame assembly.
    n_before = len(feature_list)
    feature_list = list(dict.fromkeys(feature_list))
    if len(feature_list) < n_before:
        logger.info(f"  Deduped feature_list: {n_before} -> {len(feature_list)} unique terms")

    pm = patient_matrix_df.copy()
    pm = pm[pm['feature_name'].isin(feature_list)]

    if len(pm) == 0:
        logger.warning(f"  [WARN] No matching features for {feature_type}!")
        return _empty_ml_scores(feature_list)

    # Downsample negative cohort
    pos_patients = pm[pm['label'] == 1]['patient_guid'].unique()
    neg_patients = pm[pm['label'] == 0]['patient_guid'].unique()
    n_pos = len(pos_patients)
    n_neg_target = min(len(neg_patients), n_pos * CONFIG['neg_pos_ratio'])

    if n_pos < 20:
        logger.warning(f"  [WARN] Only {n_pos} positive patients - ML rankings may be unreliable")

    sampled_neg = np.random.choice(neg_patients, size=n_neg_target, replace=False)
    keep_patients = set(pos_patients) | set(sampled_neg)
    pm = pm[pm['patient_guid'].isin(keep_patients)]

    logger.info(f"  Cohort: {n_pos:,} pos + {n_neg_target:,} neg ({CONFIG['neg_pos_ratio']}:1)")

    # Pivot to a wide patient x feature binary matrix
    logger.info("  Pivoting to wide format...")
    patient_info = pm.groupby('patient_guid').agg(label=('label', 'first')).reset_index()

    feature_pivot = pm.pivot_table(
        index='patient_guid', columns='feature_name',
        values='event_count', aggfunc='sum', fill_value=0
    )
    feature_binary = (feature_pivot > 0).astype(int)
    feature_binary = feature_binary.reindex(columns=feature_list, fill_value=0)

    data = patient_info.set_index('patient_guid').join(feature_binary).dropna(subset=['label'])
    X = data[feature_list].values
    y = data['label'].astype(int).values

    logger.info(f"  Matrix: {X.shape[0]:,} x {X.shape[1]:,}")
    logger.info(f"  Balance: {y.sum():,} pos ({y.sum()/len(y)*100:.1f}%), "
                f"{(1-y).sum():,} neg ({(1-y).sum()/len(y)*100:.1f}%)")

    return _train_ml_models(X, y, feature_list, feature_type, t_start)


# ------------------------------------------------------------------------
# SHARED ML TRAINING
# ------------------------------------------------------------------------

def _train_ml_models(X, y, feature_list, feature_type, t_start):
    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_seed'])

    # Compute sample weights for GB
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
        penalty='l1',
        solver='saga',
        C=1.0,
        max_iter=50000,
        tol=1e-3,
        random_state=CONFIG['random_seed'],
        class_weight='balanced'
    )
    # CV score with the scaler fit INSIDE each fold (pipeline) so the validation fold's statistics never
    # leak into the scaling — the previous `cross_val_score(lr, X_scaled, ...)` scaled on ALL rows first.
    lr_cv = cross_val_score(make_pipeline(StandardScaler(), clone(lr)), X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    lr.fit(X_scaled, y)                 # final fit on ALL data is for feature importance only (no held-out estimate here)
    lr_imp = np.abs(lr.coef_[0])
    logger.info(f"    AUROC: {lr_cv.mean():.4f} +/- {lr_cv.std():.4f} | "
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
    logger.info(f"    AUROC: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f} | {time.time()-t0:.1f}s")

    # * Gradient Boosting - manual CV with sample_weight
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
        gb_fold.fit(X[train_idx], y[train_idx],
                    sample_weight=sample_weights[train_idx])
        y_pred = gb_fold.predict_proba(X[val_idx])[:, 1]
        gb_cv_scores.append(roc_auc_score(y[val_idx], y_pred))
    gb_cv = np.array(gb_cv_scores)

    gb.fit(X, y, sample_weight=sample_weights)
    gb_imp = gb.feature_importances_
    logger.info(f"    AUROC: {gb_cv.mean():.4f} +/- {gb_cv.std():.4f} | {time.time()-t0:.1f}s")

    # Aggregate
    w = CONFIG['ml_weights']
    ml_scores = pd.DataFrame({
        'term': feature_list,
        'lr_importance': normalise(lr_imp),
        'rf_importance': normalise(rf_imp),
        'gb_importance': normalise(gb_imp),
        'lr_cv_auroc': lr_cv.mean(),
        'rf_cv_auroc': rf_cv.mean(),
        'gb_cv_auroc': gb_cv.mean(),
    })
    ml_scores['ml_importance'] = (
        w['lr'] * ml_scores['lr_importance'] +
        w['rf'] * ml_scores['rf_importance'] +
        w['gb'] * ml_scores['gb_importance']
    )
    ml_scores = ml_scores.sort_values('ml_importance', ascending=False).reset_index(drop=True)
    ml_scores['ml_rank'] = range(1, len(ml_scores) + 1)

    logger.info(f"  [OK] Complete: {time.time()-t_start:.1f}s total")
    return ml_scores


def _empty_ml_scores(feature_list):
    return pd.DataFrame({
        'term': feature_list, 'lr_importance': 0, 'rf_importance': 0,
        'gb_importance': 0, 'ml_importance': 0, 'ml_rank': 999,
        'lr_cv_auroc': 0, 'rf_cv_auroc': 0, 'gb_cv_auroc': 0
    })


# ------------------------------------------------------------------------
# STEP 3: BUILD COMBINED RANKING
# ------------------------------------------------------------------------

def build_full_ranking(stat_df, ml_df, feature_type):
    logger.info(f"  Building combined ranking for {feature_type}s...")

    # * Safety check: how many ML terms match stat terms
    ml_terms_in_stat = set(stat_df['term']) & set(ml_df['term'])
    if len(ml_terms_in_stat) < len(ml_df):
        logger.warning(f"  [WARN] {len(ml_df) - len(ml_terms_in_stat)} ML terms not found in stat ranking")

    combined = stat_df.merge(
        ml_df[['term', 'lr_importance', 'rf_importance', 'gb_importance',
               'ml_importance', 'ml_rank', 'lr_cv_auroc', 'rf_cv_auroc', 'gb_cv_auroc']],
        on='term', how='left'
    )

    ml_cols = ['lr_importance', 'rf_importance', 'gb_importance', 'ml_importance',
               'ml_rank', 'lr_cv_auroc', 'rf_cv_auroc', 'gb_cv_auroc']
    for col in ml_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    combined['stat_score'] = rank01(combined['log_odds_ratio'].values)
    combined['chi2_score'] = rank01(combined['chi_squared'].values)
    combined['ml_score']   = rank01(combined['ml_importance'].values)

    w = CONFIG['combined_weights']
    combined['combined_score'] = (
        w['stat_score']    * combined['stat_score'] +
        w['chi2_score']    * combined['chi2_score'] +
        w['ml_importance'] * combined['ml_score']
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

    logger.info(f"  Total: {len(combined):,} | Both agree (top {top_n}): {combined['both_agree'].sum()}")

    return combined


# ------------------------------------------------------------------------
# STEP 4: DISPLAY
# ------------------------------------------------------------------------

def print_top_n(df, n):
    header = (f"  {'Rank':<6} {'Type':<12} {'Term':<42} {'OR':>8} {'CI_Lo':>7} "
              f"{'CI_Hi':>7} {'p-val':>10} {'StatR':>6} {'MLR':>6} {'Score':>7} "
              f"{'Tier':<10} {'Agree':<5}")
    logger.info(header)
    logger.info("  " + "-" * 130)
    for _, r in df.head(n).iterrows():
        agree = "[OK]" if r.get('both_agree', False) else ""
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


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main():
    t_total = time.time()
    cancer = CONFIG['cancer_type'].upper()

    logger.info("=" * 80)
    logger.info(f"PRODUCTION - {cancer} CANCER - Phase 1.3: Feature Ranking FINAL v1.1")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: seed={CONFIG['random_seed']}, cv={CONFIG['cv_folds']}, "
                f"max_features={CONFIG['max_ml_features']}, neg:pos={CONFIG['neg_pos_ratio']}:1")
    logger.info("=" * 80)

    #  Load CSVs
    logger.info("\nLoading input files...")
    data_dir = CONFIG['data_dir']
    files = CONFIG['input_files']

    def _data_path(name):
        return os.path.join(data_dir, files[name])

    try:
        pos_obs  = pd.read_csv(_data_path('pos_obs'))
        neg_obs  = pd.read_csv(_data_path('neg_obs'))
        pos_meds = pd.read_csv(_data_path('pos_meds'))
        neg_meds = pd.read_csv(_data_path('neg_meds'))
    except FileNotFoundError as e:
        logger.error(f"[ERROR] Missing file: {e}")
        logger.error(f"  Expected in: {os.path.abspath(data_dir)}")
        sys.exit(1)

    validate_csv(pos_obs,  ['term', 'snomed_id', 'n_patient_count', 'n_patient_count_total'], 'pos_obs')
    validate_csv(neg_obs,  ['term', 'snomed_id', 'n_patient_count', 'n_patient_count_total'], 'neg_obs')
    validate_csv(pos_meds, ['term', 'med_code_id', 'n_patient_count', 'n_patient_count_total'], 'pos_meds')
    validate_csv(neg_meds, ['term', 'med_code_id', 'n_patient_count', 'n_patient_count_total'], 'neg_meds')

    # Patient matrices are REQUIRED - ML is learned on real patient data (no synthetic
    # fallback; we never fabricate patients). build_score_counts.py builds them from the cohort.
    for key in ('obs_matrix', 'meds_matrix'):
        if not os.path.exists(_data_path(key)):
            logger.error(f"[ERROR] Required patient matrix missing: {_data_path(key)}")
            logger.error("  Run build_score_counts.py first (it builds the matrices from the cohort).")
            sys.exit(1)

    obs_matrix = pd.read_csv(_data_path('obs_matrix'))
    obs_matrix.columns = obs_matrix.columns.str.strip().str.lower()
    meds_matrix = pd.read_csv(_data_path('meds_matrix'))
    meds_matrix.columns = meds_matrix.columns.str.strip().str.lower()
    logger.info(f"  [OK] Observation matrix: {obs_matrix['patient_guid'].nunique():,} patients, "
                f"{obs_matrix['feature_name'].nunique():,} features")
    logger.info(f"  [OK] Medication matrix:  {meds_matrix['patient_guid'].nunique():,} patients, "
                f"{meds_matrix['feature_name'].nunique():,} features")

    # ------------------------------------------------------------------------
    # PART 1: OBSERVATIONS
    # ------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: OBSERVATIONS")
    logger.info("=" * 80)

    obs_stat, _, _ = compute_statistical_ranking(pos_obs, neg_obs, 'observation')
    obs_features = list(dict.fromkeys(obs_stat['term'].tolist()))[:CONFIG['max_ml_features']]
    obs_ml = compute_ml_ranking_from_patient_matrix(obs_matrix, obs_features, 'observation')

    obs_full = build_full_ranking(obs_stat, obs_ml, 'observation')

    logger.info(f"\nTop 20 Observations:")
    print_top_n(obs_full, 20)

    save_csv(obs_full, f'{CONFIG["cancer_type"]}_obs_all.csv')

    # ------------------------------------------------------------------------
    # PART 2: MEDICATIONS
    # ------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: MEDICATIONS")
    logger.info("=" * 80)

    meds_stat, _, _ = compute_statistical_ranking(pos_meds, neg_meds, 'medication')
    meds_features = list(dict.fromkeys(meds_stat['term'].tolist()))[:CONFIG['max_ml_features']]
    meds_ml = compute_ml_ranking_from_patient_matrix(meds_matrix, meds_features, 'medication')

    meds_full = build_full_ranking(meds_stat, meds_ml, 'medication')

    logger.info(f"\nTop 20 Medications:")
    print_top_n(meds_full, 20)

    save_csv(meds_full, f'{CONFIG["cancer_type"]}_meds_all.csv')

    # ------------------------------------------------------------------------
    # PART 3: COMBINED
    # ------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: COMBINED (Observations + Medications)")
    logger.info("=" * 80)

    all_features = pd.concat([obs_full, meds_full], ignore_index=True)

    all_features['stat_score'] = rank01(all_features['log_odds_ratio'].values)
    all_features['chi2_score'] = rank01(all_features['chi_squared'].values)
    all_features['ml_importance_combined'] = rank01(all_features['ml_importance'].values)

    w = CONFIG['combined_weights']
    all_features['combined_score'] = (
        w['stat_score']    * all_features['stat_score'] +
        w['chi2_score']    * all_features['chi2_score'] +
        w['ml_importance'] * all_features['ml_importance_combined']
    )

    all_features = all_features.sort_values('combined_score', ascending=False).reset_index(drop=True)
    all_features['combined_rank'] = range(1, len(all_features) + 1)
    all_features['confidence_tier'] = pd.cut(
        all_features['combined_score'],
        bins=CONFIG['confidence_bins'],
        labels=CONFIG['confidence_labels']
    )

    logger.info(f"\nTop 20 Combined Features:")
    print_top_n(all_features, 20)

    save_csv(all_features, f'{CONFIG["cancer_type"]}_combined_all.csv')

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------
    n_obs  = (all_features['feature_type'] == 'observation').sum()
    n_meds = (all_features['feature_type'] == 'medication').sum()

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    logger.info(f"""
Output files ({CONFIG['output_dir']}/):
    {CONFIG['cancer_type']}_obs_all.csv       -> {len(obs_full):>6} observations
    {CONFIG['cancer_type']}_meds_all.csv      -> {len(meds_full):>6} medications
    {CONFIG['cancer_type']}_combined_all.csv  -> {len(all_features):>6} total ({n_obs} obs + {n_meds} meds)

ML importance: learned on the real patient matrices (per-patient co-occurrence; no synthetic fallback).
Scoring uses rank-based normalisation (stat / chi2 / ml each ranked to [0,1] before weighting).

Observations:  {len(obs_full):,} scored | OR>2: {(obs_full['odds_ratio']>2).sum()} | Sig (p<0.05): {(obs_full['p_value']<0.05).sum()}
Medications:   {len(meds_full):,} scored | OR>2: {(meds_full['odds_ratio']>2).sum()} | Sig (p<0.05): {(meds_full['p_value']<0.05).sum()}

Confidence (all features): VeryHigh={(all_features['confidence_tier']=='Very High').sum()} | High={(all_features['confidence_tier']=='High').sum()} | Medium={(all_features['confidence_tier']=='Medium').sum()} | Low={(all_features['confidence_tier']=='Low').sum()}""")

    logger.info(f"\nTop 5 Observations (combined ranking):")
    for _, r in all_features[all_features['feature_type'] == 'observation'].head(5).iterrows():
        logger.info(f"  #{int(r['combined_rank'])} {r['term']} "
                    f"(OR={r['odds_ratio']:.2f} [{r.get('or_ci_lower',0):.2f}-{r.get('or_ci_upper',0):.2f}], "
                    f"p={r['p_value']:.2e}, score={r['combined_score']:.3f})")

    logger.info(f"\nTop 5 Medications (combined ranking):")
    for _, r in all_features[all_features['feature_type'] == 'medication'].head(5).iterrows():
        logger.info(f"  #{int(r['combined_rank'])} {r['term']} "
                    f"(OR={r['odds_ratio']:.2f} [{r.get('or_ci_lower',0):.2f}-{r.get('or_ci_upper',0):.2f}], "
                    f"p={r['p_value']:.2e}, score={r['combined_score']:.3f})")

    total_time = time.time() - t_total
    logger.info(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("\n" + "=" * 80)
    logger.info("[OK] PHASE 1.3 COMPLETE - Ready for Clinical Review (Phase 1.4)")
    logger.info("=" * 80)

    return all_features, obs_full, meds_full


# ------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------

if __name__ == '__main__':
    for horizon in HORIZONS:
        set_horizon(horizon, YEARS)
        logger.info("#" * 80)
        logger.info(f"# HORIZON = {horizon} ({YEARS}yr)   "
                    f"(Data/{horizon}/{YEARS}yr/ -> output/{horizon}/{YEARS}yr/Scores_lung/)")
        logger.info("#" * 80)
        try:
            main()
        except Exception as e:
            logger.error(f"\n[ERROR] FATAL ({horizon}): {e}", exc_info=True)
            sys.exit(1)