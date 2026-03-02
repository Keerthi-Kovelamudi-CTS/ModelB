import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import logging
import sys
import os
import time
from datetime import datetime


# ════════════════════════════════════════════════════════════════
# CONFIGURATION — All tuneable parameters in one place
# ════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'Data')

CONFIG = {
    # General
    'random_seed': 42,
    'top_n_output': 150,
    'min_patients_per_feature': 5,

    # ML optimization
    'max_ml_features': 500,
    'cv_folds': 3,
    'neg_pos_ratio': 5,

    # Scoring weights
    'combined_weights': {
        'stat_score': 0.30,
        'chi2_score': 0.20,
        'ml_importance': 0.50
    },
    'ml_weights': {
        'lr': 0.25,
        'rf': 0.35,
        'gb': 0.40
    },

    # Input (files live in Data/)
    'data_dir': _DATA_DIR,
    'input_files': {
        'pos_obs':         'Bladder_positive_obs.csv',
        'neg_obs':         'Bladder_negative_obs.csv',
        'pos_meds':        'Bladder_positive_meds.csv',
        'neg_meds':        'Bladder_negative_meds.csv',
        'obs_matrix':      'Bladder_patient_obs_matrix.csv',
        'meds_matrix':     'Bladder_patient_meds_matrix.csv',
    },

    # Output (under Scores/output/Scores_2)
    'output_dir': os.path.join(_SCRIPT_DIR, 'output', 'Scores_2'),
    'confidence_bins':   [-0.01, 0.2, 0.5, 0.8, 1.01],
    'confidence_labels': ['Low', 'Medium', 'High', 'Very High']
}


# ════════════════════════════════════════════════════════════════
# LOGGING
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


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def normalise(arr):
    arr = np.array(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def validate_csv(df, required_cols, name):
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Found: {list(df.columns)}")
    if len(df) == 0:
        raise ValueError(f"[{name}] Empty DataFrame!")
    logger.info(f"  ✅ {name}: {len(df):,} rows validated")
    return df


def _format_csv_no_trailing_zeros(df):
    """Convert whole-number floats to int so CSV writes 74 not 74.0; leave decimals as-is."""
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


# Output column renames: internal names → display names in CSV
CSV_COLUMN_RENAMES = {'med_code_id_pos': 'DMD Code', 'snomed_id_pos': 'SNOMED_ID'}


def save_csv(df, filename, top_n=None):
    filepath = os.path.join(CONFIG['output_dir'], filename)
    output = df.head(top_n).copy() if top_n else df.copy()
    output = output.rename(columns={k: v for k, v in CSV_COLUMN_RENAMES.items() if k in output.columns})
    output = _format_csv_no_trailing_zeros(output)
    output.to_csv(filepath, index=False)
    logger.info(f"  💾 Saved: {filepath} ({len(output):,} rows)")
    return output


# ════════════════════════════════════════════════════════════════
# STEP 1: STATISTICAL RANKING
# ════════════════════════════════════════════════════════════════

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

    # Merge
    merged = pos_df.merge(neg_df, on='term', how='outer', suffixes=('_pos', '_neg'))
    merged['n_patient_count_pos'] = merged['n_patient_count_pos'].fillna(0).astype(int)
    merged['n_patient_count_neg'] = merged['n_patient_count_neg'].fillna(0).astype(int)

    # Resolve code ID
    if 'snomed_id_pos' in merged.columns:
        merged['code_id'] = merged['snomed_id_pos'].fillna(merged.get('snomed_id_neg', np.nan))
        merged['code_type'] = 'SNOMED'
    elif 'med_code_id_pos' in merged.columns:
        merged['code_id'] = merged['med_code_id_pos'].fillna(merged.get('med_code_id_neg', np.nan))
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

    # 95% CI for log(OR)
    merged['log_or_se'] = np.sqrt(1/a + 1/b + 1/c + 1/d)
    merged['or_ci_lower'] = np.exp(merged['log_odds_ratio'] - 1.96 * merged['log_or_se'])
    merged['or_ci_upper'] = np.exp(merged['log_odds_ratio'] + 1.96 * merged['log_or_se'])

    # Chi-Squared
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
        except:
            chi2_vals.append(np.nan)
            p_vals.append(np.nan)

    merged['chi_squared'] = chi2_vals
    merged['p_value'] = p_vals
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


# ════════════════════════════════════════════════════════════════
# STEP 2: ML RANKING — Real patient data (primary)
# ════════════════════════════════════════════════════════════════

def compute_ml_ranking_from_patient_matrix(patient_matrix_df, feature_list, feature_type='observation'):
    logger.info(f"\n{'='*60}")
    logger.info(f"ML RANKING (Real Patient Data): {feature_type}s")
    logger.info(f"{'='*60}")

    t_start = time.time()
    np.random.seed(CONFIG['random_seed'])

    pm = patient_matrix_df.copy()
    pm = pm[pm['feature_name'].isin(feature_list)]

    if len(pm) == 0:
        logger.warning(f"  ⚠️ No matching features for {feature_type}!")
        return _empty_ml_scores(feature_list)

    # Downsample negative cohort
    pos_patients = pm[pm['label'] == 1]['patient_guid'].unique()
    neg_patients = pm[pm['label'] == 0]['patient_guid'].unique()
    n_pos = len(pos_patients)
    n_neg_target = min(len(neg_patients), n_pos * CONFIG['neg_pos_ratio'])

    sampled_neg = np.random.choice(neg_patients, size=n_neg_target, replace=False)
    keep_patients = set(pos_patients) | set(sampled_neg)
    pm = pm[pm['patient_guid'].isin(keep_patients)]

    logger.info(f"  Cohort: {n_pos:,} pos + {n_neg_target:,} neg ({CONFIG['neg_pos_ratio']}:1)")

    # Pivot
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

    logger.info(f"  Matrix: {X.shape[0]:,} × {X.shape[1]:,}")
    logger.info(f"  Balance: {y.sum():,} pos ({y.sum()/len(y)*100:.1f}%), "
                f"{(1-y).sum():,} neg ({(1-y).sum()/len(y)*100:.1f}%)")

    return _train_ml_models(X, y, feature_list, feature_type, t_start)


# ════════════════════════════════════════════════════════════════
# STEP 2b: ML RANKING — Fallback (simulated from prevalence)
# ════════════════════════════════════════════════════════════════

def compute_ml_ranking_from_prevalence(stat_df, total_pos, total_neg, feature_type='observation'):
    logger.info(f"\n{'='*60}")
    logger.info(f"ML RANKING (FALLBACK — Simulated): {feature_type}s")
    logger.info(f"{'='*60}")
    logger.warning("  ⚠️ No patient matrix — using prevalence simulation (approximate)")

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


# ════════════════════════════════════════════════════════════════
# SHARED ML TRAINING (used by both real + fallback)
# ════════════════════════════════════════════════════════════════

def _train_ml_models(X, y, feature_list, feature_type, t_start):
    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_seed'])

    # L1 Logistic Regression
    t0 = time.time()
    logger.info("  Training L1 Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=1.0,
        max_iter=50000,          # increased from 10000 for convergence
        tol=1e-3,                # relaxed from default 1e-4
        random_state=CONFIG['random_seed'],
        class_weight='balanced'
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

    # Gradient Boosting
    t0 = time.time()
    logger.info("  Training Gradient Boosting (150 trees)...")
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        min_samples_leaf=30, random_state=CONFIG['random_seed'], subsample=0.7
    )
    gb_cv = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    gb.fit(X, y)
    gb_imp = gb.feature_importances_
    logger.info(f"    AUROC: {gb_cv.mean():.4f} ± {gb_cv.std():.4f} | {time.time()-t0:.1f}s")

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

    logger.info(f"  ✅ Complete: {time.time()-t_start:.1f}s total")
    return ml_scores


def _empty_ml_scores(feature_list):
    return pd.DataFrame({
        'term': feature_list, 'lr_importance': 0, 'rf_importance': 0,
        'gb_importance': 0, 'ml_importance': 0, 'ml_rank': 999,
        'lr_cv_auroc': 0, 'rf_cv_auroc': 0, 'gb_cv_auroc': 0
    })


# ════════════════════════════════════════════════════════════════
# STEP 3: BUILD COMBINED RANKING
# ════════════════════════════════════════════════════════════════

def build_full_ranking(stat_df, ml_df, feature_type):
    logger.info(f"  Building combined ranking for {feature_type}s...")

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

    combined['stat_score'] = normalise(combined['log_odds_ratio'].values)
    combined['chi2_score'] = normalise(combined['chi_squared'].values)

    w = CONFIG['combined_weights']
    combined['combined_score'] = (
        w['stat_score']    * combined['stat_score'] +
        w['chi2_score']    * combined['chi2_score'] +
        w['ml_importance'] * combined['ml_importance']
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


# ════════════════════════════════════════════════════════════════
# STEP 4: DISPLAY
# ════════════════════════════════════════════════════════════════

def print_top_n(df, n):
    header = f"  {'Rank':<6} {'Type':<12} {'Term':<42} {'OR':>8} {'CI_Lo':>7} {'CI_Hi':>7} {'p-val':>10} {'StatR':>6} {'MLR':>6} {'Score':>7} {'Tier':<10} {'Agree':<5}"
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


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    logger.info("=" * 80)
    logger.info("PRODUCTION — BLADDER CANCER — Phase 1.3: Feature Ranking FINAL v2")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: seed={CONFIG['random_seed']}, cv={CONFIG['cv_folds']}, "
                f"max_features={CONFIG['max_ml_features']}, neg:pos={CONFIG['neg_pos_ratio']}:1")
    logger.info("=" * 80)

    # ── Load CSVs from Data/ ──
    logger.info("\n📂 Loading input files...")
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
        logger.error(f"❌ Missing file: {e}")
        logger.error(f"  Expected input files in: {os.path.abspath(data_dir)}")
        sys.exit(1)

    validate_csv(pos_obs,  ['term', 'snomed_id', 'n_patient_count', 'n_patient_count_total'], 'pos_obs')
    validate_csv(neg_obs,  ['term', 'snomed_id', 'n_patient_count', 'n_patient_count_total'], 'neg_obs')
    validate_csv(pos_meds, ['term', 'med_code_id', 'n_patient_count', 'n_patient_count_total'], 'pos_meds')
    validate_csv(neg_meds, ['term', 'med_code_id', 'n_patient_count', 'n_patient_count_total'], 'neg_meds')

    # ── Load SEPARATE matrices for obs and meds ──
    has_obs_matrix = os.path.exists(_data_path('obs_matrix'))
    has_meds_matrix = os.path.exists(_data_path('meds_matrix'))

    obs_matrix = None
    meds_matrix = None

    if has_obs_matrix:
        logger.info(f"  ✅ Observation matrix found: {_data_path('obs_matrix')}")
        obs_matrix = pd.read_csv(_data_path('obs_matrix'))
        obs_matrix.columns = obs_matrix.columns.str.strip().str.lower()
        logger.info(f"     {obs_matrix['patient_guid'].nunique():,} patients, "
                    f"{obs_matrix['feature_name'].nunique():,} features")
    else:
        logger.warning(f"  ⚠️ Observation matrix not found — will use fallback")

    if has_meds_matrix:
        logger.info(f"  ✅ Medication matrix found: {_data_path('meds_matrix')}")
        meds_matrix = pd.read_csv(_data_path('meds_matrix'))
        meds_matrix.columns = meds_matrix.columns.str.strip().str.lower()
        logger.info(f"     {meds_matrix['patient_guid'].nunique():,} patients, "
                    f"{meds_matrix['feature_name'].nunique():,} features")
    else:
        logger.warning(f"  ⚠️ Medication matrix not found — will use fallback")

    # ════════════════════════════════════════════════════════════
    # PART 1: OBSERVATIONS
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: OBSERVATIONS")
    logger.info("=" * 80)

    obs_stat, obs_total_pos, obs_total_neg = compute_statistical_ranking(pos_obs, neg_obs, 'observation')
    obs_features = obs_stat['term'].tolist()[:CONFIG['max_ml_features']]

    if has_obs_matrix:
        obs_ml = compute_ml_ranking_from_patient_matrix(obs_matrix, obs_features, 'observation')
    else:
        obs_ml = compute_ml_ranking_from_prevalence(obs_stat, obs_total_pos, obs_total_neg, 'observation')

    obs_full = build_full_ranking(obs_stat, obs_ml, 'observation')

    logger.info(f"\n📋 Top 20 Observations:")
    print_top_n(obs_full, 20)

    save_csv(obs_full, 'bladder_obs_all.csv')
    save_csv(obs_full, 'bladder_obs_top150.csv', top_n=CONFIG['top_n_output'])

    # ════════════════════════════════════════════════════════════
    # PART 2: MEDICATIONS
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: MEDICATIONS")
    logger.info("=" * 80)

    meds_stat, meds_total_pos, meds_total_neg = compute_statistical_ranking(pos_meds, neg_meds, 'medication')
    meds_features = meds_stat['term'].tolist()[:CONFIG['max_ml_features']]

    if has_meds_matrix:
        meds_ml = compute_ml_ranking_from_patient_matrix(meds_matrix, meds_features, 'medication')
    else:
        meds_ml = compute_ml_ranking_from_prevalence(meds_stat, meds_total_pos, meds_total_neg, 'medication')

    meds_full = build_full_ranking(meds_stat, meds_ml, 'medication')

    logger.info(f"\n📋 Top 20 Medications:")
    print_top_n(meds_full, 20)

    save_csv(meds_full, 'bladder_meds_all.csv')
    save_csv(meds_full, 'bladder_meds_top150.csv', top_n=CONFIG['top_n_output'])

    # ════════════════════════════════════════════════════════════
    # PART 3: COMBINED
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: COMBINED (Observations + Medications)")
    logger.info("=" * 80)

    all_features = pd.concat([obs_full, meds_full], ignore_index=True)

    # Re-normalise across both types
    all_features['stat_score'] = normalise(all_features['log_odds_ratio'].values)
    all_features['chi2_score'] = normalise(all_features['chi_squared'].values)
    all_features['ml_importance_combined'] = normalise(all_features['ml_importance'].values)

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

    logger.info(f"\n📋 Top 20 Combined Features:")
    print_top_n(all_features, 20)

    save_csv(all_features, 'bladder_combined_all.csv')
    save_csv(all_features, 'bladder_combined_top150.csv', top_n=CONFIG['top_n_output'])

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════
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
  │   bladder_obs_top150.csv       → {min(len(obs_full), 150):>4} observations          │
  │   bladder_meds_top150.csv      → {min(len(meds_full), 150):>4} medications           │
  │                                                              │
  │ COMBINED:                                                    │
  │   bladder_combined_top150.csv  → {n_obs:>4} obs + {n_meds:>4} meds          │
  │                                                              │
  │ FULL REFERENCE:                                              │
  │   bladder_obs_all.csv          → {len(obs_full):>6} observations       │
  │   bladder_meds_all.csv         → {len(meds_full):>6} medications        │
  │   bladder_combined_all.csv     → {len(all_features):>6} total              │
  └──────────────────────────────────────────────────────────────┘

📊 ML Data Source:
  Observations: {'REAL patient matrix ✅' if has_obs_matrix else 'SIMULATED (fallback) ⚠️'}
  Medications:  {'REAL patient matrix ✅' if has_meds_matrix else 'SIMULATED (fallback) ⚠️'}

📊 Observations:  {len(obs_full):,} scored | OR>2: {(obs_full['odds_ratio']>2).sum()} | Sig: {(obs_full['p_value']<0.05).sum()}
📊 Medications:   {len(meds_full):,} scored | OR>2: {(meds_full['odds_ratio']>2).sum()} | Sig: {(meds_full['p_value']<0.05).sum()}

📊 Combined Top {CONFIG['top_n_output']}: {n_obs} obs ({n_obs/max(n_obs+n_meds,1)*100:.0f}%) + {n_meds} meds ({n_meds/max(n_obs+n_meds,1)*100:.0f}%)

📊 Confidence: VeryHigh={( combined_top150['confidence_tier']=='Very High').sum()} | High={(combined_top150['confidence_tier']=='High').sum()} | Medium={(combined_top150['confidence_tier']=='Medium').sum()} | Low={(combined_top150['confidence_tier']=='Low').sum()}""")

    # Top 5 each
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


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        all_features, obs_full, meds_full = main()
    except Exception as e:
        logger.error(f"\n❌ FATAL: {e}", exc_info=True)
        sys.exit(1)
