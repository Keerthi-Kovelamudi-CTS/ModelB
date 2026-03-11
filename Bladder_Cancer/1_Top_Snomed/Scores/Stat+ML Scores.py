"""
╔══════════════════════════════════════════════════════════════════╗
║  PRODUCTION — BLADDER CANCER — Phase 1.3                       ║
║  Combined Feature Ranking (Statistical + ML)                   ║
║  Version: 1.0                                                  ║
║                                                                ║
║  Input Files (from Data/):                                     ║
║    - Bladder_positive_obs.csv                                  ║
║    - Bladder_negative_obs.csv                                  ║
║    - Bladder_positive_meds.csv                                  ║
║    - Bladder_negative_meds.csv                                  ║
║    - Bladder_patient_obs_matrix.csv (obs matrix)               ║
║    - Bladder_patient_meds_matrix.csv (meds matrix)            ║
║                                                                ║
║  Output Files: Scores/outputs/Stat+ML Scores/                  ║
║    - bladder_obs_top150.csv                                    ║
║    - bladder_meds_top150.csv                                   ║
║    - bladder_combined_top150.csv                               ║
║    - bladder_obs_all.csv                                       ║
║    - bladder_meds_all.csv                                      ║
║    - bladder_combined_all.csv                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import logging
import sys
import os
import time
from datetime import datetime

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

# Data directory: same parent as Scores/, then Data/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'Data')

CONFIG = {
    'random_seed': 42,
    'min_patients_per_feature': 5,
    'max_ml_features': 1000,
    'top_n_output': 150,
    'cv_folds': 5,
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
    'data_dir': _DATA_DIR,
    'input_files': {
        'pos_obs':     'Bladder_positive_obs.csv',
        'neg_obs':     'Bladder_negative_obs.csv',
        'pos_meds':    'Bladder_positive_meds.csv',
        'neg_meds':    'Bladder_negative_meds.csv',
        'obs_matrix':  'Bladder_patient_obs_matrix.csv',
        'meds_matrix': 'Bladder_patient_meds_matrix.csv',
    },
    'output_dir': os.path.join(_SCRIPT_DIR, 'outputs', 'Stat+ML Scores'),
    'confidence_bins':   [-0.01, 0.2, 0.5, 0.8, 1.01],
    'confidence_labels': ['Low', 'Medium', 'High', 'Very High']
}

# ════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure structured logging to file and console."""
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
    """Min-max normalise to [0, 1]. Handles edge cases."""
    arr = np.array(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def validate_csv(df, required_cols, name):
    """Validate that a DataFrame has the required columns and is non-empty."""
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}. Found: {list(df.columns)}")
    if len(df) == 0:
        raise ValueError(f"[{name}] DataFrame is empty!")
    logger.info(f"  ✅ {name}: {len(df):,} rows, columns validated")
    return df


def _elapsed(seconds):
    """Format elapsed seconds as e.g. '2m 35s' or '45s'."""
    if seconds >= 60:
        m, s = int(seconds // 60), int(round(seconds % 60))
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


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
    """Save DataFrame to CSV in the output directory."""
    filepath = os.path.join(CONFIG['output_dir'], filename)
    output = df.head(top_n).copy() if top_n else df.copy()
    output = output.rename(columns={k: v for k, v in CSV_COLUMN_RENAMES.items() if k in output.columns})
    output = _format_csv_no_trailing_zeros(output)
    output.to_csv(filepath, index=False)
    logger.info(f"  💾 Saved: {filepath} ({len(output):,} rows)")
    return output


# ════════════════════════════════════════════════════════════════
# STEP 1: STATISTICAL RANKING (Path A)
# ════════════════════════════════════════════════════════════════

def compute_statistical_ranking(pos_df, neg_df, feature_type='observation'):
    """
    Compute Odds Ratio, Chi-Squared, Prevalence for a positive/negative pair.

    Parameters:
        pos_df: DataFrame with positive cohort aggregated SNOMED/med counts
        neg_df: DataFrame with negative cohort aggregated SNOMED/med counts
        feature_type: 'observation' or 'medication'

    Returns:
        merged DataFrame with statistical scores, total_pos, total_neg
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Statistical Ranking: {feature_type}s")
    logger.info(f"{'='*60}")

    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    pos_df.columns = pos_df.columns.str.strip().str.lower()
    neg_df.columns = neg_df.columns.str.strip().str.lower()

    # Extract totals
    total_pos = int(pos_df['n_patient_count_total'].iloc[0])
    total_neg = int(neg_df['n_patient_count_total'].iloc[0])

    logger.info(f"  Positive cohort: {total_pos:,} patients, {len(pos_df):,} unique terms")
    logger.info(f"  Negative cohort: {total_neg:,} patients, {len(neg_df):,} unique terms")

    if total_pos == 0 or total_neg == 0:
        raise ValueError(f"Cohort has zero patients! pos={total_pos}, neg={total_neg}")

    # Merge on term
    merged = pos_df.merge(neg_df, on='term', how='outer', suffixes=('_pos', '_neg'))
    merged['n_patient_count_pos'] = merged['n_patient_count_pos'].fillna(0).astype(int)
    merged['n_patient_count_neg'] = merged['n_patient_count_neg'].fillna(0).astype(int)

    logger.info(f"  Merged (union): {len(merged):,} unique terms")

    # Resolve code ID based on feature type
    if 'snomed_id_pos' in merged.columns:
        merged['code_id'] = merged['snomed_id_pos'].fillna(merged.get('snomed_id_neg', np.nan))
        merged['code_type'] = 'SNOMED'
    elif 'med_code_id_pos' in merged.columns:
        merged['code_id'] = merged['med_code_id_pos'].fillna(merged.get('med_code_id_neg', np.nan))
        merged['code_type'] = 'DMD'
    else:
        merged['code_id'] = np.nan
        merged['code_type'] = 'UNKNOWN'
        logger.warning(f"  ⚠️ Could not resolve code_id column for {feature_type}")

    # ── Prevalence ──
    merged['prevalence_pos'] = merged['n_patient_count_pos'] / total_pos
    merged['prevalence_neg'] = merged['n_patient_count_neg'] / total_neg
    merged['prevalence_diff'] = merged['prevalence_pos'] - merged['prevalence_neg']
    merged['prevalence_ratio'] = merged['prevalence_pos'] / merged['prevalence_neg'].replace(0, np.nan)

    # ── Odds Ratio (Haldane 0.5 correction) ──
    a = merged['n_patient_count_pos'] + 0.5     # cancer + has code
    b = (total_pos - merged['n_patient_count_pos']) + 0.5  # cancer + no code
    c = merged['n_patient_count_neg'] + 0.5     # no cancer + has code
    d = (total_neg - merged['n_patient_count_neg']) + 0.5  # no cancer + no code
    merged['odds_ratio'] = (a * d) / (b * c)
    merged['log_odds_ratio'] = np.log(merged['odds_ratio'])

    # ── 95% Confidence Interval for log(OR) ──
    merged['log_or_se'] = np.sqrt(1/a + 1/b + 1/c + 1/d)
    merged['or_ci_lower'] = np.exp(merged['log_odds_ratio'] - 1.96 * merged['log_or_se'])
    merged['or_ci_upper'] = np.exp(merged['log_odds_ratio'] + 1.96 * merged['log_or_se'])

    # ── Chi-Squared test ──
    chi2_vals, p_vals = [], []
    for _, row in merged.iterrows():
        table = np.array([
            [row['n_patient_count_pos'], total_pos - row['n_patient_count_pos']],
            [row['n_patient_count_neg'], total_neg - row['n_patient_count_neg']]
        ])
        try:
            if table.min() < 0:
                raise ValueError("Negative value in contingency table")
            chi2, p, _, _ = chi2_contingency(table, correction=True)
            chi2_vals.append(round(chi2, 6))
            p_vals.append(p)
        except Exception as e:
            logger.debug(f"  Chi2 failed for {row.get('term', '?')}: {e}")
            chi2_vals.append(np.nan)
            p_vals.append(np.nan)

    merged['chi_squared'] = chi2_vals
    merged['p_value'] = p_vals

    # ── Bonferroni corrected p-value ──
    n_tests = len(merged)
    merged['p_value_bonferroni'] = np.minimum(merged['p_value'] * n_tests, 1.0)

    # ── Feature type label ──
    merged['feature_type'] = feature_type

    # ── Filter: minimum patient threshold ──
    min_patients = CONFIG['min_patients_per_feature']
    before_filter = len(merged)
    merged = merged[merged['n_patient_count_pos'] >= min_patients].copy()
    logger.info(f"  After filter (>={min_patients} patients): {len(merged):,} / {before_filter:,} terms")

    # ── Rank by Odds Ratio ──
    merged = merged.sort_values('odds_ratio', ascending=False).reset_index(drop=True)
    merged['stat_rank'] = range(1, len(merged) + 1)

    # ── Log summary stats ──
    logger.info(f"  OR > 1.5: {(merged['odds_ratio'] > 1.5).sum()}")
    logger.info(f"  OR > 2.0: {(merged['odds_ratio'] > 2.0).sum()}")
    logger.info(f"  OR > 5.0: {(merged['odds_ratio'] > 5.0).sum()}")
    logger.info(f"  Significant (p < 0.05): {(merged['p_value'] < 0.05).sum()}")
    logger.info(f"  Significant after Bonferroni: {(merged['p_value_bonferroni'] < 0.05).sum()}")

    return merged, total_pos, total_neg


# ════════════════════════════════════════════════════════════════
# STEP 2: ML-BASED RANKING (Path B) — REAL PATIENT DATA
# ════════════════════════════════════════════════════════════════

def compute_ml_ranking_from_patient_matrix(patient_matrix_df, feature_list, feature_type='observation'):
    """
    Train ML models on REAL patient-level data to compute feature importance.

    Parameters:
        patient_matrix_df: DataFrame in long format (patient_guid, label, feature_name, event_count)
        feature_list: list of feature terms to include
        feature_type: 'observation' or 'medication'

    Returns:
        DataFrame with per-feature ML importance scores
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ML Ranking (Real Patient Data): {feature_type}s")
    logger.info(f"{'='*60}")

    np.random.seed(CONFIG['random_seed'])

    # Filter to only the features we want
    pm = patient_matrix_df.copy()
    pm = pm[pm['feature_name'].isin(feature_list)]

    if len(pm) == 0:
        logger.warning(f"  ⚠️ No matching features found in patient matrix for {feature_type}!")
        return pd.DataFrame({'term': feature_list, 'ml_importance': 0, 'ml_rank': 999})

    logger.info(f"  Patient matrix: {pm['patient_guid'].nunique():,} patients, {pm['feature_name'].nunique():,} features")

    # ── Pivot: long → wide (patient × feature matrix) ──
    logger.info("  Pivoting to wide format...")

    # Get patient-level demographics (one row per patient)
    patient_info = pm.groupby('patient_guid').agg(
        label=('label', 'first'),
        sex=('sex', 'first'),
        age_at_index=('age_at_index', 'first') if 'age_at_index' in pm.columns else ('label', 'first')
    ).reset_index()

    # Pivot features: binary (1 if any event, 0 if not)
    feature_pivot = pm.pivot_table(
        index='patient_guid',
        columns='feature_name',
        values='event_count',
        aggfunc='sum',
        fill_value=0
    )

    # Binary: has feature or not
    feature_binary = (feature_pivot > 0).astype(int)

    # Ensure all features are present (even if no events for some patients)
    for feat in feature_list:
        if feat not in feature_binary.columns:
            feature_binary[feat] = 0

    # Keep only requested features and maintain order
    feature_binary = feature_binary.reindex(columns=feature_list, fill_value=0)

    # Merge with labels
    data = patient_info[['patient_guid', 'label']].set_index('patient_guid').join(feature_binary)
    data = data.dropna(subset=['label'])

    X = data[feature_list].values
    y = data['label'].astype(int).values

    logger.info(f"  Feature matrix: {X.shape[0]:,} patients × {X.shape[1]:,} features")
    logger.info(f"  Class balance: {y.sum():,} positive ({y.sum()/len(y)*100:.1f}%), "
                f"{(1-y).sum():,} negative ({(1-y).sum()/len(y)*100:.1f}%)")

    if y.sum() < 10 or (1-y).sum() < 10:
        logger.warning(f"  ⚠️ Very few samples in one class — ML results may be unreliable")

    # ── Cross-validated training ──
    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_seed'])

    # ── Model 1: L1 Logistic Regression ──
    logger.info("  Training L1 Logistic Regression (5-fold CV)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=1.0,
        max_iter=10000,
        random_state=CONFIG['random_seed'],
        class_weight='balanced'
    )

    lr_cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"    CV AUROC: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

    lr.fit(X_scaled, y)
    lr_importance = np.abs(lr.coef_[0])
    logger.info(f"    Non-zero coefficients: {(lr_importance > 0).sum()} / {len(lr_importance)}")

    # ── Model 2: Random Forest ──
    logger.info("  Training Random Forest (5-fold CV)...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=20,
        random_state=CONFIG['random_seed'],
        class_weight='balanced',
        n_jobs=-1
    )

    rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"    CV AUROC: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    # ── Model 3: Gradient Boosting ──
    logger.info("  Training Gradient Boosting (5-fold CV)...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=CONFIG['random_seed'],
        subsample=0.8
    )

    gb_cv_scores = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"    CV AUROC: {gb_cv_scores.mean():.4f} ± {gb_cv_scores.std():.4f}")

    gb.fit(X, y)
    gb_importance = gb.feature_importances_

    # ── Aggregate ML importance ──
    weights = CONFIG['ml_weights']
    ml_scores = pd.DataFrame({
        'term': feature_list,
        'lr_importance': normalise(lr_importance),
        'rf_importance': normalise(rf_importance),
        'gb_importance': normalise(gb_importance),
        'lr_cv_auroc': lr_cv_scores.mean(),
        'rf_cv_auroc': rf_cv_scores.mean(),
        'gb_cv_auroc': gb_cv_scores.mean(),
    })

    ml_scores['ml_importance'] = (
        weights['lr'] * ml_scores['lr_importance'] +
        weights['rf'] * ml_scores['rf_importance'] +
        weights['gb'] * ml_scores['gb_importance']
    )

    ml_scores = ml_scores.sort_values('ml_importance', ascending=False).reset_index(drop=True)
    ml_scores['ml_rank'] = range(1, len(ml_scores) + 1)

    logger.info(f"  Top ML feature: {ml_scores.iloc[0]['term']} (score={ml_scores.iloc[0]['ml_importance']:.4f})")
    logger.info(f"  Non-zero ML features: {(ml_scores['ml_importance'] > 0).sum()}")

    return ml_scores


def compute_ml_ranking_from_prevalence(stat_df, total_pos, total_neg, feature_type='observation'):
    """
    FALLBACK: If patient-level matrix is not available,
    simulate from prevalence (less accurate but still useful).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ML Ranking (FALLBACK — Prevalence Simulated): {feature_type}s")
    logger.info(f"{'='*60}")
    logger.warning("  ⚠️ Using simulated data — results approximate. Use Query 5 for production.")

    np.random.seed(CONFIG['random_seed'])

    ml_terms = stat_df['term'].tolist()
    max_features = CONFIG['max_ml_features']
    if len(ml_terms) > max_features:
        ml_terms = stat_df.nlargest(max_features, 'n_patient_count_pos')['term'].tolist()
        logger.info(f"  Capped to top {max_features} features by patient count")

    n_pos_sample = min(int(total_pos), 5000)
    n_neg_sample = min(int(total_neg), 5000)

    logger.info(f"  Simulating {n_pos_sample:,} pos + {n_neg_sample:,} neg patients, {len(ml_terms)} features")

    X_pos = np.zeros((n_pos_sample, len(ml_terms)))
    X_neg = np.zeros((n_neg_sample, len(ml_terms)))

    for i, term in enumerate(ml_terms):
        row = stat_df[stat_df['term'] == term]
        if len(row) == 0:
            continue
        prev_pos = min(float(row['prevalence_pos'].values[0]), 1.0)
        prev_neg = min(float(row['prevalence_neg'].values[0]), 1.0)
        X_pos[:, i] = np.random.binomial(1, max(prev_pos, 0), n_pos_sample)
        X_neg[:, i] = np.random.binomial(1, max(prev_neg, 0), n_neg_sample)

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos_sample + [0] * n_neg_sample)
    shuffle_idx = np.random.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]

    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_seed'])

    # L1 Logistic
    logger.info("  Training L1 Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=10000,
                            random_state=CONFIG['random_seed'], class_weight='balanced')
    lr_cv = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"    CV AUROC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
    lr.fit(X_scaled, y)
    lr_imp = np.abs(lr.coef_[0])

    # Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=20,
                                random_state=CONFIG['random_seed'], class_weight='balanced', n_jobs=-1)
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"    CV AUROC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    # Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    min_samples_leaf=20, random_state=CONFIG['random_seed'], subsample=0.8)
    gb_cv = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"    CV AUROC: {gb_cv.mean():.4f} ± {gb_cv.std():.4f}")
    gb.fit(X, y)
    gb_imp = gb.feature_importances_

    weights = CONFIG['ml_weights']
    ml_scores = pd.DataFrame({
        'term': ml_terms,
        'lr_importance': normalise(lr_imp),
        'rf_importance': normalise(rf_imp),
        'gb_importance': normalise(gb_imp),
        'lr_cv_auroc': lr_cv.mean(),
        'rf_cv_auroc': rf_cv.mean(),
        'gb_cv_auroc': gb_cv.mean(),
    })
    ml_scores['ml_importance'] = (
        weights['lr'] * ml_scores['lr_importance'] +
        weights['rf'] * ml_scores['rf_importance'] +
        weights['gb'] * ml_scores['gb_importance']
    )
    ml_scores = ml_scores.sort_values('ml_importance', ascending=False).reset_index(drop=True)
    ml_scores['ml_rank'] = range(1, len(ml_scores) + 1)

    return ml_scores


# ════════════════════════════════════════════════════════════════
# STEP 3: BUILD FULL RANKING (Stat + ML → Combined)
# ════════════════════════════════════════════════════════════════

def build_full_ranking(stat_df, ml_df, feature_type):
    """
    Merge statistical and ML rankings, compute combined score.

    Returns:
        DataFrame with combined_rank, combined_score, confidence_tier
    """
    logger.info(f"\n  Building combined ranking for {feature_type}s...")

    combined = stat_df.merge(
        ml_df[['term', 'lr_importance', 'rf_importance', 'gb_importance',
               'ml_importance', 'ml_rank', 'lr_cv_auroc', 'rf_cv_auroc', 'gb_cv_auroc']],
        on='term', how='left'
    )

    # Fill NaN ML scores (features not in ML analysis)
    ml_cols = ['lr_importance', 'rf_importance', 'gb_importance', 'ml_importance',
               'ml_rank', 'lr_cv_auroc', 'rf_cv_auroc', 'gb_cv_auroc']
    for col in ml_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    # Normalise scores
    combined['stat_score'] = normalise(combined['log_odds_ratio'].values)
    combined['chi2_score'] = normalise(combined['chi_squared'].values)

    # Combined weighted score
    w = CONFIG['combined_weights']
    combined['combined_score'] = (
        w['stat_score']    * combined['stat_score'] +
        w['chi2_score']    * combined['chi2_score'] +
        w['ml_importance'] * combined['ml_importance']
    )

    # Rank and tier
    combined = combined.sort_values('combined_score', ascending=False).reset_index(drop=True)
    combined['combined_rank'] = range(1, len(combined) + 1)
    combined['confidence_tier'] = pd.cut(
        combined['combined_score'],
        bins=CONFIG['confidence_bins'],
        labels=CONFIG['confidence_labels']
    )
    combined['feature_type'] = feature_type

    # Agreement flag
    top_n = CONFIG['top_n_output']
    combined['stat_top150'] = combined['stat_rank'] <= top_n
    combined['ml_top150'] = combined['ml_rank'].apply(lambda x: x <= top_n if x > 0 else False)
    combined['both_agree'] = combined['stat_top150'] & combined['ml_top150']

    logger.info(f"  Total features: {len(combined):,}")
    logger.info(f"  Both stat+ML agree (top {top_n}): {combined['both_agree'].sum()}")

    return combined


# ════════════════════════════════════════════════════════════════
# STEP 4: DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════

def print_top_n(df, n, title):
    """Pretty-print top N features."""
    logger.info(f"\n  {'Rank':<6} {'Type':<12} {'Term':<45} {'OR':>8} {'StatR':>6} {'MLR':>6} {'Score':>8} {'Tier':<10} {'Agree':<6}")
    logger.info("  " + "-" * 110)
    for _, row in df.head(n).iterrows():
        agree = "✅" if row.get('both_agree', False) else ""
        logger.info(
            f"  {int(row['combined_rank']):<6} "
            f"{str(row.get('feature_type', '')):<12} "
            f"{str(row['term'])[:44]:<45} "
            f"{row['odds_ratio']:>8.2f} "
            f"{int(row['stat_rank']):>6} "
            f"{int(row['ml_rank']):>6} "
            f"{row['combined_score']:>8.3f} "
            f"{str(row['confidence_tier']):<10} "
            f"{agree:<6}"
        )


# ════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()
    logger.info("=" * 80)
    logger.info("PRODUCTION — BLADDER CANCER — Phase 1.3: Feature Ranking")
    logger.info("Observations + Medications — Separate + Combined")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # ── Load input CSVs from Data/ ──
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
        logger.error(f"❌ Missing input file: {e}")
        logger.error(f"  Expected input files in: {os.path.abspath(data_dir)}")
        sys.exit(1)

    # Validate
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
        logger.info(f"\n  ✅ Observation matrix found: {_data_path('obs_matrix')}")
        obs_matrix = pd.read_csv(_data_path('obs_matrix'))
        obs_matrix.columns = obs_matrix.columns.str.strip().str.lower()
        logger.info(f"     {obs_matrix['patient_guid'].nunique():,} patients, "
                    f"{obs_matrix['feature_name'].nunique():,} features")
    else:
        logger.warning(f"\n  ⚠️ Observation matrix not found — will use fallback")

    if has_meds_matrix:
        logger.info(f"  ✅ Medication matrix found: {_data_path('meds_matrix')}")
        meds_matrix = pd.read_csv(_data_path('meds_matrix'))
        meds_matrix.columns = meds_matrix.columns.str.strip().str.lower()
        logger.info(f"     {meds_matrix['patient_guid'].nunique():,} patients, "
                    f"{meds_matrix['feature_name'].nunique():,} features")
    else:
        logger.warning(f"  ⚠️ Medication matrix not found — will use fallback")

    t_after_load = time.perf_counter()
    logger.info(f"\n  ⏱ Load + validate: {_elapsed(t_after_load - t_start)}")

    # ════════════════════════════════════════════════════════════
    # PART 1: OBSERVATIONS
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: OBSERVATIONS")
    logger.info("=" * 80)

    obs_stat, obs_total_pos, obs_total_neg = compute_statistical_ranking(pos_obs, neg_obs, 'observation')

    obs_feature_list = obs_stat['term'].tolist()[:CONFIG['max_ml_features']]

    if has_obs_matrix:
        obs_ml = compute_ml_ranking_from_patient_matrix(obs_matrix, obs_feature_list, 'observation')
    else:
        obs_ml = compute_ml_ranking_from_prevalence(obs_stat, obs_total_pos, obs_total_neg, 'observation')

    obs_full = build_full_ranking(obs_stat, obs_ml, 'observation')

    logger.info(f"\n📋 Top 20 Observations:")
    print_top_n(obs_full, 20, "Observations")

    save_csv(obs_full, 'bladder_obs_all.csv')
    save_csv(obs_full, 'bladder_obs_top150.csv', top_n=CONFIG['top_n_output'])

    t_after_obs = time.perf_counter()
    logger.info(f"\n  ⏱ PART 1 (Observations) done at {datetime.now().strftime('%H:%M:%S')} — step: {_elapsed(t_after_obs - t_after_load)}, total: {_elapsed(t_after_obs - t_start)}")

    # ════════════════════════════════════════════════════════════
    # PART 2: MEDICATIONS
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: MEDICATIONS")
    logger.info("=" * 80)

    meds_stat, meds_total_pos, meds_total_neg = compute_statistical_ranking(pos_meds, neg_meds, 'medication')

    meds_feature_list = meds_stat['term'].tolist()[:CONFIG['max_ml_features']]

    if has_meds_matrix:
        meds_ml = compute_ml_ranking_from_patient_matrix(meds_matrix, meds_feature_list, 'medication')
    else:
        meds_ml = compute_ml_ranking_from_prevalence(meds_stat, meds_total_pos, meds_total_neg, 'medication')

    meds_full = build_full_ranking(meds_stat, meds_ml, 'medication')

    logger.info(f"\n📋 Top 20 Medications:")
    print_top_n(meds_full, 20, "Medications")

    save_csv(meds_full, 'bladder_meds_all.csv')
    save_csv(meds_full, 'bladder_meds_top150.csv', top_n=CONFIG['top_n_output'])

    t_after_meds = time.perf_counter()
    logger.info(f"\n  ⏱ PART 2 (Medications) done at {datetime.now().strftime('%H:%M:%S')} — step: {_elapsed(t_after_meds - t_after_obs)}, total: {_elapsed(t_after_meds - t_start)}")

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
    print_top_n(all_features, 20, "Combined")

    save_csv(all_features, 'bladder_combined_all.csv')
    save_csv(all_features, 'bladder_combined_top150.csv', top_n=CONFIG['top_n_output'])

    t_after_combined = time.perf_counter()
    logger.info(f"\n  ⏱ PART 3 (Combined) done at {datetime.now().strftime('%H:%M:%S')} — step: {_elapsed(t_after_combined - t_after_meds)}, total: {_elapsed(t_after_combined - t_start)}")

    # ════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ════════════════════════════════════════════════════════════
    combined_top150 = all_features.head(CONFIG['top_n_output'])
    n_obs  = (combined_top150['feature_type'] == 'observation').sum()
    n_meds = (combined_top150['feature_type'] == 'medication').sum()

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY REPORT")
    logger.info("=" * 80)

    logger.info(f"""
📂 Output Files ({CONFIG['output_dir']}/):
  ┌──────────────────────────────────────────────────────────────┐
  │ SEPARATE:                                                    │
  │   bladder_obs_top150.csv       → {min(len(obs_full), 150):>4} observations          │
  │   bladder_meds_top150.csv      → {min(len(meds_full), 150):>4} medications           │
  │                                                              │
  │ COMBINED:                                                    │
  │   bladder_combined_top150.csv  → {n_obs:>4} obs + {n_meds:>4} meds = {n_obs+n_meds:>4}     │
  │                                                              │
  │ FULL REFERENCE:                                              │
  │   bladder_obs_all.csv          → {len(obs_full):>6} observations       │
  │   bladder_meds_all.csv         → {len(meds_full):>6} medications        │
  │   bladder_combined_all.csv     → {len(all_features):>6} total features     │
  └──────────────────────────────────────────────────────────────┘

📊 Cohort Sizes:
  Positive (obs):  {obs_total_pos:,} patients
  Negative (obs):  {obs_total_neg:,} patients
  Positive (meds): {meds_total_pos:,} patients
  Negative (meds): {meds_total_neg:,} patients

📊 ML Data Source:
  Observations: {'REAL patient matrix ✅' if has_obs_matrix else 'SIMULATED (fallback) ⚠️'}
  Medications:  {'REAL patient matrix ✅' if has_meds_matrix else 'SIMULATED (fallback) ⚠️'}

📊 Observations:
  Total scored:             {len(obs_full):,}
  OR > 2.0:                 {(obs_full['odds_ratio'] > 2.0).sum()}
  OR > 5.0:                 {(obs_full['odds_ratio'] > 5.0).sum()}
  Significant (p < 0.05):   {(obs_full['p_value'] < 0.05).sum()}
  Bonferroni sig (p < 0.05): {(obs_full['p_value_bonferroni'] < 0.05).sum()}

📊 Medications:
  Total scored:             {len(meds_full):,}
  OR > 2.0:                 {(meds_full['odds_ratio'] > 2.0).sum()}
  OR > 5.0:                 {(meds_full['odds_ratio'] > 5.0).sum()}
  Significant (p < 0.05):   {(meds_full['p_value'] < 0.05).sum()}
  Bonferroni sig (p < 0.05): {(meds_full['p_value_bonferroni'] < 0.05).sum()}

📊 Combined Top {CONFIG['top_n_output']} Composition:
  Observations:   {n_obs} ({n_obs/max(n_obs+n_meds,1)*100:.0f}%)
  Medications:    {n_meds} ({n_meds/max(n_obs+n_meds,1)*100:.0f}%)

📊 Confidence Tiers (Combined Top {CONFIG['top_n_output']}):
  Very High: {(combined_top150['confidence_tier'] == 'Very High').sum()}
  High:      {(combined_top150['confidence_tier'] == 'High').sum()}
  Medium:    {(combined_top150['confidence_tier'] == 'Medium').sum()}
  Low:       {(combined_top150['confidence_tier'] == 'Low').sum()}

📊 Agreement (in top {CONFIG['top_n_output']} of BOTH stat + ML):
  Observations: {obs_full.head(150)['both_agree'].sum() if 'both_agree' in obs_full.columns else 'N/A'}
  Medications:  {meds_full.head(150)['both_agree'].sum() if 'both_agree' in meds_full.columns else 'N/A'}""")

    # Top 5 from each in combined
    logger.info(f"\n📊 Top 5 Observations (in combined ranking):")
    for _, r in combined_top150[combined_top150['feature_type'] == 'observation'].head(5).iterrows():
        logger.info(f"  #{int(r['combined_rank'])} {r['term']} "
                    f"(OR={r['odds_ratio']:.2f}, CI=[{r.get('or_ci_lower', 0):.2f}-{r.get('or_ci_upper', 0):.2f}], "
                    f"p={r['p_value']:.2e}, score={r['combined_score']:.3f})")

    logger.info(f"\n📊 Top 5 Medications (in combined ranking):")
    for _, r in combined_top150[combined_top150['feature_type'] == 'medication'].head(5).iterrows():
        logger.info(f"  #{int(r['combined_rank'])} {r['term']} "
                    f"(OR={r['odds_ratio']:.2f}, CI=[{r.get('or_ci_lower', 0):.2f}-{r.get('or_ci_upper', 0):.2f}], "
                    f"p={r['p_value']:.2e}, score={r['combined_score']:.3f})")

    t_end = time.perf_counter()
    logger.info("\n" + "=" * 80)
    logger.info("✅ PHASE 1.3 COMPLETE — Ready for Clinical Review (Phase 1.4)")
    logger.info(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total runtime: {_elapsed(t_end - t_start)}")
    logger.info("=" * 80)

    return all_features, obs_full, meds_full


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        all_features, obs_full, meds_full = main()
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)