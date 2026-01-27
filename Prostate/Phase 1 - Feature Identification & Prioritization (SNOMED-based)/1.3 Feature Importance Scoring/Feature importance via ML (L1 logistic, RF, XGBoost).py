"""
FEATURE IMPORTANCE SCORING (PATH B)
ML-BASED ANALYSIS - 12 MONTH WINDOW

Methods:
- Random Forest Feature Importance
- XGBoost Feature Importance (Gain, Cover, Weight)
- Mutual Information

Output: Ranked list of top features by ML importance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
import os

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - use script directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_dir = os.path.join(base_dir, 'Data')

# Input files
POSITIVE_FILE = os.path.join(data_dir, 'top-snomed_prostateCancer_12mo_Window.csv')
NEGATIVE_FILE = os.path.join(data_dir, 'top-snomed_no-prostate-cancer_12mo_Window.csv')

# Output directory
OUTPUT_DIR = os.path.join(base_dir, 'Results2_Analysis1.3', 'ML analysis')

# Minimum patients threshold
MIN_PATIENTS = 10

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_and_prepare_data(positive_file, negative_file):
    """
    Load cancer and control SNOMED data and create feature matrix.
    """
    # Load data
    pos = pd.read_csv(positive_file)
    neg = pd.read_csv(negative_file)
    
    # Standardize column names
    pos.columns = pos.columns.str.upper()
    neg.columns = neg.columns.str.upper()
    
    # Handle SNOMED_ID column variations
    if 'SNOMED_C_T_CONCEPT_ID' in pos.columns:
        pos['SNOMED_ID'] = pos['SNOMED_C_T_CONCEPT_ID']
    if 'SNOMED_C_T_CONCEPT_ID' in neg.columns:
        neg['SNOMED_ID'] = neg['SNOMED_C_T_CONCEPT_ID']
    
    # Clean TERM column
    pos['TERM'] = pos['TERM'].astype(str).str.strip().str.strip('"')
    neg['TERM'] = neg['TERM'].astype(str).str.strip().str.strip('"')
    
    # Get total patients per cohort
    pos_total = pos['N_PATIENT_COUNT_TOTAL'].iloc[0] if 'N_PATIENT_COUNT_TOTAL' in pos.columns else pos['N_PATIENT_COUNT'].sum()
    neg_total = neg['N_PATIENT_COUNT_TOTAL'].iloc[0] if 'N_PATIENT_COUNT_TOTAL' in neg.columns else neg['N_PATIENT_COUNT'].sum()
    
    # Calculate prevalence (percentage)
    pos['PREVALENCE'] = pos['N_PATIENT_COUNT'] / pos_total
    neg['PREVALENCE'] = neg['N_PATIENT_COUNT'] / neg_total
    
    # Create unique identifier
    pos['CODE_KEY'] = pos['TERM'].astype(str) + '|' + pos['SNOMED_ID'].astype(str)
    neg['CODE_KEY'] = neg['TERM'].astype(str) + '|' + neg['SNOMED_ID'].astype(str)
    
    # Get all unique codes
    all_codes = set(pos['CODE_KEY'].tolist() + neg['CODE_KEY'].tolist())
    
    # Create prevalence dictionaries
    pos_prev = dict(zip(pos['CODE_KEY'], pos['PREVALENCE']))
    neg_prev = dict(zip(neg['CODE_KEY'], neg['PREVALENCE']))
    
    # Create feature matrix
    feature_data = []
    
    for code in all_codes:
        pos_p = pos_prev.get(code, 0)
        neg_p = neg_prev.get(code, 0)
        
        # Get term and snomed_id
        parts = code.split('|')
        term = parts[0] if len(parts) > 0 else code
        snomed_id = parts[1] if len(parts) > 1 else ''
        
        # Get patient counts
        pos_count = pos[pos['CODE_KEY'] == code]['N_PATIENT_COUNT'].values
        neg_count = neg[neg['CODE_KEY'] == code]['N_PATIENT_COUNT'].values
        pos_n = pos_count[0] if len(pos_count) > 0 else 0
        neg_n = neg_count[0] if len(neg_count) > 0 else 0
        
        feature_data.append({
            'CODE_KEY': code,
            'TERM': term,
            'SNOMED_ID': snomed_id,
            'PREV_CANCER': pos_p,
            'PREV_CONTROL': neg_p,
            'PREV_DIFF': pos_p - neg_p,
            'PREV_RATIO': pos_p / neg_p if neg_p > 0 else (999 if pos_p > 0 else 1),
            'N_CANCER': pos_n,
            'N_CONTROL': neg_n,
            'N_TOTAL': pos_n + neg_n,
        })
    
    df = pd.DataFrame(feature_data)
    
    # Filter by minimum patients
    df = df[df['N_TOTAL'] >= MIN_PATIENTS].copy()
    
    return df, pos_total, neg_total

# ============================================================================
# CREATE BINARY FEATURE MATRIX FOR ML
# ============================================================================

def create_ml_feature_matrix(df, pos_total, neg_total):
    """
    Create a binary feature matrix for ML models.
    
    This simulates patient-level data from aggregated statistics.
    Each row = one "simulated patient"
    Each column = one SNOMED code (1 = present, 0 = absent)
    """
    # Get list of SNOMED codes (features)
    codes = df['CODE_KEY'].tolist()
    n_features = len(codes)
    
    # Number of samples to simulate (use smaller number for speed)
    n_cancer_sample = min(int(pos_total), 5000)
    n_control_sample = min(int(neg_total), 5000)
    
    # Create feature matrix
    np.random.seed(RANDOM_SEED)
    
    # Cancer patients
    X_cancer = np.zeros((n_cancer_sample, n_features))
    for i, code in enumerate(codes):
        prevalence = df[df['CODE_KEY'] == code]['PREV_CANCER'].values[0]
        X_cancer[:, i] = np.random.binomial(1, prevalence, n_cancer_sample)
    
    # Control patients
    X_control = np.zeros((n_control_sample, n_features))
    for i, code in enumerate(codes):
        prevalence = df[df['CODE_KEY'] == code]['PREV_CONTROL'].values[0]
        X_control[:, i] = np.random.binomial(1, prevalence, n_control_sample)
    
    # Combine
    X = np.vstack([X_cancer, X_control])
    y = np.array([1] * n_cancer_sample + [0] * n_control_sample)
    
    return X, y, codes

# ============================================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================

def random_forest_importance(X, y, feature_names):
    """
    Calculate feature importance using Random Forest.
    """
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf.fit(X, y)
    
    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    
    # Feature importance
    importance = rf.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'CODE_KEY': feature_names,
        'RF_IMPORTANCE': importance
    })
    
    importance_df = importance_df.sort_values('RF_IMPORTANCE', ascending=False)
    
    return importance_df, cv_scores.mean()

# ============================================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================================

def xgboost_importance(X, y, feature_names):
    """
    Calculate feature importance using XGBoost.
    Returns multiple importance types: gain, cover, weight.
    """
    if not HAS_XGBOOST:
        return None, 0
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (len(y) - sum(y)) / sum(y)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X, y)
    
    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')
    
    # Get feature importance (multiple types)
    importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
    importance_cover = xgb_model.get_booster().get_score(importance_type='cover')
    importance_weight = xgb_model.get_booster().get_score(importance_type='weight')
    
    # Create DataFrame
    importance_data = []
    for i, name in enumerate(feature_names):
        f_key = f'f{i}'
        importance_data.append({
            'CODE_KEY': name,
            'XGB_GAIN': importance_gain.get(f_key, 0),
            'XGB_COVER': importance_cover.get(f_key, 0),
            'XGB_WEIGHT': importance_weight.get(f_key, 0),
        })
    
    importance_df = pd.DataFrame(importance_data)
    
    # Normalize importance scores
    for col in ['XGB_GAIN', 'XGB_COVER', 'XGB_WEIGHT']:
        max_val = importance_df[col].max()
        if max_val > 0:
            importance_df[f'{col}_NORM'] = importance_df[col] / max_val
        else:
            importance_df[f'{col}_NORM'] = 0
    
    # Combined XGBoost score (weighted average)
    importance_df['XGB_COMBINED'] = (
        0.5 * importance_df['XGB_GAIN_NORM'] +
        0.3 * importance_df['XGB_COVER_NORM'] +
        0.2 * importance_df['XGB_WEIGHT_NORM']
    )
    
    importance_df = importance_df.sort_values('XGB_COMBINED', ascending=False)
    
    return importance_df, cv_scores.mean()

# ============================================================================
# MUTUAL INFORMATION
# ============================================================================

def mutual_information_importance(X, y, feature_names):
    """
    Calculate feature importance using Mutual Information.
    """
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'CODE_KEY': feature_names,
        'MI_SCORE': mi_scores
    })
    
    # Normalize
    max_mi = importance_df['MI_SCORE'].max()
    if max_mi > 0:
        importance_df['MI_SCORE_NORM'] = importance_df['MI_SCORE'] / max_mi
    else:
        importance_df['MI_SCORE_NORM'] = 0
    
    importance_df = importance_df.sort_values('MI_SCORE', ascending=False)
    
    return importance_df

# ============================================================================
# COMBINE ALL IMPORTANCE SCORES
# ============================================================================

def combine_importance_scores(df_base, rf_importance, xgb_importance, mi_importance):
    """
    Combine all importance scores into single DataFrame with composite ranking.
    """
    # Start with base data
    result = df_base.copy()
    
    # Merge Random Forest
    result = result.merge(
        rf_importance[['CODE_KEY', 'RF_IMPORTANCE']],
        on='CODE_KEY',
        how='left'
    )
    result['RF_IMPORTANCE'] = result['RF_IMPORTANCE'].fillna(0)
    
    # Normalize RF
    max_rf = result['RF_IMPORTANCE'].max()
    result['RF_IMPORTANCE_NORM'] = result['RF_IMPORTANCE'] / max_rf if max_rf > 0 else 0
    
    # Merge XGBoost (if available)
    if xgb_importance is not None:
        result = result.merge(
            xgb_importance[['CODE_KEY', 'XGB_GAIN', 'XGB_COMBINED']],
            on='CODE_KEY',
            how='left'
        )
        result['XGB_GAIN'] = result['XGB_GAIN'].fillna(0)
        result['XGB_COMBINED'] = result['XGB_COMBINED'].fillna(0)
    else:
        result['XGB_GAIN'] = 0
        result['XGB_COMBINED'] = 0
    
    # Merge Mutual Information
    result = result.merge(
        mi_importance[['CODE_KEY', 'MI_SCORE', 'MI_SCORE_NORM']],
        on='CODE_KEY',
        how='left'
    )
    result['MI_SCORE'] = result['MI_SCORE'].fillna(0)
    result['MI_SCORE_NORM'] = result['MI_SCORE_NORM'].fillna(0)
    
    # Calculate composite ML score
    if HAS_XGBOOST:
        result['ML_COMPOSITE_SCORE'] = (
            0.35 * result['RF_IMPORTANCE_NORM'] +
            0.40 * result['XGB_COMBINED'] +
            0.25 * result['MI_SCORE_NORM']
        )
    else:
        result['ML_COMPOSITE_SCORE'] = (
            0.60 * result['RF_IMPORTANCE_NORM'] +
            0.40 * result['MI_SCORE_NORM']
        )
    
    # Sort by composite score
    result = result.sort_values('ML_COMPOSITE_SCORE', ascending=False)
    
    # Add rank
    result['ML_RANK'] = range(1, len(result) + 1)
    
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load data
    try:
        df_base, pos_total, neg_total = load_and_prepare_data(POSITIVE_FILE, NEGATIVE_FILE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Please ensure CSV files are in the '{data_dir}' directory")
        exit(1)
    
    # Create ML feature matrix
    X, y, feature_names = create_ml_feature_matrix(df_base, pos_total, neg_total)
    
    # Random Forest importance
    rf_importance, rf_auc = random_forest_importance(X, y, feature_names)
    
    # XGBoost importance
    xgb_importance, xgb_auc = xgboost_importance(X, y, feature_names)
    
    # Mutual Information importance
    mi_importance = mutual_information_importance(X, y, feature_names)
    
    # Combine all scores
    ml_results = combine_importance_scores(df_base, rf_importance, xgb_importance, mi_importance)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save full results
    output_file = os.path.join(OUTPUT_DIR, 'ml_feature_importance_12mo.csv')
    ml_results.to_csv(output_file, index=False)
    
    # Save top 250
    top_250 = ml_results.head(250)
    top_250_file = os.path.join(OUTPUT_DIR, 'TOP_250_ML_FEATURES_12mo.csv')
    top_250.to_csv(top_250_file, index=False)
    
    # Save summary
    summary = {
        'Analysis': 'ML Feature Importance',
        'Time_Window': '12 months',
        'Total_Features': len(ml_results),
        'Cancer_Patients': pos_total,
        'Control_Patients': neg_total,
        'RF_AUC': rf_auc,
        'XGB_AUC': xgb_auc if HAS_XGBOOST else 'N/A',
        'Top_Feature_1': ml_results.iloc[0]['TERM'],
        'Top_Feature_2': ml_results.iloc[1]['TERM'],
        'Top_Feature_3': ml_results.iloc[2]['TERM'],
    }
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(OUTPUT_DIR, 'ml_summary_12mo.csv')
    summary_df.to_csv(summary_file, index=False)
