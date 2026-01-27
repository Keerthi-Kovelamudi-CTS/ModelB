"""
FINAL COMBINED ANALYSIS: STATISTICAL + ML
Select TOP 50 -100 most robust predictive SNOMEDs to finalize out of top 250 that we get from this analysis
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# LOAD RESULTS
# ============================================================================

# Get script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Load results from the correct locations
stat_file = os.path.join(base_dir, 'Results2_Analysis1.3', 'Stat_analysis', 'comprehensive_analysis_12mo.csv')
ml_file = os.path.join(base_dir, 'Results2_Analysis1.3', 'ML analysis', 'ml_feature_importance_12mo.csv')

try:
    stat_df = pd.read_csv(stat_file)
    ml_df = pd.read_csv(ml_file)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    exit(1)

# Standardize columns
stat_df.columns = stat_df.columns.str.upper()
ml_df.columns = ml_df.columns.str.upper()

# Extract TERM from CODE_KEY in ML results if needed
if 'TERM' not in ml_df.columns and 'CODE_KEY' in ml_df.columns:
    ml_df['TERM'] = ml_df['CODE_KEY'].str.split('|').str[0]

# ============================================================================
# ADD RANKS
# ============================================================================

# Sort statistical by composite score or odds ratio
if 'COMPOSITE_SCORE' in stat_df.columns:
    stat_df = stat_df.sort_values('COMPOSITE_SCORE', ascending=False)
elif 'ODDS_RATIO' in stat_df.columns:
    stat_df = stat_df.sort_values('ODDS_RATIO', ascending=False)

stat_df['STAT_RANK'] = range(1, len(stat_df) + 1)

# Sort ML by composite score
if 'ML_COMPOSITE_SCORE' in ml_df.columns:
    ml_df = ml_df.sort_values('ML_COMPOSITE_SCORE', ascending=False)

ml_df['ML_RANK'] = range(1, len(ml_df) + 1)

# ============================================================================
# MERGE RESULTS
# ============================================================================

# Get ML columns to merge
ml_cols_to_merge = ['TERM', 'ML_RANK', 'ML_COMPOSITE_SCORE']
if 'CATEGORY' in ml_df.columns:
    ml_cols_to_merge.append('CATEGORY')
# Add any RF_, XGB_, MI_ columns
for col in ml_df.columns:
    if any(prefix in col for prefix in ['RF_', 'XGB_', 'MI_']):
        if col not in ml_cols_to_merge:
            ml_cols_to_merge.append(col)

# Merge on TERM
merged = stat_df.merge(
    ml_df[ml_cols_to_merge],
    on='TERM',
    how='outer',
    suffixes=('_STAT', '_ML')
)

# Handle missing ranks
max_stat = stat_df['STAT_RANK'].max()
max_ml = ml_df['ML_RANK'].max()

merged['STAT_RANK'] = merged['STAT_RANK'].fillna(max_stat + 100)
merged['ML_RANK'] = merged['ML_RANK'].fillna(max_ml + 100)

# ============================================================================
# CALCULATE COMBINED SCORES
# ============================================================================

# Combined rank (average)
merged['COMBINED_RANK'] = (merged['STAT_RANK'] + merged['ML_RANK']) / 2

# Agreement score (how close are the two rankings)
merged['RANK_DIFF'] = abs(merged['STAT_RANK'] - merged['ML_RANK'])
merged['AGREEMENT_SCORE'] = 1 / (1 + merged['RANK_DIFF'] / 10)

# Normalize scores
if 'COMPOSITE_SCORE' in merged.columns:
    max_stat_score = merged['COMPOSITE_SCORE'].max()
    merged['STAT_SCORE_NORM'] = merged['COMPOSITE_SCORE'] / max_stat_score if max_stat_score > 0 else 0
else:
    merged['STAT_SCORE_NORM'] = 1 / merged['STAT_RANK']

if 'ML_COMPOSITE_SCORE' in merged.columns:
    max_ml_score = merged['ML_COMPOSITE_SCORE'].max()
    merged['ML_SCORE_NORM'] = merged['ML_COMPOSITE_SCORE'] / max_ml_score if max_ml_score > 0 else 0
else:
    merged['ML_SCORE_NORM'] = 1 / merged['ML_RANK']

# Final combined score (weighted average)
merged['FINAL_SCORE'] = (
    0.5 * merged['STAT_SCORE_NORM'].fillna(0) +
    0.5 * merged['ML_SCORE_NORM'].fillna(0)
)

# Sort by final score
merged = merged.sort_values('FINAL_SCORE', ascending=False)
merged['FINAL_RANK'] = range(1, len(merged) + 1)

# ============================================================================
# CATEGORIZE CODES
# ============================================================================

def categorize_code(term):
    """Categorize SNOMED codes by clinical relevance."""
    if pd.isna(term):
        return 'UNKNOWN'
    
    term_lower = str(term).lower()
    
    # Core Prostate-Specific
    if any(x in term_lower for x in ['psa', 'prostate specific antigen']):
        return 'PSA'
    
    if any(x in term_lower for x in ['nocturia', 'prostatism', 'luts', 
                                      'lower urinary tract symptom']):
        return 'LUTS'
    
    if any(x in term_lower for x in ['urinary frequency', 'increased frequency of urination',
                                      'urinary urgency', 'urgent desire to urinate',
                                      'urinary retention', 'retention of urine',
                                      'hesitancy', 'delay when starting',
                                      'weak stream', 'incomplete emptying']):
        return 'LUTS'
    
    if any(x in term_lower for x in ['prostate enlarged', 'prostatic', 'bph',
                                      'benign prostatic', 'ipss', 'rectal exam']):
        return 'PROSTATE'
    
    if any(x in term_lower for x in ['haematuria', 'hematuria', 'blood in urine',
                                      'urine blood', 'frank haematuria']):
        return 'HAEMATURIA'
    
    if 'family history of prostate' in term_lower: 
        return 'RISK_FACTOR'
    
    if 'erectile' in term_lower or 'impotence' in term_lower: 
        return 'SEXUAL_FUNCTION'
    
    # Urinary Related
    if any(x in term_lower for x in ['urine', 'urinary', 'msu', 'uti', 
                                      'urinalysis', 'cystitis']):
        return 'URINARY'
    
    # General Health
    return 'GENERAL_HEALTH'

merged['CATEGORY'] = merged['TERM'].apply(categorize_code)

# ============================================================================
# HELPER FUNCTION: Clean numeric columns
# ============================================================================

def clean_numeric_columns(df):
    """
    Convert float columns to integers where appropriate (removes .0 for whole numbers).
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        # Skip non-numeric columns
        if df_clean[col].dtype == 'object':
            continue
        
        # Only process float columns
        if pd.api.types.is_float_dtype(df_clean[col]):
            # Check if all non-null values are whole numbers
            non_null = df_clean[col].dropna()
            if len(non_null) > 0:
                # Check if all values are whole numbers (within floating point tolerance)
                is_whole = (non_null % 1 == 0) | (non_null % 1 < 1e-10)
                
                if is_whole.all():
                    # Convert to nullable integer (Int64 handles NaNs)
                    try:
                        df_clean[col] = df_clean[col].astype('Int64')
                    except (ValueError, TypeError):
                        # If conversion fails, keep as float
                        pass
    
    return df_clean

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = os.path.join(base_dir, 'Results2_Analysis1.3', 'combined stat vs ml')
os.makedirs(output_dir, exist_ok=True)

# Get TOP 250
top_250 = merged.head(250).copy()

# Clean numeric columns before saving
merged_clean = clean_numeric_columns(merged)
top_250_clean = clean_numeric_columns(top_250)

# Save complete merged results
complete_file = os.path.join(output_dir, 'complete_combined_ranking.csv')
merged_clean.to_csv(complete_file, index=False)

# Save TOP 250
top_250_file = os.path.join(output_dir, 'TOP_250_COMBINED.csv')
top_250_clean.to_csv(top_250_file, index=False)
