"""
================================================================================
PROSTATE CANCER - MAIN RISK FACTOR ANALYSIS (FORWARD-LOOKING PREDICTION)
================================================================================

PREDICTION TASK:
"Based on a patient's data up to TODAY, will they be diagnosed with 
prostate cancer in the NEXT 12 months?"

This is the main script that:
1. Loads CSV data from SQL queries (with INDEX_DATE column)
2. Analyzes ALL 107 SNOMED codes (values and symptoms)
3. Merges all features into patient-level matrix
4. Adds CANCER_FLAG (1 = diagnosed within 12 months after INDEX_DATE)
5. Outputs final dataset for ML training

KEY SETUP:
- INDEX_DATE = DATE_OF_DIAGNOSIS - 12 months (for cancer patients)
- INDEX_DATE = matched date (for control patients)
- All events must have EVENT_DATE <= INDEX_DATE (no leakage)
- TIME_WINDOW is relative to INDEX_DATE (W1 = 0-12 months before INDEX_DATE)

FEATURES INCLUDED:
- EVENT_AGE statistics (FIRST, LAST, MEAN)
- SNOMED_CODES list (for analysis)
- TERMS list (for clinical interpretation)
- HAS_{important_code} flags for key clinical codes
- RECENT vs EARLY comparison
- RECENCY_SCORE
- PSA-specific features (velocity, doubling time, thresholds)
- Window-based analysis (W1-W5 relative to INDEX_DATE)

Usage:
    python prostate_risk_factors_2.py

Input Files (must include INDEX_DATE column):
    - prostate_cancer_events_windowed-2.csv
    - prostate_control_events_windowed-2.csv

Output Files:
    - prostate_cancer_features_final.csv (ready for ML)
"""

import pandas as pd
import numpy as np
import warnings
import os
import gc

warnings.filterwarnings('ignore')

from value_trend_analysis_2 import (
    analyze_value_trends,
    analyze_values_by_window,
    clean_patient_guid,
    psa_trend_logic,
    egfr_trend_logic,
    haemoglobin_trend_logic,
    alp_trend_logic,
    mcv_trend_logic,
    default_trend_logic
)

from symptom_occurrence_analysis_2 import (
    analyze_symptom_occurrences,
    analyze_symptom_by_window
)

# ============================================================================
# ALL 107 SNOMED CODES ORGANIZED BY CATEGORY
# ============================================================================

SNOMED_CODES = {
    'PSA':  {
        'codes': [
            1030791000000100,
            1000381000000105,
            396152005,
            166160000,
            1030021000000101,
            1000481000000100,
            1006591000000104,
        ],
        'has_value':  True,
        'trend_direction': 'higher_is_worse',
        'is_psa':  True
    },
    
    'DRE': {
        'codes': [
            410007005,
            275302008,
            274296009,
            909491000006106,
            271302001,
            801000119105,
        ],
        'has_value':  False,
        'trend_direction':  None
    },
    
    'LUTS': {
        'codes': [
            11441004,
            249274008,
            981411000006101,
            307541003,
            5972002,
            267064002,
            1726491000006101,
            162116003,
            763111000000108,
            75088002,
            236648008,
            49650001,
            300471006,
        ],
        'has_value':  False,
        'trend_direction':  None
    },
    
    'HAEMATURIA': {
        'codes': [
            1038641000000104,
            999131000000106,
            34436003,
            167302009,
            167300001,
            197938001,
            1041881000000101,
            167301002,
            197941005,
        ],
        'has_value':  False,
        'trend_direction': None
    },
    
    'ERECTILE_DYSFUNCTION': {
        'codes': [
            397803000,
            860914002,
        ],
        'has_value':  False,
        'trend_direction': None
    },
    
    'BPH': {
        'codes': [266569009],
        'has_value': False,
        'trend_direction': None
    },
    
    'CHRONIC_RETENTION': {
        'codes': [236650000],
        'has_value':  False,
        'trend_direction': None
    },
    
    'FAMILY_HISTORY_PROSTATE': {
        'codes': [414205003],
        'has_value': False,
        'trend_direction':  None
    },
    
    'EGFR': {
        'codes': [
            1020291000000106,
            80274001,
            1011481000000105,
        ],
        'has_value':  True,
        'trend_direction':  'lower_is_worse'
    },
    
    'CREATININE': {
        'codes': [
            1000731000000107,
            1001011000000107,
        ],
        'has_value': True,
        'trend_direction': 'higher_is_worse'
    },
    
    'UREA': {
        'codes':  [1028131000000104],
        'has_value': True,
        'trend_direction': None
    },
    
    'CREATINE_KINASE': {
        'codes': [1000931000000105],
        'has_value':  True,
        'trend_direction': None
    },
    
    'ELECTROLYTES_GENERAL': {
        'codes': [1000641000000106],
        'has_value': True,
        'trend_direction': None
    },
    
    'ALT': {
        'codes': [1018251000000107],
        'has_value': True,
        'trend_direction': None
    },
    
    'LIVER_FUNCTION_TEST': {
        'codes': [997531000000108],
        'has_value': False,
        'trend_direction':  None
    },
    
    'BILIRUBIN': {
        'codes': [
            999691000000104,
            997561000000103,
            997591000000109,
            1026761000000106,
        ],
        'has_value': True,
        'trend_direction': None
    },
    
    'ALP': {
        'codes':  [
            1013041000000106,
            1000621000000104,
        ],
        'has_value': True,
        'trend_direction': 'higher_is_worse'
    },
    
    'SODIUM': {
        'codes':  [
            1000661000000107,
            1017381000000106,
        ],
        'has_value': True,
        'trend_direction': None
    },
    
    'POTASSIUM':  {
        'codes': [1000651000000109],
        'has_value': True,
        'trend_direction': None
    },
    
    'CALCIUM': {
        'codes': [
            935051000000108,
            1000691000000101,
        ],
        'has_value': True,
        'trend_direction':  None
    },
    
    'HAEMOGLOBIN': {
        'codes': [1022431000000105],
        'has_value': True,
        'trend_direction': 'lower_is_worse'
    },
    
    'MCV': {
        'codes':  [1022491000000106],
        'has_value': True,
        'trend_direction': 'lower_is_worse'
    },
    
    'FBC_GENERAL': {
        'codes': [1022441000000101],
        'has_value': False,
        'trend_direction':  None
    },
    
    'WBC': {
        'codes':  [1022541000000102],
        'has_value': True,
        'trend_direction': None
    },
    
    'NEUTROPHIL': {
        'codes': [1022551000000104],
        'has_value': True,
        'trend_direction': None
    },
    
    'LYMPHOCYTE': {
        'codes': [1022581000000105],
        'has_value': True,
        'trend_direction': None
    },
    
    'MONOCYTE': {
        'codes': [1022591000000107],
        'has_value': True,
        'trend_direction': None
    },
    
    'EOSINOPHIL': {
        'codes': [1022561000000101],
        'has_value': True,
        'trend_direction': None
    },
    
    'BASOPHIL': {
        'codes': [1022571000000108],
        'has_value': True,
        'trend_direction': None
    },
    
    'PLATELET': {
        'codes': [1022651000000100],
        'has_value': True,
        'trend_direction': None
    },
    
    'RBC': {
        'codes':  [1022451000000103],
        'has_value': True,
        'trend_direction': None
    },
    
    'MCH': {
        'codes':  [1022471000000107],
        'has_value':  True,
        'trend_direction': None
    },
    
    'MCHC': {
        'codes': [1022481000000109],
        'has_value': True,
        'trend_direction': None
    },
    
    'HAEMATOCRIT': {
        'codes': [1022291000000105],
        'has_value': True,
        'trend_direction': None
    },
    
    'RDW': {
        'codes': [993501000000105],
        'has_value': True,
        'trend_direction': None
    },
    
    'ESR': {
        'codes':  [1022511000000103],
        'has_value': True,
        'trend_direction': None
    },
    
    'LARGE_UNSTAINED_CELLS': {
        'codes': [993551000000106],
        'has_value': True,
        'trend_direction': None
    },
    
    'CHOLESTEROL':  {
        'codes': [1005671000000105],
        'has_value':  True,
        'trend_direction': None
    },
    
    'HDL': {
        'codes':  [
            1005681000000107,
            1010581000000101,
        ],
        'has_value': True,
        'trend_direction': None
    },
    
    'LDL': {
        'codes': [1022191000000100],
        'has_value': True,
        'trend_direction': None
    },
    
    'TRIGLYCERIDES': {
        'codes': [1005691000000109],
        'has_value': True,
        'trend_direction':  None
    },
    
    'NON_HDL':  {
        'codes': [
            1006191000000106,
            1030411000000101,
        ],
        'has_value':  True,
        'trend_direction': None
    },
    
    'FASTING_LIPIDS': {
        'codes': [854781000006103],
        'has_value': False,
        'trend_direction':  None
    },
    
    'GLUCOSE':  {
        'codes': [
            1010671000000102,
            1003141000000105,
        ],
        'has_value': True,
        'trend_direction': None
    },
    
    'HBA1C': {
        'codes': [999791000000106],
        'has_value': True,
        'trend_direction': None
    },
    
    'ALBUMIN': {
        'codes':  [1000821000000103],
        'has_value': True,
        'trend_direction': None
    },
    
    'TOTAL_PROTEIN': {
        'codes': [1000811000000109],
        'has_value': True,
        'trend_direction': None
    },
    
    'GLOBULIN': {
        'codes':  [1001231000000108],
        'has_value': True,
        'trend_direction': None
    },
    
    'INORGANIC_PHOSPHATE': {
        'codes': [1000701000000101],
        'has_value':  True,
        'trend_direction': None
    },
    
    'URINE_PROTEIN_NEGATIVE': {
        'codes': [167273002],
        'has_value': False,
        'trend_direction':  None
    },
    
    'URINE_CULTURE': {
        'codes': [1023711000000100],
        'has_value': False,
        'trend_direction': None
    },
    
    'URINE_MICROSCOPY': {
        'codes':  [1014831000000107],
        'has_value': False,
        'trend_direction': None
    },
    
    'URINE_DIPSTICK': {
        'codes': [1007881000000101],
        'has_value': False,
        'trend_direction': None
    },
    
    'URINALYSIS_NORMAL': {
        'codes': [167221003],
        'has_value': False,
        'trend_direction': None
    },
    
    'URINE_LEUCOCYTES': {
        'codes': [999291000000102],
        'has_value': False,
        'trend_direction': None
    },
    
    'URINE_CASTS': {
        'codes':  [365689007],
        'has_value':  False,
        'trend_direction': None
    },
    
    'EPITHELIAL_CELLS': {
        'codes': [992871000000105],
        'has_value': True,
        'trend_direction': None
    },
    
    'ORGANISM_COUNT': {
        'codes': [992821000000106],
        'has_value': True,
        'trend_direction': None
    },
    
    'TSH': {
        'codes':  [1022791000000101],
        'has_value': True,
        'trend_direction': None
    },
    
    'FREE_T4': {
        'codes': [1016971000000106],
        'has_value': True,
        'trend_direction': None
    },
    
    'THYROID_FUNCTION_TEST': {
        'codes': [1016851000000107],
        'has_value': False,
        'trend_direction': None
    },
    
    'CRP': {
        'codes':  [
            1001371000000100,
            999651000000107,
        ],
        'has_value': True,
        'trend_direction': None
    },
}

# ============================================================================
# CLINICALLY IMPORTANT SNOMED CODES
# ============================================================================

IMPORTANT_SNOMED_CODES = {
    'PSA': {
        'RAISED_PSA': 396152005,
        'PSA_ABNORMAL': 166160000,
    },
    
    'DRE': {
        'PROSTATE_ENLARGED': 275302008,
        'PROSTATIC_SWELLING': 271302001,
        'ABNORMALITY_DETECTED': 801000119105,
    },
    
    'LUTS': {
        'ACUTE_RETENTION': 236648008,
        'RETENTION':  267064002,
        'PROSTATISM': 11441004,
    },
    
    'HAEMATURIA':  {
        'FRANK_HAEMATURIA': 197941005,
        'PAINLESS_HAEMATURIA': 197938001,
        'BLOOD_3PLUS': 167302009,
    },
    
    'FAMILY_HISTORY_PROSTATE': {
        'FH_PROSTATE_CANCER': 414205003,
    },
    
    'BPH': {
        'BPH':  266569009,
    },
    
    'CHRONIC_RETENTION': {
        'CHRONIC_RETENTION': 236650000,
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_trend_logic(marker_name, trend_direction, is_psa=False):
    """Map trend_direction string to appropriate trend_logic function."""
    if is_psa: 
        return psa_trend_logic
    
    if trend_direction == 'higher_is_worse':
        if marker_name == 'ALP':
            return alp_trend_logic
        else:
            return default_trend_logic
    elif trend_direction == 'lower_is_worse':
        if marker_name == 'EGFR':
            return egfr_trend_logic
        elif marker_name in ['HAEMOGLOBIN', 'HB']:
            return haemoglobin_trend_logic
        elif marker_name == 'MCV':
            return mcv_trend_logic
        else:
            return default_trend_logic
    else:
        return default_trend_logic

def validate_data(df, cohort_name):
    """Validate loaded data quality for forward-looking prediction."""
    issues = []
    
    required_cols = ['PATIENT_GUID', 'SNOMED_C_T_CONCEPT_ID', 'EVENT_DATE', 'TIME_WINDOW']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    if 'INDEX_DATE' not in df.columns:
        issues.append("ERROR: INDEX_DATE column not found! Required for forward-looking prediction.")
    
    if 'PATIENT_GUID' in df.columns:
        null_guids = df['PATIENT_GUID'].isnull().sum()
        if null_guids > 0:
            issues.append(f"Null PATIENT_GUIDs: {null_guids}")
    
    if 'TIME_WINDOW' in df.columns and 'INDEX_DATE' in df.columns and 'EVENT_DATE' in df.columns:
        df_check = df[['PATIENT_GUID', 'EVENT_DATE', 'INDEX_DATE']].drop_duplicates()
        df_check['INDEX_DATE_PARSED'] = pd.to_datetime(df_check['INDEX_DATE'], errors='coerce')
        df_check['EVENT_DATE_PARSED'] = pd.to_datetime(df_check['EVENT_DATE'], errors='coerce')
        
        events_after_index = df_check[
            (df_check['EVENT_DATE_PARSED'].notna()) & 
            (df_check['INDEX_DATE_PARSED'].notna()) &
            (df_check['EVENT_DATE_PARSED'] > df_check['INDEX_DATE_PARSED'])
        ]
        
        if len(events_after_index) > 0:
            issues.append(f"CRITICAL LEAKAGE: Found {len(events_after_index)} events AFTER INDEX_DATE!")
    
    if issues:
        for issue in issues: 
            print(f"ERROR: {issue}")
    
    return len(issues) == 0

def get_patient_demographics(df):
    """Extract unique patient demographics - ONE ROW PER PATIENT."""
    demo_cols = ['PATIENT_GUID', 'SEX', 'PATIENT_ETHNICITY', 'CANCER_ID',
                 'DATE_OF_DIAGNOSIS', 'AGE_AT_DIAGNOSIS', 'AGE_AT_INDEX', 'INDEX_DATE']
    
    existing_cols = [c for c in demo_cols if c in df.columns]
    demographics = df[existing_cols].groupby('PATIENT_GUID').first().reset_index()
    
    if 'AGE_AT_INDEX' in demographics.columns:
        demographics['AGE'] = demographics['AGE_AT_INDEX']
    elif 'AGE_AT_DIAGNOSIS' in demographics.columns:
        demographics['AGE'] = demographics['AGE_AT_DIAGNOSIS']
    elif 'EVENT_AGE' in df.columns:
        age_df = df.groupby('PATIENT_GUID')['EVENT_AGE'].max().reset_index()
        age_df.columns = ['PATIENT_GUID', 'AGE']
        demographics = demographics.merge(age_df, on='PATIENT_GUID', how='left')
    
    if demographics['PATIENT_GUID'].duplicated().any():
        demographics = demographics.drop_duplicates(subset=['PATIENT_GUID'], keep='first')
    
    return demographics

# ============================================================================
# INTELLIGENT AGGREGATION FUNCTIONS
# ============================================================================

def prioritize_trend(values):
    """For trend columns, prioritize the most concerning value."""
    priority_order = [
        'Worsening',
        'Increasing frequency (worsening)',
        'Increasing',
        'Stable',
        'Stable frequency',
        'Decreasing',
        'Decreasing frequency (improving)',
        'Improving',
        'Unknown',
        'Insufficient data',
        'Unable to calculate',
        'Single occurrence'
    ]
    
    values_list = [v for v in values if pd.notna(v)]
    
    if not values_list:
        return 'Unknown'
    
    for priority in priority_order:
        if priority in values_list: 
            return priority
    
    return values_list[0]

def prioritize_trend_series(series):
    """Wrapper for prioritize_trend that works with pandas Series."""
    values_list = series.dropna().tolist()
    return prioritize_trend(values_list)

def aggregate_duplicate_rows(df, on_column='PATIENT_GUID'):
    """
    Aggregates duplicate rows intelligently based on column type/name.
    
    Aggregation rules:
    - Counts/flags (_COUNT, _PRESENT, _IS_): takes MAX
    - Averages (_MEAN, _AVG): takes MEAN of means
    - Maximums (_MAX): takes MAX
    - Minimums (_MIN): takes MIN
    - First values (_FIRST): takes FIRST
    - Last values (_LAST): takes LAST
    - Sums (_SUM, _TOTAL): takes SUM
    - Trend columns: prioritizes "Worsening"
    - String columns: takes FIRST non-null
    - Other numeric: takes MEAN
    """
    
    if on_column not in df.columns:
        return df
    
    if not df[on_column].duplicated().any():
        return df
    
    agg_dict = {}
    
    for col in df.columns:
        if col == on_column:
            continue
        
        col_upper = col.upper()
        
        if any(pattern in col_upper for pattern in ['_COUNT', '_NUM_', '_PRESENT', '_IS_', '_FLAG', '_ABOVE', '_WINDOWS_PRESENT', '_HAS_']):
            agg_dict[col] = 'max'
        elif any(pattern in col_upper for pattern in ['_MEAN', '_AVG']):
            agg_dict[col] = 'mean'
        elif '_MAX' in col_upper:
            agg_dict[col] = 'max'
        elif '_MIN' in col_upper:
            agg_dict[col] = 'min'
        elif '_FIRST' in col_upper:
            agg_dict[col] = 'first'
        elif '_LAST' in col_upper:
            agg_dict[col] = 'last'
        elif '_SUM' in col_upper or '_TOTAL' in col_upper:
            agg_dict[col] = 'sum'
        elif any(pattern in col_upper for pattern in ['_STD', '_VAR', '_SLOPE', '_VELOCITY', '_R_VALUE', '_RATIO', '_SCORE']):
            agg_dict[col] = 'mean'
        elif any(pattern in col_upper for pattern in ['_TREND', '_DIRECTION', '_STATUS']):
            agg_dict[col] = prioritize_trend_series
        elif '_DATE' in col_upper or '_CODES' in col_upper or '_TERMS' in col_upper: 
            agg_dict[col] = 'first'
        elif df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            agg_dict[col] = 'first'
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'
    
    try:
        aggregated = df.groupby(on_column, as_index=False).agg(agg_dict)
        return aggregated
    except Exception as e:
        print(f"ERROR: Aggregation failed: {e}")
        return df.drop_duplicates(subset=[on_column], keep='first')

def merge_feature_dataframes(feature_dfs, on_column='PATIENT_GUID'):
    """Merge multiple feature DataFrames on patient GUID."""
    non_empty_dfs = [df for df in feature_dfs if len(df) > 0]
    
    if len(non_empty_dfs) == 0:
        return pd.DataFrame()
    
    if len(non_empty_dfs) == 1:
        df = non_empty_dfs[0]
        if df[on_column].duplicated().any():
            return aggregate_duplicate_rows(df, on_column)
        return df
    
    cleaned_dfs = []
    
    for i, df in enumerate(non_empty_dfs):
        if on_column not in df.columns:
            continue
        
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        
        if df[on_column].duplicated().any():
            dup_guids = df[df[on_column].duplicated(keep=False)][on_column].unique()
            
            all_identical = True
            for guid in dup_guids[:5]:
                dup_rows = df[df[on_column] == guid]
                if len(dup_rows.drop(columns=[on_column]).drop_duplicates()) > 1:
                    all_identical = False
                    break
            
            if all_identical:
                df = df.drop_duplicates(subset=[on_column], keep='first')
            else:
                df = aggregate_duplicate_rows(df, on_column)
        
        cleaned_dfs.append(df)
        gc.collect()
    
    if len(cleaned_dfs) == 0:
        return pd.DataFrame()
    
    if len(cleaned_dfs) == 1:
        return cleaned_dfs[0]
    
    merged = cleaned_dfs[0]
    
    for i, df in enumerate(cleaned_dfs[1:], 1):
        merged = pd.merge(merged, df, on=on_column, how='outer')
        gc.collect()
    
    if merged[on_column].duplicated().any():
        merged = aggregate_duplicate_rows(merged, on_column)
    
    return merged

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def extract_prostate_risk_factors(df, is_cancer_cohort=True):
    """
    Extract all prostate cancer risk factors from the data.
    
    Returns: 
    --------
    pandas.DataFrame
        Patient-level feature matrix (ONE ROW PER PATIENT)
    """
    
    df['PATIENT_GUID'] = df['PATIENT_GUID'].apply(clean_patient_guid)
    df['SNOMED_C_T_CONCEPT_ID'] = pd.to_numeric(df['SNOMED_C_T_CONCEPT_ID'], errors='coerce')
    
    feature_dfs = []
    
    demographics = get_patient_demographics(df)
    feature_dfs.append(demographics)
    
    total_categories = len(SNOMED_CODES)
    processed = 0
    
    for category_name, config in SNOMED_CODES.items():
        processed += 1
        codes = config['codes']
        has_value = config.get('has_value', False)
        trend_direction = config.get('trend_direction', None)
        is_psa = config.get('is_psa', False)
        
        category_records = df[df['SNOMED_C_T_CONCEPT_ID'].isin(codes)]
        if len(category_records) == 0:
            continue
        
        important_codes_for_cat = IMPORTANT_SNOMED_CODES.get(category_name, None)
        
        if has_value:
            trend_logic_func = get_trend_logic(category_name, trend_direction, is_psa)
            
            value_results = analyze_value_trends(
                df=df,
                marker_name=category_name,
                snomed_codes=codes,
                trend_logic=trend_logic_func,
                is_psa=is_psa,
                output_prefix=category_name,
                important_codes=important_codes_for_cat
            )
            
            if len(value_results) > 0:
                if value_results['PATIENT_GUID'].duplicated().any():
                    value_results = aggregate_duplicate_rows(value_results, 'PATIENT_GUID')
                feature_dfs.append(value_results)
            
            window_results = analyze_values_by_window(
                df=df,
                marker_name=category_name,
                snomed_codes=codes,
                trend_logic=trend_logic_func,
                is_psa=is_psa,
                output_prefix=category_name
            )
            
            if len(window_results) > 0:
                if window_results['PATIENT_GUID'].duplicated().any():
                    window_results = aggregate_duplicate_rows(window_results, 'PATIENT_GUID')
                
                existing_cols = set()
                for existing_df in feature_dfs:
                    existing_cols.update(existing_df.columns)
                
                new_cols = ['PATIENT_GUID'] + [c for c in window_results.columns
                                                if c not in existing_cols or c == 'PATIENT_GUID']
                window_results = window_results[new_cols]
                
                if len(window_results.columns) > 1:
                    feature_dfs.append(window_results)
        
        else:
            occurrence_results = analyze_symptom_occurrences(
                df=df,
                symptom_name=category_name,
                snomed_codes=codes,
                output_prefix=category_name,
                important_codes=important_codes_for_cat
            )
            
            if len(occurrence_results) > 0:
                if occurrence_results['PATIENT_GUID'].duplicated().any():
                    occurrence_results = aggregate_duplicate_rows(occurrence_results, 'PATIENT_GUID')
                feature_dfs.append(occurrence_results)
            
            window_results = analyze_symptom_by_window(
                df=df,
                symptom_name=category_name,
                snomed_codes=codes,
                output_prefix=category_name
            )
            
            if len(window_results) > 0:
                if window_results['PATIENT_GUID'].duplicated().any():
                    window_results = aggregate_duplicate_rows(window_results, 'PATIENT_GUID')
                
                existing_cols = set()
                for existing_df in feature_dfs: 
                    existing_cols.update(existing_df.columns)
                
                new_cols = ['PATIENT_GUID'] + [c for c in window_results.columns
                                               if c not in existing_cols or c == 'PATIENT_GUID']
                window_results = window_results[new_cols]
                
                if len(window_results.columns) > 1:
                    feature_dfs.append(window_results)
        
        if processed % 10 == 0:
            gc.collect()
    
    merged = merge_feature_dataframes(feature_dfs, on_column='PATIENT_GUID')
    
    return merged

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete analysis."""
    
    CANCER_FILE = '/Users/keerthikovelamudi/Downloads/prostate_cancer_events_windowed-2.csv'
    CONTROL_FILE = '/Users/keerthikovelamudi/Downloads/prostate_control_events_windowed-2.csv'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FILE = os.path.join(script_dir, 'prostate_cancer_features_final.csv')
    
    if not os.path.exists(CANCER_FILE):
        print(f"ERROR: File not found: {CANCER_FILE}")
        return
    
    cancer_df = pd.read_csv(CANCER_FILE)
    
    validate_data(cancer_df, "CANCER")
    
    if 'INDEX_DATE' in cancer_df.columns and 'DATE_OF_DIAGNOSIS' in cancer_df.columns:
        cancer_patients = cancer_df[['PATIENT_GUID', 'INDEX_DATE', 'DATE_OF_DIAGNOSIS']].drop_duplicates()
        cancer_patients['INDEX_DATE_PARSED'] = pd.to_datetime(cancer_patients['INDEX_DATE'], errors='coerce')
        cancer_patients['DIAGNOSIS_PARSED'] = pd.to_datetime(cancer_patients['DATE_OF_DIAGNOSIS'], errors='coerce')
        cancer_patients['EXPECTED_INDEX'] = cancer_patients['DIAGNOSIS_PARSED'] - pd.DateOffset(months=12)
        cancer_patients['DIFF_DAYS'] = (cancer_patients['INDEX_DATE_PARSED'] - cancer_patients['EXPECTED_INDEX']).dt.days
        
        incorrect_index = cancer_patients[abs(cancer_patients['DIFF_DAYS']) > 5]
        if len(incorrect_index) > 0:
            print(f"WARNING: {len(incorrect_index)} cancer patients have INDEX_DATE not ~12 months before diagnosis")
        
        cancer_df_check = cancer_df.copy()
        cancer_df_check['INDEX_DATE_PARSED'] = pd.to_datetime(cancer_df_check['INDEX_DATE'], errors='coerce')
        cancer_df_check['EVENT_DATE_PARSED'] = pd.to_datetime(cancer_df_check['EVENT_DATE'], errors='coerce')
        
        cancer_df_check = cancer_df_check.merge(
            cancer_patients[['PATIENT_GUID', 'INDEX_DATE_PARSED']], 
            on='PATIENT_GUID', 
            how='left',
            suffixes=('', '_patient')
        )
        
        events_after_index = cancer_df_check[
            (cancer_df_check['EVENT_DATE_PARSED'].notna()) & 
            (cancer_df_check['INDEX_DATE_PARSED'].notna()) &
            (cancer_df_check['EVENT_DATE_PARSED'] > cancer_df_check['INDEX_DATE_PARSED'])
        ]
        
        if len(events_after_index) > 0:
            print(f"ERROR: {len(events_after_index)} events found AFTER INDEX_DATE!")
    
    cancer_features = extract_prostate_risk_factors(cancer_df, is_cancer_cohort=True)
    cancer_features['CANCER_FLAG'] = 1
    
    del cancer_df
    gc.collect()
    
    if not os.path.exists(CONTROL_FILE):
        print(f"ERROR: File not found: {CONTROL_FILE}")
        return
    
    control_df = pd.read_csv(CONTROL_FILE)
    
    validate_data(control_df, "CONTROL")
    
    control_features = extract_prostate_risk_factors(control_df, is_cancer_cohort=False)
    control_features['CANCER_FLAG'] = 0
    
    del control_df
    gc.collect()
    
    all_columns = set(cancer_features.columns) | set(control_features.columns)
    
    for col in all_columns:
        if col not in cancer_features.columns:
            cancer_features[col] = None
        if col not in control_features.columns:
            control_features[col] = None
    
    cancer_features = cancer_features[sorted(cancer_features.columns)]
    control_features = control_features[sorted(control_features.columns)]
    
    final_df = pd.concat([cancer_features, control_features], ignore_index=True)
    
    cols = [c for c in final_df.columns if c != 'CANCER_FLAG'] + ['CANCER_FLAG']
    final_df = final_df[cols]
    
    del cancer_features, control_features
    gc.collect()
    
    final_df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
