"""
================================================================================
PROSTATE CANCER - SYMPTOM OCCURRENCE ANALYSIS (12 Mo PREDICTION)
================================================================================

PREDICTION CONTEXT:
"Based on a patient's data up to INDEX_DATE (TODAY), will they be diagnosed 
with prostate cancer in the NEXT 12 months?"

Analyzes symptom/condition occurrences over time: 
- LUTS (Lower Urinary Tract Symptoms)
- Haematuria
- Erectile Dysfunction
- DRE findings
- Any other binary (present/absent) clinical findings

KEY SETUP:
- TIME_WINDOW is relative to INDEX_DATE (not diagnosis date)
- W1 = 0-12 months before INDEX_DATE (most recent)
- W2 = 12-24 months before INDEX_DATE
- W3 = 24-36 months before INDEX_DATE
- W4 = 36-48 months before INDEX_DATE
- W5 = 48-60 months before INDEX_DATE

FEATURES INCLUDED: 
- Basic occurrence stats (NUM_OCCURRENCES, FREQUENCY_PER_YEAR)
- EVENT_AGE statistics (FIRST, LAST, MEAN)
- SNOMED_CODES list (comma-separated)
- TERMS list (pipe-separated)
- HAS_{important_code} flags
- RECENT vs EARLY comparison
- RECENCY_SCORE
- Window-based analysis (W1-W5 relative to INDEX_DATE)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_patient_guid(guid):
    """Clean PATIENT_GUID by removing extra quotes and braces."""
    if pd.isna(guid):
        return None
    guid_str = str(guid).strip()
    guid_str = guid_str.replace('"""', '').replace('{', '').replace('}', '')
    return guid_str.strip()

def parse_date(date_str):
    """Parse date string, handling various formats."""
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str)
    except: 
        return None

# ============================================================================
# FREQUENCY TREND CALCULATION
# ============================================================================

def calculate_frequency_trend(dates):
    """
    Calculate if symptom occurrences are increasing in frequency over time.
    
    Returns:
    --------
    tuple: (slope, is_worsening)
        slope: negative = intervals getting shorter = frequency increasing
        is_worsening: 1 if frequency increasing, 0 otherwise
    """
    if len(dates) < 3:
        return None, 0
    
    dates_sorted = sorted(dates)
    
    intervals = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
    
    if len(intervals) < 2:
        return None, 0
    
    interval_indices = list(range(len(intervals)))
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(interval_indices, intervals)
        is_worsening = 1 if slope < -1 else 0
        return slope, is_worsening
    except: 
        return None, 0

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_symptom_occurrences(df, symptom_name, snomed_codes, output_prefix=None,
                                 important_codes=None):
    """
    Analyze symptom/condition occurrences and trends over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing medical records
    symptom_name : str
        Name of the symptom (e.g., 'LUTS', 'Haematuria')
    snomed_codes : list
        List of SNOMED CT codes for this symptom
    output_prefix : str, optional
        Prefix for output column names
    important_codes : dict, optional
        Dictionary of {name: code} for important codes to flag
    
    Returns: 
    --------
    pandas.DataFrame
        DataFrame with symptom occurrence analysis (ONE ROW PER PATIENT)
    """
    
    if output_prefix is None:
        output_prefix = symptom_name.upper()
    
    symptom_records = df[df['SNOMED_C_T_CONCEPT_ID'].isin(snomed_codes)].copy()
    
    if len(symptom_records) == 0:
        return pd.DataFrame()
    
    symptom_records['EVENT_DATE_PARSED'] = pd.to_datetime(symptom_records['EVENT_DATE'], errors='coerce')
    symptom_records = symptom_records[symptom_records['EVENT_DATE_PARSED'].notna()].copy()
    
    if len(symptom_records) == 0:
        return pd.DataFrame()
    
    results = []
    
    for patient_guid, patient_data in symptom_records.groupby('PATIENT_GUID'):
        patient_data = patient_data.sort_values('EVENT_DATE_PARSED').reset_index(drop=True)
        
        symptom_dates = patient_data['EVENT_DATE_PARSED']
        
        num_occurrences = len(patient_data)
        first_date = symptom_dates.min()
        last_date = symptom_dates.max()
        
        time_span_days = (last_date - first_date).days
        time_span_years = time_span_days / 365.25 if time_span_days > 0 else 0
        
        frequency_per_year = num_occurrences / time_span_years if time_span_years > 0 else None
        
        event_ages = patient_data['EVENT_AGE'].dropna() if 'EVENT_AGE' in patient_data.columns else pd.Series()
        
        if len(event_ages) > 0:
            first_event_age = event_ages.iloc[0]
            last_event_age = event_ages.iloc[-1]
            mean_event_age = round(event_ages.mean(), 1)
        else:
            first_event_age = None
            last_event_age = None
            mean_event_age = None
        
        unique_codes = patient_data['SNOMED_C_T_CONCEPT_ID'].unique()
        num_unique_codes = len(unique_codes)
        
        codes_list = ','.join([str(int(c)) for c in unique_codes if pd.notna(c)])
        
        if 'TERM' in patient_data.columns:
            unique_terms = patient_data['TERM'].dropna().unique()
            terms_list = '|'.join([str(t) for t in unique_terms[:10]])
        else: 
            terms_list = None
        
        if num_occurrences > 1:
            intervals_days = [(symptom_dates.iloc[i+1] - symptom_dates.iloc[i]).days
                              for i in range(len(symptom_dates)-1)]
            mean_interval = np.mean(intervals_days)
            
            interval_slope, freq_is_worsening = calculate_frequency_trend(symptom_dates.tolist())
            
            mid_point = len(symptom_dates) // 2
            first_half = symptom_dates.iloc[:mid_point]
            second_half = symptom_dates.iloc[mid_point:]
            
            if len(first_half) > 0 and len(second_half) > 0:
                first_half_span = (first_half.max() - first_half.min()).days / 365.25
                second_half_span = (second_half.max() - second_half.min()).days / 365.25
                
                first_half_freq = len(first_half) / first_half_span if first_half_span > 0 else None
                second_half_freq = len(second_half) / second_half_span if second_half_span > 0 else None
            else:
                first_half_freq = None
                second_half_freq = None
        else: 
            mean_interval = None
            interval_slope = None
            freq_is_worsening = 0
            first_half_freq = None
            second_half_freq = None
        
        is_worsening = freq_is_worsening
        if first_half_freq is not None and second_half_freq is not None: 
            if second_half_freq > first_half_freq * 1.1:
                is_worsening = 1
        
        result = {
            'PATIENT_GUID':  patient_guid,
            f'{output_prefix}_PRESENT': 1,
            f'{output_prefix}_NUM_OCCURRENCES': num_occurrences,
            f'{output_prefix}_TIME_SPAN_YEARS': round(time_span_years, 2),
            f'{output_prefix}_FREQUENCY_PER_YEAR': round(frequency_per_year, 2) if frequency_per_year else None,
            f'{output_prefix}_FIRST_EVENT_AGE': first_event_age,
            f'{output_prefix}_LAST_EVENT_AGE': last_event_age,
            f'{output_prefix}_MEAN_EVENT_AGE': mean_event_age,
            f'{output_prefix}_NUM_UNIQUE_CODES': num_unique_codes,
            f'{output_prefix}_SNOMED_CODES': codes_list,
            f'{output_prefix}_TERMS': terms_list,
            f'{output_prefix}_MEAN_INTERVAL_DAYS': round(mean_interval, 1) if mean_interval else None,
            f'{output_prefix}_FREQUENCY_SLOPE': round(interval_slope, 2) if interval_slope else None,
            f'{output_prefix}_IS_WORSENING': is_worsening,
            f'{output_prefix}_FIRST_HALF_FREQ': round(first_half_freq, 2) if first_half_freq else None,
            f'{output_prefix}_SECOND_HALF_FREQ': round(second_half_freq, 2) if second_half_freq else None,
        }
        
        if important_codes:
            for code_name, code_value in important_codes.items():
                has_code = 1 if code_value in unique_codes else 0
                result[f'{output_prefix}_HAS_{code_name}'] = has_code
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    return results_df

def analyze_symptom_by_window(df, symptom_name, snomed_codes, output_prefix=None):
    """
    Analyze symptom occurrences separately for each time window (W1-W5).
    
    Returns:
    --------
    pandas.DataFrame
        Window-based features for each patient (ONE ROW PER PATIENT)
    """
    if output_prefix is None:
        output_prefix = symptom_name.upper()
    
    symptom_records = df[df['SNOMED_C_T_CONCEPT_ID'].isin(snomed_codes)].copy()
    
    all_patients = df['PATIENT_GUID'].unique()
    
    results = []
    
    for patient_guid in all_patients:
        patient_data = symptom_records[symptom_records['PATIENT_GUID'] == patient_guid]
        
        result = {'PATIENT_GUID': patient_guid}
        
        result[f'{output_prefix}_EVER_PRESENT'] = 1 if len(patient_data) > 0 else 0
        result[f'{output_prefix}_TOTAL_COUNT'] = len(patient_data)
        
        window_counts = {}
        for window in ['W1', 'W2', 'W3', 'W4', 'W5']:
            window_data = patient_data[patient_data['TIME_WINDOW'] == window]
            count = len(window_data)
            window_counts[window] = count
            
            result[f'{output_prefix}_{window}_PRESENT'] = 1 if count > 0 else 0
            result[f'{output_prefix}_{window}_COUNT'] = count
        
        w5_count = window_counts['W5']
        w4_count = window_counts['W4']
        w3_count = window_counts['W3']
        w2_count = window_counts['W2']
        w1_count = window_counts['W1']
        
        recent_count = w1_count + w2_count
        early_count = w4_count + w5_count
        
        result[f'{output_prefix}_RECENT_COUNT'] = recent_count
        result[f'{output_prefix}_EARLY_COUNT'] = early_count
        
        if early_count > 0:
            result[f'{output_prefix}_RECENT_VS_EARLY_RATIO'] = round(recent_count / early_count, 2)
        elif recent_count > 0:
            result[f'{output_prefix}_RECENT_VS_EARLY_RATIO'] = 10.0
        else: 
            result[f'{output_prefix}_RECENT_VS_EARLY_RATIO'] = None
        
        result[f'{output_prefix}_ACCELERATING'] = 1 if recent_count > early_count * 1.5 else 0
        
        early_present = w5_count > 0 or w4_count > 0
        recent_present = w1_count > 0 or w2_count > 0
        result[f'{output_prefix}_NEW_IN_RECENT'] = 1 if recent_present and not early_present else 0
        
        result[f'{output_prefix}_INCREASING'] = 1 if recent_count > early_count else 0
        
        windows_present = sum([1 if window_counts[w] > 0 else 0 for w in ['W1', 'W2', 'W3', 'W4', 'W5']])
        result[f'{output_prefix}_WINDOWS_PRESENT'] = windows_present
        result[f'{output_prefix}_PERSISTENT'] = 1 if windows_present >= 3 else 0
        
        recency_score = (w1_count * 5) + (w2_count * 4) + (w3_count * 3) + (w4_count * 2) + (w5_count * 1)
        result[f'{output_prefix}_RECENCY_SCORE'] = recency_score
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    return results_df
