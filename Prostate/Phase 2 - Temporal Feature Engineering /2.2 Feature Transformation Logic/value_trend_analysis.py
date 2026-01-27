"""
================================================================================
PROSTATE CANCER - VALUE TREND ANALYSIS (12 Mo PREDICTION)
================================================================================

PREDICTION CONTEXT:
"Based on a patient's data up to INDEX_DATE (TODAY), will they be diagnosed 
with prostate cancer in the NEXT 12 months?"

Analyzes numeric values over time for lab markers: 
- PSA (with velocity, doubling time, age-specific thresholds)
- eGFR (kidney function - falling is concerning)
- Haemoglobin (anemia from bleeding - falling is concerning)
- Alkaline Phosphatase (bone metastasis - rising is concerning)
- All other lab values with numeric results

KEY SETUP:
- TIME_WINDOW is relative to INDEX_DATE (not diagnosis date)
- W1 = 0-12 months before INDEX_DATE (most recent)
- W2 = 12-24 months before INDEX_DATE
- W3 = 24-36 months before INDEX_DATE
- W4 = 36-48 months before INDEX_DATE
- W5 = 48-60 months before INDEX_DATE

OPTIMIZED: 
- Removed redundant features (MEDIAN, MIN_EVENT_AGE, MAX_EVENT_AGE)
- Removed string columns (dates kept only as reference, not for ML)
- Added EVENT_AGE statistics (FIRST, LAST, MEAN only)
- Added SNOMED code count
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

def parse_value(value_str):
    """
    Parse VALUE column, handling quotes, units, and converting to numeric.
    Enhanced to handle HealthGorilla format with units (e.g., "137.0 mmol/L")
    """
    if pd.isna(value_str):
        return None
    try:
        value_clean = str(value_str).strip().replace('"""', '').replace('"', '').strip()
        
        if value_clean == '' or value_clean.lower() in ['nan', 'none', 'null', 'na', '']:
            return None
        
        import re
        units_pattern = r'\s*(?:mg/dL|mmol/L|g/dL|ng/mL|U/L|IU/L|mL/min|pg|fL|%|mmHg|bpm|cm|kg|lb|°F|°C)\s*$'
        value_clean = re.sub(units_pattern, '', value_clean, flags=re.IGNORECASE)
        
        if '-' in value_clean and not value_clean.startswith('-'):
            parts = value_clean.split('-')
            if len(parts) >= 2 and parts[0].strip():
                value_clean = parts[0].strip()
        
        value_clean = value_clean.replace('>', '').replace('<', '').replace('≥', '').replace('≤', '').strip()
        value_clean = value_clean.replace(',', '.')
        value_clean = re.sub(r'[^\d.-]', '', value_clean)
        
        if value_clean:
            return float(value_clean)
        else:
            return None
            
    except (ValueError, TypeError):
        return None

def parse_date(date_str):
    """Parse date string, handling various formats."""
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str)
    except: 
        return None

# ============================================================================
# TREND CALCULATION FUNCTIONS
# ============================================================================

def calculate_trend(values, dates):
    """
    Calculate linear trend for values over time.
    
    Returns:
    --------
    tuple: (slope, r_value, slope_per_year)
        slope is per day; slope_per_year = slope * 365.25
    """
    if len(values) < 2:
        return None, None, None
    
    dates_numeric = [(d - dates.iloc[0]).days for d in dates]
    
    if len(set(dates_numeric)) < 2:
        return None, None, None
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
        slope_per_year = slope * 365.25
        return slope, r_value, slope_per_year
    except Exception: 
        return None, None, None

def calculate_psa_doubling_time(values, dates):
    """
    Calculate PSA doubling time in months.
    
    PSA doubling time = ln(2) / slope (where slope is from log-transformed values)
    """
    if len(values) < 2:
        return None
    
    valid_data = [(v, d) for v, d in zip(values, dates) if v > 0]
    if len(valid_data) < 2:
        return None
    
    values_clean = [v for v, d in valid_data]
    dates_clean = pd.Series([d for v, d in valid_data])
    
    log_values = np.log(values_clean)
    dates_numeric = [(d - dates_clean.iloc[0]).days for d in dates_clean]
    
    if len(set(dates_numeric)) < 2:
        return None
    
    try: 
        slope, _, _, _, _ = stats.linregress(dates_numeric, log_values)
        if slope > 0:
            doubling_time_days = np.log(2) / slope
            doubling_time_months = doubling_time_days / 30.44
            if doubling_time_months > 0 and doubling_time_months < 600:
                return round(doubling_time_months, 1)
        return None
    except Exception: 
        return None

def get_age_specific_psa_threshold(age):
    """
    Get age-specific PSA threshold based on clinical guidelines.
    
    Age-Specific Reference Ranges:
    - Age < 50: PSA > 2.5 ng/mL is concerning
    - Age 50-59: PSA > 3.5 ng/mL is concerning
    - Age 60-69: PSA > 4.5 ng/mL is concerning
    - Age 70+: PSA > 6.5 ng/mL is concerning
    """
    if age is None:
        return 4.0
    
    if age < 50:
        return 2.5
    elif age < 60:
        return 3.5
    elif age < 70:
        return 4.5
    else:
        return 6.5

# ============================================================================
# TREND LOGIC FUNCTIONS (Custom for each marker)
# ============================================================================

def psa_trend_logic(slope, first_val, last_val, r_value=None):
    """PSA trend logic: HIGHER values are WORSE."""
    if slope is None:
        return "Unknown"
    if slope > 0.0001: 
        return "Worsening"
    elif slope < -0.0001:
        return "Improving"
    else: 
        return "Stable"

def egfr_trend_logic(slope, first_val, last_val, r_value=None):
    """eGFR trend logic: LOWER values are WORSE."""
    if slope is None:
        return "Unknown"
    if slope < -0.0001:
        return "Worsening"
    elif slope > 0.0001:
        return "Improving"
    else:
        return "Stable"

def haemoglobin_trend_logic(slope, first_val, last_val, r_value=None):
    """Haemoglobin trend logic: LOWER values are WORSE."""
    if slope is None:
        return "Unknown"
    if slope < -0.0001:
        return "Worsening"
    elif slope > 0.0001:
        return "Improving"
    else: 
        return "Stable"

def alp_trend_logic(slope, first_val, last_val, r_value=None):
    """Alkaline Phosphatase trend logic: HIGHER values are WORSE."""
    if slope is None: 
        return "Unknown"
    if slope > 0.0001:
        return "Worsening"
    elif slope < -0.0001:
        return "Improving"
    else:
        return "Stable"

def mcv_trend_logic(slope, first_val, last_val, r_value=None):
    """MCV trend logic: LOWER values indicate microcytic anemia."""
    if slope is None:
        return "Unknown"
    if slope < -0.0001:
        return "Worsening"
    elif slope > 0.0001:
        return "Improving"
    else: 
        return "Stable"

def default_trend_logic(slope, first_val, last_val, r_value=None):
    """Default trend logic: Just reports direction without judgment."""
    if slope is None:
        return "Unknown"
    if slope > 0.0001:
        return "Increasing"
    elif slope < -0.0001:
        return "Decreasing"
    else: 
        return "Stable"

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_value_trends(df, marker_name, snomed_codes, trend_logic=None,
                         is_psa=False, output_prefix=None, important_codes=None):
    """
    Analyze numeric value trends over time for a specific marker. 
    
    OPTIMIZED VERSION: 
    - Removed MEDIAN (redundant with MEAN)
    - Removed MIN/MAX_EVENT_AGE (redundant with FIRST/LAST)
    - Added only useful EVENT_AGE stats
    - Added NUM_UNIQUE_CODES
    - Added HAS_{important_code} flags
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing medical records
    marker_name : str
        Name of the marker (e.g., 'PSA', 'EGFR', 'HB')
    snomed_codes : list
        List of SNOMED CT codes for this marker
    trend_logic : callable, optional
        Function that determines trend direction
    is_psa : bool
        If True, calculate PSA-specific features
    output_prefix : str, optional
        Prefix for output column names
    important_codes : dict, optional
        Dictionary of {name: code} for important codes to flag
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with value trend analysis for each patient
    """
    
    if output_prefix is None:
        output_prefix = marker_name.upper()
    
    # Filter records with marker-related SNOMED codes
    marker_records = df[df['SNOMED_C_T_CONCEPT_ID'].isin(snomed_codes)].copy()
    
    if len(marker_records) == 0:
        return pd.DataFrame()
    
    # Parse values
    marker_records['PARSED_VALUE'] = marker_records['VALUE'].apply(parse_value)
    
    # Filter records with valid numeric values
    marker_records = marker_records[marker_records['PARSED_VALUE'].notna()].copy()
    
    if len(marker_records) == 0:
        return pd.DataFrame()
    
    # Parse dates
    marker_records['EVENT_DATE_PARSED'] = pd.to_datetime(marker_records['EVENT_DATE'], errors='coerce')
    marker_records = marker_records[marker_records['EVENT_DATE_PARSED'].notna()].copy()
    
    # Group by patient and analyze
    results = []
    
    for patient_guid, patient_data in marker_records.groupby('PATIENT_GUID'):
        patient_data = patient_data.sort_values('EVENT_DATE_PARSED').reset_index(drop=True)
        
        patient_info = patient_data.iloc[0]
        
        values = patient_data['PARSED_VALUE'].tolist()
        dates = patient_data['EVENT_DATE_PARSED']
        
        num_records = len(values)
        first_value = values[0]
        last_value = values[-1]
        
        first_date = dates.min()
        last_date = dates.max()
        time_span_days = (last_date - first_date).days
        time_span_years = time_span_days / 365.25 if time_span_days > 0 else 0
        
        mean_value = np.mean(values)
        min_value = np.min(values)
        max_value = np.max(values)
        std_value = np.std(values) if len(values) > 1 else 0
        
        absolute_change = last_value - first_value
        percent_change = ((last_value - first_value) / first_value * 100) if first_value != 0 else None
        
        slope, r_value, slope_per_year = calculate_trend(values, dates)
        
        if trend_logic: 
            trend_direction = trend_logic(slope, first_value, last_value, r_value)
        else:
            trend_direction = default_trend_logic(slope, first_value, last_value, r_value)
        
        # EVENT_AGE STATISTICS
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
        
        age_at_event = patient_info.get('EVENT_AGE', None)
        
        result = {
            'PATIENT_GUID': patient_guid,
            f'{output_prefix}_NUM_RECORDS': num_records,
            f'{output_prefix}_TIME_SPAN_YEARS': round(time_span_years, 2),
            
            f'{output_prefix}_FIRST_EVENT_AGE': first_event_age,
            f'{output_prefix}_LAST_EVENT_AGE': last_event_age,
            f'{output_prefix}_MEAN_EVENT_AGE': mean_event_age,
            
            f'{output_prefix}_FIRST_VALUE': round(first_value, 2),
            f'{output_prefix}_LAST_VALUE': round(last_value, 2),
            f'{output_prefix}_MEAN':  round(mean_value, 2),
            f'{output_prefix}_MIN':  round(min_value, 2),
            f'{output_prefix}_MAX': round(max_value, 2),
            f'{output_prefix}_STD': round(std_value, 2),
            
            f'{output_prefix}_ABSOLUTE_CHANGE': round(absolute_change, 2),
            f'{output_prefix}_PERCENT_CHANGE': round(percent_change, 2) if percent_change is not None else None,
            f'{output_prefix}_SLOPE_PER_YEAR': round(slope_per_year, 4) if slope_per_year is not None else None,
            f'{output_prefix}_R_VALUE': round(r_value, 3) if r_value is not None else None,
            f'{output_prefix}_IS_WORSENING': 1 if trend_direction == "Worsening" else 0,
            
            f'{output_prefix}_NUM_UNIQUE_CODES':  num_unique_codes,
        }
        
        # PSA-SPECIFIC FEATURES
        if is_psa:
            psa_velocity = slope_per_year if slope_per_year is not None else None
            result[f'{output_prefix}_VELOCITY'] = round(psa_velocity, 3) if psa_velocity is not None else None
            
            doubling_time = calculate_psa_doubling_time(values, dates)
            result[f'{output_prefix}_DOUBLING_TIME_MONTHS'] = doubling_time
            
            threshold = get_age_specific_psa_threshold(age_at_event)
            result[f'{output_prefix}_AGE_THRESHOLD'] = threshold
            result[f'{output_prefix}_ABOVE_AGE_THRESHOLD'] = 1 if last_value > threshold else 0
            result[f'{output_prefix}_MAX_ABOVE_AGE_THRESHOLD'] = 1 if max_value > threshold else 0
            
            result[f'{output_prefix}_LAST_ABOVE_4'] = 1 if last_value > 4.0 else 0
            result[f'{output_prefix}_LAST_ABOVE_10'] = 1 if last_value > 10.0 else 0
            result[f'{output_prefix}_MAX_ABOVE_4'] = 1 if max_value > 4.0 else 0
            result[f'{output_prefix}_MAX_ABOVE_10'] = 1 if max_value > 10.0 else 0
            
            result[f'{output_prefix}_VELOCITY_ABOVE_0_75'] = 1 if (psa_velocity and psa_velocity > 0.75) else 0
            result[f'{output_prefix}_VELOCITY_ABOVE_2'] = 1 if (psa_velocity and psa_velocity > 2.0) else 0
        
        # IMPORTANT CODE FLAGS
        if important_codes: 
            for code_name, code_value in important_codes.items():
                has_code = 1 if code_value in unique_codes else 0
                result[f'{output_prefix}_HAS_{code_name}'] = has_code
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    return results_df

def analyze_values_by_window(df, marker_name, snomed_codes, trend_logic=None,
                              is_psa=False, output_prefix=None):
    """
    Analyze values separately for each time window (W1-W5).
    
    OPTIMIZED VERSION: 
    - Removed W*_MIN (less useful)
    - Removed OVERALL_MIN (less useful)
    - Kept only most useful window features
    
    Returns:
    --------
    pandas.DataFrame
        Features per window plus overall trends
    """
    if output_prefix is None:
        output_prefix = marker_name.upper()
    
    # Filter records
    marker_records = df[df['SNOMED_C_T_CONCEPT_ID'].isin(snomed_codes)].copy()
    
    if len(marker_records) == 0:
        return pd.DataFrame()
    
    # Parse values and dates
    marker_records['PARSED_VALUE'] = marker_records['VALUE'].apply(parse_value)
    marker_records['EVENT_DATE_PARSED'] = pd.to_datetime(marker_records['EVENT_DATE'], errors='coerce')
    marker_records = marker_records[
        (marker_records['PARSED_VALUE'].notna()) &
        (marker_records['EVENT_DATE_PARSED'].notna())
    ].copy()
    
    if len(marker_records) == 0:
        return pd.DataFrame()
    
    # Get all unique patients
    all_patients = df['PATIENT_GUID'].unique()
    
    results = []
    
    for patient_guid in all_patients:
        patient_data = marker_records[marker_records['PATIENT_GUID'] == patient_guid].copy()
        patient_data = patient_data.sort_values('EVENT_DATE_PARSED')
        
        result = {'PATIENT_GUID': patient_guid}
        
        # Analyze each window
        for window in ['W1', 'W2', 'W3', 'W4', 'W5']:
            window_data = patient_data[patient_data['TIME_WINDOW'] == window]
            
            if len(window_data) > 0:
                values = window_data['PARSED_VALUE'].tolist()
                result[f'{output_prefix}_{window}_PRESENT'] = 1
                result[f'{output_prefix}_{window}_COUNT'] = len(values)
                result[f'{output_prefix}_{window}_MEAN'] = round(np.mean(values), 2)
                result[f'{output_prefix}_{window}_MAX'] = round(np.max(values), 2)
                result[f'{output_prefix}_{window}_LAST'] = round(values[-1], 2)
            else:
                result[f'{output_prefix}_{window}_PRESENT'] = 0
                result[f'{output_prefix}_{window}_COUNT'] = 0
                result[f'{output_prefix}_{window}_MEAN'] = None
                result[f'{output_prefix}_{window}_MAX'] = None
                result[f'{output_prefix}_{window}_LAST'] = None
        
        # Overall statistics
        if len(patient_data) > 0:
            all_values = patient_data['PARSED_VALUE'].tolist()
            all_dates = patient_data['EVENT_DATE_PARSED']
            
            result[f'{output_prefix}_EVER_PRESENT'] = 1
            result[f'{output_prefix}_TOTAL_COUNT'] = len(all_values)
            result[f'{output_prefix}_OVERALL_MAX'] = round(np.max(all_values), 2)
            result[f'{output_prefix}_OVERALL_MEAN'] = round(np.mean(all_values), 2)
            result[f'{output_prefix}_FIRST_VALUE'] = round(all_values[0], 2)
            result[f'{output_prefix}_LAST_VALUE'] = round(all_values[-1], 2)
            result[f'{output_prefix}_CHANGE'] = round(all_values[-1] - all_values[0], 2)
            
            slope, r_value, slope_per_year = calculate_trend(all_values, all_dates)
            result[f'{output_prefix}_SLOPE_PER_YEAR'] = round(slope_per_year, 4) if slope_per_year else None
            
            if trend_logic:
                trend_dir = trend_logic(slope, all_values[0], all_values[-1], r_value)
                result[f'{output_prefix}_IS_WORSENING'] = 1 if trend_dir == "Worsening" else 0
            else:
                trend_dir = default_trend_logic(slope, all_values[0], all_values[-1], r_value)
                result[f'{output_prefix}_IS_WORSENING'] = 1 if trend_dir == "Worsening" else 0
            
            # PSA-specific
            if is_psa:
                result[f'{output_prefix}_VELOCITY'] = slope_per_year
                result[f'{output_prefix}_DOUBLING_TIME_MONTHS'] = calculate_psa_doubling_time(all_values, all_dates)
                result[f'{output_prefix}_MAX_ABOVE_4'] = 1 if max(all_values) > 4.0 else 0
                result[f'{output_prefix}_MAX_ABOVE_10'] = 1 if max(all_values) > 10.0 else 0
                result[f'{output_prefix}_LAST_ABOVE_4'] = 1 if all_values[-1] > 4.0 else 0
                result[f'{output_prefix}_LAST_ABOVE_10'] = 1 if all_values[-1] > 10.0 else 0
        else:
            result[f'{output_prefix}_EVER_PRESENT'] = 0
            result[f'{output_prefix}_TOTAL_COUNT'] = 0
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    return results_df
