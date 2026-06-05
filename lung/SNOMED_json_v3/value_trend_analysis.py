"""
Generic value Trend Analysis Script

This script analyzes numeric values over time for any symptom/condition based on 
SNOMED CT codes. It allows custom logic to determine if increasing values are 
good or bad for trend interpretation.

Usage:
    # Example: Analyze FEV values (higher is better)
    def fev_logic(slope, first_val, last_val):
        if slope > 0:
            return "Improving"  # Higher FEV is better
        elif slope < 0:
            return "Declining"   # Lower FEV is worse
        else:
            return "Stable"
    
    results, records = analyze_value_trends(
        csv_file='data.csv',
        symptom_name='FEV',
        snomed_codes=[251944000, 313222007, ...],
        trend_logic=fev_logic
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def clean_patient_guid(guid):
    """Clean patient_guid by removing extra quotes and braces."""
    if pd.isna(guid):
        return None
    guid_str = str(guid).strip()
    # Remove triple quotes and braces
    guid_str = guid_str.replace('"""', '').replace('{', '').replace('}', '')
    return guid_str.strip()


def filter_patients_by_min_records(df, min_records=150):
    """
    Keep only patients that appear at least min_records times in the DataFrame.
    """
    if df.empty or 'patient_guid' not in df.columns:
        return df.copy()
    
    cleaned_guid = df['patient_guid'].apply(clean_patient_guid)
    patient_counts = cleaned_guid.value_counts(dropna=True)
    eligible_patients = patient_counts[patient_counts >= min_records].index
    
    mask = cleaned_guid.isin(eligible_patients)
    filtered_df = df[mask].copy()
    filtered_df['patient_guid_CLEAN'] = cleaned_guid[mask]
    
    removed_patients = patient_counts.shape[0] - len(eligible_patients)
    print(f"Applied minimum record filter (>= {min_records} rows per patient).")
    print(f"Removed {removed_patients} patients; remaining patients: {filtered_df['patient_guid_CLEAN'].nunique()}.")
    print(f"Records after filtering: {len(filtered_df)}")
    
    return filtered_df


def parse_value(value_str):
    """Parse value column, handling quotes and converting to numeric."""
    if pd.isna(value_str):
        return None
    try:
        # Remove quotes and whitespace
        value_clean = str(value_str).strip().replace('"""', '').replace('"', '').strip()
        # Check for empty values
        if value_clean == '' or value_clean.lower() in ['nan', 'none', 'null', 'na']:
            return None
        # Try to convert to float
        return float(value_clean)
    except (valueError, TypeError):
        return None


def parse_date(date_str):
    """Parse date string, handling various formats."""
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str)
    except:
        return None


def default_trend_logic(slope, first_val, last_val, r_value=None):
    """
    Default trend logic: assumes higher values are better.
    
    Parameters:
    -----------
    slope : float
        Slope from linear regression (change per day)
    first_val : float
        First value in the series
    last_val : float
        Last value in the series
    r_value : float, optional
        Correlation coefficient
    
    Returns:
    --------
    str
        Trend direction: "Improving", "Declining", or "Stable"
    """
    if slope > 0.0001:
        return "Improving"
    elif slope < -0.0001:
        return "Declining"
    else:
        return "Stable"


def calculate_trend(values, dates, trend_logic=None):
    """
    Calculate linear trend for values over time using custom logic.
    
    Parameters:
    -----------
    values : list
        List of numeric values
    dates : pandas.Series
        Series of datetime objects
    trend_logic : callable, optional
        Function that takes (slope, first_val, last_val, r_value) and returns
        trend direction string. If None, uses default logic.
    
    Returns:
    --------
    tuple: (slope, r_value, trend_direction)
    """
    if len(values) < 2:
        return None, None, None
    
    # Convert dates to numeric (days since first date)
    dates_numeric = [(d - dates.iloc[0]).days for d in dates]
    
    # Check if all dates are the same (would cause regression error)
    if len(set(dates_numeric)) < 2:
        # All dates are the same, can't calculate trend
        # Check if values are changing
        if len(set(values)) > 1:
            trend_direction = "Multiple measurements on same date"
        else:
            trend_direction = "Stable"
        return None, None, trend_direction
    
    # Calculate linear regression
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
        
        # Use custom logic if provided, otherwise use default
        if trend_logic:
            first_val = values[0]
            last_val = values[-1]
            trend_direction = trend_logic(slope, first_val, last_val, r_value)
        else:
            # Default logic: higher is better
            if slope > 0.0001:
                trend_direction = "Improving"
            elif slope < -0.0001:
                trend_direction = "Declining"
            else:
                trend_direction = "Stable"
        
        return slope, r_value, trend_direction
    except Exception as e:
        return None, None, "Unable to calculate"


def analyze_value_trends(csv_file_path, symptom_name, snomed_codes, trend_logic=None, time_window_months=None, event_type='observation'):
    """
    Analyze numeric value trends over time for any symptom/condition or medication.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing medical records
    symptom_name : str
        Name of the symptom/condition or medication (e.g., 'FEV', 'Blood Pressure', 'Weight')
    snomed_codes : list
        List of SNOMED CT codes (for observations) or medication codes (for medications)
    trend_logic : callable, optional
        Function that determines trend direction based on slope and values.
        Signature: trend_logic(slope, first_val, last_val, r_value) -> str
        Should return: "Improving", "Declining", "Stable", or custom string
    time_window_months : int, optional
        If provided, only analyze records from the last N months of patient's lifetime
    event_type : str, optional
        Type of events to analyze: 'observation' or 'medication'. Default is 'observation'
    
    Returns:
    --------
    tuple: (pandas.DataFrame, pandas.DataFrame)
        DataFrame with value trend analysis for each patient, and detailed records
    """
    
    if isinstance(csv_file_path, pd.DataFrame):
        df = csv_file_path
        print(f"Using provided DataFrame ({len(df):,} rows).")
        is_lung_cancer_patient = True
    else:
        print(f"Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        print(f"Total records loaded: {len(df)}")
        is_lung_cancer_patient = "lungCancer" in str(csv_file_path)
    if is_lung_cancer_patient:
        filtered_df = filter_patients_by_min_records(df, min_records=150)
        if filtered_df.empty:
            print("No patients remain after applying the minimum record threshold (150).")
            return pd.DataFrame(), pd.DataFrame()
        df = filtered_df
    
    # Clean patient_guid
    df['patient_guid_CLEAN'] = df['patient_guid'].apply(clean_patient_guid)
    
    # Determine which columns to use based on event_type
    if event_type == 'medication':
        code_col = 'med_code_id'
        term_col = 'drug_term'
        # Convert med_code_id to numeric
        if code_col in df.columns:
            df[code_col] = pd.to_numeric(df[code_col], errors='coerce')
        else:
            print(f"Error: {code_col} column not found in CSV file")
            return pd.DataFrame(), pd.DataFrame()
    else:  # observation
        code_col = 'snomed_c_t_concept_id'
        term_col = 'term'
        # Convert snomed_c_t_concept_id to numeric
        if code_col in df.columns:
            df[code_col] = pd.to_numeric(df[code_col], errors='coerce')
        else:
            print(f"Error: {code_col} column not found in CSV file")
            return pd.DataFrame(), pd.DataFrame()
    
    # Parse dates
    df['event_date_parsed'] = df['event_date'].apply(parse_date)
    
    # Parse event_age if available
    if 'event_age' in df.columns:
        df['event_age'] = pd.to_numeric(df['event_age'], errors='coerce')
    else:
        print(f"Warning: event_age column not found in CSV file")
    
    # Filter by event_type if column exists
    if 'event_type' in df.columns:
        df = df[df['event_type'].str.lower() == event_type.lower()].copy()
        print(f"Filtered to {event_type} records: {len(df)}")
    
    # Filter records with symptom-related codes first
    symptom_records = df[df[code_col].isin(snomed_codes)].copy()
    
    print(f"Records with {symptom_name}-related codes: {len(symptom_records)}")
    
    if len(symptom_records) == 0:
        print(f"No {symptom_name}-related records found in the dataset.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter out records with empty value column before parsing
    def is_empty_value(val):
        if pd.isna(val):
            return True
        val_str = str(val).strip().replace('"""', '').replace('"', '').strip()
        return val_str == '' or val_str.lower() in ['nan', 'none', 'null', 'na']
    
    symptom_records = symptom_records[~symptom_records['value'].apply(is_empty_value)].copy()
    
    print(f"Records with non-empty value column: {len(symptom_records)}")
    
    # Parse values
    value_col_name = f'{symptom_name.upper()}_value'
    symptom_records[value_col_name] = symptom_records['value'].apply(parse_value)
    
    # Filter out records with invalid dates or non-numeric values (after parsing)
    symptom_records = symptom_records[
        (symptom_records['event_date_parsed'].notna()) & 
        (symptom_records[value_col_name].notna())
    ].copy()
    
    print(f"Records with valid dates and numeric values: {len(symptom_records)}")
    
    # Group by patient and analyze
    results = []
    
    for patient_guid, patient_data in symptom_records.groupby('patient_guid_CLEAN'):
        # Sort by event date
        patient_data_sorted = patient_data.sort_values('event_date_parsed').reset_index(drop=True)
        
        # Filter by time window if specified (last N months of patient's lifetime)
        if time_window_months is not None:
            # Get the latest event date for this patient
            max_event_date = patient_data_sorted['event_date_parsed'].max()
            # Calculate cutoff date (N months before the latest event)
            cutoff_date = max_event_date - pd.DateOffset(months=time_window_months)
            # Filter records within the time window
            patient_data_sorted = patient_data_sorted[patient_data_sorted['event_date_parsed'] >= cutoff_date].reset_index(drop=True)
            
            # Skip patient if no records in the time window
            if len(patient_data_sorted) == 0:
                continue
        
        # Get patient info
        patient_info = patient_data_sorted.iloc[0]
        
        # Extract values and dates
        values = patient_data_sorted[value_col_name].tolist()
        event_dates = patient_data_sorted['event_date_parsed']
        
        # Calculate statistics
        num_records = len(patient_data_sorted)
        first_date = event_dates.min()
        last_date = event_dates.max()
        first_value = values[0]
        last_value = values[-1]
        
        # Time span
        time_span_days = (last_date - first_date).days
        time_span_years = time_span_days / 365.25
        
        # Calculate change
        absolute_change = last_value - first_value
        percent_change = ((last_value - first_value) / first_value * 100) if first_value != 0 else None
        
        # Statistical measures
        mean_value = np.mean(values)
        median_value = np.median(values)
        min_value = np.min(values)
        max_value = np.max(values)
        std_value = np.std(values)
        
        # Calculate trend using custom logic
        slope, r_value, trend_direction = calculate_trend(values, event_dates, trend_logic)
        
        # Calculate event_age statistics if available
        if 'event_age' in patient_data_sorted.columns:
            event_ages = patient_data_sorted['event_age'].dropna()
            if len(event_ages) > 0:
                first_event_age = event_ages.iloc[0]
                last_event_age = event_ages.iloc[-1]
                min_event_age = event_ages.min()
                max_event_age = event_ages.max()
                mean_event_age = event_ages.mean()
                median_event_age = event_ages.median()
            else:
                first_event_age = None
                last_event_age = None
                min_event_age = None
                max_event_age = None
                mean_event_age = None
                median_event_age = None
        else:
            first_event_age = None
            last_event_age = None
            min_event_age = None
            max_event_age = None
            mean_event_age = None
            median_event_age = None
        
        # Get unique codes for this patient
        symptom_codes = sorted(patient_data_sorted[code_col].unique().tolist())
        
        # Get unique terms
        symptom_terms = patient_data_sorted[term_col].dropna().unique().tolist()
        
        # Create column names with symptom name and time window suffix if applicable
        suffix = f'_LAST_{time_window_months}M' if time_window_months is not None else ''
        first_date_col = f'FIRST_{symptom_name.upper()}_DATE{suffix}'
        last_date_col = f'LAST_{symptom_name.upper()}_DATE{suffix}'
        num_col = f'NUM_{symptom_name.upper()}_RECORDS{suffix}'
        first_val_col = f'FIRST_{symptom_name.upper()}_value{suffix}'
        last_val_col = f'LAST_{symptom_name.upper()}_value{suffix}'
        mean_val_col = f'MEAN_{symptom_name.upper()}_value{suffix}'
        median_val_col = f'MEDIAN_{symptom_name.upper()}_value{suffix}'
        min_val_col = f'MIN_{symptom_name.upper()}_value{suffix}'
        max_val_col = f'MAX_{symptom_name.upper()}_value{suffix}'
        std_val_col = f'STD_{symptom_name.upper()}_value{suffix}'
        codes_col = f'{symptom_name.upper()}_SNOMED_CODES{suffix}'
        terms_col = f'{symptom_name.upper()}_terms{suffix}'
        
        # Event age column names
        first_age_col = f'FIRST_{symptom_name.upper()}_event_age{suffix}'
        last_age_col = f'LAST_{symptom_name.upper()}_event_age{suffix}'
        #min_age_col = f'MIN_{symptom_name.upper()}_event_age{suffix}'
        #max_age_col = f'MAX_{symptom_name.upper()}_event_age{suffix}'
        #mean_age_col = f'MEAN_{symptom_name.upper()}_event_age{suffix}'
        median_age_col = f'MEDIAN_{symptom_name.upper()}_event_age{suffix}'
        
        # Symptom name with suffix for other columns
        symptom_name_with_suffix = f'{symptom_name}{suffix}' if time_window_months is not None else symptom_name
        
        results.append({
            'patient_guid': patient_guid,
            'sex': patient_info['sex'],
            'patient_ethnicity_16': patient_info['patient_ethnicity_16'],
            'patient_ethnicity_6': patient_info['patient_ethnicity_6'],
            'patient_age': patient_info['patient_age'],
            'cancer_class': patient_info['cancer_class'],
            #'PRACTICE_ID': patient_info['PRACTICE_ID'],
            # 'CANCER_ID': patient_info['CANCER_ID'],
            first_date_col: first_date.strftime('%Y-%m-%d') if pd.notna(first_date) else None,
            last_date_col: last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else None,
            first_age_col: round(first_event_age, 1) if first_event_age is not None else None,
            last_age_col: round(last_event_age, 1) if last_event_age is not None else None,
            #min_age_col: round(min_event_age, 1) if min_event_age is not None else None,
            #max_age_col: round(max_event_age, 1) if max_event_age is not None else None,
            #mean_age_col: round(mean_event_age, 1) if mean_event_age is not None else None,
            median_age_col: round(median_event_age, 1) if median_event_age is not None else None,
            #f'{symptom_name_with_suffix}_TIME_SPAN_DAYS': time_span_days,
            f'{symptom_name_with_suffix}_TIME_SPAN_YEARS': round(time_span_years, 2),
            num_col: num_records,
            first_val_col: round(first_value, 2),
            last_val_col: round(last_value, 2),
            mean_val_col: round(mean_value, 2),
            median_val_col: round(median_value, 2),
            min_val_col: round(min_value, 2),
            max_val_col: round(max_value, 2),
            std_val_col: round(std_value, 2),
            f'{symptom_name_with_suffix}_ABSOLUTE_CHANGE': round(absolute_change, 2),
            f'{symptom_name_with_suffix}_PERCENT_CHANGE': round(percent_change, 2) if percent_change is not None else None,
            f'{symptom_name_with_suffix}_TREND_SLOPE': round(slope, 6) if slope is not None else None,
            f'{symptom_name_with_suffix}_TREND_CORRELATION': round(r_value, 3) if r_value is not None else None,
            f'{symptom_name_with_suffix}_TREND_DIRECTION': trend_direction if trend_direction else None,
            codes_col: ', '.join(map(str, symptom_codes)),
            terms_col: ' | '.join([str(t).replace('"""', '') for t in symptom_terms[:5]])  # First 5 terms
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by patient_guid
    if len(results_df) > 0:
        results_df = results_df.sort_values('patient_guid').reset_index(drop=True)
    
    return results_df, symptom_records


def print_summary(results, symptom_name):
    """Print summary statistics for the analysis."""
    if len(results) == 0:
        print(f"No patients with {symptom_name} records found.")
        return
    
    print(f"\n{'='*80}")
    print(f"{symptom_name.upper()} value TREND ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal patients with {symptom_name} records: {len(results)}")
    
    # Display summary statistics
    num_col = f'NUM_{symptom_name.upper()}_RECORDS'
    mean_col = f'MEAN_{symptom_name.upper()}_value'
    
    print(f"\nSummary Statistics:")
    print(f"  Mean number of {symptom_name} records per patient: {results[num_col].mean():.1f}")
    if mean_col in results.columns:
        print(f"  Mean {symptom_name} value: {results[mean_col].mean():.2f}")
    print(f"  Mean time span: {results[f'{symptom_name}_TIME_SPAN_YEARS'].mean():.2f} years")
    if f'{symptom_name}_PERCENT_CHANGE' in results.columns:
        valid_changes = results[results[f'{symptom_name}_PERCENT_CHANGE'].notna()]
        if len(valid_changes) > 0:
            print(f"  Mean percent change: {valid_changes[f'{symptom_name}_PERCENT_CHANGE'].mean():.2f}%")
    
    # Trend analysis
    if f'{symptom_name}_TREND_DIRECTION' in results.columns:
        trend_counts = results[f'{symptom_name}_TREND_DIRECTION'].value_counts()
        print(f"\nTrend Direction Summary:")
        for direction, count in trend_counts.items():
            print(f"  {direction}: {count} patients")


def analyze_value_trend(csv_file, symptom_name, snomed_codes, trend_logic=None, output_prefix=None, time_window_months=None, event_type='observation'):
    """
    Main function to analyze value trends for a symptom/condition or medication.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    symptom_name : str
        Name of the symptom/condition or medication
    snomed_codes : list
        List of SNOMED CT codes (for observations) or medication codes (for medications)
    trend_logic : callable, optional
        Function that determines if increasing values are good or bad.
        Signature: trend_logic(slope, first_val, last_val, r_value) -> str
    output_prefix : str, optional
        Prefix for output files (defaults to symptom_name.lower())
    time_window_months : int, optional
        If provided, only analyze records from the last N months of patient's lifetime
    event_type : str, optional
        Type of events to analyze: 'observation' or 'medication'. Default is 'observation'
    
    Returns:
    --------
    tuple: (pandas.DataFrame, pandas.DataFrame)
        Results DataFrame and detailed records DataFrame
    """
    if output_prefix is None:
        output_prefix = symptom_name.lower().replace(' ', '_')
    
    # Run analysis
    results, symptom_records = analyze_value_trends(
        csv_file, 
        symptom_name, 
        snomed_codes, 
        trend_logic,
        time_window_months,
        event_type
    )
    
    # if len(results) > 0:
    #     # Print summary
    #     print_summary(results, symptom_name)
        
    #     # Save results to CSV
    #     output_file = f'{output_prefix}_value_trend_analysis_results.csv'
    #     results.to_csv(output_file, index=False)
    #     print(f"\nResults saved to: {output_file}")
        
    #     # Save detailed records for further analysis
    #     detailed_output = f'{output_prefix}_detailed_records.csv'
    #     value_col = f'{symptom_name.upper()}_value'
    #     if value_col in symptom_records.columns:
    #         detail_cols = ['patient_guid_CLEAN', 'event_date', 'snomed_c_t_concept_id', 'term', 
    #                       value_col, 'DATE_OF_DIAGNOSIS', 'HISTORY_PHASE']
    #     else:
    #         detail_cols = ['patient_guid_CLEAN', 'event_date', 'snomed_c_t_concept_id', 'term', 
    #                       'DATE_OF_DIAGNOSIS', 'HISTORY_PHASE']
    #     symptom_records[detail_cols].to_csv(detailed_output, index=False)
    #     print(f"Detailed {symptom_name} records saved to: {detailed_output}")
        
    #     # Display first few results
    #     print(f"\nFirst 10 patients:")
    #     first_date_col = f'FIRST_{symptom_name.upper()}_DATE'
    #     last_date_col = f'LAST_{symptom_name.upper()}_DATE'
    #     num_col = f'NUM_{symptom_name.upper()}_RECORDS'
    #     first_val_col = f'FIRST_{symptom_name.upper()}_value'
    #     last_val_col = f'LAST_{symptom_name.upper()}_value'
    #     display_cols = ['patient_guid', first_date_col, last_date_col, num_col,
    #                    first_val_col, last_val_col, 'PERCENT_CHANGE', 'TREND_DIRECTION']
    #     # Only show columns that exist
    #     display_cols = [col for col in display_cols if col in results.columns]
    #     print(results[display_cols].head(10).to_string(index=False))
    
    return results, symptom_records


# ============================================================================
# Example Trend Logic Functions
# ============================================================================

def fev_trend_logic(slope, first_val, last_val, r_value=None):
    """FEV trend logic: Higher values are better (improving lung function)."""
    if slope > 0.0001:
        return "Improving"
    elif slope < -0.0001:
        return "Declining"
    else:
        return "Stable"


def blood_pressure_trend_logic(slope, first_val, last_val, r_value=None):
    """Blood pressure trend logic: Lower values are better."""
    if slope < -0.0001:
        return "Improving"  # Decreasing BP is good
    elif slope > 0.0001:
        return "Declining"   # Increasing BP is bad
    else:
        return "Stable"


def weight_trend_logic(slope, first_val, last_val, r_value=None):
    """Weight trend logic: Context-dependent, but generally stable is good."""
    percent_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
    
    if abs(percent_change) < 5:  # Less than 5% change
        return "Stable"
    elif slope > 0.0001:
        return "Increasing"
    elif slope < -0.0001:
        return "Decreasing"
    else:
        return "Stable"


def threshold_trend_logic(threshold_value, higher_is_better=True):
    """
    Create a trend logic function based on a threshold value.
    
    Parameters:
    -----------
    threshold_value : float
        Threshold value to compare against
    higher_is_better : bool
        If True, values above threshold are good; if False, values below threshold are good
    
    Returns:
    --------
    callable
        Trend logic function
    """
    def logic(slope, first_val, last_val, r_value=None):
        if higher_is_better:
            if last_val > threshold_value and slope > 0.0001:
                return "Improving"
            elif last_val < threshold_value and slope < -0.0001:
                return "Declining"
            else:
                return "Stable"
        else:
            if last_val < threshold_value and slope < -0.0001:
                return "Improving"
            elif last_val > threshold_value and slope > 0.0001:
                return "Declining"
            else:
                return "Stable"
    
    return logic


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Main function with example usage."""
    
    csv_file = 'Lung_cancer_time_window_2008-2024_2.csv'
    
    # Example 1: Analyze FEV (higher is better)
    print("="*80)
    print("ANALYZING FEV valueS")
    print("="*80)
    fev_codes = [
        251944000,   # FEV1
        313222007,   # FEV1/FVC percent
        407576000,   # FVC/Expected FVC percent
        407602006,   # FEV1/FVC ratio
        313223002    # Percent predicted FEV1
    ]
    
    fev_results, fev_records = analyze_value_trend(
        csv_file=csv_file,
        symptom_name='FEV',
        snomed_codes=fev_codes,
        trend_logic=fev_trend_logic
    )
    
    # Example 2: Analyze with custom logic (lower is better)
    # print("\n" + "="*80)
    # print("ANALYZING BLOOD PRESSURE valueS")
    # print("="*80)
    # bp_codes = [123456789, 987654321]  # Replace with actual SNOMED codes
    # bp_results, bp_records = analyze_value_trend(
    #     csv_file=csv_file,
    #     symptom_name='BloodPressure',
    #     snomed_codes=bp_codes,
    #     trend_logic=blood_pressure_trend_logic
    # )


if __name__ == "__main__":
    main()

