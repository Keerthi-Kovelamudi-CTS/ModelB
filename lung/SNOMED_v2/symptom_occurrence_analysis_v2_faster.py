"""
Generic Symptom Occurrence and Trend Analysis Script

This script analyzes symptom/condition occurrences over time for each patient
based on symptom-related SNOMED CT codes. It can be used for any symptom/condition
by configuring the SNOMED codes and symptom name.

Usage:
    # Example 1: Analyze COPD
    analyze_symptom(
        csv_file='Lung_cancer_time_window_2008-2024_2.csv',
        symptom_name='COPD',
        snomed_codes=[723245007, 313297008, 313299006, 13645005, 204991000000107]
    )
    
    # Example 2: Analyze Emphysema
    analyze_symptom(
        csv_file='Lung_cancer_time_window_2008-2024_2.csv',
        symptom_name='Emphysema',
        snomed_codes=[87433001, 909721000006104, 68328006, 263747008]
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def clean_patient_guid(guid):
    """Clean patient_guid by removing extra quotes and braces (scalar API kept for compat)."""
    if pd.isna(guid):
        return None
    guid_str = str(guid).strip()
    guid_str = guid_str.replace('"""', '').replace('{', '').replace('}', '')
    return guid_str.strip()


def clean_patient_guid_series(series):
    """Vectorized version of clean_patient_guid: ~50-200x faster than .apply()."""
    cleaned = (series.astype(str)
               .str.replace('"""', '', regex=False)
               .str.replace('{', '', regex=False)
               .str.replace('}', '', regex=False)
               .str.strip())
    return cleaned.where(series.notna(), None)


def filter_patients_by_min_records(df, min_records=150):
    """
    Keep only patients that appear at least min_records times in the DataFrame.
    """
    if df.empty or 'patient_guid' not in df.columns:
        return df.copy()

    if 'patient_guid_CLEAN' in df.columns:
        cleaned_guid = df['patient_guid_CLEAN']
    else:
        cleaned_guid = clean_patient_guid_series(df['patient_guid'])

    patient_counts = cleaned_guid.value_counts(dropna=True)
    eligible_patients = patient_counts[patient_counts >= min_records].index

    mask = cleaned_guid.isin(eligible_patients)
    filtered_df = df[mask].copy()
    filtered_df['patient_guid_CLEAN'] = cleaned_guid[mask].values

    removed_patients = patient_counts.shape[0] - len(eligible_patients)
    print(f"Applied minimum record filter (>= {min_records} rows per patient).")
    print(f"Removed {removed_patients} patients; remaining patients: {filtered_df['patient_guid_CLEAN'].nunique()}.")
    print(f"Records after filtering: {len(filtered_df)}")

    return filtered_df


def parse_date(date_str):
    """Parse date string, handling various formats (scalar API kept for compat)."""
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return None


def _is_preprocessed(df):
    """Detect a DataFrame that has already been globally preprocessed by
    transform_features._preprocess_input — lets us skip the per-call cleaning."""
    return isinstance(df, pd.DataFrame) and \
        'patient_guid_CLEAN' in df.columns and \
        'event_date_parsed' in df.columns


def calculate_frequency_trend(dates):
    """Calculate if symptom occurrences are increasing in frequency over time."""
    if len(dates) < 3:
        return None, None
    
    # Sort dates
    dates_sorted = sorted(dates)
    
    # Calculate intervals between consecutive occurrences (in days)
    intervals = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
    
    # If intervals are decreasing, frequency is increasing (getting worse)
    if len(intervals) < 2:
        return None, None
    
    # Calculate trend of intervals (negative slope means intervals getting shorter = frequency increasing)
    interval_indices = list(range(len(intervals)))
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(interval_indices, intervals)
        
        if slope < -1:  # Intervals are getting significantly shorter
            trend = "Increasing frequency (worsening)"
        elif slope > 1:  # Intervals are getting longer
            trend = "Decreasing frequency (improving)"
        else:
            trend = "Stable frequency"
        
        return slope, trend
    except:
        return None, "Unable to calculate"


def analyze_symptom_occurrences(csv_file_path, symptom_name, snomed_codes, output_prefix=None, time_window_months=None, event_type='observation'):
    """
    Analyze symptom/condition occurrences and trends over time for each patient.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing medical records
    symptom_name : str
        Name of the symptom/condition (e.g., 'COPD', 'Emphysema', 'Asthma')
    snomed_codes : list
        List of SNOMED CT codes (for observations) or medication codes (for medications)
    output_prefix : str, optional
        Prefix for output column names
    time_window_months : int, optional
        If provided, only analyze records from the last N months of patient's lifetime
    event_type : str, optional
        Type of events to analyze: 'observation' or 'medication'. Default is 'observation'
    
    Returns:
    --------
    tuple: (pandas.DataFrame, pandas.DataFrame)
        DataFrame with symptom occurrence analysis for each patient, and detailed records
    """
    
    preprocessed = _is_preprocessed(csv_file_path)
    if preprocessed:
        df = csv_file_path
        # The data was already loaded, cleaned, dates parsed, codes coerced
        # and (if applicable) min-records filtered by the upstream caller.
    elif isinstance(csv_file_path, pd.DataFrame):
        df = csv_file_path
        print(f"Using provided DataFrame ({len(df):,} rows).")
        is_lung_cancer_patient = True
        if is_lung_cancer_patient:
            filtered_df = filter_patients_by_min_records(df, min_records=150)
            if filtered_df.empty:
                print("No patients remain after applying the minimum record threshold (150).")
                return pd.DataFrame(), pd.DataFrame()
            df = filtered_df
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

    # ── Per-call preprocessing (skipped on the fast path) ─────────────
    if not preprocessed:
        # Vectorized cleaning of patient_guid (much faster than .apply per row)
        df = df.copy()
        if 'patient_guid_CLEAN' not in df.columns:
            df['patient_guid_CLEAN'] = clean_patient_guid_series(df['patient_guid'])
        # Vectorized date parsing
        df['event_date_parsed'] = pd.to_datetime(df['event_date'], errors='coerce')
        # event_age numeric coercion
        if 'event_age' in df.columns:
            df['event_age'] = pd.to_numeric(df['event_age'], errors='coerce')
        else:
            print(f"Warning: event_age column not found in CSV file")

    # Determine which columns to use based on event_type
    if event_type == 'medication':
        code_col = 'med_code_id'
        term_col = 'drug_term'
    else:
        code_col = 'snomed_c_t_concept_id'
        term_col = 'term'

    if code_col not in df.columns:
        print(f"Error: {code_col} column not found in CSV file")
        return pd.DataFrame(), pd.DataFrame()
    if not preprocessed:
        df[code_col] = pd.to_numeric(df[code_col], errors='coerce')

    # Filter by event_type if column exists (use cached lower-cased column when available)
    if 'event_type_lower' in df.columns:
        df = df[df['event_type_lower'] == event_type.lower()]
    elif 'event_type' in df.columns:
        df = df[df['event_type'].str.lower() == event_type.lower()].copy()
    if not preprocessed:
        print(f"Filtered to {event_type} records: {len(df)}")
    
    # Filter records with symptom-related codes
    symptom_records = df[df[code_col].isin(snomed_codes)].copy()
    
    print(f"Records with {symptom_name}-related codes: {len(symptom_records)}")
    
    if len(symptom_records) == 0:
        print(f"No {symptom_name}-related records found in the dataset.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter out records with invalid dates
    symptom_records = symptom_records[symptom_records['event_date_parsed'].notna()].copy()
    
    print(f"Records with valid dates: {len(symptom_records)}")
    
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
        
        # Extract symptom occurrence dates
        symptom_dates = patient_data_sorted['event_date_parsed']
        
        # Calculate statistics
        num_symptom_occurrences = len(patient_data_sorted)
        first_symptom_date = symptom_dates.min()
        last_symptom_date = symptom_dates.max()
        
        # Time span
        time_span_days = (last_symptom_date - first_symptom_date).days
        time_span_years = time_span_days / 365.25 if time_span_days > 0 else 0
        
        # Calculate frequency (occurrences per year)
        frequency_per_year = num_symptom_occurrences / time_span_years if time_span_years > 0 else None
        
        # Calculate intervals between occurrences
        if num_symptom_occurrences > 1:
            intervals_days = [(symptom_dates.iloc[i+1] - symptom_dates.iloc[i]).days 
                            for i in range(len(symptom_dates)-1)]
            mean_interval_days = np.mean(intervals_days)
            median_interval_days = np.median(intervals_days)
            min_interval_days = np.min(intervals_days)
            max_interval_days = np.max(intervals_days)
            
            # Calculate if frequency is increasing (getting worse)
            interval_slope, frequency_trend = calculate_frequency_trend(symptom_dates.tolist())
            
            # Compare first half vs second half frequency
            mid_point = len(symptom_dates) // 2
            first_half_dates = symptom_dates.iloc[:mid_point]
            second_half_dates = symptom_dates.iloc[mid_point:]
            
            if len(first_half_dates) > 0 and len(second_half_dates) > 0:
                first_half_span = (first_half_dates.max() - first_half_dates.min()).days / 365.25
                second_half_span = (second_half_dates.max() - second_half_dates.min()).days / 365.25
                
                if first_half_span > 0 and second_half_span > 0:
                    first_half_freq = len(first_half_dates) / first_half_span
                    second_half_freq = len(second_half_dates) / second_half_span
                else:
                    first_half_freq = None
                    second_half_freq = None
            else:
                first_half_freq = None
                second_half_freq = None
        else:
            intervals_days = []
            mean_interval_days = None
            median_interval_days = None
            min_interval_days = None
            max_interval_days = None
            interval_slope = None
            frequency_trend = "Single occurrence"
            first_half_freq = None
            second_half_freq = None
        
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
        
        # Get unique symptom codes for this patient
        symptom_codes = sorted(patient_data_sorted[code_col].unique().tolist())
        
        # Get unique terms
        symptom_terms = patient_data_sorted[term_col].dropna().unique().tolist()
        
        # Determine if getting worse based on frequency trend
        is_worsening = False
        if frequency_trend:
            is_worsening = "worsening" in frequency_trend.lower() or "increasing" in frequency_trend.lower()
        if not is_worsening and first_half_freq is not None and second_half_freq is not None:
            is_worsening = second_half_freq > first_half_freq * 1.1  # 10% increase threshold
        
        # Create column names with symptom name and time window suffix if applicable
        suffix = f'_LAST_{time_window_months}M' if time_window_months is not None else ''
        num_col = f'NUM_{symptom_name.upper()}_OCCURRENCES{suffix}'
        first_date_col = f'FIRST_{symptom_name.upper()}_DATE{suffix}'
        last_date_col = f'LAST_{symptom_name.upper()}_DATE{suffix}'
        codes_col = f'{symptom_name.upper()}_SNOMED_CODES{suffix}'
        terms_col = f'{symptom_name.upper()}_terms{suffix}'
        
        # Event age column names
        first_age_col = f'FIRST_{symptom_name.upper()}_event_age{suffix}'
        last_age_col = f'LAST_{symptom_name.upper()}_event_age{suffix}'
        #min_age_col = f'MIN_{symptom_name.upper()}_event_age{suffix}'
        #max_age_col = f'MAX_{symptom_name.upper()}_event_age{suffix}'
        #mean_age_col = f'MEAN_{symptom_name.upper()}_event_age{suffix}'
        median_age_col = f'MEDIAN_{symptom_name.upper()}_event_age{suffix}'
        
        # Prefix for other columns
        output_prefix_with_suffix = f'{output_prefix}{suffix}' if time_window_months is not None else output_prefix
        
        results.append({
            'patient_guid': patient_guid,
            'sex': patient_info['sex'],
            'patient_ethnicity_16': patient_info['patient_ethnicity_16'],
            'patient_ethnicity_6': patient_info['patient_ethnicity_6'],
            'patient_age': patient_info['patient_age'],
            #'PRACTICE_ID': patient_info['PRACTICE_ID'],
            'cancer_class': patient_info['cancer_class'],
            #'CANCER_ID': patient_info['CANCER_ID'],
            num_col: num_symptom_occurrences,
            first_date_col: first_symptom_date.strftime('%Y-%m-%d') if pd.notna(first_symptom_date) else None,
            last_date_col: last_symptom_date.strftime('%Y-%m-%d') if pd.notna(last_symptom_date) else None,
            first_age_col: round(first_event_age, 1) if first_event_age is not None else None,
            last_age_col: round(last_event_age, 1) if last_event_age is not None else None,
            #min_age_col: round(min_event_age, 1) if min_event_age is not None else None,
            #max_age_col: round(max_event_age, 1) if max_event_age is not None else None,
            #mean_age_col: round(mean_event_age, 1) if mean_event_age is not None else None,
            median_age_col: round(median_event_age, 1) if median_event_age is not None else None,
            #f'{output_prefix_with_suffix}_TIME_SPAN_DAYS': time_span_days,
            f'{output_prefix_with_suffix}_TIME_SPAN_YEARS': round(time_span_years, 2) if time_span_years else None,
            f'{output_prefix_with_suffix}_FREQUENCY_PER_YEAR': round(frequency_per_year, 2) if frequency_per_year else None,
            #f'{output_prefix_with_suffix}_MEAN_INTERVAL_DAYS': round(mean_interval_days, 1) if mean_interval_days is not None else None,
            f'{output_prefix_with_suffix}_MEDIAN_INTERVAL_DAYS': round(median_interval_days, 1) if median_interval_days is not None else None,
            f'{output_prefix_with_suffix}_MIN_INTERVAL_DAYS': min_interval_days if min_interval_days is not None else None,
            f'{output_prefix_with_suffix}_MAX_INTERVAL_DAYS': max_interval_days if max_interval_days is not None else None,
            f'{output_prefix_with_suffix}_FREQUENCY_TREND': frequency_trend,
            #f'{output_prefix_with_suffix}_FREQUENCY_TREND_SLOPE': round(interval_slope, 2) if interval_slope is not None else None,
            f'{output_prefix_with_suffix}_FIRST_HALF_FREQUENCY': round(first_half_freq, 2) if first_half_freq else None,
            f'{output_prefix_with_suffix}_SECOND_HALF_FREQUENCY': round(second_half_freq, 2) if second_half_freq else None,
            f'{output_prefix_with_suffix}_IS_WORSENING': is_worsening,
            codes_col: ', '.join(map(str, symptom_codes)),
            terms_col: ' | '.join([str(t).replace('"""', '') for t in symptom_terms[:5]])  # First 5 terms
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by patient_guid
    if len(results_df) > 0:
        results_df = results_df.sort_values('patient_guid').reset_index(drop=True)
    
    return results_df, symptom_records


def print_summary(results, symptom_name, output_prefix):
    """Print summary statistics for the analysis."""
    if len(results) == 0:
        print(f"No patients with {symptom_name} records found.")
        return
    
    print(f"\n{'='*80}")
    print(f"{symptom_name.upper()} OCCURRENCE ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal patients with {symptom_name} records: {len(results)}")
    
    # Display summary statistics
    num_col = f'NUM_{symptom_name.upper()}_OCCURRENCES'
    print(f"\nSummary Statistics:")
    print(f"  Mean number of {symptom_name} occurrences per patient: {results[num_col].mean():.1f}")
    print(f"  Median number of {symptom_name} occurrences: {results[num_col].median():.1f}")
    print(f"  Range: {results[num_col].min()} - {results[num_col].max()} occurrences")
    
    valid_freq = results[results[f'{output_prefix}_FREQUENCY_PER_YEAR'].notna()]
    if len(valid_freq) > 0:
        print(f"  Mean frequency: {valid_freq[f'{output_prefix}_FREQUENCY_PER_YEAR'].mean():.2f} occurrences per year")
        print(f"  Median frequency: {valid_freq[f'{output_prefix}_FREQUENCY_PER_YEAR'].median():.2f} occurrences per year")
    
    # Worsening analysis
    worsening_count = results[f'{output_prefix}_IS_WORSENING'].sum()
    print(f"\nPatients with worsening {symptom_name} frequency: {worsening_count} ({worsening_count/len(results)*100:.1f}%)")
    
    # Frequency trend summary
    if f'{output_prefix}_FREQUENCY_TREND' in results.columns:
        trend_counts = results[f'{output_prefix}_FREQUENCY_TREND'].value_counts()
        print(f"\nFrequency Trend Summary:")
        for trend, count in trend_counts.items():
            print(f"  {trend}: {count} patients")


def analyze_symptom(csv_file, symptom_name, snomed_codes, output_prefix=None, time_window_months=None, event_type='observation'):
    """
    Main function to analyze a symptom/condition or medication.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    symptom_name : str
        Name of the symptom/condition or medication
    snomed_codes : list
        List of SNOMED CT codes (for observations) or medication codes (for medications)
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
    results, symptom_records = analyze_symptom_occurrences(csv_file, symptom_name, snomed_codes, output_prefix, time_window_months, event_type)
    
    # if len(results) > 0:
    #     # Print summary
    #     print_summary(results, symptom_name, output_prefix)
        
    #     # Save results to CSV
    #     output_file = f'{output_prefix}_occurrence_analysis_results.csv'
    #     results.to_csv(output_file, index=False)
    #     print(f"\nResults saved to: {output_file}")
        
    #     # Save detailed symptom records for further analysis
    #     detailed_output = f'{output_prefix}_detailed_records.csv'
    #     symptom_records[['patient_guid_CLEAN', 'event_date', 'snomed_c_t_concept_id', 'term', 
    #                      'DATE_OF_DIAGNOSIS', 'HISTORY_PHASE', 'DAYS_FROM_DIAGNOSIS']].to_csv(detailed_output, index=False)
    #     print(f"Detailed {symptom_name} records saved to: {detailed_output}")
        
    #     # Display first few results
    #     print(f"\nFirst 10 patients:")
    #     num_col = f'NUM_{symptom_name.upper()}_OCCURRENCES'
    #     first_date_col = f'FIRST_{symptom_name.upper()}_DATE'
    #     last_date_col = f'LAST_{symptom_name.upper()}_DATE'
    #     display_cols = ['patient_guid', num_col, first_date_col, last_date_col,
    #                    'FREQUENCY_PER_YEAR', 'FREQUENCY_TREND', 'IS_WORSENING']
    #     # Only show columns that exist
    #     display_cols = [col for col in display_cols if col in results.columns]
    #     print(results[display_cols].head(10).to_string(index=False))
    
    return results, symptom_records


# Example usage configurations
SYMPTOM_CONFIGS = {
    'COPD': {
        'snomed_codes': [
            723245007,      # Number of chronic obstructive pulmonary disease exacerbations in past year
            313297008,      # Chronic obstructive pulmonary disease
            313299006,      # Chronic obstructive pulmonary disease
            13645005,       # Chronic obstructive pulmonary disease
            204991000000107 # Suspected chronic obstructive pulmonary disease
        ]
    },
    'Emphysema': {
        'snomed_codes': [
            87433001,          # Pulmonary emphysema
            909721000006104,   # [RFC] Reason for care : Emphysema / [RFC] Emphysema
            68328006,          # Emphysema
            263747008          # Emphysema (disorder)
        ]
    }
    # Add more symptom configurations here as needed
}


def main():
    """Main function with example usage."""
    
    # Path to CSV file
    csv_file = 'Lung_cancer_time_window_2008-2024_2.csv'
    
    # Example 1: Analyze COPD
    print("="*80)
    print("ANALYZING COPD")
    print("="*80)
    copd_results, copd_records = analyze_symptom(
        csv_file=csv_file,
        symptom_name='COPD',
        snomed_codes=SYMPTOM_CONFIGS['COPD']['snomed_codes']
    )
    
    # Example 2: Analyze Emphysema
    print("\n" + "="*80)
    print("ANALYZING EMPHYSEMA")
    print("="*80)
    emphysema_results, emphysema_records = analyze_symptom(
        csv_file=csv_file,
        symptom_name='Emphysema',
        snomed_codes=SYMPTOM_CONFIGS['Emphysema']['snomed_codes']
    )
    
    # To analyze a new symptom, simply call:
    # analyze_symptom(
    #     csv_file='your_file.csv',
    #     symptom_name='YourSymptom',
    #     snomed_codes=[code1, code2, code3, ...]
    # )


if __name__ == "__main__":
    main()

