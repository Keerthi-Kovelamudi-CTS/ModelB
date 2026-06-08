"""
Lung Cancer Risk Factors Analysis

This script analyzes medical records to extract risk factors for lung cancer patients.
It supports both entire lifetime analysis and time-windowed analysis.

Features:
---------
1. ENTIRE LIFETIME ANALYSIS (default):
   - Analyzes all medical records for each patient
   - Generates features like NUM_SMOKING_OCCURRENCES, FIRST_SMOKING_DATE, etc.

2. TIME-WINDOWED ANALYSIS (optional):
   - Analyzes only the last N months of patient's medical history
   - Generates additional features with suffix _LAST_NM (e.g., NUM_SMOKING_OCCURRENCES_LAST_18M)
   - Useful for understanding recent patient status vs. lifetime trends

Usage:
------
1. Entire lifetime analysis only:
   python symptom_trend_analysis.py

2. Both lifetime and time-windowed analysis (e.g., last 18 months):
   python symptom_trend_analysis.py 18

3. Programmatic usage:
   from symptom_trend_analysis import extract_lung_risk_factors
   
   # Entire lifetime only
   results = extract_lung_risk_factors('data.csv')
   
   # Both lifetime and last 18 months
   results = extract_lung_risk_factors('data.csv', time_window_months=18)

Output:
-------
- The script generates 'all_patients_trend.csv' with all features
- Features without suffix: entire lifetime analysis
- Features with _LAST_NM suffix: last N months analysis (if time_window_months provided)

Example Output Columns:
-----------------------
Lifetime features:
  - NUM_SMOKING_OCCURRENCES
  - FIRST_SMOKING_DATE
  - LAST_SMOKING_DATE
  - smoking_FREQUENCY_PER_YEAR
  - smoking_IS_WORSENING
  
Time-windowed features (when time_window_months=18):
  - NUM_SMOKING_OCCURRENCES_LAST_18M
  - FIRST_SMOKING_DATE_LAST_18M
  - LAST_SMOKING_DATE_LAST_18M
  - smoking_LAST_18M_FREQUENCY_PER_YEAR
  - smoking_LAST_18M_IS_WORSENING
"""


import os
import pandas as pd  # type: ignore
from symptom_occurrence_analysis import (
    analyze_symptom,
    clean_patient_guid,
    clean_patient_guid_series,
    calculate_frequency_trend,
    parse_date,
    filter_patients_by_min_records,
)
from value_trend_analysis import analyze_value_trend, threshold_trend_logic
from functools import reduce


# ---------------------------------------------------------------------------
# Performance helpers: one-time vectorized preprocessing + parallelism
# ---------------------------------------------------------------------------
#
# Setting USE_GPU=1 (and having `cudf` installed, e.g. RAPIDS on a CUDA box)
# routes the I/O + vectorized cleaning step through cuDF for huge speedups on
# wide CSV inputs.  All downstream per-patient analytics still run on pandas
# (cuDF is not used inside the analyzers because they rely on pandas-only
# behaviour like ``groupby().apply``).
USE_GPU = os.environ.get("USE_GPU", "0") == "1"
N_JOBS = int(os.environ.get("RISK_FACTOR_JOBS", "-1"))  # -1 = all cores


def _preprocess_input(input_data, *, apply_min_records_filter=True, min_records=50):
    """One-time, fully vectorized preprocessing shared by every analyzer.

    The original pipeline re-loaded the CSV and re-ran ``.apply`` per row for
    every single risk factor (~50 times when a time window is set).  This
    helper does the heavy work exactly once and stamps the resulting DataFrame
    with the columns that ``_is_preprocessed`` looks for, so the analyzers
    take the fast path.

    Adds: ``patient_guid_CLEAN``, ``event_date_parsed``, ``event_type_lower``,
    coerced numeric ``snomed_c_t_concept_id``/``med_code_id``/``event_age``.
    """
    # Track whether to apply the lung-cancer 150-record filter (matches the
    # original semantics: "lungCancer" path-based for CSV inputs, always-on
    # for already-loaded DataFrames coming from BigQuery).
    if isinstance(input_data, pd.DataFrame):
        is_lung_cancer_dataset = True
    else:
        is_lung_cancer_dataset = "lungCancer" in str(input_data)

    # Optional GPU read + clean using cuDF, then back to pandas for analysis
    if USE_GPU and not isinstance(input_data, pd.DataFrame):
        try:
            import cudf  # type: ignore
            print("[GPU] Loading & cleaning with cuDF…")
            gdf = cudf.read_csv(input_data) if isinstance(input_data, str) else cudf.from_pandas(input_data)
            guid_str = gdf['patient_guid'].astype('str')
            guid_clean = (guid_str
                          .str.replace('"""', '', regex=False)
                          .str.replace('{', '', regex=False)
                          .str.replace('}', '', regex=False)
                          .str.strip())
            gdf['patient_guid_CLEAN'] = guid_clean
            gdf['event_date_parsed'] = cudf.to_datetime(gdf['event_date'], errors='coerce')
            for col in ('event_age', 'snomed_c_t_concept_id', 'med_code_id'):
                if col in gdf.columns:
                    gdf[col] = cudf.to_numeric(gdf[col], errors='coerce')
            if 'event_type' in gdf.columns:
                gdf['event_type_lower'] = gdf['event_type'].astype('str').str.lower()
            df = gdf.to_pandas()
        except Exception as gpu_err:  # noqa: BLE001
            print(f"[GPU] cuDF unavailable or failed ({gpu_err}); falling back to pandas.")
            df = pd.read_csv(input_data) if isinstance(input_data, str) else input_data.copy()
            _vectorized_clean_inplace(df)
    else:
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            print(f"Loading data from {input_data}…")
            df = pd.read_csv(input_data)
            print(f"Total records loaded: {len(df):,}")
        _vectorized_clean_inplace(df)

    # Apply the lung-cancer min-records filter once (cheaper than 25x per call)
    if apply_min_records_filter and is_lung_cancer_dataset:
        before = df['patient_guid_CLEAN'].nunique()
        counts = df['patient_guid_CLEAN'].value_counts(dropna=True)
        eligible = counts[counts >= min_records].index
        df = df[df['patient_guid_CLEAN'].isin(eligible)].copy()
        after = df['patient_guid_CLEAN'].nunique()
        print(f"Min-records filter (>= {min_records}): kept {after}/{before} patients, "
              f"{len(df):,} rows.")

    return df


def _vectorized_clean_inplace(df):
    """In-place vectorized cleaning shared by the CPU path."""
    df['patient_guid_CLEAN'] = clean_patient_guid_series(df['patient_guid'])
    df['event_date_parsed'] = pd.to_datetime(df['event_date'], errors='coerce')
    if 'event_age' in df.columns:
        df['event_age'] = pd.to_numeric(df['event_age'], errors='coerce')
    if 'snomed_c_t_concept_id' in df.columns:
        df['snomed_c_t_concept_id'] = pd.to_numeric(df['snomed_c_t_concept_id'], errors='coerce')
    if 'med_code_id' in df.columns:
        df['med_code_id'] = pd.to_numeric(df['med_code_id'], errors='coerce')
    if 'event_type' in df.columns:
        df['event_type_lower'] = df['event_type'].astype(str).str.lower()
    # Backfill the two ethnicity columns the analyzers expect when a CSV only
    # has the single 'patient_ethnicity' column (the BigQuery query splits
    # them into _16 and _6, but local CSVs may not).
    if 'patient_ethnicity_16' not in df.columns:
        df['patient_ethnicity_16'] = df.get('patient_ethnicity', pd.NA)
    if 'patient_ethnicity_6' not in df.columns:
        df['patient_ethnicity_6'] = df.get('patient_ethnicity', pd.NA)


def load_data_from_bigquery(sql_file_path, project_id="prj-cts-ai-dev-sp"):
    """
    Load data from BigQuery by executing a SQL query read from a file.

    Parameters
    ----------
    sql_file_path : str
        Path to the .sql file containing the BigQuery query.
    project_id : str
        Google Cloud project ID used to initialise the BigQuery client.

    Returns
    -------
    pandas.DataFrame
        Query results sorted by patient_guid and event_date.
    """
    from google.cloud import bigquery  # type: ignore

    print(f"Reading SQL query from: {sql_file_path}")
    with open(sql_file_path, 'r', encoding='utf-8') as fh:
        query = fh.read()

    print(f"Connecting to BigQuery project: {project_id}")
    client = bigquery.Client(project=project_id)

    print("Executing query — this may take a few minutes...")
    df = client.query(query).to_dataframe()
    print(f"Retrieved {len(df):,} rows from BigQuery.")

    sort_cols = [c for c in ['patient_guid', 'event_date'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
        print(f"Data sorted by: {sort_cols}")
    else:
        print("Warning: 'patient_guid' or 'event_date' columns not found — skipping sort.")

    return df


def merge_lifetime_and_windowed_results(lifetime_results, windowed_results):
    """
    Merge results from entire lifetime analysis with time-windowed analysis.
    
    Parameters:
    -----------
    lifetime_results : pandas.DataFrame
        Results from analyzing entire patient lifetime
    windowed_results : pandas.DataFrame
        Results from analyzing last N months of patient lifetime
    
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with both sets of features
    """
    if windowed_results.empty:
        return lifetime_results
    if lifetime_results.empty:
        return windowed_results
        
    # Merge on patient identifiers
    merged = pd.merge(
        lifetime_results, 
        windowed_results, 
        on=['patient_guid', 'sex', 'patient_ethnicity_16', 'patient_ethnicity_6', 'patient_age', 'cancer_class'],
        how='outer'
    )
    return merged

def get_ethnicity_dataframe(CSV_FILE):
    if isinstance(CSV_FILE, pd.DataFrame):
        df = CSV_FILE
    else:
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
    df['patient_guid'] = df['patient_guid'].apply(clean_patient_guid)
    
    # Filter non-empty patient_ethnicity
    df_non_empty = df[df['patient_ethnicity'].notna() & (df['patient_ethnicity'].str.strip() != '')][['patient_guid', 'patient_ethnicity']]

    # Get one non-empty patient_ethnicity per patient_guid
    df_one_code = df_non_empty.groupby('patient_guid', as_index=False).first()

    # Merge back with all unique patient_guid to keep users with no patient_ethnicity
    ethnicity_dataframe = df[['patient_guid']].drop_duplicates().merge(df_one_code, on='patient_guid', how='left')

    return ethnicity_dataframe


def analyze_visit_trend(csv_file_path, time_window_months=None):
    """
    Analyze trends of unique visit dates per patient (proxy for doctor visits).
    """
    if isinstance(csv_file_path, pd.DataFrame):
        df = csv_file_path
        if 'patient_guid_CLEAN' not in df.columns or 'event_date_parsed' not in df.columns:
            df = df.copy()
            df['patient_guid_CLEAN'] = clean_patient_guid_series(df['patient_guid'])
            df['event_date_parsed'] = pd.to_datetime(df['event_date'], errors='coerce')
    else:
        print(f"Loading data from {csv_file_path} for visit trend analysis...")
        df = pd.read_csv(csv_file_path)
        print(f"Total records loaded: {len(df)}")
        df['patient_guid_CLEAN'] = clean_patient_guid_series(df['patient_guid'])
        df['event_date_parsed'] = pd.to_datetime(df['event_date'], errors='coerce')

    # Keep only rows with valid dates
    df = df[df['event_date_parsed'].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    results = []
    for patient_guid, patient_data in df.groupby('patient_guid_CLEAN'):
        patient_data_sorted = patient_data.sort_values('event_date_parsed').reset_index(drop=True)

        # Filter by time window if specified (last N months of patient's lifetime)
        if time_window_months is not None:
            max_event_date = patient_data_sorted['event_date_parsed'].max()
            cutoff_date = max_event_date - pd.DateOffset(months=time_window_months)
            patient_data_sorted = patient_data_sorted[
                patient_data_sorted['event_date_parsed'] >= cutoff_date
            ].reset_index(drop=True)
            if len(patient_data_sorted) == 0:
                continue

        # Unique visit dates (date-level, not time)
        unique_dates = pd.Series(
            pd.to_datetime(
                pd.Series(patient_data_sorted['event_date_parsed']).dt.normalize().unique()
            )
        ).sort_values()

        if len(unique_dates) == 0:
            continue

        patient_info = patient_data_sorted.iloc[0]
        num_unique_visits = len(unique_dates)
        first_visit_date = unique_dates.iloc[0]
        last_visit_date = unique_dates.iloc[-1]

        time_span_days = (last_visit_date - first_visit_date).days
        time_span_years = time_span_days / 365.25 if time_span_days > 0 else 0
        frequency_per_year = num_unique_visits / time_span_years if time_span_years > 0 else None

        if num_unique_visits > 1:
            intervals_days = [
                (unique_dates.iloc[i + 1] - unique_dates.iloc[i]).days
                for i in range(len(unique_dates) - 1)
            ]
            mean_interval_days = sum(intervals_days) / len(intervals_days)
            median_interval_days = pd.Series(intervals_days).median()
            min_interval_days = min(intervals_days)
            max_interval_days = max(intervals_days)

            interval_slope, frequency_trend = calculate_frequency_trend(unique_dates.tolist())

            mid_point = len(unique_dates) // 2
            first_half_dates = unique_dates.iloc[:mid_point]
            second_half_dates = unique_dates.iloc[mid_point:]

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
            mean_interval_days = None
            median_interval_days = None
            min_interval_days = None
            max_interval_days = None
            interval_slope = None
            frequency_trend = "Single occurrence"
            first_half_freq = None
            second_half_freq = None

        is_worsening = False
        if frequency_trend:
            is_worsening = "worsening" in frequency_trend.lower() or "increasing" in frequency_trend.lower()
        if not is_worsening and first_half_freq is not None and second_half_freq is not None:
            is_worsening = second_half_freq > first_half_freq * 1.1

        suffix = f'_LAST_{time_window_months}M' if time_window_months is not None else ''
        results.append({
            'patient_guid': patient_guid,
            'sex': patient_info['sex'], 
            'patient_ethnicity_16': patient_info['patient_ethnicity_16'],
            'patient_ethnicity_6': patient_info['patient_ethnicity_6'],
            'patient_age': patient_info['patient_age'],
            'cancer_class': patient_info['cancer_class'],
            #'PRACTICE_ID': patient_info['PRACTICE_ID'],
            #'CANCER_ID': patient_info['CANCER_ID'],
            f'NUM_DR_VISIT_DATES{suffix}': num_unique_visits,
            f'FIRST_DR_VISIT_DATE{suffix}': first_visit_date.strftime('%Y-%m-%d') if pd.notna(first_visit_date) else None,
            f'LAST_DR_VISIT_DATE{suffix}': last_visit_date.strftime('%Y-%m-%d') if pd.notna(last_visit_date) else None,
            f'DR_VISITS_TIME_SPAN_DAYS{suffix}': time_span_days,
            f'DR_VISITS_TIME_SPAN_YEARS{suffix}': round(time_span_years, 2) if time_span_years else None,
            f'DR_VISITS_FREQUENCY_PER_YEAR{suffix}': round(frequency_per_year, 2) if frequency_per_year else None,
            f'DR_VISITS_MEAN_INTERVAL_DAYS{suffix}': round(mean_interval_days, 1) if mean_interval_days is not None else None,
            f'DR_VISITS_MEDIAN_INTERVAL_DAYS{suffix}': round(median_interval_days, 1) if median_interval_days is not None else None,
            f'DR_VISITS_MIN_INTERVAL_DAYS{suffix}': min_interval_days,
            f'DR_VISITS_MAX_INTERVAL_DAYS{suffix}': max_interval_days,
            f'DR_VISITS_FREQUENCY_TREND{suffix}': frequency_trend,
            f'DR_VISITS_FREQUENCY_TREND_SLOPE{suffix}': round(interval_slope, 2) if interval_slope is not None else None,
            f'DR_VISITS_FIRST_HALF_FREQUENCY{suffix}': round(first_half_freq, 2) if first_half_freq else None,
            f'DR_VISITS_SECOND_HALF_FREQUENCY{suffix}': round(second_half_freq, 2) if second_half_freq else None,
            f'DR_VISITS_IS_WORSENING{suffix}': is_worsening
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('patient_guid').reset_index(drop=True)
    return results_df


# ---------------------------------------------------------------------------
# Trend-logic functions (moved to module level so they pickle correctly when
# joblib uses process-based parallelism with the loky backend).
# ---------------------------------------------------------------------------

def _higher_is_better(slope, first_val, last_val, r_value=None):
    """Higher values = improving (e.g. FEV1, oxygen saturation)."""
    if slope > 0.0001:
        return "Improving"
    if slope < -0.0001:
        return "Declining"
    return "Stable"


def _lower_is_better(slope, first_val, last_val, r_value=None):
    """Higher values = declining (e.g. heart rate, BP, neutrophils)."""
    if slope > 0.0001:
        return "Declining"
    if slope < -0.0001:
        return "Improving"
    return "Stable"


# Aliases for readability at the call site.
fev_trend_logic = _higher_is_better
weight_loss_trend_logic = _higher_is_better
oxygen_level_trend_logic = _higher_is_better
alchohol_value_trend_logic = _higher_is_better
neutrophils_trend_logic = _lower_is_better
albumin_trend_logic = _lower_is_better
systolic_blood_pressure_trend_logic = _lower_is_better
diastolic_blood_pressure_trend_logic = _lower_is_better
heart_rate_trend_logic = _lower_is_better
bmi_trend_logic = _lower_is_better


def _run_symptom_task(df, name, codes, event_type, time_window_months):
    """Worker entry-point for symptom occurrence tasks (parallel-friendly)."""
    res, _ = analyze_symptom(
        csv_file=df,
        symptom_name=name,
        snomed_codes=codes,
        event_type=event_type,
        time_window_months=time_window_months,
    )
    return name, res


def _run_value_task(df, name, codes, trend_logic, event_type, time_window_months):
    """Worker entry-point for value-trend tasks (parallel-friendly)."""
    res, _ = analyze_value_trend(
        csv_file=df,
        symptom_name=name,
        snomed_codes=codes,
        trend_logic=trend_logic,
        event_type=event_type,
        time_window_months=time_window_months,
    )
    return name, res


def _run_visit_task(df, time_window_months):
    return 'visit_trend', analyze_visit_trend(df, time_window_months=time_window_months)


def extract_lung_risk_factors(CSV_FILE, time_window_months=None, n_jobs=None,
                              parallel_backend="loky"):
    """
    Extract lung cancer risk factors from medical records.

    Parameters
    ----------
    CSV_FILE : str or pandas.DataFrame
        Path to the CSV file containing medical records, or an already-loaded
        DataFrame (e.g. fetched directly from BigQuery).
    time_window_months : int, optional
        If provided, also analyze records from the last N months of patient's
        lifetime.  If None, only analyze the entire lifetime.
    n_jobs : int, optional
        Number of parallel worker processes for the 25+ risk-factor analyses.
        Default ``None`` reads ``RISK_FACTOR_JOBS`` env var (or all cores).
    parallel_backend : str
        ``"loky"`` (default, true multiprocessing) or ``"threading"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with risk factors analysis for each patient.
    """
    from joblib import Parallel, delayed

    if n_jobs is None:
        n_jobs = N_JOBS

    # ── ONE-TIME vectorized preprocessing (replaces 25× duplicate work) ──
    print(f"\n{'-'*80}\nPREPROCESSING INPUT (vectorized, one-time)\n{'-'*80}")
    df = _preprocess_input(CSV_FILE)
    print(f"Preprocessing complete: {len(df):,} rows, "
          f"{df['patient_guid_CLEAN'].nunique():,} patients.\n")

    # ============================================================================
    # 1- Analyze Smoking
    # ============================================================================
    smoking_snomed_codes = [
        77176002,         # Smoker
        65568007,         # Cigarette smoker
        225323000,        # Smoking cessation advice
        160603005,        # Light cigarette smoker (1-9 cigs/day)
        160604004,        # Moderate cigarette smoker (10-19 cigs/day)
        160605003,        # Heavy cigarette smoker (20-39 cigs/day)
        160606002,        # Very heavy cigarette smoker (40+ cigs/day)
        394873005,        # Not interested in stopping smoking
        1087441000000106, # Smoking cessation programme declined
        527151000000107,  # Smoking cessation advice declined
        365981007,        # Tobacco smoking behaviour - finding
        871641000000105,  # Referral to smoking cessation service declined
        225324006,        # Advice on effects of smoking on health
        767641000000109,  # Referral for smoking cessation service offered
        505651000000103,  # Current smoker annual review
        160625004,        # Date ceased smoking
        266918002,        # Tobacco smoking consumption
        230057008,        # Cigar consumption
        340921000000103   # Asthma trigger - tobacco smoke
    ]

    # ============================================================================
    # 2- Analyze Ex-Smoking
    # ============================================================================
    exsmoking_snomed_codes = [
        8517006,          # Ex-smoker
        281018007,        # Ex-cigarette smoker
        505761000000105,  # Ex-smoker annual review
        266921000,        # Ex-trivial cigarette smoker (<1/day)
        266922007,        # Ex-light cigarette smoker (1-9/day)
        266923002,        # Ex-moderate cigarette smoker (10-19/day)
        266924008,        # Ex-heavy cigarette smoker (20+/day)
        266925009,        # Ex-very heavy cigarette smoker (40+/day)
        492191000000103,  # Ex roll-up cigarette smoker
        266920004         # Trivial cigarette smoker (less than one cigarette/day)
    ]

    # ============================================================================
    # 3- Analyze Alchohol Category
    # ============================================================================
    alchohol_snomed_codes = [
        366371000000105,      # "Brief intervention for excessive alcohol consumption completed"
        366421000000103,      # "Extended intervention for excessive alcohol consumption completed"
        408947007             # "Health education - alcohol"
    ]

    # ============================================================================
    # 4- Analyze COPD
    # ============================================================================
    copd_snomed_codes = [
        1970531000006107,  # COPD assessment test score - cough
        1970561000006103,  # COPD assessment test score - breathless walking up hill/stairs
        1970541000006102,  # COPD assessment test score - phlegm (mucus)"
        446660005,         # Chronic obstructive pulmonary disease assessment test score
        723245007,         # Number of chronic obstructive pulmonary disease exacerbations in past year
        313297008,         # Moderate chronic obstructive pulmonary disease
        313299006,         # Severe chronic obstructive pulmonary disease
        13645005,          # COPD - Chronic obstructive pulmonary disease
        204991000000107    # Suspected chronic obstructive pulmonary disease
    ]

    # ============================================================================
    # 5- Analyze COPD Mild
    # ============================================================================
    mild_copd_snomed_codes = [
        716281000000103,      # Chronic obstructive pulmonary disease monitoring verbal invite
        716901000000101,      # Chronic obstructive pulmonary disease monitoring telephone invitation
        718241000000107,      # Issue of chronic obstructive pulmonary disease rescue pack
        527361000000107,      # CAT - COPD (chronic obstructive pulmonary disease) assessment test
        390891009       # COPD self-management plan given
    ]

    # ============================================================================
    # 6- Analyze Emphysema
    # ============================================================================
    emphysema_snomed_codes = [
        87433001,          # Pulmonary emphysema
        909721000006104,   # [RFC] Reason for care : Emphysema / [RFC] Emphysema
        68328006,          # Emphysema
        263747008          # Emphysema (disorder)
    ]

    # ============================================================================
    # 7- Analyze Fibrosis
    # ============================================================================
    fibrosis_snomed_codes = [
        700250006,          # Idiopathic pulmonary fibrosis
        909731000006101,   # [RFC] Reason for care : Pulmonary fibrosis
        51615001           # Fibrosis of lung
    ]

    # ============================================================================
    # 8- Chest Pain
    # ============================================================================
    chest_pain_snomed_codes = [
        29857009,           # Chest pain
        102589003,          # Atypical chest pain
        2237002             # Pleuritic pain
    ]

    # ============================================================================
    # 9- Chest Infection
    # ============================================================================
    chest_infection_snomed_codes = [
        312342009,           # "Chest infection - pnemonia due to unspecified organism"
        32398004,           # "Chest infection - unspecified bronchitis"
        396285007,          # "Chest infection - unspecified bronchopneumonia"
        54150009,           # "Upper respiratory infection"
        54398005,           # "Acute upper respiratory infection"
        195647007,          # "Acute respiratory infections"
        50417007,           # "Lower respiratory tract infection"
        195742007           # "Acute lower respiratory tract infection"
    ]

    # ============================================================================
    # 10- "Breathlessness"
    # ============================================================================
    breathlessness_snomed_codes = [
        267036007,           # "Breathlessness" - Dyspnoea
        391120009,           # "MRC Breathlessness Scale: grade 1"
        391123006,           # "MRC Breathlessness Scale: grade 2"
        391124000,           # "MRC Breathlessness Scale: grade 3"
        391125004            # "MRC Breathlessness Scale: grade 4"
    ]

    # ============================================================================
    # 11- "Cough"
    # ============================================================================
    cough_snomed_codes = [
        49727002,           # Cough
        161929000,          # "Chesty cough"
        11833005,           # "Dry cough"
        284523002,          # "Persistent cough"
        161947006,          # "Nocturnal cough / wheeze"
        161924005           # "Productive cough -green sputum"
    ]

    # ============================================================================
    # 12- "Red Flags"
    # ============================================================================
    redflags_snomed_codes = [
        161635002,           # History of asbestos exposure
        66857006,            # Haemoptysis
        396484008,           # "Supraclavicular lymph node biopsy
        266924008,           # Ex-heavy cigarette smoker (20-39/day)
        266925009,           # Ex-very heavy cigarette smoker (40+/day)
        160605003,           # Heavy cigarette smoker (20-39 cigs/day)
        160606002,           # Very heavy cigarette smoker (40+ cigs/day)
        427359005,           # Solitary nodule of lung
        831311000006106,     # 2 week rule referral - lung
        168734001,           # Standard chest X-ray abnormal
        276491000000101,     # Fast track referral for suspected lung cancer
        199251000000107,     # Fast track cancer referral
        162573006            # Suspected lung cancer
    ]

    # ============================================================================
    # 13- "Spirometry"
    # ============================================================================
    spirometry_snomed_codes = [
        127783003,           # Spirometry
        171255006,           # Spirometry screening
        415261001,           # Referral for spirometry
        894821000000107      # Spirometry screening invitation
    ]

    # ============================================================================
    # 14- "Ashtma"
    # ============================================================================
    asthma_snomed_codes = [
        195967001,           # Asthma
        810901000000102,           # Asthma self-management plan review
        201031000000108,           # Asthma trigger - respiratory infection
        366874008,      # Number of asthma exacerbations in past year
        443117005,          # Asthma control test score
        736056000,          # Asthma clinical management plan
        401135008,          # Health education - asthma
        919601000000107,    # Single inhaler maintenance and reliever therapy started
        170661005,          # Using inhaled steroids - normal dose
        394700004,          # Asthma annual review
        270442000,          # Asthma monitoring check done
        170614009,          # Inhaler technique observed
        754061000000100,    # Asthma review using Royal College of Physicians three questions
        370208006,          # "Asthma never causes daytime symptoms"
        170635006,          # "Asthma not disturbing sleep"
        406162001,          # "Asthma management"
        370226009,          # "Asthma treatment compliance satisfactory"
        170638008,          # "Asthma not limiting activities"
        394700004,          # "Asthma annual review"
        302331000000106,    # "Royal College of Physicians asthma assessment"
        275908000           # "Asthma monitoring"
    ]

    # ============================================================================
    # MEDICATION ANALYSIS
    # ============================================================================
    # 15- Asthma Medications (Antidepressants commonly prescribed)
    # ============================================================================
    asthma_med_codes = [
        12906411000001100,  # Fostair 100micrograms/dose / 6micrograms/dose inhaler (Chiesi Ltd)
        106511000001103,    # SertSalamol 100micrograms/dose inhaler CFC free (Teva UK Ltd)raline
        3215311000001107,   # Salamol 100micrograms/dose Easi-Breathe inhaler (Teva UK Ltd)
        9516911000001109,   # AeroChamber Plus (Trudell Medical UK Ltd)
        39113611000001102,  # Salbutamol 100micrograms/dose inhaler CFC free
        9205211000001104,   # Easyhaler Salbutamol sulfate 100micrograms/dose dry powder inhaler (Orion Pharma (UK) Ltd)
        222311000001102,    # Ventolin 100micrograms/dose Evohaler (GlaxoSmithKline UK Ltd)
        4053411000001103,   # Serevent 25micrograms/dose Evohaler (GlaxoSmithKline UK Ltd)
        726611000001102,    # Flixotide 50micrograms/dose Evohaler (GlaxoSmithKline UK Ltd)
        2831211000001109,   # Flixotide 250micrograms/dose Evohaler (GlaxoSmithKline UK Ltd)
        3184911000001108,   # Flixotide 250micrograms/dose Accuhaler (GlaxoSmithKline UK Ltd)
        3184311000001107,   # Flixotide 100micrograms/dose Accuhaler (GlaxoSmithKline UK Ltd)
        398511000001105,    # Flixotide 125micrograms/dose Evohaler (GlaxoSmithKline UK Ltd)
        42292311000001106,  # Beclometasone 100micrograms/dose inhaler
        35908811000001103   # Beclometasone 250micrograms/dose inhaler
    ]

    # ============================================================================
    # 16- Analyze FEV values
    # ============================================================================
    fev_codes = [
        251944000,   # FEV1
        313222007,   # FEV1/FVC percent
        407576000,   # FVC/Expected FVC percent
        407602006,   # FEV1/FVC ratio
        313223002    # Percent predicted FEV1
    ]

    # ============================================================================
    # 17- Weight Loss
    # ============================================================================
    weight_loss_codes = [
        27113001    # "Body weight"
    ]

    # ============================================================================
    # 18- Neutrophils
    # ============================================================================
    neutrophils_codes = [
        1022551000000104     # "Neutrophil count"
    ]

    # ============================================================================
    # 19- Albumin
    # ============================================================================
    albumin_codes = [
        1000821000000103     # "Serum albumin level"
    ]

    # ============================================================================
    # 20- Systolic Blood pressure
    # ============================================================================
    systolic_blood_pressure_codes = [
        72313002     # Systolic arterial pressure
    ]

    # ============================================================================
    # 21- Diastolic Blood pressure
    # ============================================================================
    diastolic_blood_pressure_codes = [
        1091811000000102     # Diastolic arterial pressure
    ]

    # ============================================================================
    # 22- Heart rate
    # ============================================================================
    heart_rate_codes = [
        364075005,     # "Heart rate"
        78564009       # "Pulse rate"
    ]

    # ============================================================================
    # 23- BMI
    # ============================================================================
    bmi_codes = [
        60621009     # "Body mass index"
    ]

    # ============================================================================
    # 24- Oxygen level
    # ============================================================================
    oxygen_level_codes = [
        431314004,        # "SpO2 - oxygen saturation at periphery"
        103228002,        # "Haemoglobin saturation with oxygen"
        1017311000000104  # "Blood oxygen saturation (calculated)"
    ]

    # ============================================================================
    # 25- Analyze Alchohol Value
    # ============================================================================
    alchohol_value_codes = [
        897148007,          # "Alcoholic beverage intake"
        1082641000000106,   # "Alcohol units consumed per week"
        160573003,          # "Alcohol consumption"
        1899511000006107,   # "AUDIT-C score - frequency of drinking alcohol"
        763256006,          # "AUDIT-C (Alcohol Use Disorders Identification Test - Consumption) score"
        443280005           # "Alcohol use disorders identification test score"
    ]

    # ============================================================================
    # Build the parallel task list (each tuple = one independent analysis)
    # ============================================================================
    symptom_tasks = [
        # (name,                  codes,                            event_type)
        ('SMOKING',               smoking_snomed_codes,             'observation'),
        ('EXSMOKING',             exsmoking_snomed_codes,           'observation'),
        ('AlchoholCategory',      alchohol_snomed_codes,            'observation'),
        ('COPD',                  copd_snomed_codes,                'observation'),
        ('MildCOPD',              mild_copd_snomed_codes,           'observation'),
        ('Emphysema',             emphysema_snomed_codes,           'observation'),
        ('Fibrosis',              fibrosis_snomed_codes,            'observation'),
        ('ChestPain',             chest_pain_snomed_codes,          'observation'),
        ('ChestInfection',        chest_infection_snomed_codes,     'observation'),
        ('Breathlessness',        breathlessness_snomed_codes,      'observation'),
        ('Cough',                 cough_snomed_codes,               'observation'),
        ('RedFlags',              redflags_snomed_codes,            'observation'),
        ('Spirometry',            spirometry_snomed_codes,          'observation'),
        ('Asthma',                asthma_snomed_codes,              'observation'),
        ('AsthmaMed',             asthma_med_codes,                 'medication'),
    ]

    value_tasks = [
        # (name,                    codes,                          trend_logic,                              event_type)
        ('FEV',                     fev_codes,                      fev_trend_logic,                          'observation'),
        ('WeightLoss',              weight_loss_codes,              weight_loss_trend_logic,                  'observation'),
        ('Neutrophils',             neutrophils_codes,              neutrophils_trend_logic,                  'observation'),
        ('Albumin',                 albumin_codes,                  albumin_trend_logic,                      'observation'),
        ('SystolicBloodPressure',   systolic_blood_pressure_codes,  systolic_blood_pressure_trend_logic,      'observation'),
        ('DiastolicBloodPressure',  diastolic_blood_pressure_codes, diastolic_blood_pressure_trend_logic,     'observation'),
        ('HeartRate',               heart_rate_codes,               heart_rate_trend_logic,                   'observation'),
        ('BMI',                     bmi_codes,                      bmi_trend_logic,                          'observation'),
        ('OxygenLevel',             oxygen_level_codes,             oxygen_level_trend_logic,                 'observation'),
        ('AlchoholValue',           alchohol_value_codes,           alchohol_value_trend_logic,               'observation'),
    ]

    def _build_jobs(window):
        jobs = []
        for name, codes, ev in symptom_tasks:
            jobs.append(delayed(_run_symptom_task)(df, name, codes, ev, window))
        for name, codes, logic, ev in value_tasks:
            jobs.append(delayed(_run_value_task)(df, name, codes, logic, ev, window))
        jobs.append(delayed(_run_visit_task)(df, window))
        return jobs

    # ── Lifetime analyses, all in parallel ─────────────────────────────
    print(f"Running {len(symptom_tasks) + len(value_tasks) + 1} lifetime analyses "
          f"in parallel (n_jobs={n_jobs}, backend={parallel_backend})…")
    lifetime = dict(Parallel(n_jobs=n_jobs, backend=parallel_backend, verbose=0)(
        _build_jobs(None)
    ))

    # ── Optional time-windowed analyses, also in parallel ──────────────
    windowed = {}
    if time_window_months is not None:
        print(f"\n{'='*80}\nPERFORMING TIME-WINDOWED ANALYSIS (Last {time_window_months} months) — parallel\n{'='*80}\n")
        windowed = dict(Parallel(n_jobs=n_jobs, backend=parallel_backend, verbose=0)(
            _build_jobs(time_window_months)
        ))

    # ── Merge lifetime + windowed per risk factor, then combine all ────
    combined = []
    keys = list(lifetime.keys())
    for key in keys:
        lifetime_df = lifetime.get(key, pd.DataFrame())
        if windowed:
            merged = merge_lifetime_and_windowed_results(
                lifetime_df, windowed.get(key, pd.DataFrame())
            )
        else:
            merged = lifetime_df
        if not merged.empty:
            combined.append(merged)

    if not combined:
        return pd.DataFrame()

    # Merge all DataFrames on patient identifiers using an outer join.
    lung_risk_factors_dataframe = reduce(
        lambda left, right: pd.merge(
            left, right,
            on=['patient_guid', 'sex', 'patient_ethnicity_16',
                'patient_ethnicity_6', 'patient_age', 'cancer_class'],
            how='outer',
        ),
        combined,
    )

    return lung_risk_factors_dataframe



def main(time_window_months=None, sql_file=None, project_id="prj-cts-ai-dev-sp"):
    """
    Main function to perform risk factor analysis on cancer and non-cancer patients.

    Parameters
    ----------
    time_window_months : int, optional
        If provided, also analyse records from the last N months of a patient's
        lifetime.  Default is None (entire lifetime analysis only).
        Example: time_window_months=18 for last 18 months analysis.
    sql_file : str, optional
        Path to a .sql file whose query is executed against BigQuery to obtain
        the input dataset.  When None the local CSV file is used instead.
        Example: sql_file='query_v3.sql'
    project_id : str
        Google Cloud project ID used when sql_file is provided.
        Defaults to "prj-cts-ai-dev-sp".
    """

    if sql_file is not None:
        # ── Fetch input data from BigQuery ──────────────────────────────────
        print(f"\n{'='*80}")
        print(f"FETCHING INPUT DATA FROM BIGQUERY USING: {sql_file}")
        print(f"{'='*80}\n")
        CSV_FILE = load_data_from_bigquery(sql_file, project_id=project_id)
        data_source_label = sql_file
    else:
        # ── Read from local CSV file ─────────────────────────────────────────
        #CSV_FILE = '15-lungCancer_noCancer_1_1.csv'
        CSV_FILE = 'tttttt_v2.csv'
        data_source_label = CSV_FILE

    print(f"\n{'='*80}")
    print(f"ANALYZING All PATIENTS: {data_source_label}")
    print(f"{'='*80}\n")
    #ethnicity_dataframe = get_ethnicity_dataframe(CSV_FILE)
    lung_risk_factors_dataframe = extract_lung_risk_factors(CSV_FILE, time_window_months=time_window_months)
    #result_all = pd.merge(lung_risk_factors_dataframe,ethnicity_dataframe, on='patient_guid', how='left')
    result_all = lung_risk_factors_dataframe
    # Remove code/term list helper columns from the final feature set.
    cols_to_remove = [
        c for c in result_all.columns
        if '_SNOMED_CODES' in c or '_terms' in c.lower()
    ]
    if cols_to_remove:
        result_all = result_all.drop(columns=cols_to_remove)

    col = result_all.pop('cancer_class')  # remove the column and return it
    result_all['cancer_class'] = col      # reinsert it at the end

    #output_filename = "16-all_patients_lung1000000000001.csv"
    #output_filename = "15-all_patients_trend_lungCancer_noCancer_1_1_v2.csv"
    output_filename = "all_patients_trend_tttttt_v2.csv"
    result_all.to_csv(output_filename, index=False)
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_filename}")
    print(f"Total patients: {len(result_all)}")
    print(f"Total features: {len(result_all.columns)}")
    if time_window_months is not None:
        print(f"Time window: Last {time_window_months} months")
    print(f"{'='*80}\n")



if __name__ == "__main__":
    import sys
    
    # ============================================================================
    # CONFIGURATION: Data source
    # ============================================================================
    # Option A — fetch from BigQuery (comment out Option B):
    #SQL_FILE   = 'query_v2.sql'          # or 'query_v2.sql'
    SQL_FILE = None
    PROJECT_ID = "prj-cts-ai-dev-sp"
    # Option B — read from a local CSV file (set SQL_FILE = None):
    #SQL_FILE = None

    # ============================================================================
    # CONFIGURATION: Time window
    # ============================================================================
    # Set to None for lifetime analysis only
    # Set to a number (e.g., 18) for both lifetime and windowed analysis
    time_window_months = 11  # Change this value: None, 6, 12, 18, 24, etc.

    # Command-line argument overrides the variable above (if provided)
    if len(sys.argv) > 1:
        try:
            time_window_months = int(sys.argv[1])
            print(f"Using command-line argument: {time_window_months} months")
        except ValueError:
            print(f"Invalid time_window_months value: {sys.argv[1]}")
            print("Usage: python transform_features.py [time_window_months]")
            print("Example: python transform_features.py 18")
            sys.exit(1)
    elif time_window_months is not None:
        print(f"Using configured time window: {time_window_months} months")
    else:
        print("Using lifetime analysis only (no time window)")

    main(time_window_months=time_window_months, sql_file=SQL_FILE, project_id=PROJECT_ID)





# ============================================================================
# Example 3: Analyze Asthma (example - replace with actual SNOMED codes)
# ============================================================================
# asthma_snomed_codes = [
#     195967001,  # Asthma
#     266393001,  # Bronchial asthma
#     # Add more asthma-related SNOMED codes here
# ]
# 
# asthma_results, asthma_records = analyze_symptom(
#     csv_file=CSV_FILE,
#     symptom_name='Asthma',
#     snomed_codes=asthma_snomed_codes
# )


# ============================================================================
# Example 4: Analyze Bronchitis (example - replace with actual SNOMED codes)
# ============================================================================
# bronchitis_snomed_codes = [
#     32398004,   # Chronic bronchitis
#     10509002,   # Acute bronchitis
#     # Add more bronchitis-related SNOMED codes here
# ]
# 
# bronchitis_results, bronchitis_records = analyze_symptom(
#     csv_file=CSV_FILE,
#     symptom_name='Bronchitis',
#     snomed_codes=bronchitis_snomed_codes
# )


# ============================================================================
# Example 5: Custom symptom with custom output prefix
# ============================================================================
# custom_results, custom_records = analyze_symptom(
#     csv_file=CSV_FILE,
#     symptom_name='My Custom Symptom',
#     snomed_codes=[123456789, 987654321],
#     output_prefix='custom_symptom'  # Optional: custom output file prefix
# )


# ============================================================================
# Usage Notes:
# ============================================================================
"""
To analyze any symptom/condition:

1. Identify the SNOMED CT codes for your symptom/condition
2. Call analyze_symptom() with:
   - csv_file: Path to your CSV file
   - symptom_name: Name of the symptom (e.g., 'COPD', 'Emphysema', 'Asthma')
   - snomed_codes: List of SNOMED CT codes for that symptom

3. The function returns:
   - results: DataFrame with summary statistics per patient
   - records: DataFrame with all detailed records

4. Output files are automatically created:
   - {symptom_name}_occurrence_analysis_results.csv
   - {symptom_name}_detailed_records.csv

5. The analysis includes:
   - Number of occurrences per patient
   - Frequency trends (increasing/decreasing/stable)
   - Time spans and intervals
   - Relationship to cancer diagnosis date
   - Whether the condition is worsening over time
"""

