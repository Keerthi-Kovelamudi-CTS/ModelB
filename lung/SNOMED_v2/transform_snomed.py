# Input to this file is from this query:  https://github.com/CtheSigns/AI-UK/blob/main/GCP-BigQ/unified_cancer_noncancer_v34.sql 

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


import pandas as pd  # type: ignore
from symptom_occurrence_analysis import analyze_symptom, clean_patient_guid, calculate_frequency_trend, parse_date
from value_trend_analysis import analyze_value_trend, threshold_trend_logic
from functools import reduce


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
        on=['patient_guid', 'sex', 'patient_age', 'cancer_class'],
        how='outer'
    )
    return merged

def get_ethnicity_dataframe(CSV_FILE):
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
    print(f"Loading data from {csv_file_path} for visit trend analysis...")
    df = pd.read_csv(csv_file_path)
    print(f"Total records loaded: {len(df)}")

    # Clean patient_guid and parse dates
    df['patient_guid_CLEAN'] = df['patient_guid'].apply(clean_patient_guid)
    df['event_date_parsed'] = df['event_date'].apply(parse_date)

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


def extract_lung_risk_factors(CSV_FILE, time_window_months=None):
    """
    Extract lung cancer risk factors from medical records.
    
    Parameters:
    -----------
    CSV_FILE : str
        Path to the CSV file containing medical records
    time_window_months : int, optional
        If provided, also analyze records from the last N months of patient's lifetime.
        If None, only analyze the entire lifetime.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with risk factors analysis for each patient
    """

    visit_trend_results = analyze_visit_trend(CSV_FILE)
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

    smoking_results, smoking_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='SMOKING',
        snomed_codes=smoking_snomed_codes
    )

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

    exsmoking_results, exsmoking_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='EXSMOKING',
        snomed_codes=exsmoking_snomed_codes
    )


    # ============================================================================
    # 3- Analyze COPD
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

    copd_results, copd_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='COPD',
        snomed_codes=copd_snomed_codes
    )

    #copd_results.drop(columns=columns_to_drop, axis=1, inplace=True)
    # ============================================================================
    # 4- Analyze COPD Mild
    # ============================================================================
    mild_copd_snomed_codes = [
        716281000000103,      # Chronic obstructive pulmonary disease monitoring verbal invite
        716901000000101,      # Chronic obstructive pulmonary disease monitoring telephone invitation
        718241000000107,      # Issue of chronic obstructive pulmonary disease rescue pack
        527361000000107,      # CAT - COPD (chronic obstructive pulmonary disease) assessment test
        390891009       # COPD self-management plan given
    ]

    mild_copd_results, mild_copd_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='MildCOPD',
        snomed_codes=mild_copd_snomed_codes
    )

    #copd_results.drop(columns=columns_to_drop, axis=1, inplace=True)
    # ============================================================================
    # 5- Analyze Emphysema
    # ============================================================================
    emphysema_snomed_codes = [
        87433001,          # Pulmonary emphysema
        909721000006104,   # [RFC] Reason for care : Emphysema / [RFC] Emphysema
        68328006,          # Emphysema
        263747008          # Emphysema (disorder)
    ]

    emphysema_results, emphysema_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Emphysema',
        snomed_codes=emphysema_snomed_codes
    )
    #emphysema_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 6- Analyze Fibrosis
    # ============================================================================
    fibrosis_snomed_codes = [
        700250006,          # Idiopathic pulmonary fibrosis
        909731000006101,   # [RFC] Reason for care : Pulmonary fibrosis
        51615001           # Fibrosis of lung
    ]

    fibrosis_results, fibrosis_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Fibrosis',
        snomed_codes=fibrosis_snomed_codes
    )
    #fibrosis_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 7- Chest Pain
    # ============================================================================
    chest_pain_snomed_codes = [
        29857009,           # Chest pain
        102589003,          # Atypical chest pain
        2237002             # Pleuritic pain
    ]

    chest_pain_results, chest_pain_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='ChestPain',
        snomed_codes=chest_pain_snomed_codes
    )
    #chest_pain_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 8- Chest Infection
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

    chest_infection_results, chest_infection_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='ChestInfection',
        snomed_codes=chest_infection_snomed_codes
    )
    #chest_infection_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 9- "Breathlessness"
    # ============================================================================
    breathlessness_snomed_codes = [
        267036007,           # "Breathlessness" - Dyspnoea
        391120009,           # "MRC Breathlessness Scale: grade 1"
        391123006,           # "MRC Breathlessness Scale: grade 2"
        391124000,           # "MRC Breathlessness Scale: grade 3"
        391125004            # "MRC Breathlessness Scale: grade 4"
    ]

    breathlessness_results, breathlessness_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Breathlessness',
        snomed_codes=breathlessness_snomed_codes
    )
    #breathlessness_results.drop(columns=columns_to_drop, axis=1, inplace=True)


    # ============================================================================
    # 10- "Cough"
    # ============================================================================
    cough_snomed_codes = [
        49727002,           # Cough
        161929000,          # "Chesty cough"
        11833005,           # "Dry cough"
        284523002,          # "Persistent cough"
        161947006,          # "Nocturnal cough / wheeze"
        161924005           # "Productive cough -green sputum"
    ]

    cough_results, cough_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Cough',
        snomed_codes=cough_snomed_codes
    )

    # ============================================================================
    # 11- "Red Flags"
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

    redflags_results, redflags_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='RedFlags',
        snomed_codes=redflags_snomed_codes
    )
    #redflags_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 12- "Spirometry"
    # ============================================================================
    spirometry_snomed_codes = [
        127783003,           # Spirometry
        171255006,           # Spirometry screening
        415261001,           # Referral for spirometry
        894821000000107      # Spirometry screening invitation
    ]

    spirometry_results, spirometry_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Spirometry',
        snomed_codes=spirometry_snomed_codes
    )
    #spirometry_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 13- "Ashtma"
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

    asthma_results, asthma_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Asthma',
        snomed_codes=asthma_snomed_codes
    )

    # ============================================================================
    # MEDICATION ANALYSIS EXAMPLES
    # ============================================================================
    # Note: For medications, use event_type='medication' and provide medication codes
    
    # ============================================================================
    # 14- Asthma Medications (Antidepressants commonly prescribed)
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
    
    asthma_med_results, asthma_med_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='AsthmaMed',
        snomed_codes=asthma_med_codes,
        event_type='medication'
    )

    # ============================================================================
    # 15- Analyze FEV values
    # ============================================================================

    def fev_trend_logic(slope, first_val, last_val, r_value=None):
        """
        FEV trend logic: Higher FEV values indicate better lung function.
        Increasing values = Improving, Decreasing values = Declining
        """
        if slope > 0.0001:
            return "Improving"
        elif slope < -0.0001:
            return "Declining"
        else:
            return "Stable"


    fev_codes = [
        251944000,   # FEV1
        313222007,   # FEV1/FVC percent
        407576000,   # FVC/Expected FVC percent
        407602006,   # FEV1/FVC ratio
        313223002    # Percent predicted FEV1
    ]

    fev_results, fev_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='FEV',
        snomed_codes=fev_codes,
        trend_logic=fev_trend_logic
    )

    #fev_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 16- Weight Loss
    # ============================================================================

    def weight_loss_trend_logic(slope, first_val, last_val, r_value=None):
        """
        weight_loss trend logic: weight_loss indicate worsening lung function.
        Increasing values = Improving, Decreasing values = Declining
        """
        if slope > 0.0001:
            return "Improving"
        elif slope < -0.0001:
            return "Declining"
        else:
            return "Stable"


    weight_loss_codes = [
        27113001    # "Body weight"
    ]

    weight_loss_results, weight_loss_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='WeightLoss',
        snomed_codes=weight_loss_codes,
        trend_logic=weight_loss_trend_logic
    )

    #weight_loss_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 17- Neutrophils
    # ============================================================================

    def neutrophils_trend_logic(slope, first_val, last_val, r_value=None):
        """
        neutrophils trend logic: higher neutrophils values indicate worsening lung function.
        Increasing values = Declining, Decreasing values = Improving
        """
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"


    neutrophils_codes = [
        1022551000000104     # "Neutrophil count"
    ]

    neutrophils_results, neutrophils_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='Neutrophils',
        snomed_codes=neutrophils_codes,
        trend_logic=neutrophils_trend_logic
    )

    #neutrophils_results.drop(columns=columns_to_drop, axis=1, inplace=True)

    # ============================================================================
    # 16- Albumin
    # ============================================================================

    def albumin_trend_logic(slope, first_val, last_val, r_value=None):
        """
        albumin trend logic: higher albumin values indicate worsening lung function.
        Increasing values = Declining, Decreasing values = Improving
        """
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"


    albumin_codes = [
        1000821000000103     # "Serum albumin level"
    ]

    albumin_results, albumin_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='Albumin',
        snomed_codes=albumin_codes,
        trend_logic=albumin_trend_logic
    )

    #albumin_results.drop(columns=columns_to_drop, axis=1, inplace=True)


    # ============================================================================
    # 17- Systolic Blood pressure
    # ============================================================================

    def systolic_blood_pressure_trend_logic(slope, first_val, last_val, r_value=None):
        """
        systolic blood pressure trend logic: higher systolic blood pressure values indicate worsening lung function.
        Increasing values = Declining, Decreasing values = Improving
        """
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"


    systolic_blood_pressure_codes = [
        72313002     # Systolic arterial pressure
    ]

    systolic_blood_pressure_results, systolic_blood_pressure_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='SystolicBloodPressure',
        snomed_codes=systolic_blood_pressure_codes,
        trend_logic=systolic_blood_pressure_trend_logic
    )

    # ============================================================================
    # 18- Diastolic Blood pressure
    # ============================================================================

    def diastolic_blood_pressure_trend_logic(slope, first_val, last_val, r_value=None):
        """
        diastolic blood pressure trend logic: higher diastolic blood pressure values indicate worsening lung function.
        Increasing values = Declining, Decreasing values = Improving
        """
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"


    diastolic_blood_pressure_codes = [
        1091811000000102     # Diastolic arterial pressure
    ]

    diastolic_blood_pressure_results, diastolic_blood_pressure_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='DiastolicBloodPressure',
        snomed_codes=diastolic_blood_pressure_codes,
        trend_logic=diastolic_blood_pressure_trend_logic
    )


    # ============================================================================
    # 19- Heart rate
    # ============================================================================

    def heart_rate_trend_logic(slope, first_val, last_val, r_value=None):
        """
        heart rate trend logic: higher heart rate values indicate worsening lung function.
        Increasing values = Declining, Decreasing values = Improving
        """
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"


    heart_rate_codes = [
        364075005,     # "Heart rate"
        78564009       # "Pulse rate"
    ]

    heart_rate_results, heart_rate_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='HeartRate',
        snomed_codes=heart_rate_codes,
        trend_logic=heart_rate_trend_logic
    )



    # ============================================================================
    # 20- BMI
    # ============================================================================

    def bmi_trend_logic(slope, first_val, last_val, r_value=None):
        """
        bmi trend logic: higher bmi values indicate worsening lung function.
        Increasing values = Declining, Decreasing values = Improving
        """
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"


    bmi_codes = [
        60621009     # "Body mass index"
    ]

    bmi_results, bmi_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='BMI',
        snomed_codes=bmi_codes,
        trend_logic=bmi_trend_logic
    )



    # ============================================================================
    # 21- Oxygen level
    # ============================================================================

    def oxygen_level_trend_logic(slope, first_val, last_val, r_value=None):
        """
        oxygen level trend logic: higher oxygen level values indicate worsening lung function.
        Increasing values = Declining, Decreasing values = Improving
        """
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"


    oxygen_level_codes = [
        431314004,        # "SpO2 - oxygen saturation at periphery"
        103228002,        # "Haemoglobin saturation with oxygen"
        1017311000000104  # "Blood oxygen saturation (calculated)"
    ]

    oxygen_level_results, oxygen_level_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='OxygenLevel',
        snomed_codes=oxygen_level_codes,
        trend_logic=oxygen_level_trend_logic
    )

    # ============================================================================
    # 22- Doctor visit frequency (unique visit dates)
    # ============================================================================
    #visit_trend_results = analyze_visit_trend(CSV_FILE)


    # List of DataFrames to merge (entire lifetime analysis)
    dfs = [smoking_results, exsmoking_results, copd_results, mild_copd_results, emphysema_results, 
           fibrosis_results, chest_pain_results, chest_infection_results, breathlessness_results, cough_results,
           redflags_results, spirometry_results, asthma_results, asthma_med_results, 
           fev_results, weight_loss_results, neutrophils_results, albumin_results, 
           systolic_blood_pressure_results, diastolic_blood_pressure_results, heart_rate_results, bmi_results, oxygen_level_results, visit_trend_results]

    # Filter out completely empty DataFrames
    dfs = [d for d in dfs if not d.empty]

    # If time_window_months is provided, perform windowed analysis
    if time_window_months is not None:
        print(f"\n{'='*80}")
        print(f"PERFORMING TIME-WINDOWED ANALYSIS (Last {time_window_months} months)")
        print(f"{'='*80}\n")
        
        # Perform all analyses again with time window
        smoking_results_tw, _ = analyze_symptom(CSV_FILE, 'SMOKING', smoking_snomed_codes, time_window_months=time_window_months)
        exsmoking_results_tw, _ = analyze_symptom(CSV_FILE, 'EXSMOKING', exsmoking_snomed_codes, time_window_months=time_window_months)
        copd_results_tw, _ = analyze_symptom(CSV_FILE, 'COPD', copd_snomed_codes, time_window_months=time_window_months)
        mild_copd_results_tw, _ = analyze_symptom(CSV_FILE, 'MildCOPD', mild_copd_snomed_codes, time_window_months=time_window_months)
        emphysema_results_tw, _ = analyze_symptom(CSV_FILE, 'Emphysema', emphysema_snomed_codes, time_window_months=time_window_months)
        fibrosis_results_tw, _ = analyze_symptom(CSV_FILE, 'Fibrosis', fibrosis_snomed_codes, time_window_months=time_window_months)
        chest_pain_results_tw, _ = analyze_symptom(CSV_FILE, 'ChestPain', chest_pain_snomed_codes, time_window_months=time_window_months)
        chest_infection_results_tw, _ = analyze_symptom(CSV_FILE, 'ChestInfection', chest_infection_snomed_codes, time_window_months=time_window_months)
        breathlessness_results_tw, _ = analyze_symptom(CSV_FILE, 'Breathlessness', breathlessness_snomed_codes, time_window_months=time_window_months)
        cough_results_tw, _ = analyze_symptom(CSV_FILE, 'Cough', cough_snomed_codes, time_window_months=time_window_months)
        redflags_results_tw, _ = analyze_symptom(CSV_FILE, 'RedFlags', redflags_snomed_codes, time_window_months=time_window_months)
        spirometry_results_tw, _ = analyze_symptom(CSV_FILE, 'Spirometry', spirometry_snomed_codes, time_window_months=time_window_months)
        asthma_results_tw, _ = analyze_symptom(CSV_FILE, 'Asthma', asthma_snomed_codes, time_window_months=time_window_months)
        asthma_med_results_tw, _ = analyze_symptom(CSV_FILE, 'AsthmaMed', asthma_med_codes, time_window_months=time_window_months, event_type='medication')
        fev_results_tw, _ = analyze_value_trend(CSV_FILE, 'FEV', fev_codes, trend_logic=fev_trend_logic, time_window_months=time_window_months)
        weight_loss_results_tw, _ = analyze_value_trend(CSV_FILE, 'WeightLoss', weight_loss_codes, trend_logic=weight_loss_trend_logic, time_window_months=time_window_months)
        neutrophils_results_tw, _ = analyze_value_trend(CSV_FILE, 'Neutrophils', neutrophils_codes, trend_logic=neutrophils_trend_logic, time_window_months=time_window_months)
        albumin_results_tw, _ = analyze_value_trend(CSV_FILE, 'Albumin', albumin_codes, trend_logic=albumin_trend_logic, time_window_months=time_window_months)
        systolic_blood_pressure_results_tw, _ = analyze_value_trend(CSV_FILE, 'SystolicBloodPressure', systolic_blood_pressure_codes, trend_logic=systolic_blood_pressure_trend_logic, time_window_months=time_window_months)
        diastolic_blood_pressure_results_tw, _ = analyze_value_trend(CSV_FILE, 'DiastolicBloodPressure', diastolic_blood_pressure_codes, trend_logic=diastolic_blood_pressure_trend_logic, time_window_months=time_window_months)
        heart_rate_results_tw, _ = analyze_value_trend(CSV_FILE, 'HeartRate', heart_rate_codes, trend_logic=heart_rate_trend_logic, time_window_months=time_window_months)
        bmi_results_tw, _ = analyze_value_trend(CSV_FILE, 'BMI', bmi_codes, trend_logic=bmi_trend_logic, time_window_months=time_window_months)
        oxygen_level_results_tw, _ = analyze_value_trend(CSV_FILE, 'OxygenLevel', oxygen_level_codes, trend_logic=oxygen_level_trend_logic, time_window_months=time_window_months)
        visit_trend_results_tw = analyze_visit_trend(CSV_FILE, time_window_months=time_window_months)
        
        # Merge lifetime and windowed results for each symptom
        smoking_combined = merge_lifetime_and_windowed_results(smoking_results, smoking_results_tw)
        exsmoking_combined = merge_lifetime_and_windowed_results(exsmoking_results, exsmoking_results_tw)
        copd_combined = merge_lifetime_and_windowed_results(copd_results, copd_results_tw)
        mild_copd_combined = merge_lifetime_and_windowed_results(mild_copd_results, mild_copd_results_tw)
        emphysema_combined = merge_lifetime_and_windowed_results(emphysema_results, emphysema_results_tw)
        fibrosis_combined = merge_lifetime_and_windowed_results(fibrosis_results, fibrosis_results_tw)
        chest_pain_combined = merge_lifetime_and_windowed_results(chest_pain_results, chest_pain_results_tw)
        chest_infection_combined = merge_lifetime_and_windowed_results(chest_infection_results, chest_infection_results_tw)
        breathlessness_combined = merge_lifetime_and_windowed_results(breathlessness_results, breathlessness_results_tw)
        cough_combined = merge_lifetime_and_windowed_results(cough_results, cough_results_tw)
        redflags_combined = merge_lifetime_and_windowed_results(redflags_results, redflags_results_tw)
        spirometry_combined = merge_lifetime_and_windowed_results(spirometry_results, spirometry_results_tw)
        asthma_combined = merge_lifetime_and_windowed_results(asthma_results, asthma_results_tw)
        asthma_med_combined = merge_lifetime_and_windowed_results(asthma_med_results, asthma_med_results_tw)
        fev_combined = merge_lifetime_and_windowed_results(fev_results, fev_results_tw)
        weight_loss_combined = merge_lifetime_and_windowed_results(weight_loss_results, weight_loss_results_tw)
        neutrophils_combined = merge_lifetime_and_windowed_results(neutrophils_results, neutrophils_results_tw)
        albumin_combined = merge_lifetime_and_windowed_results(albumin_results, albumin_results_tw)
        systolic_blood_pressure_combined = merge_lifetime_and_windowed_results(systolic_blood_pressure_results, systolic_blood_pressure_results_tw)
        diastolic_blood_pressure_combined = merge_lifetime_and_windowed_results(diastolic_blood_pressure_results, diastolic_blood_pressure_results_tw)
        heart_rate_combined = merge_lifetime_and_windowed_results(heart_rate_results, heart_rate_results_tw)
        bmi_combined = merge_lifetime_and_windowed_results(bmi_results, bmi_results_tw)
        oxygen_level_combined = merge_lifetime_and_windowed_results(oxygen_level_results, oxygen_level_results_tw)
        visit_trend_combined = merge_lifetime_and_windowed_results(visit_trend_results, visit_trend_results_tw)
        
        # Update dfs list with combined results
        dfs = [smoking_combined, exsmoking_combined, copd_combined, mild_copd_combined, emphysema_combined, 
               fibrosis_combined, chest_pain_combined, chest_infection_combined, breathlessness_combined, cough_combined,
               redflags_combined, spirometry_combined, asthma_combined, asthma_med_combined, 
               fev_combined, weight_loss_combined, neutrophils_combined, albumin_combined, 
               systolic_blood_pressure_combined, diastolic_blood_pressure_combined, heart_rate_combined, bmi_combined, oxygen_level_combined, visit_trend_combined]
        
        # Filter out completely empty DataFrames
        dfs = [d for d in dfs if not d.empty]

    # Merge all DataFrames on patient identifiers using an outer join
    lung_risk_factors_dataframe = reduce(lambda left, right: pd.merge(left, right, on=['patient_guid', 'sex', 'patient_age', 'cancer_class'], how='outer'), dfs)

    return lung_risk_factors_dataframe



def main(time_window_months=None):
    """
    Main function to perform risk factor analysis on cancer and non-cancer patients.
    
    Parameters:
    -----------
    time_window_months : int, optional
        If provided, also analyze records from the last N months of patient's lifetime.
        Default is None (only entire lifetime analysis).
        Example: time_window_months=18 for last 18 months analysis.
    """
    
    
    # Path to your CSV file
    CSV_FILE = '15-lungCancer_noCancer_1_1.csv'
    print(f"\n{'='*80}")
    print(f"ANALYZING All PATIENTS: {CSV_FILE}")
    print(f"{'='*80}\n")
    ethnicity_dataframe = get_ethnicity_dataframe(CSV_FILE)
    lung_risk_factors_dataframe = extract_lung_risk_factors(CSV_FILE, time_window_months=time_window_months)
    result_all = pd.merge(lung_risk_factors_dataframe,ethnicity_dataframe, on='patient_guid', how='left')

    # Remove code/term list helper columns from the final feature set.
    cols_to_remove = [
        c for c in result_all.columns
        if '_SNOMED_CODES' in c or '_terms' in c.lower()
    ]
    if cols_to_remove:
        result_all = result_all.drop(columns=cols_to_remove)

    col = result_all.pop('cancer_class')  # remove the column and return it
    result_all['cancer_class'] = col      # reinsert it at the end

    output_filename = "15-all_patients_trend_lungCancer_noCancer_1_1_v2.csv"
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
    # CONFIGURATION: Set your time window here
    # ============================================================================
    # Set to None for lifetime analysis only
    # Set to a number (e.g., 18) for both lifetime and windowed analysis
    time_window_months = 18  # Change this value: None, 6, 12, 18, 24, etc.
    
    # Command-line argument overrides the variable above (if provided)
    if len(sys.argv) > 1:
        try:
            time_window_months = int(sys.argv[1])
            print(f"Using command-line argument: {time_window_months} months")
        except ValueError:
            print(f"Invalid time_window_months value: {sys.argv[1]}")
            print("Usage: python symptom_trend_analysis.py [time_window_months]")
            print("Example: python symptom_trend_analysis.py 18")
            sys.exit(1)
    elif time_window_months is not None:
        print(f"Using configured time window: {time_window_months} months")
    else:
        print("Using lifetime analysis only (no time window)")
    
    main(time_window_months=time_window_months)


