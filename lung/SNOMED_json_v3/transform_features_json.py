"""
Lung Cancer Risk Factors Analysis — JSON-Packed Output
======================================================

This script analyzes medical records to extract risk factors for lung cancer
patients and writes a CSV whose layout is:

    patient_guid, sex, patient_ethnicity_16, patient_ethnicity_6,
    patient_age, transformed_features, cancer_class

All transformed features (smoking, COPD, FEV, BMI, doctor visits, the
``_LAST_NM`` time-windowed features, ``patient_ethnicity``, ...) are packed
into the single ``transformed_features`` column as a JSON object per row.

Both lifetime and (optionally) time-windowed analyses are supported.

Features
--------
1. ENTIRE LIFETIME ANALYSIS (default):
   - Analyzes all medical records for each patient
   - Generates features like NUM_SMOKING_OCCURRENCES, FIRST_SMOKING_DATE, etc.

2. TIME-WINDOWED ANALYSIS (optional):
   - Analyzes only the last N months of patient's medical history
   - Generates additional features with suffix _LAST_NM
     (e.g., NUM_SMOKING_OCCURRENCES_LAST_18M)

Usage
-----
1. Lifetime analysis only:
       python transform_features_json.py

2. Lifetime + last N months (e.g. 18):
       python transform_features_json.py 18

3. Programmatic usage:
       from transform_features_json import extract_lung_risk_factors_json
       df = extract_lung_risk_factors_json('data.csv', time_window_months=18)
"""

import json
import math
import sys
from datetime import date, datetime
from functools import reduce
from google.cloud import bigquery

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from symptom_occurrence_analysis import (
    analyze_symptom,
    calculate_frequency_trend,
    clean_patient_guid,
    parse_date,
)
from value_trend_analysis import analyze_value_trend


# ============================================================================
# Output layout configuration
# ============================================================================
# Columns kept as standalone (flat) columns at the FRONT of the output.
IDENTIFIER_COLUMN = 'patient_guid'
DEMOGRAPHIC_COLUMNS = [
    'sex',
    'patient_ethnicity_16',
    'patient_ethnicity_6',
    'patient_age',
]
# Column kept as a standalone column at the END of the output.
TRAILING_COLUMN = 'cancer_class'
# Name of the JSON-packed column that carries everything else.
JSON_COLUMN_NAME = 'transformed_features'


# ============================================================================
# Data ingestion helpers
# ============================================================================
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
    """
    if windowed_results.empty:
        return lifetime_results
    if lifetime_results.empty:
        return windowed_results

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

    df_non_empty = df[df['patient_ethnicity'].notna() & (df['patient_ethnicity'].str.strip() != '')][['patient_guid', 'patient_ethnicity']]
    df_one_code = df_non_empty.groupby('patient_guid', as_index=False).first()
    ethnicity_dataframe = df[['patient_guid']].drop_duplicates().merge(df_one_code, on='patient_guid', how='left')

    return ethnicity_dataframe


# ============================================================================
# Visit trend analysis
# ============================================================================
def analyze_visit_trend(csv_file_path, time_window_months=None):
    """
    Analyze trends of unique visit dates per patient (proxy for doctor visits).
    """
    if isinstance(csv_file_path, pd.DataFrame):
        df = csv_file_path
        print(f"Using provided DataFrame ({len(df):,} rows) for visit trend analysis.")
    else:
        print(f"Loading data from {csv_file_path} for visit trend analysis...")
        df = pd.read_csv(csv_file_path)
        print(f"Total records loaded: {len(df)}")

    df['patient_guid_CLEAN'] = df['patient_guid'].apply(clean_patient_guid)
    df['event_date_parsed'] = df['event_date'].apply(parse_date)

    df = df[df['event_date_parsed'].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    results = []
    for patient_guid, patient_data in df.groupby('patient_guid_CLEAN'):
        patient_data_sorted = patient_data.sort_values('event_date_parsed').reset_index(drop=True)

        if time_window_months is not None:
            max_event_date = patient_data_sorted['event_date_parsed'].max()
            cutoff_date = max_event_date - pd.DateOffset(months=time_window_months)
            patient_data_sorted = patient_data_sorted[
                patient_data_sorted['event_date_parsed'] >= cutoff_date
            ].reset_index(drop=True)
            if len(patient_data_sorted) == 0:
                continue

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


# ============================================================================
# Main feature extraction
# ============================================================================
def extract_lung_risk_factors(CSV_FILE, time_window_months=None):
    """
    Extract lung cancer risk factors from medical records and return a wide
    per-feature DataFrame.

    Parameters
    ----------
    CSV_FILE : str or pandas.DataFrame
        Path to the CSV file containing medical records, or an already-loaded
        DataFrame (e.g. fetched directly from BigQuery).
    time_window_months : int, optional
        If provided, also analyze records from the last N months of patient's
        lifetime.  If None, only analyze the entire lifetime.
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
        1970541000006102,  # COPD assessment test score - phlegm (mucus)
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

    # ============================================================================
    # 4- Analyze COPD Mild
    # ============================================================================
    mild_copd_snomed_codes = [
        716281000000103,      # Chronic obstructive pulmonary disease monitoring verbal invite
        716901000000101,      # Chronic obstructive pulmonary disease monitoring telephone invitation
        718241000000107,      # Issue of chronic obstructive pulmonary disease rescue pack
        527361000000107,      # CAT - COPD assessment test
        390891009             # COPD self-management plan given
    ]

    mild_copd_results, mild_copd_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='MildCOPD',
        snomed_codes=mild_copd_snomed_codes
    )

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

    # ============================================================================
    # 6- Analyze Fibrosis
    # ============================================================================
    fibrosis_snomed_codes = [
        700250006,          # Idiopathic pulmonary fibrosis
        909731000006101,    # [RFC] Reason for care : Pulmonary fibrosis
        51615001            # Fibrosis of lung
    ]

    fibrosis_results, fibrosis_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Fibrosis',
        snomed_codes=fibrosis_snomed_codes
    )

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

    # ============================================================================
    # 8- Chest Infection
    # ============================================================================
    chest_infection_snomed_codes = [
        312342009,          # Chest infection - pneumonia due to unspecified organism
        32398004,           # Chest infection - unspecified bronchitis
        396285007,          # Chest infection - unspecified bronchopneumonia
        54150009,           # Upper respiratory infection
        54398005,           # Acute upper respiratory infection
        195647007,          # Acute respiratory infections
        50417007,           # Lower respiratory tract infection
        195742007           # Acute lower respiratory tract infection
    ]

    chest_infection_results, chest_infection_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='ChestInfection',
        snomed_codes=chest_infection_snomed_codes
    )

    # ============================================================================
    # 9- Breathlessness
    # ============================================================================
    breathlessness_snomed_codes = [
        267036007,          # Breathlessness - Dyspnoea
        391120009,          # MRC Breathlessness Scale: grade 1
        391123006,          # MRC Breathlessness Scale: grade 2
        391124000,          # MRC Breathlessness Scale: grade 3
        391125004           # MRC Breathlessness Scale: grade 4
    ]

    breathlessness_results, breathlessness_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Breathlessness',
        snomed_codes=breathlessness_snomed_codes
    )

    # ============================================================================
    # 10- Cough
    # ============================================================================
    cough_snomed_codes = [
        49727002,           # Cough
        161929000,          # Chesty cough
        11833005,           # Dry cough
        284523002,          # Persistent cough
        161947006,          # Nocturnal cough / wheeze
        161924005           # Productive cough -green sputum
    ]

    cough_results, cough_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Cough',
        snomed_codes=cough_snomed_codes
    )

    # ============================================================================
    # 11- Red Flags
    # ============================================================================
    redflags_snomed_codes = [
        161635002,          # History of asbestos exposure
        66857006,           # Haemoptysis
        396484008,          # Supraclavicular lymph node biopsy
        266924008,          # Ex-heavy cigarette smoker (20-39/day)
        266925009,          # Ex-very heavy cigarette smoker (40+/day)
        160605003,          # Heavy cigarette smoker (20-39 cigs/day)
        160606002,          # Very heavy cigarette smoker (40+ cigs/day)
        427359005,          # Solitary nodule of lung
        831311000006106,    # 2 week rule referral - lung
        168734001,          # Standard chest X-ray abnormal
        276491000000101,    # Fast track referral for suspected lung cancer
        199251000000107,    # Fast track cancer referral
        162573006           # Suspected lung cancer
    ]

    redflags_results, redflags_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='RedFlags',
        snomed_codes=redflags_snomed_codes
    )

    # ============================================================================
    # 12- Spirometry
    # ============================================================================
    spirometry_snomed_codes = [
        127783003,          # Spirometry
        171255006,          # Spirometry screening
        415261001,          # Referral for spirometry
        894821000000107     # Spirometry screening invitation
    ]

    spirometry_results, spirometry_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Spirometry',
        snomed_codes=spirometry_snomed_codes
    )

    # ============================================================================
    # 13- Asthma
    # ============================================================================
    asthma_snomed_codes = [
        195967001,          # Asthma
        810901000000102,    # Asthma self-management plan review
        201031000000108,    # Asthma trigger - respiratory infection
        366874008,          # Number of asthma exacerbations in past year
        443117005,          # Asthma control test score
        736056000,          # Asthma clinical management plan
        401135008,          # Health education - asthma
        919601000000107,    # Single inhaler maintenance and reliever therapy started
        170661005,          # Using inhaled steroids - normal dose
        394700004,          # Asthma annual review
        270442000,          # Asthma monitoring check done
        170614009,          # Inhaler technique observed
        754061000000100,    # Asthma review using Royal College of Physicians three questions
        370208006,          # Asthma never causes daytime symptoms
        170635006,          # Asthma not disturbing sleep
        406162001,          # Asthma management
        370226009,          # Asthma treatment compliance satisfactory
        170638008,          # Asthma not limiting activities
        302331000000106,    # Royal College of Physicians asthma assessment
        275908000           # Asthma monitoring
    ]

    asthma_results, asthma_records = analyze_symptom(
        csv_file=CSV_FILE,
        symptom_name='Asthma',
        snomed_codes=asthma_snomed_codes
    )

    # ============================================================================
    # 14- Asthma Medications
    # ============================================================================
    asthma_med_codes = [
        12906411000001100,  # Fostair 100micrograms/dose / 6micrograms/dose inhaler (Chiesi Ltd)
        106511000001103,    # SertSalamol 100micrograms/dose inhaler CFC free (Teva UK Ltd)
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
    # 15- FEV values
    # ============================================================================
    def fev_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher FEV = better lung function. Increasing = Improving."""
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

    # ============================================================================
    # 16- Weight Loss
    # ============================================================================
    def weight_loss_trend_logic(slope, first_val, last_val, r_value=None):
        """Increasing body weight = Improving."""
        if slope > 0.0001:
            return "Improving"
        elif slope < -0.0001:
            return "Declining"
        else:
            return "Stable"

    weight_loss_codes = [
        27113001            # Body weight
    ]

    weight_loss_results, weight_loss_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='WeightLoss',
        snomed_codes=weight_loss_codes,
        trend_logic=weight_loss_trend_logic
    )

    # ============================================================================
    # 17- Neutrophils
    # ============================================================================
    def neutrophils_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher neutrophils = worsening lung function."""
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"

    neutrophils_codes = [
        1022551000000104    # Neutrophil count
    ]

    neutrophils_results, neutrophils_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='Neutrophils',
        snomed_codes=neutrophils_codes,
        trend_logic=neutrophils_trend_logic
    )

    # ============================================================================
    # 18- Albumin
    # ============================================================================
    def albumin_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher albumin trending up here is treated as Declining (per original logic)."""
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"

    albumin_codes = [
        1000821000000103    # Serum albumin level
    ]

    albumin_results, albumin_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='Albumin',
        snomed_codes=albumin_codes,
        trend_logic=albumin_trend_logic
    )

    # ============================================================================
    # 19- Systolic Blood Pressure
    # ============================================================================
    def systolic_blood_pressure_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher systolic BP = Declining."""
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"

    systolic_blood_pressure_codes = [
        72313002            # Systolic arterial pressure
    ]

    systolic_blood_pressure_results, systolic_blood_pressure_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='SystolicBloodPressure',
        snomed_codes=systolic_blood_pressure_codes,
        trend_logic=systolic_blood_pressure_trend_logic
    )

    # ============================================================================
    # 20- Diastolic Blood Pressure
    # ============================================================================
    def diastolic_blood_pressure_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher diastolic BP = Declining."""
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"

    diastolic_blood_pressure_codes = [
        1091811000000102    # Diastolic arterial pressure
    ]

    diastolic_blood_pressure_results, diastolic_blood_pressure_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='DiastolicBloodPressure',
        snomed_codes=diastolic_blood_pressure_codes,
        trend_logic=diastolic_blood_pressure_trend_logic
    )

    # ============================================================================
    # 21- Heart Rate
    # ============================================================================
    def heart_rate_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher heart rate = Declining."""
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"

    heart_rate_codes = [
        364075005,          # Heart rate
        78564009            # Pulse rate
    ]

    heart_rate_results, heart_rate_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='HeartRate',
        snomed_codes=heart_rate_codes,
        trend_logic=heart_rate_trend_logic
    )

    # ============================================================================
    # 22- BMI
    # ============================================================================
    def bmi_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher BMI = Declining (per original logic)."""
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"

    bmi_codes = [
        60621009            # Body mass index
    ]

    bmi_results, bmi_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='BMI',
        snomed_codes=bmi_codes,
        trend_logic=bmi_trend_logic
    )

    # ============================================================================
    # 23- Oxygen level
    # ============================================================================
    def oxygen_level_trend_logic(slope, first_val, last_val, r_value=None):
        """Higher oxygen = Declining (per original logic)."""
        if slope > 0.0001:
            return "Declining"
        elif slope < -0.0001:
            return "Improving"
        else:
            return "Stable"

    oxygen_level_codes = [
        431314004,          # SpO2 - oxygen saturation at periphery
        103228002,          # Haemoglobin saturation with oxygen
        1017311000000104    # Blood oxygen saturation (calculated)
    ]

    oxygen_level_results, oxygen_level_records = analyze_value_trend(
        csv_file=CSV_FILE,
        symptom_name='OxygenLevel',
        snomed_codes=oxygen_level_codes,
        trend_logic=oxygen_level_trend_logic
    )

    # ============================================================================
    # Aggregate all per-feature DataFrames (lifetime analysis)
    # ============================================================================
    dfs = [
        smoking_results, exsmoking_results, copd_results, mild_copd_results, emphysema_results,
        fibrosis_results, chest_pain_results, chest_infection_results, breathlessness_results, cough_results,
        redflags_results, spirometry_results, asthma_results, asthma_med_results,
        fev_results, weight_loss_results, neutrophils_results, albumin_results,
        systolic_blood_pressure_results, diastolic_blood_pressure_results, heart_rate_results,
        bmi_results, oxygen_level_results, visit_trend_results
    ]
    dfs = [d for d in dfs if not d.empty]

    # ============================================================================
    # Optional: time-windowed analysis
    # ============================================================================
    if time_window_months is not None:
        print(f"\n{'='*80}")
        print(f"PERFORMING TIME-WINDOWED ANALYSIS (Last {time_window_months} months)")
        print(f"{'='*80}\n")

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

        dfs = [
            smoking_combined, exsmoking_combined, copd_combined, mild_copd_combined, emphysema_combined,
            fibrosis_combined, chest_pain_combined, chest_infection_combined, breathlessness_combined, cough_combined,
            redflags_combined, spirometry_combined, asthma_combined, asthma_med_combined,
            fev_combined, weight_loss_combined, neutrophils_combined, albumin_combined,
            systolic_blood_pressure_combined, diastolic_blood_pressure_combined, heart_rate_combined,
            bmi_combined, oxygen_level_combined, visit_trend_combined
        ]
        dfs = [d for d in dfs if not d.empty]

    lung_risk_factors_dataframe = reduce(
        lambda left, right: pd.merge(
            left, right,
            on=['patient_guid', 'sex', 'patient_ethnicity_16', 'patient_ethnicity_6', 'patient_age', 'cancer_class'],
            how='outer'
        ),
        dfs
    )

    return lung_risk_factors_dataframe


# ============================================================================
# JSON-packing helpers
# ============================================================================
def _to_json_safe(value):
    """
    Convert a single cell value into a JSON-serializable Python primitive.

    Pandas / NumPy values (NaN, NaT, np.int64, np.float64, Timestamps, ...)
    are not natively JSON-serializable, so we coerce them to plain Python
    types here. Missing values become ``None`` (i.e. JSON ``null``).
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def _row_to_json(row, feature_columns):
    """Build a JSON string for the given row using only ``feature_columns``."""
    payload = {col: _to_json_safe(row[col]) for col in feature_columns}
    return json.dumps(payload, ensure_ascii=False, default=str)


def pack_features_as_json(wide_df):
    """
    Convert the wide per-feature DataFrame into the compact JSON layout:

        patient_guid, sex, patient_ethnicity_16, patient_ethnicity_6,
        patient_age, transformed_features, cancer_class
    """
    if wide_df.empty:
        return pd.DataFrame(
            columns=[IDENTIFIER_COLUMN, *DEMOGRAPHIC_COLUMNS, JSON_COLUMN_NAME, TRAILING_COLUMN]
        )

    front_columns = [
        c for c in [IDENTIFIER_COLUMN, *DEMOGRAPHIC_COLUMNS] if c in wide_df.columns
    ]
    trailing_columns = [c for c in [TRAILING_COLUMN] if c in wide_df.columns]

    reserved = set(front_columns) | set(trailing_columns)
    feature_columns = [c for c in wide_df.columns if c not in reserved]

    packed = wide_df[front_columns].copy()
    packed[JSON_COLUMN_NAME] = wide_df.apply(
        lambda row: _row_to_json(row, feature_columns), axis=1
    )
    for c in trailing_columns:
        packed[c] = wide_df[c].values

    ordered_columns = [*front_columns, JSON_COLUMN_NAME, *trailing_columns]
    return packed[ordered_columns]


def extract_lung_risk_factors_json(csv_file, time_window_months=None):
    """
    Run the full feature extraction pipeline and return the JSON-packed
    DataFrame (one row per patient).
    """
    #ethnicity_dataframe = get_ethnicity_dataframe(csv_file)
    merged = extract_lung_risk_factors(csv_file, time_window_months=time_window_months)
    #merged = pd.merge(wide_df, ethnicity_dataframe, on='patient_guid', how='left')

    cols_to_remove = [
        c for c in merged.columns
        if '_SNOMED_CODES' in c or '_terms' in c.lower()
    ]
    if cols_to_remove:
        merged = merged.drop(columns=cols_to_remove)

    return pack_features_as_json(merged)

def write_to_bigQ(df):
    # Assume df already exists
    df = df.copy()

    # Add partition column
    df["partition_date"] = datetime.now(timezone.utc).date()

    project_id = "prj-cts-ai-dev-sp"
    dataset_id = "prediction_emis"
    table_name = "test_prediction1"

    table_id = f"{project_id}.{dataset_id}.{table_name}"

    client = bigquery.Client(project=project_id)

    schema = [
        bigquery.SchemaField("patient_guid", "STRING"),
        bigquery.SchemaField("sex", "STRING"),
        bigquery.SchemaField("patient_ethnicity_16", "STRING"),
        bigquery.SchemaField("patient_ethnicity_6", "STRING"),
        bigquery.SchemaField("patient_age", "INT64"),
        bigquery.SchemaField("transformed_features", "STRING"),
        bigquery.SchemaField("cancer_class", "INT64"),
        bigquery.SchemaField("partition_date", "DATE"),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="partition_date",
        ),
    )

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=job_config,
    )

    job.result()

    print(f"Loaded {job.output_rows} rows into {table_id}")


# ============================================================================
# Entry point
# ============================================================================
def main(time_window_months=None, sql_file=None, project_id="prj-cts-ai-dev-sp"):
    """
    Main function — runs the full risk-factor analysis and writes a CSV with
    the JSON-packed layout.
    """
    if sql_file is not None:
        print(f"\n{'='*80}")
        print(f"FETCHING INPUT DATA FROM BIGQUERY USING: {sql_file}")
        print(f"{'='*80}\n")
        csv_file = load_data_from_bigquery(sql_file, project_id=project_id)
        data_source_label = sql_file
    else:
        csv_file = 'tttttt_v2.csv'
        data_source_label = csv_file

    print(f"\n{'='*80}")
    print(f"ANALYZING All PATIENTS (JSON-PACKED OUTPUT): {data_source_label}")
    print(f"{'='*80}\n")

    result_packed = extract_lung_risk_factors_json(
        csv_file, time_window_months=time_window_months
    )

    output_filename = "all_patients_trend_tttttt_v2_json.csv"
    result_packed.to_csv(output_filename, index=False)

    write_to_bigQ(result_packed)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_filename}")
    print(f"Total patients: {len(result_packed)}")
    print(f"Columns: {list(result_packed.columns)}")
    if time_window_months is not None:
        print(f"Time window: Last {time_window_months} months")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION: Data source
    # ============================================================================
    # Option A — fetch from BigQuery (set SQL_FILE to a .sql path):
    SQL_FILE = 'query_v2.sql'
    # SQL_FILE = None
    PROJECT_ID = "prj-cts-ai-dev-sp"

    # ============================================================================
    # CONFIGURATION: Time window
    # ============================================================================
    # Set to None for lifetime analysis only.
    # Set to a number (e.g., 18) for both lifetime and windowed analysis.
    time_window_months = 18

    if len(sys.argv) > 1:
        try:
            time_window_months = int(sys.argv[1])
            print(f"Using command-line argument: {time_window_months} months")
        except ValueError:
            print(f"Invalid time_window_months value: {sys.argv[1]}")
            print("Usage: python transform_features_json.py [time_window_months]")
            print("Example: python transform_features_json.py 18")
            sys.exit(1)
    elif time_window_months is not None:
        print(f"Using configured time window: {time_window_months} months")
    else:
        print("Using lifetime analysis only (no time window)")

    main(time_window_months=time_window_months, sql_file=SQL_FILE, project_id=PROJECT_ID)
