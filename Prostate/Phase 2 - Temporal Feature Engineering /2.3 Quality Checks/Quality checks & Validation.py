"""
This script validates that data is correctly structured for prediction.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

METADATA_COLS = [
    'PATIENT_GUID', 'CANCER_FLAG', 'SEX', 'AGE', 'PATIENT_ETHNICITY',
    'CANCER_ID', 'DATE_OF_DIAGNOSIS', 'AGE_AT_DIAGNOSIS', 
    'AGE_AT_INDEX', 'INDEX_DATE', 'COHORT', 'INDEX'
]

LEAKAGE_COLS = [
    'CANCER_ID', 'DATE_OF_DIAGNOSIS', 'AGE_AT_DIAGNOSIS', 
    'AGE_AT_INDEX', 'INDEX_DATE', 'COHORT'
]


class ValidationResult:
    """Container for validation results."""
    def __init__(self):
        self.passed = True
        self.errors = []
        self.warnings = []
    
    def add_error(self, message):
        self.passed = False
        self.errors.append(message)
    
    def add_warning(self, message):
        self.warnings.append(message)


def validate_index_date_exists(df, result):
    """Check that INDEX_DATE column exists and is populated."""
    if 'INDEX_DATE' not in df.columns:
        result.add_error("INDEX_DATE column not found")
        return
    
    null_count = df['INDEX_DATE'].isnull().sum()
    if null_count > 0:
        result.add_warning(f"{null_count} patients have null INDEX_DATE")


def validate_index_date_correctness(df, result, tolerance_days=5):
    """Validate INDEX_DATE is 12 months before diagnosis for cancer patients."""
    required_cols = ['INDEX_DATE', 'DATE_OF_DIAGNOSIS', 'CANCER_FLAG']
    if not all(col in df.columns for col in required_cols):
        result.add_warning("Cannot validate INDEX_DATE correctness - missing required columns")
        return
    
    cancer_df = df[df['CANCER_FLAG'] == 1].copy()
    if len(cancer_df) == 0:
        return
    
    cancer_patients = cancer_df[['PATIENT_GUID', 'INDEX_DATE', 'DATE_OF_DIAGNOSIS']].drop_duplicates()
    cancer_patients['INDEX_DATE_PARSED'] = pd.to_datetime(cancer_patients['INDEX_DATE'], errors='coerce')
    cancer_patients['DIAGNOSIS_PARSED'] = pd.to_datetime(cancer_patients['DATE_OF_DIAGNOSIS'], errors='coerce')
    cancer_patients['EXPECTED_INDEX'] = cancer_patients['DIAGNOSIS_PARSED'] - pd.DateOffset(months=12)
    cancer_patients['DIFF_DAYS'] = (cancer_patients['INDEX_DATE_PARSED'] - cancer_patients['EXPECTED_INDEX']).dt.days
    
    incorrect = cancer_patients[abs(cancer_patients['DIFF_DAYS']) > tolerance_days]
    if len(incorrect) > 0:
        result.add_error(f"{len(incorrect)} cancer patients have incorrect INDEX_DATE")


def validate_no_leakage_columns(df, result):
    """Check that leakage columns are not in feature set."""
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    leakage_found = [c for c in LEAKAGE_COLS if c in feature_cols]
    
    if leakage_found:
        result.add_error(f"Leakage columns found in features: {leakage_found}")


def validate_target_variable(df, result):
    """Validate CANCER_FLAG target variable."""
    if 'CANCER_FLAG' not in df.columns:
        result.add_error("CANCER_FLAG column not found")
        return
    
    cancer_count = (df['CANCER_FLAG'] == 1).sum()
    control_count = (df['CANCER_FLAG'] == 0).sum()
    
    if control_count == 0:
        result.add_error("No control patients found")
        return
    
    balance = cancer_count / control_count
    if balance < 0.1 or balance > 10.0:
        result.add_warning(f"Imbalanced dataset (ratio: {balance:.3f})")


def validate_feature_completeness(df, result, missing_threshold=0.8):
    """Check feature completeness."""
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    
    if len(feature_cols) == 0:
        result.add_warning("No feature columns found")
        return
    
    missing_pct = df[feature_cols].isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > missing_threshold]
    
    if len(high_missing) > len(feature_cols) * 0.1:
        result.add_warning(f"{len(high_missing)} features have >{missing_threshold*100:.0f}% missing values")


def validate_forward_looking_setup(df, cohort_name="Dataset"):
    """
    Comprehensive validation for forward-looking prediction setup.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset
    cohort_name : str
        Name of dataset for reporting
        
    Returns
    -------
    ValidationResult
        Object containing validation status, errors, and warnings
    """
    result = ValidationResult()
    
    logger.info(f"Validating {cohort_name}: {len(df):,} rows, {len(df.columns)} columns")
    
    validate_index_date_exists(df, result)
    validate_index_date_correctness(df, result)
    validate_no_leakage_columns(df, result)
    validate_target_variable(df, result)
    validate_feature_completeness(df, result)
    
    if result.passed:
        logger.info("Validation passed - dataset ready for modeling")
    else:
        for error in result.errors:
            logger.error(error)
    
    for warning in result.warnings:
        logger.warning(warning)
    
    return result


def validate_file(filepath):
    """
    Validate a CSV file for forward-looking prediction.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
        
    Returns
    -------
    bool
        True if validation passed, False otherwise
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False
    
    df = pd.read_csv(filepath, low_memory=False)
    result = validate_forward_looking_setup(df, os.path.basename(filepath))
    return result.passed


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'prostate_cancer_features_final.csv')
    
    success = validate_file(data_file)
    exit(0 if success else 1)
