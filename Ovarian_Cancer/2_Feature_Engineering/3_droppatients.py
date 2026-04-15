# ════════════════════════════════��══════════════════════════════
# OVARIAN CANCER — FEATURE ENGINEERING PIPELINE
# STEP 3: Data Cleaning
# STEP 4: Feature Engineering
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# STEP 3: DATA CLEANING
# ═══════════════════════════════════════════════════════════════

def clean_clinical(df):
    """Clean clinical dataframe."""
    
    print(f"  Cleaning clinical data...")
    print(f"  Starting rows: {len(df):,}")
    
    # 3a. Fix date columns
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')
    
    # 3b. Clean numeric VALUE column (lab values)
    # Set impossible values to NaN
    lab_mask = df['EVENT_TYPE'] == 'LAB VALUE'
    
    # Clinical ranges for key labs
    lab_ranges = {
        'HAEMATOLOGY': {
            'Haemoglobin concentration': (20, 250),          # g/L
            'Platelet count': (1, 2000),                      # ×10^9/L
            'White blood cell count': (0.1, 500),             # ×10^9/L
            'Red blood cell count': (0.5, 10),                # ×10^12/L
            'MCV - Mean corpuscular volume': (30, 150),       # fL
            'Serum ferritin level': (0, 5000),                # µg/L
            'Serum iron level': (0, 100),                     # µmol/L
            'Percentage hypochromic cells': (0, 100),         # %
        },
        'RENAL': {
            'eGFR using creatinine': (0, 200),                # mL/min
            'Serum creatinine level': (0, 2000),              # µmol/L
        },
        'INFLAMMATORY': {
            'Serum CRP (C reactive protein) level': (0, 500), # mg/L
            'Erythrocyte sedimentation rate': (0, 150),       # mm/hr
        },
        'METABOLIC': {
            'Serum albumin level': (5, 60),                   # g/L
        },
        'LIVER': {
            'Alkaline phosphatase level': (0, 2000),          # U/L
            'Alanine aminotransferase level': (0, 2000),      # U/L
        },
        'HORMONAL': {
            'Serum luteinising hormone level': (0, 200),      # IU/L
            'Serum FSH level': (0, 200),                      # IU/L
            'Plasma FSH level': (0, 200),                     # IU/L
            'Serum prolactin level': (0, 10000),              # mU/L
            'Serum progesterone level': (0, 200),             # nmol/L
            'Serum oestradiol level': (0, 5000),              # pmol/L
            'Serum testosterone level': (0, 50),              # nmol/L
        },
        'CANCER_MARKER': {
            'Serum lactate dehydrogenase level': (0, 5000),   # U/L
        },
        'ELECTROLYTES': {
            'Plasma corrected calcium level': (0.5, 5),       # mmol/L
        },
    }
    
    outliers_removed = 0
    for category, terms in lab_ranges.items():
        for term, (low, high) in terms.items():
            mask = (df['CATEGORY'] == category) & (df['TERM'] == term) & df['VALUE'].notna()
            out_of_range = mask & ((df['VALUE'] < low) | (df['VALUE'] > high))
            outliers_removed += out_of_range.sum()
            df.loc[out_of_range, 'VALUE'] = np.nan
    
    print(f"  Lab outliers set to NaN: {outliers_removed:,}")
    
    # 3c. Encode time windows numerically
    df['TIME_WINDOW_NUM'] = df['TIME_WINDOW'].map({
        'A': 0,      # early window (further from diagnosis)
        'B': 1,      # late window (closer to diagnosis)
        'RF': 2,     # risk factor (all time)
        'COMORB': 3  # comorbidity (all time)
    })
    
    # 3d. Create months_before_index as numeric (already exists but ensure clean)
    df['MONTHS_BEFORE_INDEX'] = pd.to_numeric(df['MONTHS_BEFORE_INDEX'], errors='coerce')
    
    print(f"  Final rows: {len(df):,}")
    return df


def clean_med(df):
    """Clean medication dataframe."""
    
    print(f"  Cleaning medication data...")
    print(f"  Starting rows: {len(df):,}")
    
    # 3a. Fix date columns
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')
    
    # 3b. Clean quantity — remove impossible values
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    impossible_qty = (df['VALUE'].notna()) & ((df['VALUE'] < 0) | (df['VALUE'] > 10000))
    print(f"  Impossible quantities set to NaN: {impossible_qty.sum():,}")
    df.loc[impossible_qty, 'VALUE'] = np.nan
    
    # 3c. DURATION_IN_DAYS is 100% null — drop it
    if 'DURATION_IN_DAYS' in df.columns:
        df = df.drop(columns=['DURATION_IN_DAYS'])
        print(f"  Dropped DURATION_IN_DAYS (100% null)")
    
    # 3d. Encode time windows
    df['TIME_WINDOW_NUM'] = df['TIME_WINDOW'].map({
        'A': 0,
        'B': 1,
    })
    
    # 3e. Clean months_before_index
    df['MONTHS_BEFORE_INDEX'] = pd.to_numeric(df['MONTHS_BEFORE_INDEX'], errors='coerce')
    
    print(f"  Final rows: {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════════
# RUN: load dropped-patients CSVs, clean, save back (for 4_Feature_engineering)
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from pathlib import Path

    DATA_PREFIX = 'ovarian'
    SCRIPT_DIR = Path(__file__).resolve().parent
    RAW_SUFFIX = {'3mo': '3m', '6mo': '6m', '12mo': '12m'}

    for window, suffix in RAW_SUFFIX.items():
        data_dir = SCRIPT_DIR / 'data' / window
        clin_path = data_dir / f'FE_{DATA_PREFIX}_dropped_patients_clinical_windowed_{suffix}.csv'
        med_path = data_dir / f'FE_{DATA_PREFIX}_dropped_patients_med_windowed_{suffix}.csv'

        if not clin_path.exists() or not med_path.exists():
            print(f"⚠ Skipping {window}: missing {clin_path.name} or {med_path.name}")
            continue

        print(f"\n{'═'*60}")
        print(f"  STEP 3 — {window}")
        print(f"{'═'*60}")

        clin = pd.read_csv(clin_path, low_memory=False)
        med = pd.read_csv(med_path, low_memory=False)

        clin = clean_clinical(clin)
        med = clean_med(med)

        clin.to_csv(clin_path, index=False)
        med.to_csv(med_path, index=False)
        print(f"  Saved: {clin_path.name}, {med_path.name}")

    print(f"\n{'═'*60}")
    print("  ✅ STEP 3 COMPLETE — Cleaned dropped-patients data saved.")
    print(f"{'═'*60}")