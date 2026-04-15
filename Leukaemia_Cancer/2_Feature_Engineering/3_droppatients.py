# ═══════════════════════════════════════════════════════════════
# LEUKAEMIA CANCER — FEATURE ENGINEERING PIPELINE
# STEP 3: Data Cleaning
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
    # Leukaemia: all events are OBSERVATION type; labs identified by CATEGORY + non-null VALUE
    # Clinical ranges for key labs in leukaemia-relevant categories
    lab_ranges = {
        'HAEM_INVESTIGATIONS': {
            'International normalised ratio': (0.5, 10),                          # INR ratio
            'Erythrocyte sedimentation rate': (0, 150),                           # mm/hr
            'Plasma C reactive protein': (0, 500),                                # mg/L
            'Immunoglobulin A': (0, 50),                                          # g/L
            'Immunoglobulin M': (0, 50),                                          # g/L
            'Immunoglobulin G': (0, 80),                                          # g/L
            'Serum lactate dehydrogenase level': (0, 5000),                       # U/L
            'Prothrombin time': (5, 100),                                         # seconds
            'Partial thromboplastin time': (5, 200),                              # seconds
            'Activated partial thromboplastin time ratio': (0.5, 10),             # ratio
            'Fibrinogen level': (0, 20),                                          # g/L
            'Serum protein electrophoresis': (0, 200),                            # g/L
            'Serum paraprotein level': (0, 100),                                  # g/L
            'Saturation of iron binding capacity': (0, 100),                      # %
            'Plasma viscosity NOS': (1.0, 3.0),                                   # mPa.s
            'Complement fourth component - C4': (0, 100),                         # mg/dL
            'Complement third component - C3': (0, 300),                          # mg/dL
            'Double stranded deoxyribonucleic acid binding autoantibody level': (0, 1000),
            'Derived fibrinogen level': (0, 20),                                  # g/L
            'Kaolin cephalin clotting time': (5, 200),                            # seconds
        },
        'BLOOD_COUNT': {
            'Large unstained cells': (0, 100),                                    # count/percentage
            'Myelocyte count': (0, 100),                                          # x10^9/L
            'Blast cell count': (0, 100),                                         # x10^9/L
            'Promyelocyte count': (0, 100),                                       # x10^9/L
            'Metamyelocyte count': (0, 100),                                      # x10^9/L
        },
        'KEY_BLOOD_TESTS': {
            'Reticulocyte count': (0, 500),                                       # x10^9/L
            'Percentage reticulocyte count': (0, 100),                            # %
            'White blood cell count': (0.1, 500),                                 # x10^9/L
            'Red blood cell folate level': (0, 2000),                             # ng/mL
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

    # 3c. Encode time windows numerically (Leukaemia: only A and B)
    df['TIME_WINDOW_NUM'] = df['TIME_WINDOW'].map({
        'A': 0,      # early window (further from diagnosis)
        'B': 1,      # late window (closer to diagnosis)
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

    # 3c. Drop DURATION_IN_DAYS if present (not expected for leukaemia data)
    if 'DURATION_IN_DAYS' in df.columns:
        null_pct = df['DURATION_IN_DAYS'].isnull().mean()
        if null_pct > 0.95:
            df = df.drop(columns=['DURATION_IN_DAYS'])
            print(f"  Dropped DURATION_IN_DAYS ({null_pct*100:.0f}% null)")

    # 3d. Encode time windows (Leukaemia: only A and B)
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

    DATA_PREFIX = 'leuk'
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
