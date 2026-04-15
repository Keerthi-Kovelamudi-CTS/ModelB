# ═══════════════════════════════════════════════════════════════
# LYMPHOMA CANCER — FEATURE ENGINEERING PIPELINE
# STEP 3: Data Cleaning (dates, lab outliers, time-window encoding)
#
# Input / Output (in-place overwrite):
#   data/{3mo,6mo,12mo}/lymphoma_{window}_{obs,med}_dropped.csv
#
# Lab ranges are based on actual Lymphoma data profiling + clinical bounds:
#   - LDH: critical prognostic marker in lymphoma (IPI score)
#   - beta-2 microglobulin: prognostic in lymphoma
#   - Immunoglobulins: elevated in lymphoplasmacytic lymphomas (Waldenström's)
#   - Complement C3/C4: unit confusion in source data (g/L vs mg/dL) → clamp upper bound
# ═══════════════════════════════════════════════════════════════

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_PREFIX = 'lymphoma'
SCRIPT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════
# LAB RANGES (physiologically plausible upper/lower bounds)
#   Values outside range → set VALUE to NaN (row preserved; just value cleared)
# ═══════════════════════════════════════════════════════════════
LAB_RANGES = {
    'CALCIUM_BONE': {
        'Serum adjusted calcium concentration': (1.0, 4.0),   # mmol/L
        'Plasma calcium level':                  (1.0, 4.0),   # mmol/L
    },
    'LAB_MARKERS': {
        'Serum CRP (C reactive protein) level':  (0, 600),    # mg/L
        'Plasma C reactive protein':             (0, 600),    # mg/L
        'Immunoglobulin A':                      (0, 20),     # g/L
        'Immunoglobulin M':                      (0, 80),     # g/L — very high in Waldenström's
        'Immunoglobulin G':                      (0, 80),     # g/L
        'Glomerular filtration rate':            (0, 200),    # mL/min/1.73m²
        'Serum lactate dehydrogenase level':     (0, 5000),   # U/L — prognostic (IPI)
        'Serum paraprotein level':               (0, 100),    # g/L
        'Plasma total bilirubin level':          (0, 500),    # µmol/L
        'Lambda light chain level':              (0, 2000),   # mg/L
        'Kappa light chain level':               (0, 2000),   # mg/L
        'Kappa/lambda light chain ratio':        (0, 500),
        'Total 25-hydroxyvitamin D level':       (0, 300),    # nmol/L
        'Carcinoembryonic antigen level':        (0, 1000),   # µg/L
        'Serum beta 2 microglobulin level':      (0, 100),    # mg/L — prognostic
        'Monoclonal component level':            (0, 100),    # g/L
        'Paraprotein profile':                   (0, 100),
    },
    'HAEMATOLOGY_INVESTIGATION_SELECTIVE': {
        'Erythrocyte sedimentation rate':        (0, 200),    # mm/hr
        'Plasma viscosity':                      (1.0, 5.0),  # mPa·s
    },
    'HAEMATOLOGICAL_ABNORMALITIES': {
        'Large unstained cells':                             (0, 100),  # %
        'Monoclonal gammopathy of uncertain significance':   (0, 100),
    },
    'AUTOIMMUNE_IMMUNE': {
        # C3/C4 have unit confusion (g/L vs mg/dL) — upper bound keeps both scales,
        # clamps only the clearly-impossible outliers (e.g., 1570).
        'Complement third component - C3':                       (0, 500),
        'Complement fourth component - C4':                      (0, 300),
        'Rheumatoid factor IgM level':                           (0, 500),
        'Rheumatoid arthritis latex test':                       (0, 500),
        'DNA (deoxyribonucleic acid) binding autoantibodies':    (0, 1000),
    },
    'PROTEIN_STUDIES': {
        'Plasma total protein':                  (20, 200),    # g/L
        'Serum protein electrophoresis':         (0, 200),
    },
}


# ═══════════════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def clean_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Clean observations dataframe."""
    print(f"  Cleaning observations...")
    print(f"  Starting rows: {len(df):,}")

    # 3a. Dates → datetime
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')

    # 3b. Numeric VALUE — ensure numeric, then clamp lab outliers to NaN
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    outliers_removed = 0
    for category, terms in LAB_RANGES.items():
        for term, (low, high) in terms.items():
            mask = (df['CATEGORY'] == category) & (df['TERM'] == term) & df['VALUE'].notna()
            oor = mask & ((df['VALUE'] < low) | (df['VALUE'] > high))
            outliers_removed += int(oor.sum())
            df.loc[oor, 'VALUE'] = np.nan
    print(f"  Lab outliers set to NaN: {outliers_removed:,}")

    # 3c. Encode time window numerically (A=0 early, B=1 late)
    df['TIME_WINDOW_NUM'] = df['TIME_WINDOW'].map({'A': 0, 'B': 1})

    # 3d. Ensure MONTHS_BEFORE_INDEX numeric
    df['MONTHS_BEFORE_INDEX'] = pd.to_numeric(df['MONTHS_BEFORE_INDEX'], errors='coerce')

    print(f"  Final rows: {len(df):,}")
    return df


def clean_med(df: pd.DataFrame) -> pd.DataFrame:
    """Clean medications dataframe."""
    print(f"  Cleaning medications...")
    print(f"  Starting rows: {len(df):,}")

    # 3a. Dates → datetime
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')

    # 3b. VALUE numeric + impossible-qty clamp (meds VALUE is usually null, but defensive)
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    impossible = df['VALUE'].notna() & ((df['VALUE'] < 0) | (df['VALUE'] > 10000))
    print(f"  Impossible quantities set to NaN: {int(impossible.sum()):,}")
    df.loc[impossible, 'VALUE'] = np.nan

    # 3c. Drop DURATION_IN_DAYS if present and mostly null (schema sanity)
    if 'DURATION_IN_DAYS' in df.columns:
        null_pct = df['DURATION_IN_DAYS'].isnull().mean()
        if null_pct > 0.95:
            df = df.drop(columns=['DURATION_IN_DAYS'])
            print(f"  Dropped DURATION_IN_DAYS ({null_pct*100:.0f}% null)")

    # 3d. Encode time window
    df['TIME_WINDOW_NUM'] = df['TIME_WINDOW'].map({'A': 0, 'B': 1})

    # 3e. MONTHS_BEFORE_INDEX numeric
    df['MONTHS_BEFORE_INDEX'] = pd.to_numeric(df['MONTHS_BEFORE_INDEX'], errors='coerce')

    print(f"  Final rows: {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════════
# RUN: load *_dropped.csv (from 1_Sanitycheck), clean, overwrite
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    for window in ('3mo', '6mo', '12mo'):
        data_dir = SCRIPT_DIR / 'data' / window
        obs_path = data_dir / f'{DATA_PREFIX}_{window}_obs_dropped.csv'
        med_path = data_dir / f'{DATA_PREFIX}_{window}_med_dropped.csv'

        if not obs_path.exists() or not med_path.exists():
            print(f"⚠ Skipping {window}: missing {obs_path.name} or {med_path.name}")
            continue

        print(f"\n{'═'*60}")
        print(f"  STEP 3 — {window}")
        print(f"{'═'*60}")

        obs = pd.read_csv(obs_path, low_memory=False)
        med = pd.read_csv(med_path, low_memory=False)

        obs = clean_obs(obs)
        med = clean_med(med)

        obs.to_csv(obs_path, index=False)
        med.to_csv(med_path, index=False)
        print(f"  Saved: {obs_path.name}, {med_path.name}")

    print(f"\n{'═'*60}")
    print(f"  ✅ STEP 3 COMPLETE — Cleaned data ready for 4_Feature_engineering.py")
    print(f"{'═'*60}")
