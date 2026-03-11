"""
CLEANUP: Remove useless features from the matrix
Run AFTER 4_feature_engineering.py
Input:  FE_Results/{window}/bladder_feature_matrix_{window}.csv
Output: cleanupfeatures/{window}/ (bladder_feature_matrix_{window}_cleaned.csv + feature list)
Usage: python 5_feature_cleanup.py [--window 3mo|6mo|12mo|12mo_250k]  (default: 12mo_250k)
"""

import os
import argparse
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Feature cleanup.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
args = parser.parse_args()
WINDOW = args.window

INPUT_FILE = os.path.join(SCRIPT_DIR, 'FE_Results', WINDOW, f'bladder_feature_matrix_{WINDOW}.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'cleanupfeatures', WINDOW)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════
if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Feature matrix not found: {INPUT_FILE}")
print("Loading feature matrix...")
print(f"  Input:  {INPUT_FILE}")
print(f"  Output: {OUTPUT_DIR}")
fm = pd.read_csv(INPUT_FILE, low_memory=False)
fm.set_index('PATIENT_GUID', inplace=True)

label = fm['LABEL']
features = fm.drop(columns=['LABEL'])

n_before = features.shape[1]
print(f"BEFORE cleanup: {n_before} features")

# ══════════════════════════════════════════════════════════════
# STEP 1: Drop features with >95% NaN
# ══════════════════════════════════════════════════════════════
nan_pct = features.isna().sum() / len(features) * 100
high_nan = nan_pct[nan_pct > 95].index.tolist()
print(f"\nDropping {len(high_nan)} features with >95% NaN:")
for c in sorted(high_nan):
    print(f"  {c:60s} NaN: {nan_pct[c]:.1f}%")
features.drop(columns=high_nan, inplace=True)

# ══════════════════════════════════════════════════════════════
# STEP 2: Drop features with >99% zero (after filling NaN→0)
# ══════════════════════════════════════════════════════════════
temp = features.fillna(0)
zero_pct = (temp == 0).sum() / len(temp) * 100
high_zero = zero_pct[zero_pct > 99].index.tolist()

# Don't drop lab value features (NaN is meaningful for labs)
lab_cols = [c for c in high_zero if c.startswith('LAB_') and not c.endswith(('_DURATION', '_TOTAL_DURATION'))]
high_zero = [c for c in high_zero if c not in lab_cols]

print(f"\nDropping {len(high_zero)} features with >99% zero:")
for c in sorted(high_zero):
    print(f"  {c:60s} zero: {zero_pct[c]:.1f}%")
features.drop(columns=high_zero, inplace=True)

# ══════════════════════════════════════════════════════════════
# STEP 3: Drop features with zero variance
# ══════════════════════════════════════════════════════════════
numeric_cols = features.select_dtypes(include=[np.number]).columns
variances = features[numeric_cols].var()
zero_var = variances[variances == 0].index.tolist()
print(f"\nDropping {len(zero_var)} features with zero variance:")
for c in sorted(zero_var):
    print(f"  {c}")
features.drop(columns=zero_var, inplace=True)

# ══════════════════════════════════════════════════════════════
# STEP 4: Check for highly correlated duplicates (>0.99)
# ══════════════════════════════════════════════════════════════
print(f"\nChecking for near-duplicate features (corr > 0.99)...")
numeric_feats = features.select_dtypes(include=[np.number])

# Sample for speed (correlation on 63K × 700 is slow)
sample = numeric_feats.sample(min(10000, len(numeric_feats)), random_state=42)
corr_matrix = sample.corr().abs()

# Find pairs
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = []
for col in upper.columns:
    highly_corr = upper.index[upper[col] > 0.99].tolist()
    for hc in highly_corr:
        # Keep the one with more info (less NaN)
        if features[col].isna().sum() <= features[hc].isna().sum():
            to_drop_corr.append(hc)
        else:
            to_drop_corr.append(col)

to_drop_corr = list(set(to_drop_corr))
print(f"Dropping {len(to_drop_corr)} near-duplicate features (corr > 0.99):")
for c in sorted(to_drop_corr)[:30]:  # show first 30
    print(f"  {c}")
if len(to_drop_corr) > 30:
    print(f"  ... and {len(to_drop_corr) - 30} more")
features.drop(columns=to_drop_corr, inplace=True, errors='ignore')

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("AFTER CLEANUP")
print("=" * 70)
print(f"Features: {features.shape[1]}")

# Recount groups
groups = {
    'Demographics': [c for c in features.columns if c.startswith(('AGE_', 'SEX_'))],
    'Observations': [c for c in features.columns if c.startswith('OBS_')],
    'Haematuria': [c for c in features.columns if c.startswith('HAEM_')],
    'Urine': [c for c in features.columns if c.startswith('URINE_')],
    'LUTS': [c for c in features.columns if c.startswith('LUTS_')],
    'Catheter/Img/Uro/Gynae': [c for c in features.columns if c.startswith(('CATH_', 'IMG_', 'URO_', 'GYNAE_', 'OTHER_'))],
    'Risk Factors': [c for c in features.columns if c.startswith('RF_')],
    'Comorbidities': [c for c in features.columns if c.startswith('COMORB_')],
    'Lab Values': [c for c in features.columns if c.startswith('LAB_')],
    'Medications': [c for c in features.columns if c.startswith('MED_')],
    'Temporal': [c for c in features.columns if c.startswith('TEMP_')],
    'Pathways': [c for c in features.columns if c.startswith('PATHWAY_')],
    'Interactions': [c for c in features.columns if c.startswith('INT_')],
    'Ratios': [c for c in features.columns if c.startswith('RATIO_')],
}

for group, cols in groups.items():
    print(f"  {group:30s}: {len(cols):4d}")

# NaN check
nan_pct = (features.isna().sum() / len(features) * 100).sort_values(ascending=False)
print(f"\nRemaining features with NaN (top 10):")
print(nan_pct[nan_pct > 0].head(10).round(1))

# ══════════════════════════════════════════════════════════════
# SAVE CLEANED VERSION
# ══════════════════════════════════════════════════════════════
cleaned = features.copy()
cleaned['LABEL'] = label

output_path = os.path.join(OUTPUT_DIR, f'bladder_feature_matrix_{WINDOW}_cleaned.csv')
cleaned.reset_index().to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"  Shape: {cleaned.shape}")

# Feature list
feature_list_path = os.path.join(OUTPUT_DIR, f'bladder_feature_list_{WINDOW}_cleaned.txt')
with open(feature_list_path, 'w') as f:
    f.write(f"Total features: {features.shape[1]}\n\n")
    for i, col in enumerate(sorted(features.columns), 1):
        nan_p = nan_pct.get(col, 0)
        f.write(f"{i:4d}. {col:60s} NaN: {nan_p:.1f}%\n")
print(f"Feature list saved to: {feature_list_path}")

print("\n" + "=" * 70)
print("CLEANUP COMPLETE ✅")
print(f"  BEFORE: {n_before} features")
print(f"  AFTER:  {features.shape[1]} features")
print("=" * 70)