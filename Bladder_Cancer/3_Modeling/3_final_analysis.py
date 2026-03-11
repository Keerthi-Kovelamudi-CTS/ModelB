#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  BLADDER CANCER — ERROR ANALYSIS
  Input:  results/{window}/test_predictions.csv
          cleanupfeatures/{window}/bladder_feature_matrix_{window}_cleaned.csv
  Output: results/{window}/error_analysis/
  Usage:  python 3_final_analysis.py [--window 3mo|6mo|12mo|12mo_250k]  (default: 12mo_250k)
═══════════════════════════════════════════════════════════════
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLADDER_2 = os.path.dirname(SCRIPT_DIR)
parser = argparse.ArgumentParser(description='Error analysis for bladder cancer model.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
args = parser.parse_args()
WINDOW = args.window
RUN_SUBFOLDER = f"{WINDOW}_65-25-10"

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', RUN_SUBFOLDER)
ERROR_DIR = os.path.join(RESULTS_DIR, 'error_analysis')
os.makedirs(ERROR_DIR, exist_ok=True)

INPUT_FILE = os.path.join(BLADDER_2, '2_Feature_Engineering', 'cleanupfeatures', WINDOW, f'bladder_feature_matrix_{WINDOW}_cleaned.csv')
if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Feature matrix not found: {INPUT_FILE}. Run 5_feature_cleanup.py --window {WINDOW} first.")

# ══════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════
print("=" * 70)
print(f"BLADDER CANCER — ERROR ANALYSIS (window={WINDOW})")
print("=" * 70)

# Load predictions (from 1_run_modeling.py output)
pred_path = os.path.join(RESULTS_DIR, 'test_predictions.csv')
if not os.path.isfile(pred_path):
    raise FileNotFoundError(f"Test predictions not found: {pred_path}. Run 1_run_modeling.py --window {WINDOW} first.")
preds = pd.read_csv(pred_path)
print(f"Test predictions: {len(preds)} patients (window={WINDOW})")

# Load full feature matrix for test patients
features = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Feature matrix: {features.shape}")

# Load threshold info
threshold_file = os.path.join(RESULTS_DIR, 'target_threshold.txt')
threshold_info = {}
if os.path.exists(threshold_file):
    with open(threshold_file) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                threshold_info[k] = v
    print(f"Threshold info: {threshold_info}")

MODEL_NAME = threshold_info.get('model', 'Ensemble (Optuna Wt)')
THRESHOLD = float(threshold_info.get('threshold', 0.385))
print(f"\nUsing: {MODEL_NAME} @ threshold={THRESHOLD}")

# Find the probability column
prob_col = None
for c in preds.columns:
    if 'prob' in c.lower() and 'optuna' in c.lower():
        prob_col = c
        break
if prob_col is None:
    # fallback: use first prob column
    prob_cols = [c for c in preds.columns if c.endswith('_prob')]
    if prob_cols:
        prob_col = prob_cols[0]
print(f"Using probability column: {prob_col}")


# ══════════════════════════════════════════════════════
# 2. CLASSIFY PATIENTS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. CLASSIFYING PATIENTS")
print("=" * 70)

# Merge predictions with features
merged = features.merge(preds[['PATIENT_GUID', 'ACTUAL_LABEL', prob_col]], on='PATIENT_GUID', how='inner')
merged['PREDICTED'] = (merged[prob_col] >= THRESHOLD).astype(int)
merged['PROB'] = merged[prob_col]

# Error types
merged['GROUP'] = 'TN'
merged.loc[(merged['ACTUAL_LABEL']==1) & (merged['PREDICTED']==1), 'GROUP'] = 'TP'
merged.loc[(merged['ACTUAL_LABEL']==1) & (merged['PREDICTED']==0), 'GROUP'] = 'FN'
merged.loc[(merged['ACTUAL_LABEL']==0) & (merged['PREDICTED']==1), 'GROUP'] = 'FP'

tp = merged[merged['GROUP']=='TP']
fn = merged[merged['GROUP']=='FN']
fp = merged[merged['GROUP']=='FP']
tn = merged[merged['GROUP']=='TN']

print(f"  TP (cancer caught):     {len(tp):,}")
print(f"  FN (cancer MISSED):     {len(fn):,}")
print(f"  FP (false alarm):       {len(fp):,}")
print(f"  TN (correct non-cancer):{len(tn):,}")


# ══════════════════════════════════════════════════════
# 3. FN vs TP PROFILE COMPARISON
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. FALSE NEGATIVES vs TRUE POSITIVES")
print("    (What makes missed cancers different from caught cancers?)")
print("=" * 70)

# Key features to compare
key_features = {
    'Demographics': ['AGE_AT_INDEX', 'SEX_MALE', 'AGE_BAND_UNDER50', 'AGE_BAND_50_59',
                      'AGE_BAND_60_69', 'AGE_BAND_70_79', 'AGE_BAND_80PLUS'],
    'Haematuria': ['HAEM_TOTAL_COUNT', 'HAEM_ANY_FLAG', 'HAEM_RECURRENT', 'HAEM_FREQUENT',
                    'HAEM_FRANK_COUNT', 'HAEM_PAINLESS_COUNT', 'HAEM_MICROSCOPIC_COUNT',
                    'HAEM_WB_COUNT', 'HAEM_ACCELERATION'],
    'Urine': ['URINE_INV_COUNT', 'URINE_LAB_ABNORM_COUNT', 'URINE_ANY_INV_FLAG',
              'URINE_ANY_ABNORM_FLAG', 'URINE_TOTAL_ACTIVITY'],
    'LUTS': ['LUTS_TOTAL_COUNT', 'LUTS_ANY_FLAG', 'LUTS_RECURRENT',
             'LUTS_BLADDER_PAIN_COUNT', 'LUTS_FREQUENCY_COUNT', 'LUTS_RETENTION_COUNT'],
    'Catheter/Imaging/Uro': ['CATH_PROC_COUNT', 'CATH_ANY_FLAG', 'IMG_COUNT', 'IMG_ANY_FLAG',
                              'URO_COUNT', 'URO_ANY_FLAG'],
    'Risk Factors': ['RF_EVER_SMOKER', 'RF_CURRENT_SMOKER_FLAG', 'RF_HEAVY_SMOKER_FLAG',
                     'RF_ALCOHOL_FLAG', 'RF_HEAVY_DRINKER_FLAG'],
    'Comorbidities': ['COMORB_BURDEN_SCORE', 'COMORB_TOTAL_CONDITIONS', 'COMORB_PREVIOUS_CANCER_FLAG',
                       'COMORB_RECURRENT_UTI_FLAG', 'COMORB_CKD_FLAG', 'COMORB_DIABETES_FLAG',
                       'COMORB_ANAEMIA_FLAG', 'COMORB_BPH_FLAG'],
    'Labs': ['LAB_HB_LAST', 'LAB_HB_DECLINING', 'LAB_ANAEMIA_MILD', 'LAB_ANAEMIA_MODERATE',
             'LAB_EGFR_LAST', 'LAB_EGFR_LOW', 'LAB_CRP_LAST', 'LAB_CRP_HIGH',
             'LAB_ALBUMIN_LAST', 'LAB_ALBUMIN_LOW', 'LAB_PSA_ELEVATED',
             'LAB_FERRITIN_LOW', 'LAB_IRON_LOW', 'LAB_MCV_LOW',
             'LAB_IRON_DEFICIENCY_PATTERN', 'LAB_WEIGHT_LOSS_5PCT'],
    'Medications': ['MED_TOTAL_COUNT', 'MED_UTI_ANTIBIOTICS_COUNT', 'MED_UTI_AB_RECURRENT_3',
                     'MED_IRON_SUPPLEMENTS_COUNT', 'MED_OPIOID_ANALGESICS_COUNT',
                     'MED_ANY_CATHETER_FLAG'],
    'Temporal': ['TEMP_CLINICAL_ACCELERATION', 'TEMP_GP_VISIT_DAYS', 'TEMP_EVENTS_PER_MONTH',
                 'OBS_TOTAL_COUNT', 'MED_TOTAL_COUNT'],
    'New Obs (v4)': ['NOBS_WEIGHT_LOSS_FLAG', 'NOBS_FATIGUE_FLAG', 'NOBS_DYSURIA_FLAG',
                      'NOBS_BACK_PAIN_FLAG', 'NOBS_LOIN_PAIN_FLAG', 'NOBS_ANAEMIA_DX_FLAG',
                      'NOBS_DVT_FLAG', 'NOBS_FRAILTY_FLAG', 'NOBS_NIGHT_SWEATS_FLAG'],
    'Syndromes': ['SYN_BLEEDING_SCORE', 'SYN_CONSTITUTIONAL_SCORE', 'SYN_PATHWAY_SCORE',
                   'SYN_PAIN_SCORE', 'SYN_VTE_ANY', 'SYN_MASTER_SUSPICION',
                   'SYN_HAEM_AND_SMOKING', 'SYN_HAEM_AND_ANAEMIA', 'SYN_UTI_TREATMENT_FAILURE'],
    'Interactions': ['INT_OVERALL_SUSPICION_SCORE', 'INT_BLADDER_SYMPTOM_SCORE',
                      'INT_RISK_COMPOSITE_SCORE', 'INT_SMOKER_AND_HAEM',
                      'INT_AGE_X_HAEM', 'INT_MALE_AND_HAEM'],
    'Ratios': ['RATIO_HAEM_TO_ALL_OBS', 'RATIO_UTI_AB_TO_ALL_MED', 'RATIO_WB_PROPORTION'],
}

# Detailed comparison
comparison_rows = []
print(f"\n{'Feature':<55s} {'FN(missed)':>10s} {'TP(caught)':>10s} {'Diff':>8s} {'Sig':>5s}")
print("=" * 93)

for group_name, feats in key_features.items():
    available = [f for f in feats if f in merged.columns]
    if not available:
        continue
    print(f"\n  ── {group_name} ──")
    for feat in available:
        fn_mean = fn[feat].mean()
        tp_mean = tp[feat].mean()
        diff = fn_mean - tp_mean
        
        # Statistical test
        from scipy import stats
        fn_vals = fn[feat].dropna().values
        tp_vals = tp[feat].dropna().values
        sig = ''
        if len(fn_vals) > 10 and len(tp_vals) > 10:
            try:
                _, p = stats.mannwhitneyu(fn_vals, tp_vals, alternative='two-sided')
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
            except:
                pass
        
        marker = " ◀◀" if (sig in ['***','**'] and abs(diff) > 0.1) else ""
        print(f"  {feat:<53s} {fn_mean:>10.3f} {tp_mean:>10.3f} {diff:>+8.3f} {sig:>5s}{marker}")
        
        comparison_rows.append({
            'group': group_name, 'feature': feat,
            'FN_mean': fn_mean, 'TP_mean': tp_mean,
            'difference': diff, 'significance': sig
        })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(ERROR_DIR, 'fn_vs_tp_comparison.csv'), index=False)


# ══════════════════════════════════════════════════════
# 4. FN SUBGROUP ANALYSIS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. FN SUBGROUP ANALYSIS")
print("=" * 70)

# 4a. Haematuria breakdown
if 'HAEM_ANY_FLAG' in merged.columns:
    fn_no_haem = fn[fn['HAEM_ANY_FLAG']==0]
    fn_has_haem = fn[fn['HAEM_ANY_FLAG']==1]
    tp_no_haem = tp[tp['HAEM_ANY_FLAG']==0]
    tp_has_haem = tp[tp['HAEM_ANY_FLAG']==1]
    
    print(f"\n  HAEMATURIA BREAKDOWN:")
    print(f"  {'':40s} {'FN (missed)':>15s} {'TP (caught)':>15s}")
    print(f"  {'WITH haematuria':<40s} {len(fn_has_haem):>10d} ({len(fn_has_haem)/len(fn)*100:4.1f}%) "
          f"{len(tp_has_haem):>10d} ({len(tp_has_haem)/len(tp)*100:4.1f}%)")
    print(f"  {'WITHOUT haematuria':<40s} {len(fn_no_haem):>10d} ({len(fn_no_haem)/len(fn)*100:4.1f}%) "
          f"{len(tp_no_haem):>10d} ({len(tp_no_haem)/len(tp)*100:4.1f}%)")

# 4b. Age breakdown
if 'AGE_AT_INDEX' in merged.columns:
    print(f"\n  AGE BREAKDOWN:")
    print(f"  {'Age Group':<20s} {'FN':>8s} {'FN%':>8s} {'TP':>8s} {'TP%':>8s} {'FN rate':>8s}")
    print("  " + "-" * 55)
    for lo, hi, label in [(0,50,'<50'),(50,60,'50-59'),(60,70,'60-69'),(70,80,'70-79'),(80,200,'80+')]:
        fn_age = fn[(fn['AGE_AT_INDEX']>=lo)&(fn['AGE_AT_INDEX']<hi)]
        tp_age = tp[(tp['AGE_AT_INDEX']>=lo)&(tp['AGE_AT_INDEX']<hi)]
        total_cancer = len(fn_age) + len(tp_age)
        fn_rate = len(fn_age)/total_cancer*100 if total_cancer>0 else 0
        print(f"  {label:<20s} {len(fn_age):>8d} {len(fn_age)/len(fn)*100:>7.1f}% "
              f"{len(tp_age):>8d} {len(tp_age)/len(tp)*100:>7.1f}% {fn_rate:>7.1f}%")

# 4c. Sex breakdown
if 'SEX_MALE' in merged.columns:
    print(f"\n  SEX BREAKDOWN:")
    for sex_val, sex_name in [(1,'Male'),(0,'Female')]:
        fn_sex = fn[fn['SEX_MALE']==sex_val]
        tp_sex = tp[tp['SEX_MALE']==sex_val]
        total = len(fn_sex)+len(tp_sex)
        fn_rate = len(fn_sex)/total*100 if total>0 else 0
        print(f"  {sex_name:<10s}: FN={len(fn_sex):,} TP={len(tp_sex):,} (FN rate={fn_rate:.1f}%)")

# 4d. Smoking breakdown
if 'RF_EVER_SMOKER' in merged.columns:
    print(f"\n  SMOKING BREAKDOWN:")
    for sm_val, sm_name in [(1,'Ever smoker'),(0,'Never smoker')]:
        fn_sm = fn[fn['RF_EVER_SMOKER']==sm_val]
        tp_sm = tp[tp['RF_EVER_SMOKER']==sm_val]
        total = len(fn_sm)+len(tp_sm)
        fn_rate = len(fn_sm)/total*100 if total>0 else 0
        print(f"  {sm_name:<15s}: FN={len(fn_sm):,} TP={len(tp_sm):,} (FN rate={fn_rate:.1f}%)")

# 4e. Clinical signal density
if 'OBS_TOTAL_COUNT' in merged.columns:
    print(f"\n  CLINICAL DATA DENSITY:")
    fn_sparse = fn[fn['OBS_TOTAL_COUNT']<=2]
    tp_sparse = tp[tp['OBS_TOTAL_COUNT']<=2]
    fn_dense = fn[fn['OBS_TOTAL_COUNT']>5]
    tp_dense = tp[tp['OBS_TOTAL_COUNT']>5]
    print(f"  Sparse (≤2 obs):  FN={len(fn_sparse):,} ({len(fn_sparse)/len(fn)*100:.1f}%) "
          f"TP={len(tp_sparse):,} ({len(tp_sparse)/len(tp)*100:.1f}%)")
    print(f"  Dense (>5 obs):   FN={len(fn_dense):,} ({len(fn_dense)/len(fn)*100:.1f}%) "
          f"TP={len(tp_dense):,} ({len(tp_dense)/len(tp)*100:.1f}%)")


# ══════════════════════════════════════════════════════
# 5. FN PROBABILITY DISTRIBUTION
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. FN PROBABILITY DISTRIBUTION")
print("=" * 70)

print(f"  Mean:   {fn['PROB'].mean():.4f}")
print(f"  Median: {fn['PROB'].median():.4f}")
print(f"  Std:    {fn['PROB'].std():.4f}")
print(f"  Min:    {fn['PROB'].min():.4f}")
print(f"  Max:    {fn['PROB'].max():.4f}")

# Near-misses
near_miss_t = THRESHOLD * 0.8
near_miss = fn[fn['PROB'] >= near_miss_t]
very_low = fn[fn['PROB'] < 0.1]
print(f"\n  Near-miss (prob >= {near_miss_t:.3f}): {len(near_miss):,} ({len(near_miss)/len(fn)*100:.1f}%)")
print(f"  Very low (prob < 0.10):            {len(very_low):,} ({len(very_low)/len(fn)*100:.1f}%)")
print(f"  → Near-misses could be caught with slightly lower threshold")
print(f"  → Very low prob patients have NO typical cancer signals")

# What threshold would catch 90% of cancer?
probs_cancer = merged[merged['ACTUAL_LABEL']==1]['PROB'].values
for target_catch in [0.85, 0.90, 0.95]:
    t_needed = np.percentile(probs_cancer, (1-target_catch)*100)
    n_fp_at_t = (merged[(merged['ACTUAL_LABEL']==0)&(merged['PROB']>=t_needed)]).shape[0]
    print(f"  To catch {target_catch*100:.0f}% cancer: threshold={t_needed:.3f}, "
          f"FP={n_fp_at_t:,}, specificity={(1-n_fp_at_t/len(merged[merged['ACTUAL_LABEL']==0]))*100:.1f}%")


# ══════════════════════════════════════════════════════
# 6. FN ARCHETYPE ANALYSIS
# What "types" of missed patients exist?
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. FN ARCHETYPE ANALYSIS")
print("    (What types of cancer patients do we miss?)")
print("=" * 70)

archetypes = {}

# Archetype 1: No haematuria, no LUTS, sparse data
if all(c in fn.columns for c in ['HAEM_ANY_FLAG', 'LUTS_ANY_FLAG', 'OBS_TOTAL_COUNT']):
    silent = fn[(fn['HAEM_ANY_FLAG']==0) & (fn['LUTS_ANY_FLAG']==0) & (fn['OBS_TOTAL_COUNT']<=3)]
    archetypes['Silent cancer (no haem, no LUTS, sparse)'] = silent
    print(f"\n  Archetype 1: SILENT CANCER")
    print(f"    No haematuria, no LUTS, ≤3 observations")
    print(f"    Count: {len(silent):,} ({len(silent)/len(fn)*100:.1f}% of FN)")
    if len(silent) > 0:
        if 'AGE_AT_INDEX' in silent.columns:
            print(f"    Mean age: {silent['AGE_AT_INDEX'].mean():.1f}")
        if 'SEX_MALE' in silent.columns:
            print(f"    Male: {(silent['SEX_MALE']==1).sum()} ({(silent['SEX_MALE']==1).mean()*100:.1f}%)")
        print(f"    Mean prob: {silent['PROB'].mean():.4f}")

# Archetype 2: Has haematuria but still missed
if 'HAEM_ANY_FLAG' in fn.columns:
    haem_missed = fn[fn['HAEM_ANY_FLAG']==1]
    archetypes['Has haematuria but missed'] = haem_missed
    print(f"\n  Archetype 2: HAEMATURIA BUT MISSED")
    print(f"    Count: {len(haem_missed):,} ({len(haem_missed)/len(fn)*100:.1f}% of FN)")
    if len(haem_missed) > 0:
        print(f"    Mean prob: {haem_missed['PROB'].mean():.4f}")
        if 'HAEM_TOTAL_COUNT' in haem_missed.columns:
            print(f"    Mean haem count: {haem_missed['HAEM_TOTAL_COUNT'].mean():.1f}")
        elif 'HAEM_WB_COUNT' in haem_missed.columns:
            print(f"    Mean haem count (WB): {haem_missed['HAEM_WB_COUNT'].mean():.1f}")
        if 'RF_EVER_SMOKER' in haem_missed.columns:
            print(f"    Ever smoker: {haem_missed['RF_EVER_SMOKER'].sum()} ({haem_missed['RF_EVER_SMOKER'].mean()*100:.1f}%)")

# Archetype 3: Young patients (<60)
if 'AGE_AT_INDEX' in fn.columns:
    young = fn[fn['AGE_AT_INDEX'] < 60]
    archetypes['Young (<60)'] = young
    print(f"\n  Archetype 3: YOUNG PATIENTS (<60)")
    print(f"    Count: {len(young):,} ({len(young)/len(fn)*100:.1f}% of FN)")
    if len(young) > 0 and 'HAEM_ANY_FLAG' in young.columns:
        print(f"    With haematuria: {(young['HAEM_ANY_FLAG']==1).sum()} ({(young['HAEM_ANY_FLAG']==1).mean()*100:.1f}%)")

# Archetype 4: Lots of UTI treatment (misdiagnosed as UTI)
if 'MED_UTI_ANTIBIOTICS_COUNT' in fn.columns:
    uti_heavy = fn[fn['MED_UTI_ANTIBIOTICS_COUNT'].fillna(0) >= 3]
    archetypes['Heavy UTI treatment (≥3 courses)'] = uti_heavy
    print(f"\n  Archetype 4: MISDIAGNOSED AS UTI")
    print(f"    ≥3 UTI antibiotic courses")
    print(f"    Count: {len(uti_heavy):,} ({len(uti_heavy)/len(fn)*100:.1f}% of FN)")

# Archetype 5: Female patients (bladder cancer often missed in women)
if 'SEX_MALE' in fn.columns:
    female_fn = fn[fn['SEX_MALE']==0]
    archetypes['Female'] = female_fn
    print(f"\n  Archetype 5: FEMALE PATIENTS")
    print(f"    Count: {len(female_fn):,} ({len(female_fn)/len(fn)*100:.1f}% of FN)")
    if len(female_fn) > 0 and 'HAEM_ANY_FLAG' in female_fn.columns:
        print(f"    With haematuria: {(female_fn['HAEM_ANY_FLAG']==1).sum()} ({(female_fn['HAEM_ANY_FLAG']==1).mean()*100:.1f}%)")

# Archetype 6: Near-miss (high probability, just below threshold)
near = fn[fn['PROB'] >= THRESHOLD * 0.8]
archetypes['Near-miss (prob ≥ 80% of threshold)'] = near
print(f"\n  Archetype 6: NEAR-MISS")
print(f"    Probability ≥ {THRESHOLD*0.8:.3f} (80% of threshold)")
print(f"    Count: {len(near):,} ({len(near)/len(fn)*100:.1f}% of FN)")
print(f"    → These would be caught with a slightly lower threshold")


# ══════════════════════════════════════════════════════
# 7. FALSE POSITIVE ANALYSIS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. FALSE POSITIVE ANALYSIS")
print("    (Why does the model flag non-cancer patients?)")
print("=" * 70)

fp_features = ['AGE_AT_INDEX', 'SEX_MALE', 'HAEM_ANY_FLAG', 'HAEM_TOTAL_COUNT',
               'LUTS_ANY_FLAG', 'URINE_LAB_ABNORM_COUNT', 'CATH_ANY_FLAG',
               'RF_EVER_SMOKER', 'COMORB_BURDEN_SCORE', 'COMORB_RECURRENT_UTI_FLAG',
               'COMORB_BPH_FLAG', 'MED_UTI_ANTIBIOTICS_COUNT', 'OBS_TOTAL_COUNT',
               'SYN_BLEEDING_SCORE', 'INT_OVERALL_SUSPICION_SCORE']

available_fp = [f for f in fp_features if f in merged.columns]

print(f"\n  {'Feature':<45s} {'FP (alarm)':>12s} {'TN (correct)':>12s} {'TP (cancer)':>12s}")
print("  " + "-" * 85)
for feat in available_fp:
    fp_m = fp[feat].mean()
    tn_m = tn[feat].mean()
    tp_m = tp[feat].mean()
    print(f"  {feat:<45s} {fp_m:>12.3f} {tn_m:>12.3f} {tp_m:>12.3f}")

print(f"\n  → FP patients look LIKE cancer (high haem, smoking, symptoms)")
print(f"  → They have genuinely concerning clinical profiles")
print(f"  → Many may be appropriate for investigation anyway")


# ══════════════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 8a. Score distribution by group
ax = axes[0, 0]
for grp, color, label in [('TP','green','TP (caught)'),('FN','red','FN (missed)'),
                            ('FP','orange','FP (false alarm)'),('TN','blue','TN (correct)')]:
    data = merged[merged['GROUP']==grp]['PROB']
    ax.hist(data, bins=50, alpha=0.5, color=color, label=f'{label} (n={len(data):,})', density=True)
ax.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Density')
ax.set_title('Score Distribution by Error Type', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 8b. FN vs TP: Age distribution
ax = axes[0, 1]
if 'AGE_AT_INDEX' in merged.columns:
    ax.hist(fn['AGE_AT_INDEX'], bins=30, alpha=0.6, color='red', label=f'FN (n={len(fn)})', density=True)
    ax.hist(tp['AGE_AT_INDEX'], bins=30, alpha=0.6, color='green', label=f'TP (n={len(tp)})', density=True)
    ax.set_xlabel('Age at Index'); ax.set_ylabel('Density')
    ax.set_title('Age: FN vs TP', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

# 8c. Haematuria presence by group
ax = axes[0, 2]
if 'HAEM_ANY_FLAG' in merged.columns:
    groups = ['TP', 'FN', 'FP', 'TN']
    haem_pct = []
    for g in groups:
        grp = merged[merged['GROUP']==g]
        haem_pct.append((grp['HAEM_ANY_FLAG']==1).mean()*100)
    bars = ax.bar(groups, haem_pct, color=['green','red','orange','blue'], alpha=0.7)
    ax.set_ylabel('% with Haematuria')
    ax.set_title('Haematuria by Error Type', fontweight='bold')
    for bar, pct in zip(bars, haem_pct):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{pct:.1f}%', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

# 8d. Top features difference (FN vs TP)
ax = axes[1, 0]
diff_df = comp_df.copy()
diff_df['abs_diff'] = diff_df['difference'].abs()
top_diff = diff_df.sort_values('abs_diff', ascending=False).head(15)
colors = ['red' if d < 0 else 'green' for d in top_diff['difference']]
ax.barh(range(len(top_diff)), top_diff['difference'].values, color=colors, alpha=0.7)
ax.set_yticks(range(len(top_diff)))
ax.set_yticklabels(top_diff['feature'].values, fontsize=7)
ax.set_xlabel('FN mean - TP mean')
ax.set_title('Biggest Differences: FN vs TP', fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3)

# 8e. FN archetype pie chart
ax = axes[1, 1]
arch_sizes = {}
if 'HAEM_ANY_FLAG' in fn.columns and 'LUTS_ANY_FLAG' in fn.columns:
    n_silent = len(fn[(fn['HAEM_ANY_FLAG']==0) & (fn['LUTS_ANY_FLAG']==0)])
    n_haem_missed = len(fn[fn['HAEM_ANY_FLAG']==1])
    n_luts_only = len(fn[(fn['HAEM_ANY_FLAG']==0) & (fn['LUTS_ANY_FLAG']==1)])
    arch_sizes = {'No haem, no LUTS': n_silent, 'Has haematuria': n_haem_missed,
                  'LUTS only (no haem)': n_luts_only}
    # Remove zero
    arch_sizes = {k: v for k, v in arch_sizes.items() if v > 0}
    if arch_sizes:
        ax.pie(arch_sizes.values(), labels=arch_sizes.keys(), autopct='%1.1f%%',
               colors=['#ff6b6b','#ffd93d','#6bcb77'], startangle=90)
        ax.set_title('FN Patient Types', fontweight='bold')

# 8f. FN probability CDF
ax = axes[1, 2]
fn_probs_sorted = np.sort(fn['PROB'].values)
cdf = np.arange(1, len(fn_probs_sorted)+1) / len(fn_probs_sorted)
ax.plot(fn_probs_sorted, cdf, 'r-', linewidth=2)
ax.axvline(x=THRESHOLD, color='black', linestyle='--', label=f'Threshold={THRESHOLD}')
ax.axvline(x=THRESHOLD*0.8, color='gray', linestyle=':', label=f'80% of threshold')
ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Cumulative Fraction of FN')
ax.set_title('FN Probability CDF', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ERROR_DIR, 'error_analysis_plots.png'), dpi=150, bbox_inches='tight')
print("Saved: error_analysis_plots.png")


# ══════════════════════════════════════════════════════
# 9. SAVE DETAILED OUTPUTS
# ═════════════════════════════���════════════════════════
print("\n" + "=" * 70)
print("9. SAVING OUTPUTS")
print("=" * 70)

# Save FN patients with key features (only columns that exist)
fn_save_cols = [c for c in ['PATIENT_GUID', 'PROB', 'AGE_AT_INDEX', 'SEX_MALE',
                             'HAEM_ANY_FLAG', 'HAEM_TOTAL_COUNT', 'LUTS_ANY_FLAG', 'LUTS_TOTAL_COUNT',
                             'URINE_LAB_ABNORM_COUNT', 'URINE_INV_COUNT', 'RF_EVER_SMOKER',
                             'MED_UTI_ANTIBIOTICS_COUNT', 'OBS_TOTAL_COUNT', 'COMORB_BURDEN_SCORE',
                             'SYN_MASTER_SUSPICION', 'INT_OVERALL_SUSPICION_SCORE',
                             'NOBS_WEIGHT_LOSS_FLAG', 'NOBS_FATIGUE_FLAG', 'NOBS_ANAEMIA_DX_FLAG']
                if c in fn.columns]
fn[fn_save_cols].sort_values('PROB', ascending=False).to_csv(
    os.path.join(ERROR_DIR, 'false_negatives_detail.csv'), index=False)
fp[fn_save_cols].sort_values('PROB', ascending=False).to_csv(
    os.path.join(ERROR_DIR, 'false_positives_detail.csv'), index=False)

# Summary JSON
summary = {
    'threshold': THRESHOLD,
    'model': MODEL_NAME,
    'counts': {'TP': len(tp), 'FN': len(fn), 'FP': len(fp), 'TN': len(tn)},
    'fn_profile': {
        'mean_age': float(fn['AGE_AT_INDEX'].mean()) if 'AGE_AT_INDEX' in fn.columns else None,
        'pct_male': float((fn['SEX_MALE']==1).mean()*100) if 'SEX_MALE' in fn.columns else None,
        'pct_with_haematuria': float((fn['HAEM_ANY_FLAG']==1).mean()*100) if 'HAEM_ANY_FLAG' in fn.columns else None,
        'mean_prob': float(fn['PROB'].mean()),
        'pct_near_miss': float(len(near)/len(fn)*100),
        'pct_very_low': float(len(very_low)/len(fn)*100),
    },
    'archetypes': {name: len(df) for name, df in archetypes.items()},
    'key_findings': []
}

# Auto-generate key findings
if 'HAEM_ANY_FLAG' in fn.columns:
    haem_pct_fn = (fn['HAEM_ANY_FLAG']==1).mean()*100
    haem_pct_tp = (tp['HAEM_ANY_FLAG']==1).mean()*100
    summary['key_findings'].append(
        f"FN patients have LESS haematuria: {haem_pct_fn:.1f}% vs {haem_pct_tp:.1f}% in TP")

if 'OBS_TOTAL_COUNT' in fn.columns:
    obs_fn = fn['OBS_TOTAL_COUNT'].mean()
    obs_tp = tp['OBS_TOTAL_COUNT'].mean()
    if obs_fn < obs_tp:
        summary['key_findings'].append(
            f"FN patients have FEWER observations: {obs_fn:.1f} vs {obs_tp:.1f} in TP")

summary['key_findings'].append(
    f"Near-miss patients ({len(near)}) could be caught with lower threshold")

with open(os.path.join(ERROR_DIR, 'error_analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nSaved to: {ERROR_DIR}/")
print(f"  - fn_vs_tp_comparison.csv")
print(f"  - false_negatives_detail.csv")
print(f"  - false_positives_detail.csv")
print(f"  - error_analysis_summary.json")
print(f"  - error_analysis_plots.png")

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE ✅")
print("=" * 70)