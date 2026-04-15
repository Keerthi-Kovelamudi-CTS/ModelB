# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — FEATURE CLEANUP
# Step 5: Clean final feature matrices for modeling
# Input:  feature_matrix_final_*.csv (from Step 4 → 4b → 4c → 4d)
# Output: feature_matrix_clean_*.csv (ready for modeling)
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'
FE_RESULTS = SCRIPT_DIR / 'results' / 'FE'
CLEANUP_RESULTS = SCRIPT_DIR / 'results' / 'Cleanup_Finalresults'
for _w in ['3mo', '6mo', '12mo']:
    (CLEANUP_RESULTS / _w).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# LOAD FINAL FEATURE MATRICES (from results/FE/)
# ═══════════════════════════════════════════════════════════════

raw_matrices = {}
for window in ['3mo', '6mo', '12mo']:
    path = FE_RESULTS / window / f"feature_matrix_final_{window}.csv"
    fm = pd.read_csv(path, index_col=0)
    raw_matrices[window] = fm
    print(f"✅ Loaded {window}: {fm.shape[0]} patients × {fm.shape[1]} features | Pos: {(fm['LABEL']==1).sum()}")


# ═══════════════════════════════════════════════════════════════
# CLEANUP FUNCTION
# ═══════════════════════════════════════════════════════════════

def cleanup_features(fm, window_name):

    print(f"\n{'═'*70}")
    print(f"  FEATURE CLEANUP — {window_name}")
    print(f"{'═'*70}")

    initial_cols = fm.shape[1]
    removed_log = {}

    fm['LABEL'] = pd.to_numeric(fm['LABEL'], errors='coerce')

    # ──────────────────────────────────────────────────────────
    # 5a. REMOVE NON-NUMERIC COLUMNS
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5a. REMOVE NON-NUMERIC COLUMNS ──")

    non_numeric = fm.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Removing {len(non_numeric)}: {non_numeric[:10]}{'...' if len(non_numeric)>10 else ''}")
        fm = fm.drop(columns=non_numeric, errors='ignore')
    else:
        print(f"  None found ✅")
    removed_log['non_numeric'] = len(non_numeric)

    # Sanitize column names — XGBoost rejects [ ] < > in feature names
    import re
    fm.columns = [re.sub(r'[\[\]<>{}/\\]', '_', str(c)) for c in fm.columns]
    fm = fm.loc[:, ~fm.columns.duplicated()]

    # ──────────────────────────────────────────────────────────
    # 5b. REMOVE DUPLICATE COLUMNS
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5b. REMOVE DUPLICATE COLUMNS ──")

    before = fm.shape[1]

    # Name duplicates
    fm = fm.loc[:, ~fm.columns.duplicated()]
    name_dupes = before - fm.shape[1]

    # LAB slope = delta bug from Step 4 (slope is just copy of delta)
    slope_cols = [c for c in fm.columns if c.endswith('_slope') and c.startswith('LAB_') and not c.startswith('LAB_TRAJ_')]
    slope_to_remove = []
    for s in slope_cols:
        d = s.replace('_slope', '_delta')
        if d in fm.columns and fm[s].equals(fm[d]):
            slope_to_remove.append(s)
    if slope_to_remove:
        fm = fm.drop(columns=slope_to_remove, errors='ignore')
        print(f"  Removed {len(slope_to_remove)} LAB slope columns (identical to delta)")

    # Value-based dedup (sample for speed)
    sample_size = min(5000, len(fm))
    fm_sample = fm.sample(n=sample_size, random_state=42)

    col_hashes = {}
    val_dupes = []
    for col in fm.columns:
        h = hash(tuple(fm_sample[col].values))
        if h in col_hashes:
            if fm_sample[col].equals(fm_sample[col_hashes[h]]):
                val_dupes.append(col)
                continue
        col_hashes[h] = col

    if val_dupes:
        fm = fm.drop(columns=val_dupes, errors='ignore')
        print(f"  Removed {len(val_dupes)} value-duplicate columns")
        for c in val_dupes[:20]:
            print(f"    {c}")
        if len(val_dupes) > 20:
            print(f"    ... and {len(val_dupes)-20} more")
    else:
        print(f"  No value-duplicates ✅")

    total_dupes = name_dupes + len(slope_to_remove) + len(val_dupes)
    removed_log['duplicates'] = total_dupes
    print(f"  Total duplicates removed: {total_dupes}")

    # ──────────────────────────────────────────────────────────
    # 5c. REMOVE >95% NULL
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5c. REMOVE HIGH-NULL FEATURES (>95%) ──")

    null_pct = fm.isnull().sum() / len(fm) * 100
    high_null = [c for c in null_pct[null_pct > 95].index if c != 'LABEL']

    if high_null:
        print(f"  Removing {len(high_null)} features:")
        for c in sorted(high_null)[:20]:
            print(f"    {c}: {null_pct[c]:.1f}%")
        if len(high_null) > 20:
            print(f"    ... and {len(high_null)-20} more")
        fm = fm.drop(columns=high_null)
    else:
        print(f"  None found ✅")
    removed_log['high_null'] = len(high_null)

    # ──────────────────────────────────────────────────────────
    # 5d. REMOVE ZERO-VARIANCE
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5d. REMOVE ZERO-VARIANCE ──")

    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != 'LABEL']
    variances = fm[numeric_cols].var()
    zero_var = variances[variances == 0].index.tolist()

    if zero_var:
        print(f"  Removing {len(zero_var)} zero-variance features:")
        for c in sorted(zero_var)[:20]:
            print(f"    {c}: constant = {fm[c].iloc[0]}")
        if len(zero_var) > 20:
            print(f"    ... and {len(zero_var)-20} more")
        fm = fm.drop(columns=zero_var)
    else:
        print(f"  None found ✅")
    removed_log['zero_var'] = len(zero_var)

    # ──────────────────────────────────────────────────────────
    # 5e. REMOVE NEAR-ZERO VARIANCE (>99% same value)
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5e. REMOVE NEAR-ZERO VARIANCE (>99% same value) ──")

    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != 'LABEL']

    nzv_cols = []
    for col in numeric_cols:
        top_pct = fm[col].value_counts(normalize=True).iloc[0] * 100
        if top_pct > 99.0:
            nzv_cols.append((col, top_pct))

    if nzv_cols:
        nzv_cols.sort(key=lambda x: -x[1])
        print(f"  Removing {len(nzv_cols)} near-zero-variance features:")
        for col, pct in nzv_cols[:30]:
            print(f"    {col}: {pct:.1f}%")
        if len(nzv_cols) > 30:
            print(f"    ... and {len(nzv_cols)-30} more")
        fm = fm.drop(columns=[c for c, _ in nzv_cols])
    else:
        print(f"  None found ✅")
    removed_log['near_zero_var'] = len(nzv_cols)

    # ──────────────────────────────────────────────────────────
    # 5f. FILL REMAINING NaN — smart by feature type
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5f. FILL REMAINING NaN ──")

    remaining = fm.isnull().sum()
    remaining = remaining[remaining > 0]

    if len(remaining) > 0:
        print(f"  {len(remaining)} features with NaN")

        median_cols = []
        zero_cols = []
        neg1_cols = []

        for col in remaining.index:
            if col == 'LABEL':
                continue

            # Lab continuous → median
            if any(x in col for x in ['LABTERM_', 'LAB_TRAJ_']):
                if any(x in col for x in ['_mean', '_min', '_max', '_first', '_last',
                                            '_delta', '_slope', '_pct_change', '_first_last_diff']):
                    median_cols.append(col)
                elif any(x in col for x in ['_std', '_cv', '_range']):
                    zero_cols.append(col)
                else:
                    median_cols.append(col)
            elif col.startswith('LAB_') and any(x in col for x in ['_mean', '_min', '_max', '_first', '_last', '_delta']):
                median_cols.append(col)
            elif col.startswith('LAB_') and '_std' in col:
                zero_cols.append(col)

            # Days/gaps → -1 (no event happened)
            elif any(x in col for x in ['_days_', 'SEQ_', '_to_index', '_to_imaging',
                                          'INV_PATTERN_symptom_to_inv_days',
                                          'MEL_PATH_symptom_to_imaging_days']):
                neg1_cols.append(col)

            # Temporal/trajectory → median
            elif any(x in col for x in ['TEMP_', 'TRAJ_', 'MONTHLY_', 'ROLLING3M_']):
                median_cols.append(col)

            # Everything else → 0
            else:
                zero_cols.append(col)

        if median_cols:
            for col in median_cols:
                val = fm[col].median()
                fm[col] = fm[col].fillna(val if pd.notna(val) else 0)
            print(f"  Filled {len(median_cols)} features with median")

        if zero_cols:
            fm[zero_cols] = fm[zero_cols].fillna(0)
            print(f"  Filled {len(zero_cols)} features with 0")

        if neg1_cols:
            fm[neg1_cols] = fm[neg1_cols].fillna(-1)
            print(f"  Filled {len(neg1_cols)} features with -1")

        leftover = fm.isnull().sum().sum()
        if leftover > 0:
            print(f"  Filling {leftover} remaining with 0")
            fm = fm.fillna(0)

        print(f"  ✅ NaN remaining: {fm.isnull().sum().sum()}")
    else:
        print(f"  ✅ No NaN")

    # Replace inf
    inf_count = np.isinf(fm.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        fm = fm.replace([np.inf, -np.inf], 0)
        print(f"  Replaced {inf_count} inf with 0")

    # ──────────────────────────────────────────────────────────
    # 5f2. REMOVE HEALTHCARE UTILIZATION CONFOUNDS
    # Lab count/max/mean features where the signal comes from
    # differential testing rates (controls tested more than cases),
    # not from actual lab values. eGFR is the worst offender.
    # NOTE: Melanoma has almost no lab features, so most of this
    # section will be no-ops.
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5f2. REMOVE HEALTHCARE UTILIZATION CONFOUNDS ──")

    # Step 1: Remove all eGFR features (pure utilization artifact)
    egfr_cols = [c for c in fm.columns if 'egfr' in c.lower() or 'eGFR' in c]
    if egfr_cols:
        print(f"  Removing {len(egfr_cols)} eGFR features (utilization artifact, not cancer signal):")
        for c in sorted(egfr_cols):
            print(f"    {c}")
        fm = fm.drop(columns=egfr_cols, errors='ignore')

    # Step 2: Detect and remove other lab count features with severe utilization bias
    # A lab count feature is confounded if controls have >15% lower zero-rate than cases
    labterm_cols = [c for c in fm.columns if c.startswith('LABTERM_') and c.endswith('_count')]
    utilization_confounds = []
    if not labterm_cols:
        print("  No lab term features found, skipping utilization confound check")
    else:
        for col in labterm_cols:
            cases_zero = (fm.loc[fm['LABEL'] == 1, col] == 0).mean() * 100
            ctrl_zero = (fm.loc[fm['LABEL'] == 0, col] == 0).mean() * 100
            diff = cases_zero - ctrl_zero  # positive = cases tested less
            if diff > 15:  # severe: >15% gap
                utilization_confounds.append((col, cases_zero, ctrl_zero, diff))
                # Also flag related features (same lab term, different suffix)
                base = col.replace('_count', '')
                related = [c for c in fm.columns if c.startswith(base) and c != col]
                for r in related:
                    if r not in [x[0] for x in utilization_confounds]:
                        utilization_confounds.append((r, -1, -1, diff))

    if utilization_confounds:
        confound_names = [c[0] for c in utilization_confounds]
        print(f"  Removing {len(confound_names)} features with severe utilization bias (>15% zero-rate gap):")
        for col, cz, ctrlz, diff in utilization_confounds[:20]:
            if cz >= 0:
                print(f"    {col}: cases_zero={cz:.1f}% ctrl_zero={ctrlz:.1f}% gap={diff:+.1f}%")
            else:
                print(f"    {col}: (related to confounded lab term)")
        if len(confound_names) > 20:
            print(f"    ... and {len(confound_names)-20} more")
        fm = fm.drop(columns=confound_names, errors='ignore')
    else:
        if labterm_cols:
            print(f"  No severe utilization confounds found ✅")

    # Step 3: Remove ALL lab test count features (test ordering frequency ≠ biology)
    # These measure how often a test was ordered, not the actual lab values.
    # Controls get routine panels; cancer patients get irregular investigation-driven tests.
    lab_count_cols = [c for c in fm.columns if c != 'LABEL' and (
        # LABTERM_*_count — per-term test frequency
        (c.startswith('LABTERM_') and c.endswith('_count')) or
        # LAB_*_count — per-category test frequency
        (c.startswith('LAB_') and c.endswith('_count') and not c.startswith('LAB_TRAJ_'))
    )]
    # Filter out any already removed
    lab_count_cols = [c for c in lab_count_cols if c in fm.columns]
    if lab_count_cols:
        print(f"  Removing {len(lab_count_cols)} lab count features (test ordering frequency, not values):")
        for c in sorted(lab_count_cols)[:20]:
            print(f"    {c}")
        if len(lab_count_cols) > 20:
            print(f"    ... and {len(lab_count_cols)-20} more")
        fm = fm.drop(columns=lab_count_cols, errors='ignore')

    # Step 4: Remove aggregate renal trajectory features (utilization-driven variability)
    renal_traj_cols = [c for c in fm.columns if c != 'LABEL' and (
        c.startswith('LAB_TRAJ_RENAL_') or
        (c.startswith('LAB_RENAL_') and any(c.endswith(s) for s in ['_std', '_range', '_cv']))
    )]
    renal_traj_cols = [c for c in renal_traj_cols if c in fm.columns]
    if renal_traj_cols:
        print(f"  Removing {len(renal_traj_cols)} renal trajectory/variability features (utilization confound):")
        for c in sorted(renal_traj_cols):
            print(f"    {c}")
        fm = fm.drop(columns=renal_traj_cols, errors='ignore')

    # Step 5: Remove lab std features where signal comes from irregular test timing
    lab_std_cols = [c for c in fm.columns if c != 'LABEL' and
        c.startswith('LAB_') and c.endswith('_std') and not c.startswith('LAB_TRAJ_')]
    lab_std_cols = [c for c in lab_std_cols if c in fm.columns]
    if lab_std_cols:
        print(f"  Removing {len(lab_std_cols)} lab std features (driven by irregular test timing):")
        for c in sorted(lab_std_cols):
            print(f"    {c}")
        fm = fm.drop(columns=lab_std_cols, errors='ignore')

    # Step 6: Remove remaining LAB_TRAJ range/cv features (same utilization issue across categories)
    lab_traj_var_cols = [c for c in fm.columns if c != 'LABEL' and
        c.startswith('LAB_TRAJ_') and any(c.endswith(s) for s in ['_range', '_cv'])]
    lab_traj_var_cols = [c for c in lab_traj_var_cols if c in fm.columns]
    if lab_traj_var_cols:
        print(f"  Removing {len(lab_traj_var_cols)} LAB_TRAJ range/cv features (utilization-driven variability):")
        for c in sorted(lab_traj_var_cols):
            print(f"    {c}")
        fm = fm.drop(columns=lab_traj_var_cols, errors='ignore')

    total_util_removed = len(egfr_cols) + len(utilization_confounds) + len(lab_count_cols) + len(renal_traj_cols) + len(lab_std_cols) + len(lab_traj_var_cols)
    removed_log['utilization_confounds'] = total_util_removed

    # ──────────────────────────────────────────────────────────
    # 5g. REMOVE HIGHLY CORRELATED (>0.98)
    # Keep the one with higher label correlation (0.98 keeps more features)
    # ──────────────────────────────────────────────────────────
    HIGH_CORR_THRESHOLD = 0.98
    print(f"\n── 5g. REMOVE HIGHLY CORRELATED (>{HIGH_CORR_THRESHOLD}) ──")

    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != 'LABEL']
    print(f"  Computing correlation for {len(numeric_cols)} features...")

    label_corr = fm[numeric_cols].corrwith(fm['LABEL']).abs()
    corr_matrix = fm[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    pair_count = 0

    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = upper.index[upper[col] > HIGH_CORR_THRESHOLD].tolist()
        for corr_col in correlated:
            if corr_col in to_drop:
                continue
            pair_count += 1
            c1 = label_corr.get(col, 0)
            c2 = label_corr.get(corr_col, 0)
            if c1 >= c2:
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    if to_drop:
        print(f"  Found {pair_count} correlated pairs → removing {len(to_drop)} features")
        for c in sorted(to_drop)[:30]:
            print(f"    {c}")
        if len(to_drop) > 30:
            print(f"    ... and {len(to_drop)-30} more")
        fm = fm.drop(columns=list(to_drop), errors='ignore')
    else:
        print(f"  None found ✅")
    removed_log['high_corr'] = len(to_drop)

    # ──────────────────────────────────────────────────────────
    # 5h. LEAKAGE CHECK
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5h. LEAKAGE CHECK ──")

    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != 'LABEL']
    label_corr_final = fm[numeric_cols].corrwith(fm['LABEL']).abs().sort_values(ascending=False)

    # Auto-remove >0.5 (in medical prediction, >0.5 correlation is highly suspicious)
    extreme = label_corr_final[label_corr_final > 0.5]
    if len(extreme) > 0:
        print(f"  Removing {len(extreme)} features with >0.5 label correlation:")
        for col, corr in extreme.items():
            is_mel = col.startswith('MEL_')
            print(f"    {col}: {corr:.4f}{' (melanoma-specific)' if is_mel else ''}")
        fm = fm.drop(columns=extreme.index.tolist(), errors='ignore')
        removed_log['leakage'] = len(extreme)
    else:
        removed_log['leakage'] = 0

    # Flag 0.3-0.5
    suspicious = label_corr_final[(label_corr_final > 0.3) & (label_corr_final <= 0.5)]
    if len(suspicious) > 0:
        print(f"  {len(suspicious)} features with 0.3-0.5 correlation (review):")
        for col, corr in suspicious.head(30).items():
            print(f"    {col}: {corr:.4f}")
    else:
        print(f"  ✅ No leakage detected")

    # ──────────────────────────────────────────────────────────
    # 5i. TOP FEATURES
    # ──────────────────────────────────────────────────────────

    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != 'LABEL']
    label_corr_final = fm[numeric_cols].corrwith(fm['LABEL']).abs().sort_values(ascending=False)

    print(f"\n  Top 30 features by |correlation with LABEL|:")
    for i, (col, corr) in enumerate(label_corr_final.head(30).items()):
        is_mel = col.startswith('MEL_')
        is_adv = any(col.startswith(p) for p in [
            'CLUSTER_','VISIT_','TRAJ_','LAB_TRAJ_','MED_ESC_',
            'INV_PATTERN_','CROSS_','DECAY_','MONTHLY_','ROLLING3M_',
            'CAT_','MEDCAT_','LABTERM_','PAIR_','CMPAIR_','SEQ_',
            'RECUR_','MEDREC_','RATE_','AGE_','AGEX_',
            'ENTROPY_','GINI_'
        ])
        if is_mel:
            marker = '◆'
        elif is_adv:
            marker = '★'
        else:
            marker = ' '
        print(f"    {marker} {i+1:2d}. {col}: {corr:.4f}")
    print(f"\n  ◆ = melanoma-specific (Step 4d) | ★ = advanced/mega (Step 4b/4c)")

    # ──────────────────────────────────────────────────────────
    # 5j. FINAL SUMMARY
    # ──────────────────────────────────────────────────────────
    print(f"\n── 5j. FINAL SUMMARY ──")
    print(f"  Initial features:   {initial_cols}")
    print(f"  Removed breakdown:")
    total_removed = 0
    for step, count in removed_log.items():
        if count > 0:
            print(f"    {step:20s}: -{count}")
            total_removed += count
    print(f"  Total removed:      {total_removed}")
    print(f"  Final features:     {fm.shape[1]}")
    print(f"  Final patients:     {fm.shape[0]}")
    print(f"  NaN: {fm.isnull().sum().sum()} | Inf: {np.isinf(fm.select_dtypes(include=[np.number])).sum().sum()}")

    # Feature group breakdown
    print(f"\n  Feature group breakdown:")
    prefixes = [
        ('MEL_NICE_', 'NICE NG14 skin'),
        ('MEL_PATH_', 'Lesion pathway'),
        ('MEL_RF_', 'Mel risk factors'),
        ('MEL_TX_', 'Mel treatment'),
        ('MEL_mimic_', 'Mel mimics'),
        ('MEL_', 'Melanoma-specific'),
        ('OBS_', 'Observation'),
        ('LAB_TRAJ_', 'Lab trajectory'),
        ('LABTERM_', 'Lab (per-term)'),
        ('LAB_', 'Lab (basic)'),
        ('INV_PATTERN_', 'Inv patterns'),
        ('INV_', 'Investigation'),
        ('AGG_', 'Aggregate'),
        ('TEMP_', 'Temporal'),
        ('MED_ESC_', 'Med escalation'),
        ('MED_AGG_', 'Med aggregate'),
        ('MEDCAT_', 'Med (per-cat)'),
        ('MEDREC_', 'Med recurrence'),
        ('MED_', 'Medication'),
        ('INT_', 'Interactions'),
        ('CROSS_', 'Cross-domain'),
        ('CLUSTER_', 'Symptom clusters'),
        ('VISIT_', 'Visit patterns'),
        ('TRAJ_', 'Trajectory'),
        ('MONTHLY_', 'Monthly bins'),
        ('ROLLING3M_', 'Rolling 3-month'),
        ('CAT_', 'Per-category granular'),
        ('PAIR_', 'Obs co-occurrence'),
        ('CMPAIR_', 'Clin-Med pairs'),
        ('SEQ_', 'Sequence'),
        ('RECUR_', 'Recurrence (obs)'),
        ('RATE_', 'Rates & ratios'),
        ('AGE_', 'Age features'),
        ('AGEX_', 'Age interactions'),
        ('ENTROPY_', 'Entropy'),
        ('GINI_', 'Gini'),
        ('DECAY_', 'Time-decay'),
    ]

    accounted = set()
    for prefix, name in prefixes:
        cols = [c for c in fm.columns if c.startswith(prefix) and c not in accounted]
        if cols:
            accounted.update(cols)
            print(f"    {name:25s}: {len(cols)}")

    other = [c for c in fm.columns if c not in accounted and c != 'LABEL']
    if other:
        print(f"    {'Other/Demographics':25s}: {len(other)}")
    print(f"    {'─'*40}")
    print(f"    {'TOTAL':25s}: {fm.shape[1]}")

    # Ratio check
    n_pos = (fm['LABEL'] == 1).sum()
    n_features = fm.shape[1] - 1  # exclude LABEL
    print(f"\n  Features-per-positive ratio: {n_features}/{n_pos} = {n_features/n_pos:.2f}")
    if n_features / n_pos > 0.3:
        print(f"  ⚠️ High ratio — model should select top 80-100 features")
    else:
        print(f"  ✅ Ratio OK")

    return fm


# ═══════════════════════════════════════════════════════════════
# RUN CLEANUP FOR ALL 3 WINDOWS
# ═══════════════════════════════════════════════════════════════

cleaned_matrices = {}

for window in ['3mo', '6mo', '12mo']:
    fm = raw_matrices[window].copy()
    cleaned = cleanup_features(fm, window.upper())
    cleaned_matrices[window] = cleaned

    out_path = CLEANUP_RESULTS / window / f"feature_matrix_clean_{window}.csv"
    cleaned.to_csv(out_path)
    print(f"\n  ✅ Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════
# CROSS-WINDOW COMPARISON
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═'*70}")
print(f"  CROSS-WINDOW COMPARISON")
print(f"{'═'*70}")

for window in ['3mo', '6mo', '12mo']:
    fm = cleaned_matrices[window]
    mel_cols = [c for c in fm.columns if c.startswith('MEL_')]
    adv_cols = [c for c in fm.columns if any(c.startswith(p) for p in
                ['CLUSTER_','VISIT_','TRAJ_','LAB_TRAJ_','MED_ESC_',
                 'INV_PATTERN_','CROSS_','DECAY_','MONTHLY_','ROLLING3M_',
                 'CAT_','MEDCAT_','LABTERM_','PAIR_','CMPAIR_','SEQ_',
                 'RECUR_','MEDREC_','RATE_','AGE_','AGEX_',
                 'ENTROPY_','GINI_'])]
    print(f"\n  {window}: {fm.shape[0]} patients × {fm.shape[1]} features")
    print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
    print(f"    Melanoma-specific (◆): {len(mel_cols)}")
    print(f"    Advanced/mega (★):      {len(adv_cols)}")
    print(f"    NaN: {fm.isnull().sum().sum()}")

cols_3 = set(cleaned_matrices['3mo'].columns)
cols_6 = set(cleaned_matrices['6mo'].columns)
cols_12 = set(cleaned_matrices['12mo'].columns)
common = cols_3 & cols_6 & cols_12

print(f"\n  Feature consistency:")
print(f"    Common across all 3: {len(common)}")
print(f"    Only in 3mo:  {len(cols_3 - cols_6 - cols_12)}")
print(f"    Only in 6mo:  {len(cols_6 - cols_3 - cols_12)}")
print(f"    Only in 12mo: {len(cols_12 - cols_3 - cols_6)}")


# ═══════════════════════════════════════════════════════════════
# MELANOMA-SPECIFIC SIGNAL CHECK
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═'*70}")
print(f"  MELANOMA-SPECIFIC FEATURES — SIGNAL RANK")
print(f"{'═'*70}")

for window in ['3mo', '6mo', '12mo']:
    fm = cleaned_matrices[window]
    mel_cols = [c for c in fm.columns if c.startswith('MEL_') and c != 'LABEL']

    if mel_cols:
        mel_corr = fm[mel_cols].corrwith(fm['LABEL']).abs().sort_values(ascending=False)
        print(f"\n  {window} — Melanoma features ranked:")
        for i, (col, corr) in enumerate(mel_corr.items()):
            print(f"    {i+1:2d}. {col}: {corr:.4f}")

print(f"\n{'═'*70}")
print(f"  ✅ STEP 5 COMPLETE — Ready for modeling")
print(f"  Files: feature_matrix_clean_{{window}}.csv")
print(f"{'═'*70}")
