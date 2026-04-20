# ═══════════════════════════════════════════════════════════════
# FEATURE CLEANUP — Config-driven
# Step 5: Clean final feature matrices for modeling
# Input:  feature_matrix_final_*.csv (from FE pipeline)
# Output: feature_matrix_clean_*.csv (ready for modeling)
# ═══════════════════════════════════════════════════════════════

import logging
import re
import warnings

import numpy as np
import pandas as pd

import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def cleanup_features(fm, window_name, cfg):
    """
    Clean a feature matrix: remove non-numeric, duplicates, high-null,
    zero/near-zero variance, utilization confounds, high correlation, leakage.

    Returns cleaned DataFrame.
    """
    PREFIX = cfg.PREFIX
    LABEL = cfg.LABEL_COL

    logger.info(f"\n{'='*70}")
    logger.info(f"  FEATURE CLEANUP - {window_name}")
    logger.info(f"{'='*70}")

    initial_cols = fm.shape[1]
    removed_log = {}

    fm[LABEL] = pd.to_numeric(fm[LABEL], errors='coerce')

    # ── 5a. REMOVE NON-NUMERIC COLUMNS ───────────────────────
    logger.info(f"\n-- 5a. REMOVE NON-NUMERIC COLUMNS --")
    non_numeric = fm.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.info(f"  Removing {len(non_numeric)}: {non_numeric[:10]}")
        fm = fm.drop(columns=non_numeric, errors='ignore')
    removed_log['non_numeric'] = len(non_numeric)

    fm.columns = [re.sub(r'[\[\]<>{}/\\]', '_', str(c)) for c in fm.columns]
    fm = fm.loc[:, ~fm.columns.duplicated()]

    # ── 5b. REMOVE DUPLICATE COLUMNS ─────────────────────────
    logger.info(f"\n-- 5b. REMOVE DUPLICATE COLUMNS --")
    before = fm.shape[1]
    fm = fm.loc[:, ~fm.columns.duplicated()]
    name_dupes = before - fm.shape[1]

    # LAB slope = delta bug
    slope_cols = [c for c in fm.columns if c.endswith('_slope') and c.startswith('LAB_') and not c.startswith('LAB_TRAJ_')]
    slope_to_remove = []
    for s in slope_cols:
        d = s.replace('_slope', '_delta')
        if d in fm.columns and fm[s].equals(fm[d]):
            slope_to_remove.append(s)
    if slope_to_remove:
        fm = fm.drop(columns=slope_to_remove, errors='ignore')
        logger.info(f"  Removed {len(slope_to_remove)} LAB slope columns (identical to delta)")

    # Value-based dedup
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
        logger.info(f"  Removed {len(val_dupes)} value-duplicate columns")

    total_dupes = name_dupes + len(slope_to_remove) + len(val_dupes)
    removed_log['duplicates'] = total_dupes

    # ── 5c. REMOVE >95% NULL ─────────────────────────────────
    logger.info(f"\n-- 5c. REMOVE HIGH-NULL FEATURES (>95%) --")
    null_pct = fm.isnull().sum() / len(fm) * 100
    high_null = [c for c in null_pct[null_pct > 95].index if c != LABEL]
    if high_null:
        logger.info(f"  Removing {len(high_null)} features")
        fm = fm.drop(columns=high_null)
    removed_log['high_null'] = len(high_null)

    # ── 5d. REMOVE ZERO-VARIANCE ─────────────────────────────
    logger.info(f"\n-- 5d. REMOVE ZERO-VARIANCE --")
    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != LABEL]
    variances = fm[numeric_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        logger.info(f"  Removing {len(zero_var)} zero-variance features")
        fm = fm.drop(columns=zero_var)
    removed_log['zero_var'] = len(zero_var)

    # ── 5e. REMOVE NEAR-ZERO VARIANCE (>99% same value) ─────
    logger.info(f"\n-- 5e. REMOVE NEAR-ZERO VARIANCE (>99% same value) --")
    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != LABEL]
    nzv_cols = []
    for col in numeric_cols:
        top_pct = fm[col].value_counts(normalize=True).iloc[0] * 100
        if top_pct > 99.0:
            nzv_cols.append((col, top_pct))
    if nzv_cols:
        nzv_cols.sort(key=lambda x: -x[1])
        logger.info(f"  Removing {len(nzv_cols)} near-zero-variance features")
        # Name any clinically-meaningful flags we're dropping so the user can
        # spot them and decide whether to widen the threshold if needed.
        flagged = [(c, p) for c, p in nzv_cols
                   if c.startswith(('ETH_', 'PROST_RF_', 'PROST_HAS_'))]
        if flagged:
            logger.info(f"    (Incl. {len(flagged)} high-signal flag(s) dropped:)")
            for c, p in flagged[:15]:
                logger.info(f"      {c}  ({p:.2f}% one value)")
        fm = fm.drop(columns=[c for c, _ in nzv_cols])
    removed_log['near_zero_var'] = len(nzv_cols)

    # ── 5f. FILL REMAINING NaN ───────────────────────────────
    logger.info(f"\n-- 5f. FILL REMAINING NaN --")
    remaining = fm.isnull().sum()
    remaining = remaining[remaining > 0]

    if len(remaining) > 0:
        logger.info(f"  {len(remaining)} features with NaN")
        median_cols, zero_cols, neg1_cols = [], [], []

        for col in remaining.index:
            if col == LABEL:
                continue
            # Lab continuous -> median
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
            # Days/gaps -> -1
            elif any(x in col for x in ['_days_', 'SEQ_', '_to_index', '_to_imaging']):
                neg1_cols.append(col)
            # Temporal/trajectory -> median
            elif any(x in col for x in ['TEMP_', 'TRAJ_', 'MONTHLY_', 'ROLLING3M_']):
                median_cols.append(col)
            else:
                zero_cols.append(col)

        if median_cols:
            for col in median_cols:
                val = fm[col].median()
                fm[col] = fm[col].fillna(val if pd.notna(val) else 0)
        if zero_cols:
            fm[zero_cols] = fm[zero_cols].fillna(0)
        if neg1_cols:
            fm[neg1_cols] = fm[neg1_cols].fillna(-1)

        leftover = fm.isnull().sum().sum()
        if leftover > 0:
            fm = fm.fillna(0)

    # Replace inf
    inf_count = np.isinf(fm.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        fm = fm.replace([np.inf, -np.inf], 0)

    # ── 5f2. REMOVE HEALTHCARE UTILIZATION CONFOUNDS ─────────
    logger.info(f"\n-- 5f2. REMOVE HEALTHCARE UTILIZATION CONFOUNDS --")

    # eGFR features
    egfr_cols = [c for c in fm.columns if 'egfr' in c.lower() or 'glomerular' in c.lower()]
    if egfr_cols:
        logger.info(f"  Removing {len(egfr_cols)} eGFR/GFR features")
        fm = fm.drop(columns=egfr_cols, errors='ignore')

    # Lab count features with severe utilization bias
    labterm_cols = [c for c in fm.columns if c.startswith('LABTERM_') and c.endswith('_count')]
    utilization_confounds = []
    for col in labterm_cols:
        cases_zero = (fm.loc[fm[LABEL] == 1, col] == 0).mean() * 100
        ctrl_zero = (fm.loc[fm[LABEL] == 0, col] == 0).mean() * 100
        diff = cases_zero - ctrl_zero
        if diff > 15:
            utilization_confounds.append((col, cases_zero, ctrl_zero, diff))
            base = col.replace('_count', '')
            related = [c for c in fm.columns if c.startswith(base) and c != col]
            for r in related:
                if r not in [x[0] for x in utilization_confounds]:
                    utilization_confounds.append((r, -1, -1, diff))
    if utilization_confounds:
        confound_names = [c[0] for c in utilization_confounds]
        logger.info(f"  Removing {len(confound_names)} features with severe utilization bias")
        fm = fm.drop(columns=confound_names, errors='ignore')

    # All lab count features
    lab_count_cols = [c for c in fm.columns if c != LABEL and (
        (c.startswith('LABTERM_') and c.endswith('_count')) or
        (c.startswith('LAB_') and c.endswith('_count') and not c.startswith('LAB_TRAJ_'))
    )]
    lab_count_cols = [c for c in lab_count_cols if c in fm.columns]
    if lab_count_cols:
        logger.info(f"  Removing {len(lab_count_cols)} lab count features")
        fm = fm.drop(columns=lab_count_cols, errors='ignore')

    # Lab std features
    lab_std_cols = [c for c in fm.columns if c != LABEL and
        c.startswith('LAB_') and c.endswith('_std') and not c.startswith('LAB_TRAJ_')]
    lab_std_cols = [c for c in lab_std_cols if c in fm.columns]
    if lab_std_cols:
        logger.info(f"  Removing {len(lab_std_cols)} lab std features")
        fm = fm.drop(columns=lab_std_cols, errors='ignore')

    # LAB_TRAJ range/cv
    lab_traj_var_cols = [c for c in fm.columns if c != LABEL and
        c.startswith('LAB_TRAJ_') and any(c.endswith(s) for s in ['_range', '_cv'])]
    lab_traj_var_cols = [c for c in lab_traj_var_cols if c in fm.columns]
    if lab_traj_var_cols:
        logger.info(f"  Removing {len(lab_traj_var_cols)} LAB_TRAJ range/cv features")
        fm = fm.drop(columns=lab_traj_var_cols, errors='ignore')

    total_util = len(egfr_cols) + len(utilization_confounds) + len(lab_count_cols) + len(lab_std_cols) + len(lab_traj_var_cols)
    removed_log['utilization_confounds'] = total_util

    # ── 5g. REMOVE HIGHLY CORRELATED (>0.98) ─────────────────
    HIGH_CORR_THRESHOLD = 0.98
    logger.info(f"\n-- 5g. REMOVE HIGHLY CORRELATED (>{HIGH_CORR_THRESHOLD}) --")

    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != LABEL]
    label_corr = fm[numeric_cols].corrwith(fm[LABEL]).abs()
    corr_matrix = fm[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = upper.index[upper[col] > HIGH_CORR_THRESHOLD].tolist()
        for corr_col in correlated:
            if corr_col in to_drop:
                continue
            c1 = label_corr.get(col, 0)
            c2 = label_corr.get(corr_col, 0)
            if c1 >= c2:
                to_drop.add(corr_col)
            else:
                to_drop.add(col)
    if to_drop:
        logger.info(f"  Removing {len(to_drop)} highly correlated features")
        fm = fm.drop(columns=list(to_drop), errors='ignore')
    removed_log['high_corr'] = len(to_drop)

    # ── 5h. LEAKAGE CHECK ────────────────────────────────────
    logger.info(f"\n-- 5h. LEAKAGE CHECK --")
    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != LABEL]
    label_corr_final = fm[numeric_cols].corrwith(fm[LABEL]).abs().sort_values(ascending=False)

    extreme = label_corr_final[label_corr_final > 0.5]
    if len(extreme) > 0:
        logger.warning(f"  Removing {len(extreme)} features with >0.5 label correlation:")
        for col, corr in extreme.items():
            logger.warning(f"    {col}: {corr:.4f}")
        fm = fm.drop(columns=extreme.index.tolist(), errors='ignore')
        removed_log['leakage'] = len(extreme)
    else:
        removed_log['leakage'] = 0

    suspicious = label_corr_final[(label_corr_final > 0.3) & (label_corr_final <= 0.5)]
    if len(suspicious) > 0:
        logger.info(f"  {len(suspicious)} features with 0.3-0.5 correlation (review):")
        for col, corr in suspicious.head(15).items():
            logger.info(f"    {col}: {corr:.4f}")

    # ── 5i. TOP FEATURES ─────────────────────────────────────
    numeric_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c != LABEL]
    label_corr_final = fm[numeric_cols].corrwith(fm[LABEL]).abs().sort_values(ascending=False)

    logger.info(f"\n  Top 30 features by |correlation with {LABEL}|:")
    for i, (col, corr) in enumerate(label_corr_final.head(30).items()):
        is_cancer = col.startswith(PREFIX)
        is_adv = any(col.startswith(p) for p in [
            'CLUSTER_', 'VISIT_', 'TRAJ_', 'LAB_TRAJ_', 'MED_ESC_',
            'INV_PATTERN_', 'CROSS_', 'DECAY_', 'MONTHLY_', 'ROLLING3M_',
            'CAT_', 'MEDCAT_', 'LABTERM_', 'PAIR_', 'CMPAIR_', 'SEQ_',
            'RECUR_', 'MEDREC_', 'RATE_', 'AGE_', 'AGEX_', 'ENTROPY_', 'GINI_'
        ])
        marker = '*' if is_cancer else ('+' if is_adv else ' ')
        logger.info(f"    {marker} {i+1:2d}. {col}: {corr:.4f}")

    # ── 5j. FINAL SUMMARY ────────────────────────────────────
    logger.info(f"\n-- FINAL SUMMARY --")
    logger.info(f"  Initial features:   {initial_cols}")
    total_removed = 0
    for step, count in removed_log.items():
        if count > 0:
            logger.info(f"    {step:20s}: -{count}")
            total_removed += count
    logger.info(f"  Total removed:      {total_removed}")
    logger.info(f"  Final features:     {fm.shape[1]}")
    logger.info(f"  Final patients:     {fm.shape[0]}")

    # Feature group breakdown
    logger.info(f"\n  Feature group breakdown:")
    accounted = set()
    for prefix_str, name in cfg.FEATURE_GROUP_DISPLAY:
        cols = [c for c in fm.columns if c.startswith(prefix_str) and c not in accounted]
        if cols:
            accounted.update(cols)
            logger.info(f"    {name:25s}: {len(cols)}")
    other = [c for c in fm.columns if c not in accounted and c != LABEL]
    if other:
        logger.info(f"    {'Other/Demographics':25s}: {len(other)}")
    logger.info(f"    {'TOTAL':25s}: {fm.shape[1]}")

    # Ratio check
    n_pos = (fm[LABEL] == 1).sum()
    n_features = fm.shape[1] - 1
    logger.info(f"\n  Features-per-positive ratio: {n_features}/{n_pos} = {n_features/max(n_pos,1):.2f}")

    return fm


def run_cleanup(cfg=None):
    """Run cleanup for all windows."""
    if cfg is None:
        cfg = config

    for _w in cfg.WINDOWS:
        (cfg.CLEANUP_RESULTS / _w).mkdir(parents=True, exist_ok=True)

    raw_matrices = {}
    for window in cfg.WINDOWS:
        path = cfg.FE_RESULTS / window / f'feature_matrix_final_{window}.csv'
        if not path.exists():
            logger.warning(f"  Missing {path}")
            continue
        fm = pd.read_csv(path, index_col=0)
        raw_matrices[window] = fm
        logger.info(f"Loaded {window}: {fm.shape[0]} patients x {fm.shape[1]} features")

    cleaned_matrices = {}
    for window in cfg.WINDOWS:
        if window not in raw_matrices:
            continue
        fm = raw_matrices[window].copy()
        cleaned = cleanup_features(fm, window.upper(), cfg)
        cleaned_matrices[window] = cleaned

        # Belt-and-suspenders: ensure column names are LightGBM-JSON and BigQuery safe.
        # FE builders already emit clean names; this catches any future addition that slips through.
        def _sanitize(name):
            s = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
            return re.sub(r'_+', '_', s).strip('_') or 'FEATURE'
        new_cols = [_sanitize(c) for c in cleaned.columns]
        seen = {}; final = []
        for c in new_cols:
            if c in seen:
                seen[c] += 1; final.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0; final.append(c)
        n_changed = sum(1 for a, b in zip(list(cleaned.columns), final) if a != b)
        if n_changed:
            logger.info(f"  Sanitized {n_changed} feature names at cleanup-save boundary")
            cleaned = cleaned.copy()
            cleaned.columns = final

        out_path = cfg.CLEANUP_RESULTS / window / f'feature_matrix_clean_{window}.csv'
        cleaned.to_csv(out_path)
        logger.info(f"  Saved: {out_path}")

    # Cross-window comparison
    logger.info(f"\n{'='*70}")
    logger.info(f"  CROSS-WINDOW COMPARISON")
    logger.info(f"{'='*70}")
    for window in cfg.WINDOWS:
        if window not in cleaned_matrices:
            continue
        fm = cleaned_matrices[window]
        cancer_cols = [c for c in fm.columns if c.startswith(cfg.PREFIX)]
        logger.info(f"  {window}: {fm.shape[0]} patients x {fm.shape[1]} features "
                     f"({len(cancer_cols)} {cfg.CANCER_NAME}-specific)")

    logger.info(f"\n  CLEANUP COMPLETE - Ready for modeling")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_cleanup()
