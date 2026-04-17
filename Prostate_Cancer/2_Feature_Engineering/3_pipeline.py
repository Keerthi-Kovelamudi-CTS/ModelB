# ═══════════════════════════════════════════════════════════════
# GENERIC FEATURE ENGINEERING PIPELINE
# Config-driven — works for any cancer type.
# Functions only — no top-level execution.
# Called by run_pipeline.py.
# ═══════════════════════════════════════════════════════════════

import logging
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import re
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

__all__ = [
    'build_clinical_features',
    'build_medication_features',
    'build_interaction_features',
    'build_advanced_features',
    'extract_maximum_features',
    'build_new_signal_features',
    'build_trend_features',
]


def _get_all_symptom_cats(cfg):
    """Flatten all categories from CLUSTER_DEFINITIONS into one deduplicated list."""
    cats = []
    for _, cat_list in cfg.CLUSTER_DEFINITIONS:
        cats.extend(cat_list)
    return list(set(cats))


# ═══════════════════════════════════════════════════════════════
# STEP 4a: BASE CLINICAL FEATURES
# ═══════════════════════════════════════════════════════════════

def build_clinical_features(clin_df, cfg):
    """Build patient-level features from observation/lab data."""
    df = clin_df.copy()
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')
    patients = df[['PATIENT_GUID', 'LABEL', 'SEX', 'AGE_AT_INDEX', 'INDEX_DATE']].drop_duplicates(subset=['PATIENT_GUID'])

    # 4a. DEMOGRAPHICS
    demo = patients[['PATIENT_GUID', 'LABEL', 'AGE_AT_INDEX']].copy()
    demo['AGE_BAND'] = (demo['AGE_AT_INDEX'] // 5) * 5

    # 4b. OBSERVATION FEATURES — per category (windowed)
    obs_df = df.copy()
    obs_categories = obs_df['CATEGORY'].unique()

    obs_A = obs_df[obs_df['TIME_WINDOW'] == 'A'].groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    obs_B = obs_df[obs_df['TIME_WINDOW'] == 'B'].groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    obs_total = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)

    obs_A.columns = [f"OBS_{c}_count_A" for c in obs_A.columns]
    obs_B.columns = [f"OBS_{c}_count_B" for c in obs_B.columns]
    obs_total.columns = [f"OBS_{c}_count_total" for c in obs_total.columns]

    obs_has_ever = (obs_total > 0).astype(int)
    obs_has_ever.columns = [c.replace('_count_total', '_has_ever') for c in obs_has_ever.columns]

    obs_accel = pd.DataFrame(index=obs_total.index)
    for cat in obs_categories:
        col_a = f"OBS_{cat}_count_A"
        col_b = f"OBS_{cat}_count_B"
        if col_a in obs_A.columns and col_b in obs_B.columns:
            merged = obs_A[[col_a]].join(obs_B[[col_b]], how='outer').fillna(0)
            obs_accel[f"OBS_{cat}_acceleration"] = (merged[col_b] > merged[col_a]).astype(int)

    obs_new_B = pd.DataFrame(index=obs_total.index)
    for cat in obs_categories:
        col_a = f"OBS_{cat}_count_A"
        col_b = f"OBS_{cat}_count_B"
        if col_a in obs_A.columns and col_b in obs_B.columns:
            merged = obs_A[[col_a]].join(obs_B[[col_b]], how='outer').fillna(0)
            obs_new_B[f"OBS_{cat}_new_in_B"] = ((merged[col_b] > 0) & (merged[col_a] == 0)).astype(int)

    # 4c. LAB VALUE FEATURES — per category
    lab_df = df[df['VALUE'].notna() & df['CATEGORY'].isin(cfg.LAB_CATEGORIES)].copy()
    lab_categories = lab_df['CATEGORY'].unique()
    lab_features = pd.DataFrame()

    for cat in lab_categories:
        cat_df = lab_df[lab_df['CATEGORY'] == cat]
        stats = cat_df.groupby('PATIENT_GUID')['VALUE'].agg(
            **{f"LAB_{cat}_mean": 'mean', f"LAB_{cat}_min": 'min', f"LAB_{cat}_max": 'max',
               f"LAB_{cat}_std": 'std', f"LAB_{cat}_count": 'count'}
        )
        last_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        last_val.name = f"LAB_{cat}_last"
        first_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first()
        first_val.name = f"LAB_{cat}_first"
        mean_A = cat_df[cat_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['VALUE'].mean()
        mean_A.name = f"LAB_{cat}_mean_A"
        mean_B = cat_df[cat_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['VALUE'].mean()
        mean_B.name = f"LAB_{cat}_mean_B"
        delta = mean_B.subtract(mean_A)
        delta.name = f"LAB_{cat}_delta"

        def _calc_slope(group):
            if len(group) < 2:
                return np.nan
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
            y = group['VALUE'].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 2:
                return np.nan
            x, y = x[mask], y[mask]
            if x.max() == x.min():
                return np.nan
            try:
                return np.polyfit(x, y, 1)[0]
            except Exception:
                return np.nan
        slope = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID').apply(_calc_slope)
        slope.name = f"LAB_{cat}_slope"
        has_ever = (stats[f"LAB_{cat}_count"] > 0).astype(int)
        has_ever.name = f"LAB_{cat}_has_ever"

        cat_features = pd.concat([stats, last_val, first_val, mean_A, mean_B, delta, slope, has_ever], axis=1)
        lab_features = cat_features if lab_features.empty else lab_features.join(cat_features, how='outer')

    # 4f. INVESTIGATION PATTERN FEATURES
    inv_cats = cfg.INVESTIGATION_CATEGORIES
    inv_df = obs_df[obs_df['CATEGORY'].isin(inv_cats)]
    inv_count_A = inv_df[inv_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    inv_count_A.name = 'INV_count_A'
    inv_count_B = inv_df[inv_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    inv_count_B.name = 'INV_count_B'
    inv_total = inv_df[inv_df['TIME_WINDOW'].isin(['A', 'B'])].groupby('PATIENT_GUID').size()
    inv_total.name = 'INV_count_total'
    inv_unique_types = inv_df[inv_df['TIME_WINDOW'].isin(['A', 'B'])].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    inv_unique_types.name = 'INV_unique_types'
    inv_accel = pd.concat([inv_count_A, inv_count_B], axis=1).fillna(0)
    inv_accel['INV_acceleration'] = (inv_accel['INV_count_B'] > inv_accel['INV_count_A']).astype(int)

    imaging_inv = inv_df[inv_df['CATEGORY'] == inv_cats[0]] if inv_cats else inv_df.head(0)
    imaging_count_A = imaging_inv[imaging_inv['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    imaging_count_A.name = 'INV_imaging_count_A'
    imaging_count_B = imaging_inv[imaging_inv['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    imaging_count_B.name = 'INV_imaging_count_B'
    imaging_total = imaging_inv[imaging_inv['TIME_WINDOW'].isin(['A', 'B'])].groupby('PATIENT_GUID').size()
    imaging_total.name = 'INV_imaging_total'

    # 4g. AGGREGATE CLINICAL FEATURES
    agg_df = df[df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    agg_unique_cats = agg_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    agg_unique_cats.name = 'AGG_unique_categories'
    agg_total_events = agg_df.groupby('PATIENT_GUID').size()
    agg_total_events.name = 'AGG_total_events'
    agg_events_A = agg_df[agg_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    agg_events_A.name = 'AGG_events_A'
    agg_events_B = agg_df[agg_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    agg_events_B.name = 'AGG_events_B'
    agg_unique_codes = agg_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
    agg_unique_codes.name = 'AGG_unique_code_ids'

    symptom_cats = cfg.SYMPTOM_CATEGORIES
    symptom_df = obs_df[obs_df['CATEGORY'].isin(symptom_cats)]
    symptom_count_A = symptom_df[symptom_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    symptom_count_A.name = 'AGG_symptom_count_A'
    symptom_count_B = symptom_df[symptom_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    symptom_count_B.name = 'AGG_symptom_count_B'
    symptom_total = symptom_df[symptom_df['TIME_WINDOW'].isin(['A', 'B'])].groupby('PATIENT_GUID').size()
    symptom_total.name = 'AGG_symptom_count_total'
    symptom_unique = symptom_df[symptom_df['TIME_WINDOW'].isin(['A', 'B'])].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    symptom_unique.name = 'AGG_symptom_unique_categories'
    symptom_merged = pd.concat([symptom_count_A, symptom_count_B], axis=1).fillna(0)
    symptom_merged['AGG_symptom_acceleration'] = (symptom_merged['AGG_symptom_count_B'] > symptom_merged['AGG_symptom_count_A']).astype(int)

    symptom_cats_A = symptom_df[symptom_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    symptom_cats_B = symptom_df[symptom_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    symptom_new_B_count = pd.concat([symptom_cats_A, symptom_cats_B], axis=1, keys=['A', 'B']).apply(
        lambda row: len(row['B'] - row['A']) if pd.notna(row['B']) and pd.notna(row['A'])
        else (len(row['B']) if pd.notna(row['B']) else 0), axis=1
    )
    symptom_new_B_count.name = 'AGG_new_symptom_cats_in_B'

    # 4h. TEMPORAL FEATURES
    temporal_df = agg_df.groupby('PATIENT_GUID')['EVENT_DATE'].agg(['min', 'max'])
    temporal_df['TEMP_event_span_days'] = (temporal_df['max'] - temporal_df['min']).dt.days
    temporal_df = temporal_df[['TEMP_event_span_days']]

    last_event = agg_df.groupby('PATIENT_GUID').agg(
        last_event_date=('EVENT_DATE', 'max'), index_date=('INDEX_DATE', 'first')
    )
    last_event['TEMP_days_last_event_to_index'] = (last_event['index_date'] - last_event['last_event_date']).dt.days

    # MERGE ALL CLINICAL FEATURES
    features = demo.set_index('PATIENT_GUID')
    for feat_df in [obs_A, obs_B, obs_total, obs_has_ever, obs_accel, obs_new_B]:
        if not feat_df.empty:
            features = features.join(feat_df, how='left')
    if not lab_features.empty:
        features = features.join(lab_features, how='left')
    for feat_s in [inv_count_A, inv_count_B, inv_total, inv_unique_types,
                   imaging_count_A, imaging_count_B, imaging_total]:
        features = features.join(feat_s, how='left')
    features = features.join(inv_accel[['INV_acceleration']], how='left')
    for feat_s in [agg_unique_cats, agg_total_events, agg_events_A, agg_events_B,
                   agg_unique_codes, symptom_count_A, symptom_count_B, symptom_total,
                   symptom_unique, symptom_new_B_count]:
        features = features.join(feat_s, how='left')
    features = features.join(symptom_merged[['AGG_symptom_acceleration']], how='left')
    features = features.join(temporal_df, how='left')
    features = features.join(last_event[['TEMP_days_last_event_to_index']], how='left')

    count_cols = [c for c in features.columns if any(x in c for x in
                  ['_count_', '_has_ever', '_flag', '_acceleration', '_new_in_B', '_total', 'AGG_', 'INV_'])]
    features[count_cols] = features[count_cols].fillna(0)
    return features


# ═══════════════════════════════════════════════════════════════
# STEP 4a (cont): MEDICATION FEATURES
# ═══════════════════════════════════════════════════════════════

def build_medication_features(med_df, cfg):
    """Build patient-level features from medication data."""
    df = med_df.copy()
    med_categories = df['CATEGORY'].unique()

    med_A = df[df['TIME_WINDOW'] == 'A'].groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    med_B = df[df['TIME_WINDOW'] == 'B'].groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    med_total = df.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)

    med_A.columns = [f"MED_{c}_count_A" for c in med_A.columns]
    med_B.columns = [f"MED_{c}_count_B" for c in med_B.columns]
    med_total.columns = [f"MED_{c}_count_total" for c in med_total.columns]

    med_has_ever = (med_total > 0).astype(int)
    med_has_ever.columns = [c.replace('_count_total', '_has_ever') for c in med_has_ever.columns]

    med_accel = pd.DataFrame(index=med_total.index)
    for cat in med_categories:
        col_a, col_b = f"MED_{cat}_count_A", f"MED_{cat}_count_B"
        if col_a in med_A.columns and col_b in med_B.columns:
            merged = med_A[[col_a]].join(med_B[[col_b]], how='outer').fillna(0)
            med_accel[f"MED_{cat}_acceleration"] = (merged[col_b] > merged[col_a]).astype(int)

    med_new_B = pd.DataFrame(index=med_total.index)
    for cat in med_categories:
        col_a, col_b = f"MED_{cat}_count_A", f"MED_{cat}_count_B"
        if col_a in med_A.columns and col_b in med_B.columns:
            merged = med_A[[col_a]].join(med_B[[col_b]], how='outer').fillna(0)
            med_new_B[f"MED_{cat}_new_in_B"] = ((merged[col_b] > 0) & (merged[col_a] == 0)).astype(int)

    if 'VALUE' in df.columns:
        med_qty = df.groupby(['PATIENT_GUID', 'CATEGORY'])['VALUE'].agg(['sum', 'mean']).unstack(fill_value=0)
        med_qty_sum = med_qty['sum']
        med_qty_sum.columns = [f"MED_{c}_total_qty" for c in med_qty_sum.columns]
        med_qty_mean = med_qty['mean']
        med_qty_mean.columns = [f"MED_{c}_mean_qty" for c in med_qty_mean.columns]
    else:
        med_qty_sum = pd.DataFrame()
        med_qty_mean = pd.DataFrame()

    med_unique = df.groupby(['PATIENT_GUID', 'CATEGORY'])['CODE_ID'].nunique().unstack(fill_value=0)
    med_unique.columns = [f"MED_{c}_unique_drugs" for c in med_unique.columns]

    # Aggregate
    med_agg_total = df.groupby('PATIENT_GUID').size()
    med_agg_total.name = 'MED_AGG_total_prescriptions'
    med_agg_cats = df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    med_agg_cats.name = 'MED_AGG_unique_categories'
    med_agg_drugs = df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
    med_agg_drugs.name = 'MED_AGG_unique_drugs'
    med_agg_A = df[df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    med_agg_A.name = 'MED_AGG_count_A'
    med_agg_B = df[df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    med_agg_B.name = 'MED_AGG_count_B'
    med_agg_merged = pd.concat([med_agg_A, med_agg_B], axis=1).fillna(0)
    med_agg_merged['MED_AGG_acceleration'] = (med_agg_merged['MED_AGG_count_B'] > med_agg_merged['MED_AGG_count_A']).astype(int)
    polypharmacy = df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    polypharmacy.name = 'MED_AGG_polypharmacy'

    # Merge
    features = pd.DataFrame(index=df['PATIENT_GUID'].unique())
    features.index.name = 'PATIENT_GUID'
    for feat_df in [med_A, med_B, med_total, med_has_ever, med_accel, med_new_B,
                    med_qty_sum, med_qty_mean, med_unique]:
        if not feat_df.empty:
            features = features.join(feat_df, how='left')
    for feat_s in [med_agg_total, med_agg_cats, med_agg_drugs, med_agg_A, med_agg_B, polypharmacy]:
        features = features.join(feat_s, how='left')
    features = features.join(med_agg_merged[['MED_AGG_acceleration']], how='left')
    features = features.fillna(0)
    return features


# ═══════════════════════════════════════════════════════════════
# STEP 4a (cont): INTERACTION FEATURES
# ═══════════════════════════════════════════════════════════════

def build_interaction_features(feature_matrix, cfg):
    """Build clinical x medication interaction features from config pairs."""
    fm = feature_matrix.copy()
    interactions = {}

    for feat_name, col_a, col_b in cfg.INTERACTION_PAIRS:
        if col_a in fm.columns and col_b in fm.columns:
            interactions[f'INT_{feat_name}'] = fm[col_a] * fm[col_b]

    # Multi-symptom burden
    if 'AGG_symptom_unique_categories' in fm.columns:
        interactions['INT_multi_symptom_burden'] = (fm['AGG_symptom_unique_categories'] >= 3).astype(int)

    # High investigation with symptoms
    if 'INV_count_total' in fm.columns and 'AGG_symptom_count_total' in fm.columns:
        interactions['INT_high_investigation_with_symptoms'] = (
            (fm['INV_count_total'] >= 3) & (fm['AGG_symptom_count_total'] >= 2)
        ).astype(int)

    int_df = pd.DataFrame(interactions, index=fm.index)
    fm = pd.concat([fm, int_df], axis=1)
    logger.info(f"  Added {len(interactions)} interaction features")
    return fm


# ═══════════════════════════════════════════════════════════════
# STEP 4b: ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════

def build_advanced_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Clusters, visit patterns, trajectories, escalation, cross-domain, decay."""
    logger.info(f"  ADVANCED FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    adv = pd.DataFrame(index=patient_list)
    adv.index.name = 'PATIENT_GUID'
    obs_df = clin.copy()
    lab_df = clin[clin['VALUE'].notna() & clin['CATEGORY'].isin(cfg.LAB_CATEGORIES)].copy()

    # GROUP 1: SYMPTOM CLUSTERING (from config)
    logger.info(f"  Building symptom clusters...")
    for group_name, cats in cfg.CLUSTER_DEFINITIONS:
        group_df = obs_df[obs_df['CATEGORY'].isin(cats)]
        has_group = group_df.groupby('PATIENT_GUID').size() > 0
        adv[f'CLUSTER_{group_name}_any'] = adv.index.map(has_group).fillna(False).astype(int)
        group_count = group_df.groupby('PATIENT_GUID').size()
        adv[f'CLUSTER_{group_name}_count'] = adv.index.map(group_count).fillna(0).astype(int)
        group_unique = group_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
        adv[f'CLUSTER_{group_name}_breadth'] = adv.index.map(group_unique).fillna(0).astype(int)
        group_B = group_df[group_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
        adv[f'CLUSTER_{group_name}_count_B'] = adv.index.map(group_B).fillna(0).astype(int)

    system_cols = [c for c in adv.columns if c.startswith('CLUSTER_') and c.endswith('_any')]
    adv['CLUSTER_multi_system_count'] = adv[system_cols].sum(axis=1)
    adv['CLUSTER_3plus_systems'] = (adv['CLUSTER_multi_system_count'] >= 3).astype(int)

    # Cross-cluster combinations
    cluster_names = [name for name, _ in cfg.CLUSTER_DEFINITIONS]
    for c1, c2 in combinations(cluster_names[:6], 2):
        col1 = f'CLUSTER_{c1}_any'
        col2 = f'CLUSTER_{c2}_any'
        if col1 in adv.columns and col2 in adv.columns:
            adv[f'CLUSTER_{c1}_AND_{c2}'] = (adv[col1] & adv[col2]).astype(int)

    # GROUP 2: VISIT PATTERNS
    logger.info(f"  Building visit patterns...")
    clin_windowed = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    visits = clin_windowed.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    adv['VISIT_unique_dates'] = adv.index.map(visits).fillna(0).astype(int)
    visits_A = clin_windowed[clin_windowed['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    visits_B = clin_windowed[clin_windowed['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    adv['VISIT_unique_dates_A'] = adv.index.map(visits_A).fillna(0).astype(int)
    adv['VISIT_unique_dates_B'] = adv.index.map(visits_B).fillna(0).astype(int)
    adv['VISIT_acceleration'] = (adv['VISIT_unique_dates_B'] > adv['VISIT_unique_dates_A']).astype(int)
    adv['VISIT_ratio_B_to_A'] = np.where(
        adv['VISIT_unique_dates_A'] > 0,
        adv['VISIT_unique_dates_B'] / adv['VISIT_unique_dates_A'],
        adv['VISIT_unique_dates_B']
    )
    events_per_patient = clin_windowed.groupby('PATIENT_GUID').size()
    adv['VISIT_events_per_visit'] = np.where(
        adv['VISIT_unique_dates'] > 0,
        adv.index.map(events_per_patient).fillna(0) / adv['VISIT_unique_dates'], 0
    )

    # GROUP 3: TEMPORAL TRAJECTORY
    logger.info(f"  Building temporal trajectories...")
    obs_windowed = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        obs_windowed['QUARTER'] = pd.cut(
            obs_windowed['MONTHS_BEFORE_INDEX'],
            bins=[0, 6, 12, 18, 24, 30, 36],
            labels=['Q1_0-6m', 'Q2_6-12m', 'Q3_12-18m', 'Q4_18-24m', 'Q5_24-30m', 'Q6_30-36m'],
            right=True
        )
        quarter_counts = obs_windowed.groupby(['PATIENT_GUID', 'QUARTER']).size().unstack(fill_value=0)
        for col in quarter_counts.columns:
            adv[f'TRAJ_events_{col}'] = adv.index.map(quarter_counts[col]).fillna(0).astype(int)
        q_cols = [c for c in adv.columns if c.startswith('TRAJ_events_Q')]
        if len(q_cols) >= 2:
            adv['TRAJ_increasing_trend'] = (adv[q_cols[0]] > adv[q_cols[-1]]).astype(int)
            adv['TRAJ_trend_ratio'] = np.where(adv[q_cols[-1]] > 0, adv[q_cols[0]] / adv[q_cols[-1]], adv[q_cols[0]])

    # GROUP 4: LAB TRAJECTORIES
    logger.info(f"  Building lab trajectories...")
    lab_windowed = lab_df[lab_df['VALUE'].notna()].copy()
    for lab_cat in cfg.LAB_CATEGORIES:
        cat_labs = lab_windowed[lab_windowed['CATEGORY'] == lab_cat].copy()
        if len(cat_labs) == 0:
            continue
        cat_labs = cat_labs.sort_values(['PATIENT_GUID', 'EVENT_DATE'])

        def calc_slope(group):
            if len(group) < 2:
                return np.nan
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
            y = group['VALUE'].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 2:
                return np.nan
            x, y = x[mask], y[mask]
            if x.max() == x.min():
                return np.nan
            try:
                return np.polyfit(x, y, 1)[0]
            except Exception:
                return np.nan

        slopes = cat_labs.groupby('PATIENT_GUID').apply(calc_slope)
        adv[f'LAB_TRAJ_{lab_cat}_slope'] = adv.index.map(slopes).astype(float)
        adv[f'LAB_TRAJ_{lab_cat}_declining'] = (adv[f'LAB_TRAJ_{lab_cat}_slope'] < 0).astype(int)
        val_range = cat_labs.groupby('PATIENT_GUID')['VALUE'].apply(lambda x: x.max() - x.min())
        adv[f'LAB_TRAJ_{lab_cat}_range'] = adv.index.map(val_range).fillna(0).astype(float)
        cv = cat_labs.groupby('PATIENT_GUID')['VALUE'].apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0)
        adv[f'LAB_TRAJ_{lab_cat}_cv'] = adv.index.map(cv).fillna(0).astype(float)
        fl_diff = cat_labs.groupby('PATIENT_GUID').apply(
            lambda x: x.iloc[-1]['VALUE'] - x.iloc[0]['VALUE'] if len(x) >= 2 else 0
        )
        adv[f'LAB_TRAJ_{lab_cat}_first_last_diff'] = adv.index.map(fl_diff).fillna(0).astype(float)

    # GROUP 5: MEDICATION ESCALATION
    logger.info(f"  Building medication escalation patterns...")
    med_windowed = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    pain_meds = cfg.PAIN_MED_CATEGORIES
    pain_med_df = med_windowed[med_windowed['CATEGORY'].isin(pain_meds)]
    pain_cats = pain_med_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_pain_category_count'] = adv.index.map(pain_cats).fillna(0).astype(int)
    pain_A = pain_med_df[pain_med_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    pain_B = pain_med_df[pain_med_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    adv['MED_ESC_pain_count_A'] = adv.index.map(pain_A).fillna(0).astype(int)
    adv['MED_ESC_pain_count_B'] = adv.index.map(pain_B).fillna(0).astype(int)
    adv['MED_ESC_pain_acceleration'] = (adv['MED_ESC_pain_count_B'] > adv['MED_ESC_pain_count_A']).astype(int)

    steroid_df = med_windowed[med_windowed['CATEGORY'] == cfg.STEROID_CATEGORY]
    has_steroid = steroid_df.groupby('PATIENT_GUID').size() > 0
    adv['MED_ESC_has_corticosteroid'] = adv.index.map(has_steroid).fillna(False).astype(int)

    repeat_cat = cfg.REPEAT_PRESCRIPTION_CATEGORY
    repeat_df = med_windowed[med_windowed['CATEGORY'] == repeat_cat]
    repeat_count = repeat_df.groupby('PATIENT_GUID').size()
    adv[f'MED_ESC_{repeat_cat.lower()}_repeat_count'] = adv.index.map(repeat_count).fillna(0).astype(int)
    adv[f'MED_ESC_{repeat_cat.lower()}_repeat_3plus'] = (adv[f'MED_ESC_{repeat_cat.lower()}_repeat_count'] >= 3).astype(int)

    med_cats_A = med_windowed[med_windowed['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    med_cats_B = med_windowed[med_windowed['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_polypharmacy_A'] = adv.index.map(med_cats_A).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_B'] = adv.index.map(med_cats_B).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_increase'] = (adv['MED_ESC_polypharmacy_B'] > adv['MED_ESC_polypharmacy_A']).astype(int)

    # GROUP 6: INVESTIGATION PATTERNS
    logger.info(f"  Building investigation patterns...")
    inv_df = obs_df[obs_df['CATEGORY'].isin(cfg.INVESTIGATION_CATEGORIES)]
    symptom_cats_all = _get_all_symptom_cats(cfg)
    symptom_df = obs_df[obs_df['CATEGORY'].isin(symptom_cats_all)]
    first_symptom = symptom_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    first_inv = inv_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap = first_inv.subtract(first_symptom).dt.days
    adv['INV_PATTERN_symptom_to_inv_days'] = adv.index.map(gap).fillna(-1).astype(float)
    adv['INV_PATTERN_has_inv_after_symptom'] = (adv['INV_PATTERN_symptom_to_inv_days'] >= 0).astype(int)
    symptom_count = symptom_df.groupby('PATIENT_GUID').size()
    inv_count = inv_df.groupby('PATIENT_GUID').size()
    adv['INV_PATTERN_symptom_count'] = adv.index.map(symptom_count).fillna(0).astype(int)
    adv['INV_PATTERN_inv_count'] = adv.index.map(inv_count).fillna(0).astype(int)

    # GROUP 7: CROSS-DOMAIN
    logger.info(f"  Building cross-domain interactions...")
    high_visits = (adv.get('VISIT_unique_dates', pd.Series(0, index=adv.index)) >= 5)
    multi_system = (adv.get('CLUSTER_multi_system_count', pd.Series(0, index=adv.index)) >= 2)
    adv['CROSS_diagnostic_odyssey'] = (high_visits & multi_system).astype(int)

    # GROUP 8: TIME-DECAY
    logger.info(f"  Building time-decay features...")
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        obs_decay = obs_windowed.copy()
        obs_decay['DECAY_WEIGHT'] = np.exp(-0.1 * obs_decay['MONTHS_BEFORE_INDEX'])
        symptom_weighted = obs_decay[obs_decay['CATEGORY'].isin(symptom_cats_all)]
        ws = symptom_weighted.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
        adv['DECAY_symptom_weighted_score'] = adv.index.map(ws).fillna(0).astype(float)
        wt = obs_decay.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
        adv['DECAY_total_weighted_score'] = adv.index.map(wt).fillna(0).astype(float)
        for cat in cfg.DECAY_CATEGORIES:
            cat_w = obs_decay[obs_decay['CATEGORY'] == cat].groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
            adv[f'DECAY_{cat}_weighted'] = adv.index.map(cat_w).fillna(0).astype(float)

    # Fill
    lab_traj_cols = [c for c in adv.columns if c.startswith('LAB_TRAJ_') and
                     any(x in c for x in ['_slope', '_range', '_cv', '_first_last_diff'])]
    for col in lab_traj_cols:
        med_val = adv[col].median()
        adv[col] = adv[col].fillna(med_val if pd.notna(med_val) else 0)
    adv = adv.fillna(0).replace([np.inf, -np.inf], 0)

    logger.info(f"  Advanced features: {adv.shape[1]}")
    return adv


# ═══════════════════════════════════════════════════════════════
# STEP 4c: MAXIMUM FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_maximum_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Monthly bins, rolling 3M, per-category granular, co-occurrence, recurrence, rates, age, entropy."""
    logger.info(f"  MAXIMUM FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in med.columns:
        med['MONTHS_BEFORE_INDEX'] = pd.to_numeric(med['MONTHS_BEFORE_INDEX'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    mega = pd.DataFrame(index=patient_list)
    mega.index.name = 'PATIENT_GUID'
    obs_windowed = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_windowed = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()

    # MONTHLY BINS
    logger.info(f"  Building monthly bins...")
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        for m_start in range(0, 13, 3):
            m_end = m_start + 3
            month_df = obs_windowed[(obs_windowed['MONTHS_BEFORE_INDEX'] > m_start) &
                                     (obs_windowed['MONTHS_BEFORE_INDEX'] <= m_end)]
            mc = month_df.groupby('PATIENT_GUID').size()
            mega[f'MONTHLY_obs_{m_start}_{m_end}m'] = mega.index.map(mc).fillna(0).astype(int)
            mu = month_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
            mega[f'MONTHLY_obs_cats_{m_start}_{m_end}m'] = mega.index.map(mu).fillna(0).astype(int)

    # ROLLING 3M
    logger.info(f"  Building rolling 3-month features...")
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        for start in range(0, 10, 3):
            end = start + 3
            window_df = obs_windowed[(obs_windowed['MONTHS_BEFORE_INDEX'] > start) &
                                      (obs_windowed['MONTHS_BEFORE_INDEX'] <= end)]
            wc = window_df.groupby('PATIENT_GUID').size()
            mega[f'ROLLING3M_count_{start}_{end}m'] = mega.index.map(wc).fillna(0).astype(int)

    # PER-CATEGORY GRANULAR
    logger.info(f"  Building per-category granular features...")
    for cat in obs_windowed['CATEGORY'].unique():
        cat_df = obs_windowed[obs_windowed['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        cc = cat_df.groupby('PATIENT_GUID').size()
        mega[f'CAT_{cat}_total'] = mega.index.map(cc).fillna(0).astype(int)
        cu = cat_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
        mega[f'CAT_{cat}_unique_codes'] = mega.index.map(cu).fillna(0).astype(int)

    # PER-MED-CATEGORY GRANULAR
    for cat in med_windowed['CATEGORY'].unique():
        cat_df = med_windowed[med_windowed['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        cc = cat_df.groupby('PATIENT_GUID').size()
        mega[f'MEDCAT_{cat}_total'] = mega.index.map(cc).fillna(0).astype(int)
        cu = cat_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
        mega[f'MEDCAT_{cat}_unique_codes'] = mega.index.map(cu).fillna(0).astype(int)

    # CO-OCCURRENCE PAIRS
    logger.info(f"  Building co-occurrence pairs...")
    patient_cats = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    top_cats = obs_windowed['CATEGORY'].value_counts().head(10).index.tolist()
    for c1, c2 in combinations(top_cats, 2):
        co = patient_cats.apply(lambda x: int(c1 in x and c2 in x) if isinstance(x, set) else 0)
        mega[f'PAIR_{c1}_x_{c2}'] = mega.index.map(co).fillna(0).astype(int)

    # RATES & RATIOS
    logger.info(f"  Building rates & ratios...")
    total_events = obs_windowed.groupby('PATIENT_GUID').size()
    total_cats = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    mega['RATE_events_per_cat'] = np.where(
        mega.index.map(total_cats).fillna(0) > 0,
        mega.index.map(total_events).fillna(0) / mega.index.map(total_cats).fillna(1), 0
    )

    # AGE INTERACTIONS
    logger.info(f"  Building age interactions...")
    age = existing_fm['AGE_AT_INDEX'].reindex(mega.index).fillna(65)
    mega['AGE_over_50'] = (age >= 50).astype(int)
    mega['AGE_over_65'] = (age >= 65).astype(int)
    mega['AGE_over_75'] = (age >= 75).astype(int)

    for cat_name in [n for n, _ in cfg.CLUSTER_DEFINITIONS[:4]]:
        col = f'CLUSTER_{cat_name}_any'
        if col in existing_fm.columns:
            mega[f'AGEX_{cat_name}_over65'] = (
                (age >= 65) & (existing_fm[col].reindex(mega.index).fillna(0) == 1)
            ).astype(int)

    # ENTROPY
    logger.info(f"  Building entropy features...")
    def calc_entropy(group):
        counts = group.value_counts(normalize=True)
        return -(counts * np.log2(counts + 1e-10)).sum()

    cat_entropy = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].apply(calc_entropy)
    mega['ENTROPY_category'] = mega.index.map(cat_entropy).fillna(0).astype(float)

    # GINI
    def calc_gini(group):
        counts = group.value_counts().values
        if len(counts) <= 1:
            return 0
        n = len(counts)
        total = counts.sum()
        if total == 0:
            return 0
        sorted_counts = np.sort(counts)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_counts) / (n * total)) - (n + 1) / n

    cat_gini = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].apply(calc_gini)
    mega['GINI_category'] = mega.index.map(cat_gini).fillna(0).astype(float)

    mega = mega.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"  Maximum features: {mega.shape[1]}")
    return mega


# ═══════════════════════════════════════════════════════════════
# STEP 4e: NEW SIGNAL FEATURES
# ═══════════════════════════════════════════════════════════════

def build_new_signal_features(clin_df, med_df, existing_fm, window_name, cfg):
    """CODE_ID-level features, recency, distinct visit counts."""
    logger.info(f"  NEW SIGNAL FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    nf = pd.DataFrame(index=patient_list)
    nf.index.name = 'PATIENT_GUID'
    obs_AB = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()

    # Per-lab-term features
    logger.info(f"  Building per-lab-term features...")
    lab_obs = obs_AB[obs_AB['VALUE'].notna() & obs_AB['CATEGORY'].isin(cfg.LAB_CATEGORIES)].copy()
    if len(lab_obs) > 0:
        top_terms = lab_obs['TERM'].value_counts().head(30).index.tolist()
        for term in top_terms:
            # Keep only [A-Za-z0-9_] so feature names are LightGBM-JSON-safe AND BigQuery-safe.
            # Truncate to 40 chars to stay readable.
            safe_name = re.sub(r'[^A-Za-z0-9_]', '_', term[:40])
            safe_name = re.sub(r'_+', '_', safe_name).strip('_') or 'TERM'
            term_df = lab_obs[lab_obs['TERM'] == term]
            if len(term_df) < 20:
                continue

            tc = term_df.groupby('PATIENT_GUID').size()
            nf[f'LABTERM_{safe_name}_count'] = nf.index.map(tc).fillna(0).astype(int)
            tm = term_df.groupby('PATIENT_GUID')['VALUE'].mean()
            nf[f'LABTERM_{safe_name}_mean'] = nf.index.map(tm).fillna(np.nan)
            tl = term_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
            nf[f'LABTERM_{safe_name}_last'] = nf.index.map(tl).fillna(np.nan)

            if len(term_df) >= 50:
                def term_slope(group):
                    if len(group) < 2:
                        return np.nan
                    x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
                    y = group['VALUE'].values.astype(float)
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() < 2:
                        return np.nan
                    x, y = x[mask], y[mask]
                    if x.max() == x.min():
                        return np.nan
                    try:
                        return np.polyfit(x, y, 1)[0]
                    except Exception:
                        return np.nan
                ts = term_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID').apply(term_slope)
                nf[f'LABTERM_{safe_name}_slope'] = nf.index.map(ts).fillna(0)

    # Recency features
    logger.info(f"  Building recency features...")
    for cat in obs_AB['CATEGORY'].unique():
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        last_event = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].max()
        index_date = obs_AB.groupby('PATIENT_GUID')['INDEX_DATE'].first()
        days_since = (index_date - last_event).dt.days
        safe_cat = cat[:30]
        nf[f'RECUR_{safe_cat}_days_since_last'] = nf.index.map(days_since).fillna(-1).astype(float)

    # Distinct visit dates per category
    logger.info(f"  Building distinct visit features...")
    for cat in obs_AB['CATEGORY'].unique():
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        visit_dates = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
        safe_cat = cat[:30]
        nf[f'RECUR_{safe_cat}_distinct_visits'] = nf.index.map(visit_dates).fillna(0).astype(int)

    nf = nf.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"  New signal features: {nf.shape[1]}")
    return nf


# ═══════════════════════════════════════════════════════════════
# STEP 4f: TREND FEATURES
# ═══════════════════════════════════════════════════════════════

def build_trend_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Per-symptom frequency trends, interval stats, worsening flags."""
    logger.info(f"  TREND FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    tf = pd.DataFrame(index=patient_list)
    tf.index.name = 'PATIENT_GUID'
    obs_AB = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()

    # Per-symptom frequency trends
    logger.info(f"  Building per-symptom frequency trends...")
    symptom_cats = cfg.SYMPTOM_CATEGORIES
    for cat in symptom_cats:
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat].copy()
        if len(cat_df) < 20:
            continue
        safe_cat = cat[:30]

        # Event count
        ec = cat_df.groupby('PATIENT_GUID').size()
        tf[f'SEQ_{safe_cat}_event_count'] = tf.index.map(ec).fillna(0).astype(int)

        # Time span
        ts = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].agg(['min', 'max'])
        ts['span_days'] = (ts['max'] - ts['min']).dt.days
        tf[f'SEQ_{safe_cat}_span_days'] = tf.index.map(ts['span_days']).fillna(0).astype(float)

        # Frequency (events per year of span)
        tf[f'SEQ_{safe_cat}_freq_per_year'] = np.where(
            tf[f'SEQ_{safe_cat}_span_days'] > 30,
            tf[f'SEQ_{safe_cat}_event_count'] / (tf[f'SEQ_{safe_cat}_span_days'] / 365.25),
            tf[f'SEQ_{safe_cat}_event_count']
        )

        # Mean interval between events
        def mean_interval(group):
            dates = group['EVENT_DATE'].sort_values()
            if len(dates) < 2:
                return np.nan
            intervals = dates.diff().dt.days.dropna()
            return intervals.mean()

        mi = cat_df.groupby('PATIENT_GUID').apply(mean_interval)
        tf[f'SEQ_{safe_cat}_mean_interval_days'] = tf.index.map(mi).fillna(-1).astype(float)

        # Worsening: more events in recent half
        if 'MONTHS_BEFORE_INDEX' in cat_df.columns:
            median_month = cat_df['MONTHS_BEFORE_INDEX'].median()
            if pd.notna(median_month) and median_month > 0:
                early = cat_df[cat_df['MONTHS_BEFORE_INDEX'] > median_month].groupby('PATIENT_GUID').size()
                late = cat_df[cat_df['MONTHS_BEFORE_INDEX'] <= median_month].groupby('PATIENT_GUID').size()
                early_mapped = tf.index.map(early).fillna(0)
                late_mapped = tf.index.map(late).fillna(0)
                tf[f'SEQ_{safe_cat}_is_worsening'] = (late_mapped > early_mapped).astype(int)

    # Per-medication frequency trends
    logger.info(f"  Building per-med frequency trends...")
    for cat in med_AB['CATEGORY'].unique():
        cat_df = med_AB[med_AB['CATEGORY'] == cat].copy()
        if len(cat_df) < 20:
            continue
        safe_cat = cat[:30]

        mc = cat_df.groupby('PATIENT_GUID').size()
        tf[f'MEDREC_{safe_cat}_count'] = tf.index.map(mc).fillna(0).astype(int)

        mu = cat_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
        tf[f'MEDREC_{safe_cat}_unique_drugs'] = tf.index.map(mu).fillna(0).astype(int)

    tf = tf.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"  Trend features: {tf.shape[1]}")
    return tf
