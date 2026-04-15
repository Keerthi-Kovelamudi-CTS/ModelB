# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — FEATURE ENGINEERING PIPELINE
# STEP 4: FEATURE ENGINEERING
# Builds patient-level feature matrix from obs + med data
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from itertools import combinations
from collections import Counter
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'
FE_RESULTS = SCRIPT_DIR / 'results' / 'FE'
for _w in ['3mo', '6mo', '12mo']:
    (FE_RESULTS / _w).mkdir(parents=True, exist_ok=True)

DATA_PREFIX = 'mel'

# Lab-related categories (Melanoma has almost NO lab values — ~60 rows in 53k)
LAB_CATEGORIES = []

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_clinical_features(clin_df):
    """
    Build patient-level features from obs data.
    Returns one row per patient.
    """

    df = clin_df.copy()
    # Ensure date columns are datetime (CSV reads as str)
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')
    patients = df[['PATIENT_GUID','LABEL','SEX','AGE_AT_INDEX','INDEX_DATE']].drop_duplicates(subset=['PATIENT_GUID'])

    # ══════════════════════════════════════════════════════════
    # 4a. DEMOGRAPHICS
    # ══════════════════════════════════════════════════════════
    demo = patients[['PATIENT_GUID','LABEL','AGE_AT_INDEX']].copy()
    demo['AGE_BAND'] = (demo['AGE_AT_INDEX'] // 5) * 5

    # ══════════════════════════════════════════════════════════
    # 4b. OBSERVATION/SYMPTOM FEATURES — per category
    # Windowed: count_A, count_B, count_total, has_ever,
    #           acceleration (B > A), new_in_B
    # Melanoma: all events are OBSERVATION (no separate EVENT_TYPE)
    # ══════════════════════════════════════════════════════════
    obs_df = df.copy()

    obs_categories = obs_df['CATEGORY'].unique()

    # Count per category per window
    obs_A = obs_df[obs_df['TIME_WINDOW'] == 'A'].groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)
    obs_B = obs_df[obs_df['TIME_WINDOW'] == 'B'].groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)
    obs_total = obs_df[obs_df['TIME_WINDOW'].isin(['A','B'])].groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)

    # Rename columns
    obs_A.columns = [f"OBS_{c}_count_A" for c in obs_A.columns]
    obs_B.columns = [f"OBS_{c}_count_B" for c in obs_B.columns]
    obs_total.columns = [f"OBS_{c}_count_total" for c in obs_total.columns]

    # Has ever flags
    obs_has_ever = (obs_total > 0).astype(int)
    obs_has_ever.columns = [c.replace('_count_total','_has_ever') for c in obs_has_ever.columns]

    # Acceleration: more events in B than A (closer to diagnosis)
    obs_accel = pd.DataFrame(index=obs_total.index)
    for cat in obs_categories:
        col_a = f"OBS_{cat}_count_A"
        col_b = f"OBS_{cat}_count_B"
        if col_a in obs_A.columns and col_b in obs_B.columns:
            merged = obs_A[[col_a]].join(obs_B[[col_b]], how='outer').fillna(0)
            obs_accel[f"OBS_{cat}_acceleration"] = (merged[col_b] > merged[col_a]).astype(int)

    # New in B: appeared in B but not in A
    obs_new_B = pd.DataFrame(index=obs_total.index)
    for cat in obs_categories:
        col_a = f"OBS_{cat}_count_A"
        col_b = f"OBS_{cat}_count_B"
        if col_a in obs_A.columns and col_b in obs_B.columns:
            merged = obs_A[[col_a]].join(obs_B[[col_b]], how='outer').fillna(0)
            obs_new_B[f"OBS_{cat}_new_in_B"] = ((merged[col_b] > 0) & (merged[col_a] == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # 4c. LAB VALUE FEATURES — per category
    # Melanoma: almost NO lab data (~60 rows in 53k total).
    # Guard: skip if insufficient data.
    # ══════════════════════════════════════════════════════════
    lab_df = df[df['VALUE'].notna()].copy() if 'VALUE' in df.columns else pd.DataFrame()

    lab_features = pd.DataFrame()

    if len(lab_df) < 50:
        print("  Insufficient lab data (melanoma has minimal lab values), skipping lab features")
    else:
        lab_categories_found = lab_df['CATEGORY'].unique()
        for cat in lab_categories_found:
            cat_df = lab_df[lab_df['CATEGORY'] == cat]
            if len(cat_df) < 10:
                continue

            stats = cat_df.groupby('PATIENT_GUID')['VALUE'].agg(
                **{
                    f"LAB_{cat}_mean": 'mean',
                    f"LAB_{cat}_min": 'min',
                    f"LAB_{cat}_max": 'max',
                    f"LAB_{cat}_std": 'std',
                    f"LAB_{cat}_count": 'count',
                }
            )

            last_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
            last_val.name = f"LAB_{cat}_last"

            first_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first()
            first_val.name = f"LAB_{cat}_first"

            mean_A = cat_df[cat_df['TIME_WINDOW']=='A'].groupby('PATIENT_GUID')['VALUE'].mean()
            mean_A.name = f"LAB_{cat}_mean_A"

            mean_B = cat_df[cat_df['TIME_WINDOW']=='B'].groupby('PATIENT_GUID')['VALUE'].mean()
            mean_B.name = f"LAB_{cat}_mean_B"

            delta = mean_B.subtract(mean_A)
            delta.name = f"LAB_{cat}_delta"

            def _calc_lab_slope(group):
                if len(group) < 2:
                    return np.nan
                x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
                y = group['VALUE'].values.astype(float)
                if x[-1] == 0:
                    return np.nan
                try:
                    return np.polyfit(x, y, 1)[0]
                except Exception:
                    return np.nan
            slope = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID').apply(_calc_lab_slope)
            slope.name = f"LAB_{cat}_slope"

            has_ever = (stats[f"LAB_{cat}_count"] > 0).astype(int)
            has_ever.name = f"LAB_{cat}_has_ever"

            cat_features = pd.concat([stats, last_val, first_val, mean_A, mean_B, delta, slope, has_ever], axis=1)

            if lab_features.empty:
                lab_features = cat_features
            else:
                lab_features = lab_features.join(cat_features, how='outer')

    # ══════════════════════════════════════════════════════════
    # 4f. INVESTIGATION PATTERN FEATURES
    # Melanoma: DERMATOSCOPY, HISTOLOGY, WIDE_EXCISION,
    #           PLASTIC_SURGERY, CRYOTHERAPY, PRIOR_SKIN_PROCEDURES
    # ══════════════════════════════════════════════════════════
    inv_cats = ['DERMATOSCOPY', 'HISTOLOGY', 'WIDE_EXCISION', 'PLASTIC_SURGERY', 'CRYOTHERAPY', 'PRIOR_SKIN_PROCEDURES']
    inv_df = obs_df[obs_df['CATEGORY'].isin(inv_cats)]

    inv_count_A = inv_df[inv_df['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size()
    inv_count_A.name = 'INV_count_A'

    inv_count_B = inv_df[inv_df['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size()
    inv_count_B.name = 'INV_count_B'

    inv_total = inv_df[inv_df['TIME_WINDOW'].isin(['A','B'])].groupby('PATIENT_GUID').size()
    inv_total.name = 'INV_count_total'

    inv_unique_types = inv_df[inv_df['TIME_WINDOW'].isin(['A','B'])].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    inv_unique_types.name = 'INV_unique_types'

    # Investigation acceleration
    inv_accel = pd.concat([inv_count_A, inv_count_B], axis=1).fillna(0)
    inv_accel['INV_acceleration'] = (inv_accel['INV_count_B'] > inv_accel['INV_count_A']).astype(int)

    # Dermatoscopy intensity
    derm_inv = inv_df[inv_df['CATEGORY'] == 'DERMATOSCOPY']
    derm_count_A = derm_inv[derm_inv['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size()
    derm_count_A.name = 'INV_dermatoscopy_count_A'
    derm_count_B = derm_inv[derm_inv['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size()
    derm_count_B.name = 'INV_dermatoscopy_count_B'
    derm_total = derm_inv[derm_inv['TIME_WINDOW'].isin(['A','B'])].groupby('PATIENT_GUID').size()
    derm_total.name = 'INV_dermatoscopy_total'

    # ══════════════════════════════════════════════════════════
    # 4g. AGGREGATE CLINICAL FEATURES
    # ══════════════════════════════════════════════════════════
    agg_df = df[df['TIME_WINDOW'].isin(['A','B'])].copy()

    # Total unique categories per patient
    agg_unique_cats = agg_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    agg_unique_cats.name = 'AGG_unique_categories'

    # Total events per patient
    agg_total_events = agg_df.groupby('PATIENT_GUID').size()
    agg_total_events.name = 'AGG_total_events'

    # Events per month (density)
    agg_events_A = agg_df[agg_df['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size()
    agg_events_A.name = 'AGG_events_A'
    agg_events_B = agg_df[agg_df['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size()
    agg_events_B.name = 'AGG_events_B'

    # Unique CODE_ID codes
    agg_unique_codes = agg_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
    agg_unique_codes.name = 'AGG_unique_code_ids'

    # Symptom-only counts (melanoma symptom categories)
    symptom_cats = [
        'SKIN_LESION', 'MOLE_NAEVUS', 'SKIN_SYMPTOMS', 'SKIN_CONDITIONS',
        'SUN_DAMAGE', 'CLINICAL_SIGNS', 'SKIN_INFECTION', 'OTHER_SKIN_LESIONS',
        'INSECT_BITES'
    ]
    symptom_df = obs_df[obs_df['CATEGORY'].isin(symptom_cats)]

    symptom_count_A = symptom_df[symptom_df['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size()
    symptom_count_A.name = 'AGG_symptom_count_A'
    symptom_count_B = symptom_df[symptom_df['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size()
    symptom_count_B.name = 'AGG_symptom_count_B'
    symptom_total = symptom_df[symptom_df['TIME_WINDOW'].isin(['A','B'])].groupby('PATIENT_GUID').size()
    symptom_total.name = 'AGG_symptom_count_total'

    symptom_unique = symptom_df[symptom_df['TIME_WINDOW'].isin(['A','B'])].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    symptom_unique.name = 'AGG_symptom_unique_categories'

    # Symptom acceleration
    symptom_merged = pd.concat([symptom_count_A, symptom_count_B], axis=1).fillna(0)
    symptom_merged['AGG_symptom_acceleration'] = (symptom_merged['AGG_symptom_count_B'] > symptom_merged['AGG_symptom_count_A']).astype(int)

    # New symptom categories in B
    symptom_cats_A = symptom_df[symptom_df['TIME_WINDOW']=='A'].groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    symptom_cats_B = symptom_df[symptom_df['TIME_WINDOW']=='B'].groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    symptom_new_B_count = pd.concat([symptom_cats_A, symptom_cats_B], axis=1, keys=['A','B']).apply(
        lambda row: len(row['B'] - row['A']) if pd.notna(row['B']) and pd.notna(row['A'])
        else (len(row['B']) if pd.notna(row['B']) else 0), axis=1
    )
    symptom_new_B_count.name = 'AGG_new_symptom_cats_in_B'

    # ══════════════════════════════════════════════════════════
    # 4h. TEMPORAL FEATURES
    # ══════════════════════════════════════════════════════════

    # Days between first and last event
    temporal_df = agg_df.groupby('PATIENT_GUID')['EVENT_DATE'].agg(['min','max'])
    temporal_df['TEMP_event_span_days'] = (temporal_df['max'] - temporal_df['min']).dt.days
    temporal_df = temporal_df[['TEMP_event_span_days']]

    # Days from last event to index date
    last_event = agg_df.groupby('PATIENT_GUID').agg(
        last_event_date=('EVENT_DATE','max'),
        index_date=('INDEX_DATE','first')
    )
    last_event['TEMP_days_last_event_to_index'] = (last_event['index_date'] - last_event['last_event_date']).dt.days

    # ══════════════════════════════════════════════════════════
    # MERGE ALL CLINICAL FEATURES
    # ══════════════════════════════════════════════════════════

    features = demo.set_index('PATIENT_GUID')

    # Observation features
    for feat_df in [obs_A, obs_B, obs_total, obs_has_ever, obs_accel, obs_new_B]:
        if not feat_df.empty:
            features = features.join(feat_df, how='left')

    # Lab features
    if not lab_features.empty:
        features = features.join(lab_features, how='left')

    # Investigation features
    for feat_s in [inv_count_A, inv_count_B, inv_total, inv_unique_types,
                   derm_count_A, derm_count_B, derm_total]:
        features = features.join(feat_s, how='left')
    features = features.join(inv_accel[['INV_acceleration']], how='left')

    # Aggregate features
    for feat_s in [agg_unique_cats, agg_total_events, agg_events_A, agg_events_B,
                   agg_unique_codes, symptom_count_A, symptom_count_B, symptom_total,
                   symptom_unique, symptom_new_B_count]:
        features = features.join(feat_s, how='left')
    features = features.join(symptom_merged[['AGG_symptom_acceleration']], how='left')

    # Temporal features
    features = features.join(temporal_df, how='left')
    features = features.join(last_event[['TEMP_days_last_event_to_index']], how='left')

    # Fill NaN with 0 for count/flag features, leave NaN for lab stats
    count_cols = [c for c in features.columns if any(x in c for x in ['_count_', '_has_ever', '_flag', '_acceleration', '_new_in_B', '_total', 'AGG_', 'INV_'])]
    features[count_cols] = features[count_cols].fillna(0)

    return features


def build_medication_features(med_df):
    """
    Build patient-level features from medication data.
    Returns one row per patient.
    """

    df = med_df.copy()

    # ══════════════════════════════════════════════════════════
    # 4i. MEDICATION FEATURES — per category
    # count_A, count_B, count_total, has_ever,
    # total_quantity, mean_quantity,
    # unique_drugs, acceleration, new_in_B
    # ══════════════════════════════════════════════════════════

    med_categories = df['CATEGORY'].unique()

    # Count per category per window
    med_A = df[df['TIME_WINDOW']=='A'].groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)
    med_B = df[df['TIME_WINDOW']=='B'].groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)
    med_total = df.groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)

    med_A.columns = [f"MED_{c}_count_A" for c in med_A.columns]
    med_B.columns = [f"MED_{c}_count_B" for c in med_B.columns]
    med_total.columns = [f"MED_{c}_count_total" for c in med_total.columns]

    # Has ever
    med_has_ever = (med_total > 0).astype(int)
    med_has_ever.columns = [c.replace('_count_total','_has_ever') for c in med_has_ever.columns]

    # Acceleration
    med_accel = pd.DataFrame(index=med_total.index)
    for cat in med_categories:
        col_a = f"MED_{cat}_count_A"
        col_b = f"MED_{cat}_count_B"
        if col_a in med_A.columns and col_b in med_B.columns:
            merged = med_A[[col_a]].join(med_B[[col_b]], how='outer').fillna(0)
            med_accel[f"MED_{cat}_acceleration"] = (merged[col_b] > merged[col_a]).astype(int)

    # New in B
    med_new_B = pd.DataFrame(index=med_total.index)
    for cat in med_categories:
        col_a = f"MED_{cat}_count_A"
        col_b = f"MED_{cat}_count_B"
        if col_a in med_A.columns and col_b in med_B.columns:
            merged = med_A[[col_a]].join(med_B[[col_b]], how='outer').fillna(0)
            med_new_B[f"MED_{cat}_new_in_B"] = ((merged[col_b] > 0) & (merged[col_a] == 0)).astype(int)

    # Total quantity per category
    if 'VALUE' in df.columns:
        med_qty = df.groupby(['PATIENT_GUID','CATEGORY'])['VALUE'].agg(['sum','mean']).unstack(fill_value=0)
        med_qty_sum = med_qty['sum']
        med_qty_sum.columns = [f"MED_{c}_total_qty" for c in med_qty_sum.columns]
        med_qty_mean = med_qty['mean']
        med_qty_mean.columns = [f"MED_{c}_mean_qty" for c in med_qty_mean.columns]
    else:
        med_qty_sum = pd.DataFrame()
        med_qty_mean = pd.DataFrame()

    # Unique drugs per category
    med_unique = df.groupby(['PATIENT_GUID','CATEGORY'])['CODE_ID'].nunique().unstack(fill_value=0)
    med_unique.columns = [f"MED_{c}_unique_drugs" for c in med_unique.columns]

    # ══════════════════════════════════════════════════════════
    # 4j. AGGREGATE MEDICATION FEATURES
    # ══════════════════════════════════════════════════════════

    # Total prescriptions
    med_agg_total = df.groupby('PATIENT_GUID').size()
    med_agg_total.name = 'MED_AGG_total_prescriptions'

    # Total unique categories
    med_agg_cats = df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    med_agg_cats.name = 'MED_AGG_unique_categories'

    # Total unique drugs
    med_agg_drugs = df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
    med_agg_drugs.name = 'MED_AGG_unique_drugs'

    # Prescriptions per window
    med_agg_A = df[df['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size()
    med_agg_A.name = 'MED_AGG_count_A'
    med_agg_B = df[df['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size()
    med_agg_B.name = 'MED_AGG_count_B'

    # Medication acceleration
    med_agg_merged = pd.concat([med_agg_A, med_agg_B], axis=1).fillna(0)
    med_agg_merged['MED_AGG_acceleration'] = (med_agg_merged['MED_AGG_count_B'] > med_agg_merged['MED_AGG_count_A']).astype(int)

    # Polypharmacy score (unique categories — proxy for complexity)
    polypharmacy = df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    polypharmacy.name = 'MED_AGG_polypharmacy'

    # ══════════════════════════════════════════════════════════
    # MERGE ALL MEDICATION FEATURES
    # ══════════════════════════════════════════════════════════

    features = pd.DataFrame(index=df['PATIENT_GUID'].unique())
    features.index.name = 'PATIENT_GUID'

    for feat_df in [med_A, med_B, med_total, med_has_ever, med_accel, med_new_B,
                    med_qty_sum, med_qty_mean, med_unique]:
        if not feat_df.empty:
            features = features.join(feat_df, how='left')

    for feat_s in [med_agg_total, med_agg_cats, med_agg_drugs, med_agg_A, med_agg_B, polypharmacy]:
        features = features.join(feat_s, how='left')
    features = features.join(med_agg_merged[['MED_AGG_acceleration']], how='left')

    # Fill NaN with 0 for all med features
    features = features.fillna(0)

    return features


def build_interaction_features(feature_matrix):
    """
    Build clinical x medication interaction features.
    Runs on the merged patient-level feature matrix.
    Melanoma-specific interactions.
    """

    fm = feature_matrix.copy()

    # ══════════════════════════════════════════════════════════
    # 4k. INTERACTION FEATURES
    # ══════════════════════════════════════════════════════════

    interactions = {}

    # Skin lesion + Dermatoscopy
    if 'OBS_SKIN_LESION_has_ever' in fm.columns and 'OBS_DERMATOSCOPY_has_ever' in fm.columns:
        interactions['INT_lesion_plus_dermatoscopy'] = fm['OBS_SKIN_LESION_has_ever'] * fm['OBS_DERMATOSCOPY_has_ever']

    # Mole + Excision
    if 'OBS_MOLE_NAEVUS_has_ever' in fm.columns and 'OBS_WIDE_EXCISION_has_ever' in fm.columns:
        interactions['INT_mole_plus_excision'] = fm['OBS_MOLE_NAEVUS_has_ever'] * fm['OBS_WIDE_EXCISION_has_ever']

    # Skin lesion + Histology
    if 'OBS_SKIN_LESION_has_ever' in fm.columns and 'OBS_HISTOLOGY_has_ever' in fm.columns:
        interactions['INT_lesion_plus_histology'] = fm['OBS_SKIN_LESION_has_ever'] * fm['OBS_HISTOLOGY_has_ever']

    # Sun damage + Skin lesion
    if 'OBS_SUN_DAMAGE_has_ever' in fm.columns and 'OBS_SKIN_LESION_has_ever' in fm.columns:
        interactions['INT_sundamage_plus_lesion'] = fm['OBS_SUN_DAMAGE_has_ever'] * fm['OBS_SKIN_LESION_has_ever']

    # Prior skin cancer + New lesion
    if 'OBS_PRIOR_SKIN_CANCER_has_ever' in fm.columns and 'OBS_SKIN_LESION_has_ever' in fm.columns:
        interactions['INT_prior_cancer_plus_lesion'] = fm['OBS_PRIOR_SKIN_CANCER_has_ever'] * fm['OBS_SKIN_LESION_has_ever']

    # Cryotherapy + Skin treatment
    if 'OBS_CRYOTHERAPY_has_ever' in fm.columns and 'OBS_SKIN_TREATMENT_has_ever' in fm.columns:
        interactions['INT_cryo_plus_treatment'] = fm['OBS_CRYOTHERAPY_has_ever'] * fm['OBS_SKIN_TREATMENT_has_ever']

    # Multi-symptom burden (3+ symptom categories)
    if 'AGG_symptom_unique_categories' in fm.columns:
        interactions['INT_multi_symptom_burden'] = (fm['AGG_symptom_unique_categories'] >= 3).astype(int)

    # High investigation with symptoms
    if 'INV_count_total' in fm.columns and 'AGG_symptom_count_total' in fm.columns:
        interactions['INT_high_investigation_with_symptoms'] = (
            (fm['INV_count_total'] >= 3) & (fm['AGG_symptom_count_total'] >= 2)
        ).astype(int)

    # Add all interactions to feature matrix
    int_df = pd.DataFrame(interactions, index=fm.index)
    fm = pd.concat([fm, int_df], axis=1)

    print(f"  Added {len(interactions)} interaction features")

    return fm


# ═══════════════════════════════════════════════════════════════
# RUN FEATURE ENGINEERING FOR ALL 3 WINDOWS
# ═══════════════════════════════════════════════════════════════

# Load dropped-patients obs + med CSVs per window
data = {}
for window, suffix in [('3mo', '3m'), ('6mo', '6m'), ('12mo', '12m')]:
    data[window] = {}
    try:
        data[window]['clinical'] = pd.read_csv(
            f"{BASE_PATH}/{window}/FE_mel_dropped_patients_obs_windowed_{suffix}.csv",
            low_memory=False
        )
        data[window]['med'] = pd.read_csv(
            f"{BASE_PATH}/{window}/FE_mel_dropped_patients_med_windowed_{suffix}.csv",
            low_memory=False
        )
        print(f"Loaded {window}: obs {data[window]['clinical'].shape[0]:,} rows, med {data[window]['med'].shape[0]:,} rows")
    except FileNotFoundError as e:
        print(f"Warning: {window}: {e}")
        data[window]['clinical'] = None
        data[window]['med'] = None

feature_matrices = {}

for window in ['3mo', '6mo', '12mo']:
    print(f"\n{'#'*70}")
    print(f"  FEATURE ENGINEERING — {window.upper()} WINDOW")
    print(f"{'#'*70}")

    clin = data[window]['clinical']
    med = data[window]['med']

    if clin is None or med is None:
        print(f"  Missing data for {window}")
        continue

    # Build clinical features
    print(f"\n  Building clinical features...")
    clin_features = build_clinical_features(clin)
    print(f"  Clinical features: {clin_features.shape[1]} features for {clin_features.shape[0]} patients")

    # Build medication features
    print(f"\n  Building medication features...")
    med_features = build_medication_features(med)
    print(f"  Medication features: {med_features.shape[1]} features for {med_features.shape[0]} patients")

    # Merge clinical + medication on patient_guid
    print(f"\n  Merging clinical + medication features...")
    feature_matrix = clin_features.join(med_features, how='left')

    # Fill NaN medication features with 0 (patients with no meds)
    med_cols = [c for c in feature_matrix.columns if c.startswith('MED_')]
    feature_matrix[med_cols] = feature_matrix[med_cols].fillna(0)

    print(f"  Merged: {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} patients")

    # Build interaction features
    print(f"\n  Building interaction features...")
    feature_matrix = build_interaction_features(feature_matrix)
    print(f"  Final: {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} patients")

    # Summary
    print(f"\n  -- FEATURE SUMMARY --")
    print(f"  Total features: {feature_matrix.shape[1]}")
    print(f"  Total patients: {feature_matrix.shape[0]}")
    print(f"  Positive (label=1): {(feature_matrix['LABEL']==1).sum()}")
    print(f"  Negative (label=0): {(feature_matrix['LABEL']==0).sum()}")

    # Feature type breakdown
    obs_cols = [c for c in feature_matrix.columns if c.startswith('OBS_')]
    lab_cols = [c for c in feature_matrix.columns if c.startswith('LAB_')]
    inv_cols = [c for c in feature_matrix.columns if c.startswith('INV_')]
    agg_cols = [c for c in feature_matrix.columns if c.startswith('AGG_')]
    temp_cols = [c for c in feature_matrix.columns if c.startswith('TEMP_')]
    int_cols = [c for c in feature_matrix.columns if c.startswith('INT_')]

    print(f"\n  Feature breakdown:")
    print(f"    Demographics:     {2}")
    print(f"    Observation:      {len(obs_cols)}")
    if len(lab_cols) > 0:
        print(f"    Lab values:       {len(lab_cols)}")
    print(f"    Investigation:    {len(inv_cols)}")
    print(f"    Aggregate:        {len(agg_cols)}")
    print(f"    Temporal:         {len(temp_cols)}")
    print(f"    Medication:       {len(med_cols)}")
    print(f"    Interaction:      {len(int_cols)}")
    print(f"    ─────────────────────────")
    print(f"    TOTAL:            {feature_matrix.shape[1]}")

    # Check for any remaining issues
    null_pct = (feature_matrix.isnull().sum() / len(feature_matrix) * 100)
    high_null = null_pct[null_pct > 50]
    if len(high_null) > 0:
        print(f"\n  Features with >50% null:")
        for col, pct in high_null.items():
            print(f"    {col}: {pct:.1f}%")
    else:
        print(f"\n  No features with >50% null")

    # Store
    feature_matrices[window] = feature_matrix

    # Save
    out_path = FE_RESULTS / window / f"feature_matrix_{window}.csv"
    feature_matrix.to_csv(out_path)
    print(f"\n  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  STEP 4 COMPLETE — FEATURE MATRICES BUILT")
print(f"{'='*70}")

for window in ['3mo', '6mo', '12mo']:
    fm = feature_matrices.get(window)
    if fm is not None:
        print(f"\n  {window}: {fm.shape[0]} patients x {fm.shape[1]} features")
        print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")

print(f"\n{'='*70}")
print(f"  Ready for STEP 5 (FE Cleanup & Feature Selection)")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — ADVANCED FEATURE ENGINEERING
# Step 4b: Squeeze more signal from existing data
# Run AFTER Step 4 (builds on existing feature matrices)
# Uses in-memory data + feature_matrices from above
# ═══════════════════════════════════════════════════════════════

RESULTS_PATH = SCRIPT_DIR.parent / '3_Modeling'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def build_advanced_features(clin_df, med_df, existing_fm, window_name):
    """
    Build advanced features that capture patterns, trajectories,
    and clinical logic beyond simple counts.
    """

    print(f"\n{'='*70}")
    print(f"  ADVANCED FEATURES — {window_name}")
    print(f"{'='*70}")

    clin = clin_df.copy()
    med = med_df.copy()

    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()

    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    med['INDEX_DATE'] = pd.to_datetime(med['INDEX_DATE'], errors='coerce')

    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')
    if 'VALUE' in med.columns:
        med['VALUE'] = pd.to_numeric(med['VALUE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in med.columns:
        med['MONTHS_BEFORE_INDEX'] = pd.to_numeric(med['MONTHS_BEFORE_INDEX'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    adv = pd.DataFrame(index=patient_list)
    adv.index.name = 'PATIENT_GUID'

    # Melanoma: all events are OBSERVATION
    obs_df = clin.copy()
    lab_df = clin[clin['VALUE'].notna()].copy() if 'VALUE' in clin.columns else pd.DataFrame()

    # ══════════════════════════════════════════════════════════
    # GROUP 1: SYMPTOM CLUSTERING — combinations that signal cancer
    # Melanoma-specific clusters
    # ══════════════════════════════════════════════════════════
    print(f"  Building symptom clusters...")

    lesion_cats = ['SKIN_LESION', 'MOLE_NAEVUS', 'OTHER_SKIN_LESIONS']
    procedure_cats = ['DERMATOSCOPY', 'CRYOTHERAPY', 'WIDE_EXCISION', 'PLASTIC_SURGERY', 'HISTOLOGY']
    treatment_cats = ['SKIN_TREATMENT', 'SUTURE_POSTOP', 'MINOR_SURGERY_ADMIN']
    risk_cats = ['SUN_DAMAGE', 'PRIOR_SKIN_CANCER', 'FAMILY_HISTORY']
    pathway_cats = ['DERMATOLOGY_PATHWAY', 'NICE_7PCL']

    for group_name, cats in [
        ('LESION', lesion_cats), ('PROCEDURE', procedure_cats), ('TREATMENT', treatment_cats),
        ('RISK', risk_cats), ('PATHWAY', pathway_cats)
    ]:
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

    adv['CLUSTER_lesion_AND_procedure'] = (adv['CLUSTER_LESION_any'] & adv['CLUSTER_PROCEDURE_any']).astype(int)
    adv['CLUSTER_procedure_AND_treatment'] = (adv['CLUSTER_PROCEDURE_any'] & adv['CLUSTER_TREATMENT_any']).astype(int)
    adv['CLUSTER_risk_AND_lesion'] = (adv['CLUSTER_RISK_any'] & adv['CLUSTER_LESION_any']).astype(int)
    adv['CLUSTER_pathway_AND_procedure'] = (adv['CLUSTER_PATHWAY_any'] & adv['CLUSTER_PROCEDURE_any']).astype(int)
    adv['CLUSTER_lesion_AND_procedure_AND_treatment'] = (adv['CLUSTER_LESION_any'] & adv['CLUSTER_PROCEDURE_any'] & adv['CLUSTER_TREATMENT_any']).astype(int)
    adv['CLUSTER_risk_AND_lesion_AND_procedure'] = (adv['CLUSTER_RISK_any'] & adv['CLUSTER_LESION_any'] & adv['CLUSTER_PROCEDURE_any']).astype(int)
    adv['CLUSTER_pathway_AND_lesion'] = (adv['CLUSTER_PATHWAY_any'] & adv['CLUSTER_LESION_any']).astype(int)
    adv['CLUSTER_3plus_systems'] = (adv['CLUSTER_multi_system_count'] >= 3).astype(int)

    # ══════════════════════════════════════════════════════════
    # GROUP 2: VISIT PATTERNS
    # ══════════════════════════════════════════════════════════
    print(f"  Building visit patterns...")

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
        adv.index.map(events_per_patient).fillna(0) / adv['VISIT_unique_dates'],
        0
    )
    cats_per_patient = clin_windowed.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['VISIT_categories_per_visit'] = np.where(
        adv['VISIT_unique_dates'] > 0,
        adv.index.map(cats_per_patient).fillna(0) / adv['VISIT_unique_dates'],
        0
    )

    # ══════════════════════════════════════════════════════════
    # GROUP 3: TEMPORAL TRAJECTORY
    # ══════════════════════════════════════════════════════════
    print(f"  Building temporal trajectories...")

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
            first_q, last_q = q_cols[-1], q_cols[0]
            adv['TRAJ_increasing_trend'] = (adv[last_q] > adv[first_q]).astype(int)
            adv['TRAJ_trend_ratio'] = np.where(adv[first_q] > 0, adv[last_q] / adv[first_q], adv[last_q])
        # Skin lesion trajectory
        lesion_obs = obs_windowed[obs_windowed['CATEGORY'] == 'SKIN_LESION']
        lesion_quarters = lesion_obs.groupby(['PATIENT_GUID', 'QUARTER']).size().unstack(fill_value=0)
        for col in lesion_quarters.columns:
            adv[f'TRAJ_lesion_{col}'] = adv.index.map(lesion_quarters[col]).fillna(0).astype(int)
        # Procedure cluster trajectory
        procedure_obs = obs_windowed[obs_windowed['CATEGORY'].isin(procedure_cats)]
        procedure_quarters = procedure_obs.groupby(['PATIENT_GUID', 'QUARTER']).size().unstack(fill_value=0)
        for col in procedure_quarters.columns:
            adv[f'TRAJ_procedure_{col}'] = adv.index.map(procedure_quarters[col]).fillna(0).astype(int)

    # ══════════════════════════════════════════════════════════
    # GROUP 4: LAB TRAJECTORIES
    # Melanoma: SKIP entirely — minimal lab data
    # ══════════════════════════════════════════════════════════
    print(f"  Skipping lab trajectories (melanoma has minimal lab data)")

    # ══════════════════════════════════════════════════════════
    # GROUP 5: MEDICATION ESCALATION PATTERNS
    # Melanoma-specific med categories
    # ══════════════════════════════════════════════════════════
    print(f"  Building medication escalation patterns...")

    med_windowed = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()

    # Topical meds
    topical_meds = ['TOPICAL_STEROIDS', 'TOPICAL_ANTIBIOTICS', 'TOPICAL_SKIN_OTHER', 'TOPICAL_STEROID_SCALP']
    topical_med_df = med_windowed[med_windowed['CATEGORY'].isin(topical_meds)]
    topical_med_cats = topical_med_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_topical_category_count'] = adv.index.map(topical_med_cats).fillna(0).astype(int)
    topical_A = topical_med_df[topical_med_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    topical_B = topical_med_df[topical_med_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    adv['MED_ESC_topical_count_A'] = adv.index.map(topical_A).fillna(0).astype(int)
    adv['MED_ESC_topical_count_B'] = adv.index.map(topical_B).fillna(0).astype(int)
    adv['MED_ESC_topical_acceleration'] = (adv['MED_ESC_topical_count_B'] > adv['MED_ESC_topical_count_A']).astype(int)

    # Systemic meds
    systemic_meds = ['SKIN_ANTIBIOTICS_ORAL', 'ANTIFUNGAL_SYSTEMIC', 'IMMUNOSUPPRESSANT']
    systemic_med_df = med_windowed[med_windowed['CATEGORY'].isin(systemic_meds)]
    has_systemic = systemic_med_df.groupby('PATIENT_GUID').size() > 0
    adv['MED_ESC_has_systemic'] = adv.index.map(has_systemic).fillna(False).astype(int)
    systemic_A = systemic_med_df[systemic_med_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size() > 0
    systemic_B = systemic_med_df[systemic_med_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size() > 0
    adv['MED_ESC_new_systemic_in_B'] = (
        adv.index.map(systemic_B).fillna(False) & ~adv.index.map(systemic_A).fillna(False)
    ).astype(int)

    # Topical to systemic escalation
    has_topical = topical_med_df.groupby('PATIENT_GUID').size() > 0
    adv['MED_ESC_topical_to_systemic'] = (
        adv.index.map(has_topical).fillna(False) & adv.index.map(has_systemic).fillna(False)
    ).astype(int)

    # Surgical meds
    surgical_meds = ['LOCAL_ANAESTHETICS', 'SUTURES', 'WOUND_DRESSINGS', 'WOUND_CARE']
    surgical_med_df = med_windowed[med_windowed['CATEGORY'].isin(surgical_meds)]
    surgical_med_cats = surgical_med_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_surgical_category_count'] = adv.index.map(surgical_med_cats).fillna(0).astype(int)
    surgical_total = surgical_med_df.groupby('PATIENT_GUID').size()
    adv['MED_ESC_surgical_total'] = adv.index.map(surgical_total).fillna(0).astype(int)
    surgical_B = surgical_med_df[surgical_med_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    adv['MED_ESC_surgical_count_B'] = adv.index.map(surgical_B).fillna(0).astype(int)

    # Surgical escalation: systemic + surgical together
    adv['MED_ESC_surgical_escalation'] = (
        (adv['MED_ESC_surgical_total'] > 0) & (adv['MED_ESC_has_systemic'] == 1)
    ).astype(int)

    # Polypharmacy escalation
    med_cats_A = med_windowed[med_windowed['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    med_cats_B = med_windowed[med_windowed['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_polypharmacy_A'] = adv.index.map(med_cats_A).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_B'] = adv.index.map(med_cats_B).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_increase'] = (adv['MED_ESC_polypharmacy_B'] > adv['MED_ESC_polypharmacy_A']).astype(int)
    total_meds = med_windowed.groupby('PATIENT_GUID').size()
    adv['MED_ESC_total_prescriptions'] = adv.index.map(total_meds).fillna(0).astype(int)

    # ══════════════════════════════════════════════════════════
    # GROUP 6: INVESTIGATION PATTERNS
    # Melanoma: DERMATOSCOPY, HISTOLOGY, WIDE_EXCISION, etc.
    # ══════════════════════════════════════════════════════════
    print(f"  Building investigation patterns...")

    inv_cats = ['DERMATOSCOPY', 'HISTOLOGY', 'WIDE_EXCISION', 'PLASTIC_SURGERY', 'CRYOTHERAPY', 'PRIOR_SKIN_PROCEDURES']
    inv_df = obs_df[obs_df['CATEGORY'].isin(inv_cats)]
    symptom_cats_all = lesion_cats + ['SKIN_SYMPTOMS', 'SKIN_CONDITIONS', 'SUN_DAMAGE',
                                       'CLINICAL_SIGNS', 'SKIN_INFECTION', 'INSECT_BITES']
    # Deduplicate
    symptom_cats_all = list(set(symptom_cats_all))
    symptom_df = obs_df[obs_df['CATEGORY'].isin(symptom_cats_all)]
    first_symptom = symptom_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    first_inv = inv_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap = first_inv.subtract(first_symptom).dt.days
    gap.name = 'INV_PATTERN_symptom_to_inv_days'
    adv[gap.name] = adv.index.map(gap).fillna(-1).astype(float)
    adv['INV_PATTERN_has_inv_after_symptom'] = (adv[gap.name] >= 0).astype(int)
    symptom_count = symptom_df.groupby('PATIENT_GUID').size()
    inv_count = inv_df.groupby('PATIENT_GUID').size()
    adv['INV_PATTERN_symptom_count'] = adv.index.map(symptom_count).fillna(0).astype(int)
    adv['INV_PATTERN_inv_count'] = adv.index.map(inv_count).fillna(0).astype(int)
    adv['INV_PATTERN_symptom_to_inv_ratio'] = np.where(
        adv['INV_PATTERN_inv_count'] > 0,
        adv['INV_PATTERN_symptom_count'] / adv['INV_PATTERN_inv_count'],
        adv['INV_PATTERN_symptom_count']
    )
    dermatoscopy_df = obs_df[obs_df['CATEGORY'] == 'DERMATOSCOPY']
    has_dermatoscopy = dermatoscopy_df.groupby('PATIENT_GUID').size() > 0
    adv['INV_PATTERN_has_dermatoscopy'] = adv.index.map(has_dermatoscopy).fillna(False).astype(int)
    histology_df = obs_df[obs_df['CATEGORY'] == 'HISTOLOGY']
    has_histology = histology_df.groupby('PATIENT_GUID').size() > 0
    adv['INV_PATTERN_has_histology'] = adv.index.map(has_histology).fillna(False).astype(int)

    # ══════════════════════════════════════════════════════════
    # GROUP 7: CROSS-DOMAIN INTERACTIONS
    # Melanoma-specific cross-domain features
    # ══════════════════════════════════════════════════════════
    print(f"  Building cross-domain interactions...")

    # CROSS_lesion_investigated (skin lesion + dermatoscopy/histology)
    has_lesion_cluster = (adv.get('CLUSTER_LESION_any', pd.Series(0, index=adv.index)) == 1)
    has_inv_derm_hist = (adv.get('INV_PATTERN_has_dermatoscopy', pd.Series(0, index=adv.index)) == 1) | \
                        (adv.get('INV_PATTERN_has_histology', pd.Series(0, index=adv.index)) == 1)
    adv['CROSS_lesion_investigated'] = (has_lesion_cluster & has_inv_derm_hist).astype(int)

    # CROSS_diagnostic_odyssey (high visits + multi-system)
    high_visits = (adv.get('VISIT_unique_dates', pd.Series(0, index=adv.index)) >= 5)
    multi_system = (adv.get('CLUSTER_multi_system_count', pd.Series(0, index=adv.index)) >= 2)
    adv['CROSS_diagnostic_odyssey'] = (high_visits & multi_system).astype(int)

    # CROSS_treated_no_investigation (skin treatment + no dermatoscopy/histology)
    has_treatment_cluster = (adv.get('CLUSTER_TREATMENT_any', pd.Series(0, index=adv.index)) == 1)
    no_inv_derm_hist = ~has_inv_derm_hist
    adv['CROSS_treated_no_investigation'] = (has_treatment_cluster & no_inv_derm_hist).astype(int)

    # CROSS_sun_damage_with_procedures (sun damage + multiple procedures)
    has_risk = (adv.get('CLUSTER_RISK_any', pd.Series(0, index=adv.index)) == 1)
    has_multi_procedures = (adv.get('CLUSTER_PROCEDURE_count', pd.Series(0, index=adv.index)) >= 2)
    adv['CROSS_sun_damage_with_procedures'] = (has_risk & has_multi_procedures).astype(int)

    # CROSS_excision_with_suture (wide excision + suture removal)
    has_excision = obs_df[obs_df['CATEGORY'] == 'WIDE_EXCISION'].groupby('PATIENT_GUID').size() > 0
    has_suture = obs_df[obs_df['CATEGORY'] == 'SUTURE_POSTOP'].groupby('PATIENT_GUID').size() > 0
    excision_mapped = adv.index.map(has_excision).fillna(False)
    suture_mapped = adv.index.map(has_suture).fillna(False)
    adv['CROSS_excision_with_suture'] = (excision_mapped & suture_mapped).astype(int)

    # ══════════════════════════════════════════════════════════
    # GROUP 8: TIME-DECAY WEIGHTED FEATURES
    # Melanoma-specific symptom categories
    # ══════════════════════════════════════════════════════════
    print(f"  Building time-decay features...")

    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        obs_windowed_copy = obs_windowed.copy()
        obs_windowed_copy['DECAY_WEIGHT'] = np.exp(-0.1 * obs_windowed_copy['MONTHS_BEFORE_INDEX'])
        symptom_weighted = obs_windowed_copy[obs_windowed_copy['CATEGORY'].isin(symptom_cats_all)]
        weighted_score = symptom_weighted.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
        weighted_score.name = 'DECAY_symptom_weighted_score'
        adv[weighted_score.name] = adv.index.map(weighted_score).fillna(0).astype(float)
        weighted_total = obs_windowed_copy.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
        weighted_total.name = 'DECAY_total_weighted_score'
        adv[weighted_total.name] = adv.index.map(weighted_total).fillna(0).astype(float)
        for cat in ['SKIN_LESION', 'MOLE_NAEVUS', 'DERMATOLOGY_PATHWAY', 'SKIN_SYMPTOMS']:
            cat_weighted = obs_windowed_copy[obs_windowed_copy['CATEGORY'] == cat]
            cat_ws = cat_weighted.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
            cat_ws.name = f'DECAY_{cat}_weighted'
            adv[cat_ws.name] = adv.index.map(cat_ws).fillna(0).astype(float)

    # Smart fill: 0 for counts/flags (no lab trajectories to median-fill)
    adv = adv.fillna(0)
    adv = adv.replace([np.inf, -np.inf], 0)

    print(f"\n  Advanced features built: {adv.shape[1]} new features")
    print(f"    CLUSTER: {len([c for c in adv.columns if c.startswith('CLUSTER_')])} | "
          f"VISIT: {len([c for c in adv.columns if c.startswith('VISIT_')])} | "
          f"TRAJ: {len([c for c in adv.columns if c.startswith('TRAJ_')])} | "
          f"MED_ESC: {len([c for c in adv.columns if c.startswith('MED_ESC_')])} | "
          f"INV_PATTERN: {len([c for c in adv.columns if c.startswith('INV_PATTERN_')])} | "
          f"CROSS: {len([c for c in adv.columns if c.startswith('CROSS_')])} | "
          f"DECAY: {len([c for c in adv.columns if c.startswith('DECAY_')])}")

    return adv


# ═══════════════════════════════════════════════════════════════
# RUN STEP 4b FOR ALL 3 WINDOWS
# ═══════════════════════════════════════════════════════════════

enhanced_matrices = {}

for window in ['3mo', '6mo', '12mo']:
    print(f"\n{'#'*70}")
    print(f"  STEP 4b — PROCESSING {window.upper()}")
    print(f"{'#'*70}")

    clin = data[window]['clinical']
    med = data[window]['med']

    if clin is None or med is None:
        print(f"  Missing data for {window}")
        continue

    existing_fm = feature_matrices[window]

    adv_features = build_advanced_features(clin, med, existing_fm, window)
    enhanced = existing_fm.join(adv_features, how='left')
    enhanced = enhanced.fillna(0)
    enhanced = enhanced.replace([np.inf, -np.inf], 0)
    enhanced = enhanced.loc[:, ~enhanced.columns.duplicated()]

    print(f"\n  Final enhanced matrix: {enhanced.shape[0]} patients x {enhanced.shape[1]} features")
    print(f"  (was {existing_fm.shape[1]} features, added {enhanced.shape[1] - existing_fm.shape[1]})")

    enhanced_matrices[window] = enhanced
    out_path = FE_RESULTS / window / f"feature_matrix_enhanced_{window}.csv"
    enhanced.to_csv(out_path)
    print(f"  Saved: {out_path}")


print(f"\n{'='*70}")
print(f"  STEP 4b COMPLETE — ENHANCED FEATURE MATRICES")
print(f"{'='*70}")

for window in ['3mo', '6mo', '12mo']:
    fm = enhanced_matrices.get(window)
    if fm is not None:
        print(f"\n  {window}: {fm.shape[0]} patients x {fm.shape[1]} features")
        print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")

print(f"\n{'='*70}")
msg = "  Run Step 5 cleanup on feature_matrix_<window>.csv or feature_matrix_enhanced_*.csv, then modeling."
print(msg)
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — MAXIMUM FEATURE EXTRACTION
# Step 4c: Extract EVERY possible signal
# Run AFTER Step 4b (builds on enhanced matrices)
# Uses in-memory data + enhanced_matrices from above
# ═══════════════════════════════════════════════════════════════

def extract_maximum_features(clin_df, med_df, existing_fm, window_name):
    """Extract every possible feature from the data."""

    print(f"\n{'='*70}")
    print(f"  MAXIMUM FEATURE EXTRACTION — {window_name}")
    print(f"{'='*70}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    med['INDEX_DATE'] = pd.to_datetime(med['INDEX_DATE'], errors='coerce')
    for col in ['VALUE', 'MONTHS_BEFORE_INDEX', 'AGE_AT_INDEX', 'EVENT_AGE']:
        if col in clin.columns:
            clin[col] = pd.to_numeric(clin[col], errors='coerce')
        if col in med.columns:
            med[col] = pd.to_numeric(med[col], errors='coerce')

    patient_list = existing_fm.index.tolist()
    mega = pd.DataFrame(index=patient_list)
    mega.index.name = 'PATIENT_GUID'

    # Melanoma: all events are OBSERVATION
    obs_df = clin.copy()
    lab_df = clin[clin['VALUE'].notna()].copy() if 'VALUE' in clin.columns else pd.DataFrame()
    obs_AB = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    lab_AB = lab_df[lab_df['TIME_WINDOW'].isin(['A', 'B'])].copy() if len(lab_df) > 0 and 'TIME_WINDOW' in lab_df.columns else pd.DataFrame()

    feature_count = 0

    # BLOCK 1: GRANULAR TIME BINNING
    print(f"\n  BLOCK 1: Monthly time bins...")
    if 'MONTHS_BEFORE_INDEX' in obs_AB.columns:
        for m in range(3, 28):
            month_events = obs_AB[obs_AB['MONTHS_BEFORE_INDEX'] == m]
            counts = month_events.groupby('PATIENT_GUID').size()
            mega[f'MONTHLY_events_m{m}'] = mega.index.map(counts).fillna(0).astype(int)
        for start in range(3, 26):
            end = start + 3
            window_events = obs_AB[
                (obs_AB['MONTHS_BEFORE_INDEX'] >= start) &
                (obs_AB['MONTHS_BEFORE_INDEX'] < end)
            ]
            counts = window_events.groupby('PATIENT_GUID').size()
            mega[f'ROLLING3M_events_m{start}_to_m{end}'] = mega.index.map(counts).fillna(0).astype(int)
        monthly_cols = [c for c in mega.columns if c.startswith('MONTHLY_events_')]
        if monthly_cols:
            mega['MONTHLY_max_events'] = mega[monthly_cols].max(axis=1)
            mega['MONTHLY_month_of_max'] = mega[monthly_cols].idxmax(axis=1).str.extract(r'm(\d+)').astype(float)
            mega['MONTHLY_nonzero_months'] = (mega[monthly_cols] > 0).sum(axis=1)
            mega['MONTHLY_mean_events'] = mega[monthly_cols].mean(axis=1)
            mega['MONTHLY_std_events'] = mega[monthly_cols].std(axis=1).fillna(0)
            mega['MONTHLY_burstiness'] = np.where(
                mega['MONTHLY_mean_events'] > 0,
                mega['MONTHLY_std_events'] / mega['MONTHLY_mean_events'],
                0
            )
    fc = len([c for c in mega.columns if c.startswith('MONTHLY_') or c.startswith('ROLLING3M_')])
    feature_count += fc
    print(f"    Added {fc} features")

    # BLOCK 2: PER-CATEGORY GRANULAR FEATURES
    print(f"\n  BLOCK 2: Per-category granular features...")
    all_obs_cats = obs_AB['CATEGORY'].unique()
    for cat in all_obs_cats:
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat]
        prefix = f'CAT_{cat}'
        if len(cat_df) < 10:
            continue
        first_occ = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        index_dates = cat_df.groupby('PATIENT_GUID')['INDEX_DATE'].first()
        days_first = (index_dates - first_occ).dt.days
        mega[f'{prefix}_days_first_to_index'] = mega.index.map(days_first).fillna(-1).astype(float)
        last_occ = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].max()
        days_last = (index_dates - last_occ).dt.days
        mega[f'{prefix}_days_last_to_index'] = mega.index.map(days_last).fillna(-1).astype(float)
        duration = (last_occ - first_occ).dt.days
        mega[f'{prefix}_duration_days'] = mega.index.map(duration).fillna(0).astype(float)
        unique_dates = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
        mega[f'{prefix}_unique_visit_dates'] = mega.index.map(unique_dates).fillna(0).astype(int)
        def mean_gap(group):
            dates = sorted(group['EVENT_DATE'].dropna().unique())
            if len(dates) < 2:
                return np.nan
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            return np.mean(gaps)
        gaps = cat_df.groupby('PATIENT_GUID').apply(mean_gap)
        mega[f'{prefix}_mean_gap_days'] = mega.index.map(gaps).fillna(-1).astype(float)
        total_count = cat_df.groupby('PATIENT_GUID').size()
        b_count = cat_df[cat_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
        ratio = b_count / total_count
        mega[f'{prefix}_B_concentration'] = mega.index.map(ratio).fillna(0).astype(float)
    fc = len([c for c in mega.columns if c.startswith('CAT_')])
    feature_count += fc
    print(f"    Added {fc} features for {len(all_obs_cats)} categories")

    # BLOCK 3: PER-CATEGORY MEDICATION GRANULAR FEATURES
    print(f"\n  BLOCK 3: Per-category medication granular features...")
    all_med_cats = med_AB['CATEGORY'].unique()
    for cat in all_med_cats:
        cat_df = med_AB[med_AB['CATEGORY'] == cat]
        prefix = f'MEDCAT_{cat}'
        if len(cat_df) < 10:
            continue
        first_rx = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        idx_dates = cat_df.groupby('PATIENT_GUID')['INDEX_DATE'].first()
        days_first = (idx_dates - first_rx).dt.days
        mega[f'{prefix}_days_first_to_index'] = mega.index.map(days_first).fillna(-1).astype(float)
        last_rx = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].max()
        days_last = (idx_dates - last_rx).dt.days
        mega[f'{prefix}_days_last_to_index'] = mega.index.map(days_last).fillna(-1).astype(float)
        rx_duration = (last_rx - first_rx).dt.days
        mega[f'{prefix}_prescribing_duration'] = mega.index.map(rx_duration).fillna(0).astype(float)
        rx_count = cat_df.groupby('PATIENT_GUID').size()
        freq = rx_count / (rx_duration.replace(0, np.nan))
        mega[f'{prefix}_rx_frequency'] = mega.index.map(freq).fillna(0).astype(float)
        unique_drugs = cat_df.groupby('PATIENT_GUID')['TERM'].nunique()
        mega[f'{prefix}_unique_drugs'] = mega.index.map(unique_drugs).fillna(0).astype(int)
        total_count = cat_df.groupby('PATIENT_GUID').size()
        b_count = cat_df[cat_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
        ratio = b_count / total_count
        mega[f'{prefix}_B_concentration'] = mega.index.map(ratio).fillna(0).astype(float)
        if 'VALUE' in cat_df.columns:
            total_qty = cat_df.groupby('PATIENT_GUID')['VALUE'].sum()
            mega[f'{prefix}_total_qty'] = mega.index.map(total_qty).fillna(0).astype(float)
            def qty_change(g):
                g = g.sort_values('EVENT_DATE')
                return g['VALUE'].iloc[-1] - g['VALUE'].iloc[0] if len(g) >= 2 else 0
            qty_delta = cat_df.groupby('PATIENT_GUID').apply(qty_change)
            mega[f'{prefix}_qty_change'] = mega.index.map(qty_delta).fillna(0).astype(float)
    fc2 = len([c for c in mega.columns if c.startswith('MEDCAT_')])
    feature_count += fc2
    print(f"    Added {fc2} features for {len(all_med_cats)} med categories")

    # BLOCK 4: LAB VALUE DEEP FEATURES
    # Melanoma: guard with insufficient data check
    print(f"\n  BLOCK 4: Per-term lab features...")
    if len(lab_AB) < 50:
        print("    Insufficient lab data, skipping lab features")
        fc3 = 0
    else:
        lab_with_vals = lab_AB[lab_AB['VALUE'].notna()].copy()
        lab_terms = lab_with_vals['TERM'].value_counts()
        top_lab_terms = lab_terms[lab_terms >= 50].index.tolist()
        for term in top_lab_terms:
            term_df = lab_with_vals[lab_with_vals['TERM'] == term]
            safe_term = term.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace('/', '_').replace('\\', '_')[:40]
            prefix = f'LABTERM_{safe_term}'
            stats = term_df.groupby('PATIENT_GUID')['VALUE'].agg(['mean', 'last', 'min', 'max', 'std', 'count'])
            stats.columns = [f'{prefix}_{s}' for s in ['mean', 'last', 'min', 'max', 'std', 'count']]
            for col in stats.columns:
                mega[col] = mega.index.map(stats[col]).fillna(np.nan)
            mean_A = term_df[term_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['VALUE'].mean()
            mean_B = term_df[term_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['VALUE'].mean()
            delta = mean_B.subtract(mean_A)
            mega[f'{prefix}_delta'] = mega.index.map(delta).fillna(np.nan)
            def calc_slope(group):
                if len(group) < 2:
                    return np.nan
                x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
                y = group['VALUE'].values.astype(float)
                if x[-1] == 0:
                    return np.nan
                try:
                    return np.polyfit(x, y, 1)[0]
                except Exception:
                    return np.nan
            slopes = term_df.groupby('PATIENT_GUID').apply(calc_slope)
            mega[f'{prefix}_slope'] = mega.index.map(slopes).fillna(np.nan)
            def pct_change(group):
                group = group.sort_values('EVENT_DATE')
                if len(group) < 2 or group['VALUE'].iloc[0] == 0:
                    return np.nan
                return (group['VALUE'].iloc[-1] / group['VALUE'].iloc[0]) - 1
            pct = term_df.groupby('PATIENT_GUID').apply(pct_change)
            mega[f'{prefix}_pct_change'] = mega.index.map(pct).fillna(np.nan)
            # Use control-only reference range to avoid case contamination
            ctrl_patients = existing_fm[existing_fm['LABEL'] == 0].index
            ctrl_vals = term_df[term_df['PATIENT_GUID'].isin(ctrl_patients)]['VALUE']
            pop_mean = ctrl_vals.mean() if len(ctrl_vals) > 10 else term_df['VALUE'].mean()
            pop_std = ctrl_vals.std() if len(ctrl_vals) > 10 else term_df['VALUE'].std()
            if pop_std > 0:
                last_vals = term_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
                is_abnormal = ((last_vals < pop_mean - 2*pop_std) | (last_vals > pop_mean + 2*pop_std)).astype(int)
                mega[f'{prefix}_abnormal'] = mega.index.map(is_abnormal).fillna(0).astype(int)
        fc3 = len([c for c in mega.columns if c.startswith('LABTERM_')])
    feature_count += fc3
    print(f"    Added {fc3} features")

    # BLOCK 5: PAIRWISE CO-OCCURRENCE
    print(f"\n  BLOCK 5: Pairwise co-occurrence...")
    top_obs_cats = obs_AB['CATEGORY'].value_counts().head(20).index.tolist()
    patient_cats = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    pair_count = 0
    for cat1, cat2 in combinations(top_obs_cats, 2):
        safe1, safe2 = cat1[:15], cat2[:15]
        cooccurrence = patient_cats.apply(lambda s: int(cat1 in s and cat2 in s) if isinstance(s, set) else 0)
        mega[f'PAIR_{safe1}_X_{safe2}'] = mega.index.map(cooccurrence).fillna(0).astype(int)
        pair_count += 1
    top_med_cats = med_AB['CATEGORY'].value_counts().head(15).index.tolist()
    patient_med_cats = med_AB.groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    # Melanoma key clinical categories
    key_clinical_cats = ['SKIN_LESION', 'MOLE_NAEVUS', 'DERMATOLOGY_PATHWAY', 'SKIN_TREATMENT',
                         'DERMATOSCOPY', 'HISTOLOGY', 'SUN_DAMAGE']
    for clin_cat in key_clinical_cats:
        for med_cat in top_med_cats:
            safe_c, safe_m = clin_cat[:15], med_cat.replace(' ', '_')[:15]
            has_clin = patient_cats.apply(lambda s: clin_cat in s if isinstance(s, set) else False)
            has_med = patient_med_cats.apply(lambda s: med_cat in s if isinstance(s, set) else False)
            both = (has_clin.reindex(mega.index).fillna(False) & has_med.reindex(mega.index).fillna(False)).astype(int)
            mega[f'CMPAIR_{safe_c}_X_{safe_m}'] = both
            pair_count += 1
    feature_count += pair_count
    print(f"    Added {pair_count} co-occurrence features")

    # BLOCK 6: SEQUENCE FEATURES
    print(f"\n  BLOCK 6: Sequence features...")
    symptom_cats = ['SKIN_LESION', 'MOLE_NAEVUS', 'SKIN_SYMPTOMS', 'SKIN_CONDITIONS',
                    'SUN_DAMAGE', 'CLINICAL_SIGNS', 'SKIN_INFECTION', 'OTHER_SKIN_LESIONS',
                    'INSECT_BITES']
    inv_cats = ['DERMATOSCOPY', 'HISTOLOGY', 'WIDE_EXCISION', 'PLASTIC_SURGERY', 'CRYOTHERAPY', 'PRIOR_SKIN_PROCEDURES']
    med_topical_cats = ['TOPICAL_STEROIDS', 'TOPICAL_ANTIBIOTICS', 'TOPICAL_SKIN_OTHER', 'TOPICAL_STEROID_SCALP']
    med_surgical_cats = ['LOCAL_ANAESTHETICS', 'SUTURES', 'WOUND_DRESSINGS', 'WOUND_CARE']
    sym_first = obs_AB[obs_AB['CATEGORY'].isin(symptom_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    inv_first = obs_AB[obs_AB['CATEGORY'].isin(inv_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    topical_med_first = med_AB[med_AB['CATEGORY'].isin(med_topical_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    surgical_med_first = med_AB[med_AB['CATEGORY'].isin(med_surgical_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap_sym_inv = (inv_first - sym_first).dt.days
    mega['SEQ_symptom_to_inv_days'] = mega.index.map(gap_sym_inv).fillna(-999).astype(float)
    mega['SEQ_inv_before_symptom'] = (mega['SEQ_symptom_to_inv_days'] < 0).astype(int)
    mega['SEQ_inv_within_30d'] = ((mega['SEQ_symptom_to_inv_days'] >= 0) & (mega['SEQ_symptom_to_inv_days'] <= 30)).astype(int)
    mega['SEQ_inv_within_90d'] = ((mega['SEQ_symptom_to_inv_days'] >= 0) & (mega['SEQ_symptom_to_inv_days'] <= 90)).astype(int)
    mega['SEQ_inv_delayed_90d'] = (mega['SEQ_symptom_to_inv_days'] > 90).astype(int)
    gap_sym_topical = (topical_med_first - sym_first).dt.days
    mega['SEQ_symptom_to_topical_days'] = mega.index.map(gap_sym_topical).fillna(-999).astype(float)
    mega['SEQ_topical_before_symptom'] = (mega['SEQ_symptom_to_topical_days'] < 0).astype(int)
    gap_sym_surgical = (surgical_med_first - sym_first).dt.days
    mega['SEQ_symptom_to_surgical_days'] = mega.index.map(gap_sym_surgical).fillna(-999).astype(float)
    for cat in symptom_cats:
        cat_first = obs_AB[obs_AB['CATEGORY'] == cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max() if 'MONTHS_BEFORE_INDEX' in obs_AB.columns else pd.Series(dtype=float)
        mega[f'SEQ_{cat}_first_months_before'] = mega.index.map(cat_first).fillna(-1).astype(float)
        cat_last = obs_AB[obs_AB['CATEGORY'] == cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min() if 'MONTHS_BEFORE_INDEX' in obs_AB.columns else pd.Series(dtype=float)
        mega[f'SEQ_{cat}_last_months_before'] = mega.index.map(cat_last).fillna(-1).astype(float)
    fc4 = len([c for c in mega.columns if c.startswith('SEQ_')])
    feature_count += fc4
    print(f"    Added {fc4} sequence features")

    # BLOCK 7: RECURRENCE FEATURES
    print(f"\n  BLOCK 7: Recurrence features...")
    for cat in all_obs_cats:
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        prefix = f'RECUR_{cat}'
        repeat_count = cat_df.groupby('PATIENT_GUID').size()
        mega[f'{prefix}_total'] = mega.index.map(repeat_count).fillna(0).astype(int)
        mega[f'{prefix}_is_recurring'] = (mega[f'{prefix}_total'] >= 2).astype(int)
        mega[f'{prefix}_is_frequent'] = (mega[f'{prefix}_total'] >= 3).astype(int)
        mega[f'{prefix}_is_persistent'] = (mega[f'{prefix}_total'] >= 4).astype(int)
    for cat in all_med_cats:
        cat_df = med_AB[med_AB['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        prefix = f'MEDREC_{cat}'
        repeat_count = cat_df.groupby('PATIENT_GUID').size()
        mega[f'{prefix}_total'] = mega.index.map(repeat_count).fillna(0).astype(int)
        mega[f'{prefix}_is_recurring'] = (mega[f'{prefix}_total'] >= 2).astype(int)
        mega[f'{prefix}_is_frequent'] = (mega[f'{prefix}_total'] >= 3).astype(int)
    fc5 = len([c for c in mega.columns if c.startswith('RECUR_') or c.startswith('MEDREC_')])
    feature_count += fc5
    print(f"    Added {fc5} recurrence features")

    # BLOCK 8: RATIO & RATE FEATURES
    print(f"\n  BLOCK 8: Ratio & rate features...")
    total_events = obs_AB.groupby('PATIENT_GUID').agg(
        first_date=('EVENT_DATE', 'min'),
        last_date=('EVENT_DATE', 'max'),
        total_count=('EVENT_DATE', 'count'),
        unique_cats=('CATEGORY', 'nunique'),
        unique_codes=('CODE_ID', 'nunique')
    )
    total_events['span_days'] = (total_events['last_date'] - total_events['first_date']).dt.days.replace(0, 1)
    total_events['event_rate_per_day'] = total_events['total_count'] / total_events['span_days']
    mega['RATE_events_per_day'] = mega.index.map(total_events['event_rate_per_day']).fillna(0).astype(float)
    total_events['cat_rate'] = total_events['unique_cats'] / (total_events['span_days'] / 30).replace(0, 1)
    mega['RATE_categories_per_month'] = mega.index.map(total_events['cat_rate']).fillna(0).astype(float)
    symptom_count = obs_AB[obs_AB['CATEGORY'].isin(symptom_cats)].groupby('PATIENT_GUID').size()
    all_count = obs_AB.groupby('PATIENT_GUID').size()
    sym_proportion = symptom_count / all_count
    mega['RATE_symptom_proportion'] = mega.index.map(sym_proportion).fillna(0).astype(float)
    inv_count = obs_AB[obs_AB['CATEGORY'].isin(inv_cats)].groupby('PATIENT_GUID').size()
    inv_proportion = inv_count / all_count
    mega['RATE_investigation_proportion'] = mega.index.map(inv_proportion).fillna(0).astype(float)
    med_total = med_AB.groupby('PATIENT_GUID').size()
    sym_total = obs_AB[obs_AB['CATEGORY'].isin(symptom_cats)].groupby('PATIENT_GUID').size()
    med_sym_ratio = med_total / sym_total.replace(0, 1)
    mega['RATE_med_to_symptom_ratio'] = mega.index.map(med_sym_ratio).fillna(0).astype(float)
    a_events = obs_AB[obs_AB['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    b_events = obs_AB[obs_AB['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    b_to_a_ratio = b_events / a_events.replace(0, 0.5)
    mega['RATE_B_to_A_event_ratio'] = mega.index.map(b_to_a_ratio).fillna(0).astype(float)
    mega['RATE_acceleration_magnitude'] = mega.index.map(b_events).fillna(0) - mega.index.map(a_events).fillna(0)
    fc6 = len([c for c in mega.columns if c.startswith('RATE_')])
    feature_count += fc6
    print(f"    Added {fc6} rate features")

    # BLOCK 9: AGE INTERACTION FEATURES
    print(f"\n  BLOCK 9: Age interaction features...")
    if 'AGE_AT_INDEX' in existing_fm.columns:
        age = existing_fm['AGE_AT_INDEX'].reindex(mega.index).fillna(0)
        mega['AGE_squared'] = age ** 2
        mega['AGE_under_40'] = (age < 40).astype(int)
        mega['AGE_40_50'] = ((age >= 40) & (age < 50)).astype(int)
        mega['AGE_50_60'] = ((age >= 50) & (age < 60)).astype(int)
        mega['AGE_60_70'] = ((age >= 60) & (age < 70)).astype(int)
        mega['AGE_70_80'] = ((age >= 70) & (age < 80)).astype(int)
        mega['AGE_over_80'] = (age >= 80).astype(int)
        for cat in ['SKIN_LESION', 'MOLE_NAEVUS', 'DERMATOLOGY_PATHWAY', 'SKIN_TREATMENT',
                    'DERMATOSCOPY', 'HISTOLOGY', 'SUN_DAMAGE']:
            has_cat_flag = obs_AB[obs_AB['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0
            has_cat_mapped = mega.index.map(has_cat_flag).fillna(False).astype(int)
            mega[f'AGEX_{cat}'] = age * has_cat_mapped
        if 'CLUSTER_multi_system_count' in existing_fm.columns:
            msc = existing_fm['CLUSTER_multi_system_count'].reindex(mega.index).fillna(0)
            mega['AGEX_multi_system'] = age * msc
    fc7 = len([c for c in mega.columns if c.startswith('AGE_') or c.startswith('AGEX_')])
    feature_count += fc7
    print(f"    Added {fc7} age features")

    # BLOCK 11: ENTROPY / DIVERSITY FEATURES
    print(f"\n  BLOCK 11: Entropy & diversity...")
    def calc_entropy(counts):
        if not hasattr(counts, 'values'):
            return 0.0
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = [c/total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    cat_counts = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].apply(lambda x: Counter(x) if len(x) else Counter())
    entropy = cat_counts.apply(calc_entropy)
    mega['ENTROPY_obs_category'] = mega.index.map(entropy).fillna(0).astype(float)
    med_cat_counts = med_AB.groupby('PATIENT_GUID')['CATEGORY'].apply(lambda x: Counter(x) if len(x) else Counter())
    med_entropy = med_cat_counts.apply(calc_entropy)
    mega['ENTROPY_med_category'] = mega.index.map(med_entropy).fillna(0).astype(float)
    code_counts = obs_AB.groupby('PATIENT_GUID')['CODE_ID'].apply(lambda x: Counter(x.dropna()) if len(x) else Counter())
    code_entropy = code_counts.apply(calc_entropy)
    mega['ENTROPY_code_id'] = mega.index.map(code_entropy).fillna(0).astype(float)
    if 'MONTHS_BEFORE_INDEX' in obs_AB.columns:
        month_counts = obs_AB.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].apply(
            lambda x: Counter(x.dropna().astype(int)) if len(x.dropna()) else Counter()
        )
        month_entropy = month_counts.apply(calc_entropy)
        mega['ENTROPY_temporal'] = mega.index.map(month_entropy).fillna(0).astype(float)
    def calc_gini(counts):
        if not hasattr(counts, 'values'):
            return 0.0
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = [c/total for c in counts.values()]
        return 1 - sum(p**2 for p in probs)
    gini = cat_counts.apply(calc_gini)
    mega['GINI_obs_category'] = mega.index.map(gini).fillna(0).astype(float)
    fc9 = len([c for c in mega.columns if c.startswith('ENTROPY_') or c.startswith('GINI_')])
    feature_count += fc9
    print(f"    Added {fc9} entropy features")

    # Smart fill: median for continuous lab features, 0 for counts/flags
    labterm_continuous = [c for c in mega.columns if c.startswith('LABTERM_') and any(x in c for x in ['_mean', '_last', '_min', '_max', '_delta', '_slope', '_pct_change'])]
    for col in labterm_continuous:
        med_val = mega[col].median()
        mega[col] = mega[col].fillna(med_val if pd.notna(med_val) else 0)
    mega = mega.fillna(0)
    mega = mega.replace([np.inf, -np.inf], 0)
    nunique = mega.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        mega = mega.drop(columns=constant_cols)
        print(f"\n  Removed {len(constant_cols)} constant columns")
    print(f"\n  TOTAL NEW FEATURES: {mega.shape[1]}")
    return mega


# RUN STEP 4c FOR ALL 3 WINDOWS
mega_matrices = {}
for window in ['3mo', '6mo', '12mo']:
    print(f"\n\n{'#'*70}")
    print(f"  STEP 4c — {window.upper()}")
    print(f"{'#'*70}")

    clin = data[window]['clinical']
    med = data[window]['med']

    if clin is None or med is None:
        print(f"  Missing data for {window}")
        continue

    existing_fm = enhanced_matrices[window]
    mega_features = extract_maximum_features(clin, med, existing_fm, window)
    final = existing_fm.join(mega_features, how='left', rsuffix='_mega')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]
    mega_matrices[window] = final
    print(f"\n  FINAL MATRIX: {final.shape[0]} patients x {final.shape[1]} features")
    print(f"  (Enhanced had {existing_fm.shape[1]}, added {final.shape[1] - existing_fm.shape[1]})")
    out_path = FE_RESULTS / window / f"feature_matrix_mega_{window}.csv"
    final.to_csv(out_path)
    print(f"  Saved: {out_path}")

print(f"\n{'='*70}")
print(f"  STEP 4c COMPLETE — MEGA FEATURE MATRICES")
print(f"{'='*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = mega_matrices.get(window)
    if fm is not None:
        print(f"\n  {window}: {fm.shape[0]} patients x {fm.shape[1]} features")
        print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
print(f"\n{'='*70}")
print("  MEGA FEATURES COMPLETE — Next: Step 4d melanoma-specific, then Step 5 cleanup -> modeling")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — MELANOMA-SPECIFIC FEATURES (NO COMPOSITE SCORES)
# Step 4d: Melanoma-specific clinical features
# Run AFTER Step 4c (builds on mega_matrices)
# Uses in-memory data + mega_matrices from above
# ═══════════════════════════════════════════════════════════════

def build_melanoma_features(clin_df, med_df, existing_fm, window_name):
    """Melanoma-specific features: NICE NG14, lesion pathway, mimics, risk factors, treatment."""

    print(f"\n{'='*70}")
    print(f"  MELANOMA-SPECIFIC FEATURES — {window_name}")
    print(f"{'='*70}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    mel = pd.DataFrame(index=patient_list)
    mel.index.name = 'PATIENT_GUID'

    obs_AB = clin[clin['TIME_WINDOW'].isin(['A','B'])].copy()
    obs_B = clin[clin['TIME_WINDOW'] == 'B'].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A','B'])].copy()

    age = existing_fm['AGE_AT_INDEX'].reindex(mel.index).fillna(50)
    sex = clin.drop_duplicates('PATIENT_GUID').set_index('PATIENT_GUID')['SEX'].reindex(mel.index).fillna('U')

    # Helper functions
    def has_cat(df, cat):
        has = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0
        return mel.index.map(has).fillna(False).astype(int)

    def count_cat(df, cat):
        cnt = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size()
        return mel.index.map(cnt).fillna(0).astype(int)

    def count_cat_B(cat):
        return count_cat(obs_B, cat)

    m = lambda s, default=0: s.reindex(mel.index).fillna(default)

    # BLOCK 1: NICE NG14 FEATURES
    print(f"  BLOCK 1: NICE NG14 features...")
    has_7pcl = has_cat(obs_AB, 'NICE_7PCL')
    has_derm_pathway = has_cat(obs_AB, 'DERMATOLOGY_PATHWAY')
    has_dermatoscopy = has_cat(obs_AB, 'DERMATOSCOPY')
    has_lesion = has_cat(obs_AB, 'SKIN_LESION')
    has_mole = has_cat(obs_AB, 'MOLE_NAEVUS')
    has_clinical_signs = has_cat(obs_AB, 'CLINICAL_SIGNS')
    has_histology = has_cat(obs_AB, 'HISTOLOGY')

    mel['MEL_NICE_7pcl_flag'] = has_7pcl
    mel['MEL_NICE_pathway_flag'] = has_derm_pathway
    mel['MEL_NICE_dermatoscopy_flag'] = has_dermatoscopy
    mel['MEL_NICE_has_lesion'] = has_lesion
    mel['MEL_NICE_has_mole'] = has_mole
    mel['MEL_NICE_has_clinical_signs'] = has_clinical_signs
    mel['MEL_NICE_pathway_complete'] = ((has_lesion == 1) & (has_dermatoscopy == 1) & (has_histology == 1)).astype(int)
    mel['MEL_NICE_cardinal_count'] = has_7pcl + has_derm_pathway + has_dermatoscopy + has_lesion + has_mole + has_clinical_signs
    mel['MEL_NICE_2plus_flags'] = (mel['MEL_NICE_cardinal_count'] >= 2).astype(int)
    mel['MEL_NICE_3plus_flags'] = (mel['MEL_NICE_cardinal_count'] >= 3).astype(int)

    # BLOCK 2: LESION PATHWAY
    print(f"  BLOCK 2: Lesion pathway...")
    # Time from lesion to dermatoscopy
    lesion_first = obs_AB[obs_AB['CATEGORY'] == 'SKIN_LESION'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    derm_first = obs_AB[obs_AB['CATEGORY'] == 'DERMATOSCOPY'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    excision_first = obs_AB[obs_AB['CATEGORY'] == 'WIDE_EXCISION'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    histology_first = obs_AB[obs_AB['CATEGORY'] == 'HISTOLOGY'].groupby('PATIENT_GUID')['EVENT_DATE'].min()

    gap_lesion_derm = (derm_first - lesion_first).dt.days
    mel['MEL_PATH_lesion_to_dermoscopy_days'] = m(gap_lesion_derm, default=-1)

    gap_derm_excision = (excision_first - derm_first).dt.days
    mel['MEL_PATH_dermoscopy_to_excision_days'] = m(gap_derm_excision, default=-1)

    gap_lesion_histology = (histology_first - lesion_first).dt.days
    mel['MEL_PATH_lesion_to_histology_days'] = m(gap_lesion_histology, default=-1)

    has_excision = has_cat(obs_AB, 'WIDE_EXCISION')
    mel['MEL_PATH_has_excision_biopsy'] = has_excision

    # Procedure count (excision + cryotherapy + plastic + histology)
    excision_count = count_cat(obs_AB, 'WIDE_EXCISION')
    cryo_count = count_cat(obs_AB, 'CRYOTHERAPY')
    plastic_count = count_cat(obs_AB, 'PLASTIC_SURGERY')
    histology_count = count_cat(obs_AB, 'HISTOLOGY')
    mel['MEL_PATH_procedure_count'] = excision_count + cryo_count + plastic_count + histology_count

    # BLOCK 3: MIMIC PATTERNS
    print(f"  BLOCK 3: Mimic patterns...")
    # Benign lesion mimic: has mole but no dermatoscopy
    mel['MEL_mimic_benign_lesion'] = ((has_mole == 1) & (has_dermatoscopy == 0)).astype(int)

    # Keratosis mimic: has other skin lesions + cryotherapy but no histology
    has_other_lesions = has_cat(obs_AB, 'OTHER_SKIN_LESIONS')
    has_cryotherapy = has_cat(obs_AB, 'CRYOTHERAPY')
    mel['MEL_mimic_keratosis'] = ((has_other_lesions == 1) & (has_cryotherapy == 1) & (has_histology == 0)).astype(int)

    # BLOCK 4: RISK FACTORS
    print(f"  BLOCK 4: Risk factors...")
    has_sun_damage = has_cat(obs_AB, 'SUN_DAMAGE')
    has_prior_skin_cancer = has_cat(obs_AB, 'PRIOR_SKIN_CANCER')
    has_family_history = has_cat(obs_AB, 'FAMILY_HISTORY')
    has_immunosuppressant = has_cat(med_AB, 'IMMUNOSUPPRESSANT')

    mel['MEL_RF_sun_damage'] = has_sun_damage
    mel['MEL_RF_prior_skin_cancer'] = has_prior_skin_cancer
    mel['MEL_RF_family_history'] = has_family_history
    mel['MEL_RF_immunosuppressed'] = has_immunosuppressant
    mel['MEL_RF_sun_plus_lesion'] = ((has_sun_damage == 1) & (has_lesion == 1)).astype(int)
    mel['MEL_RF_prior_cancer_plus_new_lesion'] = ((has_prior_skin_cancer == 1) & (has_lesion == 1)).astype(int)
    mel['MEL_RF_high_risk_age'] = (age >= 50).astype(int)
    mel['MEL_RF_male'] = (sex == 'M').astype(int)
    mel['MEL_RF_risk_factor_count'] = has_sun_damage + has_prior_skin_cancer + has_family_history + has_immunosuppressant
    mel['MEL_RF_2plus_risk_factors'] = (mel['MEL_RF_risk_factor_count'] >= 2).astype(int)

    # BLOCK 5: TREATMENT PATTERNS
    print(f"  BLOCK 5: Treatment patterns...")
    has_topical_steroids = has_cat(med_AB, 'TOPICAL_STEROIDS')
    has_topical_abx = has_cat(med_AB, 'TOPICAL_ANTIBIOTICS')
    has_local_anaesthetic = has_cat(med_AB, 'LOCAL_ANAESTHETICS')
    has_sutures = has_cat(med_AB, 'SUTURES')
    has_wound_dressing = has_cat(med_AB, 'WOUND_DRESSINGS')
    has_wound_care = has_cat(med_AB, 'WOUND_CARE')
    has_oral_abx = has_cat(med_AB, 'SKIN_ANTIBIOTICS_ORAL')

    # Topical to surgical escalation
    has_any_topical = ((has_topical_steroids == 1) | (has_topical_abx == 1)).astype(int)
    has_any_surgical = ((has_local_anaesthetic == 1) | (has_sutures == 1) | (has_wound_dressing == 1) | (has_wound_care == 1)).astype(int)
    mel['MEL_TX_topical_to_surgical'] = ((has_any_topical == 1) & (has_any_surgical == 1)).astype(int)

    mel['MEL_TX_excision_count'] = excision_count
    mel['MEL_TX_cryotherapy_count'] = cryo_count
    mel['MEL_TX_multiple_procedures'] = (mel['MEL_PATH_procedure_count'] >= 3).astype(int)
    mel['MEL_TX_surgical_escalation'] = ((has_any_surgical == 1) & (has_oral_abx == 1)).astype(int)

    # BLOCK 6: LAB FEATURES — SKIP entirely for melanoma
    print(f"  BLOCK 6: Skipping lab features (melanoma has minimal lab data)")

    mel = mel.fillna(0)
    mel = mel.replace([np.inf, -np.inf], 0)

    # Remove constant columns
    nunique = mel.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        mel = mel.drop(columns=constant)
        print(f"  Removed {len(constant)} constant columns")

    print(f"\n  Melanoma-specific features: {mel.shape[1]} features")
    for prefix, name in [
        ('MEL_NICE_', 'NICE NG14 features'), ('MEL_PATH_', 'Lesion pathway'),
        ('MEL_mimic_', 'Mimic patterns'), ('MEL_RF_', 'Risk factors'),
        ('MEL_TX_', 'Treatment patterns')
    ]:
        count = len([c for c in mel.columns if c.startswith(prefix)])
        if count > 0:
            print(f"    {name:25s}: {count}")
    return mel


# RUN STEP 4d FOR ALL 3 WINDOWS
final_matrices = {}
for window in ['3mo', '6mo', '12mo']:
    print(f"\n\n{'#'*70}")
    print(f"  STEP 4d — {window.upper()}")
    print(f"{'#'*70}")

    clin = data[window]['clinical']
    med = data[window]['med']

    if clin is None or med is None:
        print(f"  Missing data for {window}")
        continue

    existing = mega_matrices[window]
    mel_features = build_melanoma_features(clin, med, existing, window)
    final = existing.join(mel_features, how='left', rsuffix='_mel')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]
    mel_cols = [c for c in final.columns if c.startswith('MEL_')]
    nzv_remove = [col for col in mel_cols if final[col].value_counts(normalize=True).iloc[0] > 0.995]
    if nzv_remove:
        final = final.drop(columns=nzv_remove)
        print(f"  Removed {len(nzv_remove)} near-zero-var melanoma features")
    final_matrices[window] = final
    print(f"\n  FINAL: {final.shape[0]} x {final.shape[1]} features (added {final.shape[1] - existing.shape[1]} melanoma-specific)")
    out_path = FE_RESULTS / window / f"feature_matrix_final_{window}.csv"
    final.to_csv(out_path)
    print(f"  Saved: {out_path}")
    numeric_cols = [c for c in final.select_dtypes(include=[np.number]).columns if c != 'LABEL']
    label_corr = final[numeric_cols].corrwith(final['LABEL']).abs().sort_values(ascending=False)
    print(f"\n  Top 20 features:")
    for i, (col, corr) in enumerate(label_corr.head(20).items()):
        marker = '*' if col.startswith('MEL_') else ' '
        print(f"    {marker} {i+1:2d}. {col}: {corr:.4f}")
    mel_ranks = [(col, corr, rank+1) for rank, (col, corr) in enumerate(label_corr.items()) if col.startswith('MEL_')][:15]
    print(f"\n  Melanoma-specific feature rankings:")
    for col, corr, rank in mel_ranks:
        print(f"    Rank {rank:3d}: {col}: {corr:.4f}")

print(f"\n{'='*70}")
print(f"  STEP 4d COMPLETE — FINAL MATRICES (Option A, No Composite Scores)")
print(f"{'='*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = final_matrices.get(window)
    if fm is not None:
        mel_cols = [c for c in fm.columns if c.startswith('MEL_')]
        print(f"\n  {window}: {fm.shape[0]} x {fm.shape[1]} features ({len(mel_cols)} melanoma-specific)")
        print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
print(f"\n  Next: Run Step 4e (new signal features), then Step 5 cleanup -> modeling.")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — NEW SIGNAL FEATURES
# Step 4e: CODE_ID-level, recency, distinct visit counts
# Run AFTER Step 4d (builds on final_matrices)
# ═══════════════════════════════════════════════════════════════

def build_new_signal_features(clin_df, med_df, existing_fm, window_name):
    """
    Features that capture signal not already in the pipeline:
    1. Top individual CODE_ID binary features (sub-category granularity)
    2. Symptom recency scores (time-weighted importance)
    3. Distinct visit date counts per symptom (persistence signal)
    """

    print(f"\n{'='*70}")
    print(f"  NEW SIGNAL FEATURES — {window_name}")
    print(f"{'='*70}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    med['INDEX_DATE'] = pd.to_datetime(med['INDEX_DATE'], errors='coerce')
    for col in ['VALUE', 'MONTHS_BEFORE_INDEX']:
        if col in clin.columns:
            clin[col] = pd.to_numeric(clin[col], errors='coerce')
        if col in med.columns:
            med[col] = pd.to_numeric(med[col], errors='coerce')

    patient_list = existing_fm.index.tolist()
    nf = pd.DataFrame(index=patient_list)
    nf.index.name = 'PATIENT_GUID'

    # Melanoma: all events are OBSERVATION
    obs_df = clin.copy()
    obs_AB = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: TOP CODE_ID-LEVEL BINARY FEATURES
    # Individual CODE_ID codes have more specificity than categories
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 1: Top CODE_ID-level binary features...")

    # Find CODE_IDs that differ most between pos/neg cohorts
    pos_patients = set(existing_fm[existing_fm['LABEL'] == 1].index)
    neg_patients = set(existing_fm[existing_fm['LABEL'] == 0].index)
    n_pos = len(pos_patients)
    n_neg = len(neg_patients)

    # Clinical CODE_ID codes
    obs_codes = obs_AB[obs_AB['CODE_ID'].notna()].copy()
    code_patient = obs_codes.groupby('CODE_ID')['PATIENT_GUID'].apply(set)

    code_scores = []
    for code_id, patients_with in code_patient.items():
        if len(patients_with) < 20:  # Skip very rare codes
            continue
        pos_with = len(patients_with & pos_patients)
        neg_with = len(patients_with & neg_patients)
        # Rate ratio: how much more common in positive vs negative
        pos_rate = pos_with / n_pos if n_pos > 0 else 0
        neg_rate = neg_with / n_neg if n_neg > 0 else 0
        if neg_rate > 0:
            rate_ratio = pos_rate / neg_rate
        else:
            rate_ratio = pos_rate * 100 if pos_rate > 0 else 0
        # Absolute difference
        abs_diff = abs(pos_rate - neg_rate)
        code_scores.append({
            'code_id': code_id,
            'rate_ratio': rate_ratio,
            'abs_diff': abs_diff,
            'pos_rate': pos_rate,
            'neg_rate': neg_rate,
            'total_patients': len(patients_with)
        })

    if code_scores:
        score_df = pd.DataFrame(code_scores)
        # Select top 50 by absolute rate difference (most discriminative)
        top_codes = score_df.nlargest(50, 'abs_diff')

        # Get CODE_ID term names for readable column names
        code_terms = obs_codes.drop_duplicates('CODE_ID').set_index('CODE_ID')['TERM'].to_dict()

        for _, row in top_codes.iterrows():
            cid = row['code_id']
            patients_with = code_patient[cid]
            term = code_terms.get(cid, str(cid))
            safe_name = str(term).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace('{', '').replace('}', '').replace('/', '_').replace('\\', '_')[:50]
            nf[f'CODE_OBS_{safe_name}'] = nf.index.isin(patients_with).astype(int)

        print(f"    Added {len(top_codes)} obs CODE_ID features")

    # Medication CODE_ID codes
    med_codes = med_AB[med_AB['CODE_ID'].notna()].copy()
    # Use TERM instead of CODE_ID for meds (some may have -1 as CODE_ID)
    if 'TERM' in med_codes.columns:
        med_term_patient = med_codes.groupby('TERM')['PATIENT_GUID'].apply(set)

        med_scores = []
        for term, patients_with in med_term_patient.items():
            if len(patients_with) < 20:
                continue
            pos_with = len(patients_with & pos_patients)
            neg_with = len(patients_with & neg_patients)
            pos_rate = pos_with / n_pos if n_pos > 0 else 0
            neg_rate = neg_with / n_neg if n_neg > 0 else 0
            abs_diff = abs(pos_rate - neg_rate)
            med_scores.append({
                'term': term,
                'abs_diff': abs_diff,
                'pos_rate': pos_rate,
                'neg_rate': neg_rate,
                'total_patients': len(patients_with)
            })

        if med_scores:
            med_score_df = pd.DataFrame(med_scores)
            top_med_terms = med_score_df.nlargest(30, 'abs_diff')

            for _, row in top_med_terms.iterrows():
                term = row['term']
                patients_with = med_term_patient[term]
                safe_name = str(term).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace('{', '').replace('}', '').replace('/', '_').replace('\\', '_')[:50]
                nf[f'CODE_MED_{safe_name}'] = nf.index.isin(patients_with).astype(int)

            print(f"    Added {len(top_med_terms)} med CODE_ID/term features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: SYMPTOM RECENCY SCORES
    # Recent symptoms matter more — weight by inverse distance to index
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 2: Symptom recency scores...")

    key_symptoms = [
        'SKIN_LESION', 'MOLE_NAEVUS', 'SKIN_SYMPTOMS', 'SKIN_CONDITIONS',
        'SUN_DAMAGE', 'CLINICAL_SIGNS', 'SKIN_INFECTION', 'OTHER_SKIN_LESIONS',
        'DERMATOLOGY_PATHWAY', 'DERMATOSCOPY', 'HISTOLOGY', 'WIDE_EXCISION',
        'CRYOTHERAPY', 'PRIOR_SKIN_CANCER', 'FAMILY_HISTORY', 'NICE_7PCL',
        'SKIN_TREATMENT', 'SUTURE_POSTOP', 'PLASTIC_SURGERY',
        'PRIOR_SKIN_PROCEDURES', 'MINOR_SURGERY_ADMIN', 'VIROLOGY_SKIN',
        'INSECT_BITES'
    ]

    for cat in key_symptoms:
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat].copy()
        if len(cat_df) < 10:
            continue

        # Recency score: 1 / (days_to_index + 1) — higher = more recent
        if 'MONTHS_BEFORE_INDEX' in cat_df.columns:
            # Use most recent occurrence
            min_months = cat_df.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
            recency = 1.0 / (min_months + 1)
            nf[f'RECENCY_{cat}_score'] = nf.index.map(recency).fillna(0).astype(float)

            # Sum of recency across all occurrences (captures both recency + frequency)
            cat_df_copy = cat_df.copy()
            cat_df_copy['_recency'] = 1.0 / (cat_df_copy['MONTHS_BEFORE_INDEX'] + 1)
            sum_recency = cat_df_copy.groupby('PATIENT_GUID')['_recency'].sum()
            nf[f'RECENCY_{cat}_cumulative'] = nf.index.map(sum_recency).fillna(0).astype(float)

    recency_cols = [c for c in nf.columns if c.startswith('RECENCY_')]
    print(f"    Added {len(recency_cols)} recency features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: DISTINCT VISIT DATES PER SYMPTOM
    # How many separate GP visits had this symptom — persistence signal
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 3: Distinct visit dates per symptom...")

    for cat in key_symptoms:
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat]
        if len(cat_df) < 10:
            continue

        # Unique visit dates (not just event count — captures separate consultations)
        unique_visits = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
        nf[f'VISITS_{cat}_distinct_dates'] = nf.index.map(unique_visits).fillna(0).astype(int)

        # Is persistent (3+ distinct visits)
        nf[f'VISITS_{cat}_persistent'] = (nf[f'VISITS_{cat}_distinct_dates'] >= 3).astype(int)

        # Window B distinct visits (recent persistence)
        cat_B = cat_df[cat_df['TIME_WINDOW'] == 'B']
        b_visits = cat_B.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
        nf[f'VISITS_{cat}_B_distinct'] = nf.index.map(b_visits).fillna(0).astype(int)

    visit_cols = [c for c in nf.columns if c.startswith('VISITS_')]
    print(f"    Added {len(visit_cols)} visit-based features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: COMBINED RECENCY x PERSISTENCE INTERACTIONS
    # Patients with recent AND persistent symptoms are highest risk
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 4: Recency x persistence interactions...")

    interaction_count = 0
    for cat in key_symptoms:
        rec_col = f'RECENCY_{cat}_score'
        vis_col = f'VISITS_{cat}_distinct_dates'
        if rec_col in nf.columns and vis_col in nf.columns:
            # Recency x frequency product (high when both recent AND frequent)
            nf[f'RxP_{cat}_score'] = nf[rec_col] * nf[vis_col]
            interaction_count += 1

    print(f"    Added {interaction_count} recency x persistence features")

    # Clean up
    nf = nf.fillna(0).replace([np.inf, -np.inf], 0)
    nunique = nf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        nf = nf.drop(columns=constant)
        print(f"\n  Removed {len(constant)} constant columns")

    print(f"\n  New signal features: {nf.shape[1]} total")
    for prefix, name in [
        ('CODE_OBS_', 'CODE_ID obs'), ('CODE_MED_', 'CODE_ID med'),
        ('RECENCY_', 'Recency'), ('VISITS_', 'Distinct visits'), ('RxP_', 'Recency x Persistence')
    ]:
        count = len([c for c in nf.columns if c.startswith(prefix)])
        if count > 0:
            print(f"    {name:25s}: {count}")

    return nf


# RUN STEP 4e FOR ALL 3 WINDOWS
for window in ['3mo', '6mo', '12mo']:
    print(f"\n\n{'#'*70}")
    print(f"  STEP 4e — {window.upper()}")
    print(f"{'#'*70}")

    clin = data[window]['clinical']
    med = data[window]['med']

    if clin is None or med is None:
        print(f"  Missing data for {window}")
        continue

    existing = final_matrices[window]

    new_features = build_new_signal_features(clin, med, existing, window)
    final = existing.join(new_features, how='left', rsuffix='_4e')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]

    final_matrices[window] = final

    print(f"\n  FINAL: {final.shape[0]} x {final.shape[1]} features (added {new_features.shape[1]} new)")

    # Save as the new final
    out_path = FE_RESULTS / window / f"feature_matrix_final_{window}.csv"
    final.to_csv(out_path)
    print(f"  Saved: {out_path}")

print(f"\n{'='*70}")
print(f"  STEP 4e COMPLETE — NEW SIGNAL FEATURES ADDED")
print(f"{'='*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = final_matrices.get(window)
    if fm is not None:
        print(f"\n  {window}: {fm.shape[0]} x {fm.shape[1]} features")
        print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
print(f"\n  Next: Run Step 4f (trend features), then Step 5 cleanup -> modeling.")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — TREND & FREQUENCY FEATURES
# Step 4f: Per-symptom frequency trends, interval stats, worsening flags
# Per-medication trend features
# Inspired by AI-UK lung cancer v2-ensemble pipeline
# ═══════════════════════════════════════════════════════════════

def build_trend_features(clin_df, med_df, existing_fm, window_name):
    """
    Per-symptom/per-med trend features:
    1. Frequency per month per symptom
    2. Frequency trend slope per symptom
    3. Mean/median interval between occurrences per symptom
    4. First-half vs second-half frequency (acceleration magnitude)
    5. IS_WORSENING flag per symptom
    6. Lab trends SKIPPED (melanoma has minimal lab data)
    """

    print(f"\n{'='*70}")
    print(f"  TREND FEATURES — {window_name}")
    print(f"{'='*70}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    med['INDEX_DATE'] = pd.to_datetime(med['INDEX_DATE'], errors='coerce')
    for col in ['VALUE', 'MONTHS_BEFORE_INDEX']:
        if col in clin.columns:
            clin[col] = pd.to_numeric(clin[col], errors='coerce')
        if col in med.columns:
            med[col] = pd.to_numeric(med[col], errors='coerce')

    patient_list = existing_fm.index.tolist()
    tf = pd.DataFrame(index=patient_list)
    tf.index.name = 'PATIENT_GUID'

    # Melanoma: all events are OBSERVATION
    obs_df = clin.copy()
    lab_df = clin[clin['VALUE'].notna()].copy() if 'VALUE' in clin.columns else pd.DataFrame()
    obs_AB = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()

    key_symptoms = [
        'SKIN_LESION', 'MOLE_NAEVUS', 'SKIN_SYMPTOMS', 'SKIN_CONDITIONS',
        'SUN_DAMAGE', 'CLINICAL_SIGNS', 'DERMATOLOGY_PATHWAY',
        'DERMATOSCOPY', 'HISTOLOGY', 'WIDE_EXCISION',
        'CRYOTHERAPY', 'SKIN_TREATMENT', 'PRIOR_SKIN_CANCER'
    ]

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: PER-SYMPTOM FREQUENCY & TREND FEATURES
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 1: Per-symptom frequency & trend...")

    for cat in key_symptoms:
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat].copy()
        if len(cat_df) < 10:
            continue

        prefix = f'TREND_{cat}'
        per_patient = cat_df.groupby('PATIENT_GUID')

        # Total occurrences
        count = per_patient.size()
        tf[f'{prefix}_count'] = tf.index.map(count).fillna(0).astype(int)

        # Time span in days
        date_range = per_patient['EVENT_DATE'].agg(['min', 'max'])
        span_days = (date_range['max'] - date_range['min']).dt.days
        tf[f'{prefix}_time_span_days'] = tf.index.map(span_days).fillna(0).astype(float)

        # Frequency per month
        span_months = span_days / 30.44
        freq_per_month = count / span_months.replace(0, np.nan)
        tf[f'{prefix}_freq_per_month'] = tf.index.map(freq_per_month).fillna(0).astype(float)

        # Mean and median interval between occurrences (days)
        def calc_intervals(group):
            dates = sorted(group['EVENT_DATE'].dropna().unique())
            if len(dates) < 2:
                return pd.Series({'mean_interval': np.nan, 'median_interval': np.nan,
                                   'min_interval': np.nan, 'max_interval': np.nan})
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            return pd.Series({
                'mean_interval': np.mean(gaps),
                'median_interval': np.median(gaps),
                'min_interval': np.min(gaps),
                'max_interval': np.max(gaps)
            })

        intervals = per_patient.apply(calc_intervals)
        if not intervals.empty and 'mean_interval' in intervals.columns:
            tf[f'{prefix}_mean_interval_days'] = tf.index.map(intervals['mean_interval']).fillna(-1).astype(float)
            tf[f'{prefix}_median_interval_days'] = tf.index.map(intervals['median_interval']).fillna(-1).astype(float)
            tf[f'{prefix}_min_interval_days'] = tf.index.map(intervals['min_interval']).fillna(-1).astype(float)
            tf[f'{prefix}_max_interval_days'] = tf.index.map(intervals['max_interval']).fillna(-1).astype(float)

        # First-half vs second-half frequency
        if 'MONTHS_BEFORE_INDEX' in cat_df.columns:
            midpoint = cat_df.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].apply(
                lambda x: (x.max() + x.min()) / 2
            )

            def split_halves(group):
                mid = (group['MONTHS_BEFORE_INDEX'].max() + group['MONTHS_BEFORE_INDEX'].min()) / 2
                first_half = (group['MONTHS_BEFORE_INDEX'] > mid).sum()  # earlier (higher months)
                second_half = (group['MONTHS_BEFORE_INDEX'] <= mid).sum()  # recent (lower months)
                return pd.Series({'first_half': first_half, 'second_half': second_half})

            halves = per_patient.apply(split_halves)
            if not halves.empty and 'first_half' in halves.columns:
                tf[f'{prefix}_first_half_freq'] = tf.index.map(halves['first_half']).fillna(0).astype(int)
                tf[f'{prefix}_second_half_freq'] = tf.index.map(halves['second_half']).fillna(0).astype(int)
                # Acceleration ratio: second_half / first_half
                fh = tf[f'{prefix}_first_half_freq'].replace(0, 0.5)
                tf[f'{prefix}_acceleration_ratio'] = tf[f'{prefix}_second_half_freq'] / fh
                # IS_WORSENING: more recent than earlier
                tf[f'{prefix}_is_worsening'] = (tf[f'{prefix}_second_half_freq'] > tf[f'{prefix}_first_half_freq']).astype(int)

            # Frequency trend slope (linear regression of monthly counts)
            def freq_trend_slope(group):
                if len(group) < 3:
                    return np.nan
                months = group['MONTHS_BEFORE_INDEX'].values
                # Count events per month bucket
                month_range = np.arange(months.min(), months.max() + 1)
                if len(month_range) < 2:
                    return np.nan
                monthly_counts = np.array([((months >= m) & (months < m + 1)).sum() for m in month_range])
                x = month_range.astype(float)
                try:
                    slope = np.polyfit(x, monthly_counts, 1)[0]
                    return slope
                except Exception:
                    return np.nan

            slopes = per_patient.apply(freq_trend_slope)
            tf[f'{prefix}_freq_trend_slope'] = tf.index.map(slopes).fillna(0).astype(float)

    symptom_cols = [c for c in tf.columns if c.startswith('TREND_')]
    print(f"    Added {len(symptom_cols)} per-symptom trend features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: PER-MEDICATION FREQUENCY & TREND FEATURES
    # Melanoma-specific medication categories
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 2: Per-medication frequency & trend...")

    key_meds = [
        'SKIN_ANTIBIOTICS_ORAL', 'TOPICAL_STEROIDS', 'TOPICAL_ANTIBIOTICS',
        'ACTINIC_TREATMENT', 'LOCAL_ANAESTHETICS', 'WOUND_DRESSINGS',
        'ANTIFUNGAL_SYSTEMIC', 'IMMUNOSUPPRESSANT', 'TOPICAL_SKIN_OTHER',
        'ANTIVIRAL_SKIN', 'SUTURES', 'WOUND_CARE', 'TOPICAL_STEROID_SCALP'
    ]

    for cat in key_meds:
        cat_df = med_AB[med_AB['CATEGORY'] == cat].copy()
        if len(cat_df) < 10:
            continue

        safe_cat = cat.replace(' ', '_')
        prefix = f'MEDTREND_{safe_cat}'
        per_patient = cat_df.groupby('PATIENT_GUID')

        # Total prescriptions
        count = per_patient.size()
        tf[f'{prefix}_count'] = tf.index.map(count).fillna(0).astype(int)

        # Time span
        date_range = per_patient['EVENT_DATE'].agg(['min', 'max'])
        span_days = (date_range['max'] - date_range['min']).dt.days
        tf[f'{prefix}_time_span_days'] = tf.index.map(span_days).fillna(0).astype(float)

        # Frequency per month
        span_months = span_days / 30.44
        freq_per_month = count / span_months.replace(0, np.nan)
        tf[f'{prefix}_freq_per_month'] = tf.index.map(freq_per_month).fillna(0).astype(float)

        # Mean interval
        def calc_med_interval(group):
            dates = sorted(group['EVENT_DATE'].dropna().unique())
            if len(dates) < 2:
                return np.nan
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            return np.mean(gaps)

        mean_int = per_patient.apply(calc_med_interval)
        tf[f'{prefix}_mean_interval_days'] = tf.index.map(mean_int).fillna(-1).astype(float)

        # First-half vs second-half
        if 'MONTHS_BEFORE_INDEX' in cat_df.columns:
            def med_halves(group):
                mid = (group['MONTHS_BEFORE_INDEX'].max() + group['MONTHS_BEFORE_INDEX'].min()) / 2
                first_half = (group['MONTHS_BEFORE_INDEX'] > mid).sum()
                second_half = (group['MONTHS_BEFORE_INDEX'] <= mid).sum()
                return pd.Series({'first_half': first_half, 'second_half': second_half})

            halves = per_patient.apply(med_halves)
            if not halves.empty and 'first_half' in halves.columns:
                tf[f'{prefix}_first_half_freq'] = tf.index.map(halves['first_half']).fillna(0).astype(int)
                tf[f'{prefix}_second_half_freq'] = tf.index.map(halves['second_half']).fillna(0).astype(int)
                tf[f'{prefix}_is_worsening'] = (tf[f'{prefix}_second_half_freq'] > tf[f'{prefix}_first_half_freq']).astype(int)

    med_cols = [c for c in tf.columns if c.startswith('MEDTREND_')]
    print(f"    Added {len(med_cols)} per-medication trend features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: LAB TRENDS — SKIP for melanoma
    # Melanoma has almost no lab data (~60 rows in 53k)
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 3: Lab trends...")
    if len(lab_df) < 50:
        print("    Insufficient lab data, skipping lab trends (melanoma has minimal lab values)")
    else:
        # If somehow there is enough lab data, process it
        lab_AB = lab_df[lab_df['TIME_WINDOW'].isin(['A', 'B'])].copy() if 'TIME_WINDOW' in lab_df.columns else pd.DataFrame()
        lab_vals = lab_AB[lab_AB['VALUE'].notna()].copy() if len(lab_AB) > 0 else pd.DataFrame()
        if len(lab_vals) >= 50:
            lab_categories = lab_vals['CATEGORY'].unique()
            for cat in lab_categories:
                cat_labs = lab_vals[lab_vals['CATEGORY'] == cat].copy()
                if len(cat_labs) < 10:
                    continue
                prefix = f'LABTREND_{cat}'
                cat_labs = cat_labs.sort_values(['PATIENT_GUID', 'EVENT_DATE'])
                def pct_change(group):
                    if len(group) < 2:
                        return np.nan
                    first_val = group.iloc[0]['VALUE']
                    last_val = group.iloc[-1]['VALUE']
                    if first_val == 0:
                        return np.nan
                    return (last_val - first_val) / abs(first_val)
                pct = cat_labs.groupby('PATIENT_GUID').apply(pct_change)
                tf[f'{prefix}_pct_change'] = tf.index.map(pct).fillna(0).astype(float)
            lab_trend_cols = [c for c in tf.columns if c.startswith('LABTREND_')]
            print(f"    Added {len(lab_trend_cols)} lab trend features")
        else:
            print("    Insufficient lab data after filtering, skipping")

    # Clean up
    tf = tf.fillna(0).replace([np.inf, -np.inf], 0)
    nunique = tf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        tf = tf.drop(columns=constant)
        print(f"\n  Removed {len(constant)} constant columns")

    print(f"\n  Trend features: {tf.shape[1]} total")
    for prefix, name in [
        ('TREND_', 'Symptom trends'), ('MEDTREND_', 'Med trends'),
        ('LABTREND_', 'Lab category trends')
    ]:
        count = len([c for c in tf.columns if c.startswith(prefix)])
        if count > 0:
            print(f"    {name:25s}: {count}")

    return tf


# RUN STEP 4f FOR ALL 3 WINDOWS
for window in ['3mo', '6mo', '12mo']:
    print(f"\n\n{'#'*70}")
    print(f"  STEP 4f — {window.upper()}")
    print(f"{'#'*70}")

    clin = data[window]['clinical']
    med = data[window]['med']

    if clin is None or med is None:
        print(f"  Missing data for {window}")
        continue

    existing = final_matrices[window]

    trend_features = build_trend_features(clin, med, existing, window)
    final = existing.join(trend_features, how='left', rsuffix='_4f')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]

    final_matrices[window] = final

    print(f"\n  FINAL: {final.shape[0]} x {final.shape[1]} features (added {trend_features.shape[1]} new)")

    out_path = FE_RESULTS / window / f"feature_matrix_final_{window}.csv"
    final.to_csv(out_path)
    print(f"  Saved: {out_path}")

print(f"\n{'='*70}")
print(f"  STEP 4f COMPLETE — TREND FEATURES ADDED")
print(f"{'='*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = final_matrices.get(window)
    if fm is not None:
        print(f"\n  {window}: {fm.shape[0]} x {fm.shape[1]} features")
        print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
