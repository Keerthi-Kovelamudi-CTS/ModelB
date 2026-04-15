# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — FEATURE ENGINEERING PIPELINE
# STEP 4: FEATURE ENGINEERING
# Builds patient-level feature matrix from clinical + med data
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

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_clinical_features(clin_df):
    """
    Build patient-level features from clinical/obs/lab data.
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
    # ═══════════════════════════════════════��══════════════════
    obs_df = df[df['EVENT_TYPE'] == 'OBSERVATION'].copy()
    
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
    # mean_A, mean_B, last_value, min, max, delta (mean_B - mean_A),
    # count, has_ever, is_abnormal_last
    # ═══════════════════��══════════════════════════════════════
    lab_df = df[df['EVENT_TYPE'] == 'LAB VALUE'].copy()
    lab_df = lab_df[lab_df['VALUE'].notna()]
    
    lab_categories = lab_df['CATEGORY'].unique()
    
    lab_features = pd.DataFrame()
    
    for cat in lab_categories:
        cat_df = lab_df[lab_df['CATEGORY'] == cat]
        
        # Overall stats per patient
        stats = cat_df.groupby('PATIENT_GUID')['VALUE'].agg(
            **{
                f"LAB_{cat}_mean": 'mean',
                f"LAB_{cat}_min": 'min',
                f"LAB_{cat}_max": 'max',
                f"LAB_{cat}_std": 'std',
                f"LAB_{cat}_count": 'count',
            }
        )
        
        # Last value (closest to index date)
        last_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        last_val.name = f"LAB_{cat}_last"
        
        # First value (earliest in window)
        first_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first()
        first_val.name = f"LAB_{cat}_first"
        
        # Mean per window
        mean_A = cat_df[cat_df['TIME_WINDOW']=='A'].groupby('PATIENT_GUID')['VALUE'].mean()
        mean_A.name = f"LAB_{cat}_mean_A"
        
        mean_B = cat_df[cat_df['TIME_WINDOW']=='B'].groupby('PATIENT_GUID')['VALUE'].mean()
        mean_B.name = f"LAB_{cat}_mean_B"
        
        # Delta: change from A to B
        delta = mean_B.subtract(mean_A)
        delta.name = f"LAB_{cat}_delta"
        
        # Slope: actual linear regression slope over time
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
        
        # Has ever
        has_ever = (stats[f"LAB_{cat}_count"] > 0).astype(int)
        has_ever.name = f"LAB_{cat}_has_ever"
        
        # Combine
        cat_features = pd.concat([stats, last_val, first_val, mean_A, mean_B, delta, slope, has_ever], axis=1)
        
        if lab_features.empty:
            lab_features = cat_features
        else:
            lab_features = lab_features.join(cat_features, how='outer')
    
    # ══════════════════════════════════════════════════════════
    # 4d. COMORBIDITY FEATURES — binary flags
    # ══════════════════════════════════════════════════════════
    comorb_df = df[df['EVENT_TYPE'] == 'COMORBIDITY'].copy()
    
    comorb_flags = comorb_df.groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)
    comorb_flags = (comorb_flags > 0).astype(int)
    comorb_flags.columns = [f"COMORB_{c}_flag" for c in comorb_flags.columns]
    
    # Total comorbidity count
    comorb_total = pd.DataFrame({
        'COMORB_total_count': comorb_flags.sum(axis=1)
    })
    
    # ══════════════════════════════════════════════════════════
    # 4e. RISK FACTOR FEATURES — binary flags
    # ══════════════════════════════════════════════════════════
    rf_df = df[df['EVENT_TYPE'] == 'RISK FACTOR'].copy()
    
    rf_flags = rf_df.groupby(['PATIENT_GUID','CATEGORY']).size().unstack(fill_value=0)
    rf_flags = (rf_flags > 0).astype(int)
    rf_flags.columns = [f"RF_{c}_flag" for c in rf_flags.columns]
    
    # Total risk factor count
    rf_total = pd.DataFrame({
        'RF_total_count': rf_flags.sum(axis=1)
    })
    
    # ══════════════════════════════════════════════════════════
    # 4f. INVESTIGATION PATTERN FEATURES
    # ══════════════════════════════════════════════════════════
    inv_cats = ['IMAGING', 'GYNAE_PROCEDURES', 'OTHER_PROCEDURES', 'MICROBIOLOGY', 'SCREENING']
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
    
    # Gynae-specific investigation intensity
    gynae_inv = inv_df[inv_df['CATEGORY'] == 'GYNAE_PROCEDURES']
    gynae_inv_count = gynae_inv[gynae_inv['TIME_WINDOW'].isin(['A','B'])].groupby('PATIENT_GUID').size()
    gynae_inv_count.name = 'INV_gynae_count'
    
    # Imaging intensity
    imaging_inv = inv_df[inv_df['CATEGORY'] == 'IMAGING']
    imaging_count_A = imaging_inv[imaging_inv['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size()
    imaging_count_A.name = 'INV_imaging_count_A'
    imaging_count_B = imaging_inv[imaging_inv['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size()
    imaging_count_B.name = 'INV_imaging_count_B'
    imaging_total = imaging_inv[imaging_inv['TIME_WINDOW'].isin(['A','B'])].groupby('PATIENT_GUID').size()
    imaging_total.name = 'INV_imaging_total'
    
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
    
    # Unique SNOMED codes
    agg_unique_snomed = agg_df.groupby('PATIENT_GUID')['SNOMED_ID'].nunique()
    agg_unique_snomed.name = 'AGG_unique_snomed_codes'
    
    # Symptom-only counts (excluding investigations, labs, admin)
    symptom_cats = [
        'ABDOMINAL_MASS','ASCITES','ABDOMINAL_BLOATING','EARLY_SATIETY',
        'ABDOMINAL_PAIN','GI_SYMPTOMS','GYNAECOLOGICAL_BLEEDING',
        'VAGINAL_DISCHARGE','BREAST_LUMP','URINARY_SYMPTOMS',
        'URINE_ABNORMALITIES','WEIGHT_LOSS','FATIGUE','SYSTEMIC'
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
    
    # Comorbidity features
    if not comorb_flags.empty:
        features = features.join(comorb_flags, how='left')
    if not comorb_total.empty:
        features = features.join(comorb_total, how='left')
    
    # Risk factor features
    if not rf_flags.empty:
        features = features.join(rf_flags, how='left')
    if not rf_total.empty:
        features = features.join(rf_total, how='left')
    
    # Investigation features
    for feat_s in [inv_count_A, inv_count_B, inv_total, inv_unique_types, 
                   gynae_inv_count, imaging_count_A, imaging_count_B, imaging_total]:
        features = features.join(feat_s, how='left')
    features = features.join(inv_accel[['INV_acceleration']], how='left')
    
    # Aggregate features
    for feat_s in [agg_unique_cats, agg_total_events, agg_events_A, agg_events_B,
                   agg_unique_snomed, symptom_count_A, symptom_count_B, symptom_total,
                   symptom_unique, symptom_new_B_count]:
        features = features.join(feat_s, how='left')
    features = features.join(symptom_merged[['AGG_symptom_acceleration']], how='left')
    
    # Temporal features
    features = features.join(temporal_df, how='left')
    features = features.join(last_event[['TEMP_days_last_event_to_index']], how='left')
    
    # Fill NaN with 0 for count/flag features, leave NaN for lab stats
    count_cols = [c for c in features.columns if any(x in c for x in ['_count_', '_has_ever', '_flag', '_acceleration', '_new_in_B', '_total', 'AGG_', 'INV_', 'COMORB_', 'RF_'])]
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
    # total_quantity, mean_quantity, total_duration,
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
    
    # Total duration per category
    if 'DURATION_IN_DAYS' in df.columns:
        med_dur = df.groupby(['PATIENT_GUID','CATEGORY'])['DURATION_IN_DAYS'].sum().unstack(fill_value=0)
        med_dur.columns = [f"MED_{c}_total_duration" for c in med_dur.columns]
    else:
        med_dur = pd.DataFrame()
    
    # Unique drugs per category
    med_unique = df.groupby(['PATIENT_GUID','CATEGORY'])['SNOMED_ID'].nunique().unstack(fill_value=0)
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
    med_agg_drugs = df.groupby('PATIENT_GUID')['SNOMED_ID'].nunique()
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
                    med_qty_sum, med_qty_mean, med_dur, med_unique]:
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
    Build clinical×medication interaction features.
    Runs on the merged patient-level feature matrix.
    """
    
    fm = feature_matrix.copy()
    
    # ══════════════════════════════════════════════════════════
    # 4k. INTERACTION FEATURES
    # ══════════════════════════════════════════════════════════
    
    interactions = {}
    
    # Pain + Opioid: abdominal pain AND opioid prescription
    if 'OBS_ABDOMINAL_PAIN_has_ever' in fm.columns and 'MED_OPIOID ANALGESICS_has_ever' in fm.columns:
        interactions['INT_pain_plus_opioid'] = fm['OBS_ABDOMINAL_PAIN_has_ever'] * fm['MED_OPIOID ANALGESICS_has_ever']
    
    # Bloating + Imaging: bloating AND ultrasound ordered
    if 'OBS_ABDOMINAL_BLOATING_has_ever' in fm.columns and 'OBS_IMAGING_has_ever' in fm.columns:
        interactions['INT_bloating_plus_imaging'] = fm['OBS_ABDOMINAL_BLOATING_has_ever'] * fm['OBS_IMAGING_has_ever']
    
    # Bleeding + Iron: gynae bleeding AND iron supplement
    if 'OBS_GYNAECOLOGICAL_BLEEDING_has_ever' in fm.columns and 'MED_IRON SUPPLEMENTS_has_ever' in fm.columns:
        interactions['INT_bleeding_plus_iron'] = fm['OBS_GYNAECOLOGICAL_BLEEDING_has_ever'] * fm['MED_IRON SUPPLEMENTS_has_ever']
    
    # Bleeding + Haemostatic: gynae bleeding AND tranexamic acid
    if 'OBS_GYNAECOLOGICAL_BLEEDING_has_ever' in fm.columns and 'MED_HAEMOSTATIC_has_ever' in fm.columns:
        interactions['INT_bleeding_plus_haemostatic'] = fm['OBS_GYNAECOLOGICAL_BLEEDING_has_ever'] * fm['MED_HAEMOSTATIC_has_ever']
    
    # GI symptoms + Laxatives: constipation/GI AND laxative prescription
    if 'OBS_GI_SYMPTOMS_has_ever' in fm.columns and 'MED_LAXATIVES_has_ever' in fm.columns:
        interactions['INT_gi_plus_laxatives'] = fm['OBS_GI_SYMPTOMS_has_ever'] * fm['MED_LAXATIVES_has_ever']
    
    # GI symptoms + Antiemetics
    if 'OBS_GI_SYMPTOMS_has_ever' in fm.columns and 'MED_ANTIEMETICS_has_ever' in fm.columns:
        interactions['INT_gi_plus_antiemetics'] = fm['OBS_GI_SYMPTOMS_has_ever'] * fm['MED_ANTIEMETICS_has_ever']
    
    # GI symptoms + PPI
    if 'OBS_GI_SYMPTOMS_has_ever' in fm.columns and 'MED_PPI_has_ever' in fm.columns:
        interactions['INT_gi_plus_ppi'] = fm['OBS_GI_SYMPTOMS_has_ever'] * fm['MED_PPI_has_ever']
    
    # Urinary symptoms + UTI antibiotics
    if 'OBS_URINARY_SYMPTOMS_has_ever' in fm.columns and 'MED_UTI ANTIBIOTICS_has_ever' in fm.columns:
        interactions['INT_urinary_plus_uti_abx'] = fm['OBS_URINARY_SYMPTOMS_has_ever'] * fm['MED_UTI ANTIBIOTICS_has_ever']
    
    # Urinary symptoms + Bladder antispasmodics
    if 'OBS_URINARY_SYMPTOMS_has_ever' in fm.columns and 'MED_BLADDER ANTISPASMODICS_has_ever' in fm.columns:
        interactions['INT_urinary_plus_bladder_antispasm'] = fm['OBS_URINARY_SYMPTOMS_has_ever'] * fm['MED_BLADDER ANTISPASMODICS_has_ever']
    
    # Vaginal discharge + Antifungals
    if 'OBS_VAGINAL_DISCHARGE_has_ever' in fm.columns and 'MED_ANTIFUNGALS_has_ever' in fm.columns:
        interactions['INT_discharge_plus_antifungal'] = fm['OBS_VAGINAL_DISCHARGE_has_ever'] * fm['MED_ANTIFUNGALS_has_ever']
    
    # Anaemia + Iron: anaemia comorbidity AND iron supplement
    if 'COMORB_ANAEMIA_flag' in fm.columns and 'MED_IRON SUPPLEMENTS_has_ever' in fm.columns:
        interactions['INT_anaemia_plus_iron'] = fm['COMORB_ANAEMIA_flag'] * fm['MED_IRON SUPPLEMENTS_has_ever']
    
    # DVT/PE + Anticoagulant
    if 'OBS_DVT_has_ever' in fm.columns and 'MED_ANTICOAGULANTS_has_ever' in fm.columns:
        interactions['INT_dvt_plus_anticoag'] = fm['OBS_DVT_has_ever'] * fm['MED_ANTICOAGULANTS_has_ever']
    
    # Weight loss + Fatigue (cancer constellation)
    if 'OBS_WEIGHT_LOSS_has_ever' in fm.columns and 'OBS_FATIGUE_has_ever' in fm.columns:
        interactions['INT_weightloss_plus_fatigue'] = fm['OBS_WEIGHT_LOSS_has_ever'] * fm['OBS_FATIGUE_has_ever']
    
    # Abdominal mass + Ascites (advanced cancer signal)
    if 'OBS_ABDOMINAL_MASS_has_ever' in fm.columns and 'OBS_ASCITES_has_ever' in fm.columns:
        interactions['INT_mass_plus_ascites'] = fm['OBS_ABDOMINAL_MASS_has_ever'] * fm['OBS_ASCITES_has_ever']
    
    # Bloating + Early satiety + Weight loss (classic ovarian triad)
    if all(c in fm.columns for c in ['OBS_ABDOMINAL_BLOATING_has_ever','OBS_EARLY_SATIETY_has_ever','OBS_WEIGHT_LOSS_has_ever']):
        interactions['INT_ovarian_triad'] = (
            fm['OBS_ABDOMINAL_BLOATING_has_ever'] * 
            fm['OBS_EARLY_SATIETY_has_ever'] * 
            fm['OBS_WEIGHT_LOSS_has_ever']
        )
    
    # HRT risk factor + Postmenopausal
    if 'RF_HRT_flag' in fm.columns and 'OBS_POSTMENOPAUSAL_has_ever' in fm.columns:
        interactions['INT_hrt_plus_postmenopausal'] = fm['RF_HRT_flag'] * fm['OBS_POSTMENOPAUSAL_has_ever']
    
    # Multi-symptom burden: 3+ different symptom categories
    if 'AGG_symptom_unique_categories' in fm.columns:
        interactions['INT_multi_symptom_burden'] = (fm['AGG_symptom_unique_categories'] >= 3).astype(int)
    
    # High investigation intensity + symptoms
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

# Load dropped-patients clinical + med CSVs per window
data = {}
for window, suffix in [('3mo', '3m'), ('6mo', '6m'), ('12mo', '12m')]:
    data[window] = {}
    try:
        data[window]['clinical'] = pd.read_csv(
            f"{BASE_PATH}/{window}/FE_ovarian_dropped_patients_clinical_windowed_{suffix}.csv",
            low_memory=False
        )
        data[window]['med'] = pd.read_csv(
            f"{BASE_PATH}/{window}/FE_ovarian_dropped_patients_med_windowed_{suffix}.csv",
            low_memory=False
        )
        print(f"Loaded {window}: clinical {data[window]['clinical'].shape[0]:,} rows, med {data[window]['med'].shape[0]:,} rows")
    except FileNotFoundError as e:
        print(f"⚠ {window}: {e}")
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
        print(f"  ❌ Missing data for {window}")
        continue
    
    # Build clinical features
    print(f"\n  Building clinical features...")
    clin_features = build_clinical_features(clin)
    print(f"  ✅ Clinical features: {clin_features.shape[1]} features for {clin_features.shape[0]} patients")
    
    # Build medication features
    print(f"\n  Building medication features...")
    med_features = build_medication_features(med)
    print(f"  ✅ Medication features: {med_features.shape[1]} features for {med_features.shape[0]} patients")
    
    # Merge clinical + medication on patient_guid
    print(f"\n  Merging clinical + medication features...")
    feature_matrix = clin_features.join(med_features, how='left')
    
    # Fill NaN medication features with 0 (patients with no meds)
    med_cols = [c for c in feature_matrix.columns if c.startswith('MED_')]
    feature_matrix[med_cols] = feature_matrix[med_cols].fillna(0)
    
    print(f"  ✅ Merged: {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} patients")
    
    # Build interaction features
    print(f"\n  Building interaction features...")
    feature_matrix = build_interaction_features(feature_matrix)
    print(f"  ✅ Final: {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} patients")
    
    # Summary
    print(f"\n  ── FEATURE SUMMARY ──")
    print(f"  Total features: {feature_matrix.shape[1]}")
    print(f"  Total patients: {feature_matrix.shape[0]}")
    print(f"  Positive (label=1): {(feature_matrix['LABEL']==1).sum()}")
    print(f"  Negative (label=0): {(feature_matrix['LABEL']==0).sum()}")
    
    # Feature type breakdown
    obs_cols = [c for c in feature_matrix.columns if c.startswith('OBS_')]
    lab_cols = [c for c in feature_matrix.columns if c.startswith('LAB_')]
    comorb_cols = [c for c in feature_matrix.columns if c.startswith('COMORB_')]
    rf_cols_list = [c for c in feature_matrix.columns if c.startswith('RF_')]
    inv_cols = [c for c in feature_matrix.columns if c.startswith('INV_')]
    agg_cols = [c for c in feature_matrix.columns if c.startswith('AGG_')]
    temp_cols = [c for c in feature_matrix.columns if c.startswith('TEMP_')]
    int_cols = [c for c in feature_matrix.columns if c.startswith('INT_')]
    
    print(f"\n  Feature breakdown:")
    print(f"    Demographics:     {2}")
    print(f"    Observation:      {len(obs_cols)}")
    print(f"    Lab values:       {len(lab_cols)}")
    print(f"    Comorbidity:      {len(comorb_cols)}")
    print(f"    Risk factors:     {len(rf_cols_list)}")
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
        print(f"\n  ⚠️ Features with >50% null:")
        for col, pct in high_null.items():
            print(f"    {col}: {pct:.1f}%")
    else:
        print(f"\n  ✅ No features with >50% null")
    
    # Store
    feature_matrices[window] = feature_matrix
    
    # Save
    out_path = FE_RESULTS / window / f"feature_matrix_{window}.csv"
    feature_matrix.to_csv(out_path)
    print(f"\n  ✅ Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═'*70}")
print(f"  STEP 4 COMPLETE — FEATURE MATRICES BUILT")
print(f"{'═'*70}")

for window in ['3mo', '6mo', '12mo']:
    fm = feature_matrices.get(window)
    if fm is not None:
        print(f"\n  {window}: {fm.shape[0]} patients × {fm.shape[1]} features")
        print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")

print(f"\n{'═'*70}")
print(f"  ✅ Ready for STEP 5 (FE Cleanup & Feature Selection)")
print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — ADVANCED FEATURE ENGINEERING
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
    
    print(f"\n{'═'*70}")
    print(f"  ADVANCED FEATURES — {window_name}")
    print(f"{'═'*70}")
    
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
    if 'DURATION_IN_DAYS' in med.columns:
        med['DURATION_IN_DAYS'] = pd.to_numeric(med['DURATION_IN_DAYS'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in med.columns:
        med['MONTHS_BEFORE_INDEX'] = pd.to_numeric(med['MONTHS_BEFORE_INDEX'], errors='coerce')
    
    patient_list = existing_fm.index.tolist()
    adv = pd.DataFrame(index=patient_list)
    adv.index.name = 'PATIENT_GUID'
    
    obs_df = clin[clin['EVENT_TYPE'] == 'OBSERVATION'].copy()
    lab_df = clin[clin['EVENT_TYPE'] == 'LAB VALUE'].copy()
    
    # ══════════════════════════════════════════════════════════
    # GROUP 1: SYMPTOM CLUSTERING — combinations that signal cancer
    # ══════════════════════════════════════════════════════════
    print(f"  Building symptom clusters...")
    
    gi_cats = ['ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'ABDOMINAL_BLOATING', 'EARLY_SATIETY']
    gynae_cats = ['GYNAECOLOGICAL_BLEEDING', 'VAGINAL_DISCHARGE', 'OVARIAN_CYST', 'GYNAECOLOGICAL_DX']
    urinary_cats = ['URINARY_SYMPTOMS', 'URINE_ABNORMALITIES']
    constitutional_cats = ['WEIGHT_LOSS', 'FATIGUE']
    mass_cats = ['ABDOMINAL_MASS', 'ASCITES']
    
    for group_name, cats in [
        ('GI', gi_cats), ('GYNAE', gynae_cats), ('URINARY', urinary_cats),
        ('CONSTITUTIONAL', constitutional_cats), ('MASS', mass_cats)
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
    
    adv['CLUSTER_gi_AND_gynae'] = (adv['CLUSTER_GI_any'] & adv['CLUSTER_GYNAE_any']).astype(int)
    adv['CLUSTER_gi_AND_urinary'] = (adv['CLUSTER_GI_any'] & adv['CLUSTER_URINARY_any']).astype(int)
    adv['CLUSTER_gynae_AND_urinary'] = (adv['CLUSTER_GYNAE_any'] & adv['CLUSTER_URINARY_any']).astype(int)
    adv['CLUSTER_gi_AND_gynae_AND_urinary'] = (adv['CLUSTER_GI_any'] & adv['CLUSTER_GYNAE_any'] & adv['CLUSTER_URINARY_any']).astype(int)
    adv['CLUSTER_constitutional_AND_gi'] = (adv['CLUSTER_CONSTITUTIONAL_any'] & adv['CLUSTER_GI_any']).astype(int)
    adv['CLUSTER_mass_OR_ascites_any'] = adv['CLUSTER_MASS_any']
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
        pain_obs = obs_windowed[obs_windowed['CATEGORY'] == 'ABDOMINAL_PAIN']
        pain_quarters = pain_obs.groupby(['PATIENT_GUID', 'QUARTER']).size().unstack(fill_value=0)
        for col in pain_quarters.columns:
            adv[f'TRAJ_pain_{col}'] = adv.index.map(pain_quarters[col]).fillna(0).astype(int)
        gi_obs = obs_windowed[obs_windowed['CATEGORY'].isin(gi_cats)]
        gi_quarters = gi_obs.groupby(['PATIENT_GUID', 'QUARTER']).size().unstack(fill_value=0)
        for col in gi_quarters.columns:
            adv[f'TRAJ_gi_{col}'] = adv.index.map(gi_quarters[col]).fillna(0).astype(int)
    
    # ══════════════════════════════════════════════════════════
    # GROUP 4: LAB TRAJECTORIES
    # ══════════════════════════════════════════════════════════
    print(f"  Building lab trajectories...")
    
    lab_windowed = lab_df[lab_df['VALUE'].notna()].copy()
    
    for lab_cat in ['HAEMATOLOGY', 'RENAL', 'INFLAMMATORY', 'METABOLIC', 'LIVER']:
        cat_labs = lab_windowed[lab_windowed['CATEGORY'] == lab_cat].copy()
        if len(cat_labs) == 0:
            continue
        cat_labs = cat_labs.sort_values(['PATIENT_GUID', 'EVENT_DATE'])
        
        def calc_slope(group):
            if len(group) < 2:
                return np.nan
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values
            y = group['VALUE'].values
            if x[-1] == 0:
                return np.nan
            try:
                return np.polyfit(x, y, 1)[0]
            except Exception:
                return np.nan
        
        slopes = cat_labs.groupby('PATIENT_GUID').apply(calc_slope)
        slopes.name = f'LAB_TRAJ_{lab_cat}_slope'
        adv[slopes.name] = adv.index.map(slopes).astype(float)
        adv[f'LAB_TRAJ_{lab_cat}_declining'] = (adv[slopes.name] < 0).astype(int)
        val_range = cat_labs.groupby('PATIENT_GUID')['VALUE'].apply(lambda x: x.max() - x.min())
        val_range.name = f'LAB_TRAJ_{lab_cat}_range'
        adv[val_range.name] = adv.index.map(val_range).fillna(0).astype(float)
        cv = cat_labs.groupby('PATIENT_GUID')['VALUE'].apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0)
        cv.name = f'LAB_TRAJ_{lab_cat}_cv'
        adv[cv.name] = adv.index.map(cv).fillna(0).astype(float)
        first_last = cat_labs.groupby('PATIENT_GUID').apply(
            lambda x: x.iloc[-1]['VALUE'] - x.iloc[0]['VALUE'] if len(x) >= 2 else 0
        )
        first_last.name = f'LAB_TRAJ_{lab_cat}_first_last_diff'
        adv[first_last.name] = adv.index.map(first_last).fillna(0).astype(float)
    
    # ══════════════════════════════════════════════════════════
    # GROUP 5: MEDICATION ESCALATION PATTERNS
    # ══════════════════════════════════════════════════════════
    print(f"  Building medication escalation patterns...")
    
    med_windowed = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    pain_meds = ['OPIOID ANALGESICS', 'NSAIDS', 'NEUROPATHIC PAIN']
    pain_med_df = med_windowed[med_windowed['CATEGORY'].isin(pain_meds)]
    pain_med_cats = pain_med_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_pain_category_count'] = adv.index.map(pain_med_cats).fillna(0).astype(int)
    pain_A = pain_med_df[pain_med_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size()
    pain_B = pain_med_df[pain_med_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    adv['MED_ESC_pain_count_A'] = adv.index.map(pain_A).fillna(0).astype(int)
    adv['MED_ESC_pain_count_B'] = adv.index.map(pain_B).fillna(0).astype(int)
    adv['MED_ESC_pain_acceleration'] = (adv['MED_ESC_pain_count_B'] > adv['MED_ESC_pain_count_A']).astype(int)
    opioid_df = med_windowed[med_windowed['CATEGORY'] == 'OPIOID ANALGESICS']
    has_opioid = opioid_df.groupby('PATIENT_GUID').size() > 0
    adv['MED_ESC_has_opioid'] = adv.index.map(has_opioid).fillna(False).astype(int)
    opioid_A = opioid_df[opioid_df['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID').size() > 0
    opioid_B = opioid_df[opioid_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size() > 0
    adv['MED_ESC_new_opioid_in_B'] = (
        adv.index.map(opioid_B).fillna(False) & ~adv.index.map(opioid_A).fillna(False)
    ).astype(int)
    gi_meds = ['PPI', 'GI ANTISPASMODICS', 'ANTIEMETICS', 'LAXATIVES']
    gi_med_df = med_windowed[med_windowed['CATEGORY'].isin(gi_meds)]
    gi_med_cats = gi_med_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_gi_category_count'] = adv.index.map(gi_med_cats).fillna(0).astype(int)
    gi_med_total = gi_med_df.groupby('PATIENT_GUID').size()
    adv['MED_ESC_gi_total'] = adv.index.map(gi_med_total).fillna(0).astype(int)
    gi_med_B = gi_med_df[gi_med_df['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID').size()
    adv['MED_ESC_gi_count_B'] = adv.index.map(gi_med_B).fillna(0).astype(int)
    abx_cats = ['UTI ANTIBIOTICS', 'GENERAL ANTIBIOTICS']
    abx_df = med_windowed[med_windowed['CATEGORY'].isin(abx_cats)]
    abx_count = abx_df.groupby('PATIENT_GUID').size()
    adv['MED_ESC_abx_repeat_count'] = adv.index.map(abx_count).fillna(0).astype(int)
    adv['MED_ESC_abx_repeat_3plus'] = (adv['MED_ESC_abx_repeat_count'] >= 3).astype(int)
    med_cats_A = med_windowed[med_windowed['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    med_cats_B = med_windowed[med_windowed['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_polypharmacy_A'] = adv.index.map(med_cats_A).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_B'] = adv.index.map(med_cats_B).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_increase'] = (adv['MED_ESC_polypharmacy_B'] > adv['MED_ESC_polypharmacy_A']).astype(int)
    total_meds = med_windowed.groupby('PATIENT_GUID').size()
    adv['MED_ESC_total_prescriptions'] = adv.index.map(total_meds).fillna(0).astype(int)
    
    # ══════════════════════════════════════════════════════════
    # GROUP 6: INVESTIGATION PATTERNS
    # ══════════════════════════════════════════════════════════
    print(f"  Building investigation patterns...")
    
    inv_cats = ['IMAGING', 'GYNAE_PROCEDURES', 'OTHER_PROCEDURES', 'MICROBIOLOGY', 'SCREENING']
    inv_df = obs_df[obs_df['CATEGORY'].isin(inv_cats)]
    symptom_cats_all = gi_cats + gynae_cats + urinary_cats + constitutional_cats + mass_cats
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
    gynae_proc = obs_df[obs_df['CATEGORY'] == 'GYNAE_PROCEDURES']
    has_gynae_proc = gynae_proc.groupby('PATIENT_GUID').size() > 0
    adv['INV_PATTERN_has_gynae_procedure'] = adv.index.map(has_gynae_proc).fillna(False).astype(int)
    imaging_df = obs_df[obs_df['CATEGORY'] == 'IMAGING']
    has_imaging = imaging_df.groupby('PATIENT_GUID').size() > 0
    adv['INV_PATTERN_has_imaging'] = adv.index.map(has_imaging).fillna(False).astype(int)
    
    # ══════════════════════════════════════════════════════════
    # GROUP 7: CROSS-DOMAIN INTERACTIONS
    # ══════════════════════════════════════════════════════════
    print(f"  Building cross-domain interactions...")
    
    has_declining_hb = (adv.get('LAB_TRAJ_HAEMATOLOGY_declining', pd.Series(0, index=adv.index)) == 1)
    if not isinstance(has_declining_hb, pd.Series):
        has_declining_hb = pd.Series(False, index=adv.index)
    has_pain = (adv.get('CLUSTER_GI_any', pd.Series(0, index=adv.index)) == 1)
    has_pain_med = (adv.get('MED_ESC_has_opioid', pd.Series(0, index=adv.index)) == 1)
    adv['CROSS_pain_opioid_declining_hb'] = (has_pain & has_pain_med & has_declining_hb).astype(int)
    has_gi_sym = (adv.get('CLUSTER_GI_any', pd.Series(0, index=adv.index)) == 1)
    has_gi_med = (adv.get('MED_ESC_gi_total', pd.Series(0, index=adv.index)) > 0)
    no_inv = (adv.get('INV_PATTERN_inv_count', pd.Series(0, index=adv.index)) == 0)
    adv['CROSS_gi_treated_no_investigation'] = (has_gi_sym & has_gi_med & no_inv).astype(int)
    high_visits = (adv.get('VISIT_unique_dates', pd.Series(0, index=adv.index)) >= 5)
    multi_system = (adv.get('CLUSTER_multi_system_count', pd.Series(0, index=adv.index)) >= 2)
    adv['CROSS_diagnostic_odyssey'] = (high_visits & multi_system).astype(int)
    has_gynae_sym = (adv.get('CLUSTER_GYNAE_any', pd.Series(0, index=adv.index)) == 1)
    has_gynae_inv = (adv.get('INV_PATTERN_has_gynae_procedure', pd.Series(0, index=adv.index)) == 1)
    adv['CROSS_gynae_symptom_investigated'] = (has_gynae_sym & has_gynae_inv).astype(int)
    has_urinary = (adv.get('CLUSTER_URINARY_any', pd.Series(0, index=adv.index)) == 1)
    repeat_abx = (adv.get('MED_ESC_abx_repeat_count', pd.Series(0, index=adv.index)) >= 2)
    adv['CROSS_urinary_repeat_abx'] = (has_urinary & repeat_abx).astype(int)
    has_iron = pd.Series(False, index=adv.index)
    if 'MED_IRON SUPPLEMENTS_has_ever' in existing_fm.columns:
        has_iron = existing_fm['MED_IRON SUPPLEMENTS_has_ever'].reindex(adv.index).fillna(0) == 1
    has_fatigue = (adv.get('CLUSTER_CONSTITUTIONAL_any', pd.Series(0, index=adv.index)) == 1)
    adv['CROSS_anaemia_iron_fatigue'] = (has_iron & has_fatigue & has_declining_hb).astype(int)

    # ══════════════════════════════════════════════════════════
    # GROUP 8: TIME-DECAY WEIGHTED FEATURES
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
        for cat in ['ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'GYNAECOLOGICAL_BLEEDING', 'URINARY_SYMPTOMS']:
            cat_weighted = obs_windowed_copy[obs_windowed_copy['CATEGORY'] == cat]
            cat_ws = cat_weighted.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
            cat_ws.name = f'DECAY_{cat}_weighted'
            adv[cat_ws.name] = adv.index.map(cat_ws).fillna(0).astype(float)
    
    # Smart fill: median for lab trajectories, 0 for counts/flags
    lab_traj_cols = [c for c in adv.columns if c.startswith('LAB_TRAJ_') and any(x in c for x in ['_slope', '_range', '_cv', '_first_last_diff'])]
    for col in lab_traj_cols:
        med_val = adv[col].median()
        adv[col] = adv[col].fillna(med_val if pd.notna(med_val) else 0)
    adv = adv.fillna(0)
    adv = adv.replace([np.inf, -np.inf], 0)
    
    print(f"\n  ✅ Advanced features built: {adv.shape[1]} new features")
    print(f"    CLUSTER: {len([c for c in adv.columns if c.startswith('CLUSTER_')])} | "
          f"VISIT: {len([c for c in adv.columns if c.startswith('VISIT_')])} | "
          f"TRAJ: {len([c for c in adv.columns if c.startswith('TRAJ_')])} | "
          f"LAB_TRAJ: {len([c for c in adv.columns if c.startswith('LAB_TRAJ_')])} | "
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
    existing_fm = feature_matrices[window]
    
    adv_features = build_advanced_features(clin, med, existing_fm, window)
    enhanced = existing_fm.join(adv_features, how='left')
    enhanced = enhanced.fillna(0)
    enhanced = enhanced.replace([np.inf, -np.inf], 0)
    enhanced = enhanced.loc[:, ~enhanced.columns.duplicated()]
    
    print(f"\n  Final enhanced matrix: {enhanced.shape[0]} patients × {enhanced.shape[1]} features")
    print(f"  (was {existing_fm.shape[1]} features, added {enhanced.shape[1] - existing_fm.shape[1]})")
    
    enhanced_matrices[window] = enhanced
    out_path = FE_RESULTS / window / f"feature_matrix_enhanced_{window}.csv"
    enhanced.to_csv(out_path)
    print(f"  ✅ Saved: {out_path}")


print(f"\n{'═'*70}")
print(f"  STEP 4b COMPLETE — ENHANCED FEATURE MATRICES")
print(f"{'═'*70}")

for window in ['3mo', '6mo', '12mo']:
    fm = enhanced_matrices[window]
    print(f"\n  {window}: {fm.shape[0]} patients × {fm.shape[1]} features")
    print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")

print(f"\n{'═'*70}")
msg = "  ✅ Run Step 5 cleanup on feature_matrix_<window>.csv or feature_matrix_enhanced_*.csv, then modeling."
print(msg)
print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — MAXIMUM FEATURE EXTRACTION
# Step 4c: Extract EVERY possible signal
# Run AFTER Step 4b (builds on enhanced matrices)
# Uses in-memory data + enhanced_matrices from above
# ═══════════════════════════════════════════════════════════════

def extract_maximum_features(clin_df, med_df, existing_fm, window_name):
    """Extract every possible feature from the data."""
    
    print(f"\n{'═'*70}")
    print(f"  MAXIMUM FEATURE EXTRACTION — {window_name}")
    print(f"{'═'*70}")
    
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
    if 'DURATION_IN_DAYS' in med.columns:
        med['DURATION_IN_DAYS'] = pd.to_numeric(med['DURATION_IN_DAYS'], errors='coerce')
    
    patient_list = existing_fm.index.tolist()
    mega = pd.DataFrame(index=patient_list)
    mega.index.name = 'PATIENT_GUID'
    
    obs_df = clin[clin['EVENT_TYPE'] == 'OBSERVATION'].copy()
    lab_df = clin[clin['EVENT_TYPE'] == 'LAB VALUE'].copy()
    comorb_df = clin[clin['EVENT_TYPE'] == 'COMORBIDITY'].copy()
    rf_df = clin[clin['EVENT_TYPE'] == 'RISK FACTOR'].copy()
    obs_AB = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    lab_AB = lab_df[lab_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    
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
        if 'DURATION_IN_DAYS' in cat_df.columns:
            total_dur = cat_df.groupby('PATIENT_GUID')['DURATION_IN_DAYS'].sum()
            mega[f'{prefix}_total_duration'] = mega.index.map(total_dur).fillna(0).astype(float)
    fc2 = len([c for c in mega.columns if c.startswith('MEDCAT_')])
    feature_count += fc2
    print(f"    Added {fc2} features for {len(all_med_cats)} med categories")
    
    # BLOCK 4: LAB VALUE DEEP FEATURES
    print(f"\n  BLOCK 4: Per-term lab features...")
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
    print(f"    Added {fc3} features for {len(top_lab_terms)} lab terms")
    
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
    key_clinical_cats = ['ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'GYNAECOLOGICAL_BLEEDING', 'URINARY_SYMPTOMS',
                         'ABDOMINAL_BLOATING', 'WEIGHT_LOSS', 'FATIGUE']
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
    symptom_cats = ['ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'GYNAECOLOGICAL_BLEEDING', 'URINARY_SYMPTOMS',
                    'ABDOMINAL_BLOATING', 'WEIGHT_LOSS', 'FATIGUE', 'VAGINAL_DISCHARGE', 'OVARIAN_CYST']
    inv_cats = ['IMAGING', 'GYNAE_PROCEDURES', 'OTHER_PROCEDURES', 'MICROBIOLOGY']
    med_pain_cats = ['OPIOID ANALGESICS', 'NSAIDS', 'NEUROPATHIC PAIN']
    med_gi_cats = ['PPI', 'GI ANTISPASMODICS', 'ANTIEMETICS', 'LAXATIVES']
    sym_first = obs_AB[obs_AB['CATEGORY'].isin(symptom_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    inv_first = obs_AB[obs_AB['CATEGORY'].isin(inv_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    pain_med_first = med_AB[med_AB['CATEGORY'].isin(med_pain_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gi_med_first = med_AB[med_AB['CATEGORY'].isin(med_gi_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    lab_first = lab_AB.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap_sym_inv = (inv_first - sym_first).dt.days
    mega['SEQ_symptom_to_inv_days'] = mega.index.map(gap_sym_inv).fillna(-999).astype(float)
    mega['SEQ_inv_before_symptom'] = (mega['SEQ_symptom_to_inv_days'] < 0).astype(int)
    mega['SEQ_inv_within_30d'] = ((mega['SEQ_symptom_to_inv_days'] >= 0) & (mega['SEQ_symptom_to_inv_days'] <= 30)).astype(int)
    mega['SEQ_inv_within_90d'] = ((mega['SEQ_symptom_to_inv_days'] >= 0) & (mega['SEQ_symptom_to_inv_days'] <= 90)).astype(int)
    mega['SEQ_inv_delayed_90d'] = (mega['SEQ_symptom_to_inv_days'] > 90).astype(int)
    gap_sym_painmed = (pain_med_first - sym_first).dt.days
    mega['SEQ_symptom_to_painmed_days'] = mega.index.map(gap_sym_painmed).fillna(-999).astype(float)
    mega['SEQ_painmed_before_symptom'] = (mega['SEQ_symptom_to_painmed_days'] < 0).astype(int)
    gap_sym_gimed = (gi_med_first - sym_first).dt.days
    mega['SEQ_symptom_to_gimed_days'] = mega.index.map(gap_sym_gimed).fillna(-999).astype(float)
    gap_sym_lab = (lab_first - sym_first).dt.days
    mega['SEQ_symptom_to_lab_days'] = mega.index.map(gap_sym_lab).fillna(-999).astype(float)
    for cat in symptom_cats:
        cat_first = obs_AB[obs_AB['CATEGORY'] == cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max()
        mega[f'SEQ_{cat}_first_months_before'] = mega.index.map(cat_first).fillna(-1).astype(float)
        cat_last = obs_AB[obs_AB['CATEGORY'] == cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
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
        unique_snomed=('SNOMED_ID', 'nunique')
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
    lab_count = lab_AB.groupby('PATIENT_GUID').size()
    total_clinical = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].groupby('PATIENT_GUID').size()
    lab_proportion = lab_count / total_clinical
    mega['RATE_lab_proportion'] = mega.index.map(lab_proportion).fillna(0).astype(float)
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
        for cat in ['ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'GYNAECOLOGICAL_BLEEDING', 'URINARY_SYMPTOMS',
                    'WEIGHT_LOSS', 'FATIGUE', 'ABDOMINAL_BLOATING']:
            has_cat = obs_AB[obs_AB['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0
            has_cat_mapped = mega.index.map(has_cat).fillna(False).astype(int)
            mega[f'AGEX_{cat}'] = age * has_cat_mapped
        if 'CLUSTER_multi_system_count' in existing_fm.columns:
            msc = existing_fm['CLUSTER_multi_system_count'].reindex(mega.index).fillna(0)
            mega['AGEX_multi_system'] = age * msc
        postmeno = obs_AB[obs_AB['CATEGORY'] == 'POSTMENOPAUSAL'].groupby('PATIENT_GUID').size() > 0
        mega['AGEX_postmenopausal'] = age * mega.index.map(postmeno).fillna(False).astype(int)
    fc7 = len([c for c in mega.columns if c.startswith('AGE_') or c.startswith('AGEX_')])
    feature_count += fc7
    print(f"    Added {fc7} age features")
    
    # BLOCK 10: COMORBIDITY INTERACTION FEATURES
    print(f"\n  BLOCK 10: Comorbidity interactions...")
    comorb_cats = comorb_df['CATEGORY'].unique()
    patient_comorbs = comorb_df.groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    comorb_count = comorb_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    mega['COMORB_burden'] = mega.index.map(comorb_count).fillna(0).astype(int)
    mega['COMORB_high_burden'] = (mega['COMORB_burden'] >= 3).astype(int)
    for comorb in comorb_cats:
        has_comorb = patient_comorbs.apply(lambda s: comorb in s if isinstance(s, set) else False)
        has_comorb_mapped = mega.index.map(has_comorb).fillna(False).astype(int)
        for sym in ['ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'WEIGHT_LOSS', 'FATIGUE']:
            has_sym = obs_AB[obs_AB['CATEGORY'] == sym].groupby('PATIENT_GUID').size() > 0
            has_sym_mapped = mega.index.map(has_sym).fillna(False).astype(int)
            mega[f'COMORBX_{comorb[:12]}_X_{sym[:12]}'] = has_comorb_mapped * has_sym_mapped
    fc8 = len([c for c in mega.columns if c.startswith('COMORB_') or c.startswith('COMORBX_')])
    feature_count += fc8
    print(f"    Added {fc8} comorbidity interaction features")
    
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
    snomed_counts = obs_AB.groupby('PATIENT_GUID')['SNOMED_ID'].apply(lambda x: Counter(x.dropna()) if len(x) else Counter())
    snomed_entropy = snomed_counts.apply(calc_entropy)
    mega['ENTROPY_snomed'] = mega.index.map(snomed_entropy).fillna(0).astype(float)
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
    existing_fm = enhanced_matrices[window]
    mega_features = extract_maximum_features(clin, med, existing_fm, window)
    final = existing_fm.join(mega_features, how='left', rsuffix='_mega')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]
    mega_matrices[window] = final
    print(f"\n  FINAL MATRIX: {final.shape[0]} patients × {final.shape[1]} features")
    print(f"  (Enhanced had {existing_fm.shape[1]}, added {final.shape[1] - existing_fm.shape[1]})")
    out_path = FE_RESULTS / window / f"feature_matrix_mega_{window}.csv"
    final.to_csv(out_path)
    print(f"  ✅ Saved: {out_path}")

print(f"\n{'═'*70}")
print(f"  STEP 4c COMPLETE — MEGA FEATURE MATRICES")
print(f"{'═'*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = mega_matrices[window]
    print(f"\n  {window}: {fm.shape[0]} patients × {fm.shape[1]} features")
    print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
print(f"\n{'═'*70}")
print("  ✅ MEGA FEATURES COMPLETE — Next: Step 5 cleanup → modeling on mega matrices")
print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — OVARIAN-SPECIFIC FEATURES (NO COMPOSITE SCORES)
# Step 4d: Option A — Keep individual features, remove hand-built scores
# Run AFTER Step 4c (builds on mega_matrices)
# Uses in-memory data + mega_matrices from above
# ═══════════════════════════════════════════════════════════════

def build_ovarian_features(clin_df, med_df, existing_fm, window_name):
    """Ovarian-specific features: NICE cardinals, mimics, cyst, labs, pathway, treatment. No composite scores."""
    print(f"\n{'═'*70}")
    print(f"  OVARIAN-SPECIFIC (Option A) — {window_name}")
    print(f"{'═'*70}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    for col in ['VALUE', 'MONTHS_BEFORE_INDEX', 'AGE_AT_INDEX']:
        if col in clin.columns:
            clin[col] = pd.to_numeric(clin[col], errors='coerce')
        if col in med.columns:
            med[col] = pd.to_numeric(med[col], errors='coerce')
    if 'DURATION_IN_DAYS' in med.columns:
        med['DURATION_IN_DAYS'] = pd.to_numeric(med['DURATION_IN_DAYS'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    ov = pd.DataFrame(index=patient_list)
    ov.index.name = 'PATIENT_GUID'

    obs = clin[clin['EVENT_TYPE'] == 'OBSERVATION'].copy()
    lab = clin[clin['EVENT_TYPE'] == 'LAB VALUE'].copy()
    rf = clin[clin['EVENT_TYPE'] == 'RISK FACTOR'].copy()
    obs_AB = obs[obs['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    lab_AB = lab[lab['TIME_WINDOW'].isin(['A', 'B'])].copy()
    lab_vals = lab_AB[lab_AB['VALUE'].notna()].copy()

    def has_cat(df, cat):
        return df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0

    def count_cat(df, cat):
        return df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size()

    def m(series, default=0):
        return ov.index.map(series).fillna(default)

    age = existing_fm['AGE_AT_INDEX'].reindex(ov.index).fillna(50) if 'AGE_AT_INDEX' in existing_fm.columns else pd.Series(50, index=ov.index)

    # BLOCK 1: NICE CARDINAL SYMPTOMS
    print(f"  BLOCK 1: NICE cardinal symptoms...")
    has_bloating = m(has_cat(obs_AB, 'ABDOMINAL_BLOATING'), False).astype(int)
    has_satiety = m(has_cat(obs_AB, 'EARLY_SATIETY'), False).astype(int)
    has_pain = m(has_cat(obs_AB, 'ABDOMINAL_PAIN'), False).astype(int)
    has_urinary = m(has_cat(obs_AB, 'URINARY_SYMPTOMS'), False).astype(int)
    ov['NICE_bloating'] = has_bloating
    ov['NICE_early_satiety'] = has_satiety
    ov['NICE_pelvic_pain'] = has_pain
    ov['NICE_urinary_frequency'] = has_urinary
    ov['NICE_cardinal_count'] = has_bloating + has_satiety + has_pain + has_urinary
    ov['NICE_2plus_cardinals'] = (ov['NICE_cardinal_count'] >= 2).astype(int)
    ov['NICE_3plus_cardinals'] = (ov['NICE_cardinal_count'] >= 3).astype(int)
    bloat_B = count_cat(obs_AB[obs_AB['TIME_WINDOW'] == 'B'], 'ABDOMINAL_BLOATING')
    pain_B = count_cat(obs_AB[obs_AB['TIME_WINDOW'] == 'B'], 'ABDOMINAL_PAIN')
    urin_B = count_cat(obs_AB[obs_AB['TIME_WINDOW'] == 'B'], 'URINARY_SYMPTOMS')
    ov['NICE_persistent_bloating'] = (m(bloat_B) >= 2).astype(int)
    ov['NICE_persistent_pain'] = (m(pain_B) >= 2).astype(int)
    ov['NICE_persistent_urinary'] = (m(urin_B) >= 2).astype(int)
    ov['NICE_persistent_count'] = ov['NICE_persistent_bloating'] + ov['NICE_persistent_pain'] + ov['NICE_persistent_urinary']
    ov['NICE_over50_with_cardinal'] = ((age >= 50) & (ov['NICE_cardinal_count'] >= 1)).astype(int)
    ov['NICE_over50_with_2plus'] = ((age >= 50) & (ov['NICE_cardinal_count'] >= 2)).astype(int)

    # BLOCK 2: MIMIC PATTERNS
    print(f"  BLOCK 2: Mimic patterns...")
    has_gi = m(has_cat(obs_AB, 'GI_SYMPTOMS'), False).astype(int)
    has_inv = m(obs_AB[obs_AB['CATEGORY'].isin(['IMAGING', 'GYNAE_PROCEDURES', 'OTHER_PROCEDURES'])].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    ov['OV_ibs_mimic'] = ((has_gi == 1) & (has_bloating == 1) & (has_pain == 1)).astype(int)
    ov['OV_ibs_mimic_no_inv'] = ((ov['OV_ibs_mimic'] == 1) & (has_inv == 0)).astype(int)
    uti_abx_count = m(med_AB[med_AB['CATEGORY'] == 'UTI ANTIBIOTICS'].groupby('PATIENT_GUID').size())
    ov['OV_uti_mimic'] = ((has_urinary == 1) & (uti_abx_count >= 1)).astype(int)
    ov['OV_uti_mimic_recurrent'] = ((has_urinary == 1) & (uti_abx_count >= 2)).astype(int)
    has_bleed = m(has_cat(obs_AB, 'GYNAECOLOGICAL_BLEEDING'), False).astype(int)
    ov['OV_postmeno_bleeding'] = ((has_bleed == 1) & (age >= 55)).astype(int)
    ov['OV_postmeno_bleeding_with_pain'] = ((ov['OV_postmeno_bleeding'] == 1) & (has_pain == 1)).astype(int)
    ov['OV_classic_triad'] = ((has_bloating == 1) & (has_pain == 1) & (has_urinary == 1)).astype(int)
    ov['OV_classic_triad_over50'] = ((ov['OV_classic_triad'] == 1) & (age >= 50)).astype(int)

    # BLOCK 3: OVARIAN CYST INTERACTIONS
    print(f"  BLOCK 3: Ovarian cyst interactions...")
    has_cyst = m(has_cat(obs_AB, 'OVARIAN_CYST'), False).astype(int)
    has_ascites = m(has_cat(obs_AB, 'ASCITES'), False).astype(int)
    has_mass = m(has_cat(obs_AB, 'ABDOMINAL_MASS'), False).astype(int)
    has_wt_loss = m(has_cat(obs_AB, 'WEIGHT_LOSS'), False).astype(int)
    has_fatigue = m(has_cat(obs_AB, 'FATIGUE'), False).astype(int)
    ov['OV_cyst_present'] = has_cyst
    ov['OV_cyst_with_pain'] = ((has_cyst == 1) & (has_pain == 1)).astype(int)
    ov['OV_cyst_with_bloating'] = ((has_cyst == 1) & (has_bloating == 1)).astype(int)
    ov['OV_cyst_with_gi'] = ((has_cyst == 1) & (has_gi == 1)).astype(int)
    ov['OV_cyst_with_weightloss'] = ((has_cyst == 1) & (has_wt_loss == 1)).astype(int)
    ov['OV_cyst_complex'] = ((has_cyst == 1) & (ov['NICE_cardinal_count'] >= 2)).astype(int)
    ov['OV_ascites'] = has_ascites
    ov['OV_mass'] = has_mass
    ov['OV_ascites_or_mass'] = ((has_ascites == 1) | (has_mass == 1)).astype(int)
    ov['OV_ascites_with_bloating'] = ((has_ascites == 1) & (has_bloating == 1)).astype(int)

    # BLOCK 4: RISK FACTORS
    print(f"  BLOCK 4: Risk factors...")
    ov['OV_RF_peak_age'] = ((age >= 55) & (age <= 75)).astype(int)
    ov['OV_RF_high_risk_age'] = (age >= 60).astype(int)
    has_fh = m(rf[rf['CATEGORY'] == 'FAMILY_HISTORY'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    ov['OV_RF_family_history'] = has_fh
    ov['OV_RF_fh_plus_symptoms'] = ((has_fh == 1) & (ov['NICE_cardinal_count'] >= 1)).astype(int)
    has_hrt = m(
        (rf[rf['CATEGORY'] == 'HRT'].groupby('PATIENT_GUID').size() > 0) |
        (med_AB[med_AB['CATEGORY'] == 'HRT'].groupby('PATIENT_GUID').size() > 0),
        False
    ).astype(int)
    ov['OV_RF_hrt_use'] = has_hrt
    has_ocp = m(med_AB[med_AB['CATEGORY'] == 'ORAL CONTRACEPTIVE'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    ov['OV_RF_ocp_protective'] = has_ocp

    # BLOCK 5: LAB THRESHOLD FEATURES
    print(f"  BLOCK 5: Lab threshold features...")
    hb = lab_vals[lab_vals['TERM'].str.contains('Haemoglobin', case=False, na=False)]
    if len(hb) > 0:
        hb_last = hb.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        hb_first = hb.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first()
        hb_decline = hb_first - hb_last
        ov['OV_LAB_hb_anaemic'] = (m(hb_last, 999) < 120).astype(int)
        ov['OV_LAB_hb_severe_anaemia'] = (m(hb_last, 999) < 100).astype(int)
        ov['OV_LAB_hb_declining'] = (m(hb_decline) > 5).astype(int)
        ov['OV_LAB_hb_anaemic_with_pain'] = ((ov['OV_LAB_hb_anaemic'] == 1) & (has_pain == 1)).astype(int)
        ov['OV_LAB_hb_anaemic_with_fatigue'] = ((ov['OV_LAB_hb_anaemic'] == 1) & (has_fatigue == 1)).astype(int)
    crp = lab_vals[lab_vals['TERM'].str.contains('CRP|C reactive', case=False, na=False)]
    if len(crp) > 0:
        crp_last = crp.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        crp_max = crp.groupby('PATIENT_GUID')['VALUE'].max()
        ov['OV_LAB_crp_elevated'] = (m(crp_last) > 10).astype(int)
        ov['OV_LAB_crp_high'] = (m(crp_max) > 30).astype(int)
        ov['OV_LAB_crp_with_pain'] = ((ov['OV_LAB_crp_elevated'] == 1) & (has_pain == 1)).astype(int)
    alb = lab_vals[lab_vals['TERM'].str.contains('albumin', case=False, na=False)]
    if len(alb) > 0:
        alb_last = alb.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        ov['OV_LAB_albumin_low'] = (m(alb_last, 99) < 35).astype(int)
        ov['OV_LAB_albumin_low_with_wt_loss'] = ((ov['OV_LAB_albumin_low'] == 1) & (has_wt_loss == 1)).astype(int)
    plt_lab = lab_vals[lab_vals['TERM'].str.contains('Platelet count', case=False, na=False)]
    if len(plt_lab) > 0:
        plt_last = plt_lab.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        ov['OV_LAB_platelets_high'] = (m(plt_last) > 400).astype(int)
        ov['OV_LAB_platelets_high_with_symptoms'] = ((ov['OV_LAB_platelets_high'] == 1) & (ov['NICE_cardinal_count'] >= 1)).astype(int)
    egfr = lab_vals[lab_vals['TERM'].str.contains('eGFR', case=False, na=False)]
    if len(egfr) > 0:
        egfr_last = egfr.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        egfr_first = egfr.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first()
        egfr_decline = egfr_first - egfr_last
        ov['OV_LAB_egfr_declining'] = (m(egfr_decline) > 10).astype(int)
        ov['OV_LAB_egfr_low'] = (m(egfr_last, 999) < 60).astype(int)
        ov['OV_LAB_egfr_declining_with_urinary'] = ((ov['OV_LAB_egfr_declining'] == 1) & (has_urinary == 1)).astype(int)
    esr = lab_vals[lab_vals['TERM'].str.contains('Erythrocyte sedimentation', case=False, na=False)]
    if len(esr) > 0:
        esr_last = esr.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        ov['OV_LAB_esr_elevated'] = (m(esr_last) > 30).astype(int)
    alp = lab_vals[lab_vals['TERM'].str.contains('Alkaline phosphatase', case=False, na=False)]
    if len(alp) > 0:
        alp_last = alp.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        ov['OV_LAB_alp_elevated'] = (m(alp_last) > 130).astype(int)

    # BLOCK 6: DIAGNOSTIC PATHWAY
    print(f"  BLOCK 6: Diagnostic pathway...")
    has_imaging = m(has_cat(obs_AB, 'IMAGING'), False).astype(int)
    has_gynae_proc = m(has_cat(obs_AB, 'GYNAE_PROCEDURES'), False).astype(int)
    ov['OV_DX_has_imaging'] = has_imaging
    ov['OV_DX_has_gynae_procedure'] = has_gynae_proc
    ov['OV_DX_symptoms_no_imaging'] = ((ov['NICE_cardinal_count'] >= 2) & (has_imaging == 0)).astype(int)
    sym_first = obs_AB[obs_AB['CATEGORY'].isin(['ABDOMINAL_PAIN', 'ABDOMINAL_BLOATING', 'URINARY_SYMPTOMS', 'GI_SYMPTOMS'])].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    img_first = obs_AB[obs_AB['CATEGORY'] == 'IMAGING'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap = (img_first - sym_first).dt.days
    ov['OV_DX_symptom_to_imaging_days'] = m(gap, -1)
    ov['OV_DX_imaging_delayed_60d'] = (ov['OV_DX_symptom_to_imaging_days'] > 60).astype(int)
    ov['OV_DX_imaging_delayed_120d'] = (ov['OV_DX_symptom_to_imaging_days'] > 120).astype(int)

    # BLOCK 7: TREATMENT PATTERNS
    print(f"  BLOCK 7: Treatment patterns...")
    has_nsaid = m(med_AB[med_AB['CATEGORY'] == 'NSAIDS'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    has_opioid = m(med_AB[med_AB['CATEGORY'] == 'OPIOID ANALGESICS'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    has_ppi = m(med_AB[med_AB['CATEGORY'] == 'PPI'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    has_antispasm = m(med_AB[med_AB['CATEGORY'] == 'GI ANTISPASMODICS'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    has_laxative = m(med_AB[med_AB['CATEGORY'] == 'LAXATIVES'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    has_antiemetic = m(med_AB[med_AB['CATEGORY'] == 'ANTIEMETICS'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    has_iron = m(med_AB[med_AB['CATEGORY'] == 'IRON SUPPLEMENTS'].groupby('PATIENT_GUID').size() > 0, False).astype(int)
    ov['OV_TX_pain_with_opioid'] = ((has_pain == 1) & (has_opioid == 1)).astype(int)
    ov['OV_TX_pain_escalation'] = ((has_nsaid == 1) & (has_opioid == 1)).astype(int)
    ov['OV_TX_pain_escalation_with_cardinal'] = ((ov['OV_TX_pain_escalation'] == 1) & (ov['NICE_cardinal_count'] >= 2)).astype(int)
    gi_med_count = has_ppi.astype(int) + has_antispasm.astype(int) + has_laxative.astype(int) + has_antiemetic.astype(int)
    ov['OV_TX_gi_treated'] = (gi_med_count >= 1).astype(int)
    ov['OV_TX_gi_multi_treated'] = (gi_med_count >= 2).astype(int)
    ov['OV_TX_gi_treated_no_inv'] = ((ov['OV_TX_gi_treated'] == 1) & (has_inv == 0)).astype(int)
    ov['OV_TX_gi_treated_with_bloating'] = ((ov['OV_TX_gi_treated'] == 1) & (has_bloating == 1)).astype(int)
    ov['OV_TX_iron_prescribed'] = has_iron
    if 'OV_LAB_hb_declining' in ov.columns:
        ov['OV_TX_iron_with_declining_hb'] = ((has_iron == 1) & (ov['OV_LAB_hb_declining'] == 1)).astype(int)
    has_dvt = m(has_cat(obs_AB, 'DVT'), False).astype(int)
    has_pe = m(has_cat(obs_AB, 'PULMONARY_EMBOLISM'), False).astype(int)
    ov['OV_TX_vte'] = ((has_dvt == 1) | (has_pe == 1)).astype(int)
    ov['OV_TX_vte_with_symptoms'] = ((ov['OV_TX_vte'] == 1) & (ov['NICE_cardinal_count'] >= 1)).astype(int)

    ov = ov.fillna(0).replace([np.inf, -np.inf], 0)
    nunique = ov.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        ov = ov.drop(columns=constant)
        print(f"  Removed {len(constant)} constant columns")
    print(f"\n  ✅ Ovarian-specific features: {ov.shape[1]}")
    for prefix, name in [
        ('NICE_', 'NICE guidelines'), ('OV_ibs', 'IBS-mimic'), ('OV_uti', 'UTI-mimic'),
        ('OV_postmeno', 'PMB-mimic'), ('OV_classic', 'Classic triad'),
        ('OV_cyst', 'Cyst interactions'), ('OV_ascites', 'Ascites/Mass'), ('OV_mass', 'Mass'),
        ('OV_RF_', 'Risk factors'), ('OV_LAB_', 'Lab thresholds'), ('OV_DX_', 'Diagnostic pathway'),
        ('OV_TX_', 'Treatment patterns')
    ]:
        count = len([c for c in ov.columns if c.startswith(prefix)])
        if count > 0:
            print(f"    {name:25s}: {count}")
    return ov


# RUN STEP 4d FOR ALL 3 WINDOWS
final_matrices = {}
for window in ['3mo', '6mo', '12mo']:
    print(f"\n\n{'#'*70}")
    print(f"  STEP 4d — {window.upper()}")
    print(f"{'#'*70}")
    clin = data[window]['clinical']
    med = data[window]['med']
    existing = mega_matrices[window]
    ov_features = build_ovarian_features(clin, med, existing, window)
    final = existing.join(ov_features, how='left', rsuffix='_ov')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]
    ov_cols = [c for c in final.columns if c.startswith('NICE_') or c.startswith('OV_')]
    nzv_remove = [col for col in ov_cols if final[col].value_counts(normalize=True).iloc[0] > 0.995]
    if nzv_remove:
        final = final.drop(columns=nzv_remove)
        print(f"  Removed {len(nzv_remove)} near-zero-var ovarian features")
    final_matrices[window] = final
    print(f"\n  FINAL: {final.shape[0]} × {final.shape[1]} features (added {final.shape[1] - existing.shape[1]} ovarian-specific)")
    out_path = FE_RESULTS / window / f"feature_matrix_final_{window}.csv"
    final.to_csv(out_path)
    print(f"  ✅ Saved: {out_path}")
    numeric_cols = [c for c in final.select_dtypes(include=[np.number]).columns if c != 'LABEL']
    label_corr = final[numeric_cols].corrwith(final['LABEL']).abs().sort_values(ascending=False)
    print(f"\n  Top 20 features:")
    for i, (col, corr) in enumerate(label_corr.head(20).items()):
        marker = '◆' if (col.startswith('NICE_') or col.startswith('OV_')) else ' '
        print(f"    {marker} {i+1:2d}. {col}: {corr:.4f}")
    ov_ranks = [(col, corr, rank+1) for rank, (col, corr) in enumerate(label_corr.items()) if col.startswith('NICE_') or col.startswith('OV_')][:15]
    print(f"\n  Ovarian-specific feature rankings:")
    for col, corr, rank in ov_ranks:
        print(f"    Rank {rank:3d}: {col}: {corr:.4f}")

print(f"\n{'═'*70}")
print(f"  STEP 4d COMPLETE — FINAL MATRICES (Option A, No Composite Scores)")
print(f"{'═'*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = final_matrices[window]
    ov_cols = [c for c in fm.columns if c.startswith('NICE_') or c.startswith('OV_')]
    print(f"\n  {window}: {fm.shape[0]} × {fm.shape[1]} features ({len(ov_cols)} ovarian-specific)")
    print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
print(f"\n  Next: Run Step 4e (new signal features), then Step 5 cleanup → modeling.")
print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — NEW SIGNAL FEATURES
# Step 4e: SNOMED-level, recency, distinct visit counts
# Run AFTER Step 4d (builds on final_matrices)
# ═══════════════════════════════════════════════════════════════

def build_new_signal_features(clin_df, med_df, existing_fm, window_name):
    """
    Features that capture signal not already in the pipeline:
    1. Top individual SNOMED binary features (sub-category granularity)
    2. Symptom recency scores (time-weighted importance)
    3. Distinct visit date counts per symptom (persistence signal)
    """

    print(f"\n{'═'*70}")
    print(f"  NEW SIGNAL FEATURES — {window_name}")
    print(f"{'═'*70}")

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

    obs_df = clin[clin['EVENT_TYPE'] == 'OBSERVATION'].copy()
    obs_AB = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: TOP SNOMED-LEVEL BINARY FEATURES
    # Individual SNOMED codes have more specificity than categories
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 1: Top SNOMED-level binary features...")

    # Find SNOMEDs that differ most between pos/neg cohorts
    pos_patients = set(existing_fm[existing_fm['LABEL'] == 1].index)
    neg_patients = set(existing_fm[existing_fm['LABEL'] == 0].index)
    n_pos = len(pos_patients)
    n_neg = len(neg_patients)

    # Clinical SNOMED codes
    obs_snomed = obs_AB[obs_AB['SNOMED_ID'].notna()].copy()
    snomed_patient = obs_snomed.groupby('SNOMED_ID')['PATIENT_GUID'].apply(set)

    snomed_scores = []
    for snomed_id, patients_with in snomed_patient.items():
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
        snomed_scores.append({
            'snomed_id': snomed_id,
            'rate_ratio': rate_ratio,
            'abs_diff': abs_diff,
            'pos_rate': pos_rate,
            'neg_rate': neg_rate,
            'total_patients': len(patients_with)
        })

    if snomed_scores:
        score_df = pd.DataFrame(snomed_scores)
        # Select top 50 by absolute rate difference (most discriminative)
        top_snomeds = score_df.nlargest(50, 'abs_diff')

        # Get SNOMED term names for readable column names
        snomed_terms = obs_snomed.drop_duplicates('SNOMED_ID').set_index('SNOMED_ID')['TERM'].to_dict()

        for _, row in top_snomeds.iterrows():
            sid = row['snomed_id']
            patients_with = snomed_patient[sid]
            term = snomed_terms.get(sid, str(sid))
            safe_name = str(term).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace('{', '').replace('}', '').replace('/', '_').replace('\\', '_')[:50]
            nf[f'SNOMED_OBS_{safe_name}'] = nf.index.isin(patients_with).astype(int)

        print(f"    Added {len(top_snomeds)} obs SNOMED features")

    # Medication SNOMED codes
    med_snomed = med_AB[med_AB['SNOMED_ID'].notna()].copy()
    # Use TERM instead of SNOMED_ID for meds (some have -1 as SNOMED)
    if 'TERM' in med_snomed.columns:
        med_term_patient = med_snomed.groupby('TERM')['PATIENT_GUID'].apply(set)

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
                nf[f'SNOMED_MED_{safe_name}'] = nf.index.isin(patients_with).astype(int)

            print(f"    Added {len(top_med_terms)} med SNOMED/term features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: SYMPTOM RECENCY SCORES
    # Recent symptoms matter more — weight by inverse distance to index
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 2: Symptom recency scores...")

    key_symptoms = [
        'ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'ABDOMINAL_BLOATING', 'EARLY_SATIETY',
        'GYNAECOLOGICAL_BLEEDING', 'VAGINAL_DISCHARGE', 'URINARY_SYMPTOMS',
        'URINE_ABNORMALITIES', 'WEIGHT_LOSS', 'FATIGUE', 'ABDOMINAL_MASS',
        'ASCITES', 'OVARIAN_CYST'
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
    # BLOCK 4: COMBINED RECENCY × PERSISTENCE INTERACTIONS
    # Patients with recent AND persistent symptoms are highest risk
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 4: Recency × persistence interactions...")

    interaction_count = 0
    for cat in key_symptoms:
        rec_col = f'RECENCY_{cat}_score'
        vis_col = f'VISITS_{cat}_distinct_dates'
        if rec_col in nf.columns and vis_col in nf.columns:
            # Recency × frequency product (high when both recent AND frequent)
            nf[f'RxP_{cat}_score'] = nf[rec_col] * nf[vis_col]
            interaction_count += 1

    print(f"    Added {interaction_count} recency×persistence features")

    # Clean up
    nf = nf.fillna(0).replace([np.inf, -np.inf], 0)
    nunique = nf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        nf = nf.drop(columns=constant)
        print(f"\n  Removed {len(constant)} constant columns")

    print(f"\n  ✅ New signal features: {nf.shape[1]} total")
    for prefix, name in [
        ('SNOMED_OBS_', 'SNOMED obs'), ('SNOMED_MED_', 'SNOMED med'),
        ('RECENCY_', 'Recency'), ('VISITS_', 'Distinct visits'), ('RxP_', 'Recency×Persistence')
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
    existing = final_matrices[window]

    new_features = build_new_signal_features(clin, med, existing, window)
    final = existing.join(new_features, how='left', rsuffix='_4e')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]

    final_matrices[window] = final

    print(f"\n  FINAL: {final.shape[0]} × {final.shape[1]} features (added {new_features.shape[1]} new)")

    # Save as the new final
    out_path = FE_RESULTS / window / f"feature_matrix_final_{window}.csv"
    final.to_csv(out_path)
    print(f"  ✅ Saved: {out_path}")

print(f"\n{'═'*70}")
print(f"  STEP 4e COMPLETE — NEW SIGNAL FEATURES ADDED")
print(f"{'═'*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = final_matrices[window]
    print(f"\n  {window}: {fm.shape[0]} × {fm.shape[1]} features")
    print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
print(f"\n  Next: Run Step 4f (lung-style trend features), then Step 5 cleanup → modeling.")
print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — LUNG-STYLE TREND & FREQUENCY FEATURES
# Step 4f: Per-symptom frequency trends, interval stats, worsening flags
# Per-lab trend correlation and percent change
# Inspired by AI-UK lung cancer v2-ensemble pipeline
# ═══════════════════════════════════════════════════════════════

def build_trend_features(clin_df, med_df, existing_fm, window_name):
    """
    Lung-style per-symptom/per-lab trend features:
    1. Frequency per month per symptom
    2. Frequency trend slope per symptom
    3. Mean/median interval between occurrences per symptom
    4. First-half vs second-half frequency (acceleration magnitude)
    5. IS_WORSENING flag per symptom
    6. Lab percent change and trend R²
    """

    print(f"\n{'═'*70}")
    print(f"  LUNG-STYLE TREND FEATURES — {window_name}")
    print(f"{'═'*70}")

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

    obs_df = clin[clin['EVENT_TYPE'] == 'OBSERVATION'].copy()
    lab_df = clin[clin['EVENT_TYPE'] == 'LAB VALUE'].copy()
    obs_AB = obs_df[obs_df['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    lab_AB = lab_df[lab_df['TIME_WINDOW'].isin(['A', 'B'])].copy()

    key_symptoms = [
        'ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'ABDOMINAL_BLOATING', 'EARLY_SATIETY',
        'GYNAECOLOGICAL_BLEEDING', 'VAGINAL_DISCHARGE', 'URINARY_SYMPTOMS',
        'URINE_ABNORMALITIES', 'WEIGHT_LOSS', 'FATIGUE', 'ABDOMINAL_MASS',
        'ASCITES', 'OVARIAN_CYST', 'IMAGING', 'GYNAE_PROCEDURES',
        'CLINICAL_ASSESSMENT', 'MONITORING'
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
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 2: Per-medication frequency & trend...")

    key_meds = [
        'OPIOID ANALGESICS', 'NSAIDS', 'PPI', 'GI ANTISPASMODICS',
        'LAXATIVES', 'ANTIEMETICS', 'IRON SUPPLEMENTS', 'UTI ANTIBIOTICS',
        'GENERAL ANTIBIOTICS', 'BLADDER ANTISPASMODICS'
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
    # BLOCK 3: LAB PERCENT CHANGE & TREND R²
    # ══════════════════════════════════════════════════════════
    print(f"\n  BLOCK 3: Lab percent change & trend R²...")

    lab_vals = lab_AB[lab_AB['VALUE'].notna()].copy()
    lab_categories = ['HAEMATOLOGY', 'RENAL', 'INFLAMMATORY', 'METABOLIC', 'LIVER']

    for cat in lab_categories:
        cat_labs = lab_vals[lab_vals['CATEGORY'] == cat].copy()
        if len(cat_labs) < 10:
            continue

        prefix = f'LABTREND_{cat}'
        cat_labs = cat_labs.sort_values(['PATIENT_GUID', 'EVENT_DATE'])

        # Percent change (first to last value)
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

        # Trend R² (how linear is the trend)
        def trend_r2(group):
            if len(group) < 3:
                return np.nan
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
            y = group['VALUE'].values.astype(float)
            if x[-1] == 0 or np.std(y) == 0:
                return np.nan
            try:
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                return r2
            except Exception:
                return np.nan

        r2_vals = cat_labs.groupby('PATIENT_GUID').apply(trend_r2)
        tf[f'{prefix}_trend_r2'] = tf.index.map(r2_vals).fillna(0).astype(float)

        # Trend direction: 1=increasing, -1=decreasing, 0=stable
        def trend_direction(group):
            if len(group) < 2:
                return 0
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
            y = group['VALUE'].values.astype(float)
            if x[-1] == 0:
                return 0
            try:
                slope = np.polyfit(x, y, 1)[0]
                if abs(slope) < 0.001:
                    return 0
                return 1 if slope > 0 else -1
            except Exception:
                return 0

        directions = cat_labs.groupby('PATIENT_GUID').apply(trend_direction)
        tf[f'{prefix}_trend_direction'] = tf.index.map(directions).fillna(0).astype(int)

    # Per-term lab features for top lab terms
    top_lab_terms = lab_vals['TERM'].value_counts()
    top_terms = top_lab_terms[top_lab_terms >= 50].index.tolist()[:15]  # top 15 terms

    for term in top_terms:
        term_df = lab_vals[lab_vals['TERM'] == term].sort_values(['PATIENT_GUID', 'EVENT_DATE'])
        safe_term = term.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace('/', '_')[:35]
        prefix = f'LTTREND_{safe_term}'

        # Percent change
        pct = term_df.groupby('PATIENT_GUID').apply(pct_change)
        tf[f'{prefix}_pct_change'] = tf.index.map(pct).fillna(0).astype(float)

        # Trend R²
        r2 = term_df.groupby('PATIENT_GUID').apply(trend_r2)
        tf[f'{prefix}_trend_r2'] = tf.index.map(r2).fillna(0).astype(float)

    lab_trend_cols = [c for c in tf.columns if c.startswith('LABTREND_') or c.startswith('LTTREND_')]
    print(f"    Added {len(lab_trend_cols)} lab trend features")

    # Clean up
    tf = tf.fillna(0).replace([np.inf, -np.inf], 0)
    nunique = tf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        tf = tf.drop(columns=constant)
        print(f"\n  Removed {len(constant)} constant columns")

    print(f"\n  ✅ Lung-style trend features: {tf.shape[1]} total")
    for prefix, name in [
        ('TREND_', 'Symptom trends'), ('MEDTREND_', 'Med trends'),
        ('LABTREND_', 'Lab category trends'), ('LTTREND_', 'Lab term trends')
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
    existing = final_matrices[window]

    trend_features = build_trend_features(clin, med, existing, window)
    final = existing.join(trend_features, how='left', rsuffix='_4f')
    final = final.fillna(0).replace([np.inf, -np.inf], 0)
    final = final.loc[:, ~final.columns.duplicated()]

    final_matrices[window] = final

    print(f"\n  FINAL: {final.shape[0]} × {final.shape[1]} features (added {trend_features.shape[1]} new)")

    out_path = FE_RESULTS / window / f"feature_matrix_final_{window}.csv"
    final.to_csv(out_path)
    print(f"  ✅ Saved: {out_path}")

print(f"\n{'═'*70}")
print(f"  STEP 4f COMPLETE — LUNG-STYLE TREND FEATURES ADDED")
print(f"{'═'*70}")
for window in ['3mo', '6mo', '12mo']:
    fm = final_matrices[window]
    print(f"\n  {window}: {fm.shape[0]} × {fm.shape[1]} features")
    print(f"    Pos: {(fm['LABEL']==1).sum()} | Neg: {(fm['LABEL']==0).sum()}")
print(f"\n  Next: Run Step 5 cleanup on feature_matrix_final_*.csv, then modeling.")
print(f"{'═'*70}")