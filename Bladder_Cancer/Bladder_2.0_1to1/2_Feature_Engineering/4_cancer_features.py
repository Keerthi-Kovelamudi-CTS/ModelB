# ═══════════════════════════════════════════════════════════════
# BLADDER CANCER — CANCER-SPECIFIC FEATURES
# Haematuria deep features, urine workup patterns, LUTS subtypes,
# urological/imaging signals, smoking-driven risk, comorbidity scoring.
#
# Ported from Bladder_Cancer/2_Feature_Engineering/4_feature_engineering.py
# (LEVEL 3-9 blocks) and adapted to the new config-driven pipeline.
# This is the ONLY file that changes per cancer type (besides config.py).
# ═══════════════════════════════════════════════════════════════

import logging
import warnings

import numpy as np
import pandas as pd

import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

__all__ = ['build_cancer_specific_features']


def build_cancer_specific_features(clin_df, med_df, existing_fm, window_name, cfg=None):
    """
    Bladder-specific features:
      - Haematuria deep (subtypes, time span, recurrence, B/A acceleration)
      - Urine workup intensity (investigations + lab abnormalities, subtypes)
      - LUTS subtypes (bladder pain, hesitancy, retention, UTI)
      - Urological conditions / catheter / imaging
      - Smoking-driven risk + risk-factor combinations
      - Bladder comorbidity score (recurrent UTI + BPH + previous cancer)
    """
    if cfg is None:
        cfg = config
    PREFIX = cfg.PREFIX  # 'BLAD_'

    logger.info(f"  BLADDER-SPECIFIC FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')

    patients = existing_fm.index
    bf = pd.DataFrame(index=patients)
    bf.index.name = 'PATIENT_GUID'

    obs_AB = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    haem = obs_AB[obs_AB['CATEGORY'] == 'HAEMATURIA'].copy()

    # ── 1. HAEMATURIA DEEP ───────────────────────────────────────
    bf[f'{PREFIX}HAEM_total_count'] = haem.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    bf[f'{PREFIX}HAEM_unique_terms'] = haem.groupby('PATIENT_GUID')['TERM'].nunique().reindex(patients, fill_value=0) if len(haem) else 0
    for window in ['A', 'B']:
        w = haem[haem['TIME_WINDOW'] == window]
        bf[f'{PREFIX}HAEM_W{window}_count'] = w.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    # Term-pattern subtypes
    for term_pattern, col in [
        ('frank', 'HAEM_frank'),
        ('painless', 'HAEM_painless'),
        ('painful', 'HAEM_painful'),
        ('microscopic', 'HAEM_microscopic'),
        ('history|h/o', 'HAEM_history'),
    ]:
        if len(haem):
            matched = haem[haem['TERM'].str.contains(term_pattern, case=False, na=False, regex=True)]
            bf[f'{PREFIX}{col}_count'] = matched.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
        else:
            bf[f'{PREFIX}{col}_count'] = 0
    # Time span
    if 'MONTHS_BEFORE_INDEX' in haem.columns and len(haem):
        bf[f'{PREFIX}HAEM_first_months_before'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].max().reindex(patients).fillna(-1)
        bf[f'{PREFIX}HAEM_last_months_before'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min().reindex(patients).fillna(-1)
        bf[f'{PREFIX}HAEM_span_months'] = (bf[f'{PREFIX}HAEM_first_months_before'] - bf[f'{PREFIX}HAEM_last_months_before']).clip(lower=0)
        bf[f'{PREFIX}HAEM_mean_months_before'] = haem.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].mean().reindex(patients).fillna(-1)
    else:
        bf[f'{PREFIX}HAEM_first_months_before'] = -1
        bf[f'{PREFIX}HAEM_last_months_before'] = -1
        bf[f'{PREFIX}HAEM_span_months'] = 0
        bf[f'{PREFIX}HAEM_mean_months_before'] = -1
    # Recurrence
    bf[f'{PREFIX}HAEM_recurrent'] = (bf[f'{PREFIX}HAEM_total_count'] > 1).astype(int)
    bf[f'{PREFIX}HAEM_frequent'] = (bf[f'{PREFIX}HAEM_total_count'] >= 3).astype(int)
    # B/A acceleration ratio
    wa = bf[f'{PREFIX}HAEM_WA_count'].astype(float)
    wb = bf[f'{PREFIX}HAEM_WB_count'].astype(float)
    bf[f'{PREFIX}HAEM_acceleration'] = np.where(
        wa > 0, wb / wa.clip(lower=1),
        np.where(wb > 0, 2.0, 0.0)
    )
    bf[f'{PREFIX}HAEM_unique_dates'] = haem.groupby('PATIENT_GUID')['EVENT_DATE'].nunique().reindex(patients, fill_value=0) if len(haem) else 0
    bf[f'{PREFIX}HAEM_any_flag'] = (bf[f'{PREFIX}HAEM_total_count'] > 0).astype(int)

    # ── 2. URINE INVESTIGATIONS + LAB ABNORMALITIES DEEP ─────────
    ui = obs_AB[obs_AB['CATEGORY'] == 'URINE_INVESTIGATIONS']
    bf[f'{PREFIX}URINE_inv_count'] = ui.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    bf[f'{PREFIX}URINE_inv_unique_terms'] = ui.groupby('PATIENT_GUID')['TERM'].nunique().reindex(patients, fill_value=0) if len(ui) else 0
    for window in ['A', 'B']:
        w = ui[ui['TIME_WINDOW'] == window]
        bf[f'{PREFIX}URINE_inv_W{window}'] = w.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    for pat, col in [
        ('culture', 'URINE_culture'),
        ('microscopy', 'URINE_microscopy'),
        ('msu|mid-stream', 'URINE_msu'),
        ('cytolog', 'URINE_cytology'),
        ('antibacterial', 'URINE_antibacterial'),
    ]:
        if len(ui):
            matched = ui[ui['TERM'].str.contains(pat, case=False, na=False, regex=True)]
            bf[f'{PREFIX}{col}_count'] = matched.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
        else:
            bf[f'{PREFIX}{col}_count'] = 0

    ul = obs_AB[obs_AB['CATEGORY'] == 'URINE_LAB_ABNORMALITIES']
    bf[f'{PREFIX}URINE_lab_abnorm_count'] = ul.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    bf[f'{PREFIX}URINE_lab_unique_terms'] = ul.groupby('PATIENT_GUID')['TERM'].nunique().reindex(patients, fill_value=0) if len(ul) else 0
    for window in ['A', 'B']:
        w = ul[ul['TIME_WINDOW'] == window]
        bf[f'{PREFIX}URINE_lab_W{window}'] = w.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    for pat, col in [
        ('blood test', 'URINE_blood_test'),
        ('red cells|red ', 'URINE_red'),
        ('protein', 'URINE_protein'),
        ('dark|concentrated', 'URINE_dark'),
        ('leucocyte', 'URINE_leucocyte'),
        ('cytolog', 'URINE_lab_cytology'),
        ('specific gravity', 'URINE_specific_gravity'),
    ]:
        if len(ul):
            matched = ul[ul['TERM'].str.contains(pat, case=False, na=False, regex=True)]
            bf[f'{PREFIX}{col}_count'] = matched.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
        else:
            bf[f'{PREFIX}{col}_count'] = 0

    bf[f'{PREFIX}URINE_any_inv_flag'] = (bf[f'{PREFIX}URINE_inv_count'] > 0).astype(int)
    bf[f'{PREFIX}URINE_any_abnorm_flag'] = (bf[f'{PREFIX}URINE_lab_abnorm_count'] > 0).astype(int)
    bf[f'{PREFIX}URINE_total_activity'] = bf[f'{PREFIX}URINE_inv_count'] + bf[f'{PREFIX}URINE_lab_abnorm_count']

    # B/A acceleration on urine lab abnormalities (key signal)
    ulwa = bf[f'{PREFIX}URINE_lab_WA'].astype(float)
    ulwb = bf[f'{PREFIX}URINE_lab_WB'].astype(float)
    bf[f'{PREFIX}URINE_lab_acceleration'] = np.where(
        ulwa > 0, ulwb / ulwa.clip(lower=1),
        np.where(ulwb > 0, 2.0, 0.0)
    )

    # ── 3. LUTS SUBTYPES DEEP ────────────────────────────────────
    luts = obs_AB[obs_AB['CATEGORY'] == 'LUTS'].copy()
    bf[f'{PREFIX}LUTS_total_count'] = luts.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    bf[f'{PREFIX}LUTS_unique_symptoms'] = luts.groupby('PATIENT_GUID')['TERM'].nunique().reindex(patients, fill_value=0) if len(luts) else 0
    for window in ['A', 'B']:
        w = luts[luts['TIME_WINDOW'] == window]
        bf[f'{PREFIX}LUTS_W{window}_count'] = w.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    for pat, col in [
        ('bladder pain', 'LUTS_bladder_pain'),
        ('difficulty', 'LUTS_difficulty'),
        ('delay|hesitancy', 'LUTS_hesitancy'),
        ('dribbling', 'LUTS_dribbling'),
        ('frequency|polyuria', 'LUTS_frequency'),
        ('retention', 'LUTS_retention'),
        ('urinary tract infect', 'LUTS_uti'),
    ]:
        if len(luts):
            matched = luts[luts['TERM'].str.contains(pat, case=False, na=False, regex=True)]
            bf[f'{PREFIX}{col}_count'] = matched.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
        else:
            bf[f'{PREFIX}{col}_count'] = 0
    bf[f'{PREFIX}LUTS_any_flag'] = (bf[f'{PREFIX}LUTS_total_count'] > 0).astype(int)
    bf[f'{PREFIX}LUTS_recurrent'] = (bf[f'{PREFIX}LUTS_total_count'] > 1).astype(int)
    bf[f'{PREFIX}LUTS_multiple_symptoms'] = (bf[f'{PREFIX}LUTS_unique_symptoms'] > 1).astype(int)

    # ── 4. UROLOGICAL CONDITIONS / IMAGING / CATHETER ────────────
    for cat, col in [
        ('UROLOGICAL_CONDITIONS', 'UROL'),
        ('CATHETER_PROCEDURES', 'CATH'),
        ('IMAGING', 'IMAGING'),
        ('GYNAECOLOGICAL_BLEEDING', 'GYNAE_bleed'),
        ('RECURRENT_UTI', 'RECUTI'),
    ]:
        sub = obs_AB[obs_AB['CATEGORY'] == cat]
        bf[f'{PREFIX}{col}_count'] = sub.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
        bf[f'{PREFIX}{col}_any_flag'] = (bf[f'{PREFIX}{col}_count'] > 0).astype(int)

    # ── 5. RISK FACTOR COMBINATIONS (smoking is biggest bladder ca RF) ─
    smoke_cur = obs_AB[obs_AB['CATEGORY'] == 'CURRENT_SMOKER'].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    smoke_ex = obs_AB[obs_AB['CATEGORY'] == 'EX_SMOKER'].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0)
    bf[f'{PREFIX}smoking_any'] = ((smoke_cur > 0) | (smoke_ex > 0)).astype(int)
    bf[f'{PREFIX}smoking_current'] = (smoke_cur > 0).astype(int)
    bf[f'{PREFIX}smoking_ex'] = (smoke_ex > 0).astype(int)
    age = (existing_fm['AGE_AT_INDEX'] if 'AGE_AT_INDEX' in existing_fm.columns
           else pd.Series(65.0, index=patients))
    bf[f'{PREFIX}AGEX_SMOKER_over65'] = ((age >= 65) & (bf[f'{PREFIX}smoking_any'] == 1)).astype(int)
    bf[f'{PREFIX}AGEX_HAEM_over65'] = ((age >= 65) & (bf[f'{PREFIX}HAEM_any_flag'] == 1)).astype(int)

    # ── 6. BLADDER COMORBIDITY / RISK SCORE ─────────────────────
    rec_uti_flag = (obs_AB[obs_AB['CATEGORY'] == 'RECURRENT_UTI'].groupby('PATIENT_GUID').size()
                    .reindex(patients, fill_value=0) > 0).astype(int)
    bph_flag = (obs_AB[obs_AB['CATEGORY'] == 'BPH'].groupby('PATIENT_GUID').size()
                .reindex(patients, fill_value=0) > 0).astype(int)
    prev_cancer_flag = (obs_AB[obs_AB['CATEGORY'] == 'PREVIOUS_CANCER'].groupby('PATIENT_GUID').size()
                        .reindex(patients, fill_value=0) > 0).astype(int)

    bf[f'{PREFIX}RF_recurrent_uti'] = rec_uti_flag
    bf[f'{PREFIX}RF_bph'] = bph_flag
    bf[f'{PREFIX}RF_previous_cancer'] = prev_cancer_flag

    # Composite risk score (weighted) — family-history not curated in bladder codelist; omitted.
    bf[f'{PREFIX}RF_score'] = (
        prev_cancer_flag * 3
        + rec_uti_flag * 2
        + bph_flag * 1
        + bf[f'{PREFIX}smoking_current'] * 3
        + bf[f'{PREFIX}smoking_ex'] * 1
    )

    # ── 7. INTERACTIONS (haematuria × workup proxies) ────────────
    bf[f'{PREFIX}INT_haem_x_urine_inv'] = bf[f'{PREFIX}HAEM_any_flag'] * bf[f'{PREFIX}URINE_any_inv_flag']
    bf[f'{PREFIX}INT_haem_x_urine_abnorm'] = bf[f'{PREFIX}HAEM_any_flag'] * bf[f'{PREFIX}URINE_any_abnorm_flag']
    bf[f'{PREFIX}INT_haem_x_smoking'] = bf[f'{PREFIX}HAEM_any_flag'] * bf[f'{PREFIX}smoking_any']
    bf[f'{PREFIX}INT_recuti_x_haem'] = rec_uti_flag * bf[f'{PREFIX}HAEM_any_flag']
    bf[f'{PREFIX}INT_luts_x_haem'] = bf[f'{PREFIX}LUTS_any_flag'] * bf[f'{PREFIX}HAEM_any_flag']

    # ─────────────────────────────────────────────────────────────
    # Helpers + window splits for blocks 8-16 (ported from Prostate)
    # ─────────────────────────────────────────────────────────────
    obs_A = clin[clin['TIME_WINDOW'] == 'A'].copy()
    obs_B = clin[clin['TIME_WINDOW'] == 'B'].copy()

    def _count(df, cat):
        return df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)

    def _has(df, cat):
        return (_count(df, cat) > 0).astype(int)

    def _safe(col):
        return bf.get(col, pd.Series(0, index=patients)).astype(float)

    # Aggregate "constitutional" symptoms (no single CONSTITUTIONAL category in bladder schema)
    CONSTIT_CATS = ['WEIGHT_LOSS', 'FATIGUE', 'APPETITE_LOSS', 'NIGHT_SWEATS', 'ANAEMIA_DX']
    constit_B = sum(_count(obs_B, c) for c in CONSTIT_CATS)
    constit_A = sum(_count(obs_A, c) for c in CONSTIT_CATS)

    haem_B = _count(obs_B, 'HAEMATURIA')
    haem_A = _count(obs_A, 'HAEMATURIA')
    luts_B = _count(obs_B, 'LUTS')
    luts_A = _count(obs_A, 'LUTS')
    dys_B = _count(obs_B, 'DYSURIA')
    dys_A = _count(obs_A, 'DYSURIA')
    urinelab_B = _count(obs_B, 'URINE_LAB_ABNORMALITIES')
    urinelab_A = _count(obs_A, 'URINE_LAB_ABNORMALITIES')
    urineinv_B = _count(obs_B, 'URINE_INVESTIGATIONS')
    urineinv_A = _count(obs_A, 'URINE_INVESTIGATIONS')
    recuti_B = _count(obs_B, 'RECURRENT_UTI')
    recuti_A = _count(obs_A, 'RECURRENT_UTI')
    img_B = _count(obs_B, 'IMAGING')
    img_A = _count(obs_A, 'IMAGING')
    urol_B = _count(obs_B, 'UROLOGICAL_CONDITIONS')
    urol_A = _count(obs_A, 'UROLOGICAL_CONDITIONS')

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: MULTI-SYSTEM TRIFECTA (constitutional + urinary + haematuria, window B)
    # Cancer often presents with constitutional + local + bleeding signs simultaneously.
    # ══════════════════════════════════════════════════════════
    constit_flag_B = (constit_B > 0).astype(int)
    urinary_flag_B = (
        (luts_B + dys_B + recuti_B) > 0
    ).astype(int)
    bleed_flag_B = (
        (haem_B + _count(obs_B, 'GYNAECOLOGICAL_BLEEDING')) > 0
    ).astype(int)
    multi_B = constit_flag_B + urinary_flag_B + bleed_flag_B
    bf[f'{PREFIX}TRIFECTA_constit_urinary_haem_B'] = (multi_B >= 3).astype(int)
    bf[f'{PREFIX}MULTISYSTEM_PAIR_B'] = (multi_B >= 2).astype(int)
    bf[f'{PREFIX}MULTISYSTEM_count_B'] = multi_B

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: SYMPTOM VELOCITY (B - A, clipped to non-negative)
    # Recent acceleration of symptoms. HAEM/URINE_lab acceleration already in section 1-2.
    # ══════════════════════════════════════════════════════════
    bf[f'{PREFIX}VEL_luts'] = (luts_B - luts_A).clip(lower=0)
    bf[f'{PREFIX}VEL_dysuria'] = (dys_B - dys_A).clip(lower=0)
    bf[f'{PREFIX}VEL_recuti'] = (recuti_B - recuti_A).clip(lower=0)
    bf[f'{PREFIX}VEL_imaging'] = (img_B - img_A).clip(lower=0)
    bf[f'{PREFIX}VEL_urological'] = (urol_B - urol_A).clip(lower=0)
    bf[f'{PREFIX}VEL_constitutional'] = (constit_B - constit_A).clip(lower=0)
    bf[f'{PREFIX}VEL_urine_inv'] = (urineinv_B - urineinv_A).clip(lower=0)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: NEW SYMPTOM IN WINDOW B ONLY (B>0 & A==0)
    # Cancer pattern: recent-onset symptoms with no prior history.
    # ══════════════════════════════════════════════════════════
    bf[f'{PREFIX}NEW_haem_B_only'] = ((haem_B > 0) & (haem_A == 0)).astype(int)
    bf[f'{PREFIX}NEW_luts_B_only'] = ((luts_B > 0) & (luts_A == 0)).astype(int)
    bf[f'{PREFIX}NEW_dysuria_B_only'] = ((dys_B > 0) & (dys_A == 0)).astype(int)
    bf[f'{PREFIX}NEW_recuti_B_only'] = ((recuti_B > 0) & (recuti_A == 0)).astype(int)
    bf[f'{PREFIX}NEW_urinelab_B_only'] = ((urinelab_B > 0) & (urinelab_A == 0)).astype(int)
    bf[f'{PREFIX}NEW_imaging_B_only'] = ((img_B > 0) & (img_A == 0)).astype(int)
    bf[f'{PREFIX}NEW_constitutional_B_only'] = ((constit_B > 0) & (constit_A == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: AGE × CLINICAL INTERACTIONS
    # Designed to survive 1:1 age-band matching (where AGE alone becomes uninformative,
    # but age × symptom/lab/med still discriminates within bands).
    # ══════════════════════════════════════════════════════════

    # Age-derived (non-interaction) — peak bladder-cancer incidence ~age 75
    bf[f'{PREFIX}AGEX_PEAK_DIST'] = (age - 75).abs()
    bf[f'{PREFIX}AGEX_OVER_60'] = (age >= 60).astype(int)
    bf[f'{PREFIX}AGEX_OVER_70'] = (age >= 70).astype(int)
    bf[f'{PREFIX}AGEX_OVER_80'] = (age >= 80).astype(int)
    bf[f'{PREFIX}AGEX_DECILE'] = pd.qcut(
        age.rank(method='first'), q=10, labels=False, duplicates='drop'
    ).fillna(0).astype(int)

    # Age × symptom counts (window B)
    bf[f'{PREFIX}AGEX_HAEM_count_B'] = age * haem_B
    bf[f'{PREFIX}AGEX_LUTS_count_B'] = age * luts_B
    bf[f'{PREFIX}AGEX_DYSURIA_count_B'] = age * dys_B
    bf[f'{PREFIX}AGEX_URINELAB_count_B'] = age * urinelab_B
    bf[f'{PREFIX}AGEX_URINEINV_count_B'] = age * urineinv_B
    bf[f'{PREFIX}AGEX_RECUTI_count_B'] = age * recuti_B
    bf[f'{PREFIX}AGEX_IMAGING_count_B'] = age * img_B
    bf[f'{PREFIX}AGEX_UROL_count_B'] = age * urol_B
    bf[f'{PREFIX}AGEX_CONSTIT_count_B'] = age * constit_B

    # Age × symptom acceleration (B - A clipped)
    bf[f'{PREFIX}AGEX_HAEM_acceleration'] = age * (haem_B - haem_A).clip(lower=0)
    bf[f'{PREFIX}AGEX_LUTS_acceleration'] = age * (luts_B - luts_A).clip(lower=0)
    bf[f'{PREFIX}AGEX_URINELAB_acceleration'] = age * (urinelab_B - urinelab_A).clip(lower=0)
    bf[f'{PREFIX}AGEX_RECUTI_acceleration'] = age * (recuti_B - recuti_A).clip(lower=0)

    # Age × medication patterns (relevant bladder meds)
    bf[f'{PREFIX}AGEX_UTI_ABX'] = age * _has(med[med['TIME_WINDOW'].isin(['A', 'B'])], 'UTI_ANTIBIOTICS')
    bf[f'{PREFIX}AGEX_HAEMOSTATIC'] = age * _has(med[med['TIME_WINDOW'].isin(['A', 'B'])], 'HAEMOSTATIC')
    bf[f'{PREFIX}AGEX_ANTICOAG'] = age * _has(med[med['TIME_WINDOW'].isin(['A', 'B'])], 'ANTICOAGULANTS')
    bf[f'{PREFIX}AGEX_CATH_SUPPLIES'] = age * _has(
        med[med['TIME_WINDOW'].isin(['A', 'B'])], 'CATHETER_SUPPLIES_CATHETERS'
    )

    # Age × NEW-symptom-in-B
    bf[f'{PREFIX}AGEX_NEW_haem_B'] = age * bf[f'{PREFIX}NEW_haem_B_only']
    bf[f'{PREFIX}AGEX_NEW_luts_B'] = age * bf[f'{PREFIX}NEW_luts_B_only']
    bf[f'{PREFIX}AGEX_NEW_recuti_B'] = age * bf[f'{PREFIX}NEW_recuti_B_only']

    # Composite "elderly + clinical signal" flags
    elderly = (age >= 70).astype(int)
    bf[f'{PREFIX}ELDERLY_WITH_HAEM'] = (elderly & (haem_B > 0).astype(int))
    bf[f'{PREFIX}ELDERLY_WITH_LUTS'] = (elderly & (luts_B > 0).astype(int))
    bf[f'{PREFIX}ELDERLY_WITH_RECUTI'] = (elderly & (recuti_B > 0).astype(int))
    bf[f'{PREFIX}ELDERLY_WITH_URINELAB'] = (elderly & (urinelab_B > 0).astype(int))
    bf[f'{PREFIX}ELDERLY_WITH_CONSTIT'] = (elderly & (constit_B > 0).astype(int))
    bf[f'{PREFIX}ELDERLY_WITH_SMOKING'] = (elderly & bf[f'{PREFIX}smoking_any'])
    bf[f'{PREFIX}ELDERLY_NEW_HAEM_B'] = (elderly & bf[f'{PREFIX}NEW_haem_B_only'])

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: TRIPLE INTERACTIONS (age × symptom × biomarker)
    # ══════════════════════════════════════════════════════════
    smoke_any = bf[f'{PREFIX}smoking_any'].astype(float)
    smoke_cur_f = bf[f'{PREFIX}smoking_current'].astype(float)
    haem_any = bf[f'{PREFIX}HAEM_any_flag'].astype(float)
    recuti_any = bf[f'{PREFIX}RF_recurrent_uti'].astype(float)
    bf[f'{PREFIX}TRIPLE_age_haemB_smoking'] = age * haem_B * smoke_any
    bf[f'{PREFIX}TRIPLE_age_haemB_smoking_current'] = age * haem_B * smoke_cur_f
    bf[f'{PREFIX}TRIPLE_age_haemB_recUTI'] = age * haem_B * recuti_any
    bf[f'{PREFIX}TRIPLE_age_lutsB_haem'] = age * luts_B * haem_any
    bf[f'{PREFIX}TRIPLE_age_haemB_urinelab'] = age * haem_B * urinelab_B
    bf[f'{PREFIX}TRIPLE_age_haemB_imaging'] = age * haem_B * img_B
    bf[f'{PREFIX}TRIPLE_age_recutiB_haem'] = age * recuti_B * haem_any

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: COMORBIDITY BURDEN + RECENT ACTIVITY ESCALATION
    # ══════════════════════════════════════════════════════════
    n_categories = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(patients, fill_value=0).astype(int)
    bf[f'{PREFIX}COMORBIDITY_n_categories'] = n_categories
    bf[f'{PREFIX}COMORBIDITY_x_age'] = age * n_categories
    bf[f'{PREFIX}COMORBIDITY_high'] = (n_categories >= 5).astype(int)
    bf[f'{PREFIX}COMORBIDITY_x_haem'] = n_categories * haem_any
    bf[f'{PREFIX}COMORBIDITY_x_smoking'] = n_categories * smoke_any

    events_B = obs_B.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    events_A = obs_A.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    bf[f'{PREFIX}ACTIVITY_events_B'] = events_B
    bf[f'{PREFIX}ACTIVITY_events_A'] = events_A
    bf[f'{PREFIX}ACTIVITY_velocity'] = (events_B - events_A).clip(lower=0)
    bf[f'{PREFIX}ACTIVITY_acceleration'] = (events_B / events_A.replace(0, 1)).clip(upper=10)
    bf[f'{PREFIX}AGEX_ACTIVITY_velocity'] = age * bf[f'{PREFIX}ACTIVITY_velocity']

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: INVESTIGATION × SYMPTOM PAIRS
    # Cystoscopy lives inside UROLOGICAL_CONDITIONS or IMAGING in bladder schema.
    # ══════════════════════════════════════════════════════════
    has_imaging = bf[f'{PREFIX}IMAGING_any_flag'].astype(float)
    has_urol = bf[f'{PREFIX}UROL_any_flag'].astype(float)
    has_urine_inv = bf[f'{PREFIX}URINE_any_inv_flag'].astype(float)
    bf[f'{PREFIX}IMG_x_haemB'] = has_imaging * haem_B
    bf[f'{PREFIX}IMG_x_lutsB'] = has_imaging * luts_B
    bf[f'{PREFIX}UROL_x_haemB'] = has_urol * haem_B
    bf[f'{PREFIX}URINEINV_x_haemB'] = has_urine_inv * haem_B
    bf[f'{PREFIX}PATHWAY_x_age'] = age * (has_imaging + has_urol).clip(upper=2)

    # ══════════════════════════════════════════════════════════
    # BLOCK 15: FIRST-EVENT-DATE PER CATEGORY + RECENT (<12mo) FLAGS
    # "Months since first <X>" — recent-onset cancer pattern.
    # ══════════════════════════════════════════════════════════
    anchor_lookup = clin.drop_duplicates('PATIENT_GUID').set_index('PATIENT_GUID')['INDEX_DATE']
    anchor_lookup = pd.to_datetime(anchor_lookup, errors='coerce')

    def _months_since_first(category):
        sub = clin[clin['CATEGORY'] == category]
        if sub.empty:
            return pd.Series(-1.0, index=patients)
        first = sub.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        first = pd.to_datetime(first, errors='coerce')
        gap = (anchor_lookup.reindex(first.index) - first).dt.days / 30.44
        return gap.reindex(patients).fillna(-1).astype(float)

    def _months_since_first_multi(categories):
        sub = clin[clin['CATEGORY'].isin(categories)]
        if sub.empty:
            return pd.Series(-1.0, index=patients)
        first = sub.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        first = pd.to_datetime(first, errors='coerce')
        gap = (anchor_lookup.reindex(first.index) - first).dt.days / 30.44
        return gap.reindex(patients).fillna(-1).astype(float)

    bf[f'{PREFIX}FIRST_haem_months'] = _months_since_first('HAEMATURIA')
    bf[f'{PREFIX}FIRST_luts_months'] = _months_since_first('LUTS')
    bf[f'{PREFIX}FIRST_dysuria_months'] = _months_since_first('DYSURIA')
    bf[f'{PREFIX}FIRST_recuti_months'] = _months_since_first('RECURRENT_UTI')
    bf[f'{PREFIX}FIRST_urinelab_months'] = _months_since_first('URINE_LAB_ABNORMALITIES')
    bf[f'{PREFIX}FIRST_urineinv_months'] = _months_since_first('URINE_INVESTIGATIONS')
    bf[f'{PREFIX}FIRST_imaging_months'] = _months_since_first('IMAGING')
    bf[f'{PREFIX}FIRST_urological_months'] = _months_since_first('UROLOGICAL_CONDITIONS')
    bf[f'{PREFIX}FIRST_constitutional_months'] = _months_since_first_multi(CONSTIT_CATS)

    bf[f'{PREFIX}RECENT_haem'] = (
        (bf[f'{PREFIX}FIRST_haem_months'] >= 0) & (bf[f'{PREFIX}FIRST_haem_months'] < 12)
    ).astype(int)
    bf[f'{PREFIX}RECENT_luts'] = (
        (bf[f'{PREFIX}FIRST_luts_months'] >= 0) & (bf[f'{PREFIX}FIRST_luts_months'] < 12)
    ).astype(int)
    bf[f'{PREFIX}RECENT_recuti'] = (
        (bf[f'{PREFIX}FIRST_recuti_months'] >= 0) & (bf[f'{PREFIX}FIRST_recuti_months'] < 12)
    ).astype(int)
    bf[f'{PREFIX}RECENT_urinelab'] = (
        (bf[f'{PREFIX}FIRST_urinelab_months'] >= 0) & (bf[f'{PREFIX}FIRST_urinelab_months'] < 12)
    ).astype(int)
    bf[f'{PREFIX}RECENT_constitutional'] = (
        (bf[f'{PREFIX}FIRST_constitutional_months'] >= 0) & (bf[f'{PREFIX}FIRST_constitutional_months'] < 12)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 16: SEQUENTIAL X→Y PATTERNS (within Z days)
    # Captures clinical workup orderings.
    # ══════════════════════════════════════════════════════════
    def _seq(cat_x, cat_y, max_gap_days):
        x_first = clin[clin['CATEGORY'] == cat_x].groupby('PATIENT_GUID')['EVENT_DATE'].min()
        y_first = clin[clin['CATEGORY'] == cat_y].groupby('PATIENT_GUID')['EVENT_DATE'].min()
        joined = pd.concat([x_first.rename('x'), y_first.rename('y')], axis=1).dropna()
        if joined.empty:
            return pd.Series(0, index=patients, dtype=int)
        joined['gap'] = (joined['y'] - joined['x']).dt.days
        flag = ((joined['gap'] >= 0) & (joined['gap'] <= max_gap_days)).astype(int)
        return flag.reindex(patients, fill_value=0).astype(int)

    bf[f'{PREFIX}SEQ_haem_then_imaging_60d'] = _seq('HAEMATURIA', 'IMAGING', 60)
    bf[f'{PREFIX}SEQ_haem_then_urological_90d'] = _seq('HAEMATURIA', 'UROLOGICAL_CONDITIONS', 90)
    bf[f'{PREFIX}SEQ_haem_then_urineinv_60d'] = _seq('HAEMATURIA', 'URINE_INVESTIGATIONS', 60)
    bf[f'{PREFIX}SEQ_luts_then_haem_90d'] = _seq('LUTS', 'HAEMATURIA', 90)
    bf[f'{PREFIX}SEQ_luts_then_imaging_180d'] = _seq('LUTS', 'IMAGING', 180)
    bf[f'{PREFIX}SEQ_recuti_then_haem_180d'] = _seq('RECURRENT_UTI', 'HAEMATURIA', 180)
    bf[f'{PREFIX}SEQ_recuti_then_imaging_180d'] = _seq('RECURRENT_UTI', 'IMAGING', 180)
    bf[f'{PREFIX}SEQ_urinelab_then_imaging_60d'] = _seq('URINE_LAB_ABNORMALITIES', 'IMAGING', 60)

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: CHARLSON-LITE WEIGHTED COMORBIDITY
    # Weights based on bladder-cancer relevance — captures multimorbidity
    # beyond simple count from BLOCK 13.
    # ══════════════════════════════════════════════════════════
    CHARLSON_LITE_BLADDER = {
        'HAEMATURIA': 3.0,                 # primary red flag
        'URINE_LAB_ABNORMALITIES': 2.0,
        'URINE_INVESTIGATIONS': 1.0,
        'RECURRENT_UTI': 2.0,              # often masks bladder ca in women
        'BPH': 0.5,                        # DDx for LUTS
        'LUTS': 1.0,
        'DYSURIA': 1.0,
        'SUPRAPUBIC_PAIN': 1.5,
        'BACK_PAIN': 0.5,
        'LOIN_PAIN': 1.0,
        'IMAGING': 1.5,
        'UROLOGICAL_CONDITIONS': 2.0,      # bladder stones, hydronephrosis
        'CATHETER_PROCEDURES': 1.5,
        'GYNAECOLOGICAL_BLEEDING': 1.0,
        'WEIGHT_LOSS': 2.0,
        'FATIGUE': 1.0,
        'APPETITE_LOSS': 1.5,
        'NIGHT_SWEATS': 1.0,
        'ANAEMIA_DX': 2.0,                 # chronic blood loss
        'PREVIOUS_CANCER': 2.0,
        'CURRENT_SMOKER': 2.0,
        'EX_SMOKER': 1.0,
        'DVT': 1.0,
        'PULMONARY_EMBOLISM': 1.0,
        'FRAILTY': 1.0,
    }
    score = pd.Series(0.0, index=patients)
    for cat, w in CHARLSON_LITE_BLADDER.items():
        score = score + w * _has(obs_AB, cat).astype(float)
    bf[f'{PREFIX}COMORB_weighted_score'] = score
    bf[f'{PREFIX}COMORB_weighted_x_age'] = age * score
    bf[f'{PREFIX}COMORB_weighted_high'] = (score >= 5).astype(int)
    bf[f'{PREFIX}COMORB_weighted_x_haem'] = score * haem_any
    bf[f'{PREFIX}COMORB_weighted_x_smoking'] = score * smoke_any

    # ══════════════════════════════════════════════════════════
    # BLOCK 18: SYMPTOM BURST (≥3 distinct symptom categories in 30-day rolling window)
    # ══════════════════════════════════════════════════════════
    obs_dated = clin[clin['CATEGORY'].isin(cfg.SYMPTOM_CATEGORIES) & clin['EVENT_DATE'].notna()].copy()
    obs_dated['EVENT_DATE'] = pd.to_datetime(obs_dated['EVENT_DATE'], errors='coerce')

    def _max_symptoms_in_30d(group):
        if len(group) < 2:
            return group['CATEGORY'].nunique()
        g = group[['EVENT_DATE', 'CATEGORY']].drop_duplicates().sort_values('EVENT_DATE')
        max_count = 1
        n = len(g)
        i = 0
        for j in range(n):
            while (g['EVENT_DATE'].iloc[j] - g['EVENT_DATE'].iloc[i]).days > 30:
                i += 1
            window_cats = g['CATEGORY'].iloc[i:j + 1].nunique()
            if window_cats > max_count:
                max_count = window_cats
        return max_count

    if len(obs_dated) > 0:
        burst = obs_dated.groupby('PATIENT_GUID').apply(_max_symptoms_in_30d)
        bf[f'{PREFIX}SYMPTOM_max_in_30d'] = burst.reindex(patients, fill_value=0).astype(int)
    else:
        bf[f'{PREFIX}SYMPTOM_max_in_30d'] = 0
    bf[f'{PREFIX}SYMPTOM_burst_3plus'] = (bf[f'{PREFIX}SYMPTOM_max_in_30d'] >= 3).astype(int)
    bf[f'{PREFIX}SYMPTOM_burst_4plus'] = (bf[f'{PREFIX}SYMPTOM_max_in_30d'] >= 4).astype(int)

    # ── Cleanup ───────────────────────────────────────────────
    bf = bf.fillna(0).replace([np.inf, -np.inf], 0)

    # Drop constant columns (zero variance — useless for modeling)
    nunique = bf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        bf = bf.drop(columns=constant)
        logger.info(f"  Removed {len(constant)} constant columns")

    logger.info(f"  Bladder-specific features: {bf.shape[1]}")
    return bf
