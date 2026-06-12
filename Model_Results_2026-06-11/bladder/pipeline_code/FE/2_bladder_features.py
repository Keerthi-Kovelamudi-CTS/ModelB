# ═══════════════════════════════════════════════════════════════
# BLADDER CANCER — CANCER-SPECIFIC FEATURES (Step 4d)
# Hand-engineered features for bladder prediction:
#   - Haematuria dynamics (the dominant presenting sign)
#   - Smoking exposure (strongest modifiable bladder-cancer risk factor)
#   - Urinary symptoms (dysuria, recurrent UTI, LUTS) + urine lab abnormalities
#   - Loin / suprapubic / back pain
#   - Constitutional decline (weight loss, fatigue, night sweats, anaemia)
#   - Anaemia from chronic blood loss (low Hb)
#   - Previous cancer (field cancerisation)
#   - Treatment flags (UTI antibiotics, iron, pioglitazone, antispasmodics, catheter)
#   - Composite bladder-cancer risk-surrogate score
#   - Generic temporal / age / engagement scaffolding (shared with prostate/breast)
#
# NOTE: pioglitazone is a recognised drug-associated bladder-cancer risk factor —
# it is featurised explicitly.
#
# This is the ONLY file that changes per cancer type (besides config.py).
# All other pipeline files are generic.
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
    """Build bladder-specific hand-engineered features."""
    if cfg is None: cfg = config
    PREFIX = cfg.PREFIX

    logger.info(f"  BLADDER-SPECIFIC FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    pf = pd.DataFrame(index=patient_list)
    pf.index.name = 'PATIENT_GUID'

    obs_AB = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    obs_A = obs_AB[obs_AB['TIME_WINDOW'] == 'A']
    obs_B = obs_AB[obs_AB['TIME_WINDOW'] == 'B']

    age = existing_fm['AGE_AT_INDEX'].reindex(pf.index).fillna(65)

    # ── Helpers ─────────────────────────────────────────────
    def has_cat(df, cat):
        has = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0
        return has.reindex(pf.index, fill_value=False).astype(int)

    def count_cat(df, cat):
        cnt = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size()
        return cnt.reindex(pf.index, fill_value=0).astype(int)

    def has_any_cat(df, cats):
        s = pd.Series(0, index=pf.index)
        for c in cats:
            s = s | has_cat(df, c)
        return s.astype(int)

    m = lambda s, default=0: s.reindex(pf.index).fillna(default)

    def val_series(cat, agg='last', df=obs_AB):
        sub = df[(df['CATEGORY'] == cat) & df['VALUE'].notna()].sort_values('EVENT_DATE')
        if sub.empty:
            return pd.Series(np.nan, index=pf.index)
        g = sub.groupby('PATIENT_GUID')['VALUE']
        s = {'last': g.last, 'first': g.first, 'max': g.max,
             'min': g.min, 'mean': g.mean}[agg]()
        return s.reindex(pf.index)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: HAEMATURIA DYNAMICS  (the dominant presenting sign)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 1: Haematuria dynamics...")
    has_haem   = has_cat(obs_AB, 'HAEMATURIA')
    n_haem     = count_cat(obs_AB, 'HAEMATURIA')
    has_haem_B = has_cat(obs_B, 'HAEMATURIA')
    has_haem_A = has_cat(obs_A, 'HAEMATURIA')
    has_urine_abn = has_cat(obs_AB, 'URINE_LAB_ABNORMALITIES')   # microscopic-haematuria proxy

    pf[f'{PREFIX}SX_has_haematuria']        = has_haem
    pf[f'{PREFIX}SX_haematuria_count']      = n_haem
    pf[f'{PREFIX}SX_haematuria_recurrent']  = (n_haem >= 2).astype(int)
    pf[f'{PREFIX}SX_haematuria_persistent'] = (n_haem >= 3).astype(int)
    pf[f'{PREFIX}SX_haematuria_in_B']       = has_haem_B
    pf[f'{PREFIX}SX_new_haematuria_in_B']   = ((has_haem_B == 1) & (has_haem_A == 0)).astype(int)
    pf[f'{PREFIX}SX_has_urine_abnormality'] = has_urine_abn
    pf[f'{PREFIX}SX_haematuria_or_urineabn'] = ((has_haem == 1) | (has_urine_abn == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: SMOKING EXPOSURE  (strongest modifiable BC risk factor)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 2: Smoking exposure...")
    has_current_smoker = has_cat(obs_AB, 'CURRENT_SMOKER')
    has_ex_smoker      = has_cat(obs_AB, 'EX_SMOKER')
    has_cess_refused   = has_cat(obs_AB, 'SMOKING_CESSATION_REFUSED')
    has_cess_med       = has_cat(med_AB, 'SMOKING_CESSATION_MEDS')
    pf[f'{PREFIX}RF_current_smoker']  = has_current_smoker
    pf[f'{PREFIX}RF_ex_smoker']       = has_ex_smoker
    pf[f'{PREFIX}RF_ever_smoker']     = ((has_current_smoker == 1) | (has_ex_smoker == 1)).astype(int)
    pf[f'{PREFIX}RF_cessation_refused'] = has_cess_refused
    pf[f'{PREFIX}RF_smoking_burden']  = (has_current_smoker + has_ex_smoker
                                         + has_cess_refused + has_cess_med).astype(int)
    pf[f'{PREFIX}RF_active_smoker_x_age50'] = ((has_current_smoker == 1) & (age >= 50)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: URINARY SYMPTOMS (dysuria / recurrent UTI / LUTS)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 3: Urinary symptoms...")
    has_dysuria  = has_cat(obs_AB, 'DYSURIA')
    has_rec_uti  = has_cat(obs_AB, 'RECURRENT_UTI')
    n_rec_uti    = count_cat(obs_AB, 'RECURRENT_UTI')
    has_luts     = has_cat(obs_AB, 'LUTS')
    has_urine_inv = has_cat(obs_AB, 'URINE_INVESTIGATIONS')
    has_uro_cond = has_cat(obs_AB, 'UROLOGICAL_CONDITIONS')
    pf[f'{PREFIX}SX_has_dysuria']        = has_dysuria
    pf[f'{PREFIX}SX_has_recurrent_uti']  = has_rec_uti
    pf[f'{PREFIX}SX_recurrent_uti_multi'] = (n_rec_uti >= 2).astype(int)
    pf[f'{PREFIX}SX_has_luts']           = has_luts
    pf[f'{PREFIX}DX_has_urine_invest']   = has_urine_inv
    pf[f'{PREFIX}RF_has_urological_cond'] = has_uro_cond
    urinary_burden = (has_haem + has_dysuria + has_rec_uti + has_luts).astype(int)
    pf[f'{PREFIX}SX_urinary_burden']  = urinary_burden
    pf[f'{PREFIX}SX_urinary_2plus']   = (urinary_burden >= 2).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: PAIN (loin / suprapubic / back)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 4: Pain patterns...")
    has_loin     = has_cat(obs_AB, 'LOIN_PAIN')
    has_suprapub = has_cat(obs_AB, 'SUPRAPUBIC_PAIN')
    has_back     = has_cat(obs_AB, 'BACK_PAIN')
    pf[f'{PREFIX}SX_has_loin_pain']     = has_loin
    pf[f'{PREFIX}SX_has_suprapubic_pain'] = has_suprapub
    pf[f'{PREFIX}SX_has_back_pain']     = has_back
    pf[f'{PREFIX}SX_pain_burden']       = (has_loin + has_suprapub + has_back).astype(int)
    pf[f'{PREFIX}SX_loin_x_haematuria'] = ((has_loin == 1) & (has_haem == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: CONSTITUTIONAL / SYSTEMIC DECLINE
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 5: Constitutional decline...")
    has_weight_loss = has_cat(obs_AB, 'WEIGHT_LOSS')
    has_fatigue     = has_cat(obs_AB, 'FATIGUE')
    has_night_sweat = has_cat(obs_AB, 'NIGHT_SWEATS')
    has_appetite    = has_cat(obs_AB, 'APPETITE_LOSS')
    has_frailty     = has_cat(obs_AB, 'FRAILTY')
    pf[f'{PREFIX}RF_has_weight_loss']   = has_weight_loss
    pf[f'{PREFIX}RF_has_fatigue']       = has_fatigue
    pf[f'{PREFIX}RF_has_night_sweats']  = has_night_sweat
    pf[f'{PREFIX}RF_has_appetite_loss'] = has_appetite
    pf[f'{PREFIX}RF_has_frailty']       = has_frailty
    constit = (has_weight_loss + has_fatigue + has_night_sweat + has_appetite).astype(int)
    pf[f'{PREFIX}RF_constitutional_load'] = constit
    pf[f'{PREFIX}RF_constitutional_2plus'] = (constit >= 2).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: ANAEMIA (chronic blood loss → low Hb)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 6: Anaemia signals...")
    has_anaemia_dx = has_cat(obs_AB, 'ANAEMIA_DX')
    hb_min  = val_series('HAEMATOLOGY', 'min')
    hb_last = val_series('HAEMATOLOGY', 'last')
    pf[f'{PREFIX}RF_has_anaemia_dx']  = has_anaemia_dx
    pf[f'{PREFIX}LAB_hb_min']         = hb_min.fillna(-1).astype(float)
    pf[f'{PREFIX}LAB_hb_last']        = hb_last.fillna(-1).astype(float)
    pf[f'{PREFIX}LAB_anaemia_proxy']  = ((hb_min > 0) & (hb_min < 110)).fillna(False).astype(int)
    pf[f'{PREFIX}LAB_severe_anaemia'] = ((hb_min > 0) & (hb_min < 90)).fillna(False).astype(int)
    pf[f'{PREFIX}RF_anaemia_any']     = ((has_anaemia_dx == 1) | (pf[f'{PREFIX}LAB_anaemia_proxy'] == 1)).astype(int)
    pf[f'{PREFIX}RF_anaemia_x_haematuria'] = ((pf[f'{PREFIX}RF_anaemia_any'] == 1) & (has_haem == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: PREVIOUS CANCER (field cancerisation)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 7: Previous-cancer field effect...")
    has_prev_cancer = has_cat(obs_AB, 'PREVIOUS_CANCER')
    pf[f'{PREFIX}RF_has_previous_cancer'] = has_prev_cancer
    pf[f'{PREFIX}RF_prevcancer_x_haematuria'] = ((has_prev_cancer == 1) & (has_haem == 1)).astype(int)
    pf[f'{PREFIX}RF_prevcancer_x_smoker']     = ((has_prev_cancer == 1) & (pf[f'{PREFIX}RF_ever_smoker'] == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: MEDICATION FLAGS (incl. pioglitazone risk-med)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 8: Medication flags...")
    has_pioglitazone = has_cat(med_AB, 'PIOGLITAZONE')
    has_uti_abx      = has_cat(med_AB, 'UTI_ANTIBIOTICS')
    n_uti_abx        = count_cat(med_AB, 'UTI_ANTIBIOTICS')
    has_iron         = has_cat(med_AB, 'IRON_SUPPLEMENTS')
    has_antispasm    = has_cat(med_AB, 'BLADDER_ANTISPASMODICS')
    has_luts_med     = has_cat(med_AB, 'LUTS_MEDICATIONS')
    has_opioid       = has_cat(med_AB, 'OPIOID_ANALGESICS')
    has_haemostatic  = has_cat(med_AB, 'HAEMOSTATIC')
    has_catheter_med = has_any_cat(med_AB, ['CATHETER_MAINTENANCE', 'CATHETER_SUPPLIES_CATHETERS',
                                            'CATHETER_SUPPLIES_LEG_BAGS', 'CATHETER_SUPPLIES_NIGHT_BAGS'])
    pf[f'{PREFIX}TX_on_pioglitazone']  = has_pioglitazone          # drug-associated BC risk
    pf[f'{PREFIX}TX_on_uti_abx']       = has_uti_abx
    pf[f'{PREFIX}TX_recurrent_uti_abx'] = (n_uti_abx >= 3).astype(int)
    pf[f'{PREFIX}TX_on_iron']          = has_iron
    pf[f'{PREFIX}TX_on_antispasmodic'] = has_antispasm
    pf[f'{PREFIX}TX_on_luts_med']      = has_luts_med
    pf[f'{PREFIX}TX_on_opioid']        = has_opioid
    pf[f'{PREFIX}TX_on_haemostatic']   = has_haemostatic
    pf[f'{PREFIX}TX_on_catheter']      = has_catheter_med
    pf[f'{PREFIX}TX_med_burden']       = (has_uti_abx + has_iron + has_antispasm + has_luts_med
                                          + has_opioid + has_catheter_med).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: COMPOSITE BLADDER RISK-SURROGATE SCORE
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 9: Composite risk score...")
    risk = pd.Series(0.0, index=pf.index)
    # Age bands (bladder cancer peaks 60-80)
    risk = risk + (age >= 50).astype(int) * 0.5
    risk = risk + (age >= 60).astype(int) * 1.0
    risk = risk + (age >= 70).astype(int) * 1.0
    # Haematuria (dominant sign)
    risk = risk + has_haem * 2.5
    risk = risk + pf[f'{PREFIX}SX_haematuria_recurrent'] * 1.0
    risk = risk + has_urine_abn * 1.0
    # Smoking (dominant risk factor)
    risk = risk + has_current_smoker * 1.5
    risk = risk + has_ex_smoker * 0.75
    # Other urinary
    risk = risk + has_rec_uti * 0.5
    risk = risk + has_dysuria * 0.3
    risk = risk + has_loin * 0.3
    # Constitutional + anaemia (advanced disease)
    risk = risk + has_weight_loss * 1.0
    risk = risk + pf[f'{PREFIX}RF_anaemia_any'] * 1.0
    risk = risk + pf[f'{PREFIX}RF_constitutional_2plus'] * 0.5
    # Drug + prior-cancer risk
    risk = risk + has_pioglitazone * 0.5
    risk = risk + has_prev_cancer * 1.0

    pf[f'{PREFIX}RISK_score']        = risk.astype(float)
    pf[f'{PREFIX}RISK_low']          = (risk < 2.0).astype(int)
    pf[f'{PREFIX}RISK_intermediate'] = ((risk >= 2.0) & (risk < 4.0)).astype(int)
    pf[f'{PREFIX}RISK_high']         = (risk >= 4.0).astype(int)
    pf[f'{PREFIX}RISK_very_high']    = (risk >= 6.0).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: AGE × CLINICAL-STATE INTERACTIONS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 10: Age × clinical interactions...")
    age50 = (age >= 50).astype(int)
    age60 = (age >= 60).astype(int)
    age70 = (age >= 70).astype(int)
    pf[f'{PREFIX}AGEX_haem_x_age50']      = (has_haem * age50)
    pf[f'{PREFIX}AGEX_haem_x_age60']      = (has_haem * age60)
    pf[f'{PREFIX}AGEX_haem_x_age70']      = (has_haem * age70)
    pf[f'{PREFIX}AGEX_smoker_x_age60']    = (has_current_smoker * age60)
    pf[f'{PREFIX}AGEX_rec_uti_x_age60']   = (has_rec_uti * age60)
    pf[f'{PREFIX}AGEX_weightloss_x_age60'] = (has_weight_loss * age60)
    pf[f'{PREFIX}AGEX_anaemia_x_age60']   = (pf[f'{PREFIX}RF_anaemia_any'] * age60)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: TRIPLE INTERACTIONS (non-linear combos)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 11: Triple interactions...")
    pf[f'{PREFIX}TRI_haem_smoker_age60']    = (has_haem * has_current_smoker * age60)
    pf[f'{PREFIX}TRI_haem_anaemia_age60']   = (has_haem * pf[f'{PREFIX}RF_anaemia_any'] * age60)
    pf[f'{PREFIX}TRI_haem_weightloss_age60'] = (has_haem * has_weight_loss * age60)
    pf[f'{PREFIX}TRI_haem_eversmoker_recurrent'] = (has_haem * pf[f'{PREFIX}RF_ever_smoker'] * pf[f'{PREFIX}SX_haematuria_recurrent'])
    pf[f'{PREFIX}TRI_haem_prevcancer_age60'] = (has_haem * has_prev_cancer * age60)

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: RECENT-WINDOW (B) SYMPTOM BURST
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 12: Recent-window symptom burst...")
    dys_B  = has_cat(obs_B, 'DYSURIA')
    uti_B  = has_cat(obs_B, 'RECURRENT_UTI')
    wl_B   = has_cat(obs_B, 'WEIGHT_LOSS')
    sx_B = (has_haem_B + dys_B + uti_B + wl_B).astype(int)
    pf[f'{PREFIX}BURST_sx_in_B']       = (sx_B >= 1).astype(int)
    pf[f'{PREFIX}BURST_multi_sx_in_B'] = (sx_B >= 2).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: SEQUENTIAL + FIRST-EVENT-DATE PATTERNS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 13: Sequential + first-event patterns...")
    def first_event(cat, df=obs_AB):
        s = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        return s.reindex(pf.index, fill_value=np.nan)

    first_haem = first_event('HAEMATURIA')
    first_uti  = first_event('RECURRENT_UTI')
    first_wl   = first_event('WEIGHT_LOSS')
    # Recurrent UTI → haematuria within 12mo (UTI work-up uncovering tumour)
    pf[f'{PREFIX}SEQ_uti_then_haem_12m'] = (
        ((first_uti - first_haem) >= 0) & ((first_uti - first_haem) <= 12)
    ).fillna(False).astype(int)
    pf[f'{PREFIX}FIRST_haem_mo_before']  = first_haem.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_wl_mo_before']    = first_wl.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_haem_within_12mo'] = (first_haem <= 12).fillna(False).astype(int)
    pf[f'{PREFIX}FIRST_haem_within_6mo']  = (first_haem <= 6).fillna(False).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: PATIENT-LEVEL Z-SCORE LABS (engagement-adjusted)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 14: Patient z-score labs...")
    def patient_zscore(cat, agg='mean'):
        df_v = obs_AB[(obs_AB['CATEGORY'] == cat) & obs_AB['VALUE'].notna()].copy()
        if df_v.empty:
            return pd.Series(0.0, index=pf.index)
        if agg == 'mean':
            per_pt = df_v.groupby('PATIENT_GUID')['VALUE'].mean()
        elif agg == 'min':
            per_pt = df_v.groupby('PATIENT_GUID')['VALUE'].min()
        else:
            per_pt = df_v.groupby('PATIENT_GUID')['VALUE'].last()
        if per_pt.std() == 0 or pd.isna(per_pt.std()):
            return pd.Series(0.0, index=pf.index)
        z = (per_pt - per_pt.mean()) / per_pt.std()
        return z.reindex(pf.index, fill_value=0.0).astype(float)

    pf[f'{PREFIX}Z_hb_min']      = patient_zscore('HAEMATOLOGY', 'min')
    pf[f'{PREFIX}Z_crp_mean']    = patient_zscore('INFLAMMATORY', 'mean')
    pf[f'{PREFIX}Z_hb_low']      = (pf[f'{PREFIX}Z_hb_min'] < -1.5).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 15: PER-CATEGORY VELOCITY (B - A counts)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 15: Per-category velocity...")
    def cat_velocity(cat, df_AB=obs_AB):
        nA = df_AB[(df_AB['CATEGORY']==cat) & (df_AB['TIME_WINDOW']=='A')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        nB = df_AB[(df_AB['CATEGORY']==cat) & (df_AB['TIME_WINDOW']=='B')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        return (nB - nA).astype(int)

    for cat in ['HAEMATURIA','DYSURIA','RECURRENT_UTI','LUTS','LOIN_PAIN',
                'SUPRAPUBIC_PAIN','WEIGHT_LOSS','URINE_LAB_ABNORMALITIES']:
        pf[f'{PREFIX}VEL_{cat}_B_minus_A'] = cat_velocity(cat)
    pf[f'{PREFIX}VEL_uti_abx_B_minus_A'] = cat_velocity('UTI_ANTIBIOTICS', med_AB)

    # ══════════════════════════════════════════════════════════
    # BLOCK 16: TIME-DECAY INTENSITY + ACCELERATION
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 16: Time-decay intensity + acceleration...")
    KEY_CATS = ['HAEMATURIA', 'DYSURIA', 'RECURRENT_UTI', 'LUTS', 'LOIN_PAIN',
                'SUPRAPUBIC_PAIN', 'WEIGHT_LOSS', 'ANAEMIA_DX', 'FATIGUE',
                'URINE_LAB_ABNORMALITIES', 'PREVIOUS_CANCER', 'CURRENT_SMOKER']
    HALFLIFE_MO = 12.0
    def cat_decay(cat, df_AB=obs_AB):
        sub = df_AB[df_AB['CATEGORY'] == cat]
        if len(sub) == 0:
            return pd.Series(0.0, index=pf.index)
        w = np.exp(-sub['MONTHS_BEFORE_INDEX'].astype(float).clip(lower=0) / HALFLIFE_MO)
        return w.groupby(sub['PATIENT_GUID']).sum().reindex(pf.index, fill_value=0.0)
    for cat in KEY_CATS:
        pf[f'{PREFIX}DECAY_{cat}'] = cat_decay(cat).astype(float)
    pf[f'{PREFIX}DECAY_UTI_ANTIBIOTICS'] = cat_decay('UTI_ANTIBIOTICS', med_AB).astype(float)

    def cat_bin(cat, lo, hi, df_AB=obs_AB):
        sub = df_AB[(df_AB['CATEGORY'] == cat) &
                    (df_AB['MONTHS_BEFORE_INDEX'] >= lo) & (df_AB['MONTHS_BEFORE_INDEX'] < hi)]
        return sub.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    for cat in ['HAEMATURIA', 'DYSURIA', 'RECURRENT_UTI', 'LOIN_PAIN', 'WEIGHT_LOSS']:
        rec = cat_bin(cat, 1, 15); mid = cat_bin(cat, 15, 30); ear = cat_bin(cat, 30, 60)
        pf[f'{PREFIX}ACCEL_{cat}'] = ((rec - mid) - (mid - ear)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: ENGAGEMENT-NORMALIZED SIGNAL
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 17: Engagement-normalized signal...")
    total_obs_events = obs_AB.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0).clip(lower=1)
    total_med_events = med_AB.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0).clip(lower=1)

    def cat_count(cat, df_AB=obs_AB):
        return df_AB[df_AB['CATEGORY']==cat].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)

    pf[f'{PREFIX}NORM_haematuria_per_total'] = (cat_count('HAEMATURIA') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_uti_per_total']        = (cat_count('RECURRENT_UTI') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_dysuria_per_total']    = (cat_count('DYSURIA') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_uti_abx_per_med']      = (cat_count('UTI_ANTIBIOTICS', med_AB) / total_med_events).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 18: CROSS-CATEGORY CO-OCCURRENCE FLAGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 18: Cross-category co-occurrence...")
    bladder_indicators = (has_haem + has_dysuria + has_rec_uti + has_current_smoker
                          + pf[f'{PREFIX}RF_anaemia_any'] + has_weight_loss + has_prev_cancer).astype(int)
    pf[f'{PREFIX}COOC_bladder_indicators']  = bladder_indicators
    pf[f'{PREFIX}COOC_2plus_indicators']    = (bladder_indicators >= 2).astype(int)
    pf[f'{PREFIX}COOC_3plus_indicators']    = (bladder_indicators >= 3).astype(int)
    pf[f'{PREFIX}COOC_haem_and_smoker']     = ((has_haem == 1) & (pf[f'{PREFIX}RF_ever_smoker'] == 1)).astype(int)
    pf[f'{PREFIX}COOC_haem_and_anaemia']    = ((has_haem == 1) & (pf[f'{PREFIX}RF_anaemia_any'] == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 19: TIME-SINCE-LAST-EVENT (recency per category)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 19: Time-since-last-event recency...")
    def last_event(cat, df_AB=obs_AB):
        s = df_AB[df_AB['CATEGORY']==cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        return s.reindex(pf.index, fill_value=np.nan)

    last_haem = last_event('HAEMATURIA')
    last_uti  = last_event('RECURRENT_UTI')
    last_wl   = last_event('WEIGHT_LOSS')
    pf[f'{PREFIX}RECENT_haem_in_6mo']    = (last_haem <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_uti_in_6mo']     = (last_uti <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_wl_in_6mo']      = (last_wl <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_haem_gap_mo']    = last_haem.fillna(60).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 20: PER-KEY-CATEGORY TEMPORAL GRANULARITY
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 20: Per-category temporal granularity...")
    for cat in KEY_CATS:
        sub = obs_AB[obs_AB['CATEGORY'] == cat]
        if sub.empty:
            continue
        ages = sub.groupby('PATIENT_GUID')['EVENT_AGE'].agg(['min','max','mean'])
        pf[f'{PREFIX}EA_{cat}_min_age']  = ages['min'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}EA_{cat}_max_age']  = ages['max'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}EA_{cat}_age_span'] = (ages['max'] - ages['min']).reindex(pf.index, fill_value=0).astype(float)
        mbs = sub.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min','max'])
        pf[f'{PREFIX}EA_{cat}_first_mo_before'] = mbs['max'].reindex(pf.index, fill_value=999).astype(float)
        pf[f'{PREFIX}EA_{cat}_last_mo_before']  = mbs['min'].reindex(pf.index, fill_value=999).astype(float)
        nA = sub[sub['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        nB = sub[sub['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}EA_{cat}_is_worsening']   = (nB > nA).astype(int)
        pf[f'{PREFIX}EA_{cat}_is_only_recent'] = ((nB > 0) & (nA == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 21: INTERVAL + TREND SLOPE + LAST-18MO PER KEY CAT
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 21: Interval + trend + last-18mo features...")
    for cat in KEY_CATS:
        sub = obs_AB[obs_AB['CATEGORY'] == cat].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(['PATIENT_GUID', 'MONTHS_BEFORE_INDEX'])
        sub['_days_before'] = sub['MONTHS_BEFORE_INDEX'] * 30.4375
        sub['_interval'] = sub.groupby('PATIENT_GUID')['_days_before'].diff().abs()
        ivl = sub.dropna(subset=['_interval']).groupby('PATIENT_GUID')['_interval'].agg(['mean','median'])
        pf[f'{PREFIX}IVL_{cat}_mean_days']   = ivl['mean'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}IVL_{cat}_median_days'] = ivl['median'].reindex(pf.index, fill_value=-1).astype(float)

        ev_by_month = (sub.groupby(['PATIENT_GUID', 'MONTHS_BEFORE_INDEX']).size().reset_index(name='n'))
        def _slope(grp):
            if len(grp) < 2: return 0.0
            x = grp['MONTHS_BEFORE_INDEX'].astype(float).values
            y = grp['n'].astype(float).values
            if x.std() == 0: return 0.0
            return float(np.polyfit(x, y, 1)[0])
        slope = ev_by_month.groupby('PATIENT_GUID').apply(_slope)
        pf[f'{PREFIX}TRD_{cat}_slope']     = slope.reindex(pf.index, fill_value=0.0).astype(float)
        pf[f'{PREFIX}TRD_{cat}_worsening'] = (slope < 0).reindex(pf.index, fill_value=False).astype(int)

        first_half  = sub[sub['MONTHS_BEFORE_INDEX'] >= 30].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        second_half = sub[sub['MONTHS_BEFORE_INDEX'] < 30].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}HALF_{cat}_freq_ratio'] = (second_half / (first_half + 1.0)).astype(float)

        last18 = sub[sub['MONTHS_BEFORE_INDEX'] <= 18]
        n18 = last18.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}L18M_{cat}_n']   = n18.astype(int)
        pf[f'{PREFIX}L18M_{cat}_has'] = (n18 >= 1).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 22: AGE POLYNOMIALS + BANDS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 22: Age polynomials + bands...")
    age_f = age.astype(float)
    pf[f'{PREFIX}AGEP_age_sq']   = (age_f ** 2).astype(float)
    pf[f'{PREFIX}AGEP_age_cu']   = (age_f ** 3 / 100.0).astype(float)
    pf[f'{PREFIX}AGEP_age_log']  = np.log1p(age_f).astype(float)
    pf[f'{PREFIX}AGEP_age_sqrt'] = np.sqrt(age_f).astype(float)
    for lo, hi in [(30,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,99)]:
        pf[f'{PREFIX}AGEB_band_{lo}_{hi}'] = ((age_f >= lo) & (age_f < hi)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 23: PATIENT SUMMARY COUNTS + ENGAGEMENT TIMING
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 23: Summary counts + engagement timing...")
    n_distinct_obs = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    n_distinct_med = med_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}SUM_distinct_obs_cats'] = n_distinct_obs.astype(int)
    pf[f'{PREFIX}SUM_distinct_med_cats'] = n_distinct_med.astype(int)
    pf[f'{PREFIX}SUM_polypharm']         = (n_distinct_med >= 5).astype(int)
    pf[f'{PREFIX}LOG_total_obs_events']  = np.log1p(total_obs_events).astype(float)
    pf[f'{PREFIX}LOG_total_med_events']  = np.log1p(total_med_events).astype(float)
    pf[f'{PREFIX}LOG_n_distinct_cats']   = np.log1p(n_distinct_obs).astype(float)

    all_mb = obs_AB.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min','max'])
    record_span_mo = (all_mb['max'] - all_mb['min']).reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}ENG_record_span_mo']    = record_span_mo.astype(float)
    pf[f'{PREFIX}ENG_long_record_5y']    = (record_span_mo >= 60).astype(int)
    pf[f'{PREFIX}ENG_events_per_year']   = (total_obs_events / record_span_mo.clip(lower=1) * 12.0).astype(float)
    last6 = obs_AB[obs_AB['MONTHS_BEFORE_INDEX'] <= 6].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}ENG_recent_activity_6mo'] = last6.astype(int)
    pf[f'{PREFIX}ENG_active_recent']     = (last6 >= 5).astype(int)
    pf[f'{PREFIX}ENG_high_engagement']   = (n_distinct_obs >= 10).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 24: LAB-VALUE TRAJECTORY FLAGS (recent vs historical)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 24: Lab trajectory direction flags...")
    for lab_cat in ['HAEMATOLOGY', 'RENAL', 'INFLAMMATORY', 'METABOLIC']:
        sub = obs_AB[(obs_AB['CATEGORY']==lab_cat) & obs_AB['VALUE'].notna()].copy()
        if sub.empty:
            continue
        avgA = sub[sub['TIME_WINDOW']=='A'].groupby('PATIENT_GUID')['VALUE'].mean().reindex(pf.index, fill_value=np.nan)
        avgB = sub[sub['TIME_WINDOW']=='B'].groupby('PATIENT_GUID')['VALUE'].mean().reindex(pf.index, fill_value=np.nan)
        diff = (avgB - avgA)
        pf[f'{PREFIX}TRJ_{lab_cat}_B_minus_A'] = diff.fillna(0).astype(float)
        # For haemoglobin a FALLING value is the worsening direction
        if lab_cat == 'HAEMATOLOGY':
            pf[f'{PREFIX}TRJ_{lab_cat}_worsening'] = (diff < 0).fillna(False).astype(int)
        else:
            pf[f'{PREFIX}TRJ_{lab_cat}_worsening'] = (diff > 0).fillna(False).astype(int)
        pf[f'{PREFIX}TRJ_{lab_cat}_min'] = sub.groupby('PATIENT_GUID')['VALUE'].min().reindex(pf.index, fill_value=np.nan).fillna(-1).astype(float)

    # ── Done ─────────────────────────────────────────────────
    n_features = sum(1 for c in pf.columns if c.startswith(PREFIX))
    logger.info(f"  ✓ Bladder-specific features built: {n_features} columns")
    return pf
