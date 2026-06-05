# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — CANCER-SPECIFIC FEATURES
# NICE cardinal symptoms, mimic patterns (IBS/UTI/PMB), cyst/mass/ascites
# interactions, risk factors (age + family history + HRT), lab thresholds,
# diagnostic pathway gaps, treatment patterns.
#
# Ported from Ovarian_Cancer/2_Feature_Engineering/4_Feature_engineering.py
# (build_ovarian_features) and adapted to the new config-driven pipeline.
# Plus generic structural blocks (8-18) shared with Bladder/Prostate:
#   trifecta, velocity, NEW-in-B-only, AGE×, triples, comorbidity,
#   FIRST/RECENT event-date, sequential X→Y, Charlson-lite, symptom burst.
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
    Ovarian-specific features:
      - NICE cardinal symptoms (bloating / early satiety / pelvic pain / urinary)
      - Mimic patterns (IBS, UTI, postmeno bleeding, classic triad)
      - Cyst / mass / ascites interactions
      - Risk factors (peak-age, family history, HRT, OCP)
      - Lab thresholds (Hb, CRP, albumin, platelets, eGFR, ESR, ALP, LDH)
      - Diagnostic pathway (symptom-to-imaging gap)
      - Treatment patterns (pain escalation, GI multi, iron, VTE)
      - Plus generic structural blocks 8-18 (trifecta, velocity, AGE×, etc.)
    """
    if cfg is None:
        cfg = config
    PREFIX = cfg.PREFIX  # 'OVAR_'

    logger.info(f"  OVARIAN-SPECIFIC FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patients = existing_fm.index
    ov = pd.DataFrame(index=patients)
    ov.index.name = 'PATIENT_GUID'

    obs_AB = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    obs_A = clin[clin['TIME_WINDOW'] == 'A'].copy()
    obs_B = clin[clin['TIME_WINDOW'] == 'B'].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy() if 'TIME_WINDOW' in med.columns else med
    lab_AB = obs_AB[obs_AB['VALUE'].notna()].copy()

    age = (existing_fm['AGE_AT_INDEX'].reindex(patients).fillna(55)
           if 'AGE_AT_INDEX' in existing_fm.columns
           else pd.Series(55.0, index=patients))

    # ── Helpers ──────────────────────────────────────────────────
    def _has(df, cat):
        return (df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0).reindex(patients, fill_value=False).astype(int)

    def _count(df, cat):
        return df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)

    def _has_med(category):
        return (med_AB[med_AB['CATEGORY'] == category].groupby('PATIENT_GUID').size() > 0).reindex(patients, fill_value=False).astype(int)

    def _count_med(category):
        return med_AB[med_AB['CATEGORY'] == category].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)

    def _lab_latest(term_pattern, category=None):
        sub = lab_AB
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last().reindex(patients)

    def _lab_first(term_pattern, category=None):
        sub = lab_AB
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first().reindex(patients)

    def _lab_max(term_pattern, category=None):
        sub = lab_AB
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.groupby('PATIENT_GUID')['VALUE'].max().reindex(patients)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: NICE CARDINAL SYMPTOMS
    # NICE recognises persistent (≥2 in 1mo) bloating / early satiety /
    # pelvic pain / urinary frequency in women ≥50 as ovarian red flags.
    # ══════════════════════════════════════════════════════════
    has_bloating = _has(obs_AB, 'ABDOMINAL_BLOATING')
    has_satiety = _has(obs_AB, 'EARLY_SATIETY')
    has_pain = _has(obs_AB, 'ABDOMINAL_PAIN')
    has_urinary = _has(obs_AB, 'URINARY_SYMPTOMS')

    ov[f'{PREFIX}NICE_bloating'] = has_bloating
    ov[f'{PREFIX}NICE_early_satiety'] = has_satiety
    ov[f'{PREFIX}NICE_pelvic_pain'] = has_pain
    ov[f'{PREFIX}NICE_urinary_frequency'] = has_urinary
    ov[f'{PREFIX}NICE_cardinal_count'] = has_bloating + has_satiety + has_pain + has_urinary
    ov[f'{PREFIX}NICE_2plus_cardinals'] = (ov[f'{PREFIX}NICE_cardinal_count'] >= 2).astype(int)
    ov[f'{PREFIX}NICE_3plus_cardinals'] = (ov[f'{PREFIX}NICE_cardinal_count'] >= 3).astype(int)

    bloat_B = _count(obs_B, 'ABDOMINAL_BLOATING')
    pain_B = _count(obs_B, 'ABDOMINAL_PAIN')
    urin_B = _count(obs_B, 'URINARY_SYMPTOMS')
    satiety_B = _count(obs_B, 'EARLY_SATIETY')
    ov[f'{PREFIX}NICE_persistent_bloating'] = (bloat_B >= 2).astype(int)
    ov[f'{PREFIX}NICE_persistent_pain'] = (pain_B >= 2).astype(int)
    ov[f'{PREFIX}NICE_persistent_urinary'] = (urin_B >= 2).astype(int)
    ov[f'{PREFIX}NICE_persistent_satiety'] = (satiety_B >= 2).astype(int)
    ov[f'{PREFIX}NICE_persistent_count'] = (
        ov[f'{PREFIX}NICE_persistent_bloating']
        + ov[f'{PREFIX}NICE_persistent_pain']
        + ov[f'{PREFIX}NICE_persistent_urinary']
        + ov[f'{PREFIX}NICE_persistent_satiety']
    )
    ov[f'{PREFIX}NICE_over50_with_cardinal'] = ((age >= 50) & (ov[f'{PREFIX}NICE_cardinal_count'] >= 1)).astype(int)
    ov[f'{PREFIX}NICE_over50_with_2plus'] = ((age >= 50) & (ov[f'{PREFIX}NICE_cardinal_count'] >= 2)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: MIMIC PATTERNS
    # IBS-mimic, UTI-mimic, postmenopausal bleeding, classic triad
    # ══════════════════════════════════════════════════════════
    has_gi = _has(obs_AB, 'GI_SYMPTOMS')
    has_inv = (_count(obs_AB, 'IMAGING') + _count(obs_AB, 'GYNAE_PROCEDURES') + _count(obs_AB, 'OTHER_PROCEDURES') > 0).astype(int)

    ov[f'{PREFIX}OV_ibs_mimic'] = ((has_gi == 1) & (has_bloating == 1) & (has_pain == 1)).astype(int)
    ov[f'{PREFIX}OV_ibs_mimic_no_inv'] = ((ov[f'{PREFIX}OV_ibs_mimic'] == 1) & (has_inv == 0)).astype(int)

    uti_abx = _count_med('UTI_ANTIBIOTICS')
    ov[f'{PREFIX}OV_uti_mimic'] = ((has_urinary == 1) & (uti_abx >= 1)).astype(int)
    ov[f'{PREFIX}OV_uti_mimic_recurrent'] = ((has_urinary == 1) & (uti_abx >= 2)).astype(int)

    has_bleed = _has(obs_AB, 'GYNAECOLOGICAL_BLEEDING')
    has_postmeno = _has(obs_AB, 'POSTMENOPAUSAL')
    ov[f'{PREFIX}OV_postmeno_bleeding'] = ((has_bleed == 1) & (age >= 55)).astype(int)
    ov[f'{PREFIX}OV_postmeno_bleeding_with_pain'] = ((ov[f'{PREFIX}OV_postmeno_bleeding'] == 1) & (has_pain == 1)).astype(int)
    ov[f'{PREFIX}OV_postmeno_coded_with_bleed'] = ((has_postmeno == 1) & (has_bleed == 1)).astype(int)

    ov[f'{PREFIX}OV_classic_triad'] = ((has_bloating == 1) & (has_pain == 1) & (has_urinary == 1)).astype(int)
    ov[f'{PREFIX}OV_classic_triad_over50'] = ((ov[f'{PREFIX}OV_classic_triad'] == 1) & (age >= 50)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: OVARIAN CYST / MASS / ASCITES
    # Cyst + cardinals = high suspicion. Mass + ascites = late-stage triad.
    # ══════════════════════════════════════════════════════════
    has_cyst = _has(obs_AB, 'OVARIAN_CYST')
    has_ascites = _has(obs_AB, 'ASCITES')
    has_mass = _has(obs_AB, 'ABDOMINAL_MASS')
    has_wt_loss = _has(obs_AB, 'WEIGHT_LOSS')
    has_fatigue = _has(obs_AB, 'FATIGUE')

    ov[f'{PREFIX}OV_cyst_present'] = has_cyst
    ov[f'{PREFIX}OV_cyst_with_pain'] = ((has_cyst == 1) & (has_pain == 1)).astype(int)
    ov[f'{PREFIX}OV_cyst_with_bloating'] = ((has_cyst == 1) & (has_bloating == 1)).astype(int)
    ov[f'{PREFIX}OV_cyst_with_gi'] = ((has_cyst == 1) & (has_gi == 1)).astype(int)
    ov[f'{PREFIX}OV_cyst_with_weightloss'] = ((has_cyst == 1) & (has_wt_loss == 1)).astype(int)
    ov[f'{PREFIX}OV_cyst_complex'] = ((has_cyst == 1) & (ov[f'{PREFIX}NICE_cardinal_count'] >= 2)).astype(int)

    ov[f'{PREFIX}OV_ascites'] = has_ascites
    ov[f'{PREFIX}OV_mass'] = has_mass
    ov[f'{PREFIX}OV_ascites_or_mass'] = ((has_ascites == 1) | (has_mass == 1)).astype(int)
    ov[f'{PREFIX}OV_ascites_and_mass'] = ((has_ascites == 1) & (has_mass == 1)).astype(int)
    ov[f'{PREFIX}OV_ascites_with_bloating'] = ((has_ascites == 1) & (has_bloating == 1)).astype(int)
    ov[f'{PREFIX}OV_mass_with_bloating'] = ((has_mass == 1) & (has_bloating == 1)).astype(int)
    ov[f'{PREFIX}OV_late_stage_triad'] = (
        (has_mass == 1) & (has_ascites == 1) & (has_wt_loss == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: RISK FACTORS
    # Peak-age window 55-75. Family history. HRT (risk). OCP (protective).
    # ══════════════════════════════════════════════════════════
    ov[f'{PREFIX}RF_peak_age'] = ((age >= 55) & (age <= 75)).astype(int)
    ov[f'{PREFIX}RF_high_risk_age'] = (age >= 60).astype(int)
    ov[f'{PREFIX}RF_elderly'] = (age >= 70).astype(int)

    has_fh = _has(obs_AB, 'FAMILY_HISTORY')
    ov[f'{PREFIX}RF_family_history'] = has_fh
    ov[f'{PREFIX}RF_fh_plus_symptoms'] = ((has_fh == 1) & (ov[f'{PREFIX}NICE_cardinal_count'] >= 1)).astype(int)
    ov[f'{PREFIX}RF_fh_plus_age'] = ((has_fh == 1) & (age >= 50)).astype(int)

    has_hrt_obs = _has(obs_AB, 'HRT')
    has_hrt_med = _has_med('HRT')
    ov[f'{PREFIX}RF_hrt_use'] = ((has_hrt_obs == 1) | (has_hrt_med == 1)).astype(int)
    ov[f'{PREFIX}RF_ocp_protective'] = _has_med('ORAL_CONTRACEPTIVE')
    ov[f'{PREFIX}RF_obesity'] = _has(obs_AB, 'OBESITY')
    ov[f'{PREFIX}RF_reproductive_history'] = _has(obs_AB, 'REPRODUCTIVE_HISTORY')

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: LAB THRESHOLD FEATURES
    # Hb (anaemia), CRP/ESR (inflammation), albumin (advanced disease),
    # platelets (thrombocytosis = paraneoplastic), eGFR (renal),
    # ALP (liver mets), LDH (paraneoplastic / mass burden).
    # ══════════════════════════════════════════════════════════

    # Haemoglobin
    hb_last = _lab_latest('haemoglobin|hemoglobin', 'HAEMATOLOGY')
    hb_first = _lab_first('haemoglobin|hemoglobin', 'HAEMATOLOGY')
    ov[f'{PREFIX}LAB_hb_latest'] = hb_last.fillna(0)
    ov[f'{PREFIX}LAB_hb_anaemic'] = (hb_last.fillna(999) < 120).astype(int)
    ov[f'{PREFIX}LAB_hb_severe_anaemia'] = (hb_last.fillna(999) < 100).astype(int)
    hb_decline = (hb_first - hb_last).fillna(0)
    ov[f'{PREFIX}LAB_hb_declining'] = (hb_decline > 5).astype(int)
    ov[f'{PREFIX}LAB_hb_anaemic_with_pain'] = ((ov[f'{PREFIX}LAB_hb_anaemic'] == 1) & (has_pain == 1)).astype(int)
    ov[f'{PREFIX}LAB_hb_anaemic_with_fatigue'] = ((ov[f'{PREFIX}LAB_hb_anaemic'] == 1) & (has_fatigue == 1)).astype(int)
    ov[f'{PREFIX}LAB_hb_anaemic_with_bloating'] = ((ov[f'{PREFIX}LAB_hb_anaemic'] == 1) & (has_bloating == 1)).astype(int)

    # CRP
    crp_last = _lab_latest(r'CRP|C[\s-]*reactive', 'INFLAMMATORY')
    crp_max = _lab_max(r'CRP|C[\s-]*reactive', 'INFLAMMATORY')
    ov[f'{PREFIX}LAB_crp_latest'] = crp_last.fillna(0)
    ov[f'{PREFIX}LAB_crp_elevated'] = (crp_last.fillna(0) > 10).astype(int)
    ov[f'{PREFIX}LAB_crp_high'] = (crp_max.fillna(0) > 30).astype(int)
    ov[f'{PREFIX}LAB_crp_with_pain'] = ((ov[f'{PREFIX}LAB_crp_elevated'] == 1) & (has_pain == 1)).astype(int)
    ov[f'{PREFIX}LAB_crp_with_bloating'] = ((ov[f'{PREFIX}LAB_crp_elevated'] == 1) & (has_bloating == 1)).astype(int)

    # Albumin
    alb_last = _lab_latest('albumin', 'METABOLIC')
    ov[f'{PREFIX}LAB_albumin_latest'] = alb_last.fillna(0)
    ov[f'{PREFIX}LAB_albumin_low'] = (alb_last.fillna(99) < 35).astype(int)
    ov[f'{PREFIX}LAB_albumin_low_with_wt_loss'] = ((ov[f'{PREFIX}LAB_albumin_low'] == 1) & (has_wt_loss == 1)).astype(int)
    ov[f'{PREFIX}LAB_albumin_low_with_ascites'] = ((ov[f'{PREFIX}LAB_albumin_low'] == 1) & (has_ascites == 1)).astype(int)

    # Platelets — thrombocytosis is a known paraneoplastic ovarian-cancer signal
    plt_last = _lab_latest(r'Platelet\s+count|platelets', 'HAEMATOLOGY')
    ov[f'{PREFIX}LAB_platelets_latest'] = plt_last.fillna(0)
    ov[f'{PREFIX}LAB_platelets_high'] = (plt_last.fillna(0) > 400).astype(int)
    ov[f'{PREFIX}LAB_platelets_very_high'] = (plt_last.fillna(0) > 600).astype(int)
    ov[f'{PREFIX}LAB_platelets_high_with_symptoms'] = (
        (ov[f'{PREFIX}LAB_platelets_high'] == 1) & (ov[f'{PREFIX}NICE_cardinal_count'] >= 1)
    ).astype(int)

    # eGFR (renal) — hydronephrosis from pelvic mass can drop eGFR
    egfr_last = _lab_latest(r'eGFR|glomerular\s+filtration', 'RENAL')
    egfr_first = _lab_first(r'eGFR|glomerular\s+filtration', 'RENAL')
    egfr_decline = (egfr_first - egfr_last).fillna(0)
    ov[f'{PREFIX}LAB_egfr_latest'] = egfr_last.fillna(0)
    ov[f'{PREFIX}LAB_egfr_low'] = (egfr_last.fillna(999) < 60).astype(int)
    ov[f'{PREFIX}LAB_egfr_declining'] = (egfr_decline > 10).astype(int)
    ov[f'{PREFIX}LAB_egfr_declining_with_urinary'] = ((ov[f'{PREFIX}LAB_egfr_declining'] == 1) & (has_urinary == 1)).astype(int)

    # ESR
    esr_last = _lab_latest(r'Erythrocyte\s+sedimentation|ESR', 'INFLAMMATORY')
    ov[f'{PREFIX}LAB_esr_elevated'] = (esr_last.fillna(0) > 30).astype(int)

    # ALP (liver mets)
    alp_last = _lab_latest(r'Alkaline\s+phosphatase|ALP', 'LIVER')
    ov[f'{PREFIX}LAB_alp_elevated'] = (alp_last.fillna(0) > 130).astype(int)

    # LDH (paraneoplastic / mass burden) — CANCER_MARKER category
    ldh_last = _lab_latest(r'lactate\s+dehydrogenase|LDH', 'CANCER_MARKER')
    ov[f'{PREFIX}LAB_ldh_latest'] = ldh_last.fillna(0)
    ov[f'{PREFIX}LAB_ldh_elevated'] = (ldh_last.fillna(0) > 250).astype(int)
    ov[f'{PREFIX}LAB_ldh_with_mass'] = ((ov[f'{PREFIX}LAB_ldh_elevated'] == 1) & (has_mass == 1)).astype(int)

    # Calcium (hypercalcaemia of malignancy)
    ca_last = _lab_latest(r'calcium', 'ELECTROLYTES')
    ov[f'{PREFIX}LAB_calcium_latest'] = ca_last.fillna(0)
    ov[f'{PREFIX}LAB_calcium_high'] = (ca_last.fillna(0) > 2.6).astype(int)

    # Bone-met / aggressive-disease trio: ALP↑ + Hb↓ + albumin↓
    bone_signals = (
        ov[f'{PREFIX}LAB_alp_elevated'].astype(int)
        + ov[f'{PREFIX}LAB_hb_anaemic'].astype(int)
        + ov[f'{PREFIX}LAB_albumin_low'].astype(int)
    )
    ov[f'{PREFIX}LAB_aggressive_trio'] = (bone_signals >= 3).astype(int)
    ov[f'{PREFIX}LAB_aggressive_pair'] = (bone_signals >= 2).astype(int)
    ov[f'{PREFIX}LAB_aggressive_score'] = bone_signals

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: DIAGNOSTIC PATHWAY
    # Cardinal symptoms but no imaging, or delayed imaging = red flag.
    # ══════════════════════════════════════════════════════════
    has_imaging = _has(obs_AB, 'IMAGING')
    has_gynae_proc = _has(obs_AB, 'GYNAE_PROCEDURES')
    ov[f'{PREFIX}DX_has_imaging'] = has_imaging
    ov[f'{PREFIX}DX_has_gynae_procedure'] = has_gynae_proc
    ov[f'{PREFIX}DX_symptoms_no_imaging'] = (
        (ov[f'{PREFIX}NICE_cardinal_count'] >= 2) & (has_imaging == 0)
    ).astype(int)

    sym_cats = ['ABDOMINAL_PAIN', 'ABDOMINAL_BLOATING', 'URINARY_SYMPTOMS', 'GI_SYMPTOMS', 'EARLY_SATIETY']
    sym_first = obs_AB[obs_AB['CATEGORY'].isin(sym_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    img_first = obs_AB[obs_AB['CATEGORY'] == 'IMAGING'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap = (img_first - sym_first).dt.days
    ov[f'{PREFIX}DX_symptom_to_imaging_days'] = gap.reindex(patients).fillna(-1)
    ov[f'{PREFIX}DX_imaging_delayed_60d'] = (ov[f'{PREFIX}DX_symptom_to_imaging_days'] > 60).astype(int)
    ov[f'{PREFIX}DX_imaging_delayed_120d'] = (ov[f'{PREFIX}DX_symptom_to_imaging_days'] > 120).astype(int)
    ov[f'{PREFIX}DX_imaging_delayed_180d'] = (ov[f'{PREFIX}DX_symptom_to_imaging_days'] > 180).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: TREATMENT PATTERNS
    # Pain escalation, repeat GI symptom-relief without dx, iron Rx for anaemia, VTE.
    # ══════════════════════════════════════════════════════════
    has_nsaid = _has_med('NSAIDS')
    has_opioid = _has_med('OPIOID_ANALGESICS')
    has_ppi = _has_med('PPI')
    has_antispasm = _has_med('GI_ANTISPASMODICS')
    has_laxative = _has_med('LAXATIVES')
    has_antiemetic = _has_med('ANTIEMETICS')
    has_iron = _has_med('IRON_SUPPLEMENTS')
    has_diuretic = _has_med('DIURETICS')

    ov[f'{PREFIX}TX_pain_with_opioid'] = ((has_pain == 1) & (has_opioid == 1)).astype(int)
    ov[f'{PREFIX}TX_pain_escalation'] = ((has_nsaid == 1) & (has_opioid == 1)).astype(int)
    ov[f'{PREFIX}TX_pain_escalation_with_cardinal'] = (
        (ov[f'{PREFIX}TX_pain_escalation'] == 1) & (ov[f'{PREFIX}NICE_cardinal_count'] >= 2)
    ).astype(int)

    gi_med_count = has_ppi + has_antispasm + has_laxative + has_antiemetic
    ov[f'{PREFIX}TX_gi_treated'] = (gi_med_count >= 1).astype(int)
    ov[f'{PREFIX}TX_gi_multi_treated'] = (gi_med_count >= 2).astype(int)
    ov[f'{PREFIX}TX_gi_treated_no_inv'] = ((ov[f'{PREFIX}TX_gi_treated'] == 1) & (has_inv == 0)).astype(int)
    ov[f'{PREFIX}TX_gi_treated_with_bloating'] = ((ov[f'{PREFIX}TX_gi_treated'] == 1) & (has_bloating == 1)).astype(int)

    ov[f'{PREFIX}TX_iron_prescribed'] = has_iron
    ov[f'{PREFIX}TX_iron_with_declining_hb'] = ((has_iron == 1) & (ov[f'{PREFIX}LAB_hb_declining'] == 1)).astype(int)
    ov[f'{PREFIX}TX_diuretic_with_ascites'] = ((has_diuretic == 1) & (has_ascites == 1)).astype(int)

    has_dvt = _has(obs_AB, 'DVT')
    has_pe = _has(obs_AB, 'PULMONARY_EMBOLISM')
    has_anticoag = _has_med('ANTICOAGULANTS')
    ov[f'{PREFIX}TX_vte'] = ((has_dvt == 1) | (has_pe == 1)).astype(int)
    ov[f'{PREFIX}TX_vte_with_symptoms'] = ((ov[f'{PREFIX}TX_vte'] == 1) & (ov[f'{PREFIX}NICE_cardinal_count'] >= 1)).astype(int)
    ov[f'{PREFIX}TX_anticoag'] = has_anticoag

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: MULTI-SYSTEM TRIFECTA (window B)
    # Constitutional + abdominal + gynae signals simultaneously = high suspicion.
    # ══════════════════════════════════════════════════════════
    abdo_B = (
        _count(obs_B, 'ABDOMINAL_BLOATING') + _count(obs_B, 'ABDOMINAL_PAIN')
        + _count(obs_B, 'ABDOMINAL_MASS') + _count(obs_B, 'ASCITES')
        + _count(obs_B, 'EARLY_SATIETY')
    )
    gynae_B = (
        _count(obs_B, 'GYNAECOLOGICAL_BLEEDING') + _count(obs_B, 'OVARIAN_CYST')
        + _count(obs_B, 'GYNAECOLOGICAL_DX') + _count(obs_B, 'VAGINAL_DISCHARGE')
    )
    constit_B = (
        _count(obs_B, 'WEIGHT_LOSS') + _count(obs_B, 'FATIGUE')
        + _count(obs_B, 'SYSTEMIC') + _count(obs_B, 'ANAEMIA')
    )
    abdo_flag_B = (abdo_B > 0).astype(int)
    gynae_flag_B = (gynae_B > 0).astype(int)
    constit_flag_B = (constit_B > 0).astype(int)
    multi_B = abdo_flag_B + gynae_flag_B + constit_flag_B
    ov[f'{PREFIX}TRIFECTA_abdo_gynae_constit_B'] = (multi_B >= 3).astype(int)
    ov[f'{PREFIX}MULTISYSTEM_PAIR_B'] = (multi_B >= 2).astype(int)
    ov[f'{PREFIX}MULTISYSTEM_count_B'] = multi_B

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: SYMPTOM VELOCITY (B - A, clipped to non-negative)
    # ══════════════════════════════════════════════════════════
    abdo_A = (
        _count(obs_A, 'ABDOMINAL_BLOATING') + _count(obs_A, 'ABDOMINAL_PAIN')
        + _count(obs_A, 'ABDOMINAL_MASS') + _count(obs_A, 'ASCITES')
        + _count(obs_A, 'EARLY_SATIETY')
    )
    gynae_A = (
        _count(obs_A, 'GYNAECOLOGICAL_BLEEDING') + _count(obs_A, 'OVARIAN_CYST')
        + _count(obs_A, 'GYNAECOLOGICAL_DX') + _count(obs_A, 'VAGINAL_DISCHARGE')
    )
    constit_A = (
        _count(obs_A, 'WEIGHT_LOSS') + _count(obs_A, 'FATIGUE')
        + _count(obs_A, 'SYSTEMIC') + _count(obs_A, 'ANAEMIA')
    )

    bloat_A = _count(obs_A, 'ABDOMINAL_BLOATING')
    pain_A = _count(obs_A, 'ABDOMINAL_PAIN')
    urin_A = _count(obs_A, 'URINARY_SYMPTOMS')
    satiety_A = _count(obs_A, 'EARLY_SATIETY')
    img_A = _count(obs_A, 'IMAGING')
    img_B_count = _count(obs_B, 'IMAGING')

    ov[f'{PREFIX}VEL_bloating'] = (bloat_B - bloat_A).clip(lower=0)
    ov[f'{PREFIX}VEL_pain'] = (pain_B - pain_A).clip(lower=0)
    ov[f'{PREFIX}VEL_urinary'] = (urin_B - urin_A).clip(lower=0)
    ov[f'{PREFIX}VEL_satiety'] = (satiety_B - satiety_A).clip(lower=0)
    ov[f'{PREFIX}VEL_abdominal'] = (abdo_B - abdo_A).clip(lower=0)
    ov[f'{PREFIX}VEL_gynae'] = (gynae_B - gynae_A).clip(lower=0)
    ov[f'{PREFIX}VEL_constitutional'] = (constit_B - constit_A).clip(lower=0)
    ov[f'{PREFIX}VEL_imaging'] = (img_B_count - img_A).clip(lower=0)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: NEW SYMPTOM IN WINDOW B ONLY (B>0 & A==0)
    # Recent-onset cardinal symptoms = ovarian red flag.
    # ══════════════════════════════════════════════════════════
    ov[f'{PREFIX}NEW_bloating_B_only'] = ((bloat_B > 0) & (bloat_A == 0)).astype(int)
    ov[f'{PREFIX}NEW_pain_B_only'] = ((pain_B > 0) & (pain_A == 0)).astype(int)
    ov[f'{PREFIX}NEW_urinary_B_only'] = ((urin_B > 0) & (urin_A == 0)).astype(int)
    ov[f'{PREFIX}NEW_satiety_B_only'] = ((satiety_B > 0) & (satiety_A == 0)).astype(int)
    ov[f'{PREFIX}NEW_mass_B_only'] = ((_count(obs_B, 'ABDOMINAL_MASS') > 0) & (_count(obs_A, 'ABDOMINAL_MASS') == 0)).astype(int)
    ov[f'{PREFIX}NEW_ascites_B_only'] = ((_count(obs_B, 'ASCITES') > 0) & (_count(obs_A, 'ASCITES') == 0)).astype(int)
    ov[f'{PREFIX}NEW_constitutional_B_only'] = ((constit_B > 0) & (constit_A == 0)).astype(int)
    ov[f'{PREFIX}NEW_bleeding_B_only'] = (
        (_count(obs_B, 'GYNAECOLOGICAL_BLEEDING') > 0) & (_count(obs_A, 'GYNAECOLOGICAL_BLEEDING') == 0)
    ).astype(int)
    ov[f'{PREFIX}NEW_cyst_B_only'] = ((_count(obs_B, 'OVARIAN_CYST') > 0) & (_count(obs_A, 'OVARIAN_CYST') == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: AGE × CLINICAL INTERACTIONS
    # Designed to survive 1:1 age-band matching.
    # ══════════════════════════════════════════════════════════

    # Age-derived (peak ovarian-cancer incidence age ~63)
    ov[f'{PREFIX}AGEX_PEAK_DIST'] = (age - 63).abs()
    ov[f'{PREFIX}AGEX_OVER_50'] = (age >= 50).astype(int)
    ov[f'{PREFIX}AGEX_OVER_60'] = (age >= 60).astype(int)
    ov[f'{PREFIX}AGEX_OVER_70'] = (age >= 70).astype(int)
    ov[f'{PREFIX}AGEX_OVER_80'] = (age >= 80).astype(int)
    ov[f'{PREFIX}AGEX_DECILE'] = pd.qcut(
        age.rank(method='first'), q=10, labels=False, duplicates='drop'
    ).fillna(0).astype(int)

    # Age × symptom counts (window B)
    ov[f'{PREFIX}AGEX_BLOAT_count_B'] = age * bloat_B
    ov[f'{PREFIX}AGEX_PAIN_count_B'] = age * pain_B
    ov[f'{PREFIX}AGEX_URINARY_count_B'] = age * urin_B
    ov[f'{PREFIX}AGEX_SATIETY_count_B'] = age * satiety_B
    ov[f'{PREFIX}AGEX_ABDO_count_B'] = age * abdo_B
    ov[f'{PREFIX}AGEX_GYNAE_count_B'] = age * gynae_B
    ov[f'{PREFIX}AGEX_CONSTIT_count_B'] = age * constit_B
    ov[f'{PREFIX}AGEX_IMAGING_count_B'] = age * img_B_count

    # Age × symptom acceleration
    ov[f'{PREFIX}AGEX_BLOAT_acceleration'] = age * (bloat_B - bloat_A).clip(lower=0)
    ov[f'{PREFIX}AGEX_PAIN_acceleration'] = age * (pain_B - pain_A).clip(lower=0)
    ov[f'{PREFIX}AGEX_ABDO_acceleration'] = age * (abdo_B - abdo_A).clip(lower=0)
    ov[f'{PREFIX}AGEX_GYNAE_acceleration'] = age * (gynae_B - gynae_A).clip(lower=0)
    ov[f'{PREFIX}AGEX_CONSTIT_acceleration'] = age * (constit_B - constit_A).clip(lower=0)

    # Age × labs (real biology)
    ov[f'{PREFIX}AGEX_HB_anaemic'] = age * ov[f'{PREFIX}LAB_hb_anaemic'].astype(float)
    ov[f'{PREFIX}AGEX_CRP_elevated'] = age * ov[f'{PREFIX}LAB_crp_elevated'].astype(float)
    ov[f'{PREFIX}AGEX_ALBUMIN_low'] = age * ov[f'{PREFIX}LAB_albumin_low'].astype(float)
    ov[f'{PREFIX}AGEX_PLATELETS_high'] = age * ov[f'{PREFIX}LAB_platelets_high'].astype(float)
    ov[f'{PREFIX}AGEX_LDH_elevated'] = age * ov[f'{PREFIX}LAB_ldh_elevated'].astype(float)
    ov[f'{PREFIX}AGEX_CA_high'] = age * ov[f'{PREFIX}LAB_calcium_high'].astype(float)

    # Age × NEW-symptom-in-B
    ov[f'{PREFIX}AGEX_NEW_bloating_B'] = age * ov[f'{PREFIX}NEW_bloating_B_only']
    ov[f'{PREFIX}AGEX_NEW_pain_B'] = age * ov[f'{PREFIX}NEW_pain_B_only']
    ov[f'{PREFIX}AGEX_NEW_mass_B'] = age * ov[f'{PREFIX}NEW_mass_B_only']
    ov[f'{PREFIX}AGEX_NEW_ascites_B'] = age * ov[f'{PREFIX}NEW_ascites_B_only']
    ov[f'{PREFIX}AGEX_NEW_bleeding_B'] = age * ov[f'{PREFIX}NEW_bleeding_B_only']

    # Age × medications
    ov[f'{PREFIX}AGEX_OPIOID'] = age * has_opioid.astype(float)
    ov[f'{PREFIX}AGEX_HRT'] = age * ov[f'{PREFIX}RF_hrt_use'].astype(float)
    ov[f'{PREFIX}AGEX_GI_MED'] = age * ov[f'{PREFIX}TX_gi_treated'].astype(float)

    # Age × diagnostic-pathway
    ov[f'{PREFIX}AGEX_IMAGING_DELAY'] = age * ov[f'{PREFIX}DX_imaging_delayed_60d'].astype(float)
    ov[f'{PREFIX}AGEX_GYNAE_PROC'] = age * has_gynae_proc.astype(float)

    # Composite "elderly + clinical signal" (over 60 since ovarian peaks earlier than prostate)
    elderly = (age >= 60).astype(int)
    ov[f'{PREFIX}ELDERLY_WITH_BLOATING'] = (elderly & (bloat_B > 0).astype(int))
    ov[f'{PREFIX}ELDERLY_WITH_MASS'] = (elderly & has_mass)
    ov[f'{PREFIX}ELDERLY_WITH_ASCITES'] = (elderly & has_ascites)
    ov[f'{PREFIX}ELDERLY_WITH_BLEEDING'] = (elderly & has_bleed)
    ov[f'{PREFIX}ELDERLY_WITH_CARDINAL2'] = (elderly & ov[f'{PREFIX}NICE_2plus_cardinals'])
    ov[f'{PREFIX}ELDERLY_WITH_HB_LOW'] = (elderly & ov[f'{PREFIX}LAB_hb_anaemic'])
    ov[f'{PREFIX}ELDERLY_WITH_AGGRESSIVE_TRIO'] = (elderly & ov[f'{PREFIX}LAB_aggressive_trio'])
    ov[f'{PREFIX}ELDERLY_NEW_MASS_B'] = (elderly & ov[f'{PREFIX}NEW_mass_B_only'])
    ov[f'{PREFIX}ELDERLY_NEW_ASCITES_B'] = (elderly & ov[f'{PREFIX}NEW_ascites_B_only'])

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: TRIPLE INTERACTIONS (age × symptom × biomarker/symptom)
    # ══════════════════════════════════════════════════════════
    mass_f = has_mass.astype(float)
    ascites_f = has_ascites.astype(float)
    cyst_f = has_cyst.astype(float)
    fh_f = has_fh.astype(float)
    plt_high = ov[f'{PREFIX}LAB_platelets_high'].astype(float)
    crp_high = ov[f'{PREFIX}LAB_crp_elevated'].astype(float)
    hb_low = ov[f'{PREFIX}LAB_hb_anaemic'].astype(float)
    alb_low = ov[f'{PREFIX}LAB_albumin_low'].astype(float)

    ov[f'{PREFIX}TRIPLE_age_bloatB_mass'] = age * bloat_B * mass_f
    ov[f'{PREFIX}TRIPLE_age_bloatB_ascites'] = age * bloat_B * ascites_f
    ov[f'{PREFIX}TRIPLE_age_painB_mass'] = age * pain_B * mass_f
    ov[f'{PREFIX}TRIPLE_age_bloatB_pltHigh'] = age * bloat_B * plt_high
    ov[f'{PREFIX}TRIPLE_age_painB_pltHigh'] = age * pain_B * plt_high
    ov[f'{PREFIX}TRIPLE_age_constitB_albLow'] = age * constit_B * alb_low
    ov[f'{PREFIX}TRIPLE_age_bloatB_fh'] = age * bloat_B * fh_f
    ov[f'{PREFIX}TRIPLE_age_cystB_painB'] = age * cyst_f * pain_B
    ov[f'{PREFIX}TRIPLE_mass_ascites_age'] = age * mass_f * ascites_f
    ov[f'{PREFIX}TRIPLE_cardinals_pltHigh_age'] = age * ov[f'{PREFIX}NICE_cardinal_count'].astype(float) * plt_high

    # Lab biomarker pairs
    ov[f'{PREFIX}LABPAIR_plt_x_crp'] = plt_high * crp_high
    ov[f'{PREFIX}LABPAIR_hb_low_x_alb_low'] = hb_low * alb_low
    ov[f'{PREFIX}LABPAIR_plt_x_hb_low'] = plt_high * hb_low
    ov[f'{PREFIX}LABPAIR_alb_low_x_ldh'] = alb_low * ov[f'{PREFIX}LAB_ldh_elevated'].astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: COMORBIDITY BURDEN + RECENT ACTIVITY ESCALATION
    # ══════════════════════════════════════════════════════════
    n_categories = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(patients, fill_value=0).astype(int)
    ov[f'{PREFIX}COMORBIDITY_n_categories'] = n_categories
    ov[f'{PREFIX}COMORBIDITY_x_age'] = age * n_categories
    ov[f'{PREFIX}COMORBIDITY_high'] = (n_categories >= 5).astype(int)
    ov[f'{PREFIX}COMORBIDITY_x_cardinals'] = n_categories * ov[f'{PREFIX}NICE_cardinal_count'].astype(float)
    ov[f'{PREFIX}COMORBIDITY_x_mass'] = n_categories * mass_f

    events_B = obs_B.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    events_A = obs_A.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    ov[f'{PREFIX}ACTIVITY_events_B'] = events_B
    ov[f'{PREFIX}ACTIVITY_events_A'] = events_A
    ov[f'{PREFIX}ACTIVITY_velocity'] = (events_B - events_A).clip(lower=0)
    ov[f'{PREFIX}ACTIVITY_acceleration'] = (events_B / events_A.replace(0, 1)).clip(upper=10)
    ov[f'{PREFIX}AGEX_ACTIVITY_velocity'] = age * ov[f'{PREFIX}ACTIVITY_velocity']

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: INVESTIGATION × SYMPTOM PAIRS
    # ══════════════════════════════════════════════════════════
    img_f = has_imaging.astype(float)
    gynae_proc_f = has_gynae_proc.astype(float)
    ov[f'{PREFIX}IMG_x_bloatB'] = img_f * bloat_B
    ov[f'{PREFIX}IMG_x_painB'] = img_f * pain_B
    ov[f'{PREFIX}IMG_x_mass'] = img_f * mass_f
    ov[f'{PREFIX}IMG_x_ascites'] = img_f * ascites_f
    ov[f'{PREFIX}IMG_x_cyst'] = img_f * cyst_f
    ov[f'{PREFIX}GYNAE_PROC_x_bleed'] = gynae_proc_f * has_bleed.astype(float)
    ov[f'{PREFIX}PATHWAY_x_age'] = age * (img_f + gynae_proc_f).clip(upper=2)

    # ══════════════════════════════════════════════════════════
    # BLOCK 15: FIRST-EVENT-DATE PER CATEGORY + RECENT (<12mo) FLAGS
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

    ov[f'{PREFIX}FIRST_bloating_months'] = _months_since_first('ABDOMINAL_BLOATING')
    ov[f'{PREFIX}FIRST_pain_months'] = _months_since_first('ABDOMINAL_PAIN')
    ov[f'{PREFIX}FIRST_satiety_months'] = _months_since_first('EARLY_SATIETY')
    ov[f'{PREFIX}FIRST_urinary_months'] = _months_since_first('URINARY_SYMPTOMS')
    ov[f'{PREFIX}FIRST_mass_months'] = _months_since_first('ABDOMINAL_MASS')
    ov[f'{PREFIX}FIRST_ascites_months'] = _months_since_first('ASCITES')
    ov[f'{PREFIX}FIRST_cyst_months'] = _months_since_first('OVARIAN_CYST')
    ov[f'{PREFIX}FIRST_bleeding_months'] = _months_since_first('GYNAECOLOGICAL_BLEEDING')
    ov[f'{PREFIX}FIRST_gi_months'] = _months_since_first('GI_SYMPTOMS')
    ov[f'{PREFIX}FIRST_imaging_months'] = _months_since_first('IMAGING')
    ov[f'{PREFIX}FIRST_constitutional_months'] = _months_since_first_multi(
        ['WEIGHT_LOSS', 'FATIGUE', 'SYSTEMIC', 'ANAEMIA']
    )

    for col_name in ['bloating', 'pain', 'satiety', 'mass', 'ascites', 'cyst',
                     'bleeding', 'urinary', 'constitutional']:
        first_col = f'{PREFIX}FIRST_{col_name}_months'
        ov[f'{PREFIX}RECENT_{col_name}'] = (
            (ov[first_col] >= 0) & (ov[first_col] < 12)
        ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 16: SEQUENTIAL X→Y PATTERNS (within Z days)
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

    ov[f'{PREFIX}SEQ_bloating_then_imaging_60d'] = _seq('ABDOMINAL_BLOATING', 'IMAGING', 60)
    ov[f'{PREFIX}SEQ_bloating_then_imaging_180d'] = _seq('ABDOMINAL_BLOATING', 'IMAGING', 180)
    ov[f'{PREFIX}SEQ_pain_then_imaging_90d'] = _seq('ABDOMINAL_PAIN', 'IMAGING', 90)
    ov[f'{PREFIX}SEQ_mass_then_imaging_30d'] = _seq('ABDOMINAL_MASS', 'IMAGING', 30)
    ov[f'{PREFIX}SEQ_ascites_then_imaging_30d'] = _seq('ASCITES', 'IMAGING', 30)
    ov[f'{PREFIX}SEQ_cyst_then_imaging_60d'] = _seq('OVARIAN_CYST', 'IMAGING', 60)
    ov[f'{PREFIX}SEQ_bleeding_then_gynae_60d'] = _seq('GYNAECOLOGICAL_BLEEDING', 'GYNAE_PROCEDURES', 60)
    ov[f'{PREFIX}SEQ_bloating_then_gynae_120d'] = _seq('ABDOMINAL_BLOATING', 'GYNAE_PROCEDURES', 120)
    ov[f'{PREFIX}SEQ_urinary_then_imaging_90d'] = _seq('URINARY_SYMPTOMS', 'IMAGING', 90)

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: CHARLSON-LITE WEIGHTED COMORBIDITY
    # Weights tuned for ovarian-cancer relevance.
    # ══════════════════════════════════════════════════════════
    CHARLSON_LITE_OVARIAN = {
        'ABDOMINAL_MASS': 3.0,             # primary red flag
        'ASCITES': 3.0,                    # primary red flag (late stage)
        'ABDOMINAL_BLOATING': 2.0,         # NICE cardinal
        'EARLY_SATIETY': 2.0,              # NICE cardinal
        'ABDOMINAL_PAIN': 1.5,             # NICE cardinal
        'URINARY_SYMPTOMS': 1.5,           # NICE cardinal
        'GI_SYMPTOMS': 1.0,
        'OVARIAN_CYST': 2.5,
        'GYNAECOLOGICAL_BLEEDING': 1.5,    # PMB risk
        'POSTMENOPAUSAL': 1.0,
        'GYNAECOLOGICAL_DX': 1.0,
        'WEIGHT_LOSS': 2.0,
        'FATIGUE': 1.0,
        'SYSTEMIC': 1.5,
        'ANAEMIA': 1.5,
        'IMAGING': 1.0,
        'GYNAE_PROCEDURES': 2.0,
        'OTHER_PROCEDURES': 1.0,
        'FAMILY_HISTORY': 2.0,             # BRCA risk
        'DVT': 1.5,                        # paraneoplastic
        'PULMONARY_EMBOLISM': 1.5,
        'HRT': 0.5,
        'OBESITY': 0.5,
        'HYDRONEPHROSIS': 1.0,             # pelvic mass effect
    }
    score = pd.Series(0.0, index=patients)
    for cat, w in CHARLSON_LITE_OVARIAN.items():
        score = score + w * _has(obs_AB, cat).astype(float)
    ov[f'{PREFIX}COMORB_weighted_score'] = score
    ov[f'{PREFIX}COMORB_weighted_x_age'] = age * score
    ov[f'{PREFIX}COMORB_weighted_high'] = (score >= 5).astype(int)
    ov[f'{PREFIX}COMORB_weighted_x_mass'] = score * mass_f
    ov[f'{PREFIX}COMORB_weighted_x_cardinals'] = score * ov[f'{PREFIX}NICE_cardinal_count'].astype(float)

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
        ov[f'{PREFIX}SYMPTOM_max_in_30d'] = burst.reindex(patients, fill_value=0).astype(int)
    else:
        ov[f'{PREFIX}SYMPTOM_max_in_30d'] = 0
    ov[f'{PREFIX}SYMPTOM_burst_3plus'] = (ov[f'{PREFIX}SYMPTOM_max_in_30d'] >= 3).astype(int)
    ov[f'{PREFIX}SYMPTOM_burst_4plus'] = (ov[f'{PREFIX}SYMPTOM_max_in_30d'] >= 4).astype(int)

    # ── Cleanup ───────────────────────────────────────────────
    ov = ov.fillna(0).replace([np.inf, -np.inf], 0)

    nunique = ov.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        ov = ov.drop(columns=constant)
        logger.info(f"  Removed {len(constant)} constant columns")

    logger.info(f"  Ovarian-specific features: {ov.shape[1]}")
    return ov
