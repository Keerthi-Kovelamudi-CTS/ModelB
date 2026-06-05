# ═══════════════════════════════════════════════════════════════
# MELANOMA — CANCER-SPECIFIC FEATURES
# NICE NG14 + 7-point checklist features, lesion-to-procedure pathway gaps,
# mimic patterns (benign mole, keratosis cryo'd without histology), risk
# factors (Fitzpatrick / sun damage / prior BCC-SCC / personal H/O melanoma /
# family history / immunosuppression), treatment patterns (topical→surgical
# escalation, actinic pre-malignant Tx, multi-procedure burden).
#
# Melanoma codelist has NO lab categories — features are purely procedural
# and observational. Block 5 (lab thresholds) is intentionally omitted.
#
# Ported from Melanoma_Cancer/2_Feature_Engineering/4_Feature_engineering.py
# (build_melanoma_features) and adapted to the new config-driven pipeline.
# Plus generic structural blocks (8-18) shared with Bladder/Ovarian/Lymphoma/Leukaemia.
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
    Melanoma-specific features:
      - NICE NG14 / 7-PCL checklist (mole, lesion, dermatoscopy, dermatology
        referral, histology — count + ≥2/≥3 thresholds + pathway-complete)
      - Lesion pathway gaps (lesion→dermatoscopy / →biopsy / →histology / →referral)
      - Mimic patterns (benign mole no dermatoscopy, keratosis cryo'd no histology)
      - Risk factors (sun damage, prior BCC/SCC, family history, **personal H/O
        melanoma**, immunosuppression, Fitzpatrick, peak-age, male)
      - Treatment patterns (topical→surgical escalation, actinic pre-malignant Tx,
        excision/cryo/plastic procedure counts)
      - Composite SUSPICION score
      - Plus generic structural blocks 8-18 (trifecta, velocity, AGE×, etc.)
    """
    if cfg is None:
        cfg = config
    PREFIX = cfg.PREFIX  # 'MELA_'

    logger.info(f"  MELANOMA-SPECIFIC FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')

    patients = existing_fm.index
    mel = pd.DataFrame(index=patients)
    mel.index.name = 'PATIENT_GUID'

    obs_AB = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    obs_A = clin[clin['TIME_WINDOW'] == 'A'].copy()
    obs_B = clin[clin['TIME_WINDOW'] == 'B'].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy() if 'TIME_WINDOW' in med.columns else med
    med_A = med[med['TIME_WINDOW'] == 'A'].copy() if 'TIME_WINDOW' in med.columns else med.iloc[:0]
    med_B = med[med['TIME_WINDOW'] == 'B'].copy() if 'TIME_WINDOW' in med.columns else med.iloc[:0]

    age = (existing_fm['AGE_AT_INDEX'].reindex(patients).fillna(55)
           if 'AGE_AT_INDEX' in existing_fm.columns
           else pd.Series(55.0, index=patients))
    sex = (clin.drop_duplicates('PATIENT_GUID').set_index('PATIENT_GUID')['SEX']
           .reindex(patients).fillna('U')) if 'SEX' in clin.columns else pd.Series('U', index=patients)

    # ── Helpers ──────────────────────────────────────────────────
    def _has(df, cat):
        return (df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0).reindex(patients, fill_value=False).astype(int)

    def _count(df, cat):
        return df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)

    def _has_med(df, category):
        return (df[df['CATEGORY'] == category].groupby('PATIENT_GUID').size() > 0).reindex(patients, fill_value=False).astype(int)

    def _count_med(df, category):
        return df[df['CATEGORY'] == category].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: NICE NG14 / 7-POINT CHECKLIST FEATURES
    # NICE recognises 7-PCL features + dermatology referral / dermoscopy /
    # histology as the standard melanoma diagnostic pathway.
    # ══════════════════════════════════════════════════════════
    has_7pcl = _has(obs_AB, 'NICE_7PCL_FEATURES')
    has_derm_ref = _has(obs_AB, 'DERMATOLOGY_REFERRAL_CLINIC')
    has_dermoscopy = _has(obs_AB, 'DERMATOSCOPY_PHOTOGRAPHY')
    has_lesion = _has(obs_AB, 'SKIN_LESION_PRESENTATION')
    has_mole = _has(obs_AB, 'MOLE_NAEVUS_PIGMENTED_LESION')
    has_clinical_signs = _has(obs_AB, 'CLINICAL_SIGNS')
    has_histology = _has(obs_AB, 'HISTOLOGY')
    has_biopsy = _has(obs_AB, 'PRIOR_SKIN_PROCEDURES_BIOPSY')
    has_excision = _has(obs_AB, 'WIDE_EXCISION_GRAFTING')
    has_plastic = _has(obs_AB, 'PLASTIC_SURGERY')

    mel[f'{PREFIX}NICE_7pcl'] = has_7pcl
    mel[f'{PREFIX}NICE_dermatology_referral'] = has_derm_ref
    mel[f'{PREFIX}NICE_dermoscopy'] = has_dermoscopy
    mel[f'{PREFIX}NICE_has_lesion'] = has_lesion
    mel[f'{PREFIX}NICE_has_mole'] = has_mole
    mel[f'{PREFIX}NICE_clinical_signs'] = has_clinical_signs
    mel[f'{PREFIX}NICE_histology'] = has_histology
    mel[f'{PREFIX}NICE_biopsy'] = has_biopsy
    mel[f'{PREFIX}NICE_excision'] = has_excision
    mel[f'{PREFIX}NICE_plastic_surgery'] = has_plastic

    # Pathway completeness (lesion → dermoscopy → histology = full workup)
    mel[f'{PREFIX}NICE_pathway_complete'] = (
        (has_lesion == 1) & (has_dermoscopy == 1) & (has_histology == 1)
    ).astype(int)
    mel[f'{PREFIX}NICE_pathway_with_excision'] = (
        (mel[f'{PREFIX}NICE_pathway_complete'] == 1) & (has_excision == 1)
    ).astype(int)

    # Cardinal count (key NICE NG14 signals)
    mel[f'{PREFIX}NICE_cardinal_count'] = (
        has_7pcl + has_derm_ref + has_dermoscopy + has_lesion + has_mole + has_clinical_signs
    )
    mel[f'{PREFIX}NICE_2plus_flags'] = (mel[f'{PREFIX}NICE_cardinal_count'] >= 2).astype(int)
    mel[f'{PREFIX}NICE_3plus_flags'] = (mel[f'{PREFIX}NICE_cardinal_count'] >= 3).astype(int)

    # 7-PCL × dermatology referral = high-suspicion combination
    mel[f'{PREFIX}NICE_7pcl_with_referral'] = ((has_7pcl == 1) & (has_derm_ref == 1)).astype(int)
    mel[f'{PREFIX}NICE_7pcl_with_dermoscopy'] = ((has_7pcl == 1) & (has_dermoscopy == 1)).astype(int)
    mel[f'{PREFIX}NICE_mole_with_dermoscopy'] = ((has_mole == 1) & (has_dermoscopy == 1)).astype(int)

    mel[f'{PREFIX}NICE_over50_with_lesion'] = ((age >= 50) & (has_lesion == 1)).astype(int)
    mel[f'{PREFIX}NICE_over50_with_2plus'] = ((age >= 50) & (mel[f'{PREFIX}NICE_2plus_flags'] == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: LESION-TO-PROCEDURE PATHWAY GAPS
    # Time from first lesion observation to dermatoscopy / biopsy / histology /
    # excision. Long gap suggests delayed dx. Short gap = active workup.
    # ══════════════════════════════════════════════════════════
    def _first_event_date(category):
        sub = obs_AB[obs_AB['CATEGORY'] == category]
        if sub.empty:
            return pd.Series(pd.NaT, index=patients, dtype='datetime64[ns]')
        first = sub.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        return pd.to_datetime(first.reindex(patients), errors='coerce')

    lesion_first = _first_event_date('SKIN_LESION_PRESENTATION')
    mole_first = _first_event_date('MOLE_NAEVUS_PIGMENTED_LESION')
    pcl_first = _first_event_date('NICE_7PCL_FEATURES')
    derm_first = _first_event_date('DERMATOSCOPY_PHOTOGRAPHY')
    biopsy_first = _first_event_date('PRIOR_SKIN_PROCEDURES_BIOPSY')
    histo_first = _first_event_date('HISTOLOGY')
    excision_first = _first_event_date('WIDE_EXCISION_GRAFTING')
    referral_first = _first_event_date('DERMATOLOGY_REFERRAL_CLINIC')

    def _gap_days(later, earlier):
        gap = (later - earlier).dt.days
        return gap.fillna(-1).astype(float)

    mel[f'{PREFIX}PATH_lesion_to_dermoscopy_days'] = _gap_days(derm_first, lesion_first)
    mel[f'{PREFIX}PATH_lesion_to_biopsy_days'] = _gap_days(biopsy_first, lesion_first)
    mel[f'{PREFIX}PATH_lesion_to_histology_days'] = _gap_days(histo_first, lesion_first)
    mel[f'{PREFIX}PATH_lesion_to_referral_days'] = _gap_days(referral_first, lesion_first)
    mel[f'{PREFIX}PATH_dermoscopy_to_excision_days'] = _gap_days(excision_first, derm_first)
    mel[f'{PREFIX}PATH_biopsy_to_histology_days'] = _gap_days(histo_first, biopsy_first)
    mel[f'{PREFIX}PATH_mole_to_dermoscopy_days'] = _gap_days(derm_first, mole_first)
    mel[f'{PREFIX}PATH_pcl_to_excision_days'] = _gap_days(excision_first, pcl_first)

    # Delay flags
    mel[f'{PREFIX}PATH_dermoscopy_delayed_60d'] = (mel[f'{PREFIX}PATH_lesion_to_dermoscopy_days'] > 60).astype(int)
    mel[f'{PREFIX}PATH_dermoscopy_delayed_180d'] = (mel[f'{PREFIX}PATH_lesion_to_dermoscopy_days'] > 180).astype(int)
    mel[f'{PREFIX}PATH_histology_delayed_60d'] = (mel[f'{PREFIX}PATH_lesion_to_histology_days'] > 60).astype(int)
    mel[f'{PREFIX}PATH_excision_within_30d'] = (
        (mel[f'{PREFIX}PATH_dermoscopy_to_excision_days'] >= 0)
        & (mel[f'{PREFIX}PATH_dermoscopy_to_excision_days'] <= 30)
    ).astype(int)

    # Total procedure burden
    excision_count = _count(obs_AB, 'WIDE_EXCISION_GRAFTING')
    cryo_count = _count(obs_AB, 'CRYOTHERAPY')
    biopsy_count = _count(obs_AB, 'PRIOR_SKIN_PROCEDURES_BIOPSY')
    histology_count = _count(obs_AB, 'HISTOLOGY')
    plastic_count = _count(obs_AB, 'PLASTIC_SURGERY')
    minor_surg_count = _count(obs_AB, 'MINOR_SURGERY_ADMIN')
    suture_count = _count(obs_AB, 'SUTURE_POST_OP_MINOR_SURGERY')
    procedure_total = (
        excision_count + cryo_count + biopsy_count + histology_count
        + plastic_count + minor_surg_count + suture_count
    )
    mel[f'{PREFIX}PATH_procedure_count'] = procedure_total
    mel[f'{PREFIX}PATH_procedure_high'] = (procedure_total >= 3).astype(int)
    mel[f'{PREFIX}PATH_procedure_very_high'] = (procedure_total >= 5).astype(int)

    # Wide excision = strong definitive-treatment signal
    mel[f'{PREFIX}PATH_wide_excision_with_histology'] = (
        (has_excision == 1) & (has_histology == 1)
    ).astype(int)
    mel[f'{PREFIX}PATH_wide_excision_with_plastic'] = (
        (has_excision == 1) & (has_plastic == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: MIMIC PATTERNS (benign DDx)
    # Non-cancer skin conditions treated without dermoscopy/histology workup.
    # ══════════════════════════════════════════════════════════
    has_other_lesions = _has(obs_AB, 'OTHER_SKIN_LESIONS')
    has_skin_conditions = _has(obs_AB, 'SKIN_CONDITIONS')
    has_cryo = _has(obs_AB, 'CRYOTHERAPY')
    has_cryo_wart = _has(obs_AB, 'CRYOTHERAPY_WART_TREATMENT')
    has_insect = _has(obs_AB, 'INSECT_BITES')
    has_skin_infection = _has(obs_AB, 'SKIN_INFECTION_WORKUP')
    has_virology = _has(obs_AB, 'VIROLOGY_SKIN_RELATED')
    has_skin_symptoms = _has(obs_AB, 'SKIN_SYMPTOMS')

    mel[f'{PREFIX}has_other_lesions'] = has_other_lesions
    mel[f'{PREFIX}has_skin_conditions'] = has_skin_conditions
    mel[f'{PREFIX}has_cryotherapy'] = has_cryo
    mel[f'{PREFIX}has_skin_symptoms'] = has_skin_symptoms

    # Benign mole mimic: mole observed but no dermoscopy ordered
    mel[f'{PREFIX}MIMIC_benign_mole'] = ((has_mole == 1) & (has_dermoscopy == 0)).astype(int)

    # Keratosis mimic: benign DDx + cryotherapy but no histology workup
    mel[f'{PREFIX}MIMIC_keratosis_cryo_no_histo'] = (
        (has_other_lesions == 1) & (has_cryo == 1) & (has_histology == 0)
    ).astype(int)

    # Wart-treatment mimic: cryotherapy for warts (different from melanoma cryo)
    mel[f'{PREFIX}MIMIC_wart_cryo'] = ((has_cryo_wart == 1) & (has_excision == 0)).astype(int)

    # Insect-bite-only follow-up
    mel[f'{PREFIX}MIMIC_insect_bite_no_workup'] = (
        (has_insect == 1) & (has_dermoscopy == 0) & (has_histology == 0)
    ).astype(int)

    # Skin-infection-mimic: workup + viral testing instead of biopsy
    mel[f'{PREFIX}MIMIC_skin_infection_workup'] = (
        ((has_skin_infection == 1) | (has_virology == 1)) & (has_histology == 0)
    ).astype(int)

    # Symptoms-no-investigation
    mel[f'{PREFIX}MIMIC_symptoms_no_investigation'] = (
        (mel[f'{PREFIX}NICE_cardinal_count'] >= 2)
        & (has_dermoscopy == 0)
        & (has_histology == 0)
        & (has_biopsy == 0)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: RISK FACTORS
    # Sun damage, prior BCC/SCC, family history, **personal H/O melanoma**
    # (very strong recurrence signal), immunosuppression, Fitzpatrick.
    # ══════════════════════════════════════════════════════════
    has_sun_damage = _has(obs_AB, 'SUN_DAMAGE_PRE_MALIGNANT')
    has_prior_bcc_scc = _has(obs_AB, 'PRIOR_SKIN_CANCER_BCC_SCC')
    has_family_history = _has(obs_AB, 'FAMILY_HISTORY')
    has_personal_history = _has(obs_AB, 'PERSONAL_HISTORY')          # H/O melanoma — top recurrence signal
    has_risk_factor = _has(obs_AB, 'RISK_FACTOR')                    # Fitzpatrick skin type
    has_immuno_systemic = _has_med(med_AB, 'IMMUNOSUPPRESSANT_SYSTEMIC')
    has_immuno_steroid = _has_med(med_AB, 'IMMUNOSUPPRESSANT_CORTICOSTEROIDS')
    has_immunosuppression = ((has_immuno_systemic == 1) | (has_immuno_steroid == 1)).astype(int)

    mel[f'{PREFIX}RF_sun_damage'] = has_sun_damage
    mel[f'{PREFIX}RF_prior_bcc_scc'] = has_prior_bcc_scc
    mel[f'{PREFIX}RF_family_history'] = has_family_history
    mel[f'{PREFIX}RF_personal_history'] = has_personal_history
    mel[f'{PREFIX}RF_fitzpatrick'] = has_risk_factor
    mel[f'{PREFIX}RF_immunosuppressant_systemic'] = has_immuno_systemic
    mel[f'{PREFIX}RF_immunosuppressant_steroid'] = has_immuno_steroid
    mel[f'{PREFIX}RF_immunosuppressed'] = has_immunosuppression

    mel[f'{PREFIX}RF_personal_with_lesion'] = ((has_personal_history == 1) & (has_lesion == 1)).astype(int)
    mel[f'{PREFIX}RF_prior_cancer_with_new_lesion'] = ((has_prior_bcc_scc == 1) & (has_lesion == 1)).astype(int)
    mel[f'{PREFIX}RF_sun_with_lesion'] = ((has_sun_damage == 1) & (has_lesion == 1)).astype(int)
    mel[f'{PREFIX}RF_immuno_with_lesion'] = ((has_immunosuppression == 1) & (has_lesion == 1)).astype(int)
    mel[f'{PREFIX}RF_family_with_dermatology'] = ((has_family_history == 1) & (has_derm_ref == 1)).astype(int)

    mel[f'{PREFIX}RF_peak_age'] = ((age >= 50) & (age <= 80)).astype(int)
    mel[f'{PREFIX}RF_high_risk_age'] = (age >= 50).astype(int)
    mel[f'{PREFIX}RF_elderly'] = (age >= 70).astype(int)
    mel[f'{PREFIX}RF_male'] = (sex == 'M').astype(int)

    rf_count = (
        has_sun_damage + has_prior_bcc_scc + has_family_history
        + has_personal_history + has_immunosuppression + has_risk_factor
    )
    mel[f'{PREFIX}RF_risk_factor_count'] = rf_count
    mel[f'{PREFIX}RF_2plus_risk_factors'] = (rf_count >= 2).astype(int)
    mel[f'{PREFIX}RF_3plus_risk_factors'] = (rf_count >= 3).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: TREATMENT PATTERNS
    # Topical → surgical escalation. Actinic pre-malignant Tx (5FU/imiquimod/
    # ingenol) signals concerning lesion. Local anaesthetic = procedure proxy.
    # ══════════════════════════════════════════════════════════
    has_actinic_tx = _has_med(med_AB, 'ACTINIC_PRE_MALIGNANT_TREATMENT')
    has_topical_steroid = _has_med(med_AB, 'TOPICAL_STEROIDS_SKIN')
    has_topical_abx = _has_med(med_AB, 'TOPICAL_ANTIBIOTICS_SKIN')
    has_topical_other = _has_med(med_AB, 'TOPICAL_SKIN_OTHER')
    has_topical_scalp = _has_med(med_AB, 'TOPICAL_STEROID_SCALP')
    has_local_anaes = _has_med(med_AB, 'LOCAL_ANAESTHETICS_SKIN')
    has_sutures = _has_med(med_AB, 'SUTURES_WOUND_TAPE')
    has_wound_dress = _has_med(med_AB, 'WOUND_DRESSINGS')
    has_wound_care = _has_med(med_AB, 'WOUND_CARE')
    has_oral_skin_abx = _has_med(med_AB, 'SKIN_ANTIBIOTICS_ORAL')
    has_antifungal_sys = _has_med(med_AB, 'ANTIFUNGAL_SYSTEMIC')
    has_antiviral_sys = _has_med(med_AB, 'ANTIVIRAL_SYSTEMIC')

    has_any_topical = (
        (has_topical_steroid == 1) | (has_topical_abx == 1)
        | (has_topical_other == 1) | (has_topical_scalp == 1)
    ).astype(int)
    has_any_surgical_med = (
        (has_local_anaes == 1) | (has_sutures == 1)
        | (has_wound_dress == 1) | (has_wound_care == 1)
    ).astype(int)

    mel[f'{PREFIX}TX_actinic_pre_malignant'] = has_actinic_tx
    mel[f'{PREFIX}TX_actinic_with_lesion'] = ((has_actinic_tx == 1) & (has_lesion == 1)).astype(int)
    mel[f'{PREFIX}TX_actinic_with_sun_damage'] = ((has_actinic_tx == 1) & (has_sun_damage == 1)).astype(int)

    mel[f'{PREFIX}TX_topical_steroids'] = has_topical_steroid
    mel[f'{PREFIX}TX_any_topical'] = has_any_topical
    mel[f'{PREFIX}TX_local_anaesthetic'] = has_local_anaes
    mel[f'{PREFIX}TX_local_anaes_with_excision'] = ((has_local_anaes == 1) & (has_excision == 1)).astype(int)
    mel[f'{PREFIX}TX_local_anaes_with_biopsy'] = ((has_local_anaes == 1) & (has_biopsy == 1)).astype(int)
    mel[f'{PREFIX}TX_any_surgical_med'] = has_any_surgical_med
    mel[f'{PREFIX}TX_topical_to_surgical'] = ((has_any_topical == 1) & (has_any_surgical_med == 1)).astype(int)

    mel[f'{PREFIX}TX_excision_count'] = excision_count
    mel[f'{PREFIX}TX_cryotherapy_count'] = cryo_count
    mel[f'{PREFIX}TX_biopsy_count'] = biopsy_count
    mel[f'{PREFIX}TX_histology_count'] = histology_count
    mel[f'{PREFIX}TX_plastic_count'] = plastic_count
    mel[f'{PREFIX}TX_multiple_excisions'] = (excision_count >= 2).astype(int)
    mel[f'{PREFIX}TX_multiple_biopsies'] = (biopsy_count >= 2).astype(int)
    mel[f'{PREFIX}TX_multiple_procedures'] = (mel[f'{PREFIX}PATH_procedure_count'] >= 3).astype(int)

    mel[f'{PREFIX}TX_oral_skin_abx'] = has_oral_skin_abx
    mel[f'{PREFIX}TX_oral_abx_with_surgery'] = ((has_oral_skin_abx == 1) & (has_any_surgical_med == 1)).astype(int)
    mel[f'{PREFIX}TX_antifungal_systemic'] = has_antifungal_sys
    mel[f'{PREFIX}TX_antiviral_systemic'] = has_antiviral_sys

    # Surgical escalation: surgical med + procedure count + histology
    mel[f'{PREFIX}TX_surgical_workup_complete'] = (
        (has_any_surgical_med == 1) & (has_excision == 1) & (has_histology == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: COMPOSITE MELANOMA-SUSPICION SCORE
    # Hand-crafted weighted score. Personal H/O = top weight (recurrence).
    # ══════════════════════════════════════════════════════════
    susp_score = (
        has_personal_history.astype(int) * 4               # H/O melanoma — highest weight
        + has_excision.astype(int) * 3                     # wide excision = strong signal
        + has_histology.astype(int) * 2
        + has_7pcl.astype(int) * 2
        + has_dermoscopy.astype(int)
        + has_derm_ref.astype(int)
        + has_lesion.astype(int)
        + has_mole.astype(int)
        + has_biopsy.astype(int)
        + has_prior_bcc_scc.astype(int)
        + has_immunosuppression.astype(int)
        + has_actinic_tx.astype(int)
    )
    mel[f'{PREFIX}SUSPICION_score'] = susp_score
    mel[f'{PREFIX}SUSPICION_high'] = (susp_score >= 5).astype(int)
    mel[f'{PREFIX}SUSPICION_very_high'] = (susp_score >= 8).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: PERSISTENCE (≥2 events in window B)
    # ══════════════════════════════════════════════════════════
    lesion_B = _count(obs_B, 'SKIN_LESION_PRESENTATION')
    mole_B = _count(obs_B, 'MOLE_NAEVUS_PIGMENTED_LESION')
    pcl_B = _count(obs_B, 'NICE_7PCL_FEATURES')
    derm_B = _count(obs_B, 'DERMATOSCOPY_PHOTOGRAPHY')
    referral_B = _count(obs_B, 'DERMATOLOGY_REFERRAL_CLINIC')
    biopsy_B = _count(obs_B, 'PRIOR_SKIN_PROCEDURES_BIOPSY')
    histo_B = _count(obs_B, 'HISTOLOGY')

    mel[f'{PREFIX}PERSIST_lesion_B'] = (lesion_B >= 2).astype(int)
    mel[f'{PREFIX}PERSIST_mole_B'] = (mole_B >= 2).astype(int)
    mel[f'{PREFIX}PERSIST_pcl_B'] = (pcl_B >= 2).astype(int)
    mel[f'{PREFIX}PERSIST_referral_B'] = (referral_B >= 2).astype(int)
    mel[f'{PREFIX}PERSIST_biopsy_B'] = (biopsy_B >= 2).astype(int)
    mel[f'{PREFIX}PERSIST_count'] = (
        mel[f'{PREFIX}PERSIST_lesion_B']
        + mel[f'{PREFIX}PERSIST_mole_B']
        + mel[f'{PREFIX}PERSIST_pcl_B']
        + mel[f'{PREFIX}PERSIST_referral_B']
        + mel[f'{PREFIX}PERSIST_biopsy_B']
    )

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: MULTI-SYSTEM TRIFECTA (window B)
    # Lesion/mole + procedure (biopsy/excision) + investigation (dermoscopy/histology) = active workup.
    # ══════════════════════════════════════════════════════════
    lesion_or_mole_B = ((lesion_B + mole_B) > 0).astype(int)
    procedure_B = (
        (_count(obs_B, 'PRIOR_SKIN_PROCEDURES_BIOPSY')
         + _count(obs_B, 'WIDE_EXCISION_GRAFTING')
         + _count(obs_B, 'CRYOTHERAPY')) > 0
    ).astype(int)
    investigation_B = ((derm_B + histo_B) > 0).astype(int)
    pathway_B = ((referral_B + _count(obs_B, 'PLASTIC_SURGERY')) > 0).astype(int)
    multi_B = lesion_or_mole_B + procedure_B + investigation_B + pathway_B
    mel[f'{PREFIX}TRIFECTA_lesion_proc_inv_B'] = (multi_B >= 3).astype(int)
    mel[f'{PREFIX}MULTISYSTEM_PAIR_B'] = (multi_B >= 2).astype(int)
    mel[f'{PREFIX}MULTISYSTEM_count_B'] = multi_B

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: VELOCITY (B - A, clipped to non-negative)
    # ══════════════════════════════════════════════════════════
    lesion_A = _count(obs_A, 'SKIN_LESION_PRESENTATION')
    mole_A = _count(obs_A, 'MOLE_NAEVUS_PIGMENTED_LESION')
    pcl_A = _count(obs_A, 'NICE_7PCL_FEATURES')
    derm_A = _count(obs_A, 'DERMATOSCOPY_PHOTOGRAPHY')
    referral_A = _count(obs_A, 'DERMATOLOGY_REFERRAL_CLINIC')
    biopsy_A = _count(obs_A, 'PRIOR_SKIN_PROCEDURES_BIOPSY')
    histo_A = _count(obs_A, 'HISTOLOGY')
    excision_A = _count(obs_A, 'WIDE_EXCISION_GRAFTING')
    excision_B = _count(obs_B, 'WIDE_EXCISION_GRAFTING')

    mel[f'{PREFIX}VEL_lesion'] = (lesion_B - lesion_A).clip(lower=0)
    mel[f'{PREFIX}VEL_mole'] = (mole_B - mole_A).clip(lower=0)
    mel[f'{PREFIX}VEL_pcl'] = (pcl_B - pcl_A).clip(lower=0)
    mel[f'{PREFIX}VEL_dermoscopy'] = (derm_B - derm_A).clip(lower=0)
    mel[f'{PREFIX}VEL_referral'] = (referral_B - referral_A).clip(lower=0)
    mel[f'{PREFIX}VEL_biopsy'] = (biopsy_B - biopsy_A).clip(lower=0)
    mel[f'{PREFIX}VEL_histology'] = (histo_B - histo_A).clip(lower=0)
    mel[f'{PREFIX}VEL_excision'] = (excision_B - excision_A).clip(lower=0)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: NEW EVENT IN WINDOW B ONLY (B>0 & A==0)
    # ══════════════════════════════════════════════════════════
    mel[f'{PREFIX}NEW_lesion_B_only'] = ((lesion_B > 0) & (lesion_A == 0)).astype(int)
    mel[f'{PREFIX}NEW_mole_B_only'] = ((mole_B > 0) & (mole_A == 0)).astype(int)
    mel[f'{PREFIX}NEW_pcl_B_only'] = ((pcl_B > 0) & (pcl_A == 0)).astype(int)
    mel[f'{PREFIX}NEW_dermoscopy_B_only'] = ((derm_B > 0) & (derm_A == 0)).astype(int)
    mel[f'{PREFIX}NEW_referral_B_only'] = ((referral_B > 0) & (referral_A == 0)).astype(int)
    mel[f'{PREFIX}NEW_biopsy_B_only'] = ((biopsy_B > 0) & (biopsy_A == 0)).astype(int)
    mel[f'{PREFIX}NEW_histology_B_only'] = ((histo_B > 0) & (histo_A == 0)).astype(int)
    mel[f'{PREFIX}NEW_excision_B_only'] = ((excision_B > 0) & (excision_A == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: AGE × CLINICAL INTERACTIONS
    # Peak melanoma incidence ~65. Anchor PEAK_DIST to 65.
    # ══════════════════════════════════════════════════════════
    mel[f'{PREFIX}AGEX_PEAK_DIST'] = (age - 65).abs()
    mel[f'{PREFIX}AGEX_OVER_50'] = (age >= 50).astype(int)
    mel[f'{PREFIX}AGEX_OVER_60'] = (age >= 60).astype(int)
    mel[f'{PREFIX}AGEX_OVER_70'] = (age >= 70).astype(int)
    mel[f'{PREFIX}AGEX_OVER_80'] = (age >= 80).astype(int)
    mel[f'{PREFIX}AGEX_DECILE'] = pd.qcut(
        age.rank(method='first'), q=10, labels=False, duplicates='drop'
    ).fillna(0).astype(int)

    # Age × event counts (window B)
    mel[f'{PREFIX}AGEX_LESION_count_B'] = age * lesion_B
    mel[f'{PREFIX}AGEX_MOLE_count_B'] = age * mole_B
    mel[f'{PREFIX}AGEX_PCL_count_B'] = age * pcl_B
    mel[f'{PREFIX}AGEX_DERM_count_B'] = age * derm_B
    mel[f'{PREFIX}AGEX_REFERRAL_count_B'] = age * referral_B
    mel[f'{PREFIX}AGEX_BIOPSY_count_B'] = age * biopsy_B
    mel[f'{PREFIX}AGEX_HISTOLOGY_count_B'] = age * histo_B
    mel[f'{PREFIX}AGEX_EXCISION_count_B'] = age * excision_B

    # Age × acceleration
    mel[f'{PREFIX}AGEX_LESION_acceleration'] = age * (lesion_B - lesion_A).clip(lower=0)
    mel[f'{PREFIX}AGEX_MOLE_acceleration'] = age * (mole_B - mole_A).clip(lower=0)
    mel[f'{PREFIX}AGEX_BIOPSY_acceleration'] = age * (biopsy_B - biopsy_A).clip(lower=0)
    mel[f'{PREFIX}AGEX_PROCEDURE_total'] = age * procedure_total

    # Age × NEW
    mel[f'{PREFIX}AGEX_NEW_lesion_B'] = age * mel[f'{PREFIX}NEW_lesion_B_only']
    mel[f'{PREFIX}AGEX_NEW_mole_B'] = age * mel[f'{PREFIX}NEW_mole_B_only']
    mel[f'{PREFIX}AGEX_NEW_biopsy_B'] = age * mel[f'{PREFIX}NEW_biopsy_B_only']
    mel[f'{PREFIX}AGEX_NEW_excision_B'] = age * mel[f'{PREFIX}NEW_excision_B_only']

    # Age × risk factors / treatment
    mel[f'{PREFIX}AGEX_PERSONAL_HISTORY'] = age * has_personal_history.astype(float)
    mel[f'{PREFIX}AGEX_PRIOR_BCC_SCC'] = age * has_prior_bcc_scc.astype(float)
    mel[f'{PREFIX}AGEX_SUN_DAMAGE'] = age * has_sun_damage.astype(float)
    mel[f'{PREFIX}AGEX_IMMUNOSUPPRESSED'] = age * has_immunosuppression.astype(float)
    mel[f'{PREFIX}AGEX_ACTINIC_TX'] = age * has_actinic_tx.astype(float)
    mel[f'{PREFIX}AGEX_LOCAL_ANAES'] = age * has_local_anaes.astype(float)

    # Composite "elderly + clinical signal" (over 65)
    elderly = (age >= 65).astype(int)
    mel[f'{PREFIX}ELDERLY_WITH_LESION'] = (elderly & has_lesion)
    mel[f'{PREFIX}ELDERLY_WITH_MOLE'] = (elderly & has_mole)
    mel[f'{PREFIX}ELDERLY_WITH_PCL'] = (elderly & has_7pcl)
    mel[f'{PREFIX}ELDERLY_WITH_EXCISION'] = (elderly & has_excision)
    mel[f'{PREFIX}ELDERLY_WITH_HISTOLOGY'] = (elderly & has_histology)
    mel[f'{PREFIX}ELDERLY_WITH_PERSONAL_HX'] = (elderly & has_personal_history)
    mel[f'{PREFIX}ELDERLY_WITH_PRIOR_BCC_SCC'] = (elderly & has_prior_bcc_scc)
    mel[f'{PREFIX}ELDERLY_WITH_SUSPICION_HIGH'] = (elderly & mel[f'{PREFIX}SUSPICION_high'])

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: TRIPLE INTERACTIONS (age × signal × signal)
    # ══════════════════════════════════════════════════════════
    lesion_f = has_lesion.astype(float)
    mole_f = has_mole.astype(float)
    pcl_f = has_7pcl.astype(float)
    excision_f = has_excision.astype(float)
    histo_f = has_histology.astype(float)
    biopsy_f = has_biopsy.astype(float)
    personal_f = has_personal_history.astype(float)
    prior_bcc_f = has_prior_bcc_scc.astype(float)
    immuno_f = has_immunosuppression.astype(float)
    derm_f = has_dermoscopy.astype(float)

    mel[f'{PREFIX}TRIPLE_age_lesion_excision'] = age * lesion_f * excision_f
    mel[f'{PREFIX}TRIPLE_age_lesion_histology'] = age * lesion_f * histo_f
    mel[f'{PREFIX}TRIPLE_age_pcl_referral'] = age * pcl_f * has_derm_ref.astype(float)
    mel[f'{PREFIX}TRIPLE_age_pcl_excision'] = age * pcl_f * excision_f
    mel[f'{PREFIX}TRIPLE_age_mole_dermoscopy'] = age * mole_f * derm_f
    mel[f'{PREFIX}TRIPLE_age_personal_lesion'] = age * personal_f * lesion_f
    mel[f'{PREFIX}TRIPLE_age_prior_bcc_lesion'] = age * prior_bcc_f * lesion_f
    mel[f'{PREFIX}TRIPLE_age_immuno_lesion'] = age * immuno_f * lesion_f
    mel[f'{PREFIX}TRIPLE_age_lesionB_excisionB'] = age * lesion_B * excision_B
    mel[f'{PREFIX}TRIPLE_age_personal_excision'] = age * personal_f * excision_f

    # Signal pairs
    mel[f'{PREFIX}PAIR_personal_x_excision'] = personal_f * excision_f
    mel[f'{PREFIX}PAIR_pcl_x_excision'] = pcl_f * excision_f
    mel[f'{PREFIX}PAIR_pcl_x_histology'] = pcl_f * histo_f
    mel[f'{PREFIX}PAIR_lesion_x_histology'] = lesion_f * histo_f
    mel[f'{PREFIX}PAIR_biopsy_x_histology'] = biopsy_f * histo_f
    mel[f'{PREFIX}PAIR_dermoscopy_x_excision'] = derm_f * excision_f

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: COMORBIDITY BURDEN + ACTIVITY ESCALATION
    # ══════════════════════════════════════════════════════════
    n_categories = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(patients, fill_value=0).astype(int)
    mel[f'{PREFIX}COMORBIDITY_n_categories'] = n_categories
    mel[f'{PREFIX}COMORBIDITY_x_age'] = age * n_categories
    mel[f'{PREFIX}COMORBIDITY_high'] = (n_categories >= 5).astype(int)
    mel[f'{PREFIX}COMORBIDITY_x_lesion'] = n_categories * lesion_f
    mel[f'{PREFIX}COMORBIDITY_x_personal_hx'] = n_categories * personal_f

    events_B_total = obs_B.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    events_A_total = obs_A.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    mel[f'{PREFIX}ACTIVITY_events_B'] = events_B_total
    mel[f'{PREFIX}ACTIVITY_events_A'] = events_A_total
    mel[f'{PREFIX}ACTIVITY_velocity'] = (events_B_total - events_A_total).clip(lower=0)
    mel[f'{PREFIX}ACTIVITY_acceleration'] = (events_B_total / events_A_total.replace(0, 1)).clip(upper=10)
    mel[f'{PREFIX}AGEX_ACTIVITY_velocity'] = age * mel[f'{PREFIX}ACTIVITY_velocity']

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: INVESTIGATION × SIGNAL PAIRS
    # ══════════════════════════════════════════════════════════
    mel[f'{PREFIX}DERM_x_lesionB'] = derm_f * lesion_B
    mel[f'{PREFIX}DERM_x_moleB'] = derm_f * mole_B
    mel[f'{PREFIX}HISTO_x_lesionB'] = histo_f * lesion_B
    mel[f'{PREFIX}HISTO_x_pclB'] = histo_f * pcl_B
    mel[f'{PREFIX}REFERRAL_x_lesionB'] = has_derm_ref.astype(float) * lesion_B
    mel[f'{PREFIX}REFERRAL_x_pclB'] = has_derm_ref.astype(float) * pcl_B
    mel[f'{PREFIX}PATHWAY_x_age'] = age * (derm_f + histo_f + has_derm_ref.astype(float)).clip(upper=3)

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

    mel[f'{PREFIX}FIRST_lesion_months'] = _months_since_first('SKIN_LESION_PRESENTATION')
    mel[f'{PREFIX}FIRST_mole_months'] = _months_since_first('MOLE_NAEVUS_PIGMENTED_LESION')
    mel[f'{PREFIX}FIRST_pcl_months'] = _months_since_first('NICE_7PCL_FEATURES')
    mel[f'{PREFIX}FIRST_dermoscopy_months'] = _months_since_first('DERMATOSCOPY_PHOTOGRAPHY')
    mel[f'{PREFIX}FIRST_referral_months'] = _months_since_first('DERMATOLOGY_REFERRAL_CLINIC')
    mel[f'{PREFIX}FIRST_biopsy_months'] = _months_since_first('PRIOR_SKIN_PROCEDURES_BIOPSY')
    mel[f'{PREFIX}FIRST_histology_months'] = _months_since_first('HISTOLOGY')
    mel[f'{PREFIX}FIRST_excision_months'] = _months_since_first('WIDE_EXCISION_GRAFTING')
    mel[f'{PREFIX}FIRST_personal_hx_months'] = _months_since_first('PERSONAL_HISTORY')
    mel[f'{PREFIX}FIRST_sun_damage_months'] = _months_since_first('SUN_DAMAGE_PRE_MALIGNANT')

    for col_name in ['lesion', 'mole', 'pcl', 'dermoscopy', 'referral',
                     'biopsy', 'histology', 'excision']:
        first_col = f'{PREFIX}FIRST_{col_name}_months'
        mel[f'{PREFIX}RECENT_{col_name}'] = (
            (mel[first_col] >= 0) & (mel[first_col] < 12)
        ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 16: SEQUENTIAL X→Y PATTERNS (within Z days)
    # Captures the standard melanoma workup ordering.
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

    mel[f'{PREFIX}SEQ_lesion_then_dermoscopy_60d'] = _seq('SKIN_LESION_PRESENTATION', 'DERMATOSCOPY_PHOTOGRAPHY', 60)
    mel[f'{PREFIX}SEQ_lesion_then_referral_60d'] = _seq('SKIN_LESION_PRESENTATION', 'DERMATOLOGY_REFERRAL_CLINIC', 60)
    mel[f'{PREFIX}SEQ_lesion_then_biopsy_90d'] = _seq('SKIN_LESION_PRESENTATION', 'PRIOR_SKIN_PROCEDURES_BIOPSY', 90)
    mel[f'{PREFIX}SEQ_lesion_then_histology_90d'] = _seq('SKIN_LESION_PRESENTATION', 'HISTOLOGY', 90)
    mel[f'{PREFIX}SEQ_lesion_then_excision_180d'] = _seq('SKIN_LESION_PRESENTATION', 'WIDE_EXCISION_GRAFTING', 180)
    mel[f'{PREFIX}SEQ_mole_then_dermoscopy_60d'] = _seq('MOLE_NAEVUS_PIGMENTED_LESION', 'DERMATOSCOPY_PHOTOGRAPHY', 60)
    mel[f'{PREFIX}SEQ_pcl_then_excision_90d'] = _seq('NICE_7PCL_FEATURES', 'WIDE_EXCISION_GRAFTING', 90)
    mel[f'{PREFIX}SEQ_pcl_then_referral_30d'] = _seq('NICE_7PCL_FEATURES', 'DERMATOLOGY_REFERRAL_CLINIC', 30)
    mel[f'{PREFIX}SEQ_dermoscopy_then_excision_60d'] = _seq('DERMATOSCOPY_PHOTOGRAPHY', 'WIDE_EXCISION_GRAFTING', 60)
    mel[f'{PREFIX}SEQ_biopsy_then_histology_30d'] = _seq('PRIOR_SKIN_PROCEDURES_BIOPSY', 'HISTOLOGY', 30)
    mel[f'{PREFIX}SEQ_excision_then_plastic_60d'] = _seq('WIDE_EXCISION_GRAFTING', 'PLASTIC_SURGERY', 60)
    mel[f'{PREFIX}SEQ_personal_hx_then_lesion_180d'] = _seq('PERSONAL_HISTORY', 'SKIN_LESION_PRESENTATION', 180)

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: CHARLSON-LITE WEIGHTED COMORBIDITY (melanoma-tuned)
    # Personal H/O melanoma = top weight (recurrence). Wide excision +
    # histology = strong-Tx markers.
    # ══════════════════════════════════════════════════════════
    CHARLSON_LITE_MEL = {
        'PERSONAL_HISTORY': 4.0,                     # H/O melanoma — top recurrence signal
        'WIDE_EXCISION_GRAFTING': 3.0,               # definitive Tx
        'HISTOLOGY': 2.5,
        'NICE_7PCL_FEATURES': 2.5,                   # NICE NG14 cardinal
        'PRIOR_SKIN_CANCER_BCC_SCC': 2.0,
        'MOLE_NAEVUS_PIGMENTED_LESION': 1.5,
        'SKIN_LESION_PRESENTATION': 1.5,
        'DERMATOSCOPY_PHOTOGRAPHY': 1.5,
        'PRIOR_SKIN_PROCEDURES_BIOPSY': 2.0,
        'DERMATOLOGY_REFERRAL_CLINIC': 1.5,
        'PLASTIC_SURGERY': 2.0,
        'CRYOTHERAPY': 1.0,
        'SUN_DAMAGE_PRE_MALIGNANT': 1.5,
        'FAMILY_HISTORY': 1.5,
        'RISK_FACTOR': 1.0,                          # Fitzpatrick
        'CLINICAL_SIGNS': 1.0,
        'OTHER_SKIN_LESIONS': 0.5,
        'SKIN_SYMPTOMS': 0.5,
        'SKIN_TREATMENT_WOUND': 0.5,
        'SUTURE_POST_OP_MINOR_SURGERY': 1.0,
        'MINOR_SURGERY_ADMIN': 0.5,
    }
    score = pd.Series(0.0, index=patients)
    for cat, w in CHARLSON_LITE_MEL.items():
        score = score + w * _has(obs_AB, cat).astype(float)
    mel[f'{PREFIX}COMORB_weighted_score'] = score
    mel[f'{PREFIX}COMORB_weighted_x_age'] = age * score
    mel[f'{PREFIX}COMORB_weighted_high'] = (score >= 5).astype(int)
    mel[f'{PREFIX}COMORB_weighted_x_personal_hx'] = score * personal_f
    mel[f'{PREFIX}COMORB_weighted_x_excision'] = score * excision_f
    mel[f'{PREFIX}COMORB_weighted_x_suspicion'] = score * mel[f'{PREFIX}SUSPICION_score'].astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 18: SYMPTOM BURST (≥3 distinct symptom categories in 30-day rolling window)
    # Note: melanoma SYMPTOM_CATEGORIES are mostly observation/lesion descriptors;
    # signal here is generally quieter than for symptomatic cancers.
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
        mel[f'{PREFIX}SYMPTOM_max_in_30d'] = burst.reindex(patients, fill_value=0).astype(int)
    else:
        mel[f'{PREFIX}SYMPTOM_max_in_30d'] = 0
    mel[f'{PREFIX}SYMPTOM_burst_3plus'] = (mel[f'{PREFIX}SYMPTOM_max_in_30d'] >= 3).astype(int)
    mel[f'{PREFIX}SYMPTOM_burst_4plus'] = (mel[f'{PREFIX}SYMPTOM_max_in_30d'] >= 4).astype(int)

    # ── Cleanup ───────────────────────────────────────────────
    mel = mel.fillna(0).replace([np.inf, -np.inf], 0)

    nunique = mel.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        mel = mel.drop(columns=constant)
        logger.info(f"  Removed {len(constant)} constant columns")

    logger.info(f"  Melanoma-specific features: {mel.shape[1]}")
    return mel
