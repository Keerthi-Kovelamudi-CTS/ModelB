# ═══════════════════════════════════════════════════════════════
# LYMPHOMA — CANCER-SPECIFIC FEATURES
# Nodal/mass features, B-symptoms, lab prognostic (IPI surrogates:
# LDH/B2M/CRP/Ig/paraprotein), infection pattern (immunocompromise),
# treatment/steroid flags, IPI-like composite score, risk factors.
#
# Ported from Lymphoma_Cancer/2_Feature_Engineering/4_Feature_engineering.py
# (build_lymphoma_features) and adapted to the new config-driven pipeline.
# Plus generic structural blocks (8-18) shared with Bladder/Ovarian/Prostate.
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
    Lymphoma-specific features:
      - Nodal/mass (lymphadenopathy + splenomegaly + unexplained masses)
      - B-symptoms (drenching sweats, fever, weight loss) + constitutional
      - Lab prognostic (IPI surrogate: LDH, B2M, CRP, ESR, calcium, Ig, kappa/lambda, paraprotein)
      - Infection pattern (recurrent infections, antibiotic / antifungal / antiviral burden)
      - Treatment/steroid flags (corticosteroids = lymphoma Tx proxy)
      - IPI-like composite score
      - Risk factors (peak-age, elderly, sex)
      - Plus generic structural blocks 8-18 (trifecta, velocity, AGE×, etc.)
    """
    if cfg is None:
        cfg = config
    PREFIX = cfg.PREFIX  # 'LYMP_'

    logger.info(f"  LYMPHOMA-SPECIFIC FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patients = existing_fm.index
    lym = pd.DataFrame(index=patients)
    lym.index.name = 'PATIENT_GUID'

    obs_AB = clin[clin['TIME_WINDOW'].isin(['A', 'B'])].copy()
    obs_A = clin[clin['TIME_WINDOW'] == 'A'].copy()
    obs_B = clin[clin['TIME_WINDOW'] == 'B'].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy() if 'TIME_WINDOW' in med.columns else med
    med_A = med[med['TIME_WINDOW'] == 'A'].copy() if 'TIME_WINDOW' in med.columns else med.iloc[:0]
    med_B = med[med['TIME_WINDOW'] == 'B'].copy() if 'TIME_WINDOW' in med.columns else med.iloc[:0]
    lab_AB = obs_AB[obs_AB['VALUE'].notna()].copy()

    age = (existing_fm['AGE_AT_INDEX'].reindex(patients).fillna(60)
           if 'AGE_AT_INDEX' in existing_fm.columns
           else pd.Series(60.0, index=patients))
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

    def _lab_latest(term_pattern, category='LAB_MARKERS'):
        sub = lab_AB[lab_AB['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last().reindex(patients)

    def _lab_first(term_pattern, category='LAB_MARKERS'):
        sub = lab_AB[lab_AB['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first().reindex(patients)

    def _lab_max(term_pattern, category='LAB_MARKERS'):
        sub = lab_AB[lab_AB['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.groupby('PATIENT_GUID')['VALUE'].max().reindex(patients)

    def _lab_term_present(term_pattern, category='LAB_MARKERS'):
        """1 if any lab row matches term (regardless of value)."""
        sub = lab_AB[lab_AB['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(0, index=patients, dtype=int)
        present = sub['PATIENT_GUID'].unique()
        return pd.Series([1 if p in set(present) else 0 for p in patients], index=patients, dtype=int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: NODAL / MASS FEATURES
    # Lymphadenopathy + splenomegaly + unexplained masses = nodal triad.
    # ══════════════════════════════════════════════════════════
    has_lymphadenopathy = _has(obs_AB, 'LYMPHADENOPATHY')
    has_splenomegaly = _has(obs_AB, 'SPLENOMEGALY')
    has_masses = _has(obs_AB, 'UNEXPLAINED_MASSES')

    lym[f'{PREFIX}has_lymphadenopathy'] = has_lymphadenopathy
    lym[f'{PREFIX}has_splenomegaly'] = has_splenomegaly
    lym[f'{PREFIX}has_unexplained_masses'] = has_masses

    nodal_cats = ['LYMPHADENOPATHY', 'UNEXPLAINED_MASSES', 'SPLENOMEGALY']
    nodal_A = obs_A[obs_A['CATEGORY'].isin(nodal_cats)].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    nodal_B = obs_B[obs_B['CATEGORY'].isin(nodal_cats)].groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    nodal_total = nodal_A + nodal_B
    lym[f'{PREFIX}nodal_count_A'] = nodal_A
    lym[f'{PREFIX}nodal_count_B'] = nodal_B
    lym[f'{PREFIX}nodal_count_total'] = nodal_total
    lym[f'{PREFIX}nodal_acceleration'] = (nodal_B > nodal_A).astype(int)
    lym[f'{PREFIX}nodal_unique_categories'] = (
        has_lymphadenopathy + has_splenomegaly + has_masses
    )
    lym[f'{PREFIX}nodal_triad'] = (
        (has_lymphadenopathy == 1) & (has_splenomegaly == 1) & (has_masses == 1)
    ).astype(int)
    lym[f'{PREFIX}nodal_pair'] = (lym[f'{PREFIX}nodal_unique_categories'] >= 2).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: B SYMPTOMS / CONSTITUTIONAL
    # Drenching night sweats + unexplained fever + ≥10% weight loss = classic B-symptoms.
    # ══════════════════════════════════════════════════════════
    has_bsymptoms = _has(obs_AB, 'B_SYMPTOMS')
    has_constitutional = _has(obs_AB, 'CONSTITUTIONAL_SYMPTOMS')
    bsym_A = _count(obs_A, 'B_SYMPTOMS')
    bsym_B = _count(obs_B, 'B_SYMPTOMS')
    constit_A = _count(obs_A, 'CONSTITUTIONAL_SYMPTOMS')
    constit_B = _count(obs_B, 'CONSTITUTIONAL_SYMPTOMS')

    lym[f'{PREFIX}has_bsymptoms'] = has_bsymptoms
    lym[f'{PREFIX}has_constitutional'] = has_constitutional
    lym[f'{PREFIX}bsymptoms_count_A'] = bsym_A
    lym[f'{PREFIX}bsymptoms_count_B'] = bsym_B
    lym[f'{PREFIX}bsymptoms_acceleration'] = (bsym_B > bsym_A).astype(int)
    lym[f'{PREFIX}bsymptoms_persistent'] = (bsym_B >= 2).astype(int)
    lym[f'{PREFIX}constitutional_count_B'] = constit_B
    lym[f'{PREFIX}bsymptoms_or_constitutional'] = (
        (has_bsymptoms == 1) | (has_constitutional == 1)
    ).astype(int)
    lym[f'{PREFIX}bsymptoms_with_lymphadenopathy'] = (
        (has_bsymptoms == 1) & (has_lymphadenopathy == 1)
    ).astype(int)
    lym[f'{PREFIX}bsymptoms_with_splenomegaly'] = (
        (has_bsymptoms == 1) & (has_splenomegaly == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: LAB PROGNOSTIC FEATURES (IPI surrogates)
    # LDH (tumour burden), B2M, CRP/ESR (inflammation), calcium, Ig, paraprotein.
    # All filtered from LAB_MARKERS category by term pattern.
    # ══════════════════════════════════════════════════════════

    # LDH (tumour burden marker — IPI component)
    ldh_last = _lab_latest(r'lactate\s+dehydrogenase|^LDH\b')
    ldh_first = _lab_first(r'lactate\s+dehydrogenase|^LDH\b')
    lym[f'{PREFIX}LAB_ldh_latest'] = ldh_last.fillna(0)
    lym[f'{PREFIX}LAB_ldh_elevated'] = (ldh_last.fillna(0) > 250).astype(int)
    lym[f'{PREFIX}LAB_ldh_very_high'] = (ldh_last.fillna(0) > 500).astype(int)
    ldh_rise = (ldh_last - ldh_first).fillna(0)
    lym[f'{PREFIX}LAB_ldh_rising'] = (ldh_rise > 0).astype(int)
    lym[f'{PREFIX}LAB_ldh_rapid_rise'] = (ldh_rise > 100).astype(int)

    # Beta-2 microglobulin (IPI component)
    b2m_last = _lab_latest(r'beta[\s-]*2[\s-]*microglobulin')
    lym[f'{PREFIX}LAB_b2m_latest'] = b2m_last.fillna(0)
    lym[f'{PREFIX}LAB_b2m_elevated'] = (b2m_last.fillna(0) > 2.5).astype(int)

    # CRP
    crp_last = _lab_latest(r'C[\s-]*reactive\s+protein|CRP')
    crp_max = _lab_max(r'C[\s-]*reactive\s+protein|CRP')
    lym[f'{PREFIX}LAB_crp_latest'] = crp_last.fillna(0)
    lym[f'{PREFIX}LAB_crp_elevated'] = (crp_last.fillna(0) > 10).astype(int)
    lym[f'{PREFIX}LAB_crp_high'] = (crp_max.fillna(0) > 30).astype(int)

    # ESR
    esr_last = _lab_latest(r'Erythrocyte\s+sedimentation|^ESR\b')
    lym[f'{PREFIX}LAB_esr_latest'] = esr_last.fillna(0)
    lym[f'{PREFIX}LAB_esr_elevated'] = (esr_last.fillna(0) > 30).astype(int)

    # Calcium (hypercalcaemia)
    ca_last = _lab_latest(r'(?:adjusted\s+)?calcium')
    lym[f'{PREFIX}LAB_calcium_high'] = (ca_last.fillna(0) > 2.6).astype(int)

    # Bilirubin (liver involvement)
    bili_last = _lab_latest(r'bilirubin')
    lym[f'{PREFIX}LAB_bilirubin_elevated'] = (bili_last.fillna(0) > 21).astype(int)

    # GFR (renal — relevant for treatment dosing)
    gfr_last = _lab_latest(r'eGFR|glomerular\s+filtration')
    lym[f'{PREFIX}LAB_gfr_low'] = (gfr_last.fillna(999) < 60).astype(int)

    # Vitamin D (low = associated with worse lymphoma outcomes)
    vitd_last = _lab_latest(r'vitamin\s*D|25[\s-]*OH')
    lym[f'{PREFIX}LAB_vitd_low'] = (vitd_last.fillna(999) < 30).astype(int)

    # Immunoglobulins (Waldenstrom's hint = high IgM; immunoparesis = all low)
    igm_last = _lab_latest(r'Immunoglobulin\s*M\b|^IgM\b')
    iga_last = _lab_latest(r'Immunoglobulin\s*A\b|^IgA\b')
    igg_last = _lab_latest(r'Immunoglobulin\s*G\b|^IgG\b')
    lym[f'{PREFIX}LAB_igm_elevated'] = (igm_last.fillna(0) > 3).astype(int)
    lym[f'{PREFIX}LAB_igm_very_high'] = (igm_last.fillna(0) > 10).astype(int)   # Waldenström
    lym[f'{PREFIX}LAB_iga_elevated'] = (iga_last.fillna(0) > 4).astype(int)
    lym[f'{PREFIX}LAB_igg_elevated'] = (igg_last.fillna(0) > 16).astype(int)
    lym[f'{PREFIX}LAB_igg_low'] = (igg_last.fillna(99) < 6).astype(int)        # immunoparesis
    lym[f'{PREFIX}LAB_iga_low'] = (iga_last.fillna(99) < 0.7).astype(int)
    lym[f'{PREFIX}LAB_igm_low'] = (igm_last.fillna(99) < 0.4).astype(int)
    immunoparesis_signals = (
        lym[f'{PREFIX}LAB_igg_low'].astype(int)
        + lym[f'{PREFIX}LAB_iga_low'].astype(int)
        + lym[f'{PREFIX}LAB_igm_low'].astype(int)
    )
    lym[f'{PREFIX}LAB_immunoparesis_2plus'] = (immunoparesis_signals >= 2).astype(int)
    lym[f'{PREFIX}LAB_immunoparesis_score'] = immunoparesis_signals

    # Kappa/lambda free light chain ratio
    kl_ratio = _lab_latest(r'Kappa.*lambda|kappa[/_\s]*lambda|light\s+chain\s+ratio|free\s+light\s+chain\s+ratio')
    lym[f'{PREFIX}LAB_kl_ratio_abnormal'] = (
        ((kl_ratio < 0.26) | (kl_ratio > 1.65)) & kl_ratio.notna()
    ).astype(int)
    lym[f'{PREFIX}LAB_kl_ratio_extreme'] = (
        ((kl_ratio < 0.1) | (kl_ratio > 8)) & kl_ratio.notna()
    ).astype(int)

    # Paraprotein presence (lymphoma / Waldenstrom / MGUS hint)
    lym[f'{PREFIX}LAB_paraprotein_present'] = _lab_term_present(r'paraprotein')
    lym[f'{PREFIX}LAB_immunofixation'] = _lab_term_present(r'immunofixation')

    # IPI biomarker pair (LDH↑ + B2M↑ = high tumour burden)
    lym[f'{PREFIX}LAB_ipi_pair'] = (
        lym[f'{PREFIX}LAB_ldh_elevated'].astype(int)
        * lym[f'{PREFIX}LAB_b2m_elevated'].astype(int)
    )

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: HAEMATOLOGICAL ABNORMALITIES + INFECTION PATTERN
    # Recurrent infections + viral markers (EBV/CMV) = immunocompromise.
    # ══════════════════════════════════════════════════════════
    has_haem_abn = _has(obs_AB, 'HAEMATOLOGICAL_ABNORMALITIES')
    has_infection_markers = _has(obs_AB, 'INFECTION_MARKERS')
    has_recurrent_inf = _has(obs_AB, 'RECURRENT_INFECTIONS')
    has_autoimmune = _has(obs_AB, 'AUTOIMMUNE_IMMUNE')
    haem_abn_B = _count(obs_B, 'HAEMATOLOGICAL_ABNORMALITIES')
    haem_abn_A = _count(obs_A, 'HAEMATOLOGICAL_ABNORMALITIES')

    lym[f'{PREFIX}has_haem_abn'] = has_haem_abn
    lym[f'{PREFIX}has_infection_markers'] = has_infection_markers
    lym[f'{PREFIX}has_recurrent_infections'] = has_recurrent_inf
    lym[f'{PREFIX}has_autoimmune'] = has_autoimmune
    lym[f'{PREFIX}haem_abn_count_B'] = haem_abn_B
    lym[f'{PREFIX}haem_abn_acceleration'] = (haem_abn_B > haem_abn_A).astype(int)

    abx_A = _count_med(med_A, 'ANTIBIOTICS')
    abx_B = _count_med(med_B, 'ANTIBIOTICS')
    afung_B = _count_med(med_B, 'ANTIFUNGALS')
    aviral_B = _count_med(med_B, 'ANTIVIRALS')
    lym[f'{PREFIX}antibiotic_count_A'] = abx_A
    lym[f'{PREFIX}antibiotic_count_B'] = abx_B
    lym[f'{PREFIX}antibiotic_acceleration'] = (abx_B > abx_A).astype(int)
    lym[f'{PREFIX}antifungal_count_B'] = afung_B
    lym[f'{PREFIX}antiviral_count_B'] = aviral_B
    lym[f'{PREFIX}broad_antimicrobial_count_B'] = abx_B + afung_B + aviral_B
    lym[f'{PREFIX}heavy_antibiotic_use'] = (abx_B >= 3).astype(int)
    lym[f'{PREFIX}any_atypical_antimicrobial'] = ((afung_B + aviral_B) > 0).astype(int)

    # Recurrent-infection × heavy-abx = strong immunoparesis signal
    lym[f'{PREFIX}recurrent_inf_with_heavy_abx'] = (
        (has_recurrent_inf == 1) & (lym[f'{PREFIX}heavy_antibiotic_use'] == 1)
    ).astype(int)
    lym[f'{PREFIX}recurrent_inf_with_immunoparesis'] = (
        (has_recurrent_inf == 1) & (lym[f'{PREFIX}LAB_immunoparesis_2plus'] == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: TREATMENT / STEROID PATTERNS
    # Corticosteroids = lymphoma Tx proxy. Anaemia treatment + nutritional supplements = late disease.
    # ══════════════════════════════════════════════════════════
    steroid_A = _count_med(med_A, 'CORTICOSTEROIDS')
    steroid_B = _count_med(med_B, 'CORTICOSTEROIDS')
    lym[f'{PREFIX}corticosteroid_count_A'] = steroid_A
    lym[f'{PREFIX}corticosteroid_count_B'] = steroid_B
    lym[f'{PREFIX}corticosteroid_acceleration'] = (steroid_B > steroid_A).astype(int)
    lym[f'{PREFIX}heavy_steroid_use'] = (steroid_B >= 3).astype(int)

    has_anaemia_tx = _has_med(med_AB, 'ANAEMIA_TREATMENT')
    has_nutritional = _has_med(med_AB, 'NUTRITIONAL_SUPPLEMENTS')
    lym[f'{PREFIX}has_anaemia_treatment'] = has_anaemia_tx
    lym[f'{PREFIX}has_nutritional_supplements'] = has_nutritional
    lym[f'{PREFIX}has_pain_antihistamine'] = _has_med(med_AB, 'PAIN_ANTIHISTAMINES')

    # Pruritus + antihistamine use (Hodgkin signature)
    has_skin = _has(obs_AB, 'SKIN_SYMPTOMS')
    lym[f'{PREFIX}has_skin_symptoms'] = has_skin
    lym[f'{PREFIX}pruritus_with_antihistamine'] = (
        (has_skin == 1) & (lym[f'{PREFIX}has_pain_antihistamine'] == 1)
    ).astype(int)
    lym[f'{PREFIX}pruritus_with_bsymptoms'] = (
        (has_skin == 1) & (has_bsymptoms == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: COMPOSITE IPI-LIKE SURROGATE SCORE
    # International Prognostic Index components (age, LDH, stage proxies).
    # ══════════════════════════════════════════════════════════
    lym[f'{PREFIX}IPI_score'] = (
        (age >= 60).astype(int)
        + lym[f'{PREFIX}LAB_ldh_elevated'].astype(int)
        + has_bsymptoms
        + has_lymphadenopathy
        + has_splenomegaly
    )
    lym[f'{PREFIX}IPI_high'] = (lym[f'{PREFIX}IPI_score'] >= 3).astype(int)
    lym[f'{PREFIX}IPI_very_high'] = (lym[f'{PREFIX}IPI_score'] >= 4).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: RISK FACTORS
    # Lymphoma age peaks: NHL ~60-70, Hodgkin bimodal (~30 + ~70). Slight male predominance.
    # ══════════════════════════════════════════════════════════
    lym[f'{PREFIX}RF_peak_age_nhl'] = ((age >= 50) & (age <= 80)).astype(int)
    lym[f'{PREFIX}RF_elderly'] = (age >= 70).astype(int)
    lym[f'{PREFIX}RF_male'] = (sex == 'M').astype(int)
    lym[f'{PREFIX}RF_over60_with_bsymptoms'] = ((age >= 60) & (has_bsymptoms == 1)).astype(int)
    lym[f'{PREFIX}RF_over60_with_nodal'] = ((age >= 60) & (has_lymphadenopathy == 1)).astype(int)
    lym[f'{PREFIX}RF_autoimmune_with_lymphadenopathy'] = (
        (has_autoimmune == 1) & (has_lymphadenopathy == 1)
    ).astype(int)
    lym[f'{PREFIX}RF_ebv_with_bsymptoms'] = (
        (has_infection_markers == 1) & (has_bsymptoms == 1)
    ).astype(int)
    lym[f'{PREFIX}RF_high_risk_condition'] = _has(obs_AB, 'HIGH_RISK_CONDITIONS')

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: MULTI-SYSTEM TRIFECTA (window B)
    # Nodal + B-symptoms + haematological abnormality = classic lymphoma triad.
    # ══════════════════════════════════════════════════════════
    nodal_flag_B = (nodal_B > 0).astype(int)
    bsym_flag_B = (bsym_B > 0).astype(int)
    haem_flag_B = (haem_abn_B > 0).astype(int)
    constit_flag_B = (constit_B > 0).astype(int)
    multi_B = nodal_flag_B + bsym_flag_B + haem_flag_B + constit_flag_B
    lym[f'{PREFIX}TRIFECTA_nodal_bsym_haem_B'] = (multi_B >= 3).astype(int)
    lym[f'{PREFIX}MULTISYSTEM_PAIR_B'] = (multi_B >= 2).astype(int)
    lym[f'{PREFIX}MULTISYSTEM_count_B'] = multi_B

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: SYMPTOM VELOCITY (B - A, clipped to non-negative)
    # ══════════════════════════════════════════════════════════
    lymphad_B = _count(obs_B, 'LYMPHADENOPATHY')
    lymphad_A = _count(obs_A, 'LYMPHADENOPATHY')
    spleno_B = _count(obs_B, 'SPLENOMEGALY')
    spleno_A = _count(obs_A, 'SPLENOMEGALY')
    masses_B = _count(obs_B, 'UNEXPLAINED_MASSES')
    masses_A = _count(obs_A, 'UNEXPLAINED_MASSES')
    skin_B = _count(obs_B, 'SKIN_SYMPTOMS')
    skin_A = _count(obs_A, 'SKIN_SYMPTOMS')
    img_B_count = _count(obs_B, 'ABNORMAL_IMAGING')
    img_A_count = _count(obs_A, 'ABNORMAL_IMAGING')
    rec_inf_B = _count(obs_B, 'RECURRENT_INFECTIONS')
    rec_inf_A = _count(obs_A, 'RECURRENT_INFECTIONS')

    lym[f'{PREFIX}VEL_lymphadenopathy'] = (lymphad_B - lymphad_A).clip(lower=0)
    lym[f'{PREFIX}VEL_splenomegaly'] = (spleno_B - spleno_A).clip(lower=0)
    lym[f'{PREFIX}VEL_masses'] = (masses_B - masses_A).clip(lower=0)
    lym[f'{PREFIX}VEL_bsymptoms'] = (bsym_B - bsym_A).clip(lower=0)
    lym[f'{PREFIX}VEL_constitutional'] = (constit_B - constit_A).clip(lower=0)
    lym[f'{PREFIX}VEL_skin'] = (skin_B - skin_A).clip(lower=0)
    lym[f'{PREFIX}VEL_imaging'] = (img_B_count - img_A_count).clip(lower=0)
    lym[f'{PREFIX}VEL_haem_abn'] = (haem_abn_B - haem_abn_A).clip(lower=0)
    lym[f'{PREFIX}VEL_recurrent_inf'] = (rec_inf_B - rec_inf_A).clip(lower=0)
    lym[f'{PREFIX}VEL_nodal_total'] = (nodal_B - nodal_A).clip(lower=0)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: NEW SYMPTOM IN WINDOW B ONLY (B>0 & A==0)
    # Recent-onset lymphadenopathy / B-symptoms = high suspicion.
    # ══════════════════════════════════════════════════════════
    lym[f'{PREFIX}NEW_lymphadenopathy_B_only'] = ((lymphad_B > 0) & (lymphad_A == 0)).astype(int)
    lym[f'{PREFIX}NEW_splenomegaly_B_only'] = ((spleno_B > 0) & (spleno_A == 0)).astype(int)
    lym[f'{PREFIX}NEW_masses_B_only'] = ((masses_B > 0) & (masses_A == 0)).astype(int)
    lym[f'{PREFIX}NEW_bsymptoms_B_only'] = ((bsym_B > 0) & (bsym_A == 0)).astype(int)
    lym[f'{PREFIX}NEW_haem_abn_B_only'] = ((haem_abn_B > 0) & (haem_abn_A == 0)).astype(int)
    lym[f'{PREFIX}NEW_skin_B_only'] = ((skin_B > 0) & (skin_A == 0)).astype(int)
    lym[f'{PREFIX}NEW_recurrent_inf_B_only'] = ((rec_inf_B > 0) & (rec_inf_A == 0)).astype(int)
    lym[f'{PREFIX}NEW_imaging_B_only'] = ((img_B_count > 0) & (img_A_count == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: AGE × CLINICAL INTERACTIONS
    # ══════════════════════════════════════════════════════════
    # Age-derived (peak NHL ~65)
    lym[f'{PREFIX}AGEX_PEAK_DIST'] = (age - 65).abs()
    lym[f'{PREFIX}AGEX_OVER_50'] = (age >= 50).astype(int)
    lym[f'{PREFIX}AGEX_OVER_60'] = (age >= 60).astype(int)
    lym[f'{PREFIX}AGEX_OVER_70'] = (age >= 70).astype(int)
    lym[f'{PREFIX}AGEX_OVER_80'] = (age >= 80).astype(int)
    lym[f'{PREFIX}AGEX_DECILE'] = pd.qcut(
        age.rank(method='first'), q=10, labels=False, duplicates='drop'
    ).fillna(0).astype(int)

    # Age × symptom counts (B)
    lym[f'{PREFIX}AGEX_LYMPHAD_count_B'] = age * lymphad_B
    lym[f'{PREFIX}AGEX_SPLENO_count_B'] = age * spleno_B
    lym[f'{PREFIX}AGEX_MASSES_count_B'] = age * masses_B
    lym[f'{PREFIX}AGEX_BSYM_count_B'] = age * bsym_B
    lym[f'{PREFIX}AGEX_CONSTIT_count_B'] = age * constit_B
    lym[f'{PREFIX}AGEX_HAEM_ABN_count_B'] = age * haem_abn_B
    lym[f'{PREFIX}AGEX_NODAL_count_B'] = age * nodal_B
    lym[f'{PREFIX}AGEX_REC_INF_count_B'] = age * rec_inf_B
    lym[f'{PREFIX}AGEX_IMAGING_count_B'] = age * img_B_count
    lym[f'{PREFIX}AGEX_SKIN_count_B'] = age * skin_B

    # Age × symptom acceleration
    lym[f'{PREFIX}AGEX_LYMPHAD_acceleration'] = age * (lymphad_B - lymphad_A).clip(lower=0)
    lym[f'{PREFIX}AGEX_BSYM_acceleration'] = age * (bsym_B - bsym_A).clip(lower=0)
    lym[f'{PREFIX}AGEX_NODAL_acceleration'] = age * (nodal_B - nodal_A).clip(lower=0)
    lym[f'{PREFIX}AGEX_HAEM_ABN_acceleration'] = age * (haem_abn_B - haem_abn_A).clip(lower=0)

    # Age × labs
    lym[f'{PREFIX}AGEX_LDH_elevated'] = age * lym[f'{PREFIX}LAB_ldh_elevated'].astype(float)
    lym[f'{PREFIX}AGEX_B2M_elevated'] = age * lym[f'{PREFIX}LAB_b2m_elevated'].astype(float)
    lym[f'{PREFIX}AGEX_CRP_elevated'] = age * lym[f'{PREFIX}LAB_crp_elevated'].astype(float)
    lym[f'{PREFIX}AGEX_PARAPROTEIN'] = age * lym[f'{PREFIX}LAB_paraprotein_present'].astype(float)
    lym[f'{PREFIX}AGEX_IMMUNOPARESIS'] = age * lym[f'{PREFIX}LAB_immunoparesis_2plus'].astype(float)

    # Age × NEW
    lym[f'{PREFIX}AGEX_NEW_lymphad_B'] = age * lym[f'{PREFIX}NEW_lymphadenopathy_B_only']
    lym[f'{PREFIX}AGEX_NEW_bsym_B'] = age * lym[f'{PREFIX}NEW_bsymptoms_B_only']
    lym[f'{PREFIX}AGEX_NEW_masses_B'] = age * lym[f'{PREFIX}NEW_masses_B_only']

    # Age × medications
    lym[f'{PREFIX}AGEX_STEROID_count_B'] = age * steroid_B
    lym[f'{PREFIX}AGEX_BROAD_ABX_B'] = age * lym[f'{PREFIX}broad_antimicrobial_count_B']

    # Composite "elderly + clinical signal"
    elderly = (age >= 65).astype(int)
    lym[f'{PREFIX}ELDERLY_WITH_LYMPHAD'] = (elderly & has_lymphadenopathy)
    lym[f'{PREFIX}ELDERLY_WITH_BSYM'] = (elderly & has_bsymptoms)
    lym[f'{PREFIX}ELDERLY_WITH_SPLENOMEGALY'] = (elderly & has_splenomegaly)
    lym[f'{PREFIX}ELDERLY_WITH_LDH_HIGH'] = (elderly & lym[f'{PREFIX}LAB_ldh_elevated'])
    lym[f'{PREFIX}ELDERLY_WITH_PARAPROTEIN'] = (elderly & lym[f'{PREFIX}LAB_paraprotein_present'])
    lym[f'{PREFIX}ELDERLY_WITH_IMMUNOPARESIS'] = (elderly & lym[f'{PREFIX}LAB_immunoparesis_2plus'])
    lym[f'{PREFIX}ELDERLY_NEW_LYMPHAD_B'] = (elderly & lym[f'{PREFIX}NEW_lymphadenopathy_B_only'])
    lym[f'{PREFIX}ELDERLY_NEW_BSYM_B'] = (elderly & lym[f'{PREFIX}NEW_bsymptoms_B_only'])

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: TRIPLE INTERACTIONS (age × symptom × biomarker)
    # ══════════════════════════════════════════════════════════
    ldh_high = lym[f'{PREFIX}LAB_ldh_elevated'].astype(float)
    b2m_high = lym[f'{PREFIX}LAB_b2m_elevated'].astype(float)
    para = lym[f'{PREFIX}LAB_paraprotein_present'].astype(float)
    immunoparesis = lym[f'{PREFIX}LAB_immunoparesis_2plus'].astype(float)
    bsym_f = has_bsymptoms.astype(float)
    lymphad_f = has_lymphadenopathy.astype(float)
    spleno_f = has_splenomegaly.astype(float)

    lym[f'{PREFIX}TRIPLE_age_lymphad_ldh'] = age * lymphad_f * ldh_high
    lym[f'{PREFIX}TRIPLE_age_bsym_ldh'] = age * bsym_f * ldh_high
    lym[f'{PREFIX}TRIPLE_age_lymphad_bsym'] = age * lymphad_f * bsym_f
    lym[f'{PREFIX}TRIPLE_age_lymphadB_ldh'] = age * lymphad_B * ldh_high
    lym[f'{PREFIX}TRIPLE_age_bsymB_ldh'] = age * bsym_B * ldh_high
    lym[f'{PREFIX}TRIPLE_age_lymphad_b2m'] = age * lymphad_f * b2m_high
    lym[f'{PREFIX}TRIPLE_age_spleno_ldh'] = age * spleno_f * ldh_high
    lym[f'{PREFIX}TRIPLE_age_bsym_immunoparesis'] = age * bsym_f * immunoparesis
    lym[f'{PREFIX}TRIPLE_age_lymphad_paraprotein'] = age * lymphad_f * para

    # Lab biomarker pairs
    lym[f'{PREFIX}LABPAIR_ldh_x_b2m'] = ldh_high * b2m_high
    lym[f'{PREFIX}LABPAIR_ldh_x_crp'] = ldh_high * lym[f'{PREFIX}LAB_crp_elevated'].astype(float)
    lym[f'{PREFIX}LABPAIR_para_x_immunoparesis'] = para * immunoparesis
    lym[f'{PREFIX}LABPAIR_kl_abnormal_x_para'] = lym[f'{PREFIX}LAB_kl_ratio_abnormal'].astype(float) * para

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: COMORBIDITY BURDEN + RECENT ACTIVITY ESCALATION
    # ══════════════════════════════════════════════════════════
    n_categories = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(patients, fill_value=0).astype(int)
    lym[f'{PREFIX}COMORBIDITY_n_categories'] = n_categories
    lym[f'{PREFIX}COMORBIDITY_x_age'] = age * n_categories
    lym[f'{PREFIX}COMORBIDITY_high'] = (n_categories >= 5).astype(int)
    lym[f'{PREFIX}COMORBIDITY_x_lymphad'] = n_categories * lymphad_f
    lym[f'{PREFIX}COMORBIDITY_x_bsym'] = n_categories * bsym_f

    events_B = obs_B.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    events_A = obs_A.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    lym[f'{PREFIX}ACTIVITY_events_B'] = events_B
    lym[f'{PREFIX}ACTIVITY_events_A'] = events_A
    lym[f'{PREFIX}ACTIVITY_velocity'] = (events_B - events_A).clip(lower=0)
    lym[f'{PREFIX}ACTIVITY_acceleration'] = (events_B / events_A.replace(0, 1)).clip(upper=10)
    lym[f'{PREFIX}AGEX_ACTIVITY_velocity'] = age * lym[f'{PREFIX}ACTIVITY_velocity']

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: INVESTIGATION × SYMPTOM PAIRS
    # ══════════════════════════════════════════════════════════
    has_imaging_f = _has(obs_AB, 'ABNORMAL_IMAGING').astype(float)
    has_lab_markers_f = _has(obs_AB, 'LAB_MARKERS').astype(float)

    lym[f'{PREFIX}IMG_x_lymphadB'] = has_imaging_f * lymphad_B
    lym[f'{PREFIX}IMG_x_bsymB'] = has_imaging_f * bsym_B
    lym[f'{PREFIX}IMG_x_splenoB'] = has_imaging_f * spleno_B
    lym[f'{PREFIX}IMG_x_masses'] = has_imaging_f * has_masses.astype(float)
    lym[f'{PREFIX}LAB_MARKERS_x_lymphad'] = has_lab_markers_f * lymphad_f
    lym[f'{PREFIX}LAB_MARKERS_x_bsym'] = has_lab_markers_f * bsym_f
    lym[f'{PREFIX}PATHWAY_x_age'] = age * (has_imaging_f + has_lab_markers_f).clip(upper=2)

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

    lym[f'{PREFIX}FIRST_lymphadenopathy_months'] = _months_since_first('LYMPHADENOPATHY')
    lym[f'{PREFIX}FIRST_splenomegaly_months'] = _months_since_first('SPLENOMEGALY')
    lym[f'{PREFIX}FIRST_masses_months'] = _months_since_first('UNEXPLAINED_MASSES')
    lym[f'{PREFIX}FIRST_bsymptoms_months'] = _months_since_first('B_SYMPTOMS')
    lym[f'{PREFIX}FIRST_constitutional_months'] = _months_since_first('CONSTITUTIONAL_SYMPTOMS')
    lym[f'{PREFIX}FIRST_haem_abn_months'] = _months_since_first('HAEMATOLOGICAL_ABNORMALITIES')
    lym[f'{PREFIX}FIRST_skin_months'] = _months_since_first('SKIN_SYMPTOMS')
    lym[f'{PREFIX}FIRST_recurrent_inf_months'] = _months_since_first('RECURRENT_INFECTIONS')
    lym[f'{PREFIX}FIRST_imaging_months'] = _months_since_first('ABNORMAL_IMAGING')
    lym[f'{PREFIX}FIRST_infection_markers_months'] = _months_since_first('INFECTION_MARKERS')

    for col_name in ['lymphadenopathy', 'splenomegaly', 'masses', 'bsymptoms',
                     'haem_abn', 'skin', 'recurrent_inf', 'constitutional']:
        first_col = f'{PREFIX}FIRST_{col_name}_months'
        lym[f'{PREFIX}RECENT_{col_name}'] = (
            (lym[first_col] >= 0) & (lym[first_col] < 12)
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

    lym[f'{PREFIX}SEQ_lymphad_then_imaging_60d'] = _seq('LYMPHADENOPATHY', 'ABNORMAL_IMAGING', 60)
    lym[f'{PREFIX}SEQ_lymphad_then_lab_30d'] = _seq('LYMPHADENOPATHY', 'LAB_MARKERS', 30)
    lym[f'{PREFIX}SEQ_bsym_then_imaging_90d'] = _seq('B_SYMPTOMS', 'ABNORMAL_IMAGING', 90)
    lym[f'{PREFIX}SEQ_bsym_then_lab_60d'] = _seq('B_SYMPTOMS', 'LAB_MARKERS', 60)
    lym[f'{PREFIX}SEQ_masses_then_imaging_30d'] = _seq('UNEXPLAINED_MASSES', 'ABNORMAL_IMAGING', 30)
    lym[f'{PREFIX}SEQ_spleno_then_imaging_30d'] = _seq('SPLENOMEGALY', 'ABNORMAL_IMAGING', 30)
    lym[f'{PREFIX}SEQ_haem_abn_then_lab_60d'] = _seq('HAEMATOLOGICAL_ABNORMALITIES', 'LAB_MARKERS', 60)
    lym[f'{PREFIX}SEQ_recurrent_inf_then_lymphad_180d'] = _seq('RECURRENT_INFECTIONS', 'LYMPHADENOPATHY', 180)
    lym[f'{PREFIX}SEQ_skin_then_lymphad_180d'] = _seq('SKIN_SYMPTOMS', 'LYMPHADENOPATHY', 180)

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: CHARLSON-LITE WEIGHTED COMORBIDITY (lymphoma-tuned)
    # ══════════════════════════════════════════════════════════
    CHARLSON_LITE_LYMPHOMA = {
        'LYMPHADENOPATHY': 3.0,                # primary red flag
        'SPLENOMEGALY': 2.5,                   # primary red flag
        'UNEXPLAINED_MASSES': 2.5,
        'B_SYMPTOMS': 3.0,                     # primary red flag (drenching sweats etc.)
        'CONSTITUTIONAL_SYMPTOMS': 1.5,
        'HAEMATOLOGICAL_ABNORMALITIES': 2.0,   # lymphocytosis / paraprotein / immunoparesis
        'INFECTION_MARKERS': 1.5,              # EBV / CMV serology
        'RECURRENT_INFECTIONS': 2.0,           # immunoparesis proxy
        'AUTOIMMUNE_IMMUNE': 1.5,              # lymphoma risk factor
        'SKIN_SYMPTOMS': 1.5,                  # Hodgkin pruritus / cutaneous T-cell
        'ABNORMAL_IMAGING': 1.5,
        'LAB_MARKERS': 1.0,
        'PAIN_SYMPTOMS': 1.0,                  # spinal cord compression
        'HIGH_RISK_CONDITIONS': 1.0,
        'RISK_SCORES': 0.5,
    }
    score = pd.Series(0.0, index=patients)
    for cat, w in CHARLSON_LITE_LYMPHOMA.items():
        score = score + w * _has(obs_AB, cat).astype(float)
    lym[f'{PREFIX}COMORB_weighted_score'] = score
    lym[f'{PREFIX}COMORB_weighted_x_age'] = age * score
    lym[f'{PREFIX}COMORB_weighted_high'] = (score >= 5).astype(int)
    lym[f'{PREFIX}COMORB_weighted_x_lymphad'] = score * lymphad_f
    lym[f'{PREFIX}COMORB_weighted_x_ldh'] = score * ldh_high
    lym[f'{PREFIX}COMORB_weighted_x_ipi'] = score * lym[f'{PREFIX}IPI_score'].astype(float)

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
        lym[f'{PREFIX}SYMPTOM_max_in_30d'] = burst.reindex(patients, fill_value=0).astype(int)
    else:
        lym[f'{PREFIX}SYMPTOM_max_in_30d'] = 0
    lym[f'{PREFIX}SYMPTOM_burst_3plus'] = (lym[f'{PREFIX}SYMPTOM_max_in_30d'] >= 3).astype(int)
    lym[f'{PREFIX}SYMPTOM_burst_4plus'] = (lym[f'{PREFIX}SYMPTOM_max_in_30d'] >= 4).astype(int)

    # ── Cleanup ───────────────────────────────────────────────
    lym = lym.fillna(0).replace([np.inf, -np.inf], 0)

    nunique = lym.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        lym = lym.drop(columns=constant)
        logger.info(f"  Removed {len(constant)} constant columns")

    logger.info(f"  Lymphoma-specific features: {lym.shape[1]}")
    return lym
