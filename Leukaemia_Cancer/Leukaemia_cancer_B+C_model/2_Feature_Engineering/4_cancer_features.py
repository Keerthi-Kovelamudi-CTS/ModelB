# ═══════════════════════════════════════════════════════════════
# LEUKAEMIA — CANCER-SPECIFIC FEATURES
# NICE NG47 cardinal symptoms (lymphadenopathy, B-symptoms, infection,
# bleeding/bruising, bone pain, skin/lymphoproliferative), mimic patterns
# (recurrent infection abx, anaemia treated as iron-deficient),
# lab thresholds (FBC abnormalities, blasts, LDH, INR, Ig, ESR/CRP),
# diagnostic pathway, treatment patterns (immunocompromise prophylaxis,
# allopurinol for tumour lysis, tranexamic for bleeding), risk factors.
#
# Ported from Leukaemia_Cancer/2_Feature_Engineering/4_Feature_engineering.py
# (build_leukaemia_features) and adapted to the new config-driven pipeline.
# Plus generic structural blocks (8-18) shared with Bladder/Ovarian/Lymphoma.
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
    Leukaemia-specific features:
      - NICE NG47 cardinals (6: lymphadenopathy, B-symptoms, infection,
        bleeding/bruising, bone pain, skin/lymphoproliferative)
      - Mimic patterns (infection-mimic abx, anaemia-mimic iron)
      - Lab thresholds (FBC abnormalities, blasts, LDH, ESR, CRP, INR/PT,
        immunoglobulins, large unstained cells)
      - Diagnostic pathway (blood count → haem inv → lymph workup gap)
      - Treatment patterns (abx escalation, immunocompromise prophylaxis,
        allopurinol for TLS, tranexamic, steroids)
      - Risk factors (peak age 50-80, elderly, male)
      - Plus generic structural blocks 8-18 (trifecta, velocity, AGE×, etc.)
    """
    if cfg is None:
        cfg = config
    PREFIX = cfg.PREFIX  # 'LEUK_'

    logger.info(f"  LEUKAEMIA-SPECIFIC FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patients = existing_fm.index
    leuk = pd.DataFrame(index=patients)
    leuk.index.name = 'PATIENT_GUID'

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

    def _lab_latest(term_pattern, category=None):
        sub = lab_AB
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last().reindex(patients)

    def _lab_max(term_pattern, category=None):
        sub = lab_AB
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.groupby('PATIENT_GUID')['VALUE'].max().reindex(patients)

    def _lab_min(term_pattern, category=None):
        sub = lab_AB
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(np.nan, index=patients)
        return sub.groupby('PATIENT_GUID')['VALUE'].min().reindex(patients)

    def _lab_term_present(term_pattern, category=None):
        sub = lab_AB
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        if sub.empty:
            return pd.Series(0, index=patients, dtype=int)
        present = set(sub['PATIENT_GUID'].unique())
        return pd.Series([1 if p in present else 0 for p in patients], index=patients, dtype=int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: NICE NG47 CARDINAL SYMPTOMS
    # 6 cardinals: lymphadenopathy, B-symptoms, infection, bleeding/bruising,
    # bone pain, skin/lymphoproliferative.
    # ══════════════════════════════════════════════════════════
    has_lymphad = _has(obs_AB, 'LYMPHADENOPATHY')
    has_bsym = _has(obs_AB, 'B_SYMPTOMS')
    has_infection = _has(obs_AB, 'INFECTION')
    has_bleeding = _has(obs_AB, 'BLEEDING_BRUISING')
    has_bonepain = _has(obs_AB, 'BONE_PAIN')
    has_skin = _has(obs_AB, 'SKIN_LYMPHOPROLIFERATIVE')
    has_blood_abn = _has(obs_AB, 'BLOOD_COUNT_ABNORMALITIES')
    has_gi = _has(obs_AB, 'GI_SYMPTOMS')

    leuk[f'{PREFIX}has_lymphadenopathy'] = has_lymphad
    leuk[f'{PREFIX}has_bsymptoms'] = has_bsym
    leuk[f'{PREFIX}has_infection'] = has_infection
    leuk[f'{PREFIX}has_bleeding'] = has_bleeding
    leuk[f'{PREFIX}has_bonepain'] = has_bonepain
    leuk[f'{PREFIX}has_skin'] = has_skin
    leuk[f'{PREFIX}has_blood_abn'] = has_blood_abn
    leuk[f'{PREFIX}has_gi'] = has_gi

    leuk[f'{PREFIX}cardinal_count'] = (
        has_lymphad + has_bsym + has_infection + has_bleeding + has_bonepain + has_skin
    )
    leuk[f'{PREFIX}2plus_cardinals'] = (leuk[f'{PREFIX}cardinal_count'] >= 2).astype(int)
    leuk[f'{PREFIX}3plus_cardinals'] = (leuk[f'{PREFIX}cardinal_count'] >= 3).astype(int)
    leuk[f'{PREFIX}4plus_cardinals'] = (leuk[f'{PREFIX}cardinal_count'] >= 4).astype(int)

    # Persistence in window B (≥2 events)
    inf_B = _count(obs_B, 'INFECTION')
    bleed_B = _count(obs_B, 'BLEEDING_BRUISING')
    bsym_B = _count(obs_B, 'B_SYMPTOMS')
    blood_B = _count(obs_B, 'BLOOD_COUNT_ABNORMALITIES')
    leuk[f'{PREFIX}persistent_infection'] = (inf_B >= 2).astype(int)
    leuk[f'{PREFIX}persistent_bleeding'] = (bleed_B >= 2).astype(int)
    leuk[f'{PREFIX}persistent_bsymptoms'] = (bsym_B >= 2).astype(int)
    leuk[f'{PREFIX}persistent_blood_abn'] = (blood_B >= 2).astype(int)
    leuk[f'{PREFIX}persistent_count'] = (
        leuk[f'{PREFIX}persistent_infection']
        + leuk[f'{PREFIX}persistent_bleeding']
        + leuk[f'{PREFIX}persistent_bsymptoms']
        + leuk[f'{PREFIX}persistent_blood_abn']
    )

    leuk[f'{PREFIX}over50_with_cardinal'] = ((age >= 50) & (leuk[f'{PREFIX}cardinal_count'] >= 1)).astype(int)
    leuk[f'{PREFIX}over50_with_2plus'] = ((age >= 50) & (leuk[f'{PREFIX}cardinal_count'] >= 2)).astype(int)
    leuk[f'{PREFIX}over60_with_blood_abn'] = ((age >= 60) & (has_blood_abn == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: MIMIC PATTERNS
    # Recurrent infection treated empirically with abx; anaemia treated
    # with iron supplements but no overt bleeding source.
    # ══════════════════════════════════════════════════════════
    abx_total = _count_med(med_AB, 'ANTIBIOTICS') + _count_med(med_AB, 'ANTIBIOTIC_IMMUNOCOMPROMISE')
    has_iron = _has_med(med_AB, 'IRON_SUPPLEMENTS')
    has_tranexamic = _has_med(med_AB, 'TRANEXAMIC_ACID')

    leuk[f'{PREFIX}infection_mimic'] = ((has_infection == 1) & (abx_total >= 2)).astype(int)
    leuk[f'{PREFIX}infection_mimic_recurrent'] = ((has_infection == 1) & (abx_total >= 3)).astype(int)
    leuk[f'{PREFIX}anaemia_mimic'] = ((has_iron == 1) & (has_bleeding == 0)).astype(int)
    leuk[f'{PREFIX}bleeding_treated_no_dx'] = ((has_tranexamic == 1) & (has_bleeding == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: LAB THRESHOLD FEATURES
    # FBC core: WBC / lymphocyte / neutrophil / Hb / platelets — all from
    # KEY_BLOOD_TESTS via TERM regex. Plus blasts (key signal), LDH (tumour
    # burden), ESR/CRP (inflammation), INR/PT (clotting in DIC/AML),
    # immunoglobulins (immunoparesis).
    # ══════════════════════════════════════════════════════════

    # WBC (white blood cell count) — leucocytosis or leucopenia
    wbc_last = _lab_latest(r'(?:total\s+)?white\s+(?:cell|blood\s+cell)\s+count|leucocyte\s+count|^WBC\b', 'KEY_BLOOD_TESTS')
    wbc_max = _lab_max(r'(?:total\s+)?white\s+(?:cell|blood\s+cell)\s+count|leucocyte\s+count|^WBC\b', 'KEY_BLOOD_TESTS')
    leuk[f'{PREFIX}LAB_wbc_latest'] = wbc_last.fillna(0)
    leuk[f'{PREFIX}LAB_wbc_high'] = (wbc_last.fillna(0) > 11).astype(int)         # leucocytosis
    leuk[f'{PREFIX}LAB_wbc_very_high'] = (wbc_last.fillna(0) > 30).astype(int)    # marked
    leuk[f'{PREFIX}LAB_wbc_extreme'] = (wbc_max.fillna(0) > 50).astype(int)       # CLL / hyperleucocytosis
    leuk[f'{PREFIX}LAB_wbc_low'] = ((wbc_last < 4) & wbc_last.notna()).astype(int)

    # Lymphocyte count — lymphocytosis is the CLL hallmark
    lymph_last = _lab_latest(r'lymphocyte\s+count|absolute\s+lymphocyte', 'KEY_BLOOD_TESTS')
    lymph_max = _lab_max(r'lymphocyte\s+count|absolute\s+lymphocyte', 'KEY_BLOOD_TESTS')
    leuk[f'{PREFIX}LAB_lymph_latest'] = lymph_last.fillna(0)
    leuk[f'{PREFIX}LAB_lymphocytosis'] = (lymph_last.fillna(0) > 5).astype(int)        # CLL diagnostic threshold
    leuk[f'{PREFIX}LAB_lymphocytosis_high'] = (lymph_last.fillna(0) > 10).astype(int)
    leuk[f'{PREFIX}LAB_lymphocytosis_extreme'] = (lymph_max.fillna(0) > 20).astype(int)
    leuk[f'{PREFIX}LAB_lymph_low'] = ((lymph_last < 1) & lymph_last.notna()).astype(int)  # lymphopenia

    # Neutrophil count — neutropenia is immunocompromise signal
    neut_last = _lab_latest(r'neutrophil\s+count|absolute\s+neutrophil', 'KEY_BLOOD_TESTS')
    leuk[f'{PREFIX}LAB_neut_latest'] = neut_last.fillna(0)
    leuk[f'{PREFIX}LAB_neutropenia'] = ((neut_last < 1.5) & neut_last.notna()).astype(int)
    leuk[f'{PREFIX}LAB_severe_neutropenia'] = ((neut_last < 0.5) & neut_last.notna()).astype(int)

    # Haemoglobin — anaemia
    hb_last = _lab_latest(r'haemoglobin|hemoglobin', 'KEY_BLOOD_TESTS')
    hb_min = _lab_min(r'haemoglobin|hemoglobin', 'KEY_BLOOD_TESTS')
    leuk[f'{PREFIX}LAB_hb_latest'] = hb_last.fillna(0)
    leuk[f'{PREFIX}LAB_hb_low'] = (hb_last.fillna(999) < 120).astype(int)
    leuk[f'{PREFIX}LAB_hb_severe'] = (hb_last.fillna(999) < 80).astype(int)
    leuk[f'{PREFIX}LAB_hb_min_low'] = (hb_min.fillna(999) < 100).astype(int)

    # Platelets — thrombocytopenia (bleeding diathesis)
    plt_last = _lab_latest(r'platelet\s+count|platelets', 'KEY_BLOOD_TESTS')
    plt_min = _lab_min(r'platelet\s+count|platelets', 'KEY_BLOOD_TESTS')
    leuk[f'{PREFIX}LAB_plt_latest'] = plt_last.fillna(0)
    leuk[f'{PREFIX}LAB_thrombocytopenia'] = ((plt_last < 150) & plt_last.notna()).astype(int)
    leuk[f'{PREFIX}LAB_thrombocytopenia_severe'] = ((plt_last < 50) & plt_last.notna()).astype(int)
    leuk[f'{PREFIX}LAB_thrombocytopenia_critical'] = ((plt_min < 20) & plt_min.notna()).astype(int)
    leuk[f'{PREFIX}LAB_thrombocytosis'] = (plt_last.fillna(0) > 450).astype(int)  # CML / reactive

    # Blast cells (KEY leukaemia diagnostic)
    blast_last = _lab_latest(r'blast\s+cell|blasts')
    leuk[f'{PREFIX}LAB_blasts_present'] = (blast_last.fillna(0) > 0).astype(int)
    leuk[f'{PREFIX}LAB_blasts_high'] = (blast_last.fillna(0) > 5).astype(int)    # >5% suggestive
    leuk[f'{PREFIX}LAB_blasts_very_high'] = (blast_last.fillna(0) > 20).astype(int)  # AML threshold

    # Large unstained cells (LUC) — abnormal cell flag
    leuk[f'{PREFIX}LAB_luc_tested'] = _lab_term_present(r'large\s+unstained\s+cells|^LUC\b')

    # Reticulocyte count — marrow response indicator
    leuk[f'{PREFIX}LAB_reticulocyte_tested'] = _lab_term_present(r'reticulocyte', 'KEY_BLOOD_TESTS')

    # LDH (tumour burden)
    ldh_last = _lab_latest(r'lactate\s+dehydrogenase|^LDH\b', 'HAEMATOLOGY_INVESTIGATIONS')
    leuk[f'{PREFIX}LAB_ldh_latest'] = ldh_last.fillna(0)
    leuk[f'{PREFIX}LAB_ldh_elevated'] = (ldh_last.fillna(0) > 250).astype(int)
    leuk[f'{PREFIX}LAB_ldh_high'] = (ldh_last.fillna(0) > 500).astype(int)

    # Beta-2 microglobulin (CLL prognosis)
    b2m_last = _lab_latest(r'beta[\s-]*2[\s-]*microglobulin', 'HAEMATOLOGY_INVESTIGATIONS')
    leuk[f'{PREFIX}LAB_b2m_elevated'] = (b2m_last.fillna(0) > 2.5).astype(int)

    # ESR (inflammation)
    esr_last = _lab_latest(r'Erythrocyte\s+sedimentation|^ESR\b', 'HAEMATOLOGY_INVESTIGATIONS')
    leuk[f'{PREFIX}LAB_esr_elevated'] = (esr_last.fillna(0) > 30).astype(int)
    leuk[f'{PREFIX}LAB_esr_high'] = (esr_last.fillna(0) > 50).astype(int)

    # CRP
    crp_last = _lab_latest(r'C[\s-]*reactive\s+protein|CRP', 'HAEMATOLOGY_INVESTIGATIONS')
    crp_max = _lab_max(r'C[\s-]*reactive\s+protein|CRP', 'HAEMATOLOGY_INVESTIGATIONS')
    leuk[f'{PREFIX}LAB_crp_elevated'] = (crp_last.fillna(0) > 10).astype(int)
    leuk[f'{PREFIX}LAB_crp_high'] = (crp_max.fillna(0) > 30).astype(int)
    leuk[f'{PREFIX}LAB_crp_with_infection'] = (
        (leuk[f'{PREFIX}LAB_crp_elevated'] == 1) & (has_infection == 1)
    ).astype(int)

    # Plasma viscosity (CLL / monoclonal gammopathy)
    leuk[f'{PREFIX}LAB_plasma_viscosity_tested'] = _lab_term_present(r'plasma\s+viscosity', 'HAEMATOLOGY_INVESTIGATIONS')

    # Coagulation: INR / PT / APTT — DIC / acute leukaemia
    inr_last = _lab_latest(r'international\s+normalised\s+ratio|^INR\b', 'HAEMATOLOGY_INVESTIGATIONS')
    pt_last = _lab_latest(r'prothrombin\s+time|^PT\b', 'HAEMATOLOGY_INVESTIGATIONS')
    aptt_last = _lab_latest(r'(?:activated\s+)?partial\s+thromboplastin\s+time|^APTT\b', 'HAEMATOLOGY_INVESTIGATIONS')
    leuk[f'{PREFIX}LAB_inr_elevated'] = (inr_last.fillna(0) > 1.5).astype(int)
    leuk[f'{PREFIX}LAB_pt_elevated'] = (pt_last.fillna(0) > 15).astype(int)
    leuk[f'{PREFIX}LAB_aptt_elevated'] = (aptt_last.fillna(0) > 35).astype(int)
    leuk[f'{PREFIX}LAB_coagulopathy'] = (
        (leuk[f'{PREFIX}LAB_inr_elevated'] == 1)
        | (leuk[f'{PREFIX}LAB_pt_elevated'] == 1)
        | (leuk[f'{PREFIX}LAB_aptt_elevated'] == 1)
    ).astype(int)

    # Immunoglobulins (immunoparesis = low IgG/IgA/IgM in CLL)
    igm_last = _lab_latest(r'Immunoglobulin\s*M\b|^IgM\b', 'HAEMATOLOGY_INVESTIGATIONS')
    iga_last = _lab_latest(r'Immunoglobulin\s*A\b|^IgA\b', 'HAEMATOLOGY_INVESTIGATIONS')
    igg_last = _lab_latest(r'Immunoglobulin\s*G\b|^IgG\b', 'HAEMATOLOGY_INVESTIGATIONS')
    leuk[f'{PREFIX}LAB_igg_low'] = (igg_last.fillna(99) < 6).astype(int)
    leuk[f'{PREFIX}LAB_iga_low'] = (iga_last.fillna(99) < 0.7).astype(int)
    leuk[f'{PREFIX}LAB_igm_low'] = (igm_last.fillna(99) < 0.4).astype(int)
    leuk[f'{PREFIX}LAB_any_ig_tested'] = _lab_term_present(r'Immunoglobulin', 'HAEMATOLOGY_INVESTIGATIONS')
    immunoparesis_signals = (
        leuk[f'{PREFIX}LAB_igg_low'].astype(int)
        + leuk[f'{PREFIX}LAB_iga_low'].astype(int)
        + leuk[f'{PREFIX}LAB_igm_low'].astype(int)
    )
    leuk[f'{PREFIX}LAB_immunoparesis_2plus'] = (immunoparesis_signals >= 2).astype(int)
    leuk[f'{PREFIX}LAB_immunoparesis_score'] = immunoparesis_signals

    # Paraprotein
    leuk[f'{PREFIX}LAB_paraprotein_present'] = _lab_term_present(r'paraprotein', 'HAEMATOLOGY_INVESTIGATIONS')

    # Cytopenia composite (anaemia + thrombocytopenia + neutropenia = pancytopenia)
    cytopenia_signals = (
        leuk[f'{PREFIX}LAB_hb_low'].astype(int)
        + leuk[f'{PREFIX}LAB_thrombocytopenia'].astype(int)
        + leuk[f'{PREFIX}LAB_neutropenia'].astype(int)
    )
    leuk[f'{PREFIX}LAB_cytopenia_score'] = cytopenia_signals
    leuk[f'{PREFIX}LAB_bicytopenia'] = (cytopenia_signals >= 2).astype(int)
    leuk[f'{PREFIX}LAB_pancytopenia'] = (cytopenia_signals >= 3).astype(int)

    # Auto-immune screen tests done
    leuk[f'{PREFIX}LAB_autoimmune_tested'] = _has(obs_AB, 'AUTOIMMUNE_SCREEN')

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: DIAGNOSTIC PATHWAY
    # FBC → haem investigations → lymph node procedure / imaging.
    # ══════════════════════════════════════════════════════════
    has_haem_inv = _has(obs_AB, 'HAEMATOLOGY_INVESTIGATIONS')
    has_lymph_proc = _has(obs_AB, 'LYMPH_NODE_PROCEDURE')
    has_imaging = _has(obs_AB, 'IMAGING')
    has_key_blood = _has(obs_AB, 'KEY_BLOOD_TESTS')

    leuk[f'{PREFIX}DX_has_blood_count'] = has_blood_abn
    leuk[f'{PREFIX}DX_has_key_blood_tests'] = has_key_blood
    leuk[f'{PREFIX}DX_has_haem_inv'] = has_haem_inv
    leuk[f'{PREFIX}DX_has_lymph_proc'] = has_lymph_proc
    leuk[f'{PREFIX}DX_has_imaging'] = has_imaging
    leuk[f'{PREFIX}DX_blood_then_haem'] = ((has_blood_abn == 1) & (has_haem_inv == 1)).astype(int)
    leuk[f'{PREFIX}DX_full_workup'] = (
        (has_haem_inv == 1) & (has_lymph_proc == 1) & (has_imaging == 1)
    ).astype(int)
    leuk[f'{PREFIX}DX_symptoms_no_investigation'] = (
        (leuk[f'{PREFIX}cardinal_count'] >= 2) & (has_haem_inv == 0) & (has_imaging == 0)
    ).astype(int)

    # Time from first symptom to first haem investigation
    sym_cats = ['INFECTION', 'BLEEDING_BRUISING', 'B_SYMPTOMS', 'BONE_PAIN',
                'LYMPHADENOPATHY', 'SKIN_LYMPHOPROLIFERATIVE']
    sym_first = obs_AB[obs_AB['CATEGORY'].isin(sym_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    haem_first = obs_AB[obs_AB['CATEGORY'] == 'HAEMATOLOGY_INVESTIGATIONS'].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap = (haem_first - sym_first).dt.days
    leuk[f'{PREFIX}DX_symptom_to_haem_days'] = gap.reindex(patients).fillna(-1)
    leuk[f'{PREFIX}DX_haem_inv_after_symptom'] = (leuk[f'{PREFIX}DX_symptom_to_haem_days'] >= 0).astype(int)
    leuk[f'{PREFIX}DX_haem_inv_delayed_60d'] = (leuk[f'{PREFIX}DX_symptom_to_haem_days'] > 60).astype(int)
    leuk[f'{PREFIX}DX_haem_inv_delayed_120d'] = (leuk[f'{PREFIX}DX_symptom_to_haem_days'] > 120).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: TREATMENT PATTERNS
    # Antibiotic escalation, immunocompromise prophylaxis (PCP/HSV/fungal),
    # tranexamic for bleeding, allopurinol for tumour-lysis-syndrome,
    # corticosteroids as Tx proxy.
    # ══════════════════════════════════════════════════════════
    has_abx = _has_med(med_AB, 'ANTIBIOTICS')
    has_abx_immuno = _has_med(med_AB, 'ANTIBIOTIC_IMMUNOCOMPROMISE')
    has_antifungal = _has_med(med_AB, 'ANTIFUNGALS_ORAL_SYSTEMIC')
    has_antiviral = _has_med(med_AB, 'ANTIVIRALS_SYSTEMIC')
    has_steroid = _has_med(med_AB, 'CORTICOSTEROIDS_SYSTEMIC')
    has_allopurinol = _has_med(med_AB, 'ALLOPURINOL_URIC_ACID')
    has_oral_ulcer_tx = _has_med(med_AB, 'ORAL_ULCER_INFECTION_TREATMENT')
    has_antihist = _has_med(med_AB, 'ANTIHISTAMINES_PRURITUS')

    abx_A = _count_med(med_A, 'ANTIBIOTICS')
    abx_B = _count_med(med_B, 'ANTIBIOTICS')
    steroid_B = _count_med(med_B, 'CORTICOSTEROIDS_SYSTEMIC')

    leuk[f'{PREFIX}TX_antibiotic_count_A'] = abx_A
    leuk[f'{PREFIX}TX_antibiotic_count_B'] = abx_B
    leuk[f'{PREFIX}TX_antibiotic_acceleration'] = (abx_B > abx_A).astype(int)
    leuk[f'{PREFIX}TX_heavy_antibiotic'] = (abx_B >= 3).astype(int)
    leuk[f'{PREFIX}TX_abx_escalation'] = ((has_abx == 1) & (has_abx_immuno == 1)).astype(int)
    leuk[f'{PREFIX}TX_antifungal_use'] = has_antifungal
    leuk[f'{PREFIX}TX_antifungal_with_infection'] = ((has_antifungal == 1) & (has_infection == 1)).astype(int)
    leuk[f'{PREFIX}TX_antiviral_with_infection'] = ((has_antiviral == 1) & (has_infection == 1)).astype(int)
    leuk[f'{PREFIX}TX_immunocompromise_markers'] = (
        (has_antifungal == 1) | (has_abx_immuno == 1) | (has_antiviral == 1)
    ).astype(int)
    leuk[f'{PREFIX}TX_immunocompromise_with_infection'] = (
        (leuk[f'{PREFIX}TX_immunocompromise_markers'] == 1) & (has_infection == 1)
    ).astype(int)

    leuk[f'{PREFIX}TX_steroid_use'] = has_steroid
    leuk[f'{PREFIX}TX_steroid_count_B'] = steroid_B
    leuk[f'{PREFIX}TX_heavy_steroid'] = (steroid_B >= 3).astype(int)
    leuk[f'{PREFIX}TX_steroid_with_bsymptoms'] = ((has_steroid == 1) & (has_bsym == 1)).astype(int)

    leuk[f'{PREFIX}TX_iron_supplementation'] = has_iron
    leuk[f'{PREFIX}TX_tranexamic'] = has_tranexamic
    leuk[f'{PREFIX}TX_tranexamic_with_bleeding'] = (
        (has_tranexamic == 1) & (has_bleeding == 1)
    ).astype(int)

    # Allopurinol = tumour-lysis-syndrome / hyperuricaemia signal — strong leukaemia hint
    leuk[f'{PREFIX}TX_allopurinol_use'] = has_allopurinol
    leuk[f'{PREFIX}TX_allopurinol_with_blood_abn'] = (
        (has_allopurinol == 1) & (has_blood_abn == 1)
    ).astype(int)

    leuk[f'{PREFIX}TX_oral_ulcer_treatment'] = has_oral_ulcer_tx
    leuk[f'{PREFIX}TX_pruritus_treated'] = ((has_antihist == 1) & (has_skin == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: RISK FACTORS
    # Peak age 50-80 (CLL ~70, AML ~65, ALL bimodal). Slight male predominance.
    # ══════════════════════════════════════════════════════════
    leuk[f'{PREFIX}RF_peak_age'] = ((age >= 50) & (age <= 80)).astype(int)
    leuk[f'{PREFIX}RF_elderly'] = (age >= 70).astype(int)
    leuk[f'{PREFIX}RF_male'] = (sex == 'M').astype(int)
    leuk[f'{PREFIX}RF_cardinal_with_age'] = ((age >= 60) & (leuk[f'{PREFIX}cardinal_count'] >= 1)).astype(int)
    leuk[f'{PREFIX}RF_blood_abn_with_age'] = ((age >= 60) & (has_blood_abn == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: COMPOSITE LEUKAEMIA-SUSPICION SCORE
    # Hand-crafted weighted score combining classic leukaemia features.
    # ══════════════════════════════════════════════════════════
    susp_score = (
        (age >= 60).astype(int)
        + leuk[f'{PREFIX}LAB_blasts_present'].astype(int) * 3      # blasts = highest weight
        + leuk[f'{PREFIX}LAB_lymphocytosis'].astype(int) * 2
        + leuk[f'{PREFIX}LAB_pancytopenia'].astype(int) * 2
        + leuk[f'{PREFIX}LAB_immunoparesis_2plus'].astype(int)
        + has_blood_abn
        + has_lymphad
        + has_bsym
        + has_bleeding
        + has_infection
    )
    leuk[f'{PREFIX}SUSPICION_score'] = susp_score
    leuk[f'{PREFIX}SUSPICION_high'] = (susp_score >= 4).astype(int)
    leuk[f'{PREFIX}SUSPICION_very_high'] = (susp_score >= 6).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: MULTI-SYSTEM TRIFECTA (window B)
    # Blood-count-abn + B-sym + (bleeding or infection) = primary leukaemia presentation.
    # ══════════════════════════════════════════════════════════
    blood_flag_B = (blood_B > 0).astype(int)
    bsym_flag_B = (bsym_B > 0).astype(int)
    bleed_or_inf_B = ((bleed_B + inf_B) > 0).astype(int)
    constit_B = bsym_B + _count(obs_B, 'BONE_PAIN')
    constit_flag_B = (constit_B > 0).astype(int)
    multi_B = blood_flag_B + bsym_flag_B + bleed_or_inf_B + constit_flag_B
    leuk[f'{PREFIX}TRIFECTA_blood_bsym_cytopen_consequence_B'] = (multi_B >= 3).astype(int)
    leuk[f'{PREFIX}MULTISYSTEM_PAIR_B'] = (multi_B >= 2).astype(int)
    leuk[f'{PREFIX}MULTISYSTEM_count_B'] = multi_B

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: SYMPTOM VELOCITY (B - A, clipped to non-negative)
    # ══════════════════════════════════════════════════════════
    blood_A = _count(obs_A, 'BLOOD_COUNT_ABNORMALITIES')
    bsym_A = _count(obs_A, 'B_SYMPTOMS')
    inf_A = _count(obs_A, 'INFECTION')
    bleed_A = _count(obs_A, 'BLEEDING_BRUISING')
    lymphad_B_count = _count(obs_B, 'LYMPHADENOPATHY')
    lymphad_A = _count(obs_A, 'LYMPHADENOPATHY')
    bone_B = _count(obs_B, 'BONE_PAIN')
    bone_A = _count(obs_A, 'BONE_PAIN')
    skin_B = _count(obs_B, 'SKIN_LYMPHOPROLIFERATIVE')
    skin_A = _count(obs_A, 'SKIN_LYMPHOPROLIFERATIVE')
    img_B = _count(obs_B, 'IMAGING')
    img_A = _count(obs_A, 'IMAGING')
    haem_inv_B = _count(obs_B, 'HAEMATOLOGY_INVESTIGATIONS')
    haem_inv_A = _count(obs_A, 'HAEMATOLOGY_INVESTIGATIONS')

    leuk[f'{PREFIX}VEL_blood_abn'] = (blood_B - blood_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_bsymptoms'] = (bsym_B - bsym_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_infection'] = (inf_B - inf_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_bleeding'] = (bleed_B - bleed_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_lymphadenopathy'] = (lymphad_B_count - lymphad_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_bone_pain'] = (bone_B - bone_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_skin'] = (skin_B - skin_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_imaging'] = (img_B - img_A).clip(lower=0)
    leuk[f'{PREFIX}VEL_haem_investigations'] = (haem_inv_B - haem_inv_A).clip(lower=0)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: NEW SYMPTOM IN WINDOW B ONLY (B>0 & A==0)
    # ══════════════════════════════════════════════════════════
    leuk[f'{PREFIX}NEW_blood_abn_B_only'] = ((blood_B > 0) & (blood_A == 0)).astype(int)
    leuk[f'{PREFIX}NEW_bsymptoms_B_only'] = ((bsym_B > 0) & (bsym_A == 0)).astype(int)
    leuk[f'{PREFIX}NEW_infection_B_only'] = ((inf_B > 0) & (inf_A == 0)).astype(int)
    leuk[f'{PREFIX}NEW_bleeding_B_only'] = ((bleed_B > 0) & (bleed_A == 0)).astype(int)
    leuk[f'{PREFIX}NEW_lymphad_B_only'] = ((lymphad_B_count > 0) & (lymphad_A == 0)).astype(int)
    leuk[f'{PREFIX}NEW_bone_pain_B_only'] = ((bone_B > 0) & (bone_A == 0)).astype(int)
    leuk[f'{PREFIX}NEW_skin_B_only'] = ((skin_B > 0) & (skin_A == 0)).astype(int)
    leuk[f'{PREFIX}NEW_haem_inv_B_only'] = ((haem_inv_B > 0) & (haem_inv_A == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: AGE × CLINICAL INTERACTIONS
    # Peak age varies: CLL ~70, AML ~65, ALL bimodal. Anchor PEAK_DIST to 65.
    # ══════════════════════════════════════════════════════════
    leuk[f'{PREFIX}AGEX_PEAK_DIST'] = (age - 65).abs()
    leuk[f'{PREFIX}AGEX_OVER_50'] = (age >= 50).astype(int)
    leuk[f'{PREFIX}AGEX_OVER_60'] = (age >= 60).astype(int)
    leuk[f'{PREFIX}AGEX_OVER_70'] = (age >= 70).astype(int)
    leuk[f'{PREFIX}AGEX_OVER_80'] = (age >= 80).astype(int)
    leuk[f'{PREFIX}AGEX_DECILE'] = pd.qcut(
        age.rank(method='first'), q=10, labels=False, duplicates='drop'
    ).fillna(0).astype(int)

    # Age × symptom counts
    leuk[f'{PREFIX}AGEX_BLOOD_count_B'] = age * blood_B
    leuk[f'{PREFIX}AGEX_BSYM_count_B'] = age * bsym_B
    leuk[f'{PREFIX}AGEX_INFECTION_count_B'] = age * inf_B
    leuk[f'{PREFIX}AGEX_BLEEDING_count_B'] = age * bleed_B
    leuk[f'{PREFIX}AGEX_LYMPHAD_count_B'] = age * lymphad_B_count
    leuk[f'{PREFIX}AGEX_BONE_count_B'] = age * bone_B
    leuk[f'{PREFIX}AGEX_SKIN_count_B'] = age * skin_B

    # Age × symptom acceleration
    leuk[f'{PREFIX}AGEX_BLOOD_acceleration'] = age * (blood_B - blood_A).clip(lower=0)
    leuk[f'{PREFIX}AGEX_BSYM_acceleration'] = age * (bsym_B - bsym_A).clip(lower=0)
    leuk[f'{PREFIX}AGEX_INFECTION_acceleration'] = age * (inf_B - inf_A).clip(lower=0)
    leuk[f'{PREFIX}AGEX_BLEEDING_acceleration'] = age * (bleed_B - bleed_A).clip(lower=0)

    # Age × labs
    leuk[f'{PREFIX}AGEX_BLASTS'] = age * leuk[f'{PREFIX}LAB_blasts_present'].astype(float)
    leuk[f'{PREFIX}AGEX_LYMPHOCYTOSIS'] = age * leuk[f'{PREFIX}LAB_lymphocytosis'].astype(float)
    leuk[f'{PREFIX}AGEX_PANCYTOPENIA'] = age * leuk[f'{PREFIX}LAB_pancytopenia'].astype(float)
    leuk[f'{PREFIX}AGEX_IMMUNOPARESIS'] = age * leuk[f'{PREFIX}LAB_immunoparesis_2plus'].astype(float)
    leuk[f'{PREFIX}AGEX_LDH_high'] = age * leuk[f'{PREFIX}LAB_ldh_elevated'].astype(float)
    leuk[f'{PREFIX}AGEX_COAGULOPATHY'] = age * leuk[f'{PREFIX}LAB_coagulopathy'].astype(float)

    # Age × NEW
    leuk[f'{PREFIX}AGEX_NEW_blood_B'] = age * leuk[f'{PREFIX}NEW_blood_abn_B_only']
    leuk[f'{PREFIX}AGEX_NEW_bsym_B'] = age * leuk[f'{PREFIX}NEW_bsymptoms_B_only']
    leuk[f'{PREFIX}AGEX_NEW_lymphad_B'] = age * leuk[f'{PREFIX}NEW_lymphad_B_only']

    # Age × treatment
    leuk[f'{PREFIX}AGEX_ALLOPURINOL'] = age * has_allopurinol.astype(float)
    leuk[f'{PREFIX}AGEX_TRANEXAMIC'] = age * has_tranexamic.astype(float)
    leuk[f'{PREFIX}AGEX_HEAVY_ABX'] = age * leuk[f'{PREFIX}TX_heavy_antibiotic'].astype(float)
    leuk[f'{PREFIX}AGEX_IMMUNOCOMPROMISE_RX'] = age * leuk[f'{PREFIX}TX_immunocompromise_markers'].astype(float)

    # Composite "elderly + clinical signal" (over 65)
    elderly = (age >= 65).astype(int)
    leuk[f'{PREFIX}ELDERLY_WITH_BLOOD_ABN'] = (elderly & has_blood_abn)
    leuk[f'{PREFIX}ELDERLY_WITH_BSYM'] = (elderly & has_bsym)
    leuk[f'{PREFIX}ELDERLY_WITH_LYMPHAD'] = (elderly & has_lymphad)
    leuk[f'{PREFIX}ELDERLY_WITH_BLEEDING'] = (elderly & has_bleeding)
    leuk[f'{PREFIX}ELDERLY_WITH_INFECTION'] = (elderly & has_infection)
    leuk[f'{PREFIX}ELDERLY_WITH_BLASTS'] = (elderly & leuk[f'{PREFIX}LAB_blasts_present'])
    leuk[f'{PREFIX}ELDERLY_WITH_LYMPHOCYTOSIS'] = (elderly & leuk[f'{PREFIX}LAB_lymphocytosis'])
    leuk[f'{PREFIX}ELDERLY_WITH_PANCYTOPENIA'] = (elderly & leuk[f'{PREFIX}LAB_pancytopenia'])
    leuk[f'{PREFIX}ELDERLY_WITH_IMMUNOPARESIS'] = (elderly & leuk[f'{PREFIX}LAB_immunoparesis_2plus'])

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: TRIPLE INTERACTIONS (age × symptom × biomarker)
    # ══════════════════════════════════════════════════════════
    blasts = leuk[f'{PREFIX}LAB_blasts_present'].astype(float)
    lymphocytosis = leuk[f'{PREFIX}LAB_lymphocytosis'].astype(float)
    pancytopenia = leuk[f'{PREFIX}LAB_pancytopenia'].astype(float)
    thrombocytopenia = leuk[f'{PREFIX}LAB_thrombocytopenia'].astype(float)
    neutropenia = leuk[f'{PREFIX}LAB_neutropenia'].astype(float)
    ldh_high = leuk[f'{PREFIX}LAB_ldh_elevated'].astype(float)
    immunoparesis = leuk[f'{PREFIX}LAB_immunoparesis_2plus'].astype(float)
    coag = leuk[f'{PREFIX}LAB_coagulopathy'].astype(float)

    leuk[f'{PREFIX}TRIPLE_age_blood_blasts'] = age * has_blood_abn * blasts
    leuk[f'{PREFIX}TRIPLE_age_lymphad_lymphocytosis'] = age * has_lymphad * lymphocytosis
    leuk[f'{PREFIX}TRIPLE_age_bsym_blasts'] = age * has_bsym * blasts
    leuk[f'{PREFIX}TRIPLE_age_bleeding_thrombocytopenia'] = age * has_bleeding * thrombocytopenia
    leuk[f'{PREFIX}TRIPLE_age_infection_neutropenia'] = age * has_infection * neutropenia
    leuk[f'{PREFIX}TRIPLE_age_blood_pancytopenia'] = age * has_blood_abn * pancytopenia
    leuk[f'{PREFIX}TRIPLE_age_lymphad_immunoparesis'] = age * has_lymphad * immunoparesis
    leuk[f'{PREFIX}TRIPLE_age_blood_ldh'] = age * has_blood_abn * ldh_high
    leuk[f'{PREFIX}TRIPLE_age_bleeding_coag'] = age * has_bleeding * coag

    # Lab biomarker pairs
    leuk[f'{PREFIX}LABPAIR_blasts_x_pancytopenia'] = blasts * pancytopenia
    leuk[f'{PREFIX}LABPAIR_lymphocytosis_x_immunoparesis'] = lymphocytosis * immunoparesis
    leuk[f'{PREFIX}LABPAIR_thrombocytopenia_x_coag'] = thrombocytopenia * coag       # DIC pattern
    leuk[f'{PREFIX}LABPAIR_blasts_x_ldh'] = blasts * ldh_high                          # high tumour burden

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: COMORBIDITY BURDEN + RECENT ACTIVITY ESCALATION
    # ══════════════════════════════════════════════════════════
    n_categories = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(patients, fill_value=0).astype(int)
    leuk[f'{PREFIX}COMORBIDITY_n_categories'] = n_categories
    leuk[f'{PREFIX}COMORBIDITY_x_age'] = age * n_categories
    leuk[f'{PREFIX}COMORBIDITY_high'] = (n_categories >= 5).astype(int)
    leuk[f'{PREFIX}COMORBIDITY_x_blood_abn'] = n_categories * has_blood_abn.astype(float)
    leuk[f'{PREFIX}COMORBIDITY_x_lymphad'] = n_categories * has_lymphad.astype(float)

    events_B = obs_B.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    events_A = obs_A.groupby('PATIENT_GUID').size().reindex(patients, fill_value=0).astype(float)
    leuk[f'{PREFIX}ACTIVITY_events_B'] = events_B
    leuk[f'{PREFIX}ACTIVITY_events_A'] = events_A
    leuk[f'{PREFIX}ACTIVITY_velocity'] = (events_B - events_A).clip(lower=0)
    leuk[f'{PREFIX}ACTIVITY_acceleration'] = (events_B / events_A.replace(0, 1)).clip(upper=10)
    leuk[f'{PREFIX}AGEX_ACTIVITY_velocity'] = age * leuk[f'{PREFIX}ACTIVITY_velocity']

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: INVESTIGATION × SYMPTOM PAIRS
    # ══════════════════════════════════════════════════════════
    img_f = has_imaging.astype(float)
    haem_inv_f = has_haem_inv.astype(float)
    lymph_proc_f = has_lymph_proc.astype(float)

    leuk[f'{PREFIX}IMG_x_lymphad'] = img_f * has_lymphad.astype(float)
    leuk[f'{PREFIX}IMG_x_blood_abn'] = img_f * has_blood_abn.astype(float)
    leuk[f'{PREFIX}HAEM_INV_x_blood_abn'] = haem_inv_f * has_blood_abn.astype(float)
    leuk[f'{PREFIX}HAEM_INV_x_bsym'] = haem_inv_f * has_bsym.astype(float)
    leuk[f'{PREFIX}HAEM_INV_x_lymphad'] = haem_inv_f * has_lymphad.astype(float)
    leuk[f'{PREFIX}LYMPH_PROC_x_lymphad'] = lymph_proc_f * has_lymphad.astype(float)
    leuk[f'{PREFIX}PATHWAY_x_age'] = age * (img_f + haem_inv_f + lymph_proc_f).clip(upper=3)

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

    leuk[f'{PREFIX}FIRST_blood_abn_months'] = _months_since_first('BLOOD_COUNT_ABNORMALITIES')
    leuk[f'{PREFIX}FIRST_bsymptoms_months'] = _months_since_first('B_SYMPTOMS')
    leuk[f'{PREFIX}FIRST_infection_months'] = _months_since_first('INFECTION')
    leuk[f'{PREFIX}FIRST_bleeding_months'] = _months_since_first('BLEEDING_BRUISING')
    leuk[f'{PREFIX}FIRST_lymphad_months'] = _months_since_first('LYMPHADENOPATHY')
    leuk[f'{PREFIX}FIRST_bone_pain_months'] = _months_since_first('BONE_PAIN')
    leuk[f'{PREFIX}FIRST_skin_months'] = _months_since_first('SKIN_LYMPHOPROLIFERATIVE')
    leuk[f'{PREFIX}FIRST_haem_inv_months'] = _months_since_first('HAEMATOLOGY_INVESTIGATIONS')
    leuk[f'{PREFIX}FIRST_imaging_months'] = _months_since_first('IMAGING')

    for col_name in ['blood_abn', 'bsymptoms', 'infection', 'bleeding',
                     'lymphad', 'bone_pain', 'skin']:
        first_col = f'{PREFIX}FIRST_{col_name}_months'
        leuk[f'{PREFIX}RECENT_{col_name}'] = (
            (leuk[first_col] >= 0) & (leuk[first_col] < 12)
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

    leuk[f'{PREFIX}SEQ_blood_abn_then_haem_inv_30d'] = _seq('BLOOD_COUNT_ABNORMALITIES', 'HAEMATOLOGY_INVESTIGATIONS', 30)
    leuk[f'{PREFIX}SEQ_blood_abn_then_haem_inv_60d'] = _seq('BLOOD_COUNT_ABNORMALITIES', 'HAEMATOLOGY_INVESTIGATIONS', 60)
    leuk[f'{PREFIX}SEQ_lymphad_then_lymph_proc_60d'] = _seq('LYMPHADENOPATHY', 'LYMPH_NODE_PROCEDURE', 60)
    leuk[f'{PREFIX}SEQ_lymphad_then_imaging_60d'] = _seq('LYMPHADENOPATHY', 'IMAGING', 60)
    leuk[f'{PREFIX}SEQ_bsym_then_haem_inv_90d'] = _seq('B_SYMPTOMS', 'HAEMATOLOGY_INVESTIGATIONS', 90)
    leuk[f'{PREFIX}SEQ_infection_then_blood_abn_180d'] = _seq('INFECTION', 'BLOOD_COUNT_ABNORMALITIES', 180)
    leuk[f'{PREFIX}SEQ_bleeding_then_blood_abn_60d'] = _seq('BLEEDING_BRUISING', 'BLOOD_COUNT_ABNORMALITIES', 60)
    leuk[f'{PREFIX}SEQ_bone_pain_then_haem_inv_180d'] = _seq('BONE_PAIN', 'HAEMATOLOGY_INVESTIGATIONS', 180)
    leuk[f'{PREFIX}SEQ_skin_then_lymphad_180d'] = _seq('SKIN_LYMPHOPROLIFERATIVE', 'LYMPHADENOPATHY', 180)

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: CHARLSON-LITE WEIGHTED COMORBIDITY (leukaemia-tuned)
    # ══════════════════════════════════════════════════════════
    CHARLSON_LITE_LEUK = {
        'BLOOD_COUNT_ABNORMALITIES': 3.0,    # primary red flag
        'B_SYMPTOMS': 2.5,                   # constitutional cardinal
        'LYMPHADENOPATHY': 2.5,              # cardinal
        'BLEEDING_BRUISING': 2.5,            # thrombocytopenia signal
        'INFECTION': 2.0,                    # immunocompromise signal
        'BONE_PAIN': 1.5,                    # marrow infiltration
        'SKIN_LYMPHOPROLIFERATIVE': 1.5,     # cutaneous lymphoproliferation
        'GI_SYMPTOMS': 1.0,
        'IMAGING': 1.0,
        'LYMPH_NODE_PROCEDURE': 2.0,         # invasive workup
        'HAEMATOLOGY_INVESTIGATIONS': 1.5,
        'AUTOIMMUNE_SCREEN': 1.0,
        'KEY_BLOOD_TESTS': 0.5,
    }
    score = pd.Series(0.0, index=patients)
    for cat, w in CHARLSON_LITE_LEUK.items():
        score = score + w * _has(obs_AB, cat).astype(float)
    leuk[f'{PREFIX}COMORB_weighted_score'] = score
    leuk[f'{PREFIX}COMORB_weighted_x_age'] = age * score
    leuk[f'{PREFIX}COMORB_weighted_high'] = (score >= 5).astype(int)
    leuk[f'{PREFIX}COMORB_weighted_x_blood_abn'] = score * has_blood_abn.astype(float)
    leuk[f'{PREFIX}COMORB_weighted_x_blasts'] = score * blasts
    leuk[f'{PREFIX}COMORB_weighted_x_suspicion'] = score * leuk[f'{PREFIX}SUSPICION_score'].astype(float)

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
        leuk[f'{PREFIX}SYMPTOM_max_in_30d'] = burst.reindex(patients, fill_value=0).astype(int)
    else:
        leuk[f'{PREFIX}SYMPTOM_max_in_30d'] = 0
    leuk[f'{PREFIX}SYMPTOM_burst_3plus'] = (leuk[f'{PREFIX}SYMPTOM_max_in_30d'] >= 3).astype(int)
    leuk[f'{PREFIX}SYMPTOM_burst_4plus'] = (leuk[f'{PREFIX}SYMPTOM_max_in_30d'] >= 4).astype(int)

    # ── Cleanup ───────────────────────────────────────────────
    leuk = leuk.fillna(0).replace([np.inf, -np.inf], 0)

    nunique = leuk.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        leuk = leuk.drop(columns=constant)
        logger.info(f"  Removed {len(constant)} constant columns")

    logger.info(f"  Leukaemia-specific features: {leuk.shape[1]}")
    return leuk
