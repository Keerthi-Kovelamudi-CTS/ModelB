# ═══════════════════════════════════════════════════════════════
# BREAST CANCER — CANCER-SPECIFIC FEATURES (Step 4d)
# Hand-engineered features for breast prediction:
#   - HbA1c trajectory + cardiometabolic risk pattern
#   - Reproductive risk modifiers (TAH±BSO, menopause, parity)
#   - HRT exposure (strong modifiable BC risk factor)
#   - Oral contraceptive history
#   - Lifestyle risk factors (smoking, alcohol, physical activity, BMI proxy)
#   - Mental health pattern
#   - Hereditary risk (FAMHX_BREAST_OVARIAN, HISTORY_HODGKIN_RT)
#   - Cardiometabolic medication burden
#   - Composite risk-surrogate score
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
    """Build breast-specific hand-engineered features."""
    if cfg is None: cfg = config
    PREFIX = cfg.PREFIX

    logger.info(f"  BREAST-SPECIFIC FEATURES - {window_name}")

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

    age = existing_fm['AGE_AT_INDEX'].reindex(pf.index).fillna(65)

    # ── Helpers ─────────────────────────────────────────────
    def has_cat(df, cat):
        has = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0
        return has.reindex(pf.index, fill_value=False).astype(int)

    def count_cat(df, cat):
        cnt = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size()
        return cnt.reindex(pf.index, fill_value=0).astype(int)

    def has_any_cat(df, cats):
        """OR across multiple categories."""
        s = pd.Series(0, index=pf.index)
        for c in cats:
            s = s | has_cat(df, c)
        return s.astype(int)

    m = lambda s, default=0: s.reindex(pf.index).fillna(default)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: HbA1c DYNAMICS
    # Insulin resistance associated with elevated BC risk.
    # (BMI codes in this curation don't have numeric values —
    # they're presence markers — so BMI trajectory is via Layer A.)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 1: HbA1c dynamics...")

    hba1c_mask = (obs_AB['CATEGORY'] == 'LAB_HBA1C') & obs_AB['VALUE'].notna()
    hba1c = obs_AB[hba1c_mask].sort_values('EVENT_DATE')

    if not hba1c.empty:
        hba1c_latest = hba1c.groupby('PATIENT_GUID')['VALUE'].last()
        hba1c_max    = hba1c.groupby('PATIENT_GUID')['VALUE'].max()
        pf[f'{PREFIX}LAB_hba1c_latest']      = m(hba1c_latest, default=np.nan)
        pf[f'{PREFIX}LAB_hba1c_max']         = m(hba1c_max, default=np.nan)
        pf[f'{PREFIX}LAB_hba1c_diabetic']    = (m(hba1c_latest) >= 48).astype(int)
        pf[f'{PREFIX}LAB_hba1c_prediabetic'] = ((m(hba1c_latest) >= 42) & (m(hba1c_latest) < 48)).astype(int)
        pf[f'{PREFIX}LAB_hba1c_poor_ctrl']   = (m(hba1c_latest) >= 58).astype(int)
    else:
        for col in ['hba1c_latest','hba1c_max','hba1c_diabetic','hba1c_prediabetic','hba1c_poor_ctrl']:
            pf[f'{PREFIX}LAB_{col}'] = 0

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: CARDIOMETABOLIC BURDEN
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 2: Cardiometabolic burden...")

    cv_obs_cats = ['COMORBID_CVD', 'COMORBID_IHD', 'COMORBID_HTN',
                    'COMORBID_HYPERCHOL']
    # COMORBID_DM now includes at-risk + prediabetes after granularity trim
    dm_obs_cats = ['COMORBID_DM']
    cv_burden = sum(has_cat(obs_AB, c) for c in cv_obs_cats)
    dm_burden = sum(has_cat(obs_AB, c) for c in dm_obs_cats)
    cardio_total = cv_burden + dm_burden

    pf[f'{PREFIX}RF_cv_burden']                = cv_burden.astype(int)
    pf[f'{PREFIX}RF_dm_burden']                = dm_burden.astype(int)
    pf[f'{PREFIX}RF_cardiometabolic_burden']   = cardio_total.astype(int)
    pf[f'{PREFIX}RF_cardiometabolic_3plus']    = (cardio_total >= 3).astype(int)
    pf[f'{PREFIX}RF_metabolic_syndrome_proxy'] = (cardio_total >= 2).astype(int)

    # Diabetes engagement (foot exam + eye monitoring composite)
    dm_eng_cats = ['COMORBID_DM_FOOT', 'COMORBID_DM_EYE']
    pf[f'{PREFIX}RF_dm_eng_score'] = sum(has_cat(obs_AB, c) for c in dm_eng_cats).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: REPRODUCTIVE RISK MODIFIERS
    # TAH+BSO is PROTECTIVE for BC. Atrophic vaginitis = postmenopausal proxy.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 3: Reproductive risk modifiers...")

    has_tah          = has_cat(obs_AB, 'REPRO_TAH')
    has_tah_bso      = has_cat(obs_AB, 'REPRO_TAH_BSO')
    has_menopause    = has_cat(obs_AB, 'REPRO_MENOPAUSE')
    has_atrophic_vag = has_cat(obs_AB, 'REPRO_ATROPHIC_VAG')
    has_menstrual    = has_cat(obs_AB, 'REPRO_MENSTRUAL_FLOW')
    has_pms          = has_cat(obs_AB, 'REPRO_PMS')
    has_parity       = has_cat(obs_AB, 'REPRO_PARITY')

    pf[f'{PREFIX}RF_has_tah']           = has_tah
    pf[f'{PREFIX}RF_has_tah_bso']       = has_tah_bso
    pf[f'{PREFIX}RF_tah_no_bso']        = ((has_tah == 1) & (has_tah_bso == 0)).astype(int)
    pf[f'{PREFIX}RF_has_menopause']     = has_menopause
    pf[f'{PREFIX}RF_has_atrophic_vag']  = has_atrophic_vag
    pf[f'{PREFIX}RF_has_menstrual']     = has_menstrual
    pf[f'{PREFIX}RF_has_pms']           = has_pms
    pf[f'{PREFIX}RF_has_parity']        = has_parity

    # Postmenopausal status proxy: any of (menopause coded, atrophic vag, age>=55)
    postmeno_proxy = ((has_menopause == 1) | (has_atrophic_vag == 1) | (age >= 55))
    pf[f'{PREFIX}RF_postmenopausal_proxy']  = postmeno_proxy.astype(int)
    pf[f'{PREFIX}RF_early_menopause_proxy'] = ((has_menopause == 1) & (age < 50)).astype(int)

    # Active menstruation in older age = late menopause = BC risk
    pf[f'{PREFIX}RF_late_menstruation']     = ((has_menstrual == 1) & (age >= 50)).astype(int)

    # Combined reproductive activity (engagement breadth)
    repro_pathway = (has_tah + has_tah_bso + has_menopause + has_atrophic_vag
                     + has_menstrual + has_pms + has_parity)
    pf[f'{PREFIX}RF_repro_pathway_count'] = repro_pathway.astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: HRT + ORAL CONTRACEPTIVE EXPOSURE
    # HRT = strongest known modifiable BC risk factor.
    # OC = nuanced; combined-pill use has mild ↑risk during use.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 4: HRT + OC exposure...")

    has_hrt = has_cat(med_AB, 'HRT')
    has_oc  = has_cat(med_AB, 'OC')
    n_hrt   = count_cat(med_AB, 'HRT')
    n_oc    = count_cat(med_AB, 'OC')

    pf[f'{PREFIX}RF_has_hrt']            = has_hrt
    pf[f'{PREFIX}RF_hrt_event_count']    = n_hrt
    pf[f'{PREFIX}RF_hrt_multi']          = (n_hrt >= 3).astype(int)  # ≥3 events = repeat-pattern
    pf[f'{PREFIX}RF_hrt_postmeno']       = ((has_hrt == 1) & postmeno_proxy).astype(int)

    pf[f'{PREFIX}RF_has_oc']             = has_oc
    pf[f'{PREFIX}RF_oc_event_count']     = n_oc
    pf[f'{PREFIX}RF_oc_multi']           = (n_oc >= 3).astype(int)
    pf[f'{PREFIX}RF_oc_premeno']         = ((has_oc == 1) & (age < 50)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: HEREDITARY / PRIOR-CANCER RISK
    # Family history of breast/ovarian + Hodgkin's history.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 5: Hereditary / prior-cancer risk...")

    has_famhx_breast = has_cat(obs_AB, 'FAMHX_BREAST_OVARIAN')
    has_hodgkin_hx   = has_cat(obs_AB, 'HISTORY_HODGKIN_RT')

    pf[f'{PREFIX}RF_has_famhx_breast']   = has_famhx_breast
    pf[f'{PREFIX}RF_has_hodgkin_hx']     = has_hodgkin_hx
    pf[f'{PREFIX}RF_famhx_x_postmeno']   = ((has_famhx_breast == 1) & postmeno_proxy).astype(int)
    pf[f'{PREFIX}RF_famhx_x_premeno']    = ((has_famhx_breast == 1) & (age < 50)).astype(int)

    # ── Block 5b: NEW hand-engineered hereditary load (added 2026-06-01) ──
    has_brca         = has_cat(obs_AB, 'HEREDITARY_BRCA')
    has_famhx_other  = has_cat(obs_AB, 'FAMHX_OTHER_CANCERS')
    pf[f'{PREFIX}RF_has_brca']           = has_brca
    pf[f'{PREFIX}RF_has_famhx_other']    = has_famhx_other
    pf[f'{PREFIX}RF_hereditary_load']    = (has_brca + has_famhx_breast + has_famhx_other).astype(int)
    pf[f'{PREFIX}RF_strong_hereditary']  = ((has_brca == 1) | (has_famhx_breast == 1) |
                                            (has_famhx_other >= 1)).astype(int)
    pf[f'{PREFIX}RF_brca_x_age40']       = ((has_brca == 1) & (age >= 40)).astype(int)
    pf[f'{PREFIX}RF_famhx_other_x_premeno'] = ((has_famhx_other >= 1) & (age < 50)).astype(int)

    # ── Block 5c: Risk-REDUCING oophorectomy (estrogen reduction) ──
    has_oophor = has_cat(obs_AB, 'HISTORY_OOPHORECTOMY')
    pf[f'{PREFIX}RF_has_oophorectomy']           = has_oophor
    pf[f'{PREFIX}RF_oophorectomy_postmeno']      = ((has_oophor == 1) & postmeno_proxy).astype(int)
    # Note: in risk score below, oophorectomy gets NEGATIVE weight (protective)

    # ── Block 5d: Ovarian conditions (PCOS, cysts, premature menopause) ──
    has_ovarian_cond = has_cat(obs_AB, 'OVARIAN_CONDITIONS')
    pf[f'{PREFIX}RF_has_ovarian_condition']      = has_ovarian_cond

    # ══════════════════════════════════════════════════════════
    # BLOCK 5e: BREAST PRESENTING SYMPTOM BURDEN (NEW — added 2026-06-01)
    # The single highest-signal category in v4 importance file.
    # Hand-engineer composites to rescue sparse-but-strong subgroup signal.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 5e: Breast symptom burden composites...")

    has_lump   = has_cat(obs_AB, 'SYMPTOM_BREAST_LUMP')
    has_pain   = has_cat(obs_AB, 'SYMPTOM_BREAST_PAIN')
    has_nipple = has_cat(obs_AB, 'SYMPTOM_NIPPLE')
    has_benign = has_cat(obs_AB, 'BREAST_BENIGN')

    pf[f'{PREFIX}SX_has_lump']             = has_lump
    pf[f'{PREFIX}SX_has_pain']             = has_pain
    pf[f'{PREFIX}SX_has_nipple']           = has_nipple
    pf[f'{PREFIX}SX_has_benign']           = has_benign

    # Composites — rescue patients with rare-but-strong symptom signals
    sx_count = (has_lump + has_pain + has_nipple).astype(int)
    pf[f'{PREFIX}SX_any_breast_symptom']   = (sx_count >= 1).astype(int)
    pf[f'{PREFIX}SX_breast_sx_burden']     = sx_count
    pf[f'{PREFIX}SX_multi_breast_sx']      = (sx_count >= 2).astype(int)
    pf[f'{PREFIX}SX_any_breast_finding']   = ((sx_count + has_benign) >= 1).astype(int)

    # Lump × demographic interactions (lump in screening-age woman = highest cancer probability)
    pf[f'{PREFIX}SX_lump_x_age_over_50']   = ((has_lump == 1) & (age >= 50)).astype(int)
    pf[f'{PREFIX}SX_lump_x_age_over_65']   = ((has_lump == 1) & (age >= 65)).astype(int)
    pf[f'{PREFIX}SX_lump_x_postmeno']      = ((has_lump == 1) & postmeno_proxy).astype(int)
    pf[f'{PREFIX}SX_lump_x_famhx']         = ((has_lump == 1) & (has_famhx_breast == 1)).astype(int)
    pf[f'{PREFIX}SX_lump_x_hrt']           = ((has_lump == 1) & (has_hrt == 1)).astype(int)
    pf[f'{PREFIX}SX_lump_x_hereditary']    = ((has_lump == 1) & (pf[f'{PREFIX}RF_strong_hereditary'] == 1)).astype(int)

    # Nipple changes are particularly suspicious (Paget's link)
    pf[f'{PREFIX}SX_nipple_x_postmeno']    = ((has_nipple == 1) & postmeno_proxy).astype(int)

    # Multi-symptom in older woman = red flag
    pf[f'{PREFIX}SX_multi_sx_x_age_50']    = ((sx_count >= 2) & (age >= 50)).astype(int)

    # Benign disease in postmenopausal = mild cancer risk increase
    pf[f'{PREFIX}SX_benign_x_postmeno']    = ((has_benign == 1) & postmeno_proxy).astype(int)

    # Family hx of CV disease (informational — not BC risk directly)
    famhx_cv = (has_cat(obs_AB, 'COMORBID_CVD_FH_IHD_LT60')
                + has_cat(obs_AB, 'COMORBID_CVD_FH_IHD_GT60')
                + has_cat(obs_AB, 'COMORBID_CVD_FH_ANGINA_LT60'))
    pf[f'{PREFIX}RF_famhx_cv_count']     = famhx_cv.astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: LIFESTYLE RISK FACTORS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 6: Lifestyle risk factors...")

    has_smoking_obs   = has_cat(obs_AB, 'LIFE_SMOKING')
    has_alcohol_obs   = has_any_cat(obs_AB, ['LIFE_ALCOHOL', 'LIFE_ALC_EDU', 'LIFE_ALC_SCREEN'])
    has_diet_obs      = has_cat(obs_AB, 'LIFE_DIET')
    has_physact_obs   = has_cat(obs_AB, 'LIFE_PHYSACT')
    has_bmi_obs       = has_any_cat(obs_AB, ['LIFE_BMI', 'LIFE_WEIGHT_SX'])
    has_smoking_med   = has_cat(med_AB, 'SMOKING_CESSATION')
    has_weight_loss   = has_cat(med_AB, 'WEIGHT_LOSS_MEDS')

    pf[f'{PREFIX}RF_smoking_active']        = has_smoking_obs
    pf[f'{PREFIX}RF_smoking_treated']       = ((has_smoking_obs == 1) & (has_smoking_med == 1)).astype(int)
    pf[f'{PREFIX}RF_alcohol_documented']    = has_alcohol_obs
    pf[f'{PREFIX}RF_diet_intervention']     = has_diet_obs
    pf[f'{PREFIX}RF_low_physact']           = has_physact_obs
    pf[f'{PREFIX}RF_bmi_documented']        = has_bmi_obs
    pf[f'{PREFIX}RF_weight_loss_treatment'] = has_weight_loss

    lifestyle_burden = (has_smoking_obs + has_alcohol_obs + has_physact_obs +
                        has_bmi_obs + has_weight_loss)
    pf[f'{PREFIX}RF_lifestyle_burden'] = lifestyle_burden.astype(int)
    pf[f'{PREFIX}RF_lifestyle_3plus']  = (lifestyle_burden >= 3).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: MENTAL HEALTH PATTERN
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 7: Mental health pattern...")

    mh_obs_cats = ['COMORBID_DEPRESSION', 'COMORBID_DEPRESSION_MILD',
                    'COMORBID_DEPRESSION_MOD', 'COMORBID_DEPRESSION_SAD',
                    'COMORBID_DEPRESSION_SX', 'COMORBID_INSOMNIA',
                    'COMORBID_HADS', 'COMORBID_MH_CPA', 'COMORBID_MH_CARE_PLAN',
                    'COMORBID_PSYCH_STRESS', 'COMORBID_PSYCH_STRESS_HOME',
                    'COMORBID_PSYCH_BEREAVEMENT', 'COMORBID_MEMORY_LOSS']
    has_mh_obs = has_any_cat(obs_AB, mh_obs_cats)
    has_antidep_med = has_cat(med_AB, 'ANTIDEPRESSANTS')
    has_antipsy_med = has_cat(med_AB, 'ANTIPSYCHOTICS')
    has_anxiol_med  = has_cat(med_AB, 'ANXIOLYTICS_HYPNOTICS')

    pf[f'{PREFIX}TX_mh_obs']                   = has_mh_obs
    pf[f'{PREFIX}TX_on_antidep']               = has_antidep_med
    pf[f'{PREFIX}TX_on_antipsychotic']         = has_antipsy_med
    pf[f'{PREFIX}TX_on_anxiolytic']            = has_anxiol_med
    pf[f'{PREFIX}TX_mh_diagnosed_and_treated'] = (
        (has_mh_obs == 1) & ((has_antidep_med + has_antipsy_med + has_anxiol_med) >= 1)
    ).astype(int)
    pf[f'{PREFIX}TX_mh_med_count'] = (has_antidep_med + has_antipsy_med + has_anxiol_med).astype(int)
    pf[f'{PREFIX}TX_mh_polypharm'] = ((has_antidep_med + has_antipsy_med + has_anxiol_med) >= 2).astype(int)

    # Severity gradient
    pf[f'{PREFIX}TX_depression_severe'] = has_cat(obs_AB, 'COMORBID_DEPRESSION_MOD')
    pf[f'{PREFIX}TX_bereavement']       = has_cat(obs_AB, 'COMORBID_PSYCH_BEREAVEMENT')

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: CARDIOMETABOLIC MEDICATION BURDEN
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 8: CV medication burden...")

    cv_med_cats = ['ACE_INHIBITORS', 'ARBS', 'BETA_BLOCKERS', 'CCB',
                    'DIURETICS', 'STATINS', 'ANTIPLATELETS',
                    'ALPHA_BLOCKERS_CENTRAL']
    cv_med_count = sum(has_cat(med_AB, c) for c in cv_med_cats)
    pf[f'{PREFIX}TX_cv_med_count']       = cv_med_count.astype(int)
    pf[f'{PREFIX}TX_cv_polypharm']       = (cv_med_count >= 3).astype(int)
    pf[f'{PREFIX}TX_cv_polypharm_5plus'] = (cv_med_count >= 5).astype(int)

    pf[f'{PREFIX}TX_on_statin']       = has_cat(med_AB, 'STATINS')
    pf[f'{PREFIX}TX_on_antiplatelet'] = has_cat(med_AB, 'ANTIPLATELETS')
    pf[f'{PREFIX}TX_on_diuretic']     = has_cat(med_AB, 'DIURETICS')
    pf[f'{PREFIX}TX_on_betablocker']  = has_cat(med_AB, 'BETA_BLOCKERS')
    pf[f'{PREFIX}TX_on_lmwh']         = has_cat(med_AB, 'LMWH')

    pf[f'{PREFIX}TX_on_inhaler']      = has_cat(med_AB, 'RESPIRATORY_INHALERS')
    pf[f'{PREFIX}TX_on_ppi']          = has_cat(med_AB, 'PPI_H2_CHRONIC_GI')
    pf[f'{PREFIX}TX_dm_monitoring']   = has_cat(med_AB, 'DM_MONITORING')

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: COMPOSITE BREAST CANCER RISK-SURROGATE SCORE
    # Weighted by literature-derived effect-size estimates.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 9: Composite risk score...")

    risk_score = pd.Series(0.0, index=pf.index)

    # Age bands
    risk_score = risk_score + (age >= 50).astype(int) * 1.0
    risk_score = risk_score + (age >= 65).astype(int) * 1.0
    risk_score = risk_score + (age >= 75).astype(int) * 0.5

    # Metabolic burden
    risk_score = risk_score + pf[f'{PREFIX}LAB_hba1c_diabetic'] * 0.8
    risk_score = risk_score + pf[f'{PREFIX}RF_metabolic_syndrome_proxy'] * 0.8

    # Lifestyle
    risk_score = risk_score + has_alcohol_obs * 0.5
    risk_score = risk_score + has_smoking_obs * 0.5

    # Reproductive (TAH+BSO is PROTECTIVE → negative weight)
    risk_score = risk_score - has_tah_bso * 1.0
    risk_score = risk_score + pf[f'{PREFIX}RF_early_menopause_proxy'] * (-0.5)
    risk_score = risk_score + pf[f'{PREFIX}RF_late_menstruation'] * 0.5

    # Hereditary + HRT (strongest known BC risk amplifiers)
    risk_score = risk_score + has_famhx_breast * 2.0
    risk_score = risk_score + has_hrt * 1.0
    risk_score = risk_score + pf[f'{PREFIX}RF_hrt_postmeno'] * 0.5
    risk_score = risk_score + has_hodgkin_hx * 1.5
    risk_score = risk_score + has_oc * 0.3  # mild contribution

    # NEW (added 2026-06-01): hereditary load + breast symptoms + oophorectomy
    risk_score = risk_score + has_brca * 3.0                    # BRCA+ is highest-risk single marker
    risk_score = risk_score + has_famhx_other * 0.5             # FH Lynch/BRCA2-linked cancers
    risk_score = risk_score + has_lump * 2.5                    # Breast lump = strong symptom
    risk_score = risk_score + has_nipple * 1.5                  # Nipple changes (Paget's risk)
    risk_score = risk_score + has_pain * 0.5                    # Breast pain (weaker symptom)
    risk_score = risk_score + has_benign * 0.3                  # Benign breast disease (mild ↑)
    risk_score = risk_score + pf[f'{PREFIX}SX_multi_breast_sx'] * 1.0    # Multi-symptom = red flag
    risk_score = risk_score + pf[f'{PREFIX}SX_lump_x_postmeno'] * 1.0    # Lump in postmeno = highest probability
    risk_score = risk_score - has_oophor * 1.5                  # Oophorectomy = PROTECTIVE (lower estrogen)

    pf[f'{PREFIX}RISK_score']        = risk_score.astype(float)
    pf[f'{PREFIX}RISK_low']          = (risk_score < 2.0).astype(int)
    pf[f'{PREFIX}RISK_intermediate'] = ((risk_score >= 2.0) & (risk_score < 4.0)).astype(int)
    pf[f'{PREFIX}RISK_high']         = (risk_score >= 4.0).astype(int)
    pf[f'{PREFIX}RISK_very_high']    = (risk_score >= 6.0).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: AGE × CLINICAL-STATE INTERACTIONS
    # (mirrors Prostate B+C BLOCK 10)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 11: Age × clinical interactions...")
    age40  = (age >= 40).astype(int)
    age50  = (age >= 50).astype(int)
    age65  = (age >= 65).astype(int)
    age75  = (age >= 75).astype(int)
    age_under_40 = (age < 40).astype(int)  # very early-onset window
    pf[f'{PREFIX}AGEX_age40']     = age40
    pf[f'{PREFIX}AGEX_under_40']  = age_under_40
    # Age 40 specifically captures premenopausal high-risk (BRCA carriers, FH)
    pf[f'{PREFIX}AGEX_lump_x_age40']        = (has_lump * age40)
    pf[f'{PREFIX}AGEX_lump_x_under_40']     = (has_lump * age_under_40)  # very young w/ lump = red flag
    pf[f'{PREFIX}AGEX_famhx_x_age40']       = (has_famhx_breast * age40)
    pf[f'{PREFIX}AGEX_famhx_x_under_40']    = (has_famhx_breast * age_under_40)  # FH + young = BRCA suspicion
    pf[f'{PREFIX}AGEX_brca_x_age40']        = (has_brca * age40)
    pf[f'{PREFIX}AGEX_brca_x_under_40']     = (has_brca * age_under_40)
    pf[f'{PREFIX}AGEX_oc_x_age40']          = (has_oc * age40)
    pf[f'{PREFIX}AGEX_nipple_x_age40']      = (has_nipple * age40)
    pf[f'{PREFIX}AGEX_atypical_under_40']   = (sx_count * age_under_40)
    pf[f'{PREFIX}AGEX_lump_x_age50']        = (has_lump * age50)
    pf[f'{PREFIX}AGEX_lump_x_age65']        = (has_lump * age65)
    pf[f'{PREFIX}AGEX_lump_x_age75']        = (has_lump * age75)
    pf[f'{PREFIX}AGEX_nipple_x_age50']      = (has_nipple * age50)
    pf[f'{PREFIX}AGEX_pain_x_age50']        = (has_pain * age50)
    pf[f'{PREFIX}AGEX_benign_x_age50']      = (has_benign * age50)
    pf[f'{PREFIX}AGEX_famhx_x_age50']       = (has_famhx_breast * age50)
    pf[f'{PREFIX}AGEX_famhx_x_age65']       = (has_famhx_breast * age65)
    pf[f'{PREFIX}AGEX_famhx_other_x_age50'] = (has_famhx_other * age50)
    pf[f'{PREFIX}AGEX_brca_x_age50']        = (has_brca * age50)
    pf[f'{PREFIX}AGEX_hrt_x_age50']         = (has_hrt * age50)
    pf[f'{PREFIX}AGEX_hrt_x_age65']         = (has_hrt * age65)
    pf[f'{PREFIX}AGEX_oc_x_age50']          = (has_oc * age50)
    pf[f'{PREFIX}AGEX_alcohol_x_age50']     = (has_alcohol_obs * age50)
    pf[f'{PREFIX}AGEX_ovarian_x_age50']     = (has_ovarian_cond * age50)
    pf[f'{PREFIX}AGEX_atypical_x_age50']    = (sx_count * age50)  # multi-sx older woman

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: TRIPLE INTERACTIONS (non-linear combos)
    # (mirrors Prostate B+C BLOCK 11)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 12: Triple interactions...")
    pf[f'{PREFIX}TRI_lump_postmeno_famhx']   = (has_lump * postmeno_proxy.astype(int) * has_famhx_breast)
    pf[f'{PREFIX}TRI_lump_postmeno_hrt']     = (has_lump * postmeno_proxy.astype(int) * has_hrt)
    pf[f'{PREFIX}TRI_lump_age65_hereditary'] = (has_lump * age65 * pf[f'{PREFIX}RF_strong_hereditary'])
    pf[f'{PREFIX}TRI_brca_famhx_age50']      = (has_brca * has_famhx_breast * age50)
    pf[f'{PREFIX}TRI_hrt_postmeno_alcohol']  = (has_hrt * postmeno_proxy.astype(int) * has_alcohol_obs)
    pf[f'{PREFIX}TRI_hrt_postmeno_age65']    = (has_hrt * postmeno_proxy.astype(int) * age65)
    pf[f'{PREFIX}TRI_multi_sx_age50_famhx']  = ((sx_count >= 2).astype(int) * age50 * has_famhx_breast)
    pf[f'{PREFIX}TRI_nipple_postmeno_famhx'] = (has_nipple * postmeno_proxy.astype(int) * has_famhx_breast)
    pf[f'{PREFIX}TRI_benign_famhx_postmeno'] = (has_benign * has_famhx_breast * postmeno_proxy.astype(int))
    pf[f'{PREFIX}TRI_lump_pain_nipple']      = (has_lump * has_pain * has_nipple)  # all 3 symptoms

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: SYMPTOM BURST / CO-ELEVATION
    # (mirrors Prostate B+C BLOCK 16)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 13: Symptom burst patterns...")
    # Recent half of lookback (window B): 1-30mo before anchor
    obs_B = obs_AB[obs_AB['TIME_WINDOW'] == 'B']
    obs_A = obs_AB[obs_AB['TIME_WINDOW'] == 'A']

    has_lump_B   = has_cat(obs_B, 'SYMPTOM_BREAST_LUMP')
    has_pain_B   = has_cat(obs_B, 'SYMPTOM_BREAST_PAIN')
    has_nipple_B = has_cat(obs_B, 'SYMPTOM_NIPPLE')
    has_lump_A   = has_cat(obs_A, 'SYMPTOM_BREAST_LUMP')

    sx_count_B = (has_lump_B + has_pain_B + has_nipple_B).astype(int)
    pf[f'{PREFIX}BURST_sx_in_B']       = (sx_count_B >= 1).astype(int)
    pf[f'{PREFIX}BURST_multi_sx_in_B'] = (sx_count_B >= 2).astype(int)
    pf[f'{PREFIX}BURST_new_lump_in_B'] = ((has_lump_B == 1) & (has_lump_A == 0)).astype(int)  # NEW lump in recent
    pf[f'{PREFIX}BURST_escalating_sx'] = ((sx_count_B > 0) & (sx_count_B > has_lump_A + has_pain_B + has_nipple_B)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: SEQUENTIAL PATTERNS (X then Y within Z months)
    # (mirrors Prostate B+C BLOCK 14)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 14: Sequential patterns...")

    def first_event(cat, df):
        """Return min MONTHS_BEFORE_INDEX per patient for given category."""
        m = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        return m.reindex(pf.index, fill_value=np.nan)

    first_lump   = first_event('SYMPTOM_BREAST_LUMP', obs_AB)
    first_pain   = first_event('SYMPTOM_BREAST_PAIN', obs_AB)
    first_nipple = first_event('SYMPTOM_NIPPLE', obs_AB)
    first_benign = first_event('BREAST_BENIGN', obs_AB)
    first_hrt    = (med_AB[med_AB['CATEGORY'] == 'HRT']
                    .groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
                    .reindex(pf.index, fill_value=np.nan))
    first_meno   = first_event('REPRO_MENOPAUSE', obs_AB)

    # Pain → Lump within 6 months (pain preceding lump = escalating)
    pf[f'{PREFIX}SEQ_pain_then_lump_6m'] = (
        ((first_pain - first_lump) >= 0) & ((first_pain - first_lump) <= 6)
    ).fillna(False).astype(int)
    # Lump → Nipple within 6 months (extending symptom)
    pf[f'{PREFIX}SEQ_lump_then_nipple_6m'] = (
        ((first_lump - first_nipple) >= 0) & ((first_lump - first_nipple) <= 6)
    ).fillna(False).astype(int)
    # Benign disease history → recent lump (benign progressing to suspicious)
    pf[f'{PREFIX}SEQ_benign_then_lump_12m'] = (
        ((first_benign - first_lump) >= 0) & ((first_benign - first_lump) <= 12)
    ).fillna(False).astype(int)
    # HRT → Menopause documented within 12mo (typical pattern)
    pf[f'{PREFIX}SEQ_hrt_then_meno_12m'] = (
        ((first_hrt - first_meno) >= 0) & ((first_hrt - first_meno) <= 12)
    ).fillna(False).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 15: FIRST-EVENT-DATE FEATURES
    # (mirrors Prostate B+C BLOCK 13)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 15: First-event-date per category...")
    pf[f'{PREFIX}FIRST_lump_mo_before']   = first_lump.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_pain_mo_before']   = first_pain.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_nipple_mo_before'] = first_nipple.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_benign_mo_before'] = first_benign.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_hrt_mo_before']    = first_hrt.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_meno_mo_before']   = first_meno.fillna(999).astype(float)
    # Recency flags (any symptom in last 12mo of lookback)
    pf[f'{PREFIX}FIRST_lump_within_12mo']   = (first_lump <= 12).fillna(False).astype(int)
    pf[f'{PREFIX}FIRST_nipple_within_12mo'] = (first_nipple <= 12).fillna(False).astype(int)
    pf[f'{PREFIX}FIRST_benign_within_12mo'] = (first_benign <= 12).fillna(False).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 16: PATIENT-LEVEL Z-SCORE LABS
    # (mirrors Prostate B+C BLOCK 12)
    # Compute relative position vs cohort for engagement-adjusted lab signal
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 16: Patient z-score labs...")

    def patient_zscore(cat, agg='mean'):
        """Compute per-patient mean of VALUE for category, then z-score across cohort."""
        df_v = obs_AB[(obs_AB['CATEGORY'] == cat) & obs_AB['VALUE'].notna()].copy()
        if df_v.empty:
            return pd.Series(0.0, index=pf.index)
        if agg == 'mean':
            per_pt = df_v.groupby('PATIENT_GUID')['VALUE'].mean()
        elif agg == 'max':
            per_pt = df_v.groupby('PATIENT_GUID')['VALUE'].max()
        else:
            per_pt = df_v.groupby('PATIENT_GUID')['VALUE'].last()
        if per_pt.std() == 0 or pd.isna(per_pt.std()):
            return pd.Series(0.0, index=pf.index)
        z = (per_pt - per_pt.mean()) / per_pt.std()
        return z.reindex(pf.index, fill_value=0.0).astype(float)

    pf[f'{PREFIX}Z_hba1c_mean']      = patient_zscore('LAB_HBA1C', 'mean')
    pf[f'{PREFIX}Z_hba1c_max']       = patient_zscore('LAB_HBA1C', 'max')
    pf[f'{PREFIX}Z_lipids_mean']     = patient_zscore('LAB_LIPIDS', 'mean')
    pf[f'{PREFIX}Z_tsh_mean']        = patient_zscore('LAB_TSH', 'mean')
    pf[f'{PREFIX}Z_ft4_mean']        = patient_zscore('LAB_FT4', 'mean')
    pf[f'{PREFIX}Z_bmi_mean']        = patient_zscore('LIFE_BMI', 'mean')

    # Z extreme flags (clinically elevated)
    pf[f'{PREFIX}Z_hba1c_high']      = (pf[f'{PREFIX}Z_hba1c_mean'] > 1.5).astype(int)
    pf[f'{PREFIX}Z_lipids_high']     = (pf[f'{PREFIX}Z_lipids_mean'] > 1.5).astype(int)
    pf[f'{PREFIX}Z_tsh_low']         = (pf[f'{PREFIX}Z_tsh_mean'] < -1.5).astype(int)  # hyperthyroid

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: VELOCITY + COMORBIDITY-WEIGHTED RATIOS
    # (mirrors Prostate B+C BLOCK 11)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 17: Velocity + comorbidity-weighted ratios...")
    # Velocity: increase in events from window A to B
    n_lump_A = obs_A[obs_A['CATEGORY']=='SYMPTOM_BREAST_LUMP'].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    n_lump_B = obs_B[obs_B['CATEGORY']=='SYMPTOM_BREAST_LUMP'].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}VEL_lump_B_minus_A'] = (n_lump_B - n_lump_A).astype(int)

    n_benign_A = obs_A[obs_A['CATEGORY']=='BREAST_BENIGN'].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    n_benign_B = obs_B[obs_B['CATEGORY']=='BREAST_BENIGN'].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}VEL_benign_B_minus_A'] = (n_benign_B - n_benign_A).astype(int)

    # Weighted comorbidity score (count of major comorbidities)
    cm_score = (
        pf[f'{PREFIX}RF_cv_burden'] * 1.0 +
        pf[f'{PREFIX}RF_dm_burden'] * 1.0 +
        pf[f'{PREFIX}LAB_hba1c_diabetic'] * 1.5 +
        pf[f'{PREFIX}RF_metabolic_syndrome_proxy'] * 1.5
    )
    pf[f'{PREFIX}CM_weighted_score']  = cm_score.astype(float)
    pf[f'{PREFIX}CM_burden_high']     = (cm_score >= 3.0).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 18: PER-CATEGORY VELOCITY (B - A counts for many cats)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 18: Per-category velocity (escalation patterns)...")
    def cat_velocity(cat, df_AB=obs_AB):
        nA = df_AB[(df_AB['CATEGORY']==cat) & (df_AB['TIME_WINDOW']=='A')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        nB = df_AB[(df_AB['CATEGORY']==cat) & (df_AB['TIME_WINDOW']=='B')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        return (nB - nA).astype(int)

    for cat in ['SYMPTOM_BREAST_PAIN','SYMPTOM_NIPPLE','FAMHX_BREAST_OVARIAN',
                'FAMHX_OTHER_CANCERS','LIFE_ALCOHOL','LIFE_BMI','LIFE_SMOKING',
                'LAB_HBA1C','LAB_LIPIDS','OVARIAN_CONDITIONS']:
        pf[f'{PREFIX}VEL_{cat}_B_minus_A'] = cat_velocity(cat)
    pf[f'{PREFIX}VEL_hrt_B_minus_A']   = cat_velocity('HRT', med_AB)
    pf[f'{PREFIX}VEL_oc_B_minus_A']    = cat_velocity('OC', med_AB)

    # ══════════════════════════════════════════════════════════
    # BLOCK 18b: TIME-DECAY INTENSITY + ACCELERATION (restored 2026-06)
    #   DECAY = sum of exp(-months/12) per category — recency-weighted event load.
    #   ACCEL = 2nd difference across 1-15 / 15-30 / 30-60 mo bins — escalation rate.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 18b: Time-decay intensity + acceleration (restored)...")
    # KEY_CATS used by BLOCK 18b / 26 / 27 — define ONCE here, before first use
    # (was previously only defined at BLOCK 26 → UnboundLocalError when 18b ran first)
    KEY_CATS = [
        'SYMPTOM_BREAST_LUMP', 'SYMPTOM_BREAST_PAIN', 'SYMPTOM_NIPPLE',
        'BREAST_BENIGN', 'FAMHX_BREAST_OVARIAN', 'FAMHX_OTHER_CANCERS',
        'HEREDITARY_BRCA', 'HISTORY_OOPHORECTOMY',
        'LAB_HBA1C', 'LAB_LIPIDS', 'LIFE_BMI', 'LIFE_ALCOHOL',
    ]
    HALFLIFE_MO = 12.0
    def cat_decay(cat, df_AB=obs_AB):
        sub = df_AB[df_AB['CATEGORY'] == cat]
        if len(sub) == 0:
            return pd.Series(0.0, index=pf.index)
        w = np.exp(-sub['MONTHS_BEFORE_INDEX'].astype(float).clip(lower=0) / HALFLIFE_MO)
        return w.groupby(sub['PATIENT_GUID']).sum().reindex(pf.index, fill_value=0.0)
    for cat in KEY_CATS:
        pf[f'{PREFIX}DECAY_{cat}'] = cat_decay(cat).astype(float)
    pf[f'{PREFIX}DECAY_HRT'] = cat_decay('HRT', med_AB).astype(float)

    def cat_bin(cat, lo, hi, df_AB=obs_AB):
        sub = df_AB[(df_AB['CATEGORY'] == cat) &
                    (df_AB['MONTHS_BEFORE_INDEX'] >= lo) & (df_AB['MONTHS_BEFORE_INDEX'] < hi)]
        return sub.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    for cat in ['SYMPTOM_BREAST_LUMP', 'SYMPTOM_BREAST_PAIN', 'SYMPTOM_NIPPLE',
                'BREAST_BENIGN', 'LAB_LIPIDS']:
        rec = cat_bin(cat, 1, 15); mid = cat_bin(cat, 15, 30); ear = cat_bin(cat, 30, 60)
        pf[f'{PREFIX}ACCEL_{cat}'] = ((rec - mid) - (mid - ear)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 19: ENGAGEMENT-NORMALIZED FEATURES
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 19: Engagement-normalized signal...")
    total_obs_events = obs_AB.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0).clip(lower=1)
    total_med_events = med_AB.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0).clip(lower=1)

    def cat_count(cat, df_AB=obs_AB):
        return df_AB[df_AB['CATEGORY']==cat].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)

    pf[f'{PREFIX}NORM_lump_per_total']    = (cat_count('SYMPTOM_BREAST_LUMP') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_pain_per_total']    = (cat_count('SYMPTOM_BREAST_PAIN') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_nipple_per_total']  = (cat_count('SYMPTOM_NIPPLE') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_benign_per_total']  = (cat_count('BREAST_BENIGN') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_breast_sx_total']   = ((cat_count('SYMPTOM_BREAST_LUMP') + cat_count('SYMPTOM_BREAST_PAIN') + cat_count('SYMPTOM_NIPPLE')) / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_hrt_per_med']       = (cat_count('HRT', med_AB) / total_med_events).astype(float)
    pf[f'{PREFIX}NORM_famhx_per_total']   = (cat_count('FAMHX_BREAST_OVARIAN') / total_obs_events).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 20: BMI-BASED RISK (BMI is direct BC risk factor)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 20: BMI-derived risk...")
    bmi_values = obs_AB[obs_AB['CATEGORY']=='LIFE_BMI']['VALUE'].dropna()
    bmi_mean = (obs_AB[(obs_AB['CATEGORY']=='LIFE_BMI') & obs_AB['VALUE'].notna()]
                .groupby('PATIENT_GUID')['VALUE'].mean()
                .reindex(pf.index, fill_value=np.nan))
    bmi_max  = (obs_AB[(obs_AB['CATEGORY']=='LIFE_BMI') & obs_AB['VALUE'].notna()]
                .groupby('PATIENT_GUID')['VALUE'].max()
                .reindex(pf.index, fill_value=np.nan))
    pf[f'{PREFIX}BMI_mean']               = bmi_mean.fillna(-1).astype(float)
    pf[f'{PREFIX}BMI_max']                = bmi_max.fillna(-1).astype(float)
    pf[f'{PREFIX}BMI_obese']              = ((bmi_mean >= 30) & (bmi_mean < 40)).fillna(False).astype(int)
    pf[f'{PREFIX}BMI_morbid_obese']       = (bmi_mean >= 40).fillna(False).astype(int)
    pf[f'{PREFIX}BMI_overweight']         = ((bmi_mean >= 25) & (bmi_mean < 30)).fillna(False).astype(int)
    pf[f'{PREFIX}BMI_underweight']        = (bmi_mean < 18.5).fillna(False).astype(int)
    pf[f'{PREFIX}BMI_obese_postmeno']     = ((bmi_mean >= 30).fillna(False) & postmeno_proxy).astype(int)  # KEY BC risk
    pf[f'{PREFIX}BMI_obese_age65']        = ((bmi_mean >= 30).fillna(False) & (age >= 65)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 21: CROSS-CATEGORY CO-OCCURRENCE FLAGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 21: Cross-category co-occurrence...")
    # 2+ breast-pathway indicators
    breast_indicators = (has_lump + has_pain + has_nipple + has_benign +
                          has_famhx_breast + has_famhx_other + has_brca).astype(int)
    pf[f'{PREFIX}COOC_breast_indicators']        = breast_indicators
    pf[f'{PREFIX}COOC_2plus_breast_indicators']  = (breast_indicators >= 2).astype(int)
    pf[f'{PREFIX}COOC_3plus_breast_indicators']  = (breast_indicators >= 3).astype(int)
    pf[f'{PREFIX}COOC_4plus_breast_indicators']  = (breast_indicators >= 4).astype(int)
    # Symptom + hereditary load
    pf[f'{PREFIX}COOC_sx_and_hereditary']        = ((sx_count >= 1) & ((has_famhx_breast + has_famhx_other + has_brca) >= 1)).astype(int)
    # Symptom + HRT exposure
    pf[f'{PREFIX}COOC_sx_and_hrt']               = ((sx_count >= 1) & (has_hrt == 1)).astype(int)
    # Benign + hereditary (atypical hyperplasia could be hidden here)
    pf[f'{PREFIX}COOC_benign_and_hereditary']    = ((has_benign == 1) & ((has_famhx_breast + has_brca) >= 1)).astype(int)
    # PCOS + obesity
    pf[f'{PREFIX}COOC_pcos_and_obese']           = ((has_ovarian_cond == 1) & (bmi_mean >= 30).fillna(False)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 22: TIME-SINCE-LAST-EVENT (recency per category)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 22: Time-since-last-event recency...")
    def last_event(cat, df_AB=obs_AB):
        m = df_AB[df_AB['CATEGORY']==cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        return m.reindex(pf.index, fill_value=np.nan)

    last_lump   = last_event('SYMPTOM_BREAST_LUMP')
    last_pain   = last_event('SYMPTOM_BREAST_PAIN')
    last_nipple = last_event('SYMPTOM_NIPPLE')
    last_benign = last_event('BREAST_BENIGN')
    last_hrt    = last_event('HRT', med_AB)

    pf[f'{PREFIX}RECENT_lump_in_6mo']    = (last_lump <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_pain_in_6mo']    = (last_pain <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_nipple_in_6mo']  = (last_nipple <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_benign_in_6mo']  = (last_benign <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_hrt_in_3mo']     = (last_hrt <= 3).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_any_breast_sx_in_6mo'] = (
        (last_lump <= 6) | (last_pain <= 6) | (last_nipple <= 6)
    ).fillna(False).astype(int)
    # Days-since gap (capped at 60mo)
    pf[f'{PREFIX}RECENT_lump_gap_mo']    = last_lump.fillna(60).astype(float)
    pf[f'{PREFIX}RECENT_breast_sx_gap_mo'] = (
        pd.concat([last_lump, last_pain, last_nipple], axis=1).min(axis=1)
    ).fillna(60).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 23: CUMULATIVE BREAST RISK SUPER-FLAG
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 23: Cumulative breast risk super-flags...")
    # Highest-risk subgroup: lump + age50 + (postmeno OR famhx OR brca)
    pf[f'{PREFIX}SUPER_high_alarm'] = (
        (has_lump == 1) & (age50 == 1) &
        ((postmeno_proxy.astype(int) | has_famhx_breast | has_brca) >= 1)
    ).astype(int)
    # 2+ indicators in postmenopausal age
    pf[f'{PREFIX}SUPER_dual_indicator_postmeno'] = (
        (breast_indicators >= 2) & postmeno_proxy
    ).astype(int)
    # Strong family history (any cancer family hx) + lump or nipple
    pf[f'{PREFIX}SUPER_strong_famhx_sx'] = (
        ((has_famhx_breast + has_famhx_other + has_brca) >= 1) & ((has_lump + has_nipple) >= 1)
    ).astype(int)
    # HRT >5 years (long-term) + postmeno  (known BC amplifier)
    pf[f'{PREFIX}SUPER_long_hrt_postmeno'] = (
        (pf[f'{PREFIX}RF_hrt_multi'] == 1) & postmeno_proxy
    ).astype(int)
    # No protective factors (no oophorectomy AND no TAH+BSO) + 2+ risk markers
    pf[f'{PREFIX}SUPER_no_protective_high_risk'] = (
        (has_oophor == 0) & (has_tah_bso == 0) & (breast_indicators >= 2)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 24: LOG / SQRT TRANSFORMS FOR HIGH-VARIANCE COUNTS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 24: Log/sqrt transforms...")
    # Compute n_distinct_obs_cats locally (BLOCK 10 / ENG features defined AFTER this block)
    _n_distinct_obs = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}LOG_total_obs_events']     = np.log1p(total_obs_events).astype(float)
    pf[f'{PREFIX}LOG_total_med_events']     = np.log1p(total_med_events).astype(float)
    pf[f'{PREFIX}LOG_n_distinct_cats']      = np.log1p(_n_distinct_obs).astype(float)
    pf[f'{PREFIX}LOG_hrt_count']            = np.log1p(pf[f'{PREFIX}RF_hrt_event_count']).astype(float)
    pf[f'{PREFIX}LOG_oc_count']             = np.log1p(pf[f'{PREFIX}RF_oc_event_count']).astype(float)
    pf[f'{PREFIX}SQRT_breast_sx_burden']    = np.sqrt(sx_count).astype(float)
    pf[f'{PREFIX}SQRT_breast_indicators']   = np.sqrt(breast_indicators).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 25: COHORT-RELATIVE PERCENTILE BANDS (age, BMI, total events)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 25: Cohort-relative percentile bands...")
    pf[f'{PREFIX}PCT_age_decile']         = pd.qcut(age.rank(method='first'), q=10, labels=False, duplicates='drop').astype(int)
    pf[f'{PREFIX}PCT_event_decile']       = pd.qcut(total_obs_events.rank(method='first'), q=10, labels=False, duplicates='drop').astype(int)
    pf[f'{PREFIX}PCT_distinct_cats_decile'] = pd.qcut(_n_distinct_obs.rank(method='first'), q=10, labels=False, duplicates='drop').astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 26: LUNG-INSPIRED TEMPORAL GRANULARITY
    # Per-key-category: min/max/mean event AGE + interval stats + trend slope + worsening flag
    # Focused on breast-critical categories (not all 60 — that would be 3000+ features)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 26: Lung-inspired per-category temporal granularity...")
    # KEY_CATS defined earlier (before BLOCK 18b) — reused here
    for cat in KEY_CATS:
        sub = obs_AB[obs_AB['CATEGORY'] == cat]
        if sub.empty:
            continue
        # Event ages (EVENT_AGE column = patient age at event date)
        ages = sub.groupby('PATIENT_GUID')['EVENT_AGE'].agg(['min','max','mean','median'])
        pf[f'{PREFIX}EA_{cat}_min_age']    = ages['min'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}EA_{cat}_max_age']    = ages['max'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}EA_{cat}_mean_age']   = ages['mean'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}EA_{cat}_age_span']   = (ages['max'] - ages['min']).reindex(pf.index, fill_value=0).astype(float)

        # First/last event months-before-anchor
        mbs = sub.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min','max'])
        pf[f'{PREFIX}EA_{cat}_first_mo_before'] = mbs['max'].reindex(pf.index, fill_value=999).astype(float)
        pf[f'{PREFIX}EA_{cat}_last_mo_before']  = mbs['min'].reindex(pf.index, fill_value=999).astype(float)
        pf[f'{PREFIX}EA_{cat}_event_span_mo']   = (mbs['max'] - mbs['min']).reindex(pf.index, fill_value=0).astype(float)

        # Worsening flag: did frequency in B > A?
        nA = sub[sub['TIME_WINDOW']=='A'].groupby('PATIENT_GUID').size()
        nB = sub[sub['TIME_WINDOW']=='B'].groupby('PATIENT_GUID').size()
        nA = nA.reindex(pf.index, fill_value=0); nB = nB.reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}EA_{cat}_is_worsening'] = (nB > nA).astype(int)
        pf[f'{PREFIX}EA_{cat}_is_only_recent'] = ((nB > 0) & (nA == 0)).astype(int)  # NEW onset

    # ══════════════════════════════════════════════════════════
    # BLOCK 27: LUNG-STYLE INTERVAL + TREND SLOPE + LAST-18MO PER KEY CAT
    # Mirrors lung v2-ensemble feature set (NUM/MEAN/MEDIAN/MIN/MAX_INTERVAL,
    # FREQUENCY_TREND_SLOPE, FIRST_HALF/SECOND_HALF_FREQUENCY, IS_WORSENING,
    # plus LAST_18M-window variants).
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 27: Lung-style interval + trend + last-18mo features...")
    for cat in KEY_CATS:
        sub = obs_AB[obs_AB['CATEGORY'] == cat].copy()
        if sub.empty:
            continue
        # ─── Interval stats (gap between consecutive events in DAYS) ───
        sub = sub.sort_values(['PATIENT_GUID', 'MONTHS_BEFORE_INDEX'])
        # MONTHS_BEFORE_INDEX is integer months — convert to "days-before" for finer diff
        sub['_days_before'] = sub['MONTHS_BEFORE_INDEX'] * 30.4375
        intervals = sub.groupby('PATIENT_GUID')['_days_before'].diff().abs()
        sub['_interval'] = intervals
        ivl_stats = sub.dropna(subset=['_interval']).groupby('PATIENT_GUID')['_interval'].agg(['mean','median','min','max'])
        pf[f'{PREFIX}IVL_{cat}_mean_days']   = ivl_stats['mean'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}IVL_{cat}_median_days'] = ivl_stats['median'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}IVL_{cat}_min_days']    = ivl_stats['min'].reindex(pf.index, fill_value=-1).astype(float)
        pf[f'{PREFIX}IVL_{cat}_max_days']    = ivl_stats['max'].reindex(pf.index, fill_value=-1).astype(float)

        # ─── Frequency trend SLOPE (events per month, linear regression) ───
        # Per patient: regress count of events per month against month index
        ev_by_month = (sub.groupby(['PATIENT_GUID', 'MONTHS_BEFORE_INDEX']).size()
                          .reset_index(name='n'))
        def _slope(g):
            if len(g) < 2: return 0.0
            x = g['MONTHS_BEFORE_INDEX'].astype(float).values
            y = g['n'].astype(float).values
            if x.std() == 0: return 0.0
            return float(np.polyfit(x, y, 1)[0])
        slope = ev_by_month.groupby('PATIENT_GUID').apply(_slope)
        # Negative slope = increasing freq closer to anchor (worsening)
        pf[f'{PREFIX}TRD_{cat}_slope'] = slope.reindex(pf.index, fill_value=0.0).astype(float)
        pf[f'{PREFIX}TRD_{cat}_worsening'] = (slope < 0).reindex(pf.index, fill_value=False).astype(int)

        # ─── First-half vs second-half of lookback ───
        # Lookback spans 0-60mo. First half: 30-60mo, Second half: 0-30mo (closer to anchor)
        first_half = sub[sub['MONTHS_BEFORE_INDEX'] >= 30].groupby('PATIENT_GUID').size()
        second_half = sub[sub['MONTHS_BEFORE_INDEX'] < 30].groupby('PATIENT_GUID').size()
        first_half = first_half.reindex(pf.index, fill_value=0)
        second_half = second_half.reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}HALF_{cat}_first_half_n']   = first_half.astype(int)
        pf[f'{PREFIX}HALF_{cat}_second_half_n']  = second_half.astype(int)
        pf[f'{PREFIX}HALF_{cat}_freq_ratio']     = (second_half / (first_half + 1.0)).astype(float)  # +1 to avoid div0

        # ─── LAST 18 MONTHS variants ───
        last18 = sub[sub['MONTHS_BEFORE_INDEX'] <= 18]
        n18 = last18.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}L18M_{cat}_n']              = n18.astype(int)
        pf[f'{PREFIX}L18M_{cat}_has']            = (n18 >= 1).astype(int)
        pf[f'{PREFIX}L18M_{cat}_freq_per_year']  = (n18 / 1.5).astype(float)  # 18mo = 1.5yr
        if not last18.empty:
            l18_ages = last18.groupby('PATIENT_GUID')['EVENT_AGE'].agg(['mean','max'])
            pf[f'{PREFIX}L18M_{cat}_mean_age'] = l18_ages['mean'].reindex(pf.index, fill_value=-1).astype(float)
            pf[f'{PREFIX}L18M_{cat}_max_age']  = l18_ages['max'].reindex(pf.index, fill_value=-1).astype(float)
        else:
            pf[f'{PREFIX}L18M_{cat}_mean_age'] = -1.0
            pf[f'{PREFIX}L18M_{cat}_max_age']  = -1.0

    # ══════════════════════════════════════════════════════════
    # BLOCK 28: AGE POLYNOMIALS + BANDS (non-linear age effects)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 28: Age polynomials + bands...")
    age_f = age.astype(float)
    pf[f'{PREFIX}AGEP_age_sq']     = (age_f ** 2).astype(float)
    pf[f'{PREFIX}AGEP_age_cu']     = (age_f ** 3 / 100.0).astype(float)  # scaled to avoid huge nums
    pf[f'{PREFIX}AGEP_age_log']    = np.log1p(age_f).astype(float)
    pf[f'{PREFIX}AGEP_age_sqrt']   = np.sqrt(age_f).astype(float)
    # 5-year bands as one-hot
    for lo, hi in [(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,99)]:
        pf[f'{PREFIX}AGEB_band_{lo}_{hi}'] = ((age_f >= lo) & (age_f < hi)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 29: PATIENT SUMMARY COUNTS / DISTINCT-CATEGORIES
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 29: Patient summary counts...")
    # Distinct categories per window
    nA_cats = obs_A.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    nB_cats = obs_B.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}SUM_distinct_cats_A']      = nA_cats.astype(int)
    pf[f'{PREFIX}SUM_distinct_cats_B']      = nB_cats.astype(int)
    pf[f'{PREFIX}SUM_distinct_cats_B_minus_A'] = (nB_cats - nA_cats).astype(int)
    pf[f'{PREFIX}SUM_new_cats_in_B']        = (nB_cats > nA_cats).astype(int)
    # Distinct med categories
    med_AB_pid = med_AB[['PATIENT_GUID', 'CATEGORY']].drop_duplicates()
    nMedCats = med_AB_pid.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}SUM_distinct_med_cats']    = nMedCats.astype(int)
    pf[f'{PREFIX}SUM_polypharm_intensity']  = (nMedCats >= 5).astype(int)
    pf[f'{PREFIX}SUM_polypharm_severe']     = (nMedCats >= 10).astype(int)

    # Distinct LAB categories specifically
    lab_cats = ['LAB_HBA1C','LAB_LIPIDS','LAB_TSH','LAB_TSH_PLASMA','LAB_FT3','LAB_FT4',
                'LAB_FT4_SERUM','LAB_TFT','LAB_TFT_TEST','LAB_TPO_AB','LAB_GONADOTROPHIN',
                'LAB_FSH','LAB_SPIRO_FLOW','LAB_SPIRO_VOLUMES','LAB_BP_NORMAL']
    n_distinct_labs = obs_AB[obs_AB['CATEGORY'].isin(lab_cats)].groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}SUM_distinct_lab_cats']    = n_distinct_labs.astype(int)
    pf[f'{PREFIX}SUM_lab_heavy_user']       = (n_distinct_labs >= 5).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 30: CROSS-WINDOW RATIOS (recent vs historical)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 30: Cross-window lab/symptom ratios...")
    def window_count(cat, window):
        return obs_AB[(obs_AB['CATEGORY']==cat) & (obs_AB['TIME_WINDOW']==window)].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)

    for cat in ['LAB_HBA1C','LAB_LIPIDS','LIFE_BMI','LIFE_ALCOHOL',
                'SYMPTOM_BREAST_LUMP','BREAST_BENIGN','HRT']:
        df_src = med_AB if cat == 'HRT' else obs_AB
        nA_c = df_src[(df_src['CATEGORY']==cat) & (df_src['TIME_WINDOW']=='A')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        nB_c = df_src[(df_src['CATEGORY']==cat) & (df_src['TIME_WINDOW']=='B')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}XW_{cat}_B_over_A']  = (nB_c / (nA_c + 1.0)).astype(float)
        pf[f'{PREFIX}XW_{cat}_B_minus_A'] = (nB_c - nA_c).astype(int)
        pf[f'{PREFIX}XW_{cat}_growth']    = (nB_c > nA_c).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 31: SAME-MONTH SYMPTOM CO-OCCURRENCE (close-in-time clustering)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 31: Symptom temporal clustering...")
    sx_cats = ['SYMPTOM_BREAST_LUMP', 'SYMPTOM_BREAST_PAIN', 'SYMPTOM_NIPPLE']
    sx_events = obs_AB[obs_AB['CATEGORY'].isin(sx_cats)].copy()
    if not sx_events.empty:
        # Number of months in which the patient had any symptom (frequency-by-month)
        n_months_with_sx = sx_events.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].nunique().reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}TC_n_months_with_sx']    = n_months_with_sx.astype(int)
        pf[f'{PREFIX}TC_sustained_sx']        = (n_months_with_sx >= 3).astype(int)  # 3+ months with sx = persistent
        # Symptom span (max - min months before)
        sx_span = sx_events.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min', 'max'])
        pf[f'{PREFIX}TC_sx_span_mo']          = (sx_span['max'] - sx_span['min']).reindex(pf.index, fill_value=0).astype(float)
        # Same-month multi-symptom: ≥2 distinct sx categories in the same month
        multi_in_same_mo = sx_events.groupby(['PATIENT_GUID', 'MONTHS_BEFORE_INDEX'])['CATEGORY'].nunique().reset_index()
        had_multi = multi_in_same_mo[multi_in_same_mo['CATEGORY'] >= 2].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        pf[f'{PREFIX}TC_multi_sx_same_mo']    = (had_multi >= 1).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 32: HEALTHCARE ENGAGEMENT TIMING
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 32: Healthcare engagement timing...")
    # Years in record: span of all events
    all_mb = obs_AB.groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].agg(['min','max'])
    record_span_mo = (all_mb['max'] - all_mb['min']).reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}ENG_record_span_mo']     = record_span_mo.astype(float)
    pf[f'{PREFIX}ENG_record_span_years']  = (record_span_mo / 12.0).astype(float)
    pf[f'{PREFIX}ENG_long_record_5y']     = (record_span_mo >= 60).astype(int)

    # Total events per year of record
    pf[f'{PREFIX}ENG_events_per_year']    = (total_obs_events / record_span_mo.clip(lower=1) * 12.0).astype(float)
    # Recent activity density: events in last 12mo / record_span
    last12 = obs_AB[obs_AB['MONTHS_BEFORE_INDEX'] <= 12].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}ENG_recent_density']     = (last12 / total_obs_events.clip(lower=1)).astype(float)

    # Visit-density flag: high activity in last 6 months
    last6 = obs_AB[obs_AB['MONTHS_BEFORE_INDEX'] <= 6].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}ENG_recent_activity_6mo'] = last6.astype(int)
    pf[f'{PREFIX}ENG_active_recent']      = (last6 >= 5).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 33: LAB-VALUE TRAJECTORY FLAGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 33: Lab trajectory direction flags...")
    for lab_cat in ['LAB_HBA1C', 'LAB_LIPIDS', 'LIFE_BMI', 'LAB_FT4', 'LAB_TSH']:
        sub = obs_AB[(obs_AB['CATEGORY']==lab_cat) & obs_AB['VALUE'].notna()].copy()
        if sub.empty:
            continue
        # Recent (B) mean vs historical (A) mean
        avgA = sub[sub['TIME_WINDOW']=='A'].groupby('PATIENT_GUID')['VALUE'].mean()
        avgB = sub[sub['TIME_WINDOW']=='B'].groupby('PATIENT_GUID')['VALUE'].mean()
        avgA = avgA.reindex(pf.index, fill_value=np.nan)
        avgB = avgB.reindex(pf.index, fill_value=np.nan)
        diff = (avgB - avgA)
        pf[f'{PREFIX}TRJ_{lab_cat}_B_minus_A_val']  = diff.fillna(0).astype(float)
        pf[f'{PREFIX}TRJ_{lab_cat}_worsening']      = (diff > 0).fillna(False).astype(int)
        pf[f'{PREFIX}TRJ_{lab_cat}_improving']      = (diff < 0).fillna(False).astype(int)
        # Max value flag
        max_val = sub.groupby('PATIENT_GUID')['VALUE'].max().reindex(pf.index, fill_value=np.nan)
        pf[f'{PREFIX}TRJ_{lab_cat}_max']            = max_val.fillna(-1).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: ENGAGEMENT-BIAS FLAGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 10: Engagement-bias flags...")

    n_distinct_obs_cats = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0)
    pf[f'{PREFIX}ENG_n_distinct_obs_cats'] = n_distinct_obs_cats.astype(int)
    pf[f'{PREFIX}ENG_n_distinct_med_cats'] = (
        med_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique().reindex(pf.index, fill_value=0).astype(int)
    )
    pf[f'{PREFIX}ENG_high_engagement'] = (n_distinct_obs_cats >= 10).astype(int)

    # Female-specific care pathway engagement
    female_care = has_any_cat(obs_AB, ['REPRO_MENOPAUSE', 'REPRO_MENSTRUAL_FLOW',
                                        'REPRO_PMS', 'REPRO_ATROPHIC_VAG',
                                        'REPRO_PARITY', 'REPRO_TAH', 'REPRO_TAH_BSO'])
    pf[f'{PREFIX}ENG_female_care_pathway'] = female_care

    pf[f'{PREFIX}ENG_polypharm_total'] = pf[f'{PREFIX}TX_cv_med_count'] + pf[f'{PREFIX}TX_mh_med_count']

    # Asthma engagement (collapsed: disease+severity + control combined)
    asthma_cats = ['COMORBID_ASTHMA', 'COMORBID_ASTHMA_CONTROL']
    asthma_eng = sum(has_cat(obs_AB, c) for c in asthma_cats)
    pf[f'{PREFIX}ENG_asthma_score'] = asthma_eng.astype(int)

    # ── Done ─────────────────────────────────────────────────
    n_features = sum(1 for c in pf.columns if c.startswith(PREFIX))
    logger.info(f"  ✓ Breast-specific features built: {n_features} columns")
    return pf
