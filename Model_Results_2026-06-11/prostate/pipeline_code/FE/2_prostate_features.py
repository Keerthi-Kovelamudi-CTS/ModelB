# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — CANCER-SPECIFIC FEATURES (Step 4d)
# Hand-engineered features for prostate prediction:
#   - PSA dynamics (level, velocity/yr, doubling time, free/total ratio)
#   - Lower-urinary-tract symptoms (LUTS) + IPSS severity + retention
#   - Digital rectal exam (DRE) findings
#   - Prostatic conditions (BPH), erectile dysfunction
#   - Bone-metastasis signals (ALP, calcium, pelvic/bone pain)
#   - Constitutional decline (weight loss, low albumin, anaemia)
#   - Treatment flags (5ARI, alpha-blockers, anticholinergics, ED meds)
#   - Composite PSA-weighted risk-surrogate score
#   - Generic temporal / age / engagement scaffolding (shared with breast)
#
# LEAKAGE RULE: PSA presence / test-counts are NOT used (ordering a PSA test is
# part of the diagnostic workup). Only PSA VALUE-derived features are built.
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
    """Build prostate-specific hand-engineered features."""
    if cfg is None: cfg = config
    PREFIX = cfg.PREFIX

    logger.info(f"  PROSTATE-SPECIFIC FEATURES - {window_name}")

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
        """Per-patient aggregate of numeric VALUE for a category (last/first/max/mean)."""
        sub = df[(df['CATEGORY'] == cat) & df['VALUE'].notna()].sort_values('EVENT_DATE')
        if sub.empty:
            return pd.Series(np.nan, index=pf.index)
        g = sub.groupby('PATIENT_GUID')['VALUE']
        s = {'last': g.last, 'first': g.first, 'max': g.max,
             'min': g.min, 'mean': g.mean}[agg]()
        return s.reindex(pf.index)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: PSA DYNAMICS  (VALUE-ONLY — presence/counts are leaky)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 1: PSA dynamics...")

    psa = obs_AB[(obs_AB['CATEGORY'] == 'PSA') & obs_AB['VALUE'].notna()].sort_values('EVENT_DATE')

    if not psa.empty:
        g = psa.groupby('PATIENT_GUID')
        psa_latest = g['VALUE'].last()
        psa_first  = g['VALUE'].first()
        psa_max    = g['VALUE'].max()
        psa_n      = g['VALUE'].size()

        pf[f'{PREFIX}LAB_psa_latest'] = m(psa_latest, default=np.nan)
        pf[f'{PREFIX}LAB_psa_first']  = m(psa_first, default=np.nan)
        pf[f'{PREFIX}LAB_psa_max']    = m(psa_max, default=np.nan)

        # Threshold flags (standard referral / risk bands)
        pf[f'{PREFIX}LAB_psa_elevated_4']  = (m(psa_latest) > 4.0).astype(int)
        pf[f'{PREFIX}LAB_psa_elevated_10'] = (m(psa_latest) > 10.0).astype(int)
        pf[f'{PREFIX}LAB_psa_elevated_20'] = (m(psa_latest) > 20.0).astype(int)
        pf[f'{PREFIX}LAB_psa_very_high']   = (m(psa_latest) > 100.0).astype(int)
        pf[f'{PREFIX}LAB_psa_max_elevated_4'] = (m(psa_max) > 4.0).astype(int)

        # Delta (first → last)
        psa_delta = psa_latest.subtract(psa_first)
        pf[f'{PREFIX}LAB_psa_delta']      = m(psa_delta, default=0)
        pf[f'{PREFIX}LAB_psa_rising']     = (m(psa_delta) > 0).astype(int)
        pf[f'{PREFIX}LAB_psa_rapid_rise'] = (m(psa_delta) > 2.0).astype(int)

        # Velocity (ng/mL per YEAR — clinical standard >0.75 suspicious).
        # Floor the span at 0.5yr: two readings <6mo apart can't give a reliable annualised
        # velocity and a tiny denominator would explode the value, so set it NaN there.
        span_days = (g['EVENT_DATE'].max() - g['EVENT_DATE'].min()).dt.days
        span_years = (span_days / 365.25)
        vel_yr = psa_delta / span_years.where(span_years >= 0.5)
        pf[f'{PREFIX}LAB_psa_velocity_per_year'] = m(vel_yr, default=0)
        pf[f'{PREFIX}LAB_psa_velocity_high']      = (m(vel_yr) > 0.75).astype(int)
        pf[f'{PREFIX}LAB_psa_velocity_very_high'] = (m(vel_yr) > 2.0).astype(int)

        # Percent change across the window (guarded)
        psa_pct = (psa_delta / psa_first.replace(0, np.nan)) * 100.0
        pf[f'{PREFIX}LAB_psa_pct_change'] = m(psa_pct, default=0)
        pf[f'{PREFIX}LAB_psa_pct_up_50']  = (m(psa_pct) > 50).astype(int)
        pf[f'{PREFIX}LAB_psa_pct_up_100'] = (m(psa_pct) > 100).astype(int)

        # Doubling time (months): ln(2) * span_mo / ln(last/first), guarded
        span_mo = (span_days / 30.4375)
        ratio = (psa_latest / psa_first.replace(0, np.nan))
        with np.errstate(all='ignore'):
            dt = np.log(2.0) * span_mo / np.log(ratio.where(ratio > 0))
        dt = dt.where((ratio > 1) & (span_mo > 0))           # only meaningful when rising
        pf[f'{PREFIX}LAB_psa_doubling_time'] = m(dt, default=np.nan)
        pf[f'{PREFIX}LAB_psa_fast_doubling'] = ((m(dt, default=999) > 0) & (m(dt, default=999) < 12)).astype(int)

        pf[f'{PREFIX}LAB_psa_n_readings'] = m(psa_n, default=0).astype(int)
    else:
        for col in ['psa_latest','psa_first','psa_max','psa_elevated_4','psa_elevated_10',
                    'psa_elevated_20','psa_very_high','psa_max_elevated_4','psa_delta',
                    'psa_rising','psa_rapid_rise','psa_velocity_per_year','psa_velocity_high',
                    'psa_velocity_very_high','psa_pct_change','psa_pct_up_50','psa_pct_up_100',
                    'psa_doubling_time','psa_fast_doubling','psa_n_readings']:
            pf[f'{PREFIX}LAB_{col}'] = 0

    # Free PSA + free/total ratio (low ratio <0.15 = suspicious for cancer)
    fpsa_latest = val_series('PSA_FREE', 'last')
    ratio_latest = val_series('PSA_RATIO', 'last')
    pf[f'{PREFIX}LAB_free_psa_latest']  = fpsa_latest.fillna(-1).astype(float)
    pf[f'{PREFIX}LAB_psa_ratio_latest'] = ratio_latest.fillna(-1).astype(float)
    # Codelists may store ratio as a fraction (<=1) or as percent (<=100); handle both.
    _r = ratio_latest.copy()
    _r_frac = _r.where(_r <= 1.0, _r / 100.0)
    pf[f'{PREFIX}LAB_psa_ratio_low'] = ((_r_frac > 0) & (_r_frac < 0.15)).fillna(False).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: LUTS / IPSS / URINARY-RETENTION BURDEN
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 2: LUTS / IPSS / urinary burden...")

    has_luts      = has_cat(obs_AB, 'LUTS')
    n_luts        = count_cat(obs_AB, 'LUTS')
    has_retention = has_cat(obs_AB, 'URINARY_RETENTION')
    has_catheter  = has_cat(obs_AB, 'CATHETER')
    has_ipss      = has_cat(obs_AB, 'IPSS')

    pf[f'{PREFIX}SX_has_luts']        = has_luts
    pf[f'{PREFIX}SX_luts_count']      = n_luts
    pf[f'{PREFIX}SX_luts_multi']      = (n_luts >= 3).astype(int)
    pf[f'{PREFIX}SX_has_retention']   = has_retention
    pf[f'{PREFIX}SX_has_catheter']    = has_catheter
    pf[f'{PREFIX}SX_has_ipss']        = has_ipss

    ipss_latest = val_series('IPSS', 'last')
    ipss_max    = val_series('IPSS', 'max')
    pf[f'{PREFIX}SX_ipss_latest']  = ipss_latest.fillna(-1).astype(float)
    pf[f'{PREFIX}SX_ipss_max']     = ipss_max.fillna(-1).astype(float)
    pf[f'{PREFIX}SX_ipss_severe']  = (ipss_latest >= 20).fillna(False).astype(int)   # severe LUTS
    pf[f'{PREFIX}SX_ipss_moderate'] = ((ipss_latest >= 8) & (ipss_latest < 20)).fillna(False).astype(int)

    urinary_burden = (has_luts + has_retention + has_catheter + has_ipss).astype(int)
    pf[f'{PREFIX}SX_urinary_burden']   = urinary_burden
    pf[f'{PREFIX}SX_urinary_2plus']    = (urinary_burden >= 2).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: DRE (digital rectal exam) FINDINGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 3: DRE findings...")
    has_dre = has_cat(obs_AB, 'DRE')
    n_dre   = count_cat(obs_AB, 'DRE')
    pf[f'{PREFIX}DX_has_dre']        = has_dre
    pf[f'{PREFIX}DX_dre_count']      = n_dre
    pf[f'{PREFIX}DX_dre_repeat']     = (n_dre >= 2).astype(int)
    pf[f'{PREFIX}DX_dre_with_luts']  = ((has_dre == 1) & (has_luts == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: PROSTATIC CONDITIONS (BPH) + SEXUAL / ED
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 4: Prostatic conditions + sexual health...")
    has_prostatic = has_cat(obs_AB, 'PROSTATIC_CONDITIONS')
    has_ed        = has_cat(obs_AB, 'ERECTILE_DYSFUNCTION')
    has_sexual    = has_cat(obs_AB, 'SEXUAL_REPRODUCTIVE')
    has_hormonal  = has_cat(obs_AB, 'HORMONAL')
    pf[f'{PREFIX}RF_has_prostatic_cond'] = has_prostatic
    pf[f'{PREFIX}RF_has_ed']             = has_ed
    pf[f'{PREFIX}RF_has_sexual']         = has_sexual
    pf[f'{PREFIX}RF_has_hormonal']       = has_hormonal
    pf[f'{PREFIX}RF_prostatic_x_luts']   = ((has_prostatic == 1) & (has_luts == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: FAMILY HISTORY / HEREDITARY RISK
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 5: Family history...")
    has_famhx = has_cat(obs_AB, 'FAMILY_HISTORY')
    pf[f'{PREFIX}RF_has_famhx']       = has_famhx
    pf[f'{PREFIX}RF_famhx_x_age50']   = ((has_famhx == 1) & (age >= 50)).astype(int)
    pf[f'{PREFIX}RF_famhx_x_psa4']    = ((has_famhx == 1) & (pf.get(f'{PREFIX}LAB_psa_elevated_4', 0) == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: BONE-METASTASIS SIGNAL (ALP, calcium, pelvic/bone pain)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 6: Bone-metastasis signals...")
    alp_latest = val_series('ALP_BONE_MARKER', 'last')
    alp_max    = val_series('ALP_BONE_MARKER', 'max')
    ca_max     = val_series('CALCIUM', 'max')
    pf[f'{PREFIX}LAB_alp_latest']     = alp_latest.fillna(-1).astype(float)
    pf[f'{PREFIX}LAB_alp_max']        = alp_max.fillna(-1).astype(float)
    pf[f'{PREFIX}LAB_alp_elevated']   = (alp_latest > 130).fillna(False).astype(int)   # bone-met marker
    pf[f'{PREFIX}LAB_calcium_high']   = (ca_max > 2.6).fillna(False).astype(int)        # hypercalcaemia

    has_bone_pain = has_cat(obs_AB, 'PAIN_PELVIC_BONE')
    has_bone_musc = has_cat(obs_AB, 'BONE_MUSCLE')
    pf[f'{PREFIX}SX_has_bone_pain']   = has_bone_pain
    bone_met_signal = (pf[f'{PREFIX}LAB_alp_elevated'] + pf[f'{PREFIX}LAB_calcium_high']
                       + has_bone_pain + has_bone_musc).astype(int)
    pf[f'{PREFIX}RF_bone_met_signal']      = bone_met_signal
    pf[f'{PREFIX}RF_bone_met_2plus']       = (bone_met_signal >= 2).astype(int)
    pf[f'{PREFIX}RF_bonepain_x_alp']       = ((has_bone_pain == 1) & (pf[f'{PREFIX}LAB_alp_elevated'] == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: CONSTITUTIONAL / SYSTEMIC DECLINE
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 7: Constitutional decline...")
    has_constit = has_cat(obs_AB, 'CONSTITUTIONAL')
    has_weight  = has_any_cat(obs_AB, ['WEIGHT', 'BODY_WEIGHT'])
    alb_latest  = val_series('ALBUMIN_PROTEIN', 'last')
    pf[f'{PREFIX}RF_has_constitutional'] = has_constit
    pf[f'{PREFIX}RF_has_weight_event']   = has_weight
    pf[f'{PREFIX}LAB_albumin_low']       = (alb_latest < 35).fillna(False).astype(int)   # low albumin
    # Anaemia from FBC haemoglobin (low Hb common in advanced disease)
    hb = val_series('FBC_HAEMATOLOGY', 'min')
    pf[f'{PREFIX}LAB_anaemia_proxy']     = ((hb > 0) & (hb < 110)).fillna(False).astype(int)
    pf[f'{PREFIX}RF_constitutional_load'] = (has_constit + has_weight
                                             + pf[f'{PREFIX}LAB_albumin_low']).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: UROLOGICAL PATHOLOGY (haematuria / UTI / urine markers)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 8: Urological pathology...")
    has_haematuria = has_cat(obs_AB, 'HAEMATURIA')
    has_uti        = has_cat(obs_AB, 'UTI')
    has_urine_mark = has_cat(obs_AB, 'URINE_MARKERS')
    has_uro_path   = has_cat(obs_AB, 'UROLOGY_PATHWAY')
    pf[f'{PREFIX}SX_has_haematuria']  = has_haematuria
    pf[f'{PREFIX}SX_has_uti']         = has_uti
    pf[f'{PREFIX}SX_has_urine_marker'] = has_urine_mark
    pf[f'{PREFIX}DX_has_uro_pathway'] = has_uro_path
    pf[f'{PREFIX}SX_haematuria_x_luts'] = ((has_haematuria == 1) & (has_luts == 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: MEDICATION FLAGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 9: Medication flags...")
    has_5ari       = has_cat(med_AB, '5ARI')
    has_alpha      = has_cat(med_AB, 'ALPHA_BLOCKERS')
    has_anticholin = has_cat(med_AB, 'ANTICHOLINERGICS')
    has_ed_med     = has_cat(med_AB, 'ED_MEDICATIONS')
    has_uti_abx    = has_cat(med_AB, 'UTI_ANTIBIOTICS')
    has_pain_esc   = has_cat(med_AB, 'PAIN_ESCALATION')
    pf[f'{PREFIX}TX_on_5ari']          = has_5ari
    pf[f'{PREFIX}TX_on_alpha_blocker'] = has_alpha
    pf[f'{PREFIX}TX_on_anticholinergic'] = has_anticholin
    pf[f'{PREFIX}TX_on_ed_med']        = has_ed_med
    pf[f'{PREFIX}TX_on_uti_abx']       = has_uti_abx
    pf[f'{PREFIX}TX_on_pain_escalation'] = has_pain_esc
    pf[f'{PREFIX}TX_bph_treated']      = ((has_5ari == 1) | (has_alpha == 1)).astype(int)
    pf[f'{PREFIX}TX_dual_bph_therapy'] = ((has_5ari == 1) & (has_alpha == 1)).astype(int)
    n_uti_abx = count_cat(med_AB, 'UTI_ANTIBIOTICS')
    pf[f'{PREFIX}TX_recurrent_uti_abx'] = (n_uti_abx >= 3).astype(int)
    pf[f'{PREFIX}TX_med_burden'] = (has_5ari + has_alpha + has_anticholin + has_ed_med
                                    + has_uti_abx + has_pain_esc).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: COMPOSITE PROSTATE RISK-SURROGATE SCORE
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 10: Composite risk score...")
    risk = pd.Series(0.0, index=pf.index)
    # Age bands
    risk = risk + (age >= 50).astype(int) * 0.5
    risk = risk + (age >= 65).astype(int) * 1.0
    risk = risk + (age >= 75).astype(int) * 1.0
    # PSA level (dominant signal)
    risk = risk + pf.get(f'{PREFIX}LAB_psa_elevated_4', 0) * 2.0
    risk = risk + pf.get(f'{PREFIX}LAB_psa_elevated_10', 0) * 2.0
    risk = risk + pf.get(f'{PREFIX}LAB_psa_elevated_20', 0) * 2.0
    risk = risk + pf.get(f'{PREFIX}LAB_psa_very_high', 0) * 3.0
    risk = risk + pf.get(f'{PREFIX}LAB_psa_velocity_high', 0) * 1.0
    risk = risk + pf.get(f'{PREFIX}LAB_psa_ratio_low', 0) * 1.5
    # DRE + urinary
    risk = risk + has_dre * 0.5
    risk = risk + has_luts * 0.5
    risk = risk + pf[f'{PREFIX}SX_ipss_severe'] * 0.5
    risk = risk + has_retention * 0.5
    risk = risk + has_haematuria * 0.5
    # Bone-met + constitutional (advanced disease)
    risk = risk + bone_met_signal * 1.0
    risk = risk + pf[f'{PREFIX}LAB_albumin_low'] * 0.5
    risk = risk + pf[f'{PREFIX}LAB_anaemia_proxy'] * 0.5
    # Family history
    risk = risk + has_famhx * 1.0

    pf[f'{PREFIX}RISK_score']        = risk.astype(float)
    pf[f'{PREFIX}RISK_low']          = (risk < 2.0).astype(int)
    pf[f'{PREFIX}RISK_intermediate'] = ((risk >= 2.0) & (risk < 4.0)).astype(int)
    pf[f'{PREFIX}RISK_high']         = (risk >= 4.0).astype(int)
    pf[f'{PREFIX}RISK_very_high']    = (risk >= 6.0).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: AGE × CLINICAL-STATE INTERACTIONS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 11: Age × clinical interactions...")
    age50 = (age >= 50).astype(int)
    age65 = (age >= 65).astype(int)
    age75 = (age >= 75).astype(int)
    psa4  = pf.get(f'{PREFIX}LAB_psa_elevated_4', pd.Series(0, index=pf.index))
    psa10 = pf.get(f'{PREFIX}LAB_psa_elevated_10', pd.Series(0, index=pf.index))
    pf[f'{PREFIX}AGEX_psa4_x_age50']     = (psa4 * age50)
    pf[f'{PREFIX}AGEX_psa4_x_age65']     = (psa4 * age65)
    pf[f'{PREFIX}AGEX_psa10_x_age65']    = (psa10 * age65)
    pf[f'{PREFIX}AGEX_luts_x_age65']     = (has_luts * age65)
    pf[f'{PREFIX}AGEX_dre_x_age65']      = (has_dre * age65)
    pf[f'{PREFIX}AGEX_retention_x_age65'] = (has_retention * age65)
    pf[f'{PREFIX}AGEX_famhx_x_age50']    = (has_famhx * age50)
    pf[f'{PREFIX}AGEX_bonepain_x_age65'] = (has_bone_pain * age65)
    pf[f'{PREFIX}AGEX_haematuria_x_age65'] = (has_haematuria * age65)

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: TRIPLE INTERACTIONS (non-linear combos)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 12: Triple interactions...")
    pf[f'{PREFIX}TRI_psa4_dre_age65']     = (psa4 * has_dre * age65)
    pf[f'{PREFIX}TRI_psa4_luts_age65']    = (psa4 * has_luts * age65)
    pf[f'{PREFIX}TRI_psa10_famhx_age50']  = (psa10 * has_famhx * age50)
    pf[f'{PREFIX}TRI_bonepain_alp_psa']   = (has_bone_pain * pf[f'{PREFIX}LAB_alp_elevated'] * psa4)
    pf[f'{PREFIX}TRI_luts_retention_age65'] = (has_luts * has_retention * age65)
    pf[f'{PREFIX}TRI_psavel_psa4_age65']  = (pf.get(f'{PREFIX}LAB_psa_velocity_high', 0) * psa4 * age65)

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: RECENT-WINDOW (B) SYMPTOM BURST
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 13: Recent-window symptom burst...")
    luts_B   = has_cat(obs_B, 'LUTS')
    luts_A   = has_cat(obs_A, 'LUTS')
    haem_B   = has_cat(obs_B, 'HAEMATURIA')
    ret_B    = has_cat(obs_B, 'URINARY_RETENTION')
    bone_B   = has_cat(obs_B, 'PAIN_PELVIC_BONE')
    sx_B = (luts_B + haem_B + ret_B + bone_B).astype(int)
    pf[f'{PREFIX}BURST_sx_in_B']        = (sx_B >= 1).astype(int)
    pf[f'{PREFIX}BURST_multi_sx_in_B']  = (sx_B >= 2).astype(int)
    pf[f'{PREFIX}BURST_new_luts_in_B']  = ((luts_B == 1) & (luts_A == 0)).astype(int)
    pf[f'{PREFIX}BURST_new_haematuria_B'] = ((haem_B == 1) & (has_cat(obs_A, 'HAEMATURIA') == 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: SEQUENTIAL + FIRST-EVENT-DATE PATTERNS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 14: Sequential + first-event patterns...")

    def first_event(cat, df=obs_AB):
        s = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        return s.reindex(pf.index, fill_value=np.nan)

    first_luts = first_event('LUTS')
    first_ret  = first_event('URINARY_RETENTION')
    first_haem = first_event('HAEMATURIA')
    first_dre  = first_event('DRE')

    # LUTS → retention within 12mo (progressive obstruction)
    pf[f'{PREFIX}SEQ_luts_then_retention_12m'] = (
        ((first_luts - first_ret) >= 0) & ((first_luts - first_ret) <= 12)
    ).fillna(False).astype(int)
    pf[f'{PREFIX}FIRST_luts_mo_before']  = first_luts.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_haem_mo_before']  = first_haem.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_dre_mo_before']   = first_dre.fillna(999).astype(float)
    pf[f'{PREFIX}FIRST_luts_within_12mo'] = (first_luts <= 12).fillna(False).astype(int)
    pf[f'{PREFIX}FIRST_haem_within_12mo'] = (first_haem <= 12).fillna(False).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 15: PATIENT-LEVEL Z-SCORE LABS (engagement-adjusted)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 15: Patient z-score labs...")

    def patient_zscore(cat, agg='mean'):
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

    pf[f'{PREFIX}Z_psa_max']     = patient_zscore('PSA', 'max')
    pf[f'{PREFIX}Z_psa_mean']    = patient_zscore('PSA', 'mean')
    pf[f'{PREFIX}Z_alp_max']     = patient_zscore('ALP_BONE_MARKER', 'max')
    pf[f'{PREFIX}Z_calcium_max'] = patient_zscore('CALCIUM', 'max')
    pf[f'{PREFIX}Z_psa_high']    = (pf[f'{PREFIX}Z_psa_max'] > 1.5).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 16: PER-CATEGORY VELOCITY (B - A counts)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 16: Per-category velocity...")
    def cat_velocity(cat, df_AB=obs_AB):
        nA = df_AB[(df_AB['CATEGORY']==cat) & (df_AB['TIME_WINDOW']=='A')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        nB = df_AB[(df_AB['CATEGORY']==cat) & (df_AB['TIME_WINDOW']=='B')].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
        return (nB - nA).astype(int)

    for cat in ['LUTS','URINARY_RETENTION','HAEMATURIA','PAIN_PELVIC_BONE',
                'PROSTATIC_CONDITIONS','DRE','UTI','ERECTILE_DYSFUNCTION']:
        pf[f'{PREFIX}VEL_{cat}_B_minus_A'] = cat_velocity(cat)
    pf[f'{PREFIX}VEL_alpha_B_minus_A'] = cat_velocity('ALPHA_BLOCKERS', med_AB)

    # ══════════════════════════════════════════════════════════
    # BLOCK 17: TIME-DECAY INTENSITY + ACCELERATION
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 17: Time-decay intensity + acceleration...")
    KEY_CATS = ['LUTS', 'URINARY_RETENTION', 'HAEMATURIA', 'PAIN_PELVIC_BONE',
                'PROSTATIC_CONDITIONS', 'DRE', 'PSA', 'ERECTILE_DYSFUNCTION',
                'ALP_BONE_MARKER', 'FAMILY_HISTORY', 'CONSTITUTIONAL', 'IPSS']
    HALFLIFE_MO = 12.0
    def cat_decay(cat, df_AB=obs_AB):
        sub = df_AB[df_AB['CATEGORY'] == cat]
        if len(sub) == 0:
            return pd.Series(0.0, index=pf.index)
        w = np.exp(-sub['MONTHS_BEFORE_INDEX'].astype(float).clip(lower=0) / HALFLIFE_MO)
        return w.groupby(sub['PATIENT_GUID']).sum().reindex(pf.index, fill_value=0.0)
    for cat in KEY_CATS:
        pf[f'{PREFIX}DECAY_{cat}'] = cat_decay(cat).astype(float)
    pf[f'{PREFIX}DECAY_ALPHA_BLOCKERS'] = cat_decay('ALPHA_BLOCKERS', med_AB).astype(float)

    def cat_bin(cat, lo, hi, df_AB=obs_AB):
        sub = df_AB[(df_AB['CATEGORY'] == cat) &
                    (df_AB['MONTHS_BEFORE_INDEX'] >= lo) & (df_AB['MONTHS_BEFORE_INDEX'] < hi)]
        return sub.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)
    for cat in ['LUTS', 'HAEMATURIA', 'URINARY_RETENTION', 'PAIN_PELVIC_BONE', 'DRE']:
        rec = cat_bin(cat, 1, 15); mid = cat_bin(cat, 15, 30); ear = cat_bin(cat, 30, 60)
        pf[f'{PREFIX}ACCEL_{cat}'] = ((rec - mid) - (mid - ear)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 18: ENGAGEMENT-NORMALIZED SIGNAL
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 18: Engagement-normalized signal...")
    total_obs_events = obs_AB.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0).clip(lower=1)
    total_med_events = med_AB.groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0).clip(lower=1)

    def cat_count(cat, df_AB=obs_AB):
        return df_AB[df_AB['CATEGORY']==cat].groupby('PATIENT_GUID').size().reindex(pf.index, fill_value=0)

    pf[f'{PREFIX}NORM_luts_per_total']       = (cat_count('LUTS') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_haematuria_per_total'] = (cat_count('HAEMATURIA') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_bonepain_per_total']   = (cat_count('PAIN_PELVIC_BONE') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_dre_per_total']        = (cat_count('DRE') / total_obs_events).astype(float)
    pf[f'{PREFIX}NORM_alpha_per_med']        = (cat_count('ALPHA_BLOCKERS', med_AB) / total_med_events).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 19: CROSS-CATEGORY CO-OCCURRENCE FLAGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 19: Cross-category co-occurrence...")
    prostate_indicators = (psa4 + has_dre + has_luts + has_retention + has_haematuria
                           + has_bone_pain + has_famhx).astype(int)
    pf[f'{PREFIX}COOC_prostate_indicators']       = prostate_indicators
    pf[f'{PREFIX}COOC_2plus_indicators']          = (prostate_indicators >= 2).astype(int)
    pf[f'{PREFIX}COOC_3plus_indicators']          = (prostate_indicators >= 3).astype(int)
    pf[f'{PREFIX}COOC_psa_and_dre']               = ((psa4 == 1) & (has_dre == 1)).astype(int)
    pf[f'{PREFIX}COOC_psa_and_bone_signal']       = ((psa4 == 1) & (bone_met_signal >= 1)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 20: TIME-SINCE-LAST-EVENT (recency per category)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 20: Time-since-last-event recency...")
    def last_event(cat, df_AB=obs_AB):
        s = df_AB[df_AB['CATEGORY']==cat].groupby('PATIENT_GUID')['MONTHS_BEFORE_INDEX'].min()
        return s.reindex(pf.index, fill_value=np.nan)

    last_luts = last_event('LUTS')
    last_haem = last_event('HAEMATURIA')
    last_bone = last_event('PAIN_PELVIC_BONE')
    pf[f'{PREFIX}RECENT_luts_in_6mo']  = (last_luts <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_haem_in_6mo']  = (last_haem <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_bone_in_6mo']  = (last_bone <= 6).fillna(False).astype(int)
    pf[f'{PREFIX}RECENT_luts_gap_mo']  = last_luts.fillna(60).astype(float)

    # ══════════════════════════════════════════════════════════
    # BLOCK 21: PER-KEY-CATEGORY TEMPORAL GRANULARITY (event-age + worsening)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 21: Per-category temporal granularity...")
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
    # BLOCK 22: INTERVAL + TREND SLOPE + LAST-18MO PER KEY CAT
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 22: Interval + trend + last-18mo features...")
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
    # BLOCK 23: AGE POLYNOMIALS + BANDS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 23: Age polynomials + bands...")
    age_f = age.astype(float)
    pf[f'{PREFIX}AGEP_age_sq']   = (age_f ** 2).astype(float)
    pf[f'{PREFIX}AGEP_age_cu']   = (age_f ** 3 / 100.0).astype(float)
    pf[f'{PREFIX}AGEP_age_log']  = np.log1p(age_f).astype(float)
    pf[f'{PREFIX}AGEP_age_sqrt'] = np.sqrt(age_f).astype(float)
    for lo, hi in [(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,99)]:
        pf[f'{PREFIX}AGEB_band_{lo}_{hi}'] = ((age_f >= lo) & (age_f < hi)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 24: PATIENT SUMMARY COUNTS + ENGAGEMENT TIMING
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 24: Summary counts + engagement timing...")
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
    # BLOCK 25: LAB-VALUE TRAJECTORY FLAGS (recent vs historical)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 25: Lab trajectory direction flags...")
    for lab_cat in ['PSA', 'ALP_BONE_MARKER', 'CALCIUM', 'ALBUMIN_PROTEIN', 'IPSS']:
        sub = obs_AB[(obs_AB['CATEGORY']==lab_cat) & obs_AB['VALUE'].notna()].copy()
        if sub.empty:
            continue
        avgA = sub[sub['TIME_WINDOW']=='A'].groupby('PATIENT_GUID')['VALUE'].mean().reindex(pf.index, fill_value=np.nan)
        avgB = sub[sub['TIME_WINDOW']=='B'].groupby('PATIENT_GUID')['VALUE'].mean().reindex(pf.index, fill_value=np.nan)
        diff = (avgB - avgA)
        pf[f'{PREFIX}TRJ_{lab_cat}_B_minus_A'] = diff.fillna(0).astype(float)
        pf[f'{PREFIX}TRJ_{lab_cat}_worsening'] = (diff > 0).fillna(False).astype(int)
        pf[f'{PREFIX}TRJ_{lab_cat}_max']       = sub.groupby('PATIENT_GUID')['VALUE'].max().reindex(pf.index, fill_value=np.nan).fillna(-1).astype(float)

    # ── Done ─────────────────────────────────────────────────
    n_features = sum(1 for c in pf.columns if c.startswith(PREFIX))
    logger.info(f"  ✓ Prostate-specific features built: {n_features} columns")
    return pf
