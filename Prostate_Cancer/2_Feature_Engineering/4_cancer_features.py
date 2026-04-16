# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — CANCER-SPECIFIC FEATURES (Step 4d)
# PSA dynamics, urinary symptom patterns, bone metastasis signals,
# treatment flags, and a Gleason/risk surrogate score.
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
    """
    Prostate-specific features: PSA dynamics, urinary patterns,
    bone pain signals, treatment flags, risk surrogate score.

    Parameters
    ----------
    clin_df : pd.DataFrame  — observation/lab rows for this window
    med_df  : pd.DataFrame  — medication rows for this window
    existing_fm : pd.DataFrame — current feature matrix (index=PATIENT_GUID)
    window_name : str — e.g. '3MO'
    cfg : module, optional — config module (defaults to config)

    Returns
    -------
    pd.DataFrame — cancer-specific features, indexed by PATIENT_GUID
    """
    if cfg is None:
        cfg = config

    PREFIX = cfg.PREFIX  # 'PROST_'

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
    obs_A = clin[clin['TIME_WINDOW'] == 'A'].copy()
    obs_B = clin[clin['TIME_WINDOW'] == 'B'].copy()
    med_AB = med[med['TIME_WINDOW'].isin(['A', 'B'])].copy()
    med_A = med[med['TIME_WINDOW'] == 'A'].copy()
    med_B = med[med['TIME_WINDOW'] == 'B'].copy()

    age = existing_fm['AGE_AT_INDEX'].reindex(pf.index).fillna(65)
    sex_raw = clin.drop_duplicates('PATIENT_GUID').set_index('PATIENT_GUID')['SEX']
    sex = sex_raw.reindex(pf.index).fillna('M')  # prostate cancer is male

    # ── Helpers ───────────────────────────────────────────────
    def has_cat(df, cat):
        has = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0
        return pf.index.map(has).fillna(False).astype(int)

    def count_cat(df, cat):
        cnt = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size()
        return pf.index.map(cnt).fillna(0).astype(int)

    m = lambda s, default=0: s.reindex(pf.index).fillna(default)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: PSA DYNAMICS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 1: PSA dynamics...")

    pf[f'{PREFIX}has_psa_monitoring'] = has_cat(obs_AB, 'PSA')
    psa_count_A = count_cat(obs_A, 'PSA')
    psa_count_B = count_cat(obs_B, 'PSA')
    pf[f'{PREFIX}psa_count_A'] = psa_count_A
    pf[f'{PREFIX}psa_count_B'] = psa_count_B
    pf[f'{PREFIX}psa_count_total'] = count_cat(obs_AB, 'PSA')
    pf[f'{PREFIX}psa_acceleration'] = (psa_count_B > psa_count_A).astype(int)

    # PSA lab values
    lab_AB = obs_AB[obs_AB['VALUE'].notna()].copy()

    def get_lab_latest(category, term_pattern):
        mask = (lab_AB['CATEGORY'] == category) & (
            lab_AB['TERM'].str.contains(term_pattern, case=False, na=False)
        )
        vals = lab_AB[mask].sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        return vals

    def get_lab_first(category, term_pattern):
        mask = (lab_AB['CATEGORY'] == category) & (
            lab_AB['TERM'].str.contains(term_pattern, case=False, na=False)
        )
        vals = lab_AB[mask].sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first()
        return vals

    try:
        # Get all PSA readings per patient (sorted by date)
        psa_mask = (lab_AB['CATEGORY'] == 'PSA') & (
            lab_AB['TERM'].str.contains('prostate specific antigen|Serum PSA', case=False, na=False)
        )
        psa_all = lab_AB[psa_mask].sort_values('EVENT_DATE')

        psa_latest = psa_all.groupby('PATIENT_GUID')['VALUE'].last()
        psa_first = psa_all.groupby('PATIENT_GUID')['VALUE'].first()
        psa_max = psa_all.groupby('PATIENT_GUID')['VALUE'].max()
        psa_count = psa_all.groupby('PATIENT_GUID')['VALUE'].count()

        pf[f'{PREFIX}LAB_psa_latest'] = m(psa_latest, default=np.nan)
        pf[f'{PREFIX}LAB_psa_first'] = m(psa_first, default=np.nan)
        pf[f'{PREFIX}LAB_psa_max'] = m(psa_max, default=np.nan)
        pf[f'{PREFIX}LAB_psa_test_count'] = m(psa_count, default=0)

        # Threshold flags
        pf[f'{PREFIX}LAB_psa_elevated_4'] = (m(psa_latest) > 4.0).astype(int)
        pf[f'{PREFIX}LAB_psa_elevated_10'] = (m(psa_latest) > 10.0).astype(int)
        pf[f'{PREFIX}LAB_psa_elevated_20'] = (m(psa_latest) > 20.0).astype(int)
        pf[f'{PREFIX}LAB_psa_very_high'] = (m(psa_latest) > 100.0).astype(int)

        # PSA delta (change from first to last)
        psa_delta = psa_latest.subtract(psa_first)
        pf[f'{PREFIX}LAB_psa_delta'] = m(psa_delta, default=0)
        pf[f'{PREFIX}LAB_psa_rising'] = (m(psa_delta) > 0).astype(int)
        pf[f'{PREFIX}LAB_psa_rapid_rise'] = (m(psa_delta) > 2.0).astype(int)

        # PSA velocity (change per month)
        def psa_velocity(group):
            if len(group) < 2:
                return np.nan
            first_date = group['EVENT_DATE'].iloc[0]
            last_date = group['EVENT_DATE'].iloc[-1]
            months = (last_date - first_date).days / 30.44
            if months < 1:
                return np.nan
            return (group['VALUE'].iloc[-1] - group['VALUE'].iloc[0]) / months
        psa_vel = psa_all.groupby('PATIENT_GUID').apply(psa_velocity)
        pf[f'{PREFIX}LAB_psa_velocity'] = m(psa_vel, default=0)
        pf[f'{PREFIX}LAB_psa_velocity_high'] = (m(psa_vel) > 0.75).astype(int)  # >0.75 ng/mL/yr is suspicious

        # PSA doubling time (months) — ln(2) / slope
        def psa_doubling_time(group):
            if len(group) < 2:
                return np.nan
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].iloc[0]).dt.days.values.astype(float) / 30.44  # months
            y = group['VALUE'].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 2:
                return np.nan
            x, y = x[mask], y[mask]
            if x.max() == x.min() or y[0] <= 0 or y[-1] <= 0:
                return np.nan
            try:
                log_y = np.log(y + 0.01)
                slope = np.polyfit(x, log_y, 1)[0]
                if slope <= 0:
                    return np.nan  # not doubling
                return np.log(2) / slope  # months to double
            except Exception:
                return np.nan
        psa_dt = psa_all.groupby('PATIENT_GUID').apply(psa_doubling_time)
        pf[f'{PREFIX}LAB_psa_doubling_time'] = m(psa_dt, default=np.nan)
        pf[f'{PREFIX}LAB_psa_fast_doubling'] = (m(psa_dt) < 24).astype(int)  # <24mo is concerning
        pf[f'{PREFIX}LAB_psa_very_fast_doubling'] = (m(psa_dt) < 12).astype(int)  # <12mo is high risk

        # PSA percentile (where does this patient fall vs all patients)
        all_psa_values = psa_latest.dropna()
        if len(all_psa_values) > 10:
            percentiles = all_psa_values.rank(pct=True)
            pf[f'{PREFIX}LAB_psa_percentile'] = m(percentiles, default=0.5)
            pf[f'{PREFIX}LAB_psa_top5pct'] = (m(percentiles) >= 0.95).astype(int)
            pf[f'{PREFIX}LAB_psa_top10pct'] = (m(percentiles) >= 0.90).astype(int)
            pf[f'{PREFIX}LAB_psa_top25pct'] = (m(percentiles) >= 0.75).astype(int)

        # PSA age-adjusted: PSA / age (older men have naturally higher PSA)
        pf[f'{PREFIX}LAB_psa_age_ratio'] = np.where(
            age > 0, m(psa_latest, default=0) / age, 0
        )
        pf[f'{PREFIX}LAB_psa_age_adjusted_high'] = (
            ((age < 60) & (m(psa_latest) > 3.0)) |
            ((age >= 60) & (age < 70) & (m(psa_latest) > 4.0)) |
            ((age >= 70) & (m(psa_latest) > 5.0))
        ).astype(int)

        # Window B PSA vs Window A PSA (recent vs early)
        psa_A = psa_all[psa_all['TIME_WINDOW'] == 'A'].groupby('PATIENT_GUID')['VALUE'].mean()
        psa_B = psa_all[psa_all['TIME_WINDOW'] == 'B'].groupby('PATIENT_GUID')['VALUE'].mean()
        psa_window_delta = psa_B.subtract(psa_A)
        pf[f'{PREFIX}LAB_psa_window_delta'] = m(psa_window_delta, default=0)
        pf[f'{PREFIX}LAB_psa_window_rising'] = (m(psa_window_delta) > 0).astype(int)

    except Exception as e:
        logger.warning(f"  PSA features failed: {e}, defaulting to 0")
        for col in ['LAB_psa_latest', 'LAB_psa_first', 'LAB_psa_max']:
            pf[f'{PREFIX}{col}'] = np.nan
        for col in ['LAB_psa_elevated_4', 'LAB_psa_elevated_10', 'LAB_psa_elevated_20',
                     'LAB_psa_very_high', 'LAB_psa_delta', 'LAB_psa_rising', 'LAB_psa_rapid_rise',
                     'LAB_psa_velocity', 'LAB_psa_velocity_high', 'LAB_psa_fast_doubling',
                     'LAB_psa_very_fast_doubling', 'LAB_psa_test_count']:
            pf[f'{PREFIX}{col}'] = 0

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: URINARY SYMPTOM PATTERNS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 2: Urinary symptom patterns...")

    pf[f'{PREFIX}has_luts'] = has_cat(obs_AB, 'LUTS')
    urinary_A = count_cat(obs_A, 'LUTS')
    urinary_B = count_cat(obs_B, 'LUTS')
    pf[f'{PREFIX}urinary_count_A'] = urinary_A
    pf[f'{PREFIX}urinary_count_B'] = urinary_B
    pf[f'{PREFIX}urinary_acceleration'] = (urinary_B > urinary_A).astype(int)

    pf[f'{PREFIX}has_haematuria'] = has_cat(obs_AB, 'HAEMATURIA')
    pf[f'{PREFIX}haematuria_count_B'] = count_cat(obs_B, 'HAEMATURIA')

    pf[f'{PREFIX}has_erectile_dysfunction'] = has_cat(obs_AB, 'ERECTILE_DYSFUNCTION')

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: BONE / METASTATIC SIGNALS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 3: Bone/metastatic signals...")

    pf[f'{PREFIX}has_pelvic_bone_pain'] = has_cat(obs_AB, 'PAIN_PELVIC_BONE')
    pf[f'{PREFIX}has_bone_muscle'] = has_cat(obs_AB, 'BONE_MUSCLE')
    pf[f'{PREFIX}has_any_bone_symptom'] = (
        (pf[f'{PREFIX}has_pelvic_bone_pain'] == 1) | (pf[f'{PREFIX}has_bone_muscle'] == 1)
    ).astype(int)

    bone_cats = ['PAIN_PELVIC_BONE', 'BONE_MUSCLE']
    bone_A = obs_A[obs_A['CATEGORY'].isin(bone_cats)].groupby('PATIENT_GUID').size()
    bone_B = obs_B[obs_B['CATEGORY'].isin(bone_cats)].groupby('PATIENT_GUID').size()
    pf[f'{PREFIX}bone_count_A'] = m(bone_A)
    pf[f'{PREFIX}bone_count_B'] = m(bone_B)
    pf[f'{PREFIX}bone_acceleration'] = (m(bone_B) > m(bone_A)).astype(int)

    # Alkaline phosphatase (bone metastasis marker)
    try:
        alp = get_lab_latest('ALP_BONE_MARKER', 'alkaline phosphatase')
        pf[f'{PREFIX}LAB_alp_latest'] = m(alp, default=np.nan)
        pf[f'{PREFIX}LAB_alp_elevated'] = (m(alp) > 130).astype(int)  # upper normal ~130 U/L
    except Exception as e:
        logger.warning(f"  ALP features failed: {e}, defaulting")
        pf[f'{PREFIX}LAB_alp_latest'] = np.nan
        pf[f'{PREFIX}LAB_alp_elevated'] = 0

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: OTHER LAB PROGNOSTIC FEATURES
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 4: Lab prognostic features...")

    # CRP (inflammation)
    try:
        crp = get_lab_latest('HAEMATOLOGY', 'C reactive protein')
        pf[f'{PREFIX}LAB_crp_latest'] = m(crp, default=np.nan)
        pf[f'{PREFIX}LAB_crp_elevated'] = (m(crp) > 10).astype(int)
    except Exception as e:
        logger.warning(f"  CRP features failed: {e}")
        pf[f'{PREFIX}LAB_crp_latest'] = np.nan
        pf[f'{PREFIX}LAB_crp_elevated'] = 0

    # Haemoglobin (anaemia — advanced disease)
    try:
        hb = get_lab_latest('FBC_HAEMATOLOGY', 'haemoglobin|hemoglobin')
        pf[f'{PREFIX}LAB_hb_latest'] = m(hb, default=np.nan)
        pf[f'{PREFIX}LAB_hb_low'] = (m(hb) < 120).astype(int)  # <120 g/L = anaemia in males
    except Exception as e:
        logger.warning(f"  Hb features failed: {e}")
        pf[f'{PREFIX}LAB_hb_latest'] = np.nan
        pf[f'{PREFIX}LAB_hb_low'] = 0

    # Testosterone (relevant for hormonal therapy monitoring)
    try:
        testo = get_lab_latest('HORMONAL', 'testosterone')
        pf[f'{PREFIX}LAB_testosterone_latest'] = m(testo, default=np.nan)
        pf[f'{PREFIX}LAB_testosterone_low'] = (m(testo) < 8).astype(int)  # <8 nmol/L = low
    except Exception as e:
        logger.warning(f"  Testosterone features failed: {e}")
        pf[f'{PREFIX}LAB_testosterone_latest'] = np.nan
        pf[f'{PREFIX}LAB_testosterone_low'] = 0

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: TREATMENT FLAGS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 5: Treatment flags...")

    pf[f'{PREFIX}has_alpha_blockers'] = has_cat(med_AB, 'ALPHA_BLOCKERS')
    pf[f'{PREFIX}has_5ari'] = has_cat(med_AB, '5ARI')
    pf[f'{PREFIX}uti_abx_count_A'] = count_cat(med_A, 'UTI_ANTIBIOTICS')
    pf[f'{PREFIX}uti_abx_count_B'] = count_cat(med_B, 'UTI_ANTIBIOTICS')
    pf[f'{PREFIX}pain_med_count_B'] = count_cat(med_B, 'PAIN_ESCALATION')
    pf[f'{PREFIX}has_anticholinergics'] = has_cat(med_AB, 'ANTICHOLINERGICS')
    pf[f'{PREFIX}has_catheter_supplies'] = has_cat(med_AB, 'CATHETER_SUPPLIES')
    pf[f'{PREFIX}has_ed_meds'] = has_cat(med_AB, 'ED_MEDICATIONS')

    # BPH treatment combo (alpha blocker + 5-ARI = managed benign disease)
    pf[f'{PREFIX}bph_combo_treatment'] = (
        (pf[f'{PREFIX}has_alpha_blockers'] == 1) &
        (pf[f'{PREFIX}has_5ari'] == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: DIAGNOSTIC PATHWAY FEATURES
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 6: Diagnostic pathway features...")

    pf[f'{PREFIX}has_urology_imaging'] = has_cat(obs_AB, 'UROLOGY_IMAGING')
    pf[f'{PREFIX}has_dre'] = has_cat(obs_AB, 'DRE')
    pf[f'{PREFIX}has_cystoscopy'] = has_cat(obs_AB, 'CYSTOSCOPY')
    pf[f'{PREFIX}has_urology_pathway'] = has_cat(obs_AB, 'UROLOGY_PATHWAY')
    pf[f'{PREFIX}has_ipss'] = has_cat(obs_AB, 'IPSS')
    pf[f'{PREFIX}has_urinary_retention'] = has_cat(obs_AB, 'URINARY_RETENTION')
    pf[f'{PREFIX}has_prostatic_conditions'] = has_cat(obs_AB, 'PROSTATIC_CONDITIONS')
    pf[f'{PREFIX}has_family_history'] = has_cat(obs_AB, 'FAMILY_HISTORY')

    # Symptom to investigation gap
    urinary_cats = ['LUTS', 'HAEMATURIA', 'URINARY_RETENTION']
    first_symptom = obs_AB[obs_AB['CATEGORY'].isin(urinary_cats)].groupby('PATIENT_GUID')['EVENT_DATE'].min()
    inv_cats = ['UROLOGY_IMAGING', 'CYSTOSCOPY', 'DRE']
    imaging_df = obs_AB[obs_AB['CATEGORY'].isin(inv_cats)]
    first_imaging = imaging_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap = first_imaging.subtract(first_symptom).dt.days
    pf[f'{PREFIX}DX_symptom_to_imaging_days'] = m(gap, default=-1)

    # ══════════════════════════════════════════════════════════
    # BLOCK 7: RISK SURROGATE SCORE
    # Loosely based on D'Amico risk stratification
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 7: Risk surrogate score...")

    pf[f'{PREFIX}risk_score'] = (
        (age >= 65).astype(int) +
        pf.get(f'{PREFIX}LAB_psa_elevated_10', pd.Series(0, index=pf.index)).astype(int) +
        pf.get(f'{PREFIX}has_luts', pd.Series(0, index=pf.index)).astype(int) +
        pf.get(f'{PREFIX}has_any_bone_symptom', pd.Series(0, index=pf.index)).astype(int) +
        pf.get(f'{PREFIX}has_urology_imaging', pd.Series(0, index=pf.index)).astype(int)
    )

    # ══════════════════════════════════════════════════════════
    # BLOCK 8: RISK FACTORS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 8: Risk factors...")

    pf[f'{PREFIX}RF_peak_age'] = ((age >= 50) & (age <= 80)).astype(int)
    pf[f'{PREFIX}RF_elderly'] = (age >= 70).astype(int)
    pf[f'{PREFIX}RF_over65_with_psa'] = (
        (age >= 65) & (pf[f'{PREFIX}has_psa_monitoring'] == 1)
    ).astype(int)
    pf[f'{PREFIX}RF_over65_with_luts'] = (
        (age >= 65) & (pf[f'{PREFIX}has_luts'] == 1)
    ).astype(int)
    pf[f'{PREFIX}RF_psa_plus_bone'] = (
        (pf.get(f'{PREFIX}LAB_psa_elevated_4', pd.Series(0, index=pf.index)) == 1) &
        (pf[f'{PREFIX}has_any_bone_symptom'] == 1)
    ).astype(int)

    # ── Clean up ──────────────────────────────────────────────
    pf = pf.fillna(0)
    pf = pf.replace([np.inf, -np.inf], 0)

    # Remove constant columns
    nunique = pf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        pf = pf.drop(columns=constant)
        logger.info(f"  Removed {len(constant)} constant columns")

    logger.info(f"  Prostate-specific features: {pf.shape[1]} features")

    # Feature breakdown
    for prefix_check, name in [
        (f'{PREFIX}has_', 'Binary flags'),
        (f'{PREFIX}psa_', 'PSA dynamics'),
        (f'{PREFIX}urinary', 'Urinary patterns'),
        (f'{PREFIX}bone', 'Bone/metastatic'),
        (f'{PREFIX}LAB_', 'Lab prognostic'),
        (f'{PREFIX}DX_', 'Diagnostic pathway'),
        (f'{PREFIX}risk_', 'Risk score'),
        (f'{PREFIX}RF_', 'Risk factors'),
        (f'{PREFIX}bph_', 'BPH treatment'),
        (f'{PREFIX}antibiotic', 'Antibiotic'),
        (f'{PREFIX}pain_', 'Pain meds'),
        (f'{PREFIX}corticosteroid', 'Corticosteroid'),
    ]:
        count = len([c for c in pf.columns if c.startswith(prefix_check)])
        if count > 0:
            logger.info(f"    {name:25s}: {count}")

    return pf
