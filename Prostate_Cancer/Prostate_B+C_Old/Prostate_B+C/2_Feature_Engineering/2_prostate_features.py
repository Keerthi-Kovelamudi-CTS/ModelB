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
        # Returns a Series indexed by pf.index (not Index), so .clip()/.subtract() work
        has = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size() > 0
        return has.reindex(pf.index, fill_value=False).astype(int)

    def count_cat(df, cat):
        cnt = df[df['CATEGORY'] == cat].groupby('PATIENT_GUID').size()
        return cnt.reindex(pf.index, fill_value=0).astype(int)

    m = lambda s, default=0: s.reindex(pf.index).fillna(default)

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: PSA DYNAMICS
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 1: PSA dynamics...")

    # PSA: use VALUE features only. Count/has-PSA features are leaky —
    # they reflect GP testing behavior, not patient biology.

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

        pf[f'{PREFIX}LAB_psa_latest'] = m(psa_latest, default=np.nan)
        pf[f'{PREFIX}LAB_psa_first'] = m(psa_first, default=np.nan)
        pf[f'{PREFIX}LAB_psa_max'] = m(psa_max, default=np.nan)

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

        # PSA velocity (ng/mL/month, retained for backward compat) and the
        # clinically-standard per-year version. The previous _velocity_high flag
        # compared a per-month value against a per-year threshold (0.75 ng/mL/yr),
        # so it only fired at ~9 ng/mL/yr — 12x too strict. Fixed below.
        def psa_velocity_per_month(group):
            if len(group) < 2:
                return np.nan
            first_date = group['EVENT_DATE'].iloc[0]
            last_date = group['EVENT_DATE'].iloc[-1]
            months = (last_date - first_date).days / 30.44
            if months < 1:
                return np.nan
            return (group['VALUE'].iloc[-1] - group['VALUE'].iloc[0]) / months
        psa_vel = pd.to_numeric(
            psa_all.groupby('PATIENT_GUID').apply(psa_velocity_per_month),
            errors='coerce',
        )
        psa_vel_yr = psa_vel * 12.0
        pf[f'{PREFIX}LAB_psa_velocity'] = m(psa_vel, default=0)              # ng/mL per month
        pf[f'{PREFIX}LAB_psa_velocity_per_year'] = m(psa_vel_yr, default=0)  # ng/mL per year — clinical standard
        pf[f'{PREFIX}LAB_psa_velocity_high'] = (m(psa_vel_yr) > 0.75).astype(int)   # >0.75 ng/mL/yr suspicious
        pf[f'{PREFIX}LAB_psa_velocity_very_high'] = (m(psa_vel_yr) > 2.0).astype(int)  # >2.0 ng/mL/yr high concern

        # PSA percent change across the window (first vs last reading, guarded)
        def psa_pct_change(group):
            if len(group) < 2:
                return np.nan
            first = float(group['VALUE'].iloc[0])
            last = float(group['VALUE'].iloc[-1])
            if first <= 0.1:
                return np.nan
            return (last - first) / first * 100.0
        psa_pct = pd.to_numeric(
            psa_all.groupby('PATIENT_GUID').apply(psa_pct_change),
            errors='coerce',
        )
        pf[f'{PREFIX}LAB_psa_pct_change'] = m(psa_pct, default=0)
        pf[f'{PREFIX}LAB_psa_pct_up_50'] = (m(psa_pct) > 50).astype(int)
        pf[f'{PREFIX}LAB_psa_pct_up_100'] = (m(psa_pct) > 100).astype(int)

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
        psa_dt = pd.to_numeric(
            psa_all.groupby('PATIENT_GUID').apply(psa_doubling_time),
            errors='coerce',
        )
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

        # Age-banded PSA percentile: rank PSA within each 10-year age band.
        # PSA>4 means different things at 50 vs 75 — within-band percentile captures that.
        age_bands = pd.cut(age, bins=[0, 50, 60, 70, 80, 150],
                           labels=['<50', '50-59', '60-69', '70-79', '80+'],
                           include_lowest=True)
        psa_with_band = pd.DataFrame({'psa': psa_latest, 'band': age_bands})
        band_pct = psa_with_band.groupby('band', observed=True)['psa'].rank(pct=True)
        pf[f'{PREFIX}LAB_psa_age_band_percentile'] = m(band_pct, default=0.5)
        pf[f'{PREFIX}LAB_psa_age_band_top10pct'] = (m(band_pct) >= 0.90).astype(int)
        pf[f'{PREFIX}LAB_psa_age_band_top5pct'] = (m(band_pct) >= 0.95).astype(int)

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
                     'LAB_psa_very_fast_doubling']:
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
    # BLOCK 6b: EXPLICIT CRITICAL-CODE FLAGS
    # Boolean flags for high-signal SNOMED terms that would otherwise be
    # diluted inside category-level counts.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 6b: Critical-code flags...")

    def _term_flag(df, pattern, category=None):
        """1 if any row matches TERM regex (optionally within a CATEGORY), else 0."""
        sub = df
        if category is not None:
            sub = sub[sub['CATEGORY'] == category]
        if sub.empty:
            return pd.Series(0, index=pf.index, dtype=int)
        mask = sub['TERM'].astype(str).str.contains(pattern, case=False, na=False, regex=True)
        has = sub[mask].groupby('PATIENT_GUID').size() > 0
        return pf.index.map(has).fillna(False).astype(int)

    # Critical clinical codes — matched at the TERM level for clarity
    pf[f'{PREFIX}HAS_FRANK_HAEMATURIA'] = _term_flag(obs_AB, r'frank\s+haematuria|macroscopic\s+haematuria|visible\s+haematuria', 'HAEMATURIA')
    pf[f'{PREFIX}HAS_MICROSCOPIC_HAEMATURIA'] = _term_flag(obs_AB, r'microscopic\s+haematuria|non[\s-]*visible\s+haematuria', 'HAEMATURIA')
    pf[f'{PREFIX}HAS_DRE_ABNORMAL'] = _term_flag(obs_AB, r'abnormal|hard|nodul|irregular|suspicious', 'DRE')
    pf[f'{PREFIX}HAS_RAISED_PSA'] = _term_flag(obs_AB, r'rais(e|ed)\s+psa|psa\s+raised|elevated\s+psa|psa\s+elevated', 'PSA')
    pf[f'{PREFIX}HAS_PSA_ABNORMAL'] = _term_flag(obs_AB, r'abnormal.*psa|psa.*abnormal|serum\s+psa.*abnormal', 'PSA')
    pf[f'{PREFIX}HAS_URINARY_RETENTION_CODE'] = _term_flag(obs_AB, r'retention\s+of\s+urine|urinary\s+retention', 'URINARY_RETENTION')
    pf[f'{PREFIX}HAS_BONE_METS_CODE'] = _term_flag(obs_AB, r'bone\s+metasta|metastat.*bone|secondary\s+malignant.*bone')
    pf[f'{PREFIX}HAS_PROSTATE_BIOPSY'] = _term_flag(obs_AB, r'biopsy.*prostat|prostat.*biopsy|trus\s+biopsy')

    # Granular smoking flags (was collapsed into OBS_SMOKING_has_ever)
    pf[f'{PREFIX}HAS_CURRENT_SMOKER'] = _term_flag(obs_AB, r'current\s+smoker|smokes|daily\s+smok', 'SMOKING')
    pf[f'{PREFIX}HAS_FORMER_SMOKER'] = _term_flag(obs_AB, r'ex[\s-]*smoker|former\s+smoker|stopped\s+smoking|smoking\s+cessation', 'SMOKING')
    pf[f'{PREFIX}HAS_NEVER_SMOKER'] = _term_flag(obs_AB, r'never\s+smok|non[\s-]*smoker|lifetime\s+non[\s-]*smoker', 'SMOKING')

    # Granular family-history flags (was collapsed into OBS_FAMILY_HISTORY_has_ever)
    pf[f'{PREFIX}HAS_FAMHX_PROSTATE_CA'] = _term_flag(obs_AB, r'family\s+history.*prostat|prostat.*family\s+history', 'FAMILY_HISTORY')
    pf[f'{PREFIX}HAS_FAMHX_ANY_CANCER'] = _term_flag(obs_AB, r'family\s+history.*(cancer|malignan|neoplasm|carcinoma)', 'FAMILY_HISTORY')
    pf[f'{PREFIX}HAS_FAMHX_MALE_GENITAL_CA'] = _term_flag(obs_AB, r'family\s+history.*(male\s+genital|testic|penile)', 'FAMILY_HISTORY')

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
    pf[f'{PREFIX}RF_over65_with_psa_elevated'] = (
        (age >= 65) &
        (pf.get(f'{PREFIX}LAB_psa_elevated_4', pd.Series(0, index=pf.index)) == 1)
    ).astype(int)
    pf[f'{PREFIX}RF_over65_with_luts'] = (
        (age >= 65) & (pf[f'{PREFIX}has_luts'] == 1)
    ).astype(int)
    pf[f'{PREFIX}RF_psa_plus_bone'] = (
        (pf.get(f'{PREFIX}LAB_psa_elevated_4', pd.Series(0, index=pf.index)) == 1) &
        (pf[f'{PREFIX}has_any_bone_symptom'] == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 9: COMPOSITE / VALUE-BASED CLINICAL FEATURES
    # Non-leaky additions: IPSS severity from VALUE, calcium,
    # bone-metastasis biomarker trio, multi-system symptom co-occurrence,
    # family-history composite. All built from value/category data
    # already present pre-index.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 9: Composite / value-based features...")

    # ── 9a: IPSS severity tiers (clinical symptom score) ─────
    try:
        ipss_rows = clin[clin['CATEGORY'] == 'IPSS'].sort_values('EVENT_DATE')
        ipss_latest = ipss_rows.groupby('PATIENT_GUID')['VALUE'].last()
        ipss_max = ipss_rows.groupby('PATIENT_GUID')['VALUE'].max()
        pf[f'{PREFIX}LAB_ipss_latest'] = m(ipss_latest, default=np.nan)
        pf[f'{PREFIX}LAB_ipss_max'] = m(ipss_max, default=np.nan)
        pf[f'{PREFIX}LAB_ipss_severe'] = (m(ipss_latest) >= 20).astype(int)        # severe LUTS
        pf[f'{PREFIX}LAB_ipss_moderate'] = (
            (m(ipss_latest) >= 8) & (m(ipss_latest) < 20)
        ).astype(int)
        pf[f'{PREFIX}LAB_ipss_mild'] = (
            (m(ipss_latest) >= 1) & (m(ipss_latest) < 8)
        ).astype(int)
    except Exception as e:
        logger.warning(f"  IPSS tier features failed: {e}, defaulting")
        for col in ['LAB_ipss_latest', 'LAB_ipss_max']:
            pf[f'{PREFIX}{col}'] = np.nan
        for col in ['LAB_ipss_severe', 'LAB_ipss_moderate', 'LAB_ipss_mild']:
            pf[f'{PREFIX}{col}'] = 0

    # ── 9b: Serum calcium (bone metastasis correlate) ─────────
    try:
        ca = get_lab_latest('CALCIUM', 'calcium')
        pf[f'{PREFIX}LAB_ca_latest'] = m(ca, default=np.nan)
        pf[f'{PREFIX}LAB_ca_high'] = (m(ca) > 2.6).astype(int)        # mmol/L upper normal
        pf[f'{PREFIX}LAB_ca_very_high'] = (m(ca) > 2.85).astype(int)  # hypercalcaemia
    except Exception as e:
        logger.warning(f"  Calcium features failed: {e}, defaulting")
        pf[f'{PREFIX}LAB_ca_latest'] = np.nan
        pf[f'{PREFIX}LAB_ca_high'] = 0
        pf[f'{PREFIX}LAB_ca_very_high'] = 0

    # ── 9c: Bone-metastasis biomarker trio (ALP↑ + Ca↑ + Hb↓) ─
    alp_hi = pf.get(f'{PREFIX}LAB_alp_elevated', pd.Series(0, index=pf.index)).astype(int)
    ca_hi = pf.get(f'{PREFIX}LAB_ca_high', pd.Series(0, index=pf.index)).astype(int)
    hb_lo = pf.get(f'{PREFIX}LAB_hb_low', pd.Series(0, index=pf.index)).astype(int)
    bone_signals = alp_hi + ca_hi + hb_lo
    pf[f'{PREFIX}BONE_MET_TRIO'] = (bone_signals >= 3).astype(int)
    pf[f'{PREFIX}BONE_MET_PAIR'] = (bone_signals >= 2).astype(int)
    pf[f'{PREFIX}BONE_MET_SCORE'] = bone_signals

    # ── 9d: Multi-system trifecta (constitutional + urinary + bone, window B) ─
    constitutional_B = (count_cat(obs_B, 'CONSTITUTIONAL') > 0).astype(int)
    urinary_B_any = (
        count_cat(obs_B, 'LUTS') + count_cat(obs_B, 'HAEMATURIA') +
        count_cat(obs_B, 'URINARY_RETENTION') > 0
    ).astype(int)
    bone_B_any = (
        count_cat(obs_B, 'PAIN_PELVIC_BONE') + count_cat(obs_B, 'BONE_MUSCLE') > 0
    ).astype(int)
    multisystem_count = constitutional_B + urinary_B_any + bone_B_any
    pf[f'{PREFIX}TRIFECTA_constit_urinary_bone_B'] = (multisystem_count >= 3).astype(int)
    pf[f'{PREFIX}MULTISYSTEM_PAIR_B'] = (multisystem_count >= 2).astype(int)

    # ── 9e: Family-history composite (prostate-relevant cancers) ─
    fh_prostate = pf.get(f'{PREFIX}HAS_FAMHX_PROSTATE_CA', pd.Series(0, index=pf.index)).astype(int)
    fh_male_genital = pf.get(f'{PREFIX}HAS_FAMHX_MALE_GENITAL_CA', pd.Series(0, index=pf.index)).astype(int)
    fh_any_cancer = pf.get(f'{PREFIX}HAS_FAMHX_ANY_CANCER', pd.Series(0, index=pf.index)).astype(int)
    pf[f'{PREFIX}FH_prostate_relevant'] = ((fh_prostate + fh_male_genital) > 0).astype(int)
    pf[f'{PREFIX}FH_score'] = fh_prostate * 2 + fh_male_genital + fh_any_cancer  # weighted

    # ══════════════════════════════════════════════════════════
    # BLOCK 10: AGE × CLINICAL-STATE INTERACTIONS
    # Designed to survive age-band matching (where AGE_AT_INDEX alone
    # becomes uninformative, but age × symptoms/labs/meds still
    # discriminates within bands — "70yo with LUTS" vs "70yo without").
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 10: Age × clinical interactions...")

    def _safe(col):
        return pf.get(col, pd.Series(0, index=pf.index)).astype(float)

    # ── 10a: Age-derived features (non-interaction) ─────────
    pf[f'{PREFIX}AGEX_PEAK_DIST'] = (age - 75).abs()                  # distance from peak prostate ca incidence age
    pf[f'{PREFIX}AGEX_OVER_60'] = (age >= 60).astype(int)
    pf[f'{PREFIX}AGEX_OVER_70'] = (age >= 70).astype(int)
    pf[f'{PREFIX}AGEX_OVER_80'] = (age >= 80).astype(int)
    pf[f'{PREFIX}AGEX_DECILE'] = pd.qcut(age.rank(method='first'), q=10, labels=False, duplicates='drop').fillna(0).astype(int)

    # ── 10b: Age × symptom counts (window B = recent) ────────
    luts_B   = (count_cat(obs_B, 'LUTS')).astype(float)
    haem_B   = (count_cat(obs_B, 'HAEMATURIA')).astype(float)
    pain_B   = (count_cat(obs_B, 'PAIN_PELVIC_BONE')).astype(float)
    bone_B_c = (count_cat(obs_B, 'BONE_MUSCLE')).astype(float)
    cons_B   = (count_cat(obs_B, 'CONSTITUTIONAL')).astype(float)
    ret_B    = (count_cat(obs_B, 'URINARY_RETENTION')).astype(float)
    pf[f'{PREFIX}AGEX_LUTS_count_B']         = age * luts_B
    pf[f'{PREFIX}AGEX_HAEMATURIA_count_B']   = age * haem_B
    pf[f'{PREFIX}AGEX_PAIN_count_B']         = age * pain_B
    pf[f'{PREFIX}AGEX_BONE_count_B']         = age * bone_B_c
    pf[f'{PREFIX}AGEX_CONSTITUTIONAL_B']     = age * cons_B
    pf[f'{PREFIX}AGEX_URINARY_RETENTION_B']  = age * ret_B

    # ── 10c: Age × symptom acceleration (B vs A) ─────────────
    luts_A = (count_cat(obs_A, 'LUTS')).astype(float)
    pain_A = (count_cat(obs_A, 'PAIN_PELVIC_BONE')).astype(float)
    bone_A_c = (count_cat(obs_A, 'BONE_MUSCLE')).astype(float)
    pf[f'{PREFIX}AGEX_LUTS_acceleration'] = age * (luts_B - luts_A).clip(lower=0)
    pf[f'{PREFIX}AGEX_BONE_acceleration'] = age * ((pain_B + bone_B_c) - (pain_A + bone_A_c)).clip(lower=0)

    # ── 10d: Age × labs (real biology — bone mets, anaemia, inflammation) ─
    pf[f'{PREFIX}AGEX_ALP_elevated'] = age * _safe(f'{PREFIX}LAB_alp_elevated')
    pf[f'{PREFIX}AGEX_HB_low']        = age * _safe(f'{PREFIX}LAB_hb_low')
    pf[f'{PREFIX}AGEX_CRP_elevated']  = age * _safe(f'{PREFIX}LAB_crp_elevated')
    pf[f'{PREFIX}AGEX_TESTO_low']     = age * _safe(f'{PREFIX}LAB_testosterone_low')
    pf[f'{PREFIX}AGEX_CA_high']       = age * _safe(f'{PREFIX}LAB_ca_high')

    # ── 10e: Age × PSA (only when PSA codes present) ─────────
    pf[f'{PREFIX}AGEX_RAISED_PSA']    = age * _safe(f'{PREFIX}HAS_RAISED_PSA')
    pf[f'{PREFIX}AGEX_PSA_ABNORMAL']  = age * _safe(f'{PREFIX}HAS_PSA_ABNORMAL')
    # PSA value features (present only when PSA in codelist; safely zero otherwise)
    pf[f'{PREFIX}AGEX_PSA_max']       = age * _safe(f'{PREFIX}LAB_psa_max')
    pf[f'{PREFIX}AGEX_PSA_velocity']  = age * _safe(f'{PREFIX}LAB_psa_velocity_per_year')

    # ── 10f: Age × medication patterns ───────────────────────
    pf[f'{PREFIX}AGEX_ALPHA_BLOCKERS'] = age * _safe(f'{PREFIX}has_alpha_blockers')
    pf[f'{PREFIX}AGEX_5ARI']            = age * _safe(f'{PREFIX}has_5ari')
    pf[f'{PREFIX}AGEX_PAIN_MED_B']      = age * _safe(f'{PREFIX}pain_med_count_B')
    pf[f'{PREFIX}AGEX_UTI_ABX_B']       = age * _safe(f'{PREFIX}uti_abx_count_B')
    pf[f'{PREFIX}AGEX_ANTICHOLINERGICS'] = age * _safe(f'{PREFIX}has_anticholinergics')

    # ── 10g: Age × diagnostic-pathway flags ──────────────────
    pf[f'{PREFIX}AGEX_UROLOGY_PATHWAY']  = age * _safe(f'{PREFIX}has_urology_pathway')
    pf[f'{PREFIX}AGEX_DRE_ABNORMAL']     = age * _safe(f'{PREFIX}HAS_DRE_ABNORMAL')
    pf[f'{PREFIX}AGEX_PROSTATE_BIOPSY']  = age * _safe(f'{PREFIX}HAS_PROSTATE_BIOPSY')
    pf[f'{PREFIX}AGEX_BONE_METS_CODE']   = age * _safe(f'{PREFIX}HAS_BONE_METS_CODE')

    # ── 10h: Composite "elderly + clinical signal" flags ──────
    elderly = (age >= 70).astype(int)
    pf[f'{PREFIX}ELDERLY_WITH_LUTS']         = (elderly & (luts_B > 0).astype(int))
    pf[f'{PREFIX}ELDERLY_WITH_RAISED_PSA']   = (elderly & _safe(f'{PREFIX}HAS_RAISED_PSA').astype(int))
    pf[f'{PREFIX}ELDERLY_WITH_BONE_SYMPTOM'] = (elderly & _safe(f'{PREFIX}has_any_bone_symptom').astype(int))
    pf[f'{PREFIX}ELDERLY_WITH_ALP_HIGH']     = (elderly & _safe(f'{PREFIX}LAB_alp_elevated').astype(int))
    pf[f'{PREFIX}ELDERLY_WITH_HB_LOW']       = (elderly & _safe(f'{PREFIX}LAB_hb_low').astype(int))
    pf[f'{PREFIX}ELDERLY_WITH_HAEMATURIA']   = (elderly & (haem_B > 0).astype(int))
    pf[f'{PREFIX}ELDERLY_WITH_RETENTION']    = (elderly & (ret_B > 0).astype(int))
    pf[f'{PREFIX}ELDERLY_WITH_BONE_METS']    = (elderly & _safe(f'{PREFIX}HAS_BONE_METS_CODE').astype(int))
    pf[f'{PREFIX}ELDERLY_BONE_TRIO']         = (elderly & _safe(f'{PREFIX}BONE_MET_TRIO').astype(int))

    # ── 10i: Age vs investigation-pathway gap ────────────────
    # If older patient took a long time from symptom to imaging → red flag
    sti = pf.get(f'{PREFIX}DX_symptom_to_imaging_days', pd.Series(-1, index=pf.index)).astype(float)
    valid_gap = sti >= 0
    pf[f'{PREFIX}AGEX_SYMPTOM_TO_IMAGING'] = np.where(valid_gap, age * sti, 0)

    # ══════════════════════════════════════════════════════════
    # BLOCK 11: TRIPLE INTERACTIONS, VELOCITY, COMORBIDITY, LAB RATIOS
    # High-value composite features that should survive age-band matching:
    #   - Triple interactions (age × symptoms × PSA/ALP)
    #   - Velocity: B-vs-A acceleration of symptoms/visits
    #   - Comorbidity burden: count of distinct categories active
    #   - Lab ratios: PSA × ALP, PSA × Hb_low (metastasis biomarker pairs)
    #   - Pre-anchor "approach to dx" pattern (recent escalation)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 11: Triple interactions, velocity, comorbidity, lab ratios...")

    has_raised_psa = _safe(f'{PREFIX}HAS_RAISED_PSA')
    psa_max        = _safe(f'{PREFIX}LAB_psa_max')
    alp_elevated   = _safe(f'{PREFIX}LAB_alp_elevated')
    hb_low         = _safe(f'{PREFIX}LAB_hb_low')
    ca_high        = _safe(f'{PREFIX}LAB_ca_high')
    crp_elevated   = _safe(f'{PREFIX}LAB_crp_elevated')
    bone_count_B   = bone_B_c + pain_B   # from BLOCK 10
    bone_count_A   = bone_A_c + pain_A
    famhx_score    = _safe(f'{PREFIX}FH_score')

    # ── 11a: Triple interactions (age × symptom × PSA/ALP) ────
    # "Elderly + LUTS + raised PSA" = high-risk cancer profile
    pf[f'{PREFIX}TRIPLE_age_lutsB_raisedPSA']  = age * luts_B * has_raised_psa
    pf[f'{PREFIX}TRIPLE_age_haemB_raisedPSA']  = age * haem_B * has_raised_psa
    pf[f'{PREFIX}TRIPLE_age_boneB_raisedPSA']  = age * bone_count_B * has_raised_psa
    pf[f'{PREFIX}TRIPLE_age_lutsB_psamax']     = age * luts_B * psa_max
    pf[f'{PREFIX}TRIPLE_age_boneB_alp']        = age * bone_count_B * alp_elevated

    # ── 11b: Lab biomarker pairs (real biology, no decision-proxy) ──
    pf[f'{PREFIX}LABPAIR_psa_x_alp']        = psa_max * alp_elevated
    pf[f'{PREFIX}LABPAIR_psa_x_hb_low']     = psa_max * hb_low
    pf[f'{PREFIX}LABPAIR_psa_x_ca_high']    = psa_max * ca_high
    pf[f'{PREFIX}LABPAIR_psa_x_crp']        = psa_max * crp_elevated
    pf[f'{PREFIX}LABPAIR_alp_hb_low']       = alp_elevated * hb_low
    pf[f'{PREFIX}LABPAIR_psa_alp_hb']       = psa_max * alp_elevated * hb_low  # bone-met aggressive trio
    pf[f'{PREFIX}LAB_psa_per_age']          = psa_max / age.clip(lower=1)

    # ── 11c: Symptom velocity (recent acceleration) ───────────
    # B - A clipped to non-negative (only count INCREASES, not decreases)
    pf[f'{PREFIX}VEL_luts']         = (luts_B - luts_A).clip(lower=0)
    pf[f'{PREFIX}VEL_haem']         = (haem_B - (count_cat(obs_A, 'HAEMATURIA').astype(float))).clip(lower=0)
    pf[f'{PREFIX}VEL_bone']         = (bone_count_B - bone_count_A).clip(lower=0)
    pf[f'{PREFIX}VEL_constitutional'] = (cons_B - (count_cat(obs_A, 'CONSTITUTIONAL').astype(float))).clip(lower=0)
    pf[f'{PREFIX}VEL_retention']    = (ret_B - (count_cat(obs_A, 'URINARY_RETENTION').astype(float))).clip(lower=0)

    # ── 11d: New symptom appearance in window B (vs absent in A) ──
    pf[f'{PREFIX}NEW_luts_B_only']  = ((luts_B > 0) & (luts_A == 0)).astype(int)
    pf[f'{PREFIX}NEW_haem_B_only']  = ((haem_B > 0) & (count_cat(obs_A, 'HAEMATURIA') == 0)).astype(int)
    pf[f'{PREFIX}NEW_bone_B_only']  = ((bone_count_B > 0) & (bone_count_A == 0)).astype(int)
    pf[f'{PREFIX}NEW_retention_B_only'] = ((ret_B > 0) & (count_cat(obs_A, 'URINARY_RETENTION') == 0)).astype(int)
    pf[f'{PREFIX}AGEX_NEW_luts_B']  = age * pf[f'{PREFIX}NEW_luts_B_only']
    pf[f'{PREFIX}AGEX_NEW_bone_B']  = age * pf[f'{PREFIX}NEW_bone_B_only']

    # ── 11e: Comorbidity burden (distinct categories with events) ──
    cat_counts = obs_AB.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    n_categories = cat_counts.reindex(pf.index, fill_value=0).astype(int)
    pf[f'{PREFIX}COMORBIDITY_n_categories']     = n_categories
    pf[f'{PREFIX}COMORBIDITY_x_age']            = age * n_categories
    pf[f'{PREFIX}COMORBIDITY_high']             = (n_categories >= 5).astype(int)
    pf[f'{PREFIX}COMORBIDITY_x_raisedPSA']      = n_categories * has_raised_psa

    # ── 11f: Family-history × age (real risk amplifier) ───────
    pf[f'{PREFIX}AGEX_FH_score']    = age * famhx_score
    pf[f'{PREFIX}FH_x_raisedPSA']   = famhx_score * has_raised_psa
    pf[f'{PREFIX}FH_x_LUTS_B']      = famhx_score * luts_B

    # ── 11g: Recent activity escalation (events in window B vs A) ──
    n_events_B = obs_B.groupby('PATIENT_GUID').size()
    n_events_A = obs_A.groupby('PATIENT_GUID').size()
    events_B = n_events_B.reindex(pf.index, fill_value=0).astype(float)
    events_A = n_events_A.reindex(pf.index, fill_value=0).astype(float)
    # Normalize by window length (window B is ~30mo, A is ~30mo too in 60mo lookback)
    pf[f'{PREFIX}ACTIVITY_events_B']        = events_B
    pf[f'{PREFIX}ACTIVITY_events_A']        = events_A
    pf[f'{PREFIX}ACTIVITY_velocity']        = (events_B - events_A).clip(lower=0)
    pf[f'{PREFIX}ACTIVITY_acceleration']   = (events_B / events_A.replace(0, 1)).clip(upper=10)  # max 10x to avoid inf
    pf[f'{PREFIX}AGEX_ACTIVITY_velocity']   = age * pf[f'{PREFIX}ACTIVITY_velocity']

    # ── 11h: Investigation × symptom (clinically meaningful pairs) ──
    has_imaging  = _safe(f'{PREFIX}has_urology_imaging')
    has_pathway  = _safe(f'{PREFIX}has_urology_pathway')
    pf[f'{PREFIX}IMG_x_lutsB']            = has_imaging * luts_B
    pf[f'{PREFIX}IMG_x_raisedPSA']        = has_imaging * has_raised_psa
    pf[f'{PREFIX}PATHWAY_x_age']          = has_pathway * age

    # ══════════════════════════════════════════════════════════
    # BLOCK 12: PATIENT Z-SCORE LABS
    # "Is this patient's PSA high *for them*?" rather than absolute.
    # PSA naturally varies 2-10× by patient baseline → personal z-score
    # captures real change much better than population thresholds.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 12: Patient z-score labs...")

    def patient_zscore(category, term_pattern=None, label='lab'):
        """Compute patient-personal z-score: (latest - patient_mean) / patient_std.
        Returns 0 for patients with <2 readings (no personal baseline)."""
        sub = clin[clin['CATEGORY'] == category]
        if term_pattern:
            sub = sub[sub['TERM'].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)]
        sub = sub.dropna(subset=['VALUE']).copy()
        sub['VALUE'] = pd.to_numeric(sub['VALUE'], errors='coerce')
        sub = sub.dropna(subset=['VALUE'])
        if sub.empty:
            return pd.Series(0.0, index=pf.index), pd.Series(0, index=pf.index)
        sub = sub.sort_values('EVENT_DATE')
        grp = sub.groupby('PATIENT_GUID')['VALUE']
        means = grp.mean()
        stds = grp.std().replace(0, np.nan)  # std=0 means single reading
        latest = grp.last()
        zscore = ((latest - means) / stds).fillna(0).clip(-10, 10)
        n_reads = grp.count()
        return (zscore.reindex(pf.index, fill_value=0),
                n_reads.reindex(pf.index, fill_value=0).astype(int))

    z_psa, n_psa     = patient_zscore('PSA')
    z_alp, n_alp     = patient_zscore('ALP_BONE_MARKER', 'alkaline phosphatase')
    z_hb, n_hb       = patient_zscore('FBC_HAEMATOLOGY', 'haemoglobin|hemoglobin')
    z_creat, n_creat = patient_zscore('RENAL_FUNCTION', 'creatinine')
    z_ca, n_ca       = patient_zscore('CALCIUM', 'calcium')

    pf[f'{PREFIX}ZSCORE_psa']         = z_psa
    pf[f'{PREFIX}ZSCORE_psa_high']    = (z_psa > 1.5).astype(int)
    pf[f'{PREFIX}ZSCORE_psa_extreme'] = (z_psa > 3.0).astype(int)
    pf[f'{PREFIX}ZSCORE_alp']         = z_alp
    pf[f'{PREFIX}ZSCORE_hb']          = z_hb
    pf[f'{PREFIX}ZSCORE_hb_low']      = (z_hb < -1.5).astype(int)
    pf[f'{PREFIX}ZSCORE_creat']       = z_creat
    pf[f'{PREFIX}ZSCORE_ca']          = z_ca
    pf[f'{PREFIX}ZSCORE_n_psa_reads'] = n_psa
    pf[f'{PREFIX}ZSCORE_n_alp_reads'] = n_alp

    # ══════════════════════════════════════════════════════════
    # BLOCK 13: FIRST-EVENT-DATE PER CATEGORY
    # months_since_first_<X> captures "long-standing" vs "new" symptom
    # patterns. Cancer often shows recent-onset symptoms.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 13: First-event-date features...")

    def months_since_first(category):
        sub = clin[clin['CATEGORY'] == category]
        if sub.empty:
            return pd.Series(-1.0, index=pf.index)
        first = sub.groupby('PATIENT_GUID')['EVENT_DATE'].min()
        anchor = clin.drop_duplicates('PATIENT_GUID').set_index('PATIENT_GUID')['INDEX_DATE']
        anchor = pd.to_datetime(anchor, errors='coerce')
        first = pd.to_datetime(first, errors='coerce')
        gap = (anchor.reindex(first.index) - first).dt.days / 30.44
        return pf.index.map(gap).fillna(-1).astype(float)

    pf[f'{PREFIX}FIRST_luts_months']         = months_since_first('LUTS')
    pf[f'{PREFIX}FIRST_haem_months']         = months_since_first('HAEMATURIA')
    pf[f'{PREFIX}FIRST_pain_months']         = months_since_first('PAIN_PELVIC_BONE')
    pf[f'{PREFIX}FIRST_bone_months']         = months_since_first('BONE_MUSCLE')
    pf[f'{PREFIX}FIRST_constitutional_months'] = months_since_first('CONSTITUTIONAL')
    pf[f'{PREFIX}FIRST_retention_months']    = months_since_first('URINARY_RETENTION')
    pf[f'{PREFIX}FIRST_psa_months']          = months_since_first('PSA')
    pf[f'{PREFIX}FIRST_imaging_months']      = months_since_first('UROLOGY_IMAGING')
    pf[f'{PREFIX}FIRST_dre_months']          = months_since_first('DRE')

    # "Recent-onset" flags (<12 months since first event)
    pf[f'{PREFIX}RECENT_luts']  = ((pf[f'{PREFIX}FIRST_luts_months'] >= 0) & (pf[f'{PREFIX}FIRST_luts_months'] < 12)).astype(int)
    pf[f'{PREFIX}RECENT_haem']  = ((pf[f'{PREFIX}FIRST_haem_months'] >= 0) & (pf[f'{PREFIX}FIRST_haem_months'] < 12)).astype(int)
    pf[f'{PREFIX}RECENT_bone']  = ((pf[f'{PREFIX}FIRST_bone_months'] >= 0) & (pf[f'{PREFIX}FIRST_bone_months'] < 12)).astype(int)

    # ══════════════════════════════════════════════════════════
    # BLOCK 14: SEQUENTIAL PATTERNS (X then Y within Z days)
    # Cancer often follows specific symptom orderings:
    # LUTS → HAEMATURIA, BONE_PAIN → CONSTITUTIONAL, etc.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 14: Sequential symptom patterns...")

    def seq_pattern(cat_x, cat_y, max_gap_days):
        """1 if patient had cat_x event followed by cat_y event within max_gap_days."""
        x_first = clin[clin['CATEGORY'] == cat_x].groupby('PATIENT_GUID')['EVENT_DATE'].min()
        y_first = clin[clin['CATEGORY'] == cat_y].groupby('PATIENT_GUID')['EVENT_DATE'].min()
        joined = pd.concat([x_first.rename('x'), y_first.rename('y')], axis=1).dropna()
        if joined.empty:
            return pd.Series(0, index=pf.index, dtype=int)
        joined['gap'] = (joined['y'] - joined['x']).dt.days
        valid = (joined['gap'] >= 0) & (joined['gap'] <= max_gap_days)
        flag = valid.astype(int)
        return pf.index.map(flag).fillna(0).astype(int)

    pf[f'{PREFIX}SEQ_luts_then_haem_90d']      = seq_pattern('LUTS', 'HAEMATURIA', 90)
    pf[f'{PREFIX}SEQ_luts_then_imaging_180d']  = seq_pattern('LUTS', 'UROLOGY_IMAGING', 180)
    pf[f'{PREFIX}SEQ_luts_then_retention_180d'] = seq_pattern('LUTS', 'URINARY_RETENTION', 180)
    pf[f'{PREFIX}SEQ_haem_then_imaging_60d']   = seq_pattern('HAEMATURIA', 'UROLOGY_IMAGING', 60)
    pf[f'{PREFIX}SEQ_pain_then_constitutional_90d'] = seq_pattern('PAIN_PELVIC_BONE', 'CONSTITUTIONAL', 90)
    pf[f'{PREFIX}SEQ_bone_then_constitutional_90d'] = seq_pattern('BONE_MUSCLE', 'CONSTITUTIONAL', 90)
    pf[f'{PREFIX}SEQ_psa_then_imaging_60d']    = seq_pattern('PSA', 'UROLOGY_IMAGING', 60)
    pf[f'{PREFIX}SEQ_psa_then_dre_60d']        = seq_pattern('PSA', 'DRE', 60)

    # ══════════════════════════════════════════════════════════
    # BLOCK 15: WEIGHTED COMORBIDITY SCORE
    # Charlson-inspired: weight categories by clinical importance.
    # Captures multimorbidity beyond simple count from BLOCK 11.
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 15: Weighted comorbidity score...")

    # Weights based on prostate-cancer relevance + Charlson tradition
    CHARLSON_LITE = {
        'BONE_MUSCLE': 1.0,
        'PAIN_PELVIC_BONE': 1.0,
        'CONSTITUTIONAL': 2.0,         # systemic illness (weight loss/cachexia)
        'URINARY_RETENTION': 2.0,      # serious urinary
        'HAEMATURIA': 2.0,             # red flag
        'LUTS': 1.0,
        'ERECTILE_DYSFUNCTION': 0.5,
        'CATHETER': 1.5,
        'PROSTATIC_CONDITIONS': 1.0,
        'FAMILY_HISTORY': 1.5,
        'CYSTOSCOPY': 2.0,             # invasive workup
        'UROLOGY_IMAGING': 1.5,
        'UROLOGY_PATHWAY': 1.5,
        'DRE': 1.0,
        'IPSS': 0.5,
    }
    score = pd.Series(0.0, index=pf.index)
    for cat, w in CHARLSON_LITE.items():
        has_it = (count_cat(obs_AB, cat) > 0).astype(int)
        score = score + w * has_it
    pf[f'{PREFIX}COMORB_weighted_score']       = score
    pf[f'{PREFIX}COMORB_weighted_x_age']       = age * score
    pf[f'{PREFIX}COMORB_weighted_high']        = (score >= 5).astype(int)
    pf[f'{PREFIX}COMORB_weighted_x_raisedPSA'] = score * has_raised_psa
    pf[f'{PREFIX}COMORB_weighted_x_psamax']    = score * psa_max

    # ══════════════════════════════════════════════════════════
    # BLOCK 16: PSA JUMP + LAB CO-ELEVATION + SYMPTOM BURST
    # Three additional clinically validated patterns:
    #  - Single-reading PSA jump (sudden rise = concerning)
    #  - PSA + ALP co-elevation (bone metastasis pattern)
    #  - Multi-symptom burst (3+ different symptoms in 30-day window)
    # ══════════════════════════════════════════════════════════
    logger.info(f"  BLOCK 16: PSA jump + lab co-elevation + symptom burst...")

    # ── 16a: PSA single-reading jump detection ──
    psa_lab_all = clin[(clin['CATEGORY'] == 'PSA') &
                        pd.to_numeric(clin['VALUE'], errors='coerce').notna()].copy()
    psa_lab_all['VALUE'] = pd.to_numeric(psa_lab_all['VALUE'], errors='coerce')

    def _psa_max_jump(group):
        """Returns named Series with max_abs and max_rel jumps between consecutive readings.
        Named keys ensure groupby().apply() yields a DataFrame (not MultiIndex Series)
        regardless of pandas version."""
        if len(group) < 2:
            return pd.Series({'max_abs': 0.0, 'max_rel': 0.0})
        g = group.sort_values('EVENT_DATE')
        diffs = g['VALUE'].diff().dropna()
        if len(diffs) == 0:
            return pd.Series({'max_abs': 0.0, 'max_rel': 0.0})
        max_abs = float(diffs.max())
        prev = g['VALUE'].shift(1).iloc[1:]
        rel = (diffs / prev.clip(lower=0.1)).replace([np.inf, -np.inf], 0).fillna(0)
        max_rel = float(rel.max()) if len(rel) > 0 else 0.0
        return pd.Series({'max_abs': max_abs, 'max_rel': max_rel})

    if len(psa_lab_all) > 0:
        jumps = psa_lab_all.groupby('PATIENT_GUID').apply(_psa_max_jump)
        pf[f'{PREFIX}PSA_max_single_jump']     = jumps['max_abs'].reindex(pf.index, fill_value=0).astype(float)
        pf[f'{PREFIX}PSA_max_relative_jump']   = jumps['max_rel'].reindex(pf.index, fill_value=0).astype(float).clip(0, 50)
        pf[f'{PREFIX}PSA_jump_over_2']         = (pf[f'{PREFIX}PSA_max_single_jump'] > 2.0).astype(int)
        pf[f'{PREFIX}PSA_jump_over_5']         = (pf[f'{PREFIX}PSA_max_single_jump'] > 5.0).astype(int)
        pf[f'{PREFIX}PSA_relative_jump_50pct'] = (pf[f'{PREFIX}PSA_max_relative_jump'] > 0.5).astype(int)

    # ── 16b: PSA + ALP co-elevation (bone metastasis pattern) ──
    # ALP > 130 U/L is upper-normal-range; combined with raised PSA suggests bone mets.
    alp_lab = clin[(clin['CATEGORY'] == 'ALP_BONE_MARKER') &
                    pd.to_numeric(clin['VALUE'], errors='coerce').notna()].copy()
    if len(alp_lab) > 0:
        alp_lab['VALUE'] = pd.to_numeric(alp_lab['VALUE'], errors='coerce')
        alp_max = alp_lab.groupby('PATIENT_GUID')['VALUE'].max()
        pf[f'{PREFIX}LAB_alp_max'] = alp_max.reindex(pf.index, fill_value=0).astype(float)
        pf[f'{PREFIX}LAB_alp_high'] = (pf[f'{PREFIX}LAB_alp_max'] > 130.0).astype(int)
        pf[f'{PREFIX}LABPAIR_psa_alp_both_high'] = (
            (psa_max > 4.0).astype(int) * (pf[f'{PREFIX}LAB_alp_max'] > 130.0).astype(int)
        )
        pf[f'{PREFIX}LABPAIR_psa_alp_both_very_high'] = (
            (psa_max > 10.0).astype(int) * (pf[f'{PREFIX}LAB_alp_max'] > 200.0).astype(int)
        )

    # PSA + Hb (anemia of malignancy: PSA rising + Hb dropping)
    hb_lab = clin[(clin['CATEGORY'] == 'FBC_HAEMATOLOGY') &
                   pd.to_numeric(clin['VALUE'], errors='coerce').notna() &
                   clin['TERM'].astype(str).str.contains('Haemoglobin', case=False, na=False)].copy()
    if len(hb_lab) > 0:
        hb_lab['VALUE'] = pd.to_numeric(hb_lab['VALUE'], errors='coerce')
        hb_min = hb_lab.groupby('PATIENT_GUID')['VALUE'].min()
        pf[f'{PREFIX}LAB_hb_min'] = hb_min.reindex(pf.index, fill_value=0).astype(float)
        # Hb < 130 g/L is mild anaemia in males
        pf[f'{PREFIX}LABPAIR_psa_high_anaemia'] = (
            (psa_max > 4.0).astype(int) * ((pf[f'{PREFIX}LAB_hb_min'].between(1, 130)).astype(int))
        )

    # ── 16c: Symptom burst detection (multiple new symptoms in same 30-day window) ──
    obs_dated = clin[clin['CATEGORY'].isin(cfg.SYMPTOM_CATEGORIES) &
                      clin['EVENT_DATE'].notna()].copy()
    obs_dated['EVENT_DATE'] = pd.to_datetime(obs_dated['EVENT_DATE'], errors='coerce')

    def _max_symptoms_in_30d(group):
        """Returns max number of distinct symptom CATEGORIES seen in any 30-day rolling window."""
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
        pf[f'{PREFIX}SYMPTOM_max_in_30d'] = burst.reindex(pf.index, fill_value=0).astype(int)
        pf[f'{PREFIX}SYMPTOM_burst_3plus'] = (pf[f'{PREFIX}SYMPTOM_max_in_30d'] >= 3).astype(int)
        pf[f'{PREFIX}SYMPTOM_burst_4plus'] = (pf[f'{PREFIX}SYMPTOM_max_in_30d'] >= 4).astype(int)

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
        (f'{PREFIX}AGEX_', 'Age × clinical interactions'),
        (f'{PREFIX}ELDERLY_', 'Elderly + clinical composites'),
        (f'{PREFIX}TRIPLE_', 'Triple interactions (age × symptom × biomarker)'),
        (f'{PREFIX}LABPAIR_', 'Lab biomarker pairs'),
        (f'{PREFIX}VEL_', 'Symptom velocity (B-A acceleration)'),
        (f'{PREFIX}NEW_', 'New symptoms in window B only'),
        (f'{PREFIX}COMORBIDITY_', 'Comorbidity burden'),
        (f'{PREFIX}FH_', 'Family-history composites'),
        (f'{PREFIX}ACTIVITY_', 'Recent event activity'),
        (f'{PREFIX}IMG_', 'Investigation × symptom'),
        (f'{PREFIX}PATHWAY_', 'Pathway × age'),
        (f'{PREFIX}ZSCORE_', 'Patient personal z-scores (lab)'),
        (f'{PREFIX}FIRST_', 'Months-since-first per category'),
        (f'{PREFIX}RECENT_', 'Recent-onset (<12mo) flags'),
        (f'{PREFIX}SEQ_', 'Sequential X→Y patterns'),
        (f'{PREFIX}COMORB_', 'Weighted comorbidity score'),
        (f'{PREFIX}PSA_jump', 'PSA single-reading jump'),
        (f'{PREFIX}LABPAIR_psa_alp', 'PSA + ALP co-elevation (bone mets)'),
        (f'{PREFIX}LABPAIR_psa_high_anaemia', 'PSA + low Hb (anemia of malignancy)'),
        (f'{PREFIX}SYMPTOM_burst', 'Multi-symptom burst (30-day)'),
        (f'{PREFIX}SYMPTOM_max', 'Max symptoms in 30-day window'),
    ]:
        count = len([c for c in pf.columns if c.startswith(prefix_check)])
        if count > 0:
            logger.info(f"    {name:25s}: {count}")

    return pf
