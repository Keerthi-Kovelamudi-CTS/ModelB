"""
Enhanced per-patient FE for the Lung B+C model.

Adds transform-features beyond the base FE, proven useful in our other
B+C models. All occurrence-based transforms use ``days_before_anchor`` (the fixed,
leakage-safe prediction anchor) and are applied ONLY to genuine presenting-symptom
categories — never to investigations (spirometry/FEV/imaging) or diagnostic-pathway
codes (2WW / fast-track referrals, "suspected lung cancer"), which would leak.

Features (per symptom category):
  *_RECENCY_MONTHS    months from anchor to the MOST RECENT occurrence (lower = more recent)
  *_DECAY_INTENSITY   sum of exp(-months_before/tau) over occurrences (recency-weighted load)
  *_ACCEL             2nd-difference of occurrence counts across time bins (ramping-up signal)
  *_RECENT_RATIO      occurrences in the most-recent allowed window / lifetime occurrences
  *_PRESENT           1 if the category ever occurs

Aggregate / dose / interaction:
  SYMPTOM_BURDEN      count of distinct presenting-symptom categories present
  PACK_YEARS_MAX      max recorded cigarette pack-years (direct codes)
  CIGS_PER_DAY_MAX    max recorded cigarettes/day
  INT_SMOKING_x_AGE, INT_SMOKING_x_COPD, INT_AGE_x_HAEMOPTYSIS

The two entry points are merged into the feature matrix on ``patient_guid`` by the
orchestrator (gated by env FE_ENHANCED, default on).
"""
import os
import numpy as np
import pandas as pd

# Leakage-safe presenting-symptom categories (codes copied from the orchestrator;
# HAEMOPTYSIS is isolated from redflags — the rest of redflags is referral/workup leakage).
SYMPTOM_CODES = {
    'COUGH':          [49727002, 161929000, 11833005, 284523002, 161947006, 161924005],
    'BREATHLESSNESS': [267036007, 391120009, 391123006, 391124000, 391125004],
    'CHESTPAIN':      [29857009, 102589003, 2237002],
    'CHESTINFECTION': [312342009, 32398004, 396285007, 54150009, 54398005,
                       195647007, 50417007, 195742007],
    'HAEMOPTYSIS':    [66857006],
    'CLUBBING':       [164457001],   # O/E finger clubbing — high-specificity lung-ca sign (own feature, not just in redflags)
}

PACK_YEAR_CODES = [315609007, 401201003]            # "Pack years", "Cigarette pack-years" (value = pack-years)
CIGS_PER_DAY_CODES = [230056004, 65568007]          # "Cigarette consumption", "Cigarette smoker" (value = cigs/day)

# Chronic comorbidity diagnoses (land on the problem list far more than transient symptoms).
# Used for the clinician-curated problem-list flags below.
COMORBIDITY_CODES = {
    'COPD':      [723245007, 313297008, 313299006, 13645005, 204991000000107],
    'EMPHYSEMA': [87433001, 909721000006104, 68328006, 263747008],
    'FIBROSIS':  [700250006, 909731000006101, 51615001],
    'ASTHMA':    [195967001],
}
# CareRecord_Problem value strings (profiled on 104.1M rows: clean, low-cardinality).
ACTIVE_STATUS_VALUE = 'active problem'
SIGNIFICANT_VALUE = 'significant problem'

# Patient-relative time bands (V2-parity). Windows count back from EACH PATIENT'S most-recent
# event (gap-agnostic, inference-safe — we don't know when a patient presents, so anchor on their
# last record and go back). Same bands for every horizon. Generalizes V1's single _LAST_NM window
# to multiple disjoint bands. Gated by env FE_BANDS (default '0' = OFF, preserves the production
# matrix); set FE_BANDS=1 to add them. FE_BAND_ANCHOR=cohort_anchor uses months-before-anchor instead.
BANDS_RELATIVE = [(0, 6), (6, 18), (18, 36), (36, 72), (72, 999)]   # final = open-ended catch-all
CUMULATIVE_WINDOWS = [6, 12, 24, 60]   # cumulative "last-N months" windows (from start of data)

DECAY_TAU_MONTHS = 12.0
# Time bins measured from the DATA CUTOFF (= start of available data), so band 0 is the freshest
# events for EVERY horizon. (Previously hardcoded 12-18/18-30/30-54 = the 12mo gap, which left the
# 1mo horizon's fresh 1-12mo events in no bin -> accel ignored them. Fixed: the dynamics functions
# re-reference _mb to its own min before binning.)
BIN_RECENT = (0.0, 6.0)
BIN_MID = (6.0, 18.0)
BIN_OLD = (18.0, 42.0)
RECENT_RATIO_CUTOFF = 24.0   # "recent" = within the most-recent 24mo of available data


def _months_before(df, ref_col='days_before_anchor'):
    """Months-before-anchor per row. Uses days_before_anchor when present; else falls
    back to (patient's last event - event) so the module still works on older extracts."""
    dba = pd.to_numeric(df.get(ref_col), errors='coerce')
    if dba is None or dba.isna().all():
        maxd = df.groupby('patient_guid_CLEAN')['event_date_parsed'].transform('max')
        dba = (maxd - df['event_date_parsed']).dt.days
    return dba / 30.44


def compute_symptom_dynamics(df, symptom_codes=None, ref_col='days_before_anchor',
                             decay_tau_months=DECAY_TAU_MONTHS):
    """Per-patient recency / decay-intensity / acceleration / recent-ratio / burden."""
    symptom_codes = symptom_codes or SYMPTOM_CODES
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_mb'] = _months_before(work, ref_col)
    work = work[work['_mb'].notna()]
    work['_mb'] = work['_mb'] - work['_mb'].min()   # re-reference to data cutoff -> bins start at 0

    patients = df['patient_guid_CLEAN'].dropna().unique()
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))
    present_cols = []

    for cat, codes in symptom_codes.items():
        sub = work[work[code_col].isin(codes)]
        rec_col = f'{cat}_RECENCY_MONTHS'
        dec_col = f'{cat}_DECAY_INTENSITY'
        acc_col = f'{cat}_ACCEL'
        rat_col = f'{cat}_RECENT_RATIO'
        prs_col = f'{cat}_PRESENT'
        present_cols.append(prs_col)
        if sub.empty:
            out[rec_col] = np.nan
            out[dec_col] = 0.0
            out[acc_col] = 0.0
            out[rat_col] = 0.0
            out[prs_col] = 0
            continue
        g = sub.groupby('patient_guid_CLEAN')
        total = g.size()
        recency = g['_mb'].min()
        decay = (sub.assign(_w=np.exp(-sub['_mb'] / decay_tau_months))
                 .groupby('patient_guid_CLEAN')['_w'].sum())
        def _binct(lo, hi):
            return sub[(sub['_mb'] > lo) & (sub['_mb'] <= hi)].groupby('patient_guid_CLEAN').size()
        c_recent = _binct(BIN_RECENT[0] - 1e-9, BIN_RECENT[1]).reindex(total.index, fill_value=0)
        c_mid = _binct(*BIN_MID).reindex(total.index, fill_value=0)
        c_old = _binct(*BIN_OLD).reindex(total.index, fill_value=0)
        accel = c_recent - 2 * c_mid + c_old
        recent = sub[sub['_mb'] <= RECENT_RATIO_CUTOFF].groupby('patient_guid_CLEAN').size()
        recent_ratio = recent.reindex(total.index, fill_value=0) / total

        out[rec_col] = recency
        out[dec_col] = decay
        out[acc_col] = accel
        out[rat_col] = recent_ratio
        out[prs_col] = (total > 0).astype(int)

    # fill occurrence-based cols (missing patient = 0); recency stays NaN (no occurrence)
    for c in out.columns:
        if c.endswith('_RECENCY_MONTHS'):
            continue
        out[c] = out[c].fillna(0)
    out['SYMPTOM_BURDEN'] = out[present_cols].sum(axis=1)
    return out.reset_index()


def compute_smoking_dose(df):
    """Per-patient max recorded pack-years and cigarettes/day (leakage-safe risk dose)."""
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_v'] = pd.to_numeric(work.get('value'), errors='coerce')
    patients = df['patient_guid_CLEAN'].dropna().unique()
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))
    py = (work[work[code_col].isin(PACK_YEAR_CODES) & work['_v'].notna()]
          .groupby('patient_guid_CLEAN')['_v'].max())
    cpd = (work[work[code_col].isin(CIGS_PER_DAY_CODES) & work['_v'].notna()]
           .groupby('patient_guid_CLEAN')['_v'].max())
    out['PACK_YEARS_MAX'] = py
    out['CIGS_PER_DAY_MAX'] = cpd
    return out.reset_index()


def compute_problem_flags(df, codes_map=None):
    """Per-patient clinician-curated problem-list features (CareRecord_Problem).

    `Active Problem`  -> active flag; `Significant Problem` -> significant flag.
    Emits per-category `<CAT>_HAS_ACTIVE_PROBLEM` / `<CAT>_HAS_SIGNIFICANT_PROBLEM`,
    plus `NUM_ACTIVE_SIGNIFICANT_PROBLEMS` and `SIGNIFICANT_PROBLEM_BURDEN`.
    Leakage-safe (comorbidity/symptom categories only). Returns just the patient_guid
    column (no-op) if the cohort SQL didn't emit the two problem columns.
    """
    codes_map = codes_map or {**SYMPTOM_CODES, **COMORBIDITY_CODES}
    code_col = 'snomed_c_t_concept_id'
    patients = df['patient_guid_CLEAN'].dropna().unique()
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))

    has_status = 'problem_status_description' in df.columns
    has_sig = 'significance_description' in df.columns
    if not (has_status or has_sig):
        return out.reset_index()      # columns absent in this extract -> skip cleanly

    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_active'] = (work['problem_status_description'].astype(str).str.strip().str.lower()
                       == ACTIVE_STATUS_VALUE) if has_status else False
    work['_sig'] = (work['significance_description'].astype(str).str.strip().str.lower()
                    == SIGNIFICANT_VALUE) if has_sig else False

    all_codes = {c for codes in codes_map.values() for c in codes}
    for cat, codes in codes_map.items():
        sub = work[work[code_col].isin(codes)]
        a_col, s_col = f'{cat}_HAS_ACTIVE_PROBLEM', f'{cat}_HAS_SIGNIFICANT_PROBLEM'
        if sub.empty:
            out[a_col] = 0
            out[s_col] = 0
            continue
        g = sub.groupby('patient_guid_CLEAN')
        out[a_col] = g['_active'].max()
        out[s_col] = g['_sig'].max()

    asig = work[work['_active'] & work['_sig'] & work[code_col].isin(all_codes)]
    out['NUM_ACTIVE_SIGNIFICANT_PROBLEMS'] = asig.groupby('patient_guid_CLEAN').size()
    out = out.fillna(0)
    sig_cols = [c for c in out.columns if c.endswith('_HAS_SIGNIFICANT_PROBLEM')]
    out['SIGNIFICANT_PROBLEM_BURDEN'] = out[sig_cols].sum(axis=1)
    flag_cols = [c for c in out.columns
                 if c.endswith(('_HAS_ACTIVE_PROBLEM', '_HAS_SIGNIFICANT_PROBLEM'))]
    out[flag_cols] = out[flag_cols].astype(int)
    return out.reset_index()


NEUTROPHIL_CODE = [1022551000000104]
LYMPHOCYTE_CODE = [1022581000000105]
PLATELET_CODE = [1022651000000100]


def compute_blood_ratios(df):
    """Per-patient NLR (neutrophil/lymphocyte) and PLR (platelet/lymphocyte) from the most
    recent value of each — recognised lung-cancer inflammatory markers. Missing -> NaN
    (the modeling step median-imputes, which is correct for 'no CBC')."""
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_v'] = pd.to_numeric(work.get('value'), errors='coerce')
    work['_dba'] = pd.to_numeric(work.get('days_before_anchor'), errors='coerce')

    def last_val(codes):
        s = work[work[code_col].isin(codes) & work['_v'].notna()].sort_values('_dba')
        return s.groupby('patient_guid_CLEAN')['_v'].first()   # smallest days_before_anchor = most recent

    neut, lymph, plat = last_val(NEUTROPHIL_CODE), last_val(LYMPHOCYTE_CODE), last_val(PLATELET_CODE)
    patients = df['patient_guid_CLEAN'].dropna().unique()
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))
    lymph_safe = lymph.replace(0, np.nan)
    out['NLR'] = neut / lymph_safe
    out['PLR'] = plat / lymph_safe
    return out.reset_index()


# ── Phase-1 value-driving families (xpoll is Phase 2; these need no train reference) ──────────
# Per-analyte VALUE codes (copied from transform_features value_tasks) — for fuller level stats.
LAB_VALUE_CODES = {
    'HAEMOGLOBIN': [1022431000000105, 271026005],
    'PLATELET':    [1022651000000100],
    'LYMPHOCYTE':  [1022581000000105],
    'NEUTROPHIL':  [1022551000000104],
    'CRP':         [1001371000000100, 999651000000107],
    'ESR':         [1022511000000103],
    'CALCIUM':     [1000691000000101, 935051000000108],
    'SODIUM':      [1000661000000107, 1017381000000106],
    'MCV':         [1022491000000106],
    'ALBUMIN':     [1000821000000103],
}

# Symptom/comorbidity clusters (multi-system presentation often precedes lung-ca diagnosis).
CLUSTER_DEFS = {
    'RESP_SYMPTOM':  ['COUGH', 'BREATHLESSNESS', 'CHESTPAIN', 'CHESTINFECTION', 'HAEMOPTYSIS'],
    'RESP_COMORBID': ['COPD', 'EMPHYSEMA', 'FIBROSIS', 'ASTHMA'],
}


def compute_lab_stats(df, lab_codes=None):
    """Per-analyte LEVEL stats (latest/max/min/mean + measured-flag). Lung's value_trend gives
    slope/first/last; this adds the magnitude summaries. Value cols stay NaN when never measured
    (median-imputed downstream); *_MEASURED is a 0/1 flag."""
    lab_codes = lab_codes or LAB_VALUE_CODES
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_v'] = pd.to_numeric(work.get('value'), errors='coerce')
    work['_dba'] = pd.to_numeric(work.get('days_before_anchor'), errors='coerce')
    work = work[work['_v'].notna()]
    out = pd.DataFrame(index=pd.Index(df['patient_guid_CLEAN'].dropna().unique(), name='patient_guid'))
    for name, codes in lab_codes.items():
        sub = work[work[code_col].isin(codes)]
        if sub.empty:
            for sfx in ('_LATEST', '_VMAX', '_VMIN', '_VMEAN'):
                out[f'{name}{sfx}'] = np.nan
            out[f'{name}_MEASURED'] = 0
            continue
        g = sub.groupby('patient_guid_CLEAN')
        out[f'{name}_LATEST'] = sub.sort_values('_dba').groupby('patient_guid_CLEAN')['_v'].first()  # most recent
        out[f'{name}_VMAX'] = g['_v'].max()
        out[f'{name}_VMIN'] = g['_v'].min()
        out[f'{name}_VMEAN'] = g['_v'].mean()
        out[f'{name}_MEASURED'] = (g['_v'].size() > 0).astype(int)
        # value acceleration: slope(recent half) - slope(older half) (>=4 measurements). +ve = steepening
        s2 = sub.sort_values(['patient_guid_CLEAN', '_dba'], ascending=[True, False])   # oldest first
        s2['_n'] = s2.groupby('patient_guid_CLEAN')['_v'].transform('size')
        s2['_r'] = s2.groupby('patient_guid_CLEAN').cumcount()
        h = s2[s2['_n'] >= 4].copy()
        if h.empty:
            out[f'{name}_VALUE_ACCEL'] = np.nan
        else:
            h['_half'] = np.where(h['_r'] < h['_n'] / 2.0, 'old', 'new')
            h['_x'] = -h['_dba']
            s_old = _patient_slope(h[h['_half'] == 'old'], '_x', '_v')
            s_new = _patient_slope(h[h['_half'] == 'new'], '_x', '_v')
            out[f'{name}_VALUE_ACCEL'] = s_new - s_old
    meas = [c for c in out.columns if c.endswith('_MEASURED')]
    out[meas] = out[meas].fillna(0).astype(int)
    return out.reset_index()


def compute_clusters(df, symptom_codes=None, comorbid_codes=None):
    """Per-patient cluster co-occurrence: # distinct member-categories present in each cluster
    (multi-symptom presentation). Count-like -> 0 when absent."""
    allcodes = {**(symptom_codes or SYMPTOM_CODES), **(comorbid_codes or COMORBIDITY_CODES)}
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    patients = df['patient_guid_CLEAN'].dropna().unique()
    pres = {cat: (work[work[code_col].isin(codes)].groupby('patient_guid_CLEAN').size() > 0).astype(int)
            for cat, codes in allcodes.items()}
    pres_df = pd.DataFrame(pres).reindex(patients).fillna(0)
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))
    for cl, members in CLUSTER_DEFS.items():
        mem = [m for m in members if m in pres_df.columns]
        out[f'{cl}_CLUSTER_COUNT'] = pres_df[mem].sum(axis=1) if mem else 0
        out[f'{cl}_CLUSTER_ANY'] = (out[f'{cl}_CLUSTER_COUNT'] > 0).astype(int)
    return out.fillna(0).reset_index()


def compute_consultation_dynamics(df, ref_col='days_before_anchor'):
    """Per-patient healthcare-utilisation dynamics from distinct encounter-days: total / recent-rate /
    acceleration / recency. Rising consultation frequency can be prodromal. Guarded by the 12mo gap;
    VERIFY via the leadtime screen (same caveat as NLR/PLR). Counts/rates -> 0; recency -> NaN."""
    work = df.copy()
    work['_mb'] = _months_before(work, ref_col)
    work = work[work['_mb'].notna()]
    work['_mb'] = work['_mb'] - work['_mb'].min()   # re-reference to data cutoff -> bins start at 0
    key = 'event_date_parsed' if 'event_date_parsed' in work.columns else '_mb'
    enc = work.drop_duplicates(['patient_guid_CLEAN', key])      # one encounter per patient-day
    out = pd.DataFrame(index=pd.Index(df['patient_guid_CLEAN'].dropna().unique(), name='patient_guid'))
    g = enc.groupby('patient_guid_CLEAN')
    total = g.size()
    out['CONSULT_TOTAL'] = total
    out['CONSULT_RECENCY_MONTHS'] = g['_mb'].min()

    def _binct(lo, hi):
        return enc[(enc['_mb'] > lo) & (enc['_mb'] <= hi)].groupby('patient_guid_CLEAN').size()
    c_recent = _binct(BIN_RECENT[0] - 1e-9, BIN_RECENT[1]).reindex(total.index, fill_value=0)
    c_mid = _binct(*BIN_MID).reindex(total.index, fill_value=0)
    c_old = _binct(*BIN_OLD).reindex(total.index, fill_value=0)
    out['CONSULT_ACCEL'] = c_recent - 2 * c_mid + c_old
    recent = enc[enc['_mb'] <= RECENT_RATIO_CUTOFF].groupby('patient_guid_CLEAN').size().reindex(total.index, fill_value=0)
    out['CONSULT_RECENT_RATE'] = recent / total
    for c in ('CONSULT_TOTAL', 'CONSULT_ACCEL', 'CONSULT_RECENT_RATE'):
        out[c] = out[c].fillna(0)
    return out.reset_index()


def compute_xpoll(df, ref=None, min_band_n=20):
    """Phase 2: cohort + age-band PERCENTILE-RANK of each lab's most-recent value.
    Where a patient's lab sits vs the population is a strong, scale-free signal.

    LEAKAGE-SAFE via a fit-on-train reference:
      * FIT  (ref=None): returns (features_df, ref_dict) — the sorted value arrays (overall + per
        age-decade-band) computed from THIS df (the training cohort).
      * APPLY (ref given): returns features_df — percentiles of this df's patients against the
        train-fit ref. So the held-out is scored against the training distribution only.
    Emits `{ANALYTE}_PCTILE` (cohort) and `{ANALYTE}_AGEBAND_PCTILE`. NaN when unmeasured
    (median-imputed downstream)."""
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_v'] = pd.to_numeric(work.get('value'), errors='coerce')
    work['_dba'] = pd.to_numeric(work.get('days_before_anchor'), errors='coerce')
    work = work[work['_v'].notna()]
    patients = df['patient_guid_CLEAN'].dropna().unique()
    age = pd.to_numeric(df.groupby('patient_guid_CLEAN')['patient_age'].first(), errors='coerce')
    band = (age // 10 * 10)                                   # age-decade band
    fitting = ref is None
    ref = {} if fitting else dict(ref)
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))

    def _pct(v, arr):
        if arr is None or len(arr) == 0 or pd.isna(v):
            return np.nan
        return float(np.searchsorted(np.asarray(arr), v) / len(arr))

    for name, codes in LAB_VALUE_CODES.items():
        latest = (work[work[code_col].isin(codes)].sort_values('_dba')
                  .groupby('patient_guid_CLEAN')['_v'].first()).reindex(patients)   # most-recent value/pt
        if fitting:
            allv = np.sort(latest.dropna().values).tolist()
            bands = {}
            bvec = band.reindex(patients).values
            for b in pd.unique(band.dropna().values):
                bv = latest.values[bvec == b]
                bv = bv[~pd.isna(bv)]
                if len(bv) >= min_band_n:
                    bands[str(int(b))] = np.sort(bv).tolist()
            ref[name] = {'all': allv, 'bands': bands}
        rn = ref.get(name, {'all': [], 'bands': {}})
        out[f'{name}_PCTILE'] = [_pct(latest.get(p, np.nan), rn['all']) for p in patients]
        out[f'{name}_AGEBAND_PCTILE'] = [
            _pct(latest.get(p, np.nan),
                 rn['bands'].get(str(int(band.get(p))) if pd.notna(band.get(p)) else '', rn['all']))
            for p in patients]
    out = out.reset_index()
    return (out, ref) if fitting else out


def _patient_slope(sub, xcol, ycol, key='patient_guid_CLEAN'):
    """Vectorized per-patient OLS slope via summed moments (no per-group apply)."""
    d = sub[[key, xcol, ycol]].copy()
    d['_xy'] = d[xcol] * d[ycol]
    d['_xx'] = d[xcol] * d[xcol]
    a = d.groupby(key).agg(n=(xcol, 'size'), Sx=(xcol, 'sum'), Sy=(ycol, 'sum'),
                           Sxy=('_xy', 'sum'), Sxx=('_xx', 'sum'))
    den = (a['n'] * a['Sxx'] - a['Sx'] ** 2).replace(0, np.nan)
    return (a['n'] * a['Sxy'] - a['Sx'] * a['Sy']) / den


def compute_band_features(df, bands=None, relative=True):
    """Per-CONCEPT, per-time-band occurrence (+ value mean/latest for lab concepts).

    `relative=True` -> windows count back from each patient's most-recent event (gap-agnostic,
    inference-safe; V1's _LAST_NM idea generalized to multiple disjoint bands, SAME bands for
    every horizon). `relative=False` -> fixed months-before-anchor. count/present -> 0 when absent;
    value cols -> NaN (median-imputed downstream)."""
    bands = bands or BANDS_RELATIVE
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_v'] = pd.to_numeric(work.get('value'), errors='coerce')
    work['_dba'] = pd.to_numeric(work.get('days_before_anchor'), errors='coerce')
    work['_mb'] = _months_before(work)
    work = work[work['_mb'].notna()]
    # Band 0 ALWAYS = start of available data (the SQL already applied the gap cutoff, so don't
    # re-offset by it). relative -> per-patient last event; else -> cohort data cutoff (global min).
    if relative:
        work['_mb'] = work['_mb'] - work.groupby('patient_guid_CLEAN')['_mb'].transform('min')
    else:
        work['_mb'] = work['_mb'] - work['_mb'].min()

    patients = df['patient_guid_CLEAN'].dropna().unique()
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))
    occ_defs = {**SYMPTOM_CODES, **COMORBIDITY_CODES}
    for lo, hi in bands:
        b = work[(work['_mb'] >= lo) & (work['_mb'] < hi)]
        tag = f'w{int(lo)}_{int(hi)}'
        for cat, codes in occ_defs.items():
            cnt = b[b[code_col].isin(codes)].groupby('patient_guid_CLEAN').size()
            out[f'{cat}_count_{tag}'] = cnt
            out[f'{cat}_present_{tag}'] = (cnt > 0).astype(int)
        for cat, codes in LAB_VALUE_CODES.items():
            sub = b[b[code_col].isin(codes) & b['_v'].notna()]
            if sub.empty:
                out[f'{cat}_val_mean_{tag}'] = np.nan
                out[f'{cat}_val_latest_{tag}'] = np.nan
                continue
            out[f'{cat}_val_mean_{tag}'] = sub.groupby('patient_guid_CLEAN')['_v'].mean()
            out[f'{cat}_val_latest_{tag}'] = (sub.sort_values('_dba')      # smallest days = most recent
                                              .groupby('patient_guid_CLEAN')['_v'].first())
            out[f'{cat}_val_slope_{tag}'] = _patient_slope(                # within-band value trend
                sub.assign(_x=-sub['_dba']), '_x', '_v')
    cnt_cols = [c for c in out.columns if '_count_w' in c or '_present_w' in c]
    out[cnt_cols] = out[cnt_cols].fillna(0)
    return out.reset_index()


def compute_cumulative_features(df, windows=None, relative=True):
    """Per-CONCEPT cumulative 'last-N months' windows (count + value mean/latest for lab concepts).
    Overlapping windows from the start of available data; complements the disjoint bands. Generalizes
    V1's single _LAST_NM to the full set. count -> 0 when absent; value cols -> NaN."""
    windows = windows or CUMULATIVE_WINDOWS
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    if 'event_type_lower' in work.columns:
        work = work[work['event_type_lower'] == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_v'] = pd.to_numeric(work.get('value'), errors='coerce')
    work['_dba'] = pd.to_numeric(work.get('days_before_anchor'), errors='coerce')
    work['_mb'] = _months_before(work)
    work = work[work['_mb'].notna()]
    if relative:
        work['_mb'] = work['_mb'] - work.groupby('patient_guid_CLEAN')['_mb'].transform('min')
    else:
        work['_mb'] = work['_mb'] - work['_mb'].min()

    patients = df['patient_guid_CLEAN'].dropna().unique()
    out = pd.DataFrame(index=pd.Index(patients, name='patient_guid'))
    occ_defs = {**SYMPTOM_CODES, **COMORBIDITY_CODES}
    for n in windows:
        w = work[work['_mb'] < n]
        tag = f'last{n}'
        for cat, codes in occ_defs.items():
            out[f'{cat}_count_{tag}'] = w[w[code_col].isin(codes)].groupby('patient_guid_CLEAN').size()
        for cat, codes in LAB_VALUE_CODES.items():
            sub = w[w[code_col].isin(codes) & w['_v'].notna()]
            if sub.empty:
                out[f'{cat}_val_mean_{tag}'] = np.nan
                out[f'{cat}_val_latest_{tag}'] = np.nan
                continue
            out[f'{cat}_val_mean_{tag}'] = sub.groupby('patient_guid_CLEAN')['_v'].mean()
            out[f'{cat}_val_latest_{tag}'] = (sub.sort_values('_dba')
                                              .groupby('patient_guid_CLEAN')['_v'].first())
    cnt_cols = [c for c in out.columns if '_count_last' in c]
    out[cnt_cols] = out[cnt_cols].fillna(0)
    return out.reset_index()


def add_interactions(mat):
    """Add multiplicative interaction terms on the merged feature matrix (in place-safe)."""
    mat = mat.copy()
    age = pd.to_numeric(mat.get('patient_age'), errors='coerce')
    packyears = pd.to_numeric(mat.get('PACK_YEARS_MAX'), errors='coerce').fillna(0)

    def _flag(col):
        if col in mat.columns:
            return (pd.to_numeric(mat[col], errors='coerce').fillna(0) > 0).astype(float)
        return pd.Series(0.0, index=mat.index)

    smoke    = _flag('NUM_SMOKING_OCCURRENCES')
    copd     = _flag('NUM_COPD_OCCURRENCES')
    haemo    = _flag('HAEMOPTYSIS_PRESENT')
    breath   = _flag('BREATHLESSNESS_PRESENT')
    chestinf = _flag('CHESTINFECTION_PRESENT')
    cough    = _flag('COUGH_PRESENT')
    clubbing = _flag('CLUBBING_PRESENT')

    mat['INT_SMOKING_x_AGE']            = smoke * age
    mat['INT_SMOKING_x_COPD']           = smoke * copd
    mat['INT_AGE_x_HAEMOPTYSIS']        = age * haemo
    # extra curated pairs (Phase 1)
    mat['INT_SMOKING_x_HAEMOPTYSIS']    = smoke * haemo
    mat['INT_COPD_x_HAEMOPTYSIS']       = copd * haemo
    mat['INT_SMOKING_x_CHESTINFECTION'] = smoke * chestinf
    mat['INT_COPD_x_CHESTINFECTION']    = copd * chestinf
    mat['INT_AGE_x_BREATHLESSNESS']     = age * breath
    mat['INT_PACKYEARS_x_AGE']          = packyears * age
    mat['INT_HAEMOPTYSIS_x_COUGH']      = haemo * cough
    # finger clubbing is high-specificity — amplify it in a smoker / with haemoptysis
    mat['INT_CLUBBING_x_SMOKING']       = clubbing * smoke
    mat['INT_CLUBBING_x_HAEMOPTYSIS']   = clubbing * haemo
    return mat


def enrich(matrix, df):
    """Merge all enhanced features into the feature matrix. Returns the enriched matrix.
    No-op-safe: if env FE_ENHANCED == '0', returns the matrix unchanged."""
    if os.environ.get('FE_ENHANCED', '1') == '0':
        print('[enhanced_features] FE_ENHANCED=0 -> skipping enhanced features.')
        return matrix
    n0 = matrix.shape[1]
    dyn = compute_symptom_dynamics(df)
    dose = compute_smoking_dose(df)
    prob = compute_problem_flags(df)
    ratios = compute_blood_ratios(df)
    labs = compute_lab_stats(df)                 # Phase 1: fuller per-analyte level stats
    clusters = compute_clusters(df)              # Phase 1: symptom/comorbid cluster co-occurrence
    consults = compute_consultation_dynamics(df) # Phase 1: consultation-frequency dynamics
    matrix = matrix.merge(dyn, on='patient_guid', how='left')
    matrix = matrix.merge(dose, on='patient_guid', how='left')
    matrix = matrix.merge(ratios, on='patient_guid', how='left')   # NLR/PLR (NaN->imputed downstream)
    matrix = matrix.merge(labs, on='patient_guid', how='left')
    matrix = matrix.merge(clusters, on='patient_guid', how='left')
    matrix = matrix.merge(consults, on='patient_guid', how='left')
    if os.environ.get('FE_BANDS', '0') == '1':   # patient-relative per-concept time bands (opt-in)
        rel = os.environ.get('FE_BAND_ANCHOR', 'patient_last') == 'patient_last'
        bands = compute_band_features(df, relative=rel)
        matrix = matrix.merge(bands, on='patient_guid', how='left')
        cumul = compute_cumulative_features(df, relative=rel)        # cumulative last-N windows
        matrix = matrix.merge(cumul, on='patient_guid', how='left')
        fill0 = [c for c in matrix.columns
                 if '_count_w' in c or '_present_w' in c or '_count_last' in c]
        matrix[fill0] = matrix[fill0].fillna(0)
        print(f'[enhanced_features] FE_BANDS=1 -> added {bands.shape[1] - 1} band + '
              f'{cumul.shape[1] - 1} cumulative per-concept cols '
              f'(anchor={os.environ.get("FE_BAND_ANCHOR", "patient_last")}).')
    if prob.shape[1] > 1:                       # has feature cols beyond patient_guid
        matrix = matrix.merge(prob, on='patient_guid', how='left')
    matrix = add_interactions(matrix)
    # Count/flag/rate cols -> 0 when a patient is absent from a sub-table. VALUE-level cols
    # (*_LATEST/_VMAX/_VMIN/_VMEAN, *_RECENCY_MONTHS) stay NaN -> median-imputed downstream.
    fill0 = [c for c in matrix.columns
             if c.endswith(('_DECAY_INTENSITY', '_ACCEL', '_RECENT_RATIO', '_PRESENT',
                            '_HAS_ACTIVE_PROBLEM', '_HAS_SIGNIFICANT_PROBLEM',
                            '_MEASURED', '_CLUSTER_COUNT', '_CLUSTER_ANY'))
             or c in ('SYMPTOM_BURDEN', 'PACK_YEARS_MAX', 'CIGS_PER_DAY_MAX',
                      'NUM_ACTIVE_SIGNIFICANT_PROBLEMS', 'SIGNIFICANT_PROBLEM_BURDEN',
                      'CONSULT_TOTAL', 'CONSULT_ACCEL', 'CONSULT_RECENT_RATE')]
    matrix[fill0] = matrix[fill0].fillna(0)
    print(f'[enhanced_features] added {matrix.shape[1] - n0} columns (symptom dynamics + smoking dose '
          f'+ problem-list flags + NLR/PLR + lab level-stats + clusters + consultation dynamics + interactions).')
    return matrix
