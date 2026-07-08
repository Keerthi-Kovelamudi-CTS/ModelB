"""
Clinical concept-level feature engineering, operating on the unified `ev` event stream.
====================================================================================
These are the hand-curated CLINICAL-CONCEPT features (symptom dynamics, comorbidity problem-list
flags, multi-system clusters, smoking dose, per-analyte lab level-stats, interaction terms). In the
TIERED design they are computed ONLY on the curated codes; every OTHER codelist code is handled
by the generic per-code families in build_features.py (no code is treated twice).

SCHEMA: operates on `ev` (see build_features.load_events) — one row per event with the unified
int64 `code` (snomed for observations, med_code_id for medications), `value_num`, `days`
(days_before_anchor), `months`, `event_age`, `event_type` ("observation"/"medication"),
`active`/`sig` (0/1 problem-list flags), and per-patient `patient_guid`. `code`/`value_num`/`days`
are already numeric.

ALL functions return a DataFrame indexed by `patient_guid` (index name "patient_guid") so build()
can `patients.join(block)` them directly — they do NOT reset_index. Fill rules: count/flag/rate -> 0
(genuine absence); value/level/recency -> NaN (median-imputed downstream, never fake-0).

NOTE: NLR/PLR/LMR/CRP-albumin blood ratios are NOT here — they are already implemented as
`blood_ratio_features` in build_features.py.
"""
import numpy as np
import pandas as pd

# ── Code maps ────────────────────────────────────────────────────────────────────────────────
# Leakage-safe presenting-symptom categories. HAEMOPTYSIS isolated from referral/workup redflags.
SYMPTOM_CODES = {
    'COUGH':          [49727002, 161929000, 11833005, 284523002, 161947006, 161924005],
    'BREATHLESSNESS': [267036007, 391120009, 391123006, 391124000, 391125004],
    'CHESTPAIN':      [29857009, 102589003, 2237002],
    'CHESTINFECTION': [312342009, 32398004, 396285007, 54150009, 54398005,
                       195647007, 50417007, 195742007],
    'HAEMOPTYSIS':    [66857006],
    'CLUBBING':       [164457001],   # O/E finger clubbing — high-specificity lung-ca sign
}

PACK_YEAR_CODES = [315609007, 401201003]            # "Pack years" (value = pack-years)
CIGS_PER_DAY_CODES = [230056004, 65568007]          # "Cigarette consumption" (value = cigs/day)

# Chronic comorbidity diagnoses (problem-list flags).
COMORBIDITY_CODES = {
    'COPD':      [723245007, 313297008, 313299006, 13645005, 204991000000107],
    'EMPHYSEMA': [87433001, 909721000006104, 68328006, 263747008],
    'FIBROSIS':  [700250006, 909731000006101, 51615001],
    'ASTHMA':    [195967001],
}

# Medication groups (med_code_id; unified into `code` for medication rows in V3).
MED_CODES = {
    'ASTHMA_MED': [12906411000001100, 106511000001103, 3215311000001107, 9516911000001109,
                   39113611000001102, 9205211000001104, 222311000001102, 4053411000001103,
                   726611000001102, 2831211000001109, 3184911000001108, 3184311000001107,
                   398511000001105, 42292311000001106, 35908811000001103],
    'SMOKING_CESSATION_MED': [
        88711000001104, 330811000001104, 352511000001102, 441911000001109, 457011000001103,
        662311000001108, 703311000001108, 714711000001106, 768511000001107, 868611000001109,
        2833611000001105, 2834111000001100, 2834711000001104, 2835311000001104, 3052211000001105,
        3053411000001100, 3054111000001107, 3054911000001105, 3055311000001108, 3062511000001104,
        3063211000001108, 3064011000001101, 3065411000001106, 3065811000001108, 3208111000001103,
        3214111000001106, 3215211000001104, 3216011000001100, 3217711000001100, 3229811000001105,
        3230611000001103, 3501611000001105, 3505611000001108, 3506011000001105, 3559411000001101,
        3559511000001102, 3559611000001103, 4990511000001109, 4990811000001107, 5181211000001109,
        5181511000001107, 5612711000001105, 5623211000001100, 8148611000001100, 8148911000001106,
        9178211000001104, 9178511000001101, 10143711000001104, 10143911000001102, 10144111000001103,
        10971311000001100, 10971611000001105, 10981211000001103, 10984311000001109, 11548311000001106,
        11548611000001101, 13113011000001109, 14752711000001102, 14979111000001100, 14979311000001103,
        14979811000001107, 14984411000001104, 15244211000001102, 15244511000001104, 16237311000001103,
        16250511000001109, 16528011000001108, 18244311000001102, 18549811000001104, 18561411000001109,
        18562311000001106, 19370611000001106, 19370911000001100, 19475011000001104, 19482211000001108,
        21145711000001106, 21146211000001105, 21520411000001104, 22735911000001103, 22787211000001107,
        23072811000001107, 26351211000001107, 26351611000001109, 30223311000001102, 30781011000001101,
        32473711000001107, 34557011000001101, 34863411000001106, 34864011000001100, 35599911000001105,
        35721211000001108, 35914111000001102, 36563611000001102, 36565311000001108, 36566311000001103,
        37764511000001100, 38095911000001102, 38897211000001102, 38961311000001109, 39022411000001107,
        39111611000001101, 39112811000001106, 39707011000001106, 39707111000001107, 41512611000001105,
        41512911000001104, 41513211000001102, 42296911000001106, 42509011000001109, 44960711000001103,
        45331611000001108, 45331811000001107],
    'COPD_INHALER': [
        42292811000001106, 39417611000001109, 38894511000001107, 3378211000001106, 3380011000001106,
        9478911000001107, 9479011000001103, 12146911000001103, 12197411000001102, 20985511000001101,
        21495411000001107, 21496211000001102, 24644611000001108, 24645511000001105, 27567911000001101,
        27890611000001109, 28007211000001102, 28049611000001104, 28357211000001106, 28365011000001100,
        29971311000001100, 29987211000001108, 33594911000001100, 33596311000001107, 34681611000001100,
        34952211000001104, 37677711000001102, 37678011000001103, 37692511000001100, 37692711000001105,
        38893611000001108, 39343611000001104, 39993311000001105, 40752211000001109],
}

# CareRecord_Problem value strings.
ACTIVE_STATUS_VALUE = 'active problem'
SIGNIFICANT_VALUE = 'significant problem'

# Per-analyte VALUE codes — for fuller per-analyte level stats.
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

# Dynamics constants (match build_features).
DECAY_TAU_MONTHS = 12.0
BIN_RECENT = (0.0, 6.0)
BIN_MID = (6.0, 18.0)
BIN_OLD = (18.0, 42.0)
RECENT_RATIO_CUTOFF = 24.0
RECENT_BURDEN_WINDOWS = [6, 12]   # # distinct symptom categories present in the last N months


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────
def _empty():
    return pd.DataFrame(index=pd.Index([], name='patient_guid'))


def _patients_index(ev):
    return pd.Index(ev['patient_guid'].dropna().unique(), name='patient_guid')


def _obs(ev):
    """Observation rows only (V3 `code` already int64; no coercion needed)."""
    return ev[ev['event_type'].eq('observation')]


def _patient_slope(sub, xcol, ycol, key='patient_guid'):
    """Vectorized per-patient OLS slope via summed moments (no per-group apply)."""
    d = sub[[key, xcol, ycol]].copy()
    d['_xy'] = d[xcol] * d[ycol]
    d['_xx'] = d[xcol] * d[xcol]
    a = d.groupby(key).agg(n=(xcol, 'size'), Sx=(xcol, 'sum'), Sy=(ycol, 'sum'),
                           Sxy=('_xy', 'sum'), Sxx=('_xx', 'sum'))
    den = (a['n'] * a['Sxx'] - a['Sx'] ** 2).replace(0, np.nan)
    return (a['n'] * a['Sxy'] - a['Sx'] * a['Sy']) / den


# ── Concept feature families ────────────────────────────────────────────────────────────────
def compute_symptom_dynamics(ev, symptom_codes=None, decay_tau_months=DECAY_TAU_MONTHS):
    """Per-patient recency / decay-intensity / acceleration / recent-ratio / presence / burden per
    symptom category. recency/first-occurrence/span = RAW months (staleness from anchor); decay/accel/
    recent_ratio referenced to the patient's OWN most-recent event (inference-safe)."""
    symptom_codes = symptom_codes or SYMPTOM_CODES
    work = _obs(ev).copy()
    work['_mb'] = work['months']                                   # months-before-anchor (already computed)
    work = work[work['_mb'].notna()]
    # per-patient relative months: anchor on the patient's OWN most-recent event -> starts at 0
    work['_mbrel'] = work['_mb'] - work.groupby('patient_guid')['_mb'].transform('min')

    out = pd.DataFrame(index=_patients_index(ev))
    present_cols = []
    recent_present = {w: [] for w in RECENT_BURDEN_WINDOWS}

    for cat, codes in symptom_codes.items():
        sub = work[work['code'].isin(codes)]
        rec_col, dec_col = f'{cat}_RECENCY_MONTHS', f'{cat}_DECAY_INTENSITY'
        acc_col, rat_col = f'{cat}_ACCEL', f'{cat}_RECENT_RATIO'
        prs_col = f'{cat}_PRESENT'
        foc_col, span_col = f'{cat}_FIRST_OCCURRENCE_MONTHS', f'{cat}_SYMPTOM_SPAN_MONTHS'
        present_cols.append(prs_col)
        if sub.empty:
            out[rec_col] = np.nan
            out[dec_col] = 0.0
            out[acc_col] = 0.0
            out[rat_col] = 0.0
            out[prs_col] = 0
            out[foc_col] = np.nan
            out[span_col] = np.nan
            continue
        g = sub.groupby('patient_guid')
        total = g.size()
        recency = g['_mb'].min()                                   # RAW staleness from anchor
        first_occ = g['_mb'].max()                                 # RAW earliest occurrence
        decay = (sub.assign(_w=np.exp(-sub['_mbrel'] / decay_tau_months))
                 .groupby('patient_guid')['_w'].sum())

        def _binct(lo, hi):
            return sub[(sub['_mbrel'] > lo) & (sub['_mbrel'] <= hi)].groupby('patient_guid').size()
        c_recent = _binct(BIN_RECENT[0] - 1e-9, BIN_RECENT[1]).reindex(total.index, fill_value=0)
        c_mid = _binct(*BIN_MID).reindex(total.index, fill_value=0)
        c_old = _binct(*BIN_OLD).reindex(total.index, fill_value=0)
        accel = c_recent - 2 * c_mid + c_old
        recent = sub[sub['_mbrel'] <= RECENT_RATIO_CUTOFF].groupby('patient_guid').size()
        recent_ratio = recent.reindex(total.index, fill_value=0) / total

        out[rec_col] = recency
        out[dec_col] = decay
        out[acc_col] = accel
        out[rat_col] = recent_ratio
        out[prs_col] = (total > 0).astype(int)
        out[foc_col] = first_occ
        out[span_col] = first_occ - recency
        for w in RECENT_BURDEN_WINDOWS:
            cw = sub[sub['_mbrel'] <= w].groupby('patient_guid').size()
            recent_present[w].append((cw > 0).astype(int))

    for w in RECENT_BURDEN_WINDOWS:
        out[f'RECENT_SYMPTOM_BURDEN_{w}MO'] = (
            pd.concat(recent_present[w], axis=1).reindex(out.index).sum(axis=1)
            if recent_present[w] else 0)

    for c in out.columns:                                          # month cols stay NaN; rest -> 0
        if c.endswith('_MONTHS'):
            continue
        out[c] = out[c].fillna(0)
    out['SYMPTOM_BURDEN'] = out[present_cols].sum(axis=1) if present_cols else 0
    out.index.name = 'patient_guid'
    return out


def compute_smoking_dose(ev):
    """Per-patient max recorded pack-years and cigarettes/day (leakage-safe risk dose)."""
    work = _obs(ev)
    out = pd.DataFrame(index=_patients_index(ev))
    val = work[work['value_num'].notna()]
    py = val[val['code'].isin(PACK_YEAR_CODES)].groupby('patient_guid')['value_num'].max()
    cpd = val[val['code'].isin(CIGS_PER_DAY_CODES)].groupby('patient_guid')['value_num'].max()
    out['PACK_YEARS_MAX'] = py
    out['CIGS_PER_DAY_MAX'] = cpd                                  # value cols stay NaN -> imputed
    out.index.name = 'patient_guid'
    return out


def compute_problem_flags(ev, codes_map=None):
    """Per-patient clinician-curated problem-list flags (CareRecord_Problem) per category:
    `<CAT>_HAS_ACTIVE_PROBLEM` / `<CAT>_HAS_SIGNIFICANT_PROBLEM` + NUM_ACTIVE_SIGNIFICANT_PROBLEMS +
    SIGNIFICANT_PROBLEM_BURDEN. Uses V3's precomputed `active`/`sig` 0/1 flags."""
    codes_map = codes_map or {**SYMPTOM_CODES, **COMORBIDITY_CODES}
    out = pd.DataFrame(index=_patients_index(ev))
    work = _obs(ev)
    if not ({'active', 'sig'} <= set(work.columns)):
        out.index.name = 'patient_guid'
        return out                                                 # flags absent -> skip cleanly

    all_codes = {c for codes in codes_map.values() for c in codes}
    for cat, codes in codes_map.items():
        sub = work[work['code'].isin(codes)]
        a_col, s_col = f'{cat}_HAS_ACTIVE_PROBLEM', f'{cat}_HAS_SIGNIFICANT_PROBLEM'
        if sub.empty:
            out[a_col] = 0
            out[s_col] = 0
            continue
        g = sub.groupby('patient_guid')
        out[a_col] = g['active'].max()
        out[s_col] = g['sig'].max()

    asig = work[(work['active'] == 1) & (work['sig'] == 1) & work['code'].isin(all_codes)]
    out['NUM_ACTIVE_SIGNIFICANT_PROBLEMS'] = asig.groupby('patient_guid').size()
    out = out.fillna(0)
    sig_cols = [c for c in out.columns if c.endswith('_HAS_SIGNIFICANT_PROBLEM')]
    out['SIGNIFICANT_PROBLEM_BURDEN'] = out[sig_cols].sum(axis=1) if sig_cols else 0
    flag_cols = [c for c in out.columns
                 if c.endswith(('_HAS_ACTIVE_PROBLEM', '_HAS_SIGNIFICANT_PROBLEM'))]
    out[flag_cols] = out[flag_cols].astype(int)
    out.index.name = 'patient_guid'
    return out


def compute_lab_stats(ev, lab_codes=None):
    """Per-analyte LEVEL stats (latest/max/min/mean + measured-flag + value-accel). Value cols stay
    NaN when never measured (median-imputed downstream); *_MEASURED is a 0/1 flag."""
    lab_codes = lab_codes or LAB_VALUE_CODES
    work = _obs(ev)
    work = work[work['value_num'].notna()]
    out = pd.DataFrame(index=_patients_index(ev))
    for name, codes in lab_codes.items():
        sub = work[work['code'].isin(codes)]
        if sub.empty:
            for sfx in ('_LATEST', '_VMAX', '_VMIN', '_VMEAN'):
                out[f'{name}{sfx}'] = np.nan
            out[f'{name}_MEASURED'] = 0
            continue
        g = sub.groupby('patient_guid')
        out[f'{name}_LATEST'] = sub.sort_values('days').groupby('patient_guid')['value_num'].first()
        out[f'{name}_VMAX'] = g['value_num'].max()
        out[f'{name}_VMIN'] = g['value_num'].min()
        out[f'{name}_VMEAN'] = g['value_num'].mean()
        out[f'{name}_MEASURED'] = (g['value_num'].size() > 0).astype(int)
        # value acceleration: slope(recent half) - slope(older half) (>=4 measurements)
        s2 = sub.sort_values(['patient_guid', 'days'], ascending=[True, False])   # oldest first
        s2['_n'] = s2.groupby('patient_guid')['value_num'].transform('size')
        s2['_r'] = s2.groupby('patient_guid').cumcount()
        h = s2[s2['_n'] >= 4].copy()
        if h.empty:
            out[f'{name}_VALUE_ACCEL'] = np.nan
        else:
            h['_half'] = np.where(h['_r'] < h['_n'] / 2.0, 'old', 'new')
            h['_x'] = -h['days']
            s_old = _patient_slope(h[h['_half'] == 'old'], '_x', 'value_num')
            s_new = _patient_slope(h[h['_half'] == 'new'], '_x', 'value_num')
            out[f'{name}_VALUE_ACCEL'] = s_new - s_old
    meas = [c for c in out.columns if c.endswith('_MEASURED')]
    out[meas] = out[meas].fillna(0).astype(int)
    out.index.name = 'patient_guid'
    return out


def compute_clusters(ev, symptom_codes=None, comorbid_codes=None):
    """Per-patient cluster co-occurrence: # distinct member-categories present in each cluster
    (multi-symptom presentation). Count-like -> 0 when absent."""
    allcodes = {**(symptom_codes or SYMPTOM_CODES), **(comorbid_codes or COMORBIDITY_CODES)}
    work = _obs(ev)
    patients = _patients_index(ev)
    pres = {cat: (work[work['code'].isin(codes)].groupby('patient_guid').size() > 0).astype(int)
            for cat, codes in allcodes.items()}
    pres_df = pd.DataFrame(pres).reindex(patients).fillna(0)
    out = pd.DataFrame(index=patients)
    for cl, members in CLUSTER_DEFS.items():
        mem = [m for m in members if m in pres_df.columns]
        out[f'{cl}_CLUSTER_COUNT'] = pres_df[mem].sum(axis=1) if mem else 0
        out[f'{cl}_CLUSTER_ANY'] = (out[f'{cl}_CLUSTER_COUNT'] > 0).astype(int)
    out = out.fillna(0)
    out.index.name = 'patient_guid'
    return out


def interaction_features(ev):
    """Derives the concept presence-flags (smoking, COPD, haemoptysis, breathlessness,
    chest-infection, cough, clubbing) + patient age (max event_age) + pack-years directly from
    `ev`/concept code-maps, then builds the INT_* multiplicative products. All outputs numeric."""
    obs = _obs(ev)
    patients = _patients_index(ev)
    out = pd.DataFrame(index=patients)

    age = ev.groupby('patient_guid')['event_age'].max().reindex(patients)   # patient age at the prediction point

    py = (obs[obs['code'].isin(PACK_YEAR_CODES) & obs['value_num'].notna()]
          .groupby('patient_guid')['value_num'].max())
    packyears = py.reindex(patients).fillna(0.0)

    def _present(codes):
        cnt = obs[obs['code'].isin(codes)].groupby('patient_guid').size()
        return (cnt.reindex(patients).fillna(0) > 0).astype(float)

    smoke    = _present(PACK_YEAR_CODES + CIGS_PER_DAY_CODES)      # smoking-status presence (dose codes)
    copd     = _present(COMORBIDITY_CODES['COPD'])
    haemo    = _present(SYMPTOM_CODES['HAEMOPTYSIS'])
    breath   = _present(SYMPTOM_CODES['BREATHLESSNESS'])
    chestinf = _present(SYMPTOM_CODES['CHESTINFECTION'])
    cough    = _present(SYMPTOM_CODES['COUGH'])
    clubbing = _present(SYMPTOM_CODES['CLUBBING'])

    out['INT_SMOKING_x_AGE']            = smoke * age
    out['INT_SMOKING_x_COPD']           = smoke * copd
    out['INT_AGE_x_HAEMOPTYSIS']        = age * haemo
    out['INT_SMOKING_x_HAEMOPTYSIS']    = smoke * haemo
    out['INT_COPD_x_HAEMOPTYSIS']       = copd * haemo
    out['INT_SMOKING_x_CHESTINFECTION'] = smoke * chestinf
    out['INT_COPD_x_CHESTINFECTION']    = copd * chestinf
    out['INT_AGE_x_BREATHLESSNESS']     = age * breath
    out['INT_PACKYEARS_x_AGE']          = packyears * age
    out['INT_HAEMOPTYSIS_x_COUGH']      = haemo * cough
    out['INT_CLUBBING_x_SMOKING']       = clubbing * smoke
    out['INT_CLUBBING_x_HAEMOPTYSIS']   = clubbing * haemo
    out.index.name = 'patient_guid'
    return out
