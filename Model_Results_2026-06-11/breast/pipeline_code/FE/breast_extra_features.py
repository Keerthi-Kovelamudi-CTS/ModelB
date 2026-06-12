"""
Breast EXTRA features — computed post-FE from the raw cohort events, which carry signals the
mature FE drops at 0_preprocess (it restricts to OUT_COLS). Two families, ported from the lung
enhanced_features approach and merged onto the mature FE matrix in run_lookback_experiment.build_matrix:

  1. Clinician problem-list flags (CareRecord_Problem): problem_status_description == 'Active Problem'
     and significance_description == 'Significant Problem'. Per condition-category:
       <CAT>_HAS_ACTIVE_PROBLEM / <CAT>_HAS_SIGNIFICANT_PROBLEM
     plus NUM_ACTIVE_SIGNIFICANT_PROBLEMS and SIGNIFICANT_PROBLEM_BURDEN.
     (LAB_*/LIFE_* categories excluded — measurements/lifestyle aren't clinician "problems".)
  2. Blood-count ratios: NLR (neutrophil/lymphocyte) and PLR (platelet/lymphocyte) from the most
     recent value of each — recognised inflammatory markers. Missing -> NaN (modeling median-imputes).

These columns were NEVER used as features before; this is the breast analogue of what lung did.
Leakage-safe by construction: condition/symptom categories only, and the 12-month pre-anchor gap
(months_before=12) excludes any diagnostic-workup blood test from NLR/PLR.

TODO (post-training): verify NLR/PLR + the problem-flags with tools/leadtime_screen — if any are
strong AND tightly time-locked to diagnosis, drop them. FBC coverage (2026-06-10): cancer 97% /
non-cancer 80%. See memory: breast_extra_features_leakage.
"""
import json
import os
import numpy as np
import pandas as pd

ACTIVE_STATUS_VALUE = 'active problem'
SIGNIFICANT_VALUE = 'significant problem'

# Standard FBC SNOMED codes (cancer-agnostic; not in the breast codelist but present in the raw
# obs events, which the cohort SQL pulls un-filtered).
NEUTROPHIL_CODE = [1022551000000104]
LYMPHOCYTE_CODE = [1022581000000105]
PLATELET_CODE   = [1022651000000100]

_CODELIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "codelist2.0", "code_category_mapping_2.0.json")


def _clean_guid(s):
    return s.astype(str).str.replace(r'[{}"]', '', regex=True).str.strip().str.upper()


def _problem_codes_map():
    """obs categories that denote conditions/symptoms (exclude LAB_/LIFE_) -> {category: {int codes}}."""
    d = json.load(open(_CODELIST))
    by = {}
    for code, cat in d["obs"].items():
        if cat.startswith(("LAB_", "LIFE_")):
            continue
        by.setdefault(cat, set()).add(int(code))
    return by


def compute_problem_flags(df):
    code_col = 'snomed_c_t_concept_id'
    g_all = _clean_guid(df['patient_guid'])
    out = pd.DataFrame(index=pd.Index(g_all.dropna().unique(), name='patient_guid'))

    has_status = 'problem_status_description' in df.columns
    has_sig = 'significance_description' in df.columns
    if not (has_status or has_sig):
        return out.reset_index()            # columns absent -> no-op (just the guid column)

    work = df.copy()
    work['_g'] = g_all.values
    if 'event_type' in work.columns:
        work = work[work['event_type'].astype(str).str.lower() == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_active'] = ((work['problem_status_description'].astype(str).str.strip().str.lower()
                        == ACTIVE_STATUS_VALUE) if has_status else False)
    work['_sig'] = ((work['significance_description'].astype(str).str.strip().str.lower()
                     == SIGNIFICANT_VALUE) if has_sig else False)

    codes_map = _problem_codes_map()
    all_codes = {c for cs in codes_map.values() for c in cs}
    for cat, codes in codes_map.items():
        sub = work[work[code_col].isin(codes)]
        a_col, s_col = f'{cat}_HAS_ACTIVE_PROBLEM', f'{cat}_HAS_SIGNIFICANT_PROBLEM'
        if sub.empty:
            out[a_col] = 0
            out[s_col] = 0
            continue
        gp = sub.groupby('_g')
        out[a_col] = gp['_active'].max()
        out[s_col] = gp['_sig'].max()

    asig = work[work['_active'] & work['_sig'] & work[code_col].isin(all_codes)]
    out['NUM_ACTIVE_SIGNIFICANT_PROBLEMS'] = asig.groupby('_g').size()
    out = out.fillna(0)
    # cast flags to int BEFORE deriving burden, so all emitted cols are numeric (else an object
    # dtype gets dropped by the modeling's numeric-only select)
    flag_cols = [c for c in out.columns
                 if c.endswith(('_HAS_ACTIVE_PROBLEM', '_HAS_SIGNIFICANT_PROBLEM'))]
    out[flag_cols] = out[flag_cols].astype(int)
    sig_cols = [c for c in flag_cols if c.endswith('_HAS_SIGNIFICANT_PROBLEM')]
    out['SIGNIFICANT_PROBLEM_BURDEN'] = out[sig_cols].sum(axis=1).astype(int)
    out['NUM_ACTIVE_SIGNIFICANT_PROBLEMS'] = out['NUM_ACTIVE_SIGNIFICANT_PROBLEMS'].astype(int)
    return out.reset_index()


def compute_blood_ratios(df):
    code_col = 'snomed_c_t_concept_id'
    work = df.copy()
    work['_g'] = _clean_guid(work['patient_guid']).values
    if 'event_type' in work.columns:
        work = work[work['event_type'].astype(str).str.lower() == 'observation']
    work[code_col] = pd.to_numeric(work.get(code_col), errors='coerce')
    work['_v'] = pd.to_numeric(work.get('value'), errors='coerce')
    work['_dba'] = pd.to_numeric(work.get('days_before_anchor'), errors='coerce')

    def last_val(codes):
        s = work[work[code_col].isin(codes) & work['_v'].notna()].sort_values('_dba')
        return s.groupby('_g')['_v'].first()        # smallest days_before_anchor = most recent

    neut, lymph, plat = last_val(NEUTROPHIL_CODE), last_val(LYMPHOCYTE_CODE), last_val(PLATELET_CODE)
    out = pd.DataFrame(index=pd.Index(work['_g'].dropna().unique(), name='patient_guid'))
    lymph_safe = lymph.replace(0, np.nan)
    out['NLR'] = neut / lymph_safe
    out['PLR'] = plat / lymph_safe
    return out.reset_index()


def compute_extras(df):
    """Problem-list flags + NLR/PLR for every patient in df, keyed by cleaned patient_guid."""
    pf = compute_problem_flags(df)
    br = compute_blood_ratios(df)
    return pf.merge(br, on='patient_guid', how='outer')
