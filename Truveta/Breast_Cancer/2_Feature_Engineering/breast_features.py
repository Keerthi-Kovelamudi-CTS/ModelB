"""
Breast-specific feature engineering — Layer B (hand-engineered), Layer C
(engagement / code-type-aware), and Layer D (interactions).

Called from 1_build_features.py after the Layer A category aggregation has
been computed. All feature definitions live in config.py — this module just
realises them against the actual per-patient data.

Inputs:
    merged      long-format (patient_id, code_id, category, code_type, count,
                value_mean, duration_sum, first_event, last_event)
    by_pc       (patient_id, category) aggregates with columns
                count / value_mean / duration_sum / present / recency_days /
                event_density. The same frame Layer A pivots from.
    spine       per-patient row (patient_id, sex, ethnicity, state_or_province,
                patient_age, anchor_date, label, cancer_id)

Output:
    DataFrame indexed by patient_id with Layer B + C + D columns.
"""

import re

import numpy as np
import pandas as pd

import config


def _slug(category: str) -> str:
    """'Lab - Lipid Panel' -> 'lab_lipid_panel' (matches 1_build_features.slug)."""
    return re.sub(r'[^a-z0-9]+', '_', str(category).lower()).strip('_')


# ─────────────────────────────────────────────────────────────────────────────
# Layer B — breast-specific hand-engineered
# ─────────────────────────────────────────────────────────────────────────────

def _category_lookup(by_pc: pd.DataFrame, category: str, value_col: str) -> pd.Series:
    """Return a per-patient Series of `value_col` for the requested category.
    Patients without any event in this category get NaN."""
    sub = by_pc.loc[by_pc['category'] == category, ['patient_id', value_col]]
    if sub.empty:
        return pd.Series(dtype='float64', name=value_col)
    return sub.set_index('patient_id')[value_col]


def build_layer_b(by_pc: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Hand-engineered breast-specific features. Returns DF indexed by patient_id."""
    out = pd.DataFrame(index=pd.Index(spine['patient_id'].unique(), name='patient_id'))

    for name, kind, params in config.BREAST_FEATURES:
        if kind == 'category_duration_years':
            dur = _category_lookup(by_pc, params['category'], 'duration_sum')
            out[name] = (dur / 365.0).reindex(out.index).fillna(0.0)

        elif kind == 'category_duration_threshold':
            dur = _category_lookup(by_pc, params['category'], 'duration_sum')
            out[name] = (dur >= params['threshold_days']).reindex(out.index).fillna(False).astype('int8')

        elif kind == 'category_value_threshold':
            val = _category_lookup(by_pc, params['category'], 'value_mean')
            out[name] = (val >= params['threshold']).reindex(out.index).fillna(False).astype('int8')

        elif kind == 'spine_threshold':
            col = params['column']
            ser = spine.set_index('patient_id')[col]
            out[name] = (ser >= params['threshold']).reindex(out.index).fillna(False).astype('int8')

        else:
            raise ValueError(f"Unknown BREAST_FEATURES kind: {kind!r} for {name!r}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer C — engagement / code-type-aware
# ─────────────────────────────────────────────────────────────────────────────

def build_layer_c(merged: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Cross-category engagement features. Returns DF indexed by patient_id."""
    out = pd.DataFrame(index=pd.Index(spine['patient_id'].unique(), name='patient_id'))

    # 1. Distinct vocabularies per patient
    out['ENG_N_DISTINCT_VOCABS'] = (
        merged.groupby('patient_id')['code_type'].nunique()
        .reindex(out.index).fillna(0).astype('int16')
    )

    # 2. Distinct categories with ≥1 event
    out['ENG_N_DISTINCT_CATEGORIES'] = (
        merged.groupby('patient_id')['category'].nunique()
        .reindex(out.index).fillna(0).astype('int16')
    )

    # 3. Total events across all categories
    out['ENG_TOTAL_EVENTS'] = (
        merged.groupby('patient_id')['count'].sum()
        .reindex(out.index).fillna(0).astype('int32')
    )

    # 4. Event density: events per active month
    span_days = (
        merged.groupby('patient_id').agg(
            first=('first_event', 'min'),
            last=('last_event', 'max'),
        )
    )
    span_days['active_days'] = (
        pd.to_datetime(span_days['last']) - pd.to_datetime(span_days['first'])
    ).dt.days.clip(lower=1)  # ≥1 to avoid div by zero
    span_days['active_months'] = span_days['active_days'] / 30.4375
    span_days = span_days.reindex(out.index)
    out['ENG_EVENT_DENSITY_OVERALL'] = (
        out['ENG_TOTAL_EVENTS'] / span_days['active_months'].replace(0, np.nan)
    ).fillna(0.0)

    # 5. Most-recent event across all categories — days from anchor_date
    anchor = pd.to_datetime(spine.set_index('patient_id')['anchor_date'])
    last_per_pt = pd.to_datetime(
        merged.groupby('patient_id')['last_event'].max()
    )
    recency = (anchor - last_per_pt).dt.days.reindex(out.index)
    out['ENG_RECENCY_MIN_DAYS'] = recency.fillna(9999).astype('int32')

    # 6. Lab orders without value (observation events where value_mean is NULL)
    obs = merged[merged['event_type'] == 'observation']
    no_val = obs[obs['value_mean'].isna()]
    out['ENG_LAB_ORDERS_NO_VALUE'] = (
        no_val.groupby('patient_id')['count'].sum()
        .reindex(out.index).fillna(0).astype('int32')
    )

    # 7. Patient-level ACCELERATION (EMIS ACCEL_* parity): total recent-half vs
    #    older-half events across ALL curated categories, using the patient's
    #    own-timeline split (count_h1 / count_h2 from 0_extract.py).
    if 'count_h1' in merged.columns and 'count_h2' in merged.columns:
        tot = merged.groupby('patient_id')[['count_h1', 'count_h2']].sum()
        h1 = tot['count_h1'].reindex(out.index).fillna(0.0)
        h2 = tot['count_h2'].reindex(out.index).fillna(0.0)
        out['ENG_ACCEL_H2_OVER_H1']  = (h2 / (h1 + 1.0)).astype('float32')   # ratio (>1 ⇒ accelerating)
        out['ENG_ACCEL_H2_MINUS_H1'] = (h2 - h1).astype('float32')           # net acceleration

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer D — cross-feature interactions
# ─────────────────────────────────────────────────────────────────────────────

def build_layer_d(parent_features: pd.DataFrame) -> pd.DataFrame:
    """Interactions computed from already-built parent columns (Layers B/C/E/F).
    For binary × binary the result is the AND product (kept as int8).
    For numeric × binary the result is the numeric value gated by the binary.

    NOTE: many INTERACTION_PAIRS reference Layer E features (BRCA_E_*), so this
    MUST be called AFTER Layers B, C, E (and F) are built and joined — otherwise
    those pairs silently skip. build_all() passes the combined parent frame.
    """
    out = pd.DataFrame(index=parent_features.index)
    combined = parent_features

    for name, left, right in config.INTERACTION_PAIRS:
        if left not in combined.columns or right not in combined.columns:
            # Skip with a warning rather than crashing — a missing parent feature
            # usually means a category had zero codes in the curated list.
            print(f"  [layer-d] skipping {name}: missing {left if left not in combined.columns else right}")
            continue
        lv = combined[left]
        rv = combined[right]
        # If both are binary-ish (0/1), output is int8 AND-product.
        if lv.dropna().isin([0, 1]).all() and rv.dropna().isin([0, 1]).all():
            out[name] = (lv.fillna(0).astype('int8') & rv.fillna(0).astype('int8'))
        else:
            # Numeric × binary → gated numeric. Numeric × numeric → product.
            out[name] = (lv.fillna(0) * rv.fillna(0)).astype('float32')

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_layer_g(by_pc: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Layer G — lung-style temporal trend / worsening, per GATED category.

    Uses recent-half (count_h2) vs older-half (count_h1) event counts emitted by
    0_extract.py. Built ONLY for config.TREND_CATEGORIES (the STRONG/MEDIUM
    categories) to add temporal resolution where it matters without exploding the
    feature space. Per category emits:
        __trend_h2_over_h1  recent/older activity ratio (>1 ⇒ escalating)
        __is_worsening      1 if more events in the recent half than the older half
        __recent_frac       fraction of events in the recent half  ∈ [0,1]
    Mirrors the lung SNOMED_json_v3 trend logic and EMIS SEQ_*/ACCEL_* features.
    """
    idx = pd.Index(spine['patient_id'].unique(), name='patient_id')
    out = pd.DataFrame(index=idx)
    if 'count_h1' not in by_pc.columns or 'count_h2' not in by_pc.columns:
        print("  [layer-g] count_h1/count_h2 absent (pre-trend extract) — skipping trend features")
        return out
    for cat in config.TREND_CATEGORIES:
        sub = by_pc.loc[by_pc['category'] == cat, ['patient_id', 'count_h1', 'count_h2']]
        if sub.empty:
            h1 = pd.Series(0.0, index=idx); h2 = pd.Series(0.0, index=idx)
        else:
            g = sub.groupby('patient_id')[['count_h1', 'count_h2']].sum()
            h1 = g['count_h1'].reindex(idx).fillna(0.0)
            h2 = g['count_h2'].reindex(idx).fillna(0.0)
        s = _slug(cat)
        # The split is patient-relative (midpoint of each patient's OWN event
        # timeline), so short-history patients still get a within-history trend.
        # h1==0 now only occurs for a single-event / single-date patient — genuinely
        # no trend to measure — so trend is NEUTRAL (0) there; their recent activity
        # is still captured by Layer A count/recency.
        has_baseline = h1 > 0
        out[f'{s}__trend_h2_over_h1'] = (h2 / (h1 + 1.0)).where(has_baseline, 0.0).astype('float32')
        out[f'{s}__is_worsening']     = ((h1 > 0) & (h2 > h1)).astype('int8')
        out[f'{s}__recent_frac']      = (h2 / (h1 + h2).replace(0, np.nan)).where(has_baseline, 0.0).fillna(0.0).astype('float32')
    print(f"  Layer G: {out.shape[1]} trend features across {len(config.TREND_CATEGORIES)} gated categories")
    return out


def build_all(merged: pd.DataFrame, by_pc: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Build Layers B + C + D + E + F + G (extended hand-engineered) and concatenate.
    Returns wide per-patient DF indexed by patient_id."""
    print("Building Layer B (breast-specific hand-engineered)...")
    b = build_layer_b(by_pc, spine)
    print(f"  {b.shape[1]} features")

    print("Building Layer C (engagement / code-type-aware)...")
    c = build_layer_c(merged, spine)
    print(f"  {c.shape[1]} features")

    print("Building Layer E (extended composites + interactions)...")
    e = build_layer_e(by_pc, merged, spine)
    print(f"  {e.shape[1]} features")

    print("Building Layer F (EMIS Blocks 15/22/28 port — first-event-age, recency, age polynomials)...")
    f = build_layer_f(by_pc, spine)
    print(f"  {f.shape[1]} features")

    print("Building Layer G (lung-style trend / worsening, gated categories)...")
    g = build_layer_g(by_pc, spine)
    print(f"  {g.shape[1]} features")

    # Layer D LAST: its interactions reference Layer B/C/E features, so all
    # parents must exist before resolving INTERACTION_PAIRS (else they skip).
    print("Building Layer D (interactions)...")
    parents = b.join(c, how='outer').join(e, how='outer').join(f, how='outer')
    d = build_layer_d(parents)
    print(f"  {d.shape[1]} features")

    extra = (b.join(c, how='outer')
              .join(d, how='outer')
              .join(e, how='outer')
              .join(f, how='outer')
              .join(g, how='outer'))
    print(f"Layer B+C+D+E+F+G total: {extra.shape[1]} features")
    return extra


# ─────────────────────────────────────────────────────────────────────────────
# Layer F — EMIS Blocks 15, 22, 28 ported to Truveta (2026-06-01 v6)
# Blocks 18 (per-category velocity) and 27 (lung-style temporal) require
# event-level dates that Truveta's 0_extract.py aggregates away — skipped here
# until 0_extract.py is changed to emit event-level rows (or a richer aggregation).
# ─────────────────────────────────────────────────────────────────────────────

# Categories targeted (highest-signal subset; mirrors EMIS top-importance blocks).
_KEY_CATEGORIES_F = [
    ('Symptoms - Breast Lump',                  'LUMP'),
    ('Symptoms - Pain',                         'PAIN'),
    ('Symptoms - General',                      'GEN_SX'),
    ('Comorbidity - Benign Breast',             'BENIGN'),
    ('Comorbidity - Atypical Hyperplasia',      'ATYP'),
    ('History - Family Hx Breast Cancer',       'FH_BRCA'),
    ('History - Family Hx Other Cancers',       'FH_OTH'),
    ('History - Hereditary Risk',               'BRCA'),
    ('History - Oophorectomy',                  'OOPH'),
    ('Medication - Hormone Therapy (HRT)',      'HRT'),
    ('Lifestyle - Alcohol Use',                 'ALC'),
    ('Comorbidity - Menopause',                 'MENO'),
]


def build_layer_f(by_pc: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """First-event ages (Block 15) + per-category recency (Block 22) + age polynomials (Block 28).

    Mirrors the EMIS hand-features that consistently rank in the top-10 importance.
    """
    spine = spine.set_index('patient_id')
    idx = spine.index.unique()
    out = pd.DataFrame(index=pd.Index(idx, name='patient_id'))

    anchor = pd.to_datetime(spine['anchor_date']).reindex(out.index)
    age    = pd.to_numeric(spine['patient_age'], errors='coerce').reindex(out.index).fillna(50)

    # ── BLOCK 28 port: Age polynomials + bands ────────────────────────────────
    # EMIS top-importance features were BRCA_AGEP_age_cu and BRCA_AGEP_age_log.
    out['BRCA_F_AGEP_age']          = age.astype('float32')
    out['BRCA_F_AGEP_age_sq']       = (age ** 2).astype('float32')
    out['BRCA_F_AGEP_age_cu']       = (age ** 3).astype('float32')
    out['BRCA_F_AGEP_age_log']      = np.log1p(age).astype('float32')
    out['BRCA_F_AGEP_age_sqrt']     = np.sqrt(age.clip(lower=0)).astype('float32')
    out['BRCA_F_AGEP_age_inv']      = (1.0 / age.clip(lower=1)).astype('float32')
    out['BRCA_F_AGEP_age_decile']   = pd.qcut(age, q=10, labels=False, duplicates='drop').astype('float32').fillna(0)
    out['BRCA_F_AGEP_band_lt40']    = (age < 40).astype('int8')
    out['BRCA_F_AGEP_band_40_50']   = ((age >= 40) & (age < 50)).astype('int8')
    out['BRCA_F_AGEP_band_50_55']   = ((age >= 50) & (age < 55)).astype('int8')
    out['BRCA_F_AGEP_band_55_65']   = ((age >= 55) & (age < 65)).astype('int8')
    out['BRCA_F_AGEP_band_65_75']   = ((age >= 65) & (age < 75)).astype('int8')
    out['BRCA_F_AGEP_band_75plus']  = (age >= 75).astype('int8')

    # ── BLOCK 15 port: First-event-age + months-before-anchor per key category ──
    # For each key category, compute the patient's age at first event, and how
    # many months before the anchor that first event was. NaN-filled with
    # sentinel values for patients without that category.
    for full_cat, short in _KEY_CATEGORIES_F:
        sub = by_pc.loc[by_pc['category'] == full_cat, ['patient_id', 'first_event', 'last_event']]
        if sub.empty:
            # Category has no events at all — emit zero/sentinel cols so schema is stable.
            out[f'BRCA_F_EA_{short}_first_age']       = np.float32(0.0)
            out[f'BRCA_F_EA_{short}_mo_before']       = np.float32(9999.0)
            out[f'BRCA_F_R_{short}_recency_days']     = np.float32(9999.0)
            out[f'BRCA_F_R_{short}_last_age']         = np.float32(0.0)
            continue

        sub = sub.set_index('patient_id')
        first_dt = pd.to_datetime(sub['first_event']).reindex(out.index)
        last_dt  = pd.to_datetime(sub['last_event']).reindex(out.index)

        # Months from first event to anchor (positive = first event is in the past)
        mo_before_first = ((anchor - first_dt).dt.days / 30.4375)
        out[f'BRCA_F_EA_{short}_mo_before'] = mo_before_first.fillna(9999.0).astype('float32')

        # Age at first event
        age_at_first = age - (mo_before_first / 12.0)
        out[f'BRCA_F_EA_{short}_first_age'] = age_at_first.fillna(0.0).clip(lower=0, upper=130).astype('float32')

        # ── BLOCK 22 port: per-category recency ─────────────────────────────────
        recency_days = (anchor - last_dt).dt.days
        out[f'BRCA_F_R_{short}_recency_days'] = recency_days.fillna(9999).astype('float32')

        # Age at last event
        age_at_last = age - (recency_days / 365.25)
        out[f'BRCA_F_R_{short}_last_age'] = age_at_last.fillna(0.0).clip(lower=0, upper=130).astype('float32')

    # ── Composite recency aggregates across all breast symptom categories ──────
    sx_categories = ['Symptoms - Breast Lump', 'Symptoms - Pain', 'Symptoms - General']
    sx_sub = by_pc[by_pc['category'].isin(sx_categories)][['patient_id', 'last_event']]
    if not sx_sub.empty:
        most_recent_sx = pd.to_datetime(sx_sub.groupby('patient_id')['last_event'].max()).reindex(out.index)
        gap = (anchor - most_recent_sx).dt.days / 30.4375
        out['BRCA_F_R_breast_sx_gap_mo']   = gap.fillna(9999).astype('float32')
        out['BRCA_F_R_any_breast_sx_6mo']  = (gap <= 6).fillna(False).astype('int8')
        out['BRCA_F_R_any_breast_sx_12mo'] = (gap <= 12).fillna(False).astype('int8')
    else:
        out['BRCA_F_R_breast_sx_gap_mo']   = np.float32(9999.0)
        out['BRCA_F_R_any_breast_sx_6mo']  = np.int8(0)
        out['BRCA_F_R_any_breast_sx_12mo'] = np.int8(0)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer E — EXTENDED hand-engineered composites & interactions (2026-06-01)
# Mirrors EMIS Blocks 5e-25: symptom burden, age×clinical, triples,
# hereditary load, oophorectomy protection, BMI risk, co-occurrence,
# recency, super-flags. Uses Truveta category naming convention.
# ─────────────────────────────────────────────────────────────────────────────

def _has_cat(by_pc: pd.DataFrame, category: str, index) -> pd.Series:
    """Binary 0/1 per patient: does this patient have ≥1 event in category?"""
    sub = by_pc.loc[by_pc['category'] == category, 'patient_id']
    return pd.Series(index.isin(sub.values).astype(int), index=index)


def _cat_count(by_pc: pd.DataFrame, category: str, index) -> pd.Series:
    """Per-patient count for category."""
    sub = by_pc.loc[by_pc['category'] == category, ['patient_id', 'count']]
    if sub.empty:
        return pd.Series(0, index=index, dtype='int32')
    return sub.set_index('patient_id')['count'].reindex(index, fill_value=0).astype('int32')


def build_layer_e(by_pc: pd.DataFrame, merged: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Extended composites + age interactions + triple interactions + super-flags."""
    spine = spine.set_index('patient_id')
    idx = spine.index.unique()
    out = pd.DataFrame(index=pd.Index(idx, name='patient_id'))

    age = pd.to_numeric(spine['patient_age'], errors='coerce').reindex(out.index).fillna(50).astype(int)
    age40 = (age >= 40).astype(int)
    age50 = (age >= 50).astype(int)
    age65 = (age >= 65).astype(int)
    age75 = (age >= 75).astype(int)
    age_under_40 = (age < 40).astype(int)
    postmeno = (age >= 55).astype(int)

    # ── Has-flags for new categories (Truveta naming) ─────────────────────
    has_lump   = _has_cat(by_pc, 'Symptoms - Breast Lump',   out.index)
    has_pain   = _has_cat(by_pc, 'Symptoms - Pain',          out.index)
    has_nipple = _has_cat(by_pc, 'Symptoms - General',       out.index)
    has_benign = _has_cat(by_pc, 'Comorbidity - Benign Breast', out.index)
    has_atyp   = _has_cat(by_pc, 'Comorbidity - Atypical Hyperplasia', out.index)
    has_famhx_b = _has_cat(by_pc, 'History - Family Hx Breast Cancer', out.index)
    has_famhx_o = _has_cat(by_pc, 'History - Family Hx Other Cancers', out.index)
    has_brca    = _has_cat(by_pc, 'History - Hereditary Risk', out.index)
    has_oophor  = _has_cat(by_pc, 'History - Oophorectomy', out.index)
    has_ovar_fail = _has_cat(by_pc, 'Comorbidity - Ovarian Failure', out.index)
    has_ovar_cond = _has_cat(by_pc, 'Comorbidity - Ovarian Conditions', out.index)
    has_meno    = _has_cat(by_pc, 'Comorbidity - Menopause', out.index)
    has_screen_mammo = _has_cat(by_pc, 'Screening - Mammography', out.index)
    has_radn    = _has_cat(by_pc, 'History - Radiation Exposure', out.index)
    has_hrt     = _has_cat(by_pc, 'Medication - Hormone Therapy (HRT)', out.index)
    has_alcohol = _has_cat(by_pc, 'Lifestyle - Alcohol Use', out.index)

    # ── BLOCK E1: Has-flags + symptom burden composites ──────────────────
    out['BRCA_E_has_lump']     = has_lump
    out['BRCA_E_has_pain']     = has_pain
    out['BRCA_E_has_nipple']   = has_nipple
    out['BRCA_E_has_benign']   = has_benign
    out['BRCA_E_has_atypical'] = has_atyp
    out['BRCA_E_has_famhx_breast'] = has_famhx_b
    out['BRCA_E_has_famhx_other']  = has_famhx_o
    out['BRCA_E_has_brca_test']    = has_brca
    out['BRCA_E_has_oophorectomy'] = has_oophor
    out['BRCA_E_has_ovarian_failure']    = has_ovar_fail
    out['BRCA_E_has_ovarian_condition']  = has_ovar_cond
    out['BRCA_E_has_menopause']    = has_meno
    out['BRCA_E_has_screen_mammo'] = has_screen_mammo
    out['BRCA_E_has_radiation']    = has_radn

    sx_count = (has_lump + has_pain + has_nipple).astype(int)
    out['BRCA_E_breast_sx_burden']  = sx_count
    out['BRCA_E_any_breast_sx']     = (sx_count >= 1).astype(int)
    out['BRCA_E_multi_breast_sx']   = (sx_count >= 2).astype(int)

    # ── BLOCK E2: Hereditary load ─────────────────────────────────────────
    hereditary_load = (has_brca + has_famhx_b + has_famhx_o + has_atyp).astype(int)
    out['BRCA_E_hereditary_load']      = hereditary_load
    out['BRCA_E_strong_hereditary']    = (hereditary_load >= 1).astype(int)
    out['BRCA_E_multi_hereditary']     = (hereditary_load >= 2).astype(int)

    # ── BLOCK E3: Age × Clinical interactions ─────────────────────────────
    # Age 40 features (captures premenopausal / BRCA-carrier age window)
    out['BRCA_E_age40']               = age40
    out['BRCA_E_under_40']            = age_under_40
    out['BRCA_E_lump_x_age40']        = (has_lump * age40)
    out['BRCA_E_lump_x_under_40']     = (has_lump * age_under_40)
    out['BRCA_E_famhx_x_age40']       = (has_famhx_b * age40)
    out['BRCA_E_famhx_x_under_40']    = (has_famhx_b * age_under_40)
    out['BRCA_E_brca_x_age40']        = (has_brca * age40)
    out['BRCA_E_brca_x_under_40']     = (has_brca * age_under_40)
    out['BRCA_E_nipple_x_age40']      = (has_nipple * age40)
    out['BRCA_E_atyp_x_age40']        = (has_atyp * age40)
    out['BRCA_E_atyp_x_under_40']     = (has_atyp * age_under_40)
    out['BRCA_E_lump_x_age50']        = (has_lump * age50)
    out['BRCA_E_lump_x_age65']        = (has_lump * age65)
    out['BRCA_E_lump_x_age75']        = (has_lump * age75)
    out['BRCA_E_nipple_x_age50']      = (has_nipple * age50)
    out['BRCA_E_benign_x_age50']      = (has_benign * age50)
    out['BRCA_E_famhx_x_age50']       = (has_famhx_b * age50)
    out['BRCA_E_famhx_other_x_age50'] = (has_famhx_o * age50)
    out['BRCA_E_brca_x_age50']        = (has_brca * age50)
    out['BRCA_E_atypical_x_age50']    = (has_atyp * age50)
    out['BRCA_E_hrt_x_age50']         = (has_hrt * age50)
    out['BRCA_E_hrt_x_age65']         = (has_hrt * age65)
    out['BRCA_E_meno_x_age50']        = (has_meno * age50)
    out['BRCA_E_radn_x_age50']        = (has_radn * age50)
    out['BRCA_E_alcohol_x_age50']     = (has_alcohol * age50)

    # ── BLOCK E4: Triple interactions ─────────────────────────────────────
    out['BRCA_E_tri_lump_postmeno_famhx']   = (has_lump * postmeno * has_famhx_b)
    out['BRCA_E_tri_lump_postmeno_hrt']     = (has_lump * postmeno * has_hrt)
    out['BRCA_E_tri_lump_age65_hereditary'] = (has_lump * age65 * out['BRCA_E_strong_hereditary'])
    out['BRCA_E_tri_brca_famhx_age50']      = (has_brca * has_famhx_b * age50)
    out['BRCA_E_tri_atyp_famhx_age50']      = (has_atyp * has_famhx_b * age50)
    out['BRCA_E_tri_hrt_postmeno_alcohol']  = (has_hrt * postmeno * has_alcohol)
    out['BRCA_E_tri_multi_sx_age50_famhx']  = (out['BRCA_E_multi_breast_sx'] * age50 * has_famhx_b)
    out['BRCA_E_tri_nipple_postmeno_famhx'] = (has_nipple * postmeno * has_famhx_b)
    out['BRCA_E_tri_benign_famhx_postmeno'] = (has_benign * has_famhx_b * postmeno)
    out['BRCA_E_tri_lump_pain_nipple']      = (has_lump * has_pain * has_nipple)

    # ── BLOCK E5: Cross-category co-occurrence ────────────────────────────
    breast_indicators = (has_lump + has_pain + has_nipple + has_benign +
                          has_famhx_b + has_famhx_o + has_brca + has_atyp).astype(int)
    out['BRCA_E_breast_indicators']         = breast_indicators
    out['BRCA_E_2plus_breast_indicators']   = (breast_indicators >= 2).astype(int)
    out['BRCA_E_3plus_breast_indicators']   = (breast_indicators >= 3).astype(int)
    out['BRCA_E_4plus_breast_indicators']   = (breast_indicators >= 4).astype(int)
    out['BRCA_E_sx_and_hereditary']         = ((sx_count >= 1) & (hereditary_load >= 1)).astype(int)
    out['BRCA_E_sx_and_hrt']                = ((sx_count >= 1) & (has_hrt == 1)).astype(int)
    out['BRCA_E_benign_and_hereditary']     = ((has_benign == 1) & (hereditary_load >= 1)).astype(int)
    out['BRCA_E_atyp_and_famhx']            = ((has_atyp == 1) & (has_famhx_b == 1)).astype(int)

    # ── BLOCK E6: Cumulative SUPER risk flags ────────────────────────────
    out['BRCA_E_super_high_alarm'] = (
        (has_lump == 1) & (age50 == 1) &
        ((postmeno | has_famhx_b | has_brca) >= 1)
    ).astype(int)
    out['BRCA_E_super_dual_indicator_postmeno'] = (
        (breast_indicators >= 2) & (postmeno == 1)
    ).astype(int)
    out['BRCA_E_super_strong_famhx_sx'] = (
        (hereditary_load >= 1) & ((has_lump + has_nipple) >= 1)
    ).astype(int)
    out['BRCA_E_super_atyp_high_risk'] = (
        (has_atyp == 1) & ((has_famhx_b + has_brca) >= 1)
    ).astype(int)
    out['BRCA_E_super_no_protective_high_risk'] = (
        (has_oophor == 0) & (breast_indicators >= 2)
    ).astype(int)

    # ── BLOCK E7: Risk SCORE composite (Truveta version) ──────────────────
    risk_score = pd.Series(0.0, index=out.index)
    risk_score = risk_score + age50.astype(float) * 1.0
    risk_score = risk_score + age65.astype(float) * 1.0
    risk_score = risk_score + age75.astype(float) * 0.5
    risk_score = risk_score + has_brca.astype(float) * 3.0
    risk_score = risk_score + has_famhx_b.astype(float) * 2.0
    risk_score = risk_score + has_famhx_o.astype(float) * 0.5
    risk_score = risk_score + has_atyp.astype(float) * 2.5
    risk_score = risk_score + has_lump.astype(float) * 2.5
    risk_score = risk_score + has_nipple.astype(float) * 1.5
    risk_score = risk_score + has_pain.astype(float) * 0.5
    risk_score = risk_score + has_benign.astype(float) * 0.3
    risk_score = risk_score + has_hrt.astype(float) * 1.0
    risk_score = risk_score + has_radn.astype(float) * 1.5
    risk_score = risk_score + out['BRCA_E_multi_breast_sx'].astype(float) * 1.0
    risk_score = risk_score - has_oophor.astype(float) * 1.5  # PROTECTIVE
    risk_score = risk_score - has_meno.astype(float) * 0.3   # mild protective if early
    out['BRCA_E_RISK_score']      = risk_score.astype('float32')
    out['BRCA_E_RISK_high']       = (risk_score >= 4.0).astype(int)
    out['BRCA_E_RISK_very_high']  = (risk_score >= 6.0).astype(int)

    # ── BLOCK E8: Engagement-normalized signals ───────────────────────────
    total_obs = merged.groupby('patient_id')['count'].sum().reindex(out.index, fill_value=0).clip(lower=1).astype('float32')
    lump_ct   = _cat_count(by_pc, 'Symptoms - Breast Lump', out.index).astype('float32')
    out['BRCA_E_norm_lump_per_total']    = (lump_ct / total_obs).astype('float32')
    out['BRCA_E_log_total_events']       = np.log1p(total_obs).astype('float32')
    out['BRCA_E_sqrt_breast_sx_burden']  = np.sqrt(sx_count).astype('float32')
    out['BRCA_E_sqrt_breast_indicators'] = np.sqrt(breast_indicators).astype('float32')

    return out
