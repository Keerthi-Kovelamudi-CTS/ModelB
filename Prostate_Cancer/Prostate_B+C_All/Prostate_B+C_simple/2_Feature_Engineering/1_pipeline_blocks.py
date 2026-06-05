# ═══════════════════════════════════════════════════════════════
# GENERIC FEATURE ENGINEERING PIPELINE
# Config-driven — works for any cancer type.
# Functions only — no top-level execution.
# Called by run_pipeline.py.
# ═══════════════════════════════════════════════════════════════

import logging
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import re
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

__all__ = [
    'build_clinical_features',
    'build_medication_features',
    'build_interaction_features',
    'build_advanced_features',
    'extract_maximum_features',
    'build_new_signal_features',
    'build_trend_features',
]


def _get_all_symptom_cats(cfg):
    """Flatten all categories from CLUSTER_DEFINITIONS into one deduplicated list."""
    cats = []
    for _, cat_list in cfg.CLUSTER_DEFINITIONS:
        cats.extend(cat_list)
    return list(set(cats))


# ═══════════════════════════════════════════════════════════════
# CLINICAL-BIN HELPERS (replaces A/B halves with acute/subacute/recent_past/historical)
# ═══════════════════════════════════════════════════════════════

def _bin_names(cfg):
    """Return the clinical bin names in order (acute → historical)."""
    return getattr(cfg, 'CLINICAL_BIN_NAMES', ['acute', 'subacute', 'recent_past', 'historical'])


def _adjacent_pairs(cfg):
    """Return adjacent-bin pairs for delta/acceleration features (recent → earlier)."""
    return getattr(cfg, 'ADJACENT_BIN_PAIRS', [
        ('acute',       'subacute'),
        ('subacute',    'recent_past'),
        ('recent_past', 'historical'),
    ])


def _all_bins_mask(df, cfg):
    """Boolean mask selecting events in ANY clinical bin (i.e. with a non-null TIME_WINDOW)."""
    return df['TIME_WINDOW'].isin(_bin_names(cfg))


def _per_bin_counts(df, group_cols, cfg):
    """Compute per-bin event counts grouped by `group_cols`.

    Returns: dict[bin_name → DataFrame of counts indexed by group_cols].
    DataFrames are wide (unstacked on last group_col).
    """
    out = {}
    for b in _bin_names(cfg):
        sub = df[df['TIME_WINDOW'] == b]
        if len(sub) == 0:
            continue
        out[b] = sub.groupby(group_cols).size()
        if isinstance(group_cols, list) and len(group_cols) > 1:
            out[b] = out[b].unstack(fill_value=0)
    return out


# ═══════════════════════════════════════════════════════════════
# STEP 4a: BASE CLINICAL FEATURES
# ═══════════════════════════════════════════════════════════════

def build_clinical_features(clin_df, cfg, window_name='12mo'):
    """Build patient-level features from observation/lab data."""
    df = clin_df.copy()
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')
    # Include ethnicity columns (added in preprocess OUT_COLS) so they can flow
    # into the per-patient feature matrix via encoded demographic features below.
    _demo_cols_src = ['PATIENT_GUID', 'LABEL', 'SEX', 'AGE_AT_INDEX', 'INDEX_DATE']
    for _c in ('PATIENT_ETHNICITY_16', 'PATIENT_ETHNICITY_6'):
        if _c in df.columns:
            _demo_cols_src.append(_c)
    patients = df[_demo_cols_src].drop_duplicates(subset=['PATIENT_GUID'])

    # 4a. DEMOGRAPHICS — include encoded SEX + ethnicity so they survive cleanup.
    # LOAD-OR-FIT: if encoder_mappings.json already exists (e.g. training run
    # already happened, this is a holdout-FE or inference-FE pass), load it
    # and apply the same integer mapping. Otherwise fit fresh + save.
    # This keeps integer indices stable across training / holdout / inference.
    demo = patients[['PATIENT_GUID', 'LABEL', 'AGE_AT_INDEX']].copy()
    demo['AGE_BAND'] = (demo['AGE_AT_INDEX'] // 5) * 5

    import json as _json
    _enc_path = cfg.FE_RESULTS / window_name.lower() / 'encoder_mappings.json'
    _encoder_mappings = None
    try:
        if _enc_path.exists():
            with open(_enc_path, encoding='utf-8') as _f:
                _encoder_mappings = _json.load(_f)
    except Exception:
        _encoder_mappings = None

    if _encoder_mappings is None:
        # First run (training) — fit deterministic mappings + save.
        _encoder_mappings = {'sex_male': {'M': 1, 'F': 0}}
        for _eth_col, _key in (
            ('PATIENT_ETHNICITY_16', 'ethnicity_16_enc'),
            ('PATIENT_ETHNICITY_6',  'ethnicity_6_enc'),
        ):
            if _eth_col in patients.columns:
                _cats = sorted(patients[_eth_col].dropna().astype(str).unique())
                _encoder_mappings[_key] = {v: i for i, v in enumerate(_cats)}
        try:
            _enc_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_enc_path, 'w', encoding='utf-8') as _f:
                _json.dump(_encoder_mappings, _f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # non-fatal — re-fits will be lossy but won't crash

    # Apply mappings (whether loaded or freshly fitted). For Prostate/Ovarian
    # SEX_MALE is constant → step 5d zero-variance drop removes it later.
    demo['SEX_MALE'] = patients['SEX'].astype(str).str.upper().map(
        _encoder_mappings.get('sex_male', {'M': 1, 'F': 0})
    ).fillna(-1).astype(int)
    for _eth_col, _out_col, _key in (
        ('PATIENT_ETHNICITY_16', 'ETHNICITY_16_ENC', 'ethnicity_16_enc'),
        ('PATIENT_ETHNICITY_6',  'ETHNICITY_6_ENC',  'ethnicity_6_enc'),
    ):
        if _eth_col in patients.columns:
            demo[_out_col] = patients[_eth_col].astype(str).map(
                _encoder_mappings.get(_key, {})
            ).fillna(-1).astype(int)

    # 4b. OBSERVATION FEATURES — per category × per clinical bin
    obs_df = df.copy()
    obs_categories = obs_df['CATEGORY'].unique()
    bin_names = _bin_names(cfg)         # acute, subacute, recent_past, historical
    adj_pairs = _adjacent_pairs(cfg)    # [(acute,subacute), (subacute,recent_past), ...]

    # Per-bin counts: one DataFrame per bin, wide (one column per category)
    obs_per_bin = {}
    for b in bin_names:
        sub = obs_df[obs_df['TIME_WINDOW'] == b]
        if len(sub) == 0:
            continue
        per_b = sub.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
        per_b.columns = [f"OBS_{c}_count_{b}" for c in per_b.columns]
        obs_per_bin[b] = per_b

    # Total across all bins
    obs_total = obs_df[_all_bins_mask(obs_df, cfg)].groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    obs_total.columns = [f"OBS_{c}_count_total" for c in obs_total.columns]

    obs_has_ever = (obs_total > 0).astype(int)
    obs_has_ever.columns = [c.replace('_count_total', '_has_ever') for c in obs_has_ever.columns]

    # Adjacent-bin acceleration: e.g. count_acute > count_subacute → patient's activity is
    # accelerating in the most recent bin. Captures "things getting worse recently".
    obs_accel = pd.DataFrame(index=obs_total.index)
    for cat in obs_categories:
        for recent_bin, earlier_bin in adj_pairs:
            col_recent  = f"OBS_{cat}_count_{recent_bin}"
            col_earlier = f"OBS_{cat}_count_{earlier_bin}"
            df_r = obs_per_bin.get(recent_bin)
            df_e = obs_per_bin.get(earlier_bin)
            if df_r is None or df_e is None or col_recent not in df_r.columns or col_earlier not in df_e.columns:
                continue
            merged = df_r[[col_recent]].join(df_e[[col_earlier]], how='outer').fillna(0)
            obs_accel[f"OBS_{cat}_accel_{recent_bin}_vs_{earlier_bin}"] = (
                merged[col_recent] > merged[col_earlier]
            ).astype(int)

    # "New in acute" — category that appeared in the most-recent bin but NOT in any earlier bin.
    # Clinically: "first-time presentation of this symptom in the recent window."
    # No-op in the no-bins variant (single bin → no "earlier" bins to compare against).
    obs_new_acute = pd.DataFrame(index=obs_total.index)
    _bins = _bin_names(cfg)
    recent_bin_name = _bins[0] if _bins else None
    earlier_bin_names = _bins[1:] if len(_bins) > 1 else []
    df_acute = obs_per_bin.get(recent_bin_name) if recent_bin_name else None
    if df_acute is not None and earlier_bin_names:
        for cat in obs_categories:
            col_acute = f"OBS_{cat}_count_{recent_bin_name}"
            if col_acute not in df_acute.columns:
                continue
            # Sum across all EARLIER bins for this category
            earlier_sum = pd.Series(0, index=df_acute.index)
            for earlier_b in earlier_bin_names:
                df_e = obs_per_bin.get(earlier_b)
                if df_e is None:
                    continue
                col_e = f"OBS_{cat}_count_{earlier_b}"
                if col_e in df_e.columns:
                    earlier_sum = earlier_sum.add(df_e[col_e].reindex(df_acute.index, fill_value=0), fill_value=0)
            obs_new_acute[f"OBS_{cat}_new_in_{recent_bin_name}"] = (
                (df_acute[col_acute] > 0) & (earlier_sum == 0)
            ).astype(int)

    # 4b-extra. ABBAS-STYLE per-category temporal features (FREQUENCY_PER_YEAR + TIME_SPAN_YEARS)
    # — captures chronicity vs acute presentation. Computed over ALL bins.
    obs_AB_dated = obs_df[_all_bins_mask(obs_df, cfg)].copy()
    obs_AB_dated['EVENT_DATE'] = pd.to_datetime(obs_AB_dated['EVENT_DATE'], errors='coerce')
    obs_AB_dated = obs_AB_dated.dropna(subset=['EVENT_DATE'])
    obs_temporal = pd.DataFrame(index=obs_total.index)
    if len(obs_AB_dated) > 0:
        # Per (patient, category): first/last event date
        gb = obs_AB_dated.groupby(['PATIENT_GUID', 'CATEGORY'])['EVENT_DATE']
        first_dt = gb.min().unstack(fill_value=pd.NaT)
        last_dt = gb.max().unstack(fill_value=pd.NaT)
        for cat in obs_categories:
            if cat not in first_dt.columns or cat not in last_dt.columns:
                continue
            span_days = (last_dt[cat] - first_dt[cat]).dt.days.fillna(0).clip(lower=0)
            span_years = (span_days / 365.25).reindex(obs_total.index, fill_value=0).astype(float)
            count_total = obs_total.get(f"OBS_{cat}_count_total", pd.Series(0, index=obs_total.index)).astype(float)
            # FREQUENCY_PER_YEAR: events per year over the patient's category history span
            # If span is 0 (single event), frequency = count itself (treated as "1 year of activity")
            freq = count_total / span_years.clip(lower=1.0 / 365.25)
            obs_temporal[f"OBS_{cat}_TIME_SPAN_YEARS"] = span_years
            obs_temporal[f"OBS_{cat}_FREQUENCY_PER_YEAR"] = freq.clip(upper=10000)  # cap extreme values

    # 4c. LAB VALUE FEATURES — per category
    lab_df = df[df['VALUE'].notna() & df['CATEGORY'].isin(cfg.LAB_CATEGORIES)].copy()
    lab_categories = lab_df['CATEGORY'].unique()
    lab_features = pd.DataFrame()
    bad_dir = getattr(cfg, 'LAB_BAD_DIRECTION', {})
    worsening_rules = getattr(cfg, 'LAB_WORSENING_RULES', {})

    for cat in lab_categories:
        cat_df = lab_df[lab_df['CATEGORY'] == cat]
        stats = cat_df.groupby('PATIENT_GUID')['VALUE'].agg(
            **{f"LAB_{cat}_mean": 'mean', f"LAB_{cat}_min": 'min', f"LAB_{cat}_max": 'max',
               f"LAB_{cat}_std": 'std', f"LAB_{cat}_count": 'count'}
        )
        last_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
        last_val.name = f"LAB_{cat}_last"
        first_val = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].first()
        first_val.name = f"LAB_{cat}_first"
        # Per-bin lab means: mean_acute, mean_subacute, mean_recent_past, mean_historical
        per_bin_means = {}
        for b in bin_names:
            m = cat_df[cat_df['TIME_WINDOW'] == b].groupby('PATIENT_GUID')['VALUE'].mean()
            m.name = f"LAB_{cat}_mean_{b}"
            per_bin_means[b] = m

        # Adjacent-bin deltas: e.g. mean_acute - mean_subacute (acute spike)
        bin_deltas = []
        bin_pct_changes = []
        for recent_bin, earlier_bin in adj_pairs:
            mr = per_bin_means.get(recent_bin)
            me = per_bin_means.get(earlier_bin)
            if mr is None or me is None:
                continue
            d = mr.subtract(me)
            d.name = f"LAB_{cat}_delta_{recent_bin}_minus_{earlier_bin}"
            bin_deltas.append(d)
            pc = (mr.subtract(me) / me.replace(0, np.nan).abs()).replace([np.inf, -np.inf], np.nan)
            pc.name = f"LAB_{cat}_pct_change_{recent_bin}_vs_{earlier_bin}"
            bin_pct_changes.append(pc)

        # Slope + Pearson r of the linear fit. r tells us how *consistent* the trend is —
        # a big slope with low r is noise; a moderate slope with high r is real signal.
        # Tree models can use r as a confidence gate on the slope feature.
        def _calc_slope_and_r(group):
            if len(group) < 2:
                return pd.Series({'slope': np.nan, 'r': np.nan})
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
            y = group['VALUE'].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 2:
                return pd.Series({'slope': np.nan, 'r': np.nan})
            x, y = x[mask], y[mask]
            if x.max() == x.min():
                return pd.Series({'slope': np.nan, 'r': np.nan})
            try:
                s = float(np.polyfit(x, y, 1)[0])
                xm, ym = x.mean(), y.mean()
                den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
                r = float(((x - xm) * (y - ym)).sum() / den) if den > 0 else np.nan
                return pd.Series({'slope': s, 'r': r})
            except Exception:
                return pd.Series({'slope': np.nan, 'r': np.nan})

        slope_r = cat_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID').apply(_calc_slope_and_r)
        slope = slope_r['slope']
        slope.name = f"LAB_{cat}_slope"
        trend_r = slope_r['r']
        trend_r.name = f"LAB_{cat}_trend_r"
        has_ever = (stats[f"LAB_{cat}_count"] > 0).astype(int)
        has_ever.name = f"LAB_{cat}_has_ever"

        # Missingness indicator: 1 if patient has NO values for this lab category.
        # Distinguishes "true zero" from "not measured" — tree models often use this strongly.
        is_missing = (stats[f"LAB_{cat}_count"].fillna(0) == 0).astype(int)
        is_missing.name = f"LAB_{cat}_is_missing"

        # Worsening flag — pluggable rule takes precedence over LAB_BAD_DIRECTION.
        # Custom rule signature: rule(slope, first, last, mean, max, count) -> 0/1 Series.
        rule = worsening_rules.get(cat)
        direction = bad_dir.get(cat)
        worsening = None
        if callable(rule):
            try:
                rule_out = rule(
                    slope, first_val, last_val,
                    stats[f"LAB_{cat}_mean"], stats[f"LAB_{cat}_max"], stats[f"LAB_{cat}_count"],
                )
                worsening = pd.Series(rule_out, index=slope.index).astype(int)
            except Exception:
                df_for_rule = pd.DataFrame({
                    'slope': slope, 'first': first_val, 'last': last_val,
                    'mean': stats[f"LAB_{cat}_mean"],
                    'max':  stats[f"LAB_{cat}_max"],
                    'count': stats[f"LAB_{cat}_count"],
                })
                worsening = df_for_rule.apply(
                    lambda r: int(rule(r['slope'], r['first'], r['last'],
                                       r['mean'], r['max'], r['count'])),
                    axis=1,
                ).astype(int)
            worsening.name = f"LAB_{cat}_worsening"
        elif direction == 'up':
            worsening = (slope > 0).astype(int)
            worsening.name = f"LAB_{cat}_worsening"
        elif direction == 'down':
            worsening = (slope < 0).astype(int)
            worsening.name = f"LAB_{cat}_worsening"

        parts = [stats, last_val, first_val]
        parts.extend(per_bin_means.values())     # 4 per-bin means
        parts.extend(bin_deltas)                  # 3 adjacent-bin deltas
        parts.extend(bin_pct_changes)             # 3 adjacent-bin pct changes
        parts.extend([slope, trend_r, has_ever, is_missing])
        if worsening is not None:
            parts.append(worsening)
        cat_features = pd.concat(parts, axis=1)
        lab_features = cat_features if lab_features.empty else lab_features.join(cat_features, how='outer')

    # 4f. INVESTIGATION PATTERN FEATURES — per clinical bin
    inv_cats = cfg.INVESTIGATION_CATEGORIES
    inv_df = obs_df[obs_df['CATEGORY'].isin(inv_cats)]

    inv_per_bin = {}
    for b in bin_names:
        s = inv_df[inv_df['TIME_WINDOW'] == b].groupby('PATIENT_GUID').size()
        s.name = f"INV_count_{b}"
        inv_per_bin[b] = s
    inv_total = inv_df[_all_bins_mask(inv_df, cfg)].groupby('PATIENT_GUID').size()
    inv_total.name = 'INV_count_total'
    inv_unique_types = inv_df[_all_bins_mask(inv_df, cfg)].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    inv_unique_types.name = 'INV_unique_types'
    # Acceleration: any adjacent-bin pair where the recent bin > earlier bin
    inv_accel_df = pd.DataFrame()
    for recent_bin, earlier_bin in adj_pairs:
        sr = inv_per_bin.get(recent_bin); se = inv_per_bin.get(earlier_bin)
        if sr is None or se is None:
            continue
        merged = pd.concat([sr, se], axis=1).fillna(0)
        inv_accel_df[f'INV_accel_{recent_bin}_vs_{earlier_bin}'] = (
            merged[f"INV_count_{recent_bin}"] > merged[f"INV_count_{earlier_bin}"]
        ).astype(int)

    imaging_inv = inv_df[inv_df['CATEGORY'] == inv_cats[0]] if inv_cats else inv_df.head(0)
    imaging_per_bin = {}
    for b in bin_names:
        s = imaging_inv[imaging_inv['TIME_WINDOW'] == b].groupby('PATIENT_GUID').size()
        s.name = f"INV_imaging_count_{b}"
        imaging_per_bin[b] = s
    imaging_total = imaging_inv[_all_bins_mask(imaging_inv, cfg)].groupby('PATIENT_GUID').size()
    imaging_total.name = 'INV_imaging_total'

    # 4g. AGGREGATE CLINICAL FEATURES — per clinical bin
    agg_df = df[_all_bins_mask(df, cfg)].copy()
    agg_unique_cats = agg_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    agg_unique_cats.name = 'AGG_unique_categories'
    agg_total_events = agg_df.groupby('PATIENT_GUID').size()
    agg_total_events.name = 'AGG_total_events'
    agg_events_per_bin = {}
    for b in bin_names:
        s = agg_df[agg_df['TIME_WINDOW'] == b].groupby('PATIENT_GUID').size()
        s.name = f"AGG_events_{b}"
        agg_events_per_bin[b] = s
    agg_unique_codes = agg_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
    agg_unique_codes.name = 'AGG_unique_code_ids'

    symptom_cats = cfg.SYMPTOM_CATEGORIES
    symptom_df = obs_df[obs_df['CATEGORY'].isin(symptom_cats)]
    symptom_per_bin = {}
    for b in bin_names:
        s = symptom_df[symptom_df['TIME_WINDOW'] == b].groupby('PATIENT_GUID').size()
        s.name = f"AGG_symptom_count_{b}"
        symptom_per_bin[b] = s
    symptom_total = symptom_df[_all_bins_mask(symptom_df, cfg)].groupby('PATIENT_GUID').size()
    symptom_total.name = 'AGG_symptom_count_total'
    symptom_unique = symptom_df[_all_bins_mask(symptom_df, cfg)].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    symptom_unique.name = 'AGG_symptom_unique_categories'
    symptom_accel_df = pd.DataFrame()
    for recent_bin, earlier_bin in adj_pairs:
        sr = symptom_per_bin.get(recent_bin); se = symptom_per_bin.get(earlier_bin)
        if sr is None or se is None:
            continue
        merged = pd.concat([sr, se], axis=1).fillna(0)
        symptom_accel_df[f'AGG_symptom_accel_{recent_bin}_vs_{earlier_bin}'] = (
            merged[f"AGG_symptom_count_{recent_bin}"] > merged[f"AGG_symptom_count_{earlier_bin}"]
        ).astype(int)

    # "New symptom categories in recent bin" — categories appearing in the recent bin but in NO earlier bin
    # No-op in the no-bins variant (single bin → no "earlier" bins to compare against).
    _sbins = _bin_names(cfg)
    _recent_b = _sbins[0] if _sbins else None
    _earlier_bs = _sbins[1:] if len(_sbins) > 1 else []
    if _recent_b is not None and _earlier_bs:
        symptom_cats_acute = symptom_df[symptom_df['TIME_WINDOW'] == _recent_b].groupby('PATIENT_GUID')['CATEGORY'].apply(set)
        earlier_cats = (
            symptom_df[symptom_df['TIME_WINDOW'].isin(_earlier_bs)]
            .groupby('PATIENT_GUID')['CATEGORY'].apply(set)
        )
        symptom_new_acute_count = pd.concat([symptom_cats_acute, earlier_cats], axis=1, keys=['acute', 'earlier']).apply(
            lambda row: len(row['acute'] - row['earlier']) if pd.notna(row['acute']) and pd.notna(row['earlier'])
            else (len(row['acute']) if pd.notna(row['acute']) else 0), axis=1
        )
        symptom_new_acute_count.name = f'AGG_new_symptom_cats_in_{_recent_b}'
    else:
        symptom_new_acute_count = pd.Series(dtype='int64', name='AGG_new_symptom_cats_in_recent')

    # 4h. TEMPORAL FEATURES
    temporal_df = agg_df.groupby('PATIENT_GUID')['EVENT_DATE'].agg(['min', 'max'])
    temporal_df['TEMP_event_span_days'] = (temporal_df['max'] - temporal_df['min']).dt.days
    temporal_df = temporal_df[['TEMP_event_span_days']]

    last_event = agg_df.groupby('PATIENT_GUID').agg(
        last_event_date=('EVENT_DATE', 'max'), index_date=('INDEX_DATE', 'first')
    )
    last_event['TEMP_days_last_event_to_index'] = (last_event['index_date'] - last_event['last_event_date']).dt.days

    # MERGE ALL CLINICAL FEATURES
    features = demo.set_index('PATIENT_GUID')

    # Per-bin observation counts (acute/subacute/recent_past/historical)
    for b, per_b in obs_per_bin.items():
        features = features.join(per_b, how='left')
    for feat_df in [obs_total, obs_has_ever, obs_accel, obs_new_acute, obs_temporal]:
        if not feat_df.empty:
            features = features.join(feat_df, how='left')
    if not lab_features.empty:
        features = features.join(lab_features, how='left')
    # Investigation features per bin
    for s in inv_per_bin.values():
        features = features.join(s, how='left')
    features = features.join(inv_total, how='left').join(inv_unique_types, how='left')
    if not inv_accel_df.empty:
        features = features.join(inv_accel_df, how='left')
    for s in imaging_per_bin.values():
        features = features.join(s, how='left')
    features = features.join(imaging_total, how='left')
    # Aggregate features
    for s in [agg_unique_cats, agg_total_events, agg_unique_codes,
              symptom_total, symptom_unique, symptom_new_acute_count]:
        features = features.join(s, how='left')
    for s in agg_events_per_bin.values():
        features = features.join(s, how='left')
    for s in symptom_per_bin.values():
        features = features.join(s, how='left')
    if not symptom_accel_df.empty:
        features = features.join(symptom_accel_df, how='left')
    features = features.join(temporal_df, how='left')
    features = features.join(last_event[['TEMP_days_last_event_to_index']], how='left')

    count_cols = [c for c in features.columns if any(x in c for x in
                  ['_count_', '_has_ever', '_flag', '_accel', '_new_in_acute', '_total', 'AGG_', 'INV_',
                   '_worsening'])]
    features[count_cols] = features[count_cols].fillna(0)

    # Missingness indicators: NaN after left-join means the patient had zero rows for
    # that lab category → fill with 1 (truly missing), not 0.
    missing_cols = [c for c in features.columns if c.endswith('_is_missing')]
    if missing_cols:
        features[missing_cols] = features[missing_cols].fillna(1).astype(int)
    return features


# ═══════════════════════════════════════════════════════════════
# STEP 4a (cont): MEDICATION FEATURES
# ═══════════════════════════════════════════════════════════════

def build_medication_features(med_df, cfg):
    """Build patient-level features from medication data."""
    df = med_df.copy()
    med_categories = df['CATEGORY'].unique()

    bin_names = _bin_names(cfg)
    adj_pairs = _adjacent_pairs(cfg)
    # Per-bin med counts
    med_per_bin = {}
    for b in bin_names:
        sub = df[df['TIME_WINDOW'] == b]
        if len(sub) == 0:
            continue
        per_b = sub.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
        per_b.columns = [f"MED_{c}_count_{b}" for c in per_b.columns]
        med_per_bin[b] = per_b

    med_total = df.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    med_total.columns = [f"MED_{c}_count_total" for c in med_total.columns]

    med_has_ever = (med_total > 0).astype(int)
    med_has_ever.columns = [c.replace('_count_total', '_has_ever') for c in med_has_ever.columns]

    # Adjacent-bin acceleration per category
    med_accel = pd.DataFrame(index=med_total.index)
    for cat in med_categories:
        for recent_bin, earlier_bin in adj_pairs:
            col_r = f"MED_{cat}_count_{recent_bin}"
            col_e = f"MED_{cat}_count_{earlier_bin}"
            dfr = med_per_bin.get(recent_bin); dfe = med_per_bin.get(earlier_bin)
            if dfr is None or dfe is None or col_r not in dfr.columns or col_e not in dfe.columns:
                continue
            merged = dfr[[col_r]].join(dfe[[col_e]], how='outer').fillna(0)
            med_accel[f"MED_{cat}_accel_{recent_bin}_vs_{earlier_bin}"] = (
                merged[col_r] > merged[col_e]
            ).astype(int)

    # New-in-recent: med category appeared in the most-recent bin but NOT in any earlier bin.
    # No-op in the no-bins variant (single bin → no "earlier" bins to compare against).
    med_new_acute = pd.DataFrame(index=med_total.index)
    _mbins = _bin_names(cfg)
    _med_recent = _mbins[0] if _mbins else None
    _med_earlier = _mbins[1:] if len(_mbins) > 1 else []
    df_acute = med_per_bin.get(_med_recent) if _med_recent else None
    if df_acute is not None and _med_earlier:
        for cat in med_categories:
            col_acute = f"MED_{cat}_count_{_med_recent}"
            if col_acute not in df_acute.columns:
                continue
            earlier_sum = pd.Series(0, index=df_acute.index)
            for earlier_b in _med_earlier:
                dfe = med_per_bin.get(earlier_b)
                if dfe is None:
                    continue
                ce = f"MED_{cat}_count_{earlier_b}"
                if ce in dfe.columns:
                    earlier_sum = earlier_sum.add(dfe[ce].reindex(df_acute.index, fill_value=0), fill_value=0)
            med_new_acute[f"MED_{cat}_new_in_{_med_recent}"] = (
                (df_acute[col_acute] > 0) & (earlier_sum == 0)
            ).astype(int)

    if 'VALUE' in df.columns:
        med_qty = df.groupby(['PATIENT_GUID', 'CATEGORY'])['VALUE'].agg(['sum', 'mean']).unstack(fill_value=0)
        med_qty_sum = med_qty['sum']
        med_qty_sum.columns = [f"MED_{c}_total_qty" for c in med_qty_sum.columns]
        med_qty_mean = med_qty['mean']
        med_qty_mean.columns = [f"MED_{c}_mean_qty" for c in med_qty_mean.columns]
    else:
        med_qty_sum = pd.DataFrame()
        med_qty_mean = pd.DataFrame()

    med_unique = df.groupby(['PATIENT_GUID', 'CATEGORY'])['CODE_ID'].nunique().unstack(fill_value=0)
    med_unique.columns = [f"MED_{c}_unique_drugs" for c in med_unique.columns]

    # Aggregate
    med_agg_total = df.groupby('PATIENT_GUID').size()
    med_agg_total.name = 'MED_AGG_total_prescriptions'
    med_agg_cats = df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    med_agg_cats.name = 'MED_AGG_unique_categories'
    med_agg_drugs = df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
    med_agg_drugs.name = 'MED_AGG_unique_drugs'
    # Per-bin aggregate med counts
    med_agg_per_bin = {}
    for b in bin_names:
        s = df[df['TIME_WINDOW'] == b].groupby('PATIENT_GUID').size()
        s.name = f'MED_AGG_count_{b}'
        med_agg_per_bin[b] = s
    med_agg_merged = pd.DataFrame()
    for recent_bin, earlier_bin in adj_pairs:
        sr = med_agg_per_bin.get(recent_bin); se = med_agg_per_bin.get(earlier_bin)
        if sr is None or se is None:
            continue
        merged = pd.concat([sr, se], axis=1).fillna(0)
        med_agg_merged[f'MED_AGG_accel_{recent_bin}_vs_{earlier_bin}'] = (
            merged[f"MED_AGG_count_{recent_bin}"] > merged[f"MED_AGG_count_{earlier_bin}"]
        ).astype(int)
    polypharmacy = df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    polypharmacy.name = 'MED_AGG_polypharmacy'

    # Merge
    features = pd.DataFrame(index=df['PATIENT_GUID'].unique())
    features.index.name = 'PATIENT_GUID'
    # Per-bin med counts (acute / subacute / recent_past / historical)
    for per_b in med_per_bin.values():
        features = features.join(per_b, how='left')
    for feat_df in [med_total, med_has_ever, med_accel, med_new_acute,
                    med_qty_sum, med_qty_mean, med_unique]:
        if not feat_df.empty:
            features = features.join(feat_df, how='left')
    for feat_s in [med_agg_total, med_agg_cats, med_agg_drugs, polypharmacy]:
        features = features.join(feat_s, how='left')
    # Per-bin aggregate counts
    for s in med_agg_per_bin.values():
        features = features.join(s, how='left')
    if not med_agg_merged.empty:
        features = features.join(med_agg_merged, how='left')
    features = features.fillna(0)
    return features


# ═══════════════════════════════════════════════════════════════
# STEP 4a (cont): INTERACTION FEATURES
# ═══════════════════════════════════════════════════════════════

def build_interaction_features(feature_matrix, cfg):
    """Build clinical x medication interaction features from config pairs."""
    fm = feature_matrix.copy()
    interactions = {}

    for feat_name, col_a, col_b in cfg.INTERACTION_PAIRS:
        if col_a in fm.columns and col_b in fm.columns:
            interactions[f'INT_{feat_name}'] = fm[col_a] * fm[col_b]

    # Multi-symptom burden
    if 'AGG_symptom_unique_categories' in fm.columns:
        interactions['INT_multi_symptom_burden'] = (fm['AGG_symptom_unique_categories'] >= 3).astype(int)

    # High investigation with symptoms
    if 'INV_count_total' in fm.columns and 'AGG_symptom_count_total' in fm.columns:
        interactions['INT_high_investigation_with_symptoms'] = (
            (fm['INV_count_total'] >= 3) & (fm['AGG_symptom_count_total'] >= 2)
        ).astype(int)

    int_df = pd.DataFrame(interactions, index=fm.index)
    fm = pd.concat([fm, int_df], axis=1)
    logger.info(f"  Added {len(interactions)} interaction features")
    return fm


# ═══════════════════════════════════════════════════════════════
# STEP 4b: ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════

def build_advanced_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Clusters, visit patterns, trajectories, escalation, cross-domain, decay."""
    logger.info(f"  ADVANCED FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    adv = pd.DataFrame(index=patient_list)
    adv.index.name = 'PATIENT_GUID'
    obs_df = clin.copy()
    lab_df = clin[clin['VALUE'].notna() & clin['CATEGORY'].isin(cfg.LAB_CATEGORIES)].copy()

    # GROUP 1: SYMPTOM CLUSTERING (from config)
    logger.info(f"  Building symptom clusters...")
    for group_name, cats in cfg.CLUSTER_DEFINITIONS:
        group_df = obs_df[obs_df['CATEGORY'].isin(cats)]
        has_group = group_df.groupby('PATIENT_GUID').size() > 0
        adv[f'CLUSTER_{group_name}_any'] = adv.index.map(has_group).fillna(False).astype(int)
        group_count = group_df.groupby('PATIENT_GUID').size()
        adv[f'CLUSTER_{group_name}_count'] = adv.index.map(group_count).fillna(0).astype(int)
        group_unique = group_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
        adv[f'CLUSTER_{group_name}_breadth'] = adv.index.map(group_unique).fillna(0).astype(int)
        group_B = group_df[group_df['TIME_WINDOW'] == 'all'].groupby('PATIENT_GUID').size()
        adv[f'CLUSTER_{group_name}_count_acute'] = adv.index.map(group_B).fillna(0).astype(int)

    system_cols = [c for c in adv.columns if c.startswith('CLUSTER_') and c.endswith('_any')]
    adv['CLUSTER_multi_system_count'] = adv[system_cols].sum(axis=1)
    adv['CLUSTER_3plus_systems'] = (adv['CLUSTER_multi_system_count'] >= 3).astype(int)

    # Cross-cluster combinations
    cluster_names = [name for name, _ in cfg.CLUSTER_DEFINITIONS]
    for c1, c2 in combinations(cluster_names[:6], 2):
        col1 = f'CLUSTER_{c1}_any'
        col2 = f'CLUSTER_{c2}_any'
        if col1 in adv.columns and col2 in adv.columns:
            adv[f'CLUSTER_{c1}_AND_{c2}'] = (adv[col1] & adv[col2]).astype(int)

    # GROUP 2: VISIT PATTERNS
    logger.info(f"  Building visit patterns...")
    clin_windowed = clin[clin['TIME_WINDOW'].isin(_bin_names(cfg))].copy()
    visits = clin_windowed.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    adv['VISIT_unique_dates'] = adv.index.map(visits).fillna(0).astype(int)
    visits_A = clin_windowed[clin_windowed['TIME_WINDOW'] == 'all'].groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    visits_B = clin_windowed[clin_windowed['TIME_WINDOW'] == 'all'].groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    adv['VISIT_unique_dates_A'] = adv.index.map(visits_A).fillna(0).astype(int)
    adv['VISIT_unique_dates_B'] = adv.index.map(visits_B).fillna(0).astype(int)
    adv['VISIT_acceleration'] = (adv['VISIT_unique_dates_B'] > adv['VISIT_unique_dates_A']).astype(int)
    adv['VISIT_ratio_B_to_A'] = np.where(
        adv['VISIT_unique_dates_A'] > 0,
        adv['VISIT_unique_dates_B'] / adv['VISIT_unique_dates_A'],
        adv['VISIT_unique_dates_B']
    )
    events_per_patient = clin_windowed.groupby('PATIENT_GUID').size()
    adv['VISIT_events_per_visit'] = np.where(
        adv['VISIT_unique_dates'] > 0,
        adv.index.map(events_per_patient).fillna(0) / adv['VISIT_unique_dates'], 0
    )

    # GROUP 3: TEMPORAL TRAJECTORY
    logger.info(f"  Building temporal trajectories...")
    obs_windowed = obs_df[obs_df['TIME_WINDOW'].isin(_bin_names(cfg))].copy()
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        obs_windowed['QUARTER'] = pd.cut(
            obs_windowed['MONTHS_BEFORE_INDEX'],
            bins=[0, 6, 12, 18, 24, 30, 36],
            labels=['Q1_0-6m', 'Q2_6-12m', 'Q3_12-18m', 'Q4_18-24m', 'Q5_24-30m', 'Q6_30-36m'],
            right=True
        )
        quarter_counts = obs_windowed.groupby(['PATIENT_GUID', 'QUARTER']).size().unstack(fill_value=0)
        for col in quarter_counts.columns:
            adv[f'TRAJ_events_{col}'] = adv.index.map(quarter_counts[col]).fillna(0).astype(int)
        q_cols = [c for c in adv.columns if c.startswith('TRAJ_events_Q')]
        if len(q_cols) >= 2:
            adv['TRAJ_increasing_trend'] = (adv[q_cols[0]] > adv[q_cols[-1]]).astype(int)
            adv['TRAJ_trend_ratio'] = np.where(adv[q_cols[-1]] > 0, adv[q_cols[0]] / adv[q_cols[-1]], adv[q_cols[0]])

    # GROUP 4: LAB TRAJECTORIES
    logger.info(f"  Building lab trajectories...")
    lab_windowed = lab_df[lab_df['VALUE'].notna()].copy()
    for lab_cat in cfg.LAB_CATEGORIES:
        cat_labs = lab_windowed[lab_windowed['CATEGORY'] == lab_cat].copy()
        if len(cat_labs) == 0:
            continue
        cat_labs = cat_labs.sort_values(['PATIENT_GUID', 'EVENT_DATE'])

        def calc_slope(group):
            if len(group) < 2:
                return np.nan
            x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
            y = group['VALUE'].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 2:
                return np.nan
            x, y = x[mask], y[mask]
            if x.max() == x.min():
                return np.nan
            try:
                return np.polyfit(x, y, 1)[0]
            except Exception:
                return np.nan

        slopes = cat_labs.groupby('PATIENT_GUID').apply(calc_slope)
        adv[f'LAB_TRAJ_{lab_cat}_slope'] = adv.index.map(slopes).astype(float)
        adv[f'LAB_TRAJ_{lab_cat}_declining'] = (adv[f'LAB_TRAJ_{lab_cat}_slope'] < 0).astype(int)
        val_range = cat_labs.groupby('PATIENT_GUID')['VALUE'].apply(lambda x: x.max() - x.min())
        adv[f'LAB_TRAJ_{lab_cat}_range'] = adv.index.map(val_range).fillna(0).astype(float)
        cv = cat_labs.groupby('PATIENT_GUID')['VALUE'].apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0)
        adv[f'LAB_TRAJ_{lab_cat}_cv'] = adv.index.map(cv).fillna(0).astype(float)
        fl_diff = cat_labs.groupby('PATIENT_GUID').apply(
            lambda x: x.iloc[-1]['VALUE'] - x.iloc[0]['VALUE'] if len(x) >= 2 else 0
        )
        adv[f'LAB_TRAJ_{lab_cat}_first_last_diff'] = adv.index.map(fl_diff).fillna(0).astype(float)

    # GROUP 5: MEDICATION ESCALATION
    logger.info(f"  Building medication escalation patterns...")
    med_windowed = med[med['TIME_WINDOW'].isin(_bin_names(cfg))].copy()
    pain_meds = cfg.PAIN_MED_CATEGORIES
    pain_med_df = med_windowed[med_windowed['CATEGORY'].isin(pain_meds)]
    pain_cats = pain_med_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_pain_category_count'] = adv.index.map(pain_cats).fillna(0).astype(int)
    pain_A = pain_med_df[pain_med_df['TIME_WINDOW'] == 'all'].groupby('PATIENT_GUID').size()
    pain_B = pain_med_df[pain_med_df['TIME_WINDOW'] == 'all'].groupby('PATIENT_GUID').size()
    adv['MED_ESC_pain_count_subacute'] = adv.index.map(pain_A).fillna(0).astype(int)
    adv['MED_ESC_pain_count_acute'] = adv.index.map(pain_B).fillna(0).astype(int)
    adv['MED_ESC_pain_acceleration'] = (adv['MED_ESC_pain_count_acute'] > adv['MED_ESC_pain_count_subacute']).astype(int)

    steroid_df = med_windowed[med_windowed['CATEGORY'] == cfg.STEROID_CATEGORY]
    has_steroid = steroid_df.groupby('PATIENT_GUID').size() > 0
    adv['MED_ESC_has_corticosteroid'] = adv.index.map(has_steroid).fillna(False).astype(int)

    repeat_cat = cfg.REPEAT_PRESCRIPTION_CATEGORY
    repeat_df = med_windowed[med_windowed['CATEGORY'] == repeat_cat]
    repeat_count = repeat_df.groupby('PATIENT_GUID').size()
    adv[f'MED_ESC_{repeat_cat.lower()}_repeat_count'] = adv.index.map(repeat_count).fillna(0).astype(int)
    adv[f'MED_ESC_{repeat_cat.lower()}_repeat_3plus'] = (adv[f'MED_ESC_{repeat_cat.lower()}_repeat_count'] >= 3).astype(int)

    med_cats_A = med_windowed[med_windowed['TIME_WINDOW'] == 'all'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    med_cats_B = med_windowed[med_windowed['TIME_WINDOW'] == 'all'].groupby('PATIENT_GUID')['CATEGORY'].nunique()
    adv['MED_ESC_polypharmacy_A'] = adv.index.map(med_cats_A).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_B'] = adv.index.map(med_cats_B).fillna(0).astype(int)
    adv['MED_ESC_polypharmacy_increase'] = (adv['MED_ESC_polypharmacy_B'] > adv['MED_ESC_polypharmacy_A']).astype(int)

    # GROUP 6: INVESTIGATION PATTERNS
    logger.info(f"  Building investigation patterns...")
    inv_df = obs_df[obs_df['CATEGORY'].isin(cfg.INVESTIGATION_CATEGORIES)]
    symptom_cats_all = _get_all_symptom_cats(cfg)
    symptom_df = obs_df[obs_df['CATEGORY'].isin(symptom_cats_all)]
    first_symptom = symptom_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    first_inv = inv_df.groupby('PATIENT_GUID')['EVENT_DATE'].min()
    gap = first_inv.subtract(first_symptom).dt.days
    adv['INV_PATTERN_symptom_to_inv_days'] = adv.index.map(gap).fillna(-1).astype(float)
    adv['INV_PATTERN_has_inv_after_symptom'] = (adv['INV_PATTERN_symptom_to_inv_days'] >= 0).astype(int)
    symptom_count = symptom_df.groupby('PATIENT_GUID').size()
    inv_count = inv_df.groupby('PATIENT_GUID').size()
    adv['INV_PATTERN_symptom_count'] = adv.index.map(symptom_count).fillna(0).astype(int)
    adv['INV_PATTERN_inv_count'] = adv.index.map(inv_count).fillna(0).astype(int)

    # GROUP 7: CROSS-DOMAIN
    logger.info(f"  Building cross-domain interactions...")
    high_visits = (adv.get('VISIT_unique_dates', pd.Series(0, index=adv.index)) >= 5)
    multi_system = (adv.get('CLUSTER_multi_system_count', pd.Series(0, index=adv.index)) >= 2)
    adv['CROSS_diagnostic_odyssey'] = (high_visits & multi_system).astype(int)

    # GROUP 8: TIME-DECAY
    logger.info(f"  Building time-decay features...")
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        obs_decay = obs_windowed.copy()
        obs_decay['DECAY_WEIGHT'] = np.exp(-0.1 * obs_decay['MONTHS_BEFORE_INDEX'])
        symptom_weighted = obs_decay[obs_decay['CATEGORY'].isin(symptom_cats_all)]
        ws = symptom_weighted.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
        adv['DECAY_symptom_weighted_score'] = adv.index.map(ws).fillna(0).astype(float)
        wt = obs_decay.groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
        adv['DECAY_total_weighted_score'] = adv.index.map(wt).fillna(0).astype(float)
        for cat in cfg.DECAY_CATEGORIES:
            cat_w = obs_decay[obs_decay['CATEGORY'] == cat].groupby('PATIENT_GUID')['DECAY_WEIGHT'].sum()
            adv[f'DECAY_{cat}_weighted'] = adv.index.map(cat_w).fillna(0).astype(float)

    # Fill
    lab_traj_cols = [c for c in adv.columns if c.startswith('LAB_TRAJ_') and
                     any(x in c for x in ['_slope', '_range', '_cv', '_first_last_diff'])]
    for col in lab_traj_cols:
        med_val = adv[col].median()
        adv[col] = adv[col].fillna(med_val if pd.notna(med_val) else 0)
    adv = adv.fillna(0).replace([np.inf, -np.inf], 0)

    logger.info(f"  Advanced features: {adv.shape[1]}")
    return adv


# ═══════════════════════════════════════════════════════════════
# STEP 4c: MAXIMUM FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_maximum_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Monthly bins, rolling 3M, per-category granular, co-occurrence, recurrence, rates, age, entropy."""
    logger.info(f"  MAXIMUM FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in med.columns:
        med['MONTHS_BEFORE_INDEX'] = pd.to_numeric(med['MONTHS_BEFORE_INDEX'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    mega = pd.DataFrame(index=patient_list)
    mega.index.name = 'PATIENT_GUID'
    obs_windowed = clin[clin['TIME_WINDOW'].isin(_bin_names(cfg))].copy()
    med_windowed = med[med['TIME_WINDOW'].isin(_bin_names(cfg))].copy()

    # MONTHLY BINS
    logger.info(f"  Building monthly bins...")
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        for m_start in range(0, 13, 3):
            m_end = m_start + 3
            month_df = obs_windowed[(obs_windowed['MONTHS_BEFORE_INDEX'] > m_start) &
                                     (obs_windowed['MONTHS_BEFORE_INDEX'] <= m_end)]
            mc = month_df.groupby('PATIENT_GUID').size()
            mega[f'MONTHLY_obs_{m_start}_{m_end}m'] = mega.index.map(mc).fillna(0).astype(int)
            mu = month_df.groupby('PATIENT_GUID')['CATEGORY'].nunique()
            mega[f'MONTHLY_obs_cats_{m_start}_{m_end}m'] = mega.index.map(mu).fillna(0).astype(int)

    # ROLLING 3M
    logger.info(f"  Building rolling 3-month features...")
    if 'MONTHS_BEFORE_INDEX' in obs_windowed.columns:
        for start in range(0, 10, 3):
            end = start + 3
            window_df = obs_windowed[(obs_windowed['MONTHS_BEFORE_INDEX'] > start) &
                                      (obs_windowed['MONTHS_BEFORE_INDEX'] <= end)]
            wc = window_df.groupby('PATIENT_GUID').size()
            mega[f'ROLLING3M_count_{start}_{end}m'] = mega.index.map(wc).fillna(0).astype(int)

    # PER-CATEGORY GRANULAR
    logger.info(f"  Building per-category granular features...")
    for cat in obs_windowed['CATEGORY'].unique():
        cat_df = obs_windowed[obs_windowed['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        cc = cat_df.groupby('PATIENT_GUID').size()
        mega[f'CAT_{cat}_total'] = mega.index.map(cc).fillna(0).astype(int)
        cu = cat_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
        mega[f'CAT_{cat}_unique_codes'] = mega.index.map(cu).fillna(0).astype(int)

    # PER-MED-CATEGORY GRANULAR
    for cat in med_windowed['CATEGORY'].unique():
        cat_df = med_windowed[med_windowed['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        cc = cat_df.groupby('PATIENT_GUID').size()
        mega[f'MEDCAT_{cat}_total'] = mega.index.map(cc).fillna(0).astype(int)
        cu = cat_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
        mega[f'MEDCAT_{cat}_unique_codes'] = mega.index.map(cu).fillna(0).astype(int)

    # CO-OCCURRENCE PAIRS
    logger.info(f"  Building co-occurrence pairs...")
    patient_cats = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    top_cats = obs_windowed['CATEGORY'].value_counts().head(10).index.tolist()
    for c1, c2 in combinations(top_cats, 2):
        co = patient_cats.apply(lambda x: int(c1 in x and c2 in x) if isinstance(x, set) else 0)
        mega[f'PAIR_{c1}_x_{c2}'] = mega.index.map(co).fillna(0).astype(int)

    # RATES & RATIOS
    logger.info(f"  Building rates & ratios...")
    total_events = obs_windowed.groupby('PATIENT_GUID').size()
    total_cats = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    mega['RATE_events_per_cat'] = np.where(
        mega.index.map(total_cats).fillna(0) > 0,
        mega.index.map(total_events).fillna(0) / mega.index.map(total_cats).fillna(1), 0
    )

    # AGE INTERACTIONS
    logger.info(f"  Building age interactions...")
    age = existing_fm['AGE_AT_INDEX'].reindex(mega.index).fillna(65)
    mega['AGE_over_50'] = (age >= 50).astype(int)
    mega['AGE_over_65'] = (age >= 65).astype(int)
    mega['AGE_over_75'] = (age >= 75).astype(int)

    for cat_name in [n for n, _ in cfg.CLUSTER_DEFINITIONS[:4]]:
        col = f'CLUSTER_{cat_name}_any'
        if col in existing_fm.columns:
            mega[f'AGEX_{cat_name}_over65'] = (
                (age >= 65) & (existing_fm[col].reindex(mega.index).fillna(0) == 1)
            ).astype(int)

    # ENTROPY
    logger.info(f"  Building entropy features...")
    def calc_entropy(group):
        counts = group.value_counts(normalize=True)
        return -(counts * np.log2(counts + 1e-10)).sum()

    cat_entropy = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].apply(calc_entropy)
    mega['ENTROPY_category'] = mega.index.map(cat_entropy).fillna(0).astype(float)

    # GINI
    def calc_gini(group):
        counts = group.value_counts().values
        if len(counts) <= 1:
            return 0
        n = len(counts)
        total = counts.sum()
        if total == 0:
            return 0
        sorted_counts = np.sort(counts)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_counts) / (n * total)) - (n + 1) / n

    cat_gini = obs_windowed.groupby('PATIENT_GUID')['CATEGORY'].apply(calc_gini)
    mega['GINI_category'] = mega.index.map(cat_gini).fillna(0).astype(float)

    mega = mega.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"  Maximum features: {mega.shape[1]}")
    return mega


# ═══════════════════════════════════════════════════════════════
# STEP 4e: NEW SIGNAL FEATURES
# ═══════════════════════════════════════════════════════════════

def build_new_signal_features(clin_df, med_df, existing_fm, window_name, cfg):
    """CODE_ID-level features, recency, distinct visit counts."""
    logger.info(f"  NEW SIGNAL FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    clin['INDEX_DATE'] = pd.to_datetime(clin['INDEX_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'VALUE' in clin.columns:
        clin['VALUE'] = pd.to_numeric(clin['VALUE'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    nf = pd.DataFrame(index=patient_list)
    nf.index.name = 'PATIENT_GUID'
    obs_AB = clin[clin['TIME_WINDOW'].isin(_bin_names(cfg))].copy()
    med_AB = med[med['TIME_WINDOW'].isin(_bin_names(cfg))].copy()

    # Per-lab-term features
    logger.info(f"  Building per-lab-term features...")
    lab_obs = obs_AB[obs_AB['VALUE'].notna() & obs_AB['CATEGORY'].isin(cfg.LAB_CATEGORIES)].copy()
    if len(lab_obs) > 0:
        top_terms = lab_obs['TERM'].value_counts().head(30).index.tolist()
        for term in top_terms:
            # Keep only [A-Za-z0-9_] so feature names are LightGBM-JSON-safe AND BigQuery-safe.
            # Truncate to 40 chars to stay readable.
            safe_name = re.sub(r'[^A-Za-z0-9_]', '_', term[:40])
            safe_name = re.sub(r'_+', '_', safe_name).strip('_') or 'TERM'
            term_df = lab_obs[lab_obs['TERM'] == term]
            if len(term_df) < 20:
                continue

            tc = term_df.groupby('PATIENT_GUID').size()
            nf[f'LABTERM_{safe_name}_count'] = nf.index.map(tc).fillna(0).astype(int)
            tm = term_df.groupby('PATIENT_GUID')['VALUE'].mean()
            nf[f'LABTERM_{safe_name}_mean'] = nf.index.map(tm).fillna(np.nan)
            tl = term_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID')['VALUE'].last()
            nf[f'LABTERM_{safe_name}_last'] = nf.index.map(tl).fillna(np.nan)

            if len(term_df) >= 50:
                def term_slope(group):
                    if len(group) < 2:
                        return np.nan
                    x = (group['EVENT_DATE'] - group['EVENT_DATE'].min()).dt.days.values.astype(float)
                    y = group['VALUE'].values.astype(float)
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() < 2:
                        return np.nan
                    x, y = x[mask], y[mask]
                    if x.max() == x.min():
                        return np.nan
                    try:
                        return np.polyfit(x, y, 1)[0]
                    except Exception:
                        return np.nan
                ts = term_df.sort_values('EVENT_DATE').groupby('PATIENT_GUID').apply(term_slope)
                nf[f'LABTERM_{safe_name}_slope'] = nf.index.map(ts).fillna(0)

    # Recency features
    logger.info(f"  Building recency features...")
    for cat in obs_AB['CATEGORY'].unique():
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        last_event = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].max()
        index_date = obs_AB.groupby('PATIENT_GUID')['INDEX_DATE'].first()
        days_since = (index_date - last_event).dt.days
        safe_cat = cat[:30]
        nf[f'RECUR_{safe_cat}_days_since_last'] = nf.index.map(days_since).fillna(-1).astype(float)

    # Distinct visit dates per category
    logger.info(f"  Building distinct visit features...")
    for cat in obs_AB['CATEGORY'].unique():
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat]
        if len(cat_df) < 20:
            continue
        visit_dates = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
        safe_cat = cat[:30]
        nf[f'RECUR_{safe_cat}_distinct_visits'] = nf.index.map(visit_dates).fillna(0).astype(int)

    nf = nf.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"  New signal features: {nf.shape[1]}")
    return nf


# ═══════════════════════════════════════════════════════════════
# STEP 4f: TREND FEATURES
# ═══════════════════════════════════════════════════════════════

def build_trend_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Per-symptom frequency trends, interval stats, worsening flags."""
    logger.info(f"  TREND FEATURES - {window_name}")

    clin = clin_df.copy()
    med = med_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    med.columns = med.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')
    med['EVENT_DATE'] = pd.to_datetime(med['EVENT_DATE'], errors='coerce')
    if 'MONTHS_BEFORE_INDEX' in clin.columns:
        clin['MONTHS_BEFORE_INDEX'] = pd.to_numeric(clin['MONTHS_BEFORE_INDEX'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    tf = pd.DataFrame(index=patient_list)
    tf.index.name = 'PATIENT_GUID'
    obs_AB = clin[clin['TIME_WINDOW'].isin(_bin_names(cfg))].copy()
    med_AB = med[med['TIME_WINDOW'].isin(_bin_names(cfg))].copy()

    # Per-symptom frequency trends
    logger.info(f"  Building per-symptom frequency trends...")
    symptom_cats = cfg.SYMPTOM_CATEGORIES
    for cat in symptom_cats:
        cat_df = obs_AB[obs_AB['CATEGORY'] == cat].copy()
        if len(cat_df) < 20:
            continue
        safe_cat = cat[:30]

        # Event count
        ec = cat_df.groupby('PATIENT_GUID').size()
        tf[f'SEQ_{safe_cat}_event_count'] = tf.index.map(ec).fillna(0).astype(int)

        # Time span
        ts = cat_df.groupby('PATIENT_GUID')['EVENT_DATE'].agg(['min', 'max'])
        ts['span_days'] = (ts['max'] - ts['min']).dt.days
        tf[f'SEQ_{safe_cat}_span_days'] = tf.index.map(ts['span_days']).fillna(0).astype(float)

        # Frequency (events per year of span)
        tf[f'SEQ_{safe_cat}_freq_per_year'] = np.where(
            tf[f'SEQ_{safe_cat}_span_days'] > 30,
            tf[f'SEQ_{safe_cat}_event_count'] / (tf[f'SEQ_{safe_cat}_span_days'] / 365.25),
            tf[f'SEQ_{safe_cat}_event_count']
        )

        # Mean interval between events
        def mean_interval(group):
            dates = group['EVENT_DATE'].sort_values()
            if len(dates) < 2:
                return np.nan
            intervals = dates.diff().dt.days.dropna()
            return intervals.mean()

        mi = cat_df.groupby('PATIENT_GUID').apply(mean_interval)
        tf[f'SEQ_{safe_cat}_mean_interval_days'] = tf.index.map(mi).fillna(-1).astype(float)

        # Interval-trend slope: regress consecutive interval-lengths against their index.
        # Negative slope = intervals shrinking = events compressing toward index = worsening.
        def interval_slope(group):
            dates = group['EVENT_DATE'].sort_values().dropna()
            if len(dates) < 3:
                return np.nan
            intervals = dates.diff().dt.days.dropna().to_numpy(dtype=float)
            if len(intervals) < 2 or np.all(intervals == intervals[0]):
                return np.nan
            x = np.arange(len(intervals), dtype=float)
            try:
                return float(np.polyfit(x, intervals, 1)[0])
            except Exception:
                return np.nan

        islope = cat_df.groupby('PATIENT_GUID').apply(interval_slope)
        tf[f'SEQ_{safe_cat}_interval_slope'] = tf.index.map(islope).fillna(0).astype(float)

        # Half-vs-half frequency split within the lookback window.
        # MONTHS_BEFORE_INDEX is months back from index, so MBI=0 is "now".
        # h1 = older half (MBI > half), h2 = recent half (MBI <= half).
        # h2_over_h1 > 1 → events accelerating toward index.
        if 'MONTHS_BEFORE_INDEX' in cat_df.columns:
            mbi_max = cat_df['MONTHS_BEFORE_INDEX'].max()
            if pd.notna(mbi_max) and mbi_max > 0:
                half = mbi_max / 2.0
                h1 = cat_df[cat_df['MONTHS_BEFORE_INDEX'] >  half].groupby('PATIENT_GUID').size()
                h2 = cat_df[cat_df['MONTHS_BEFORE_INDEX'] <= half].groupby('PATIENT_GUID').size()
                h1m = h1.reindex(tf.index, fill_value=0).astype(float)
                h2m = h2.reindex(tf.index, fill_value=0).astype(float)
                tf[f'SEQ_{safe_cat}_h1_freq']      = h1m
                tf[f'SEQ_{safe_cat}_h2_freq']      = h2m
                tf[f'SEQ_{safe_cat}_h2_over_h1']   = h2m / h1m.clip(lower=1)
                tf[f'SEQ_{safe_cat}_h2_minus_h1']  = h2m - h1m
                tf[f'SEQ_{safe_cat}_is_worsening'] = (h2m > h1m).astype(int)

    # Per-medication frequency trends
    logger.info(f"  Building per-med frequency trends...")
    for cat in med_AB['CATEGORY'].unique():
        cat_df = med_AB[med_AB['CATEGORY'] == cat].copy()
        if len(cat_df) < 20:
            continue
        safe_cat = cat[:30]

        mc = cat_df.groupby('PATIENT_GUID').size()
        tf[f'MEDREC_{safe_cat}_count'] = tf.index.map(mc).fillna(0).astype(int)

        mu = cat_df.groupby('PATIENT_GUID')['CODE_ID'].nunique()
        tf[f'MEDREC_{safe_cat}_unique_drugs'] = tf.index.map(mu).fillna(0).astype(int)

    tf = tf.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"  Trend features: {tf.shape[1]}")
    return tf


def build_acceleration_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Fine-grained acceleration features beyond binary _acceleration flags.

    Captures: 'symptoms not just present, but ramping up' — the clinical
    signature of imminent cancer. Numeric ratios + acute-onset clusters +
    PSA recent-vs-lifetime velocity comparison.

    Non-leakage: all derived from events in the lookback window only.
    """
    logger.info(f"  ACCELERATION FEATURES - {window_name}")

    clin = clin_df.copy()
    clin.columns = clin.columns.str.strip().str.upper()
    clin['EVENT_DATE'] = pd.to_datetime(clin['EVENT_DATE'], errors='coerce')

    patient_list = existing_fm.index.tolist()
    af = pd.DataFrame(index=patient_list)
    af.index.name = 'PATIENT_GUID'

    obs_AB = clin[clin['TIME_WINDOW'].isin(_bin_names(cfg))].copy()
    obs_A = obs_AB[obs_AB['TIME_WINDOW'] == 'all']
    obs_B = obs_AB[obs_AB['TIME_WINDOW'] == 'all']

    # 1. Total events ratio: B / max(A, 1) — "how many more events recently?"
    total_A = obs_A.groupby('PATIENT_GUID').size()
    total_B = obs_B.groupby('PATIENT_GUID').size()
    a_map = total_A.reindex(af.index, fill_value=0).astype(float)
    b_map = total_B.reindex(af.index, fill_value=0).astype(float)
    af['ACCEL_total_events_B_over_A'] = b_map / a_map.clip(lower=1)
    af['ACCEL_total_events_B_minus_A'] = b_map - a_map

    # 2. Diversity (unique categories) ratio + change
    unique_A = obs_A.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    unique_B = obs_B.groupby('PATIENT_GUID')['CATEGORY'].nunique()
    ua = unique_A.reindex(af.index, fill_value=0).astype(float)
    ub = unique_B.reindex(af.index, fill_value=0).astype(float)
    af['ACCEL_unique_cats_B_over_A'] = ub / ua.clip(lower=1)
    af['ACCEL_unique_cats_B_minus_A'] = ub - ua

    # 3. Acute-onset cluster: count of categories present only in B (zero in A)
    cats_per_patient_A = obs_A.groupby('PATIENT_GUID')['CATEGORY'].apply(set)
    cats_per_patient_B = obs_B.groupby('PATIENT_GUID')['CATEGORY'].apply(set)

    def _new_count(pid):
        b = cats_per_patient_B.get(pid, set())
        a = cats_per_patient_A.get(pid, set())
        return len(b - a)
    af['ACCEL_new_categories_in_B'] = af.index.to_series().map(_new_count).fillna(0).astype(int)
    af['ACCEL_has_acute_onset'] = (af['ACCEL_new_categories_in_B'] >= 2).astype(int)

    # 4. Visit-rate (unique event-dates) ratio
    visit_A = obs_A.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    visit_B = obs_B.groupby('PATIENT_GUID')['EVENT_DATE'].nunique()
    va = visit_A.reindex(af.index, fill_value=0).astype(float)
    vb = visit_B.reindex(af.index, fill_value=0).astype(float)
    af['ACCEL_visits_B_over_A'] = vb / va.clip(lower=1)

    # 5. Number of categories where count_B > count_A (count of accelerating categories)
    # Align BOTH index (patients) and columns (categories) so comparison works
    cnt_per_cat_A = obs_A.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    cnt_per_cat_B = obs_B.groupby(['PATIENT_GUID', 'CATEGORY']).size().unstack(fill_value=0)
    common_cols = list(set(cnt_per_cat_A.columns) | set(cnt_per_cat_B.columns))
    cnt_per_cat_A = cnt_per_cat_A.reindex(index=af.index, columns=common_cols, fill_value=0)
    cnt_per_cat_B = cnt_per_cat_B.reindex(index=af.index, columns=common_cols, fill_value=0)
    accel_cats = (cnt_per_cat_B > cnt_per_cat_A).sum(axis=1)
    af['ACCEL_n_accelerating_categories'] = accel_cats.astype(int)

    # 6. PSA recent-vs-lifetime velocity (prostate-aware acceleration)
    psa_lab = clin[(clin['CATEGORY'].isin(['PSA', 'PSA_FREE', 'PSA_RATIO'])) &
                    pd.to_numeric(clin['VALUE'], errors='coerce').notna()].copy()
    psa_lab['VALUE'] = pd.to_numeric(psa_lab['VALUE'], errors='coerce')

    def _velocity_recent_vs_lifetime(group):
        """Named Series so groupby().apply() yields a DataFrame across pandas versions."""
        if len(group) < 2:
            return pd.Series({'recent_v': np.nan, 'life_v': np.nan})
        g = group.sort_values('EVENT_DATE')
        g_recent = g[g['TIME_WINDOW'] == 'all']
        days_full = (g['EVENT_DATE'].iloc[-1] - g['EVENT_DATE'].iloc[0]).days
        life_v = ((g['VALUE'].iloc[-1] - g['VALUE'].iloc[0]) /
                  max(days_full, 1) * 365.25) if days_full > 30 else np.nan
        if len(g_recent) >= 2:
            days_rec = (g_recent['EVENT_DATE'].iloc[-1] - g_recent['EVENT_DATE'].iloc[0]).days
            rec_v = ((g_recent['VALUE'].iloc[-1] - g_recent['VALUE'].iloc[0]) /
                     max(days_rec, 1) * 365.25) if days_rec > 14 else np.nan
        else:
            rec_v = np.nan
        return pd.Series({'recent_v': rec_v, 'life_v': life_v})

    if len(psa_lab) > 0:
        vels = psa_lab.groupby('PATIENT_GUID').apply(_velocity_recent_vs_lifetime)
        af['ACCEL_PSA_recent_velocity_per_year'] = vels['recent_v'].reindex(af.index, fill_value=0).astype(float)
        af['ACCEL_PSA_lifetime_velocity_per_year'] = vels['life_v'].reindex(af.index, fill_value=0).astype(float)
        # Acceleration = recent velocity / lifetime velocity (capped)
        denom = af['ACCEL_PSA_lifetime_velocity_per_year'].abs().clip(lower=0.05)
        af['ACCEL_PSA_velocity_acceleration'] = (
            af['ACCEL_PSA_recent_velocity_per_year'] / denom
        ).clip(-50, 50)
        af['ACCEL_PSA_recent_velocity_high'] = (af['ACCEL_PSA_recent_velocity_per_year'] > 0.75).astype(int)
        af['ACCEL_PSA_recent_velocity_very_high'] = (af['ACCEL_PSA_recent_velocity_per_year'] > 2.0).astype(int)

    af = af.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"  Acceleration features: {af.shape[1]}")
    return af
