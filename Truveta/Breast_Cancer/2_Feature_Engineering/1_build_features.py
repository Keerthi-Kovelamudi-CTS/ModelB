"""
Build the wide per-patient feature matrix — heavy-FE design.

Four layers (see config.py for details):

    Layer A — Category aggregations: 67 categories × 6 aggs (~400 features)
              count / present / value_mean / duration_sum / recency_days / event_density
    Layer B — Breast-specific hand-engineered (HRT, alcohol, age thresholds)
    Layer C — Engagement / code-type-aware
    Layer D — Cross-feature interactions

Output:
    data/features/breast_feature_matrix.parquet

Usage:
    python 1_build_features.py
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd

import config
import breast_features
from io_utils import read_table, write_table


def slug(category: str) -> str:
    """Make a clean column-name slug from a category label.
    'Lab - Lipid Panel' -> 'lab_lipid_panel'
    'Medication - GI / Acid Suppression' -> 'medication_gi_acid_suppression'
    """
    s = str(category).lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = s.strip('_')
    return s


def build_layer_a(merged: pd.DataFrame, spine: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer A: 67 categories × 6 aggregations. Returns (by_pc_long, wide_df)."""
    # ── Per-patient × category aggregates ──────────────────────────────────
    agg_spec = dict(
        count        = ('count',        'sum'),
        value_mean   = ('value_mean',   'mean'),
        duration_sum = ('duration_sum', 'sum'),
        first_event  = ('first_event',  'min'),
        last_event   = ('last_event',   'max'),
    )
    # Carry recent/older half-window counts for Layer G (lung-style trend), when
    # the extract provides them (0_extract.py emits count_h1 / count_h2).
    if 'count_h1' in merged.columns and 'count_h2' in merged.columns:
        agg_spec['count_h1'] = ('count_h1', 'sum')
        agg_spec['count_h2'] = ('count_h2', 'sum')
    by_pc = merged.groupby(['patient_id', 'category'], dropna=False).agg(**agg_spec).reset_index()
    by_pc['present'] = (by_pc['count'] > 0).astype('int8')

    # Recency: days from anchor_date to most-recent event in this category.
    anchor = pd.to_datetime(spine.set_index('patient_id')['anchor_date'])
    last_ev = pd.to_datetime(by_pc['last_event'])
    by_pc['recency_days'] = (
        anchor.reindex(by_pc['patient_id']).values - last_ev.values
    )
    by_pc['recency_days'] = pd.to_timedelta(by_pc['recency_days']).dt.days

    # Event density: events / active months within (first_event..last_event].
    first_ev = pd.to_datetime(by_pc['first_event'])
    active_days = (last_ev - first_ev).dt.days.clip(lower=1)
    active_months = active_days / 30.4375
    by_pc['event_density'] = (by_pc['count'] / active_months).astype('float32')

    print(f"  built (patient, category) long frame: {len(by_pc):,} rows")

    # ── Pivot wide ─────────────────────────────────────────────────────────
    wide_pieces = []
    for kind in config.CATEGORY_AGGREGATIONS:
        if kind not in by_pc.columns:
            continue
        piece = by_pc.pivot_table(
            index='patient_id', columns='category',
            values=kind, aggfunc='first',
        )
        piece.columns = [f"{slug(c)}__{kind}" for c in piece.columns]
        wide_pieces.append(piece)
    wide = pd.concat(wide_pieces, axis=1)
    wide.index.name = 'patient_id'

    # ── Fills per agg kind (sentinels in config.AGG_FILL_VALUES) ───────────
    for col in wide.columns:
        kind = col.rsplit('__', 1)[-1]
        if kind in config.AGG_FILL_VALUES:
            wide[col] = wide[col].fillna(config.AGG_FILL_VALUES[kind])
        # value_mean intentionally NOT filled — NaN encodes "never measured"

    return by_pc, wide


def main(window: str):
    # 1. Load inputs (per-window subdirs).
    raw_dir       = config.RAW_DIR / window
    features_dir  = config.FEATURES_DIR / window
    features_dir.mkdir(parents=True, exist_ok=True)
    agg_path      = raw_dir / 'breast_patient_code_aggregates.csv'
    spine_path    = raw_dir / 'breast_spine.csv'
    codelist_path = raw_dir / 'breast_codelist.parquet'

    for p in (agg_path, spine_path, codelist_path):
        if not p.exists():
            sys.exit(f"Missing input: {p}\nRun 0_extract.py first.")

    agg = pd.read_csv(agg_path)
    spine = pd.read_csv(spine_path)
    codelist = read_table(codelist_path)

    # Dedupe spine: BQ cohort can contain the same patient in both arms (cancer + non-cancer
    # placeholder sampling). Keep cancer-positive row when duplicated.
    n_dup = spine['patient_id'].duplicated().sum()
    if n_dup:
        spine = (spine.sort_values('label', ascending=False)
                       .drop_duplicates('patient_id', keep='first')
                       .reset_index(drop=True))
        print(f"  Deduped spine: removed {n_dup} duplicate patient rows (kept cancer-positive)")

    print(f"Loaded aggregates: {len(agg):,} rows "
          f"({agg['patient_id'].nunique():,} patients × {agg['code_id'].nunique():,} codes)")
    print(f"Loaded spine:      {len(spine):,} patients")
    print(f"Loaded codelist:   {len(codelist):,} codes across "
          f"{codelist['category'].nunique()} categories")

    agg['code_id']      = agg['code_id'].astype('Int64')
    codelist['code_id'] = codelist['code_id'].astype('Int64')

    # 2. Join code-level aggregates with codelist to get the category for each row.
    merged = agg.merge(
        codelist[['code_id', 'category', 'code_type']],
        on='code_id', how='inner',
    )
    if merged.empty:
        sys.exit("No rows after joining aggregates to codelist — check code_id matches.")

    # 3. Per-patient observation-event handling.
    obs_event_counts = (
        merged[merged['event_type'] == 'observation']
        .groupby('patient_id')['count'].sum()
    )
    n_before = spine['patient_id'].nunique()
    if config.KEEP_SPARSE_PATIENTS:
        # Approach B: keep EVERY cohort patient. Patients with no/few curated
        # events stay in the spine and fall through the left-join below as
        # all-zero feature rows (the EMIS "__PLACEHOLDER__" analog). No drop.
        rich = set(obs_event_counts[obs_event_counts >= config.MIN_OBS_EVENTS_PER_PATIENT].index)
        all_patients = set(spine['patient_id'])
        n_sparse = len(all_patients - rich)
        print(f"Approach B (KEEP_SPARSE_PATIENTS=True): {n_before:,} patients total | "
              f"{len(rich):,} with ≥{config.MIN_OBS_EVENTS_PER_PATIENT} curated obs events | "
              f"{n_sparse:,} sparse → all-zero placeholder rows "
              f"({100 * n_sparse / max(n_before, 1):.1f}%)")
    else:
        # Approach A: drop patients below the curated-observation threshold.
        eligible_patients = obs_event_counts[obs_event_counts >= config.MIN_OBS_EVENTS_PER_PATIENT].index
        spine = spine[spine['patient_id'].isin(eligible_patients)].copy()
        merged = merged[merged['patient_id'].isin(eligible_patients)].copy()
        print(f"Patients after min_obs ≥ {config.MIN_OBS_EVENTS_PER_PATIENT}: {len(spine):,} "
              f"(dropped {n_before - len(spine):,})")

    # 4. Layer A — category aggregations (67 categories × 6 aggs).
    print("Building Layer A (category aggregations)...")
    by_pc, layer_a = build_layer_a(merged, spine)
    print(f"  Layer A: {layer_a.shape[1]:,} feature columns across "
          f"{by_pc['category'].nunique()} categories × {len(config.CATEGORY_AGGREGATIONS)} aggregations")

    # 5. Layer B + C + D — breast-specific, engagement, interactions.
    extra = breast_features.build_all(merged, by_pc, spine)

    # 6. Concat all layers + spine.
    all_feats = layer_a.join(extra, how='outer')
    # Backfill any sentinel-fill columns that came in via Layer A but not yet filled
    # (extra already handled internally by breast_features.build_all)
    out = spine.set_index('patient_id').join(all_feats, how='left')

    # Final fills for any patients in spine without features (shouldn't happen,
    # but safe). Use the same per-kind sentinels as Layer A.
    for col in layer_a.columns:
        kind = col.rsplit('__', 1)[-1]
        if kind in config.AGG_FILL_VALUES and col in out.columns:
            out[col] = out[col].fillna(config.AGG_FILL_VALUES[kind])
    out = out.reset_index()

    # Reorder: spine first, features after.
    spine_cols = [c for c in config.SPINE_COLUMNS if c in out.columns]
    feature_col_names = [c for c in out.columns if c not in spine_cols]
    out = out[spine_cols + feature_col_names]

    # 7. Write.
    out_path = features_dir / 'breast_feature_matrix.parquet'
    write_table(out, out_path, index=False)
    print(f"\n[{window}] Wrote feature matrix: {out_path}")
    print(f"  shape: {out.shape}  (patients × cols)")
    print(f"  label distribution: {out['label'].value_counts().to_dict()}")
    print(f"  sex distribution:   {out['sex'].value_counts(dropna=False).to_dict()}")
    print(f"  feature breakdown:")
    print(f"    Layer A (categories × aggs): {layer_a.shape[1]:,}")
    print(f"    Layer B+C+D:                 {extra.shape[1]:,}")
    print(f"    Spine:                       {len(spine_cols)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', required=True,
                    help=f"Which window's features to build (e.g. {config.WINDOWS}, "
                         "or a custom label like 'holdout_1mo'). The script reads "
                         "data/raw/<window>/* and writes data/features/<window>/*.")
    args = ap.parse_args()
    main(args.window)
