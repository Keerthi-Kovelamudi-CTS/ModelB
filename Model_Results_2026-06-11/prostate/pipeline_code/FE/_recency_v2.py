"""
v2 temporal FE — multi-scale CUMULATIVE recency windows + recency-anchored escalation.

Replaces the coarse fixed A/B split (which put a calendar cut at 10.5y over a 20y
lookback → empty 'older' half for most patients → degenerate trends). Instead, per
category we count events in cumulative windows of increasing depth, measured from
the prediction GAP (the near edge of the lookback). Cumulative/nested windows:
  - capture escalation SHAPE at the resolution where signal lives (recent months),
  - FLEX to patient history length (short history → longer windows just equal the
    total; no degenerate empty bin),
  - account for the gap: the 12mo model has no events < 12mo before anchor, so
    'last 3mo' = the 3 most-recent AVAILABLE months (gap + 3), not absolute months.

Gated behind env FE_V2=1 so it's additive and A/B-able vs the current design.
Cancer-agnostic: operates on the preprocessed obs/med frames (PATIENT_GUID,
CATEGORY, MONTHS_BEFORE_INDEX) + the existing feature matrix's patient index.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# recency depths (months) added to the gap; plus an all-time total
RECENCY_WINDOWS = [3, 6, 12, 24, 60]
MIN_EVENTS_PER_CAT = 20   # same threshold as build_trend_features


def build_recency_window_features(clin_df, med_df, existing_fm, window_name, cfg):
    """Return a per-patient DataFrame of cumulative recency-window features.

    Columns per qualifying category C (domain OBS/MED):
      RECW_{dom}_{C}_last{W}mo   — cumulative count within (gap + W) months
      RECW_{dom}_{C}_total       — all-time count in the lookback
      RECW_{dom}_{C}_recent_frac — last-6mo share of total (flexes to history)
      RECW_{dom}_{C}_escalation  — recent (last-6mo) rate vs older (6–24mo) rate
                                    (recency-anchored trend; >1 = accelerating)
    """
    patients = existing_fm.index
    rf = pd.DataFrame(index=patients)
    rf.index.name = 'PATIENT_GUID'

    for src, dom in [(clin_df, 'OBS'), (med_df, 'MED')]:
        try:
            d = src.copy()
            d.columns = d.columns.str.strip().str.upper()
            need = {'MONTHS_BEFORE_INDEX', 'CATEGORY', 'PATIENT_GUID'}
            if not need.issubset(d.columns):
                continue
            d['MONTHS_BEFORE_INDEX'] = pd.to_numeric(d['MONTHS_BEFORE_INDEX'], errors='coerce')
            d = d[d['MONTHS_BEFORE_INDEX'].notna() & d['CATEGORY'].notna()]
            # Exclude Approach-B placeholder fillers (no real recency signal)
            d = d[d['CATEGORY'].astype(str) != '__PLACEHOLDER__']
            if d.empty:
                continue
            # near edge of the lookback (the prediction gap), e.g. ~12 for the 12mo model
            gap = max(0, int(np.floor(d['MONTHS_BEFORE_INDEX'].min())))

            # ALL observed categories (OBS + MED) with enough events — comprehensive
            # (A2 spirit; the model decides). The >=MIN_EVENTS_PER_CAT gate below keeps
            # it from exploding on rare categories.
            cats = list(pd.unique(d['CATEGORY']))

            for cat in cats:
                cd = d[d['CATEGORY'] == cat]
                if len(cd) < MIN_EVENTS_PER_CAT:
                    continue
                safe = str(cat)[:28].replace(' ', '_').replace('/', '_')
                wc = {}
                for w in RECENCY_WINDOWS:
                    cnt = cd[cd['MONTHS_BEFORE_INDEX'] <= gap + w].groupby('PATIENT_GUID').size()
                    col = f'RECW_{dom}_{safe}_last{w}mo'
                    rf[col] = rf.index.map(cnt).fillna(0.0).astype(float)
                    wc[w] = rf[col]
                tot = rf.index.map(cd.groupby('PATIENT_GUID').size()).fillna(0.0).astype(float)
                rf[f'RECW_{dom}_{safe}_total'] = tot
                # recent_frac + escalation need the 6mo & 24mo windows; guard in case
                # RECENCY_WINDOWS is edited so this never KeyErrors.
                if 6 in wc:
                    rf[f'RECW_{dom}_{safe}_recent_frac'] = np.where(tot > 0, wc[6] / tot, 0.0)
                if 6 in wc and 24 in wc:
                    recent_rate = wc[6] / 6.0
                    older_rate = (wc[24] - wc[6]).clip(lower=0) / 18.0
                    rf[f'RECW_{dom}_{safe}_escalation'] = np.where(
                        older_rate > 0, recent_rate / older_rate,
                        np.where(recent_rate > 0, 18.0, 0.0))
        except Exception as e:  # noqa: BLE001 — never let v2 features sink the pipeline
            logger.warning(f"  RECENCY-WINDOW v2 ({dom}) skipped: {e}")

    rf = rf.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    rf = rf.loc[:, ~rf.columns.duplicated()]
    logger.info(f"  RECENCY-WINDOW v2 ({window_name}): {rf.shape[1]} features (gap-anchored, cumulative)")
    return rf
