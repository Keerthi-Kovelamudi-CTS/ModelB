"""
v2 cross-pollination FE — feature TYPES proven useful in one cancer, generalised
to all (gated by env FE_V2=1, same as recency-v2; additive + A/B-able).

This file = the MECHANICAL tier (cancer-agnostic, derives inputs from each cfg /
the data). The CLINICAL tier (MIMIC, NICE) lives in _xpoll_clinical.py with
per-cancer definitions.

Mechanical features (per qualifying category, >=MIN_EVENTS cohort-total):
  PERSIST  — XPOLL_PERSIST_{cat}: >=2 events of a symptom category in the recent
             window (gap..gap+6mo). Generalises Melanoma's persistence flags.
  LABPCT   — XPOLL_LABPCT_{lab}_pctile / _top5 / _top10 / _agebandpct: the patient's
             value for a lab category ranked vs the whole cohort (and within their
             10-yr age band). Generalises Prostate's PSA-percentile.
  LABPAIR  — XPOLL_LABPAIR_{a}_x_{b}: product of two labs' "elevated" flags
             (elevated = value >= cohort 75th pct). Generalises Prostate's LABPAIR
             (PSA x ALP etc.). Built for the top labs by cohort coverage.

REFERENCE REUSE (LOAD-OR-FIT, mirrors encoder_mappings.json):
  LABPCT/LABPAIR are cohort-relative, which would (a) make a 1:10 cohort encode
  shared patients differently from 1:1, and (b) skew train-vs-serve (holdout would
  rank vs the holdout cohort). To avoid both, the FIRST (training/1:1) run SAVES the
  cohort lab reference (sorted value arrays + per-age-band arrays + q75) to
  cfg.FE_RESULTS/<window>/xpoll_ref.json. Any later run that finds that file REUSES
  it: percentiles via searchsorted on the saved arrays, elevated via the saved q75,
  and the SAME feature set (same labs / persist cats / pair keys). For continuous
  labs (ties negligible) searchsorted == rank(pct=True), so shared patients match.
"""
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_EVENTS = 20            # cohort-total floor (matches recency-v2)
RECENT_MONTHS = 6          # "recent" window for PERSIST (added to the gap)
MAX_LABS_FOR_PAIRS = 8     # cap pair explosion: top-N labs by coverage


def _ref_path(cfg, window_name):
    try:
        return cfg.FE_RESULTS / str(window_name).lower() / 'xpoll_ref.json'
    except Exception:
        return None


def _lab_categories(obs):
    """Categories that behave like labs: a numeric VALUE present on most events."""
    if 'VALUE' not in obs.columns:
        return []
    v = pd.to_numeric(obs['VALUE'], errors='coerce')
    obs = obs.assign(_v=v)
    cov = obs.groupby('CATEGORY')['_v'].apply(lambda s: s.notna().mean())
    n = obs.groupby('CATEGORY')['_v'].apply(lambda s: s.notna().sum())
    labs = [c for c in cov.index if cov[c] >= 0.5 and n[c] >= MIN_EVENTS]
    # rank by coverage*count so the pair-cap keeps the best-populated labs
    labs.sort(key=lambda c: n[c], reverse=True)
    return labs


def _safe(s, n):
    return str(s)[:n].replace(' ', '_').replace('/', '_')


def build_xpoll_features(clin_df, med_df, existing_fm, window_name, cfg):
    patients = existing_fm.index
    xf = pd.DataFrame(index=patients)
    xf.index.name = 'PATIENT_GUID'

    # LOAD-OR-FIT reference (mirrors encoder_mappings.json)
    rpath = _ref_path(cfg, window_name)
    ref = None
    if rpath is not None:
        try:
            if rpath.exists():
                with open(rpath) as f:
                    ref = json.load(f)
                logger.info(f"  XPOLL: REUSING saved reference {rpath}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  XPOLL: failed to load reference ({e}) — refitting")
            ref = None
    fit = ref is None
    new_ref = {'persist_cats': [], 'labs': {}, 'pair_keys': []}

    try:
        obs = clin_df.copy()
        obs.columns = obs.columns.str.strip().str.upper()
        if not {'CATEGORY', 'PATIENT_GUID', 'MONTHS_BEFORE_INDEX'}.issubset(obs.columns):
            logger.warning("  XPOLL: missing columns — skipped"); return xf
        obs['MONTHS_BEFORE_INDEX'] = pd.to_numeric(obs['MONTHS_BEFORE_INDEX'], errors='coerce')
        obs = obs[obs['MONTHS_BEFORE_INDEX'].notna() & obs['CATEGORY'].notna()]
        obs = obs[obs['CATEGORY'].astype(str) != '__PLACEHOLDER__']
        if obs.empty:
            return xf
        gap = max(0, int(np.floor(obs['MONTHS_BEFORE_INDEX'].min())))

        # ---- PERSIST (symptom categories) ----
        if fit:
            sym = list(getattr(cfg, 'SYMPTOM_CATEGORIES', [])) or list(pd.unique(obs['CATEGORY']))
        else:
            sym = list(ref.get('persist_cats', []))   # reuse: same categories as 1:1
        recent = obs[obs['MONTHS_BEFORE_INDEX'] <= gap + RECENT_MONTHS]
        for cat in sym:
            cd = obs[obs['CATEGORY'] == cat]
            if fit and len(cd) < MIN_EVENTS:
                continue
            safe = _safe(cat, 26)
            cnt = recent[recent['CATEGORY'] == cat].groupby('PATIENT_GUID').size()
            xf[f'XPOLL_PERSIST_{safe}'] = (xf.index.map(cnt).fillna(0) >= 2).astype(int)
            if fit:
                new_ref['persist_cats'].append(cat)

        # ---- LAB percentile / elevated (cohort + age band) ----
        obs['_V'] = pd.to_numeric(obs.get('VALUE'), errors='coerce')
        ageband = None
        if 'AGE_AT_INDEX' in existing_fm.columns:
            ageband = (existing_fm['AGE_AT_INDEX'].fillna(0) // 10 * 10).astype(int)

        if fit:
            labs = [(lab, _safe(lab, 24)) for lab in _lab_categories(obs)]
        else:
            labs = [(d['orig'], safe) for safe, d in ref.get('labs', {}).items()]

        elevated = {}   # safe -> per-patient elevated flag (for pairs)
        for lab, safe in labs:
            ld = obs[(obs['CATEGORY'] == lab) & obs['_V'].notna()]
            pv = ld.groupby('PATIENT_GUID')['_V'].max() if not ld.empty else pd.Series(dtype=float)
            pser = pd.Series(xf.index.map(pv), index=xf.index, dtype=float)
            present = pser.notna()
            if fit and present.sum() < MIN_EVENTS:
                continue

            if fit:
                pct = pser[present].rank(pct=True)
                pser_pct = pser.index.map(pct).astype(float)
                thr = float(pser[present].quantile(0.75))
                ld_ref = {'orig': lab,
                          'sorted': np.sort(pser[present].values).tolist(),
                          'q75': thr, 'ageband': {}}
            else:
                d = ref['labs'][safe]
                sorted_ref = np.asarray(d['sorted'], dtype=float)
                n = max(1, len(sorted_ref))
                vals = pser.values.astype(float)
                pc = np.searchsorted(sorted_ref, vals, side='right') / n
                pser_pct = pd.Series(np.where(present.values, pc, np.nan), index=xf.index)
                thr = float(d['q75'])

            xf[f'XPOLL_LABPCT_{safe}_pctile'] = pser_pct.astype(float).fillna(0.5)
            xf[f'XPOLL_LABPCT_{safe}_top10'] = (xf[f'XPOLL_LABPCT_{safe}_pctile'] >= 0.90).astype(int)
            xf[f'XPOLL_LABPCT_{safe}_top5'] = (xf[f'XPOLL_LABPCT_{safe}_pctile'] >= 0.95).astype(int)

            if ageband is not None:
                ab = ageband.reindex(xf.index)
                if fit:
                    band_pct = pser.groupby(ab).rank(pct=True)
                    xf[f'XPOLL_LABPCT_{safe}_agebandpct'] = band_pct.astype(float).fillna(0.5)
                    for b in pd.unique(ab.dropna()):
                        bvals = pser[(ab == b) & present]
                        ld_ref['ageband'][str(int(b))] = np.sort(bvals.values).tolist()
                else:
                    band_ref = ref['labs'][safe].get('ageband', {})
                    out = pd.Series(np.nan, index=xf.index)
                    for b in pd.unique(ab.dropna()):
                        arr = np.asarray(band_ref.get(str(int(b)), []), dtype=float)
                        m = (ab == b) & present
                        if len(arr) and m.any():
                            out[m] = np.searchsorted(arr, pser[m].values, side='right') / len(arr)
                    xf[f'XPOLL_LABPCT_{safe}_agebandpct'] = out.astype(float).fillna(0.5)

            elevated[safe] = (pser >= thr).fillna(False).astype(int)
            if fit:
                new_ref['labs'][safe] = ld_ref

        # ---- LAB pairs (top labs by coverage) ----
        if fit:
            keys = list(elevated.keys())[:MAX_LABS_FOR_PAIRS]
            new_ref['pair_keys'] = keys
        else:
            keys = [k for k in ref.get('pair_keys', []) if k in elevated]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                xf[f'XPOLL_LABPAIR_{a}_x_{b}'] = (elevated[a] * elevated[b]).astype(int)

        # ---- save reference on the fitting (training/1:1) run ----
        if fit and rpath is not None:
            try:
                rpath.parent.mkdir(parents=True, exist_ok=True)
                with open(rpath, 'w') as f:
                    json.dump(new_ref, f)
                logger.info(f"  XPOLL: SAVED reference -> {rpath} "
                            f"({len(new_ref['labs'])} labs, {len(new_ref['persist_cats'])} persist cats)")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  XPOLL: could not save reference ({e})")

    except Exception as e:  # noqa: BLE001 — never sink the pipeline
        logger.warning(f"  XPOLL mechanical skipped: {e}")

    xf = xf.fillna(0).replace([np.inf, -np.inf], 0)
    xf = xf.loc[:, ~xf.columns.duplicated()]
    logger.info(f"  XPOLL mechanical ({window_name}): {xf.shape[1]} features "
                f"[{'FIT' if fit else 'REUSE'}]")
    return xf
