"""
Lookback experiment for the Lung model: 5yr / 10yr / 20yr / lifetime, on the 12mo 1:1 config.
TRAINING + INTERNAL val/test ONLY (80/10/10 split of the training cohort). No held-out here:
the real-world held-out (500/50k) is a separate, later concern and is NOT run in this pipeline.

Design:
  * The 4 lookbacks are NESTED -> extract LIFETIME events ONCE (cached) and derive shorter
    lookbacks by filtering days_before_anchor. No extra BigQuery scans.
  * SAME PATIENTS across all lookbacks: every lookback's matrix is reindexed to the full
    lifetime cohort; a patient with no events in a shorter window becomes an all-zero row
    ("no recent signal"). So only feature richness varies between lookbacks, not the patient set.

Outputs -> {GAP}mo_1to{NC_RATIO}/lookback/{5yr,10yr,20yr,lifetime}/ (features, model, metrics.csv)
        -> {GAP}mo_1to{NC_RATIO}/lookback/lookback_internal_summary.csv

Run all four:  python run_lookback_experiment.py
Run just one:  python run_lookback_experiment.py 5yr     (reuses cached events)
"""
import os, sys, re, importlib.util
import numpy as np, pandas as pd

# Value-dependent features: for a no-window-data patient, 0 would be a FAKE value (e.g. Hb=0),
# so these are left NaN -> median-imputed (neutral). Everything else (counts/occurrence/flags)
# is genuinely 0 when there are no events.
VALUE_PAT = re.compile(r'_value|_val_|TREND_SLOPE|TREND_CORRELATION|PERCENT_CHANGE|ABSOLUTE_CHANGE|'
                       r'TREND_DIRECTION|_EVENT_AGE|RECENCY_MONTHS|VALUE_ACCEL|\bNLR\b|\bPLR\b|PACK_YEARS|'
                       r'CIGS_PER_DAY|_LATEST|_VMAX|_VMIN|_VMEAN|PCTILE',
                       re.IGNORECASE)   # _val_ catches band/cumulative _val_mean/_val_slope (stay NaN, not fake-0)

ROOT = os.path.dirname(os.path.abspath(__file__))
FE_DIR = os.path.join(ROOT, "2_FE"); MODEL_DIR = os.path.join(ROOT, "3_Modeling")
GCS = "gs://cancer_transformed_ai/lung_lookback"
# NC_RATIO env: "1" -> 1:1 cohort (12mo_1to1.sql), "10" -> 1:10 (12mo_1to10.sql). Each ratio gets
# its OWN output dir + events cache, so 1:1 and 1:10 coexist without clobbering.
NC_RATIO = os.environ.get("NC_RATIO", "1")
# EXP_SUFFIX env: isolates an A/B run's outputs into its own folder (e.g. "_minrec0") WITHOUT
# changing the SQL filename (which keys off NC_RATIO). Default "" = the normal dir.
EXP_SUFFIX = os.environ.get("EXP_SUFFIX", "")
GAP = os.environ.get("GAP", "12")            # "12"->12-month gap (12mo_*.sql); "1"->1-month gap (1mo_*.sql); own dir+cache per gap
LOOKBACK_DIR = os.path.join(ROOT, f"{GAP}mo_1to{NC_RATIO}{EXP_SUFFIX}", "lookback"); os.makedirs(LOOKBACK_DIR, exist_ok=True)
LOOKBACKS = [5, 10, 20, 100]        # years; 100 == lifetime
ID_COLS = ["cancer_class", "sex", "patient_ethnicity_16", "patient_ethnicity_6", "patient_age"]
sys.path.insert(0, FE_DIR); sys.path.insert(0, MODEL_DIR)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

rp      = _load(os.path.join(ROOT, "_run_pipeline.py"), "_run_pipeline")
fe      = _load(os.path.join(FE_DIR, "transform_features.py"), "transform_features")
metrics = _load(os.path.join(MODEL_DIR, "lung_metrics.py"), "lung_metrics")
import json
import enhanced_features as ef        # Phase-2 compute_xpoll (fit-on-train percentile reference)

TRAIN_EVENTS = os.path.join(LOOKBACK_DIR, "train_events_lifetime.csv")


def _tag(L):
    return "lifetime" if L == 100 else f"{L}yr"


def ensure_events():
    """Extract the lifetime TRAIN events ONCE (via _run_pipeline's shared bq export); cache.
    Internal-only pipeline -> no held-out extraction (held-out is a separate, later concern)."""
    if os.path.exists(TRAIN_EVENTS):
        print(f"[cache] reuse {TRAIN_EVENTS}")
    else:
        rp.export_query_to_csv(open(os.path.join(FE_DIR, "SQL", f"{GAP}mo_1to{NC_RATIO}.sql")).read(),
                               TRAIN_EVENTS, gcs_prefix=GCS, tag=f"train_lifetime_{GAP}mo_r{NC_RATIO}")
    return TRAIN_EVENTS


def full_patient_frame(events_df):
    """One row per cohort patient (cleaned guid + ID cols + label) — the FULL patient set that
    every lookback's matrix is aligned to, so all lookbacks share identical patients."""
    f = events_df.drop_duplicates("patient_guid").copy()
    f["patient_guid"] = fe.clean_patient_guid_series(f["patient_guid"])
    keep = ["patient_guid"] + [c for c in ID_COLS if c in f.columns]
    return f[keep].drop_duplicates("patient_guid").reset_index(drop=True)


def build_matrix(events_df, L, out_csv, full_frame, xpoll_fit=True, xpoll_ref_path=None):
    """Filter to the lookback window, build FE matrix (+ Phase-2 xpoll percentiles), then REINDEX
    to the full patient set (missing patients -> all-zero feature row). Returns matrix path.

    xpoll_fit=True  -> FIT the percentile reference on this (train) build + save to xpoll_ref_path.
    xpoll_fit=False -> APPLY a previously-saved reference (held-out build) so labs are ranked vs
                       the TRAINING distribution only (leakage-safe)."""
    dba = pd.to_numeric(events_df["days_before_anchor"], errors="coerce")
    df = events_df[dba <= L * 365.25].copy()
    mat = fe.extract_lung_risk_factors(df, time_window_months=None)
    drop = [c for c in mat.columns if "_SNOMED_CODES" in c or "_terms" in c.lower()]
    mat = mat.drop(columns=drop, errors="ignore")

    # Phase-2 xpoll cohort/age-band lab percentiles — fit on train, apply on held-out.
    xdf = df.copy()
    xdf["patient_guid_CLEAN"] = fe.clean_patient_guid_series(xdf["patient_guid"])
    xdf["event_type_lower"] = (xdf["event_type"].astype(str).str.lower()
                               if "event_type" in xdf.columns else "observation")
    try:
        if xpoll_fit:
            xp, xref = ef.compute_xpoll(xdf, ref=None)
            if xpoll_ref_path:
                json.dump(xref, open(xpoll_ref_path, "w"))
        else:
            xref = json.load(open(xpoll_ref_path)) if (xpoll_ref_path and os.path.exists(xpoll_ref_path)) else None
            xp = ef.compute_xpoll(xdf, ref=xref) if xref is not None else None
        if xp is not None:
            mat = mat.merge(xp, on="patient_guid", how="left")
    except Exception as _e:
        print(f"  [xpoll] skipped: {_e}")

    feat_cols = [c for c in mat.columns if c not in ID_COLS and c != "patient_guid"]
    present = set(mat["patient_guid"])
    aligned = full_frame.merge(mat[["patient_guid"] + feat_cols], on="patient_guid", how="left")
    missing = ~aligned["patient_guid"].isin(present)         # patients with no events in this window
    value_cols = [c for c in feat_cols if VALUE_PAT.search(c)]
    count_cols = [c for c in feat_cols if c not in set(value_cols)]
    aligned.loc[missing, count_cols] = aligned.loc[missing, count_cols].fillna(0)   # no events -> 0 (correct)
    # value/lab/ratio cols stay NaN for no-data patients -> median-imputed downstream (NOT fake-0)
    if "cancer_class" in aligned.columns:
        aligned["cancer_class"] = aligned.pop("cancer_class")
    aligned.to_csv(out_csv, index=False)
    print(f"  [{_tag(L)}] {len(df):,} events; matrix {aligned.shape[0]} patients "
          f"({int(missing.sum())} no-window-data, zero-filled) x {aligned.shape[1]} cols -> {out_csv}")
    return out_csv


def run_one_lookback(L, train_ev_df, full_frame):
    """Train + INTERNAL val/test for a single lookback (no held-out). Returns metric rows."""
    tag = _tag(L); od = os.path.join(LOOKBACK_DIR, tag); os.makedirs(od, exist_ok=True)
    print(f"\n{'='*70}\nLOOKBACK = {tag}  (train + internal val/test)\n{'='*70}")
    tr_feat = build_matrix(train_ev_df, L, os.path.join(od, "train_features.csv"), full_frame,
                           xpoll_fit=True, xpoll_ref_path=os.path.join(od, f"xpoll_ref_{tag}.json"))
    p = rp.run_model(tr_feat, od, tag, "1to1")              # saves model_{tag}_1to1.joblib
    s_int = p.best_model.predict_proba(p.X_test)[:, 1]
    y_int = np.asarray(p.y_test).astype(int)
    # persist raw scores so ANY operating point can be recomputed later without re-running
    np.savez(os.path.join(od, f"internal_scores_{tag}.npz"), y=y_int, s=s_int)
    # persist per-model test scores too → enables per-model bootstrap CIs offline (each base
    # model in p.results carries its y_pred_proba on the SAME internal test set).
    try:
        _pm = {n: np.asarray(r["y_pred_proba"], dtype="float32")
               for n, r in getattr(p, "results", {}).items()
               if r.get("y_pred_proba") is not None}
        if _pm:
            _names = list(_pm.keys())
            np.savez(os.path.join(od, f"per_model_scores_{tag}.npz"), y=y_int,
                     names=np.array(_names, dtype=object),
                     scores=np.vstack([_pm[n] for n in _names]))
            print(f"[per-model] saved test scores for {len(_names)} models -> per_model_scores_{tag}.npz")
    except Exception as _e:
        print(f"[per-model] skipped: {_e}")
    # Free (unpinned) operating point only — Youden's J (max sens+spec-1). No pinned-sensitivity points.
    from sklearn.metrics import roc_curve as _roc
    _fpr, _tpr, _th = _roc(y_int, s_int)
    _Tfree = float(_th[int((_tpr - _fpr).argmax())])
    _rf = metrics.eval_with_ci(y_int, s_int, threshold=_Tfree, label=f"{tag} internal @free(Youden)")
    print("\n" + metrics.format_result(_rf))
    rows = [metrics.results_to_row(_rf, tag, "internal_free")]
    pd.DataFrame(rows).to_csv(os.path.join(od, "metrics.csv"), index=False)
    return rows


def main():
    print("=" * 70 + "\nLOOKBACK SWEEP (5/10/20/lifetime) — 12mo 1:1 — TRAIN + INTERNAL only\n" + "=" * 70)
    tr = ensure_events()
    train_ev = pd.read_csv(tr, low_memory=False)
    full_frame = full_patient_frame(train_ev)
    print(f"full cohort: {len(full_frame):,} patients (same set for every lookback)")
    rows = []
    for L in LOOKBACKS:
        rows += run_one_lookback(L, train_ev, full_frame)
    summ = os.path.join(LOOKBACK_DIR, "lookback_internal_summary.csv")
    pd.DataFrame(rows).to_csv(summ, index=False)
    print(f"\n{'='*70}\nDONE (internal) -> {summ}\nNext: pick a lookback from the internal summary (held-out evaluation is a separate step).\n{'='*70}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        main()
    else:
        L = {"5yr": 5, "10yr": 10, "20yr": 20, "lifetime": 100}.get(arg)
        assert L is not None, f"unknown lookback '{arg}' (use 5yr|10yr|20yr|lifetime|all)"
        tr = ensure_events()
        train_ev = pd.read_csv(tr, low_memory=False)
        run_one_lookback(L, train_ev, full_patient_frame(train_ev))
