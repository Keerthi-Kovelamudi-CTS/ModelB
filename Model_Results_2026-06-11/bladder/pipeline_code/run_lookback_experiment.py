"""
Bladder HYBRID lookback experiment: the MATURE bladder FE engine (8-phase, ~1-2k rich
features) + the IMPROVED lung-style modeling chain (scale_pos_weight, cumimp99, isotonic
per-age-band calibration, 80/10/10). TRAIN + INTERNAL val/test only — the real-world
held-out (2000/50k) is a SEPARATE inference step (evaluate_holdout.py).

Lookbacks: 5yr / 10yr / 20yr / lifetime, on the 12mo 1:1 cohort. The 4 lookbacks are NESTED
— extract LIFETIME events ONCE (cached) and derive shorter windows by filtering
days_before_anchor. SAME PATIENTS across all lookbacks: a patient with no events inside the
window is carried through as a stub so the mature FE's __PLACEHOLDER__ mechanism gives them
the standard all-absent feature row (exactly how they'd be scored in deployment).

Flow per lookback L:
  filter events days_before_anchor<=L*365.25  (+ stub rows for no-window-data patients)
  -> bridge schema to UPPERCASE              (the mature 0_preprocess expects UPPER cols)
  -> 0_preprocess.py                          (categorise via codelist + A/B window + placeholders)
  -> _sanity_check.run_sanity_check           (split obs/med, dedup, drops)
  -> _clean_data.run_clean_data               (leak guard: EVENT_DATE < INDEX_DATE)
  -> run_pipeline.run_feature_engineering     (8-phase) -> feature_matrix_final_12mo.parquet
  -> modeling schema (patient_guid/cancer_class, numeric features) -> rp.run_model -> eval_with_ci

Each lookback runs the file-based mature FE in its OWN isolated work dir (config paths are
re-pointed per lookback) so the 4 lookbacks never clobber each other's parquet.

Outputs -> 12mo_1:1/lookback_mature/{5yr,10yr,20yr,lifetime}/ (features, model, metrics.csv)
        -> 12mo_1:1/lookback_mature/lookback_internal_summary.csv

Run all four:  cd B+C_Newmodel && python run_lookback_experiment.py
Run just one:  python run_lookback_experiment.py 5yr
Held-out eval: python evaluate_holdout.py 20yr        (after picking a lookback)
"""
import os, sys, re, subprocess, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FE_DIR = os.path.join(ROOT, "FE")
MODEL_DIR = os.path.join(ROOT, "Modeling")
CODELIST = os.path.join(ROOT, "codelist2.0", "code_category_mapping_2.0.json")
GCS = "gs://cancer_transformed_ai/bladder_lookback"
NC_RATIO = os.environ.get("NC_RATIO", "1")   # "1"->1:1 (<GAP>mo_1to1.sql), "10"->1:10 (<GAP>mo_1to10.sql); own dir+cache per ratio
GAP = os.environ.get("GAP", "12")            # "12"->12-month gap (12mo_*.sql); "1"->1-month gap (1mo_*.sql); own dir+cache per gap
LOOKBACK_DIR = os.path.join(ROOT, f"{GAP}mo_1:{NC_RATIO}", "lookback_mature")
os.makedirs(LOOKBACK_DIR, exist_ok=True)
LOOKBACKS = [5, 10, 20, 100]        # years; 100 == lifetime
TARGET_SENS = 0.90
TARGET_SENS_LIST = [0.90, 0.93, 0.95]   # internal report: Spec/PPV at each sensitivity

# Value-dependent mature-FE features: for a no-window-data patient a fabricated 0 would be a
# FAKE measurement (e.g. oestradiol=0), so these stay NaN -> median-imputed downstream.
# (Stub patients are normally handled by the mature FE's placeholder rows; this regex only
# governs the belt-and-braces fill if any cohort patient is still missing after FE.)
VALUE_PAT = re.compile(r'VAL_|_LATEST|_VMAX|_VMIN|_VMEAN|SLOPE|TRAJ|PCTILE|XPOLL_LABP|_VALUE|\bNLR\b|\bPLR\b',
                       re.IGNORECASE)

sys.path.insert(0, ROOT)
sys.path.insert(0, FE_DIR)
sys.path.insert(0, MODEL_DIR)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


rp      = _load(os.path.join(ROOT, "_run_pipeline.py"), "_run_pipeline")
metrics = _load(os.path.join(MODEL_DIR, "bladder_metrics.py"), "bladder_metrics")

# Mature FE modules (FE_DIR is on sys.path). run_pipeline imports `config` + the digit-prefixed
# phase modules; importing it here wires the whole mature FE. We re-point config's paths per
# lookback before invoking the steps (run_pipeline + phases read the SAME config object).
import config as bladder_cfg            # FE/config.py  (CANCER_NAME/DATA_PREFIX='bladder')
import _sanity_check as sanity_mod     # run_sanity_check(cfg)
import _clean_data as clean_mod        # run_clean_data(cfg)
import run_pipeline as fe_pipeline     # run_feature_engineering(cfg)
import bladder_extra_features as extra   # problem-list flags + NLR/PLR (post-FE enrichment from raw events)

TRAIN_EVENTS   = os.path.join(LOOKBACK_DIR, "train_events_lifetime.csv")
HELDOUT_EVENTS = os.path.join(LOOKBACK_DIR, "heldout_events_lifetime.csv")

# lowercase SQL columns -> UPPERCASE expected by 0_preprocess (which then renames
# ANCHOR_DATE->INDEX_DATE, PATIENT_AGE->AGE_AT_INDEX, CANCER_CLASS->LABEL itself).
UPPER_MAP = {
    'patient_guid': 'PATIENT_GUID', 'anchor_date': 'ANCHOR_DATE', 'cancer_class': 'CANCER_CLASS',
    'sex': 'SEX', 'patient_ethnicity_16': 'PATIENT_ETHNICITY_16', 'patient_ethnicity_6': 'PATIENT_ETHNICITY_6',
    'patient_age': 'PATIENT_AGE', 'age_at_anchor': 'AGE_AT_ANCHOR', 'event_date': 'EVENT_DATE',
    'event_age': 'EVENT_AGE', 'days_before_anchor': 'DAYS_BEFORE_ANCHOR', 'event_type': 'EVENT_TYPE',
    'snomed_c_t_concept_id': 'SNOMED_C_T_CONCEPT_ID', 'term': 'TERM', 'associated_text': 'ASSOCIATED_TEXT',
    'value': 'VALUE', 'med_code_id': 'MED_CODE_ID', 'drug_term': 'DRUG_TERM',
}


def _tag(L):
    return "lifetime" if L == 100 else f"{L}yr"


def _clean_guid(s):
    """Match the cohort guid convention: strip braces/quotes, upper-case."""
    return s.astype(str).str.replace(r'[{}"]', '', regex=True).str.strip().str.upper()


def bridge_schema(df):
    return df.rename(columns={c: UPPER_MAP.get(c.lower(), c.upper()) for c in df.columns})


def ensure_events():
    """Extract lifetime train + held-out events ONCE (via _run_pipeline's shared bq export); cache."""
    if os.path.exists(TRAIN_EVENTS):
        print(f"[cache] reuse {TRAIN_EVENTS}")
    else:
        rp.export_query_to_csv(open(os.path.join(FE_DIR, "SQL", f"{GAP}mo_1to{NC_RATIO}.sql")).read(),
                               TRAIN_EVENTS, gcs_prefix=GCS, tag=f"bladder_train_lifetime_{GAP}mo_r{NC_RATIO}")
    if os.path.exists(HELDOUT_EVENTS):
        print(f"[cache] reuse {HELDOUT_EVENTS}")
    else:
        rp.export_query_to_csv(open(os.path.join(FE_DIR, "SQL", f"heldout_test_{GAP}mo.sql")).read(),
                               HELDOUT_EVENTS, gcs_prefix=GCS, tag=f"bladder_heldout_lifetime_{GAP}mo")
    return TRAIN_EVENTS, HELDOUT_EVENTS


def full_patient_frame(events_df):
    """One row per cohort patient (cleaned guid + label) — the FULL set every lookback aligns to."""
    f = events_df.drop_duplicates("patient_guid").copy()
    f["patient_guid"] = _clean_guid(f["patient_guid"])
    cc = pd.to_numeric(f["cancer_class"], errors="coerce").astype("Int64")
    return (pd.DataFrame({"patient_guid": f["patient_guid"].values, "cancer_class": cc.values})
            .drop_duplicates("patient_guid").reset_index(drop=True))


def _set_cfg_paths(work):
    """Re-point the mature FE config at an isolated per-lookback work dir (covers both the
    cfg-arg and module-level `import config` access patterns since it's the same object)."""
    work = Path(work)
    bladder_cfg.BASE_PATH       = work / "data"
    bladder_cfg._RESULTS_ROOT   = work / "results"
    bladder_cfg.FE_RESULTS      = work / "results" / "3_feature_engineering"
    bladder_cfg.SANITY_RESULTS  = work / "results" / "1_sanity_check"
    bladder_cfg.CLEANUP_RESULTS = work / "results" / "5_cleanup"
    bladder_cfg.WINDOWS         = ["12mo"]
    for p in (bladder_cfg.BASE_PATH / "12mo", bladder_cfg.BASE_PATH / "raw",
              bladder_cfg.FE_RESULTS, bladder_cfg.SANITY_RESULTS, bladder_cfg.CLEANUP_RESULTS):
        p.mkdir(parents=True, exist_ok=True)


def run_mature_fe(events_df, L, work):
    """Filter to the lookback, keep ALL cohort patients (stub no-window-data ones so the mature
    FE placeholders them), run the mature FE chain, return the feature matrix in modeling schema
    (patient_guid + cancer_class + features)."""
    dba = pd.to_numeric(events_df["days_before_anchor"], errors="coerce")
    df = events_df[dba <= L * 365.25].copy()

    # carry no-window-data patients through as stubs (uncategorised code -> 0_preprocess placeholders them)
    kept = set(df["patient_guid"].astype(str))
    miss = events_df.drop_duplicates("patient_guid")
    miss = miss[~miss["patient_guid"].astype(str).isin(kept)].copy()
    if len(miss):
        miss["snomed_c_t_concept_id"] = 0      # not in the codelist -> NULL category -> placeholder row
        if "med_code_id" in miss.columns: miss["med_code_id"] = pd.NA
        miss["event_type"] = "observation"
        miss["value"] = pd.NA
        miss["days_before_anchor"] = 0
        df = pd.concat([df, miss[df.columns]], ignore_index=True)

    df = bridge_schema(df)
    _set_cfg_paths(work)

    raw_csv = str(Path(bladder_cfg.BASE_PATH) / "raw" / "bladder_12mo_raw.csv")
    df.to_csv(raw_csv, index=False)
    out_dir = str(Path(bladder_cfg.BASE_PATH) / "12mo")

    # 0_preprocess.py is argparse-only -> drive via subprocess; writes {out_dir}/bladder_12mo.parquet
    r = subprocess.run([sys.executable, os.path.join(FE_DIR, "0_preprocess.py"),
                        "--raw", raw_csv, "--out-dir", out_dir,
                        "--mapping", CODELIST, "--prefix", "bladder_12mo"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"0_preprocess failed:\n{r.stderr[-1800:]}"

    sanity_mod.run_sanity_check(bladder_cfg)      # split obs/med, dedup, drops
    clean_mod.run_clean_data(bladder_cfg)         # leak guard
    fe_pipeline.run_feature_engineering(bladder_cfg)   # 8-phase FE

    fm_path = Path(bladder_cfg.FE_RESULTS) / "12mo" / "feature_matrix_final_12mo.parquet"
    fm = pd.read_parquet(fm_path).reset_index()  # index is PATIENT_GUID
    fm = fm.rename(columns={"PATIENT_GUID": "patient_guid", "LABEL": "cancer_class"})
    return fm


def build_matrix(events_df, L, out_csv, full_frame):
    """Mature-FE matrix -> modeling-ready CSV (numeric features, cancer_class last), aligned to
    the full cohort (stubs already keep everyone; any residual missing -> count 0 / value NaN)."""
    fm = run_mature_fe(events_df, L, os.path.join(LOOKBACK_DIR, f"fe_work_{_tag(L)}"))
    fm["patient_guid"] = _clean_guid(fm["patient_guid"])
    # SAFETY DEDUP: the mature FE indexes by PATIENT_GUID (one row/patient, no multi-key
    # merge), so it does NOT have the lung-FE fan-out bug — but guard anyway so a patient can
    # never be double-counted or leaked across the train/test split.
    _n0 = len(fm)
    fm = fm.groupby("patient_guid", as_index=False, sort=False).first()
    if len(fm) < _n0:
        print(f"[dedup] collapsed {_n0 - len(fm)} duplicate rows -> {len(fm)} unique patients")
    fm = fm.drop(columns=["cancer_class"], errors="ignore")            # take the label from full_frame

    # problem-list flags + NLR/PLR from the lookback-filtered raw events (signals the mature FE
    # drops at 0_preprocess: problem_status_description / significance_description + FBC values)
    dba = pd.to_numeric(events_df["days_before_anchor"], errors="coerce")
    extras = extra.compute_extras(events_df[dba <= L * 365.25].copy())
    fm = fm.merge(extras, on="patient_guid", how="left")

    feats = fm.drop(columns=["patient_guid"], errors="ignore").select_dtypes(include=[np.number])
    fm = pd.concat([fm["patient_guid"], feats], axis=1)
    feat_cols = list(feats.columns)

    aligned = full_frame.merge(fm, on="patient_guid", how="left")
    present = set(fm["patient_guid"])
    missing = ~aligned["patient_guid"].isin(present)
    value_cols = [c for c in feat_cols if VALUE_PAT.search(c)]
    count_cols = [c for c in feat_cols if c not in set(value_cols)]
    aligned.loc[missing, count_cols] = aligned.loc[missing, count_cols].fillna(0)
    cc = aligned.pop("cancer_class"); aligned["cancer_class"] = cc     # label last
    aligned.to_csv(out_csv, index=False)
    print(f"  [{_tag(L)}] matrix {aligned.shape[0]} patients "
          f"({int(missing.sum())} no-window-data) x {aligned.shape[1]} cols "
          f"({len(value_cols)} value-cols) -> {out_csv}")
    return out_csv


def run_one_lookback(L, train_ev_df, full_frame):
    tag = _tag(L); od = os.path.join(LOOKBACK_DIR, tag); os.makedirs(od, exist_ok=True)
    print(f"\n{'='*70}\nLOOKBACK = {tag}  (mature FE + improved modeling, train + internal)\n{'='*70}")
    tr_feat = build_matrix(train_ev_df, L, os.path.join(od, "train_features.csv"), full_frame)
    p = rp.run_model(tr_feat, od, tag, "1to1")                  # saves model_{tag}_1to1.joblib
    s_int = p.best_model.predict_proba(p.X_test)[:, 1]
    y_int = np.asarray(p.y_test).astype(int)
    np.savez(os.path.join(od, f"internal_scores_{tag}.npz"), y=y_int, s=s_int)   # any operating point later
    # per-model test scores → per-model bootstrap CIs offline (each base model in p.results).
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
    rows = []
    # Free (unpinned) operating point — Youden's J (max sens+spec-1); the PRIMARY report point.
    from sklearn.metrics import roc_curve as _roc
    _fpr, _tpr, _th = _roc(y_int, s_int)
    _Tfree = float(_th[int((_tpr - _fpr).argmax())])
    _rf = metrics.eval_with_ci(y_int, s_int, threshold=_Tfree, label=f"{tag} internal @free(Youden)")
    print("\n" + metrics.format_result(_rf))
    rows.append(metrics.results_to_row(_rf, tag, "internal_free"))
    for ts in TARGET_SENS_LIST:                              # Spec/PPV at Sens 0.90 / 0.93 / 0.95
        T = metrics.threshold_at_sensitivity(y_int, s_int, ts)
        r = metrics.eval_with_ci(y_int, s_int, threshold=T, label=f"{tag} internal @Sens>={ts:.2f}")
        print("\n" + metrics.format_result(r))
        rows.append(metrics.results_to_row(r, tag, f"internal_sens{int(round(ts*100))}"))
    pd.DataFrame(rows).to_csv(os.path.join(od, "metrics.csv"), index=False)
    return rows


def main():
    print("=" * 70 + "\nBLADDER HYBRID LOOKBACK SWEEP (5/10/20/lifetime) — mature FE + improved modeling\n" + "=" * 70)
    tr, _ = ensure_events()
    train_ev = pd.read_csv(tr, low_memory=False)
    full_frame = full_patient_frame(train_ev)
    print(f"full cohort: {len(full_frame):,} patients (same set for every lookback)")
    rows = []
    for L in LOOKBACKS:
        rows += run_one_lookback(L, train_ev, full_frame)
    summ = os.path.join(LOOKBACK_DIR, "lookback_internal_summary.csv")
    pd.DataFrame(rows).to_csv(summ, index=False)
    print(f"\n{'='*70}\nDONE (internal) -> {summ}\nNext: pick a lookback, then `python evaluate_holdout.py <lookback>`\n{'='*70}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        main()
    else:
        L = {"5yr": 5, "10yr": 10, "20yr": 20, "lifetime": 100}.get(arg)
        assert L is not None, f"unknown lookback '{arg}' (use 5yr|10yr|20yr|lifetime|all)"
        tr, _ = ensure_events()
        train_ev = pd.read_csv(tr, low_memory=False)
        run_one_lookback(L, train_ev, full_patient_frame(train_ev))
