"""
Held-out evaluation + Platt recalibration for the data-driven Lung model — touch-once labelled cohort.

Runs the FE + stable-feature model on the held-out cohort (the 500 cancer / 50k non-cancer):

  SQL/heldout_test_{GAP}mo.sql  --(BigQuery)-->  events
   -> build(h, sql_path=heldout SQL, fit_split=False) via the SAME FE engine as training (pandas
      build_features.py; SAME codelist + families, at the model's lookback; nothing fit on held-out, leak-safe)
   -> transform_external (TRAIN-fitted encoders/median-impute/scaler from the saved model) -> X
   -> raw = model["model"].predict_proba(X)             (the uncalibrated ensemble's raw scores)

  Honest reporting via a DISJOINT split of the held-out (stratified, fixed seed) — fitting Platt and
  choosing the operating threshold on the SAME patients we then report on would bias the metrics:
   -> CALIB (30%): fit the Platt recalibrator + pick the Youden operating threshold here only.
   -> TEST  (70%): report AUROC/AUPRC + Sens/Spec/PPV/NPV @ that threshold, plus Brier/ECE — all on
      patients NOT used to fit Platt or choose the threshold, so the numbers are unbiased.
   -> the CALIB-fit Platt is saved as the deployable recalibration artifact.

Run (VM/env with BigQuery + the trained model at ../3_Modeling_outputs/{Astra|Nova}/{yr}yr_OFF/model_{h}.joblib):
    python evaluate_heldout.py 12mo 5      # horizon [years lookback]; default 12mo, config FE_YEARS_BEFORE
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
FE_DIR = os.path.join(HERE, "..", "2_FE")
sys.path.insert(0, os.path.dirname(HERE))      # V3 root, for config + artifacts
import config as C
import artifacts                               # atomic writes (no partial Platt/report/features on interruption)

# 1st positional arg = horizon; hard-error on a typo (don't silently fall back to 12mo).
if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and not sys.argv[1].isdigit() \
        and sys.argv[1] not in ("12mo", "1mo"):
    raise SystemExit(f"evaluate_heldout: unrecognized horizon '{sys.argv[1]}' (valid: 12mo, 1mo)")
WINDOW = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("12mo", "1mo") else "12mo"
GAP = WINDOW.replace("mo", "")
# 2nd arg = FE lookback window in years (matches the model being evaluated). REQUIRED — no year is
# hardcoded; the run selects it. (Falls back to FE_YEARS_BEFORE only if it's been set > 0.)
YEARS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else int(getattr(C, "FE_YEARS_BEFORE", 0) or 0)
if YEARS <= 0:
    raise SystemExit(f"evaluate_heldout: pass the lookback window in years, e.g. `evaluate_heldout.py {WINDOW} 10` "
                     f"(no default is hardcoded).")
# --force (or env HELDOUT_FORCE=1) rebuilds the held-out FE matrix from scratch instead of reusing the
# cached parquet — matches `run_v3.py --force` (which now passes it through). Without it the cached
# matrix is reused, which is the right default but goes STALE if the FE code changed since it was built.
FORCE = ("--force" in sys.argv) or (os.environ.get("HELDOUT_FORCE", "").lower() in ("1", "true", "yes"))
OUT_DIR = os.path.join(HERE, "..", "3_Modeling_outputs", C.artifact_subdir(WINDOW, YEARS))
MODEL = os.path.join(OUT_DIR, f"model_{WINDOW}.joblib")
SQL = os.path.join(HERE, "SQL", f"heldout_test_{GAP}mo.sql")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _transform_external(df_raw, md):
    """Replicate LungCancerPredictor.transform_external using the SAVED model dict (no class instance):
    encoders -> reindex to TRAIN feature_names -> value cols filled w/ TRAIN median, rest -> 0 -> scale.
    Returns the scaled array the base ensemble expects (identical pipeline to training, no refit)."""
    d = df_raw.copy()
    for col, mp in (md.get("encoders") or {}).items():
        d[col] = (d[col].fillna("Unknown").astype(str).map(mp).fillna(-1) if col in d.columns else -1)
    d = d.reindex(columns=md["feature_names"])               # same columns/order as train; missing -> NaN
    d = d.apply(pd.to_numeric, errors="coerce").astype("float64")   # never nullable Int64 (parquet cache)
    vcols = set(md.get("value_cols") or [])
    val = [c for c in md["feature_names"] if c in vcols]
    if val and md.get("impute_medians") is not None:
        d[val] = d[val].fillna(md["impute_medians"])         # value -> TRAIN median
    d = d.replace([np.inf, -np.inf], np.nan)                 # inf -> NaN (then median/0 fill), scaler-safe
    return md["scaler"].transform(d.fillna(0.0).values)      # remaining (count) -> 0, then TRAIN scaler


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    assert os.path.exists(SQL), f"held-out SQL missing: {SQL}"
    assert os.path.exists(MODEL), f"trained model missing: {MODEL} (run run_v3.py first)"
    print(f"{'='*70}\nHELD-OUT EVAL + PLATT — {WINDOW} {YEARS}yr (gap {GAP}mo)\n{'='*70}")

    # 1) held-out FE -> transform to the model's features -> raw scores. CHUNKED (config.HELDOUT_CHUNK>0)
    #    builds + scores in patient-batches so the wide (~290k-col) FE matrix is NEVER materialized:
    #    each batch's wide matrix is reduced to the model's feature columns (transform_external) and the
    #    small reduced X is accumulated. Peak memory = one batch's wide matrix (+ the reduced full X).
    #    The schema (value_codes / cross-top_cats / cohort-global month-zero) is computed ONCE on the full
    #    held-out cohort and reused for every batch, so chunked output == non-chunked output exactly.
    # held-out FE matrix lives WITH the training FE under 2_FE/output/ (parallel to features_p005_{h}.parquet),
    # not in the modeling-output dir — so both the internal and held-out FE matrices sit in the FE folder.
    ho_matrix = os.path.join(FE_DIR, "output", C.artifact_subdir(WINDOW, YEARS), f"heldout_features_{WINDOW}.parquet")
    if FORCE and os.path.exists(ho_matrix):
        os.remove(ho_matrix)
        print(f"[heldout] --force: deleted cached matrix ({ho_matrix})")
    # FE engine: pandas build_features.py only (Polars removed). Same engine as training.
    fe = _load(os.path.join(FE_DIR, "build_features.py"), "fe_engine")
    # Register lung_training first so joblib can unpickle CalibratedLungModel (pickled __module__ == "lung_training").
    _load(os.path.join(HERE, "..", "3_Modeling", "lung_training.py"), "lung_training")
    md = joblib.load(MODEL)
    # C1 guard: rebuild held-out FE with the SAME flags the model was TRAINED under (align config.* from the
    # model's stamped fe_config). Without this, a model trained with an arm ON but evaluated under the default
    # config would silently NaN-impute its trained features and report degraded metrics.
    _fec = md.get("fe_config")
    if _fec:
        for _k, _v in _fec.items():
            if _v is not None and getattr(C, _k, None) != _v:
                print(f"[heldout] aligning FE flag {_k}: config={getattr(C, _k, None)} -> model-trained={_v}")
                setattr(C, _k, _v)
    CHUNK = int(getattr(C, "HELDOUT_CHUNK", 0) or 0)

    # staleness guard for the cached matrix. stage_heldout busts the *report* when the model changes
    # but NOT this parquet, so without --force a matrix built for a PREVIOUS feature set would be reused and
    # silently median/0-imputed by transform_external -> degraded metrics. The keep-features build reindexes to the
    # model's feature_names, so a fresh same-model cache contains all of them; if the cache is missing a
    # non-trivial fraction, it was built for a different model -> rebuild instead of trusting it.
    # the matrix also carries an FE-LOGIC staleness manifest (content-hash of the FE source in its
    # upstream). the column-presence check catches COLUMN drift (model feature-set change); the FE-source content-hash manifest catches VALUE drift — an FE-logic
    # edit (e.g. how _decay_intensity is computed) with identical column names. Without it, only --force
    # (or the column check) would rebuild, so a logic change could be scored on a stale-valued cache.
    _fe_src = os.path.join(FE_DIR, "build_features.py")   # pandas-only (Polars removed)
    # PULL-before-rebuild (aligns with the internal run_v3 stages' ensure_local): if the matrix isn't local
    # but a GCS-published one exists, download it instead of rebuilding from scratch. The staleness checks
    # below still validate it (column-presence + FE-source hash) and rebuild if it's stale. Skipped under
    # --force (we explicitly want a fresh build then). Reads are always allowed (GCS_WRITE gates writes only).
    if not FORCE and getattr(C, "GCS_ARTIFACTS", ""):
        _gcs_matrix = (C.GCS_ARTIFACTS.rstrip("/") + "/2_FE/output/"
                       + C.artifact_subdir(WINDOW, YEARS) + f"/heldout_features_{WINDOW}.parquet")
        artifacts.ensure_local(ho_matrix, _gcs_matrix)
    _use_cache = os.path.exists(ho_matrix)
    if _use_cache:
        mat = pd.read_parquet(ho_matrix)
        _miss = [f for f in md["feature_names"] if f not in mat.columns]
        # FE-LOGIC staleness — compare ONLY the FE-source content hash (mirrors events._cache_fresh),
        # DELIBERATELY NOT git_sha (build_key embeds it, but checking it would force a full held-out rebuild
        # on every commit, even docs). A build_features edit changes the file hash -> rebuild; an unrelated
        # commit does not. Legacy matrices without a manifest are trusted (backward compatible).
        _m = artifacts.read_manifest(ho_matrix)
        _fe_drift = (_m is not None and
                     (_m.get("upstream") or {}).get(os.path.basename(_fe_src)) != artifacts.file_hash(_fe_src))
        # M5: FE-FLAG drift — cached matrix built under different FE flags than THIS model's stamped fe_config
        # (an arm toggled without --force keeps column names, so the column + FE-source checks pass and a
        # stale-valued matrix would be scored). Belt to the --force braces.
        _fec_want = (artifacts.build_key(config_slice=(md.get("fe_config") or {})).get("config") or {})
        _fe_cfg_drift = (_m is not None and _fec_want and (_m.get("config") or {}) != _fec_want)
        if len(_miss) > 0.01 * max(1, len(md["feature_names"])):
            print(f"[heldout] cached matrix is STALE for this model — {len(_miss)}/{len(md['feature_names'])} "
                  f"feature(s) absent (model retrained with a different feature set?) -> rebuilding")
            os.remove(ho_matrix); _use_cache = False
        elif _fe_drift:
            print(f"[heldout] cached matrix is STALE — FE source ({os.path.basename(_fe_src)}) changed since "
                  f"it was built -> rebuilding (git_sha is intentionally ignored)")
            os.remove(ho_matrix); _use_cache = False
        elif _fe_cfg_drift:
            print(f"[heldout] cached matrix is STALE — built under different FE flags than this model's "
                  f"fe_config -> rebuilding")
            os.remove(ho_matrix); _use_cache = False
    if _use_cache:
        print(f"[heldout] cached matrix: {mat.shape[0]:,} patients x {mat.shape[1]:,} cols")
        y = mat["cancer_class"].astype(int).to_numpy()
        X = _transform_external(mat, md)
        raw = md["model"].predict_proba(X)[:, 1]
    elif CHUNK > 0:
        shared = fe.build(WINDOW, sql_path=SQL, fit_split=False, years=YEARS, schema_only=True)
        pids = shared["roster"]["patient_guid"].drop_duplicates().to_numpy()
        nb = int(np.ceil(len(pids) / CHUNK))
        print(f"[heldout] CHUNKED held-out FE+score: {len(pids):,} patients in {nb} chunk(s) of {CHUNK} "
              f"(wide matrix never materialized)")
        Xs, ys, seen = [], [], []
        for bi in range(nb):
            batch = set(pids[bi * CHUNK:(bi + 1) * CHUNK])
            mb, _ = fe.build(WINDOW, sql_path=SQL, fit_split=False, years=YEARS,
                             patient_ids=batch, shared=shared, keep_features=md["feature_names"])
            # HARD-FAIL (no fallback): a chunk must return EXACTLY its requested patients, once each. A
            # silently-dropped/duplicated/misrouted patient would corrupt the accumulated matrix.
            _g = mb["patient_guid"].astype(str).to_numpy()
            if len(_g) != len(set(_g)):
                raise SystemExit(f"[heldout] chunk {bi+1}/{nb} produced DUPLICATE patients -> abort (no fallback)")
            if set(_g) != {str(b) for b in batch}:
                raise SystemExit(f"[heldout] chunk {bi+1}/{nb} patient set != requested batch "
                                 f"(got {len(_g)}, asked {len(batch)}) -> abort (no fallback)")
            _mf = [f for f in md["feature_names"] if f not in mb.columns]
            if _mf:
                raise SystemExit(f"[heldout] chunk {bi+1}/{nb} missing {len(_mf)} model feature column(s) "
                                 f"e.g. {_mf[:5]} — keep-features build is broken -> abort (no fallback)")
            Xs.append(_transform_external(mb, md)); ys.append(mb["cancer_class"].astype(int).to_numpy()); seen.extend(_g)
            print(f"[heldout]   chunk {bi + 1}/{nb}: {mb.shape[0]:,} patients -> X {Xs[-1].shape}")
        if len(seen) != len(set(seen)) or set(seen) != {str(x) for x in pids}:
            raise SystemExit(f"[heldout] chunked patient set incomplete/duplicated across chunks "
                             f"(accumulated {len(seen)} / {len(set(seen))} unique vs {len(pids)} expected) -> abort")
        X = np.vstack(Xs); y = np.concatenate(ys)
        raw = md["model"].predict_proba(X)[:, 1]
        print(f"[heldout] chunked done: accumulated X {X.shape} ({len(set(seen)):,} unique patients, integrity OK)")
    else:
        mat, _ = fe.build(WINDOW, sql_path=SQL, fit_split=False, years=YEARS,
                          keep_features=md["feature_names"])   # keep-features build ONLY the model's features (touch-once; nothing fit on held-out)
        # HARD-FAIL (no fallback): a FRESH build must carry every model feature as a column and no duplicate
        # patients. This is the "doesn't match stable features even after rerun" case — error loudly here
        # rather than let _transform_external silently median/0-impute a structurally-missing column.
        _mf = [f for f in md["feature_names"] if f not in mat.columns]
        if _mf:
            raise SystemExit(f"[heldout] fresh build is MISSING {len(_mf)}/{len(md['feature_names'])} model "
                             f"feature column(s) e.g. {_mf[:5]} — keep-features build broken -> abort (no silent impute)")
        if not mat["patient_guid"].astype(str).is_unique:
            raise SystemExit("[heldout] fresh build produced DUPLICATE patient_guids -> abort (no fallback)")
        os.makedirs(os.path.dirname(ho_matrix), exist_ok=True)
        artifacts.atomic_write(ho_matrix, lambda t: mat.to_parquet(t, index=False))
        artifacts.write_manifest(ho_matrix, config_slice=(md.get("fe_config") or {}), upstream=[_fe_src])   # FE-source hash (value drift) + fe_config (flag drift, M5)
        print(f"[heldout] built matrix: {mat.shape[0]:,} patients x {mat.shape[1]:,} cols -> {ho_matrix}")
        y = mat["cancer_class"].astype(int).to_numpy()
        X = _transform_external(mat, md)
        raw = md["model"].predict_proba(X)[:, 1]
    print(f"[heldout] {len(y):,} patients ({int(y.sum())} cancer / {int((1-y).sum())} non-cancer)")

    # cohort-integrity guards (fail loud; deliberately NOT a hardcoded cohort size — a legitimate cohort
    # change shouldn't crash the eval). A truncated BQ pull, a join that multiplied patients, or an
    # impute/scale bug would otherwise silently inflate or corrupt the reported metrics:
    #   - both classes present (else AUROC/PPV are undefined / meaningless)
    #   - the scored matrix X is finite (transform_external median/0-fills, so any NaN/inf = real bug)
    #   - no duplicated patients in the FE matrix (non-chunked path, where the full matrix is in scope)
    assert int(y.sum()) > 0 and int((1 - y).sum()) > 0, \
        f"[heldout] cohort is single-class (pos={int(y.sum())}/{len(y)}) — cannot evaluate"
    assert np.isfinite(X).all(), \
        "[heldout] non-finite values in scored matrix X after transform_external (impute/scale bug)"
    if 'mat' in dir() and "patient_guid" in getattr(mat, "columns", []):
        _dups = int(mat["patient_guid"].duplicated().sum())
        assert _dups == 0, f"[heldout] {_dups} duplicated patient_guid in FE matrix (preserve-all/join bug)"
    _expN = int(getattr(C, "HELDOUT_EXPECTED_N", 0) or 0)
    if _expN and len(y) != _expN:
        print(f"[heldout] WARNING: cohort size {len(y):,} != expected {_expN:,} "
              f"(config.HELDOUT_EXPECTED_N) — possible truncated/partial pull; metrics may be off")

    # 3) DISJOINT calib/test split of the held-out (stratified, fixed seed). Fitting Platt and
    #    choosing the operating threshold on the same patients we report on would bias the metrics,
    #    so we carve a 30% CALIB slice for both of those, and report ONLY on the 70% TEST slice.
    cal_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=0.70, random_state=getattr(C, "RANDOM_STATE", 42), stratify=y)
    raw_cal, raw_te = raw[cal_idx], raw[te_idx]
    y_cal, y_te = y[cal_idx], y[te_idx]
    print(f"[heldout] calib {len(y_cal):,} ({int(y_cal.sum())} cancer) / "
          f"test {len(y_te):,} ({int(y_te.sum())} cancer)  (stratified, seed 42)")

    # 4) fit the deployable Platt recalibrator on CALIB ONLY (the saved deployment artifact), and pick
    #    the Youden operating threshold on CALIB. Both are then APPLIED to the unseen TEST slice.
    # SINGLE DEPLOYMENT CALIBRATOR (resolve the which-calibrator ambiguity): this Platt is THE
    # deployment calibrator. It is fit on the held-out CALIB slice, which is at the REAL ~1:100 prevalence,
    # so platt(raw) is prevalence-correct by construction (that's why the held-out PPV/Brier/ECE are
    # realistic). The training isotonic `calibrated_model` is fit on the BALANCED 1:1 training data, so it
    # is NOT prevalence-correct — it exists ONLY as predict_unseen's fallback when this Platt artifact is
    # absent (and predict_unseen scale-gates the operating threshold to the calibrator actually used).
    # Platt and isotonic are alternative single calibrations of the raw base — never stacked.
    platt = LogisticRegression(max_iter=1000).fit(raw_cal.reshape(-1, 1), y_cal)
    artifacts.atomic_write(os.path.join(OUT_DIR, f"platt_calib_{WINDOW}.joblib"),
                           lambda t: joblib.dump(platt, t))
    p_cal = platt.predict_proba(raw_cal.reshape(-1, 1))[:, 1]
    p_te = platt.predict_proba(raw_te.reshape(-1, 1))[:, 1]
    fpr, tpr, thr = roc_curve(y_cal, p_cal)            # threshold chosen on CALIB
    cut = thr[int(np.argmax(tpr - fpr))]

    # 5) report everything on the DISJOINT TEST slice (calibrated probabilities, calib-chosen threshold)
    au = roc_auc_score(y_te, p_te); ap = average_precision_score(y_te, p_te)
    pred = (p_te >= cut).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum()); fp = int(((pred == 1) & (y_te == 0)).sum())
    tn = int(((pred == 0) & (y_te == 0)).sum()); fn = int(((pred == 0) & (y_te == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0

    # DEPLOYMENT ARTIFACT: persist the operating threshold (on the Platt-calibrated probability scale)
    # next to platt_calib so predict_unseen labels patients at the SAME operating point we report here —
    # not the raw model's 0.5 cut. The Sens/Spec are the honest TEST-slice values this cut achieves.
    import json as _json
    _opthr = {"threshold": float(cut), "scale": "platt_calibrated",
              "chosen_on": "calib_youden", "reported_on": "test_70pct",
              "sens": round(sens, 4), "spec": round(spec, 4), "ppv": round(ppv, 4), "npv": round(npv, 4),
              "test_prevalence": round(float(y_te.mean()), 6), "window": WINDOW, "gap_months": GAP}
    artifacts.atomic_write_text(os.path.join(OUT_DIR, f"operating_threshold_{WINDOW}.json"),
                                _json.dumps(_opthr, indent=2))

    # full clinical suite with 95% bootstrap CIs on the TEST slice (same calib-chosen threshold)
    lm = _load(os.path.join(HERE, "..", "3_Modeling", "lung_metrics.py"), "lung_metrics")
    ci = lm.eval_with_ci(y_te, p_te, threshold=float(cut), label=f"held-out {WINDOW} {YEARS}yr")

    # WITHIN-AGE-BAND discrimination on the TEST slice — the honest "beyond age" number (global AUROC is
    # inflated by easy young-vs-old separation). Headline metric for the AGE_OFFSET de-confounding work.
    _wa_block = ""
    if 'mat' in dir() and "age_at_prediction" in getattr(mat, "columns", []):
        try:
            _age_te = pd.to_numeric(mat["age_at_prediction"], errors="coerce").to_numpy()[te_idx]
            _warows, _wau, _wap, _waglob = lm.within_age_metrics(y_te, p_te, _age_te)
            _wa_block = "\n" + lm.format_within_age(_warows, _wau, _wap, _waglob, label=f"held-out {WINDOW}")
        except Exception as _wae:
            raise RuntimeError(f"[heldout] within-age-band metrics FAILED (no soft fallback): {_wae}") from _wae

    lines = [
        f"# Held-out — lung {WINDOW} (gap {GAP}mo) | reported on the 70% TEST slice n={len(y_te):,} "
        f"(pos {int(y_te.sum())}, prevalence {100*y_te.mean():.2f}%) | Platt + threshold fit on the disjoint 30% calib",
        f"AUROC {au:.4f}   AUPRC {ap:.4f}",
        f"@Youden (threshold chosen on calib, applied to test): Sens {sens*100:.1f}  Spec {spec*100:.1f}  "
        f"PPV {ppv*100:.1f}  NPV {npv*100:.1f}  (cut={cut:.4f})",
        f"-- 95% bootstrap CIs (TEST slice, calib-chosen threshold) --",
        lm.format_result(ci),
        f"-- calibration on the TEST slice (out-of-sample: Platt fit on calib) --",
        f"{'cal':9s} {'Brier':>8s} {'ECE':>8s}",
        f"{'raw':9s} {brier_score_loss(y_te, raw_te):>8.4f} {_ece(raw_te, y_te):>8.4f}   (uncalibrated)",
        f"{'Platt':9s} {brier_score_loss(y_te, p_te):>8.4f} {_ece(p_te, y_te):>8.4f}   (calib-fit, out-of-sample)",
    ]
    report = "\n".join(lines) + _wa_block
    print("\n" + report)
    artifacts.atomic_write_text(os.path.join(OUT_DIR, f"heldout_recalib_{WINDOW}.txt"), report + "\n")
    print(f"\n-> heldout_recalib_{WINDOW}.txt + platt_calib_{WINDOW}.joblib + operating_threshold_{WINDOW}.json "
          f"(deployment artifacts)  (in {OUT_DIR})")

    # (Subgroup / fairness audit MOVED below the threshold artifacts so its fail-loud cannot block the
    # deployable model / Platt / threshold writes.)

    # Per-age-band operating thresholds (deployment artifact operating_threshold_by_age_{WINDOW}.json).
    # The single global Youden cut gives very uneven sensitivity by age — the model is age-dominated, so at
    # the low global cut nearly everyone old clears it (old bands get ~0% specificity) while under-50s rarely
    # do (~0% sensitivity). A per-band cut equalizes it: for each age band with enough calib positives, pick
    # the CALIB threshold achieving a target sensitivity; under-powered bands fall back to the global cut.
    # predict_unseen applies the band-appropriate cut. The global operating_threshold_{WINDOW}.json stays the
    # default; this is an opt-in refinement keyed on the patient's age band.
    if 'mat' in dir():
        try:
            _abcols = [c for c in mat.columns if c.startswith("ageband_")]
            if _abcols:
                _A = mat[_abcols].fillna(0.0).to_numpy()
                _lab = np.array([c[len("ageband_"):] for c in _abcols])
                _band = np.where(_A.sum(1) > 0, _lab[_A.argmax(1)], "Unknown")
                _bc, _bt = _band[cal_idx], _band[te_idx]
                _target = float(getattr(C, "AGEBAND_TARGET_SENS", 0.90))
                _min_pos = 20
                _by_band = {}
                for b in sorted(set(_bc)):
                    mb = _bc == b
                    if int(y_cal[mb].sum()) >= _min_pos:
                        _by_band[b] = float(lm.threshold_at_sensitivity(y_cal[mb], p_cal[mb], _target))
                _agethr = {"by_band": _by_band, "global": float(cut), "target_sens": _target,
                           "min_band_pos": _min_pos, "scale": "platt_calibrated", "window": WINDOW}
                artifacts.atomic_write_text(os.path.join(OUT_DIR, f"operating_threshold_by_age_{WINDOW}.json"),
                                            _json.dumps(_agethr, indent=2))
                print(f"\n[heldout] per-age-band operating points (target Sens {_target:.0%}; bands with "
                      f"<{_min_pos} calib pos fall back to the global cut {cut:.4f}) — TEST slice:")
                print(f"    {'band':<8}{'thr':>9}{'Sens':>8}{'Spec':>8}   vs global Sens/Spec")
                for b in sorted(set(_bt)):
                    mb = _bt == b
                    if mb.sum() == 0:
                        continue
                    yb = y_te[mb]
                    def _ss(pred):
                        tp = int(((pred == 1) & (yb == 1)).sum()); fn = int(((pred == 0) & (yb == 1)).sum())
                        tn = int(((pred == 0) & (yb == 0)).sum()); fp = int(((pred == 1) & (yb == 0)).sum())
                        return (tp / (tp + fn) if tp + fn else float("nan"),
                                tn / (tn + fp) if tn + fp else float("nan"))
                    thr_b = _by_band.get(b, cut)
                    se, sp = _ss((p_te[mb] >= thr_b).astype(int))
                    sg, spg = _ss((p_te[mb] >= cut).astype(int))
                    tag = "" if b in _by_band else "  (global fallback)"
                    print(f"    {b:<8}{thr_b:>9.4f}{se*100:>7.1f}%{sp*100:>7.1f}%   (global {sg*100:.0f}/{spg*100:.0f}){tag}")
                # AGGREGATE one-number comparison: overall Sens/Spec/FP under the single GLOBAL cut vs under
                # the per-age-band cuts (each patient judged at their own band's threshold). The ΔFP is the
                # headline benefit of per-age-band operating points. Appended to heldout_recalib_{WINDOW}.txt.
                _thr_vec = np.array([_by_band.get(b, cut) for b in _bt])
                def _ovr(pred):
                    tp = int(((pred == 1) & (y_te == 1)).sum()); fn = int(((pred == 0) & (y_te == 1)).sum())
                    tn = int(((pred == 0) & (y_te == 0)).sum()); fp = int(((pred == 1) & (y_te == 0)).sum())
                    return (tp / (tp + fn) if tp + fn else float("nan"),
                            tn / (tn + fp) if tn + fp else float("nan"), fp)
                _seg, _spg2, _fpg = _ovr((p_te >= cut).astype(int))
                _seb, _spb, _fpb = _ovr((p_te >= _thr_vec).astype(int))
                _agg = ("\nOVERALL operating-point comparison (TEST slice, n=%d):\n" % len(y_te)
                        + "    @ global threshold        : Sens %.1f  Spec %.1f  FP %d\n" % (_seg*100, _spg2*100, _fpg)
                        + "    @ per-age-band thresholds : Sens %.1f  Spec %.1f  FP %d   (dFP %+d, dSpec %+.1fpp)" %
                          (_seb*100, _spb*100, _fpb, _fpb-_fpg, (_spb-_spg2)*100))
                print("[heldout]" + _agg)
                artifacts.atomic_write_text(os.path.join(OUT_DIR, f"heldout_recalib_{WINDOW}.txt"), report + "\n" + _agg + "\n")
        except Exception as _abe:
            raise RuntimeError(f"[heldout] per-age-band threshold computation FAILED (no soft fallback — "
                               f"fix it rather than silently shipping the global-only cut): {_abe}") from _abe
    else:
        # M8: chunked path (HELDOUT_CHUNK>0) has no `mat` in scope, so the per-age-band threshold artifact is
        # NOT written -> predict_unseen's default THRESHOLD_REFINE='age' would silently fall back to the global
        # cut. Warn loudly so a chunked deploy run doesn't silently ship without the by-age operating points.
        print(f"[heldout][WARN] per-age-band thresholds NOT written (chunked path, no strata in scope) -> "
              f"operating_threshold_by_age_{WINDOW}.json absent; serve will fall back to the GLOBAL cut. "
              f"Run non-chunked (HELDOUT_CHUNK=0) to produce per-age operating points.")

    # Per-RECORD-DENSITY operating thresholds (operating_threshold_by_density_{WINDOW}.json). Mirrors the
    # per-age artifact for the OTHER confound: the model over-scores rich-record patients and under-scores
    # sparse ones (Sens 45% sparse -> 95% rich). Density proxy = sum of per-category *_count columns (always
    # present; same proxy subgroup_audit uses). Quartile EDGES are fit on CALIB and PERSISTED so serve assigns
    # identical quartiles; each quartile with enough calib positives gets a target-sens CALIB cut. Applied at
    # inference only when THRESHOLD_REFINE='density' (default 'age' preserves current behaviour).
    if 'mat' in dir():
        try:
            _ccols = [c for c in mat.columns if c.endswith("_count")]
            if _ccols:
                _dens = np.nan_to_num(mat[_ccols].to_numpy(dtype=float)).sum(1)
                _dc, _dt = _dens[cal_idx], _dens[te_idx]
                _edges = [float(x) for x in np.quantile(_dc, [0.25, 0.5, 0.75])]
                _q_cal, _q_te = np.digitize(_dc, _edges), np.digitize(_dt, _edges)   # 0..3
                _target = float(getattr(C, "AGEBAND_TARGET_SENS", 0.90)); _min_pos = 20
                _by_q = {}
                for q in range(4):
                    mq = _q_cal == q
                    if int(y_cal[mq].sum()) >= _min_pos:
                        _by_q[str(q)] = float(lm.threshold_at_sensitivity(y_cal[mq], p_cal[mq], _target))
                _densthr = {"by_quartile": _by_q, "edges": _edges, "global": float(cut), "target_sens": _target,
                            "min_q_pos": _min_pos, "density": "sum_count_cols", "scale": "platt_calibrated", "window": WINDOW}
                artifacts.atomic_write_text(os.path.join(OUT_DIR, f"operating_threshold_by_density_{WINDOW}.json"),
                                            _json.dumps(_densthr, indent=2))
                _qn = {0: "Q1_sparse", 1: "Q2", 2: "Q3", 3: "Q4_rich"}
                print(f"\n[heldout] per-density-quartile operating points (target Sens {_target:.0%}) — TEST slice:")
                print(f"    {'quartile':<10}{'thr':>9}{'Sens':>8}{'Spec':>8}   vs global")
                for q in range(4):
                    mq = _q_te == q
                    if mq.sum() == 0:
                        continue
                    yb = y_te[mq]
                    def _ss(pred, _yb=yb):
                        tp = int(((pred == 1) & (_yb == 1)).sum()); fn = int(((pred == 0) & (_yb == 1)).sum())
                        tn = int(((pred == 0) & (_yb == 0)).sum()); fp = int(((pred == 1) & (_yb == 0)).sum())
                        return (tp / (tp + fn) if tp + fn else float("nan"), tn / (tn + fp) if tn + fp else float("nan"))
                    thr_q = _by_q.get(str(q), cut)
                    se, sp = _ss((p_te[mq] >= thr_q).astype(int)); sg, spg = _ss((p_te[mq] >= cut).astype(int))
                    tag = "" if str(q) in _by_q else "  (global fallback)"
                    print(f"    {_qn[q]:<10}{thr_q:>9.4f}{se*100:>7.1f}%{sp*100:>7.1f}%   (global {sg*100:.0f}/{spg*100:.0f}){tag}")
        except Exception as _de:
            raise RuntimeError(f"[heldout] per-density-quartile threshold computation FAILED (no soft fallback — "
                               f"fix it rather than silently shipping the global-only cut): {_de}") from _de

    # Subgroup / fairness audit on the TEST slice: Sens/Spec/PPV + calibration within sex x ethnicity x
    # age-band x record-density quartile, with a disparity flag -> subgroup_audit_{WINDOW}.csv. Runs HERE,
    # after every deployable artifact (model / Platt / global + per-age + per-density thresholds) is written,
    # so it can FAIL LOUD (no soft fallback) without blocking deployment. Needs the full FE matrix (non-chunked).
    if 'mat' in dir():
        sys.path.insert(0, HERE)
        import subgroup_audit as _sg
        _sg.run_audit(mat.iloc[te_idx], y_te, p_te, float(cut), OUT_DIR, WINDOW, YEARS)

    # Explainability on the held-out TEST slice (items #4 + #5): (1) per-patient signed-SHAP top risk
    # factors + a global SHAP summary, and (2) the key risk-factor drivers of each confusion-matrix
    # segment (TP/FP/FN/TN), using the reported operating prediction `pred`. SHAP is TREE-EXACT ONLY
    # (explainability._shap_values): tree models + soft-voting tree ensembles -> exact TreeExplainer on
    # ALL patients (REQUIRES xgboost<3 — shap-0.49 can't parse xgboost 3.x's vector base_score; pinned in
    # requirements.txt). There is NO KernelExplainer — a non-tree model would raise
    # (AdaBoost is demoted off, so this never happens). EXPLAIN_MAX is now a no-op (tree-exact always
    # explains ALL patients); EXPLAIN=0 skips. Outputs -> {OUT_DIR}/explainability/ (patient_explanations.csv,
    # segment_drivers.csv, *.png), then mirrored — same as every other artifact — to GCS at
    # {GCS_ARTIFACTS}/3_Modeling_outputs/{config}/explainability/ when GCS is on (no-op for local runs).
    # GCS_ARTIFACTS = GCS_ROOT/GCS_VARIANT (variant-scoped), NOT the common GCS_ROOT.
    if os.environ.get("EXPLAIN", "1").strip().lower() not in ("0", "false", "no"):
        try:
            sys.path.insert(0, os.path.join(HERE, "..", "3_Modeling"))
            import explainability as _xai
            feats = md.get("feature_names")        # == columns of the transformed X (reindexed to feature_names)
            if getattr(C, "CATEGORIZED", False):
                # CATEGORIZED mode: features are already "<category>_<family>" (the FE grouped by category),
                # so SHAP explains drivers PER CATEGORY directly. No SNOMED->term map is needed (category
                # labels are already human-readable); humanize() leaves non-coded names unchanged.
                code_terms = None
                print(f"[heldout] CATEGORIZED: SHAP factors/segment-drivers are category-level "
                      f"(e.g. 'respiratory_clinic_count'); no per-code term map applied.")
            else:
                try:                               # code->term map from THIS config's own codelist (no hardcode)
                    _cl = pd.read_csv(os.path.join(FE_DIR, "codelist", C.codelist_name(WINDOW, YEARS)))
                    code_terms = dict(zip(_cl["Code"], _cl["Name"]))
                except Exception as _ce:
                    print(f"[heldout] codelist term-map unavailable ({_ce}); SHAP labels stay code-only")
                    code_terms = None
            xai_dir = os.path.join(OUT_DIR, "explainability")
            # patient_guid + segment in patient_explanations.csv -> identify each FP/FN and join age/density
            _guids = (mat["patient_guid"].to_numpy()[te_idx]
                      if 'mat' in dir() and "patient_guid" in getattr(mat, "columns", []) else None)
            _xai.explain(md["model"], X[te_idx], feats, y_te, pred, p_te, xai_dir,
                         title=f"{WINDOW} {YEARS}yr held-out",
                         max_explained=(int(os.environ["EXPLAIN_MAX"]) if os.environ.get("EXPLAIN_MAX") else None),
                         code_terms=code_terms, patient_guids=_guids)
            # mirror the explainability folder to GCS (variant-scoped; no-op when GCS off / local-only run)
            gcs_artifacts = getattr(C, "GCS_ARTIFACTS", "")
            if gcs_artifacts:
                gcs_xai = f"{gcs_artifacts.rstrip('/')}/3_Modeling_outputs/{C.artifact_subdir(WINDOW, YEARS)}/explainability"
                try:
                    sent = artifacts.upload_dir(xai_dir, gcs_xai)
                    print(f"[heldout] uploaded {len(sent)} explainability files -> {gcs_xai}/")
                except Exception as _ue:
                    print(f"[heldout] explainability GCS upload skipped: {_ue}")
        except Exception as _e:
            # LOUD (not a one-line swallow): the metrics/recalib above are already written, so a SHAP/
            # xgboost hiccup must not lose them — but it must be diagnosable, since SHAP is a requested
            # deliverable. Print the full traceback so a broken explainer can never masquerade as success.
            import traceback
            print(f"[heldout] explainability FAILED — metrics + recalib above are still valid and saved.\n"
                  f"{traceback.format_exc()}")
            raise   # no soft fallback: SHAP is a requested deliverable -> surface the failure (everything
                    # deployable is already written above, so re-raising loses nothing but never fakes success)


if __name__ == "__main__":
    main()
