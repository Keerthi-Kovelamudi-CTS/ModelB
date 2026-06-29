"""
End-to-end driver — Lung Cancer model (codelist-driven, split-first)
=====================================================================
One command runs the whole pipeline for BOTH horizons x EVERY lookback window by default, with
per-stage caching and optional Google Cloud Storage mirroring so reruns and teammates reuse artifacts.

The split is built ONCE per horizon (cohort membership is lookback-independent). Everything else runs
per (horizon, lookback) — CRUCIALLY the codelist is RE-discovered for each lookback on THAT lookback's
own train data, so no codelist is ever reused across lookbacks.

Stages per (horizon, lookback) — each is cache-skipped if its output already exists locally or in GCS,
unless --force (cache-skip = reuse the existing artifact, pulling from GCS if needed; nothing is lost):
  0. split    make_split.py            ONE canonical train/valid/test split per horizon -> SPLIT_DIR
  1. codelist 1_Top_Snomed/*           counts -> Combined Scoring -> codelist, on THIS lookback's TRAIN
                                        data -> {Astra|Nova}_{yr}yr_OFF_codelist.csv
  2. fe       2_FE/build_features      this lookback's codelist + cohort (BigQuery) -> feature matrix
  3. stable   2_FE/stability_select    k-fold cumimp99 -> driving codes; stamps the split onto the matrix
  4. train    3_Modeling/lung_training  train on the stable matrix (split honored from the stamped column,
                                        validation-based model/threshold selection, isotonic calibration)
Optional:
  --heldout   run 4_Heldout/evaluate_heldout after training (touch-once labelled cohort)

Output naming: Astra = 12mo horizon, Nova = 1mo horizon; the OFF suffix is the fixed data-driven artifact tag.
CATEGORIZED variant: FE groups events by hand-assigned CATEGORY (2_FE/categorized_codelist/{Astra|Nova}_{yr}yr_categories.csv),
not per individual SNOMED; Phase-1 per-code scoring is skipped (see stage_codelist).
Per-(horizon, lookback) artifacts (under V3_categorized/) are mirrored to GCS at GCS_ARTIFACTS/<relative path>,
where GCS_ARTIFACTS = GCS_ROOT/GCS_VARIANT (this variant = V3_categorized; the snomed-level pipeline = V3_snomed):
  2_FE/output/{Astra|Nova}/{yr}yr_OFF/features_p005_{h}.parquet, features_p005_{h}_stable.parquet, stable_features_{h}.csv
  3_Modeling_outputs/{Astra|Nova}/{yr}yr_OFF/model_{h}.joblib, results_{h}.png, heldout_recalib_{h}.txt, platt_calib_{h}.joblib
  3_Modeling_outputs/{Astra|Nova}/{yr}yr_OFF/explainability/  (SHAP — uploaded by evaluate_heldout itself)
COMMON across both variants (NOT under GCS_VARIANT): the canonical split lives in SPLIT_DIR (GCS_ROOT/splits,
a gs:// path written directly by the split stage) and the lifetime event cache in GCS_ROOT/raw_events.

Config: ../config.py (single source — GCS_ROOT, GCS_VARIANT, GCS_ARTIFACTS, SPLIT_DIR, CATEGORIZED, HORIZONS, FE_WINDOWS, FE_ENGINE, …).

Run (VM with BigQuery + the env; see requirements.txt):
    python run_v3.py                      # 12mo & 1mo x AUTO-DISCOVERED windows (from category-map files), GCS on
    python run_v3.py 12mo --windows 10    # one horizon, single chosen lookback (must have its category map)
    python run_v3.py --heldout            # full sweep + held-out eval
    python run_v3.py --force              # recompute every stage (ignore cache)
    python run_v3.py --no-gcs             # local only (no GCS read/write)
    python run_v3.py --heldout 2>&1 | tee run.log   # capture the FULL run log (all stages + stderr) via the shell
    # FE engine is pandas build_features.py ONLY (the Polars backend was removed).
    # Full run-log capture is the shell/scheduler's job (tee / cron / Airflow / k8s) — captures subprocess
    # stages + tracebacks too. A structured run record is written to 3_Modeling_outputs/run_metadata_*.json.
"""
import os
import sys
import argparse
import importlib.util
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # config + splits + artifacts
import config as C
import splits
import artifacts                          # atomic writes + provenance/freshness manifests

FE_DIR = os.path.join(HERE, "2_FE")
MODEL_DIR = os.path.join(HERE, "3_Modeling")
SNOMED_DIR = os.path.join(HERE, "1_Top_Snomed")
HELDOUT_DIR = os.path.join(HERE, "4_Heldout")

# LOGGING: the pipeline prints operational + report output to stdout/stderr. Full run-log capture is left
# to the shell / scheduler (e.g. `python run_v3.py 2>&1 | tee run.log`, or cron/Airflow/k8s), which capture
# EVERYTHING — including the subprocess stages (split/codelist/heldout) and tracebacks — live and correctly.
# A structured machine-readable run record (git sha + UTC + argv + full config) is written to
# 3_Modeling_outputs/run_metadata_*.json. (We deliberately do NOT wrap sys.stdout in-app: it would miss
# subprocess-stage output and stderr.)


# ----------------------------------------------------------------------------- module loading
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(script, args, cwd):
    """Run a pipeline script in a fresh process (isolates its module-level arg parsing)."""
    cmd = [sys.executable, script] + list(args)
    print(f"    $ {' '.join(cmd)}   (cwd={os.path.relpath(cwd, HERE)})")
    subprocess.run(cmd, cwd=cwd, check=True)


# ----------------------------------------------------------------------------- GCS cache layer
_GCS = {"fs": None, "tried": False, "enabled": True}


def _fs():
    """A gcsfs filesystem, or None if GCS is disabled / unavailable / GCS_ARTIFACTS unset."""
    if not _GCS["enabled"] or not getattr(C, "GCS_ARTIFACTS", ""):
        return None
    if not _GCS["tried"]:
        _GCS["tried"] = True
        try:
            import gcsfs
            _GCS["fs"] = gcsfs.GCSFileSystem()
        except Exception as e:
            print(f"[gcs] disabled (gcsfs unavailable: {e})")
            _GCS["fs"] = None
    return _GCS["fs"]


def _key(local):
    # Variant-scoped: 2_FE + 3_Modeling_outputs mirror under GCS_ARTIFACTS (= GCS_ROOT/GCS_VARIANT).
    # splits + raw_events are NOT routed here — they use GCS_ROOT directly (common to both variants).
    rel = os.path.relpath(local, HERE).replace(os.sep, "/")
    return C.GCS_ARTIFACTS.rstrip("/") + "/" + rel


def ensure_local(local):
    """Make `local` (and, best-effort, its .manifest.json sidecar) present by pulling from GCS if only
    there. Return True if the artifact is present after."""
    present = os.path.exists(local)
    fs = _fs()
    if not present and fs is not None:
        key = _key(local)
        try:
            if fs.exists(key):
                os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
                fs.get(key, local)
                print(f"[gcs] pulled {key}")
                present = True
        except Exception as e:
            print(f"[gcs][warn] pull failed for {key}: {e}")
    if present and fs is not None and not os.path.exists(artifacts.manifest_path(local)):
        try:                                            # pull the sidecar so freshness is judged locally
            mkey = _key(artifacts.manifest_path(local))
            if fs.exists(mkey):
                fs.get(mkey, artifacts.manifest_path(local))
        except Exception:
            pass
    return present


def publish(*locals_):
    """Mirror produced artifacts AND their manifest sidecars to GCS (no-op if GCS off / file missing).
    Honors the SAME global GCS_WRITE switch as artifacts.atomic_write (artifacts._gcs_blocked): with
    GCS_WRITE=0 this is a no-op, so a non-mutating run can NEVER overwrite production via the publish path
    (the --no-gcs flag also forces GCS_WRITE=0). Without this, GCS_WRITE=0 gated only atomic_write/upload_dir
    while publish() still pushed — the dual-gate bug that clobbered retrain-2 held-out artifacts (2026-06-19)."""
    fs = _fs()
    if fs is None:
        return
    if artifacts._gcs_blocked(C.GCS_ARTIFACTS):   # one gate for ALL GCS writes (atomic_write + publish)
        print("[gcs] publish skipped — GCS_WRITE=0 (non-mutating run; prod untouched)")
        return
    for local in locals_:
        for f in (local, artifacts.manifest_path(local)):
            if not os.path.exists(f):
                continue
            key = _key(f)
            try:
                fs.put(f, key)
                print(f"[gcs] pushed -> {key}")
            except Exception as e:
                print(f"[gcs][warn] push failed for {key}: {e}")


def need(local, force, config_slice=None, upstream=None):
    """True if the stage must (re)compute: forced, missing, or its manifest is STALE — i.e. the git
    commit, the relevant config, or an upstream input that produced it no longer matches. Legacy
    artifacts without a manifest are trusted on existence (backward compatible)."""
    if force:
        return True
    if not ensure_local(local):
        return True
    why = artifacts.stale_reason(local, config_slice, upstream)
    if why:
        print(f"[cache] {os.path.basename(local)} STALE ({why}) -> recompute")
        return True
    return False


def _cfg(*keys, **extra):
    """A JSON-able config slice for a stage's manifest: the named config.py values + extras (horizon,
    years, …). Code edits are caught separately by git_sha; this captures the env-overridable knobs."""
    d = {k: getattr(C, k, None) for k in keys}
    d.update(extra)
    return d


def _done(path, config_slice=None, upstream=None):
    """Stamp the artifact's provenance/freshness manifest (git_sha + config slice + upstream hashes),
    then mirror the artifact AND its manifest to GCS."""
    if os.path.exists(path):
        artifacts.write_manifest(path, config_slice, upstream)
    publish(path)


# ----------------------------------------------------------------------------- stages
def stage_codelist(h, yr, force):
    """Phase 1 for THIS (horizon, lookback): per-code counts -> combined scoring -> codelist, scored on
    this lookback's TRAIN events. Each lookback gets its OWN codelist ({Astra|Nova}_{yr}yr_OFF_codelist.csv) —
    no codelist is ever reused across lookbacks. Requires the canonical split (built first)."""
    if getattr(C, "CATEGORIZED", False):
        cat = os.path.join(FE_DIR, "categorized_codelist", C.categorized_name(h, yr))
        assert os.path.exists(cat), (f"[codelist] CATEGORIZED mode: hand-curated mapping missing: {cat}\n"
                                     f"  Create it (cols Code,Name,Category) before running FE.")
        print(f"[codelist] {h}/{yr}yr: CATEGORIZED mode — using hand-curated {os.path.basename(cat)} "
              f"(Phase-1 per-code scoring skipped)")
        return
    codelist = os.path.join(FE_DIR, "codelist", C.codelist_name(h, yr))
    # Stamp the scoring knobs too (Combined Scoring sources them from config) so a change to any of them
    # invalidates the codelist artifact rather than silently reusing a stale one.
    slc = _cfg("TOP_N", "OR_MIN", "PREV_MIN", "MIN_VALUE_FRAC", "SCORING_YEARS_BEFORE", "NEG_POS_RATIO",
               "SCORING_TOP_N", "SCORING_MAX_ML_FEATURES", "SCORING_CV_FOLDS", "MIN_PATIENTS_PER_FEATURE",
               horizon=h, years=yr)
    up = [splits.split_path(h)]
    if not need(codelist, force, slc, up):
        print(f"[codelist] {h}/{yr}yr: present & fresh — skip")
        return
    print(f"[codelist] {h}/{yr}yr: scoring codes on {yr}yr TRAIN data ...")
    _run("build_score_counts.py", [h, str(yr)], SNOMED_DIR)
    _run("Combined Scoring (ML+Stat).py", [h, str(yr)], SNOMED_DIR)
    _run("build_codelist.py", [h, str(yr)], SNOMED_DIR)
    _done(codelist, slc, up)


def stage_split(h, force):
    """Create the canonical split for horizon `h` if it isn't already saved (in SPLIT_DIR / GCS)."""
    if not force and splits.exists(splits.split_path(h)):
        print(f"[split] {h}: canonical split present at {splits.split_path(h)} — skip")
        return
    _run("make_split.py", [h], HERE)        # writes directly to SPLIT_DIR (gs:// handled by splits)


def _fe_dir(h, yr):
    return os.path.join(FE_DIR, "output", C.artifact_subdir(h, yr))


def _model_dir(h, yr):
    return os.path.join(HERE, "3_Modeling_outputs", C.artifact_subdir(h, yr))


def stage_fe(h, yr, force, engine):
    """Build the feature matrix for horizon `h` at lookback `yr` years."""
    out = os.path.join(_fe_dir(h, yr), f"features_p005_{h}.parquet")
    slc = _cfg("FE_ENGINE", "CATEGORIZED", "MIN_VALUE_FRAC", "AGE_PROXY_FE", "BLOOD_RATIOS_FE", "CLINICAL_FE",
               "DENSITY_PROXY_FE", "ADMIN_NOISE_FE", "CHRONICITY_FE", "REDFLAG_FE", "WEIGHT_TREND_FE", "WEIGHT_TREND_CODE", "CATEGORY_PRUNE_FE", "CROSS_TOP_K",
               "MED_FE", "MED_DUR_CAP_DAYS",
               horizon=h, years=yr, engine=engine)   # ALL FE-shaping flags in the cache key -> env-toggling busts a stale FE matrix
    # CATEGORIZED mode reads the hand-curated category map (not the per-code codelist) as its upstream.
    codelist_up = (os.path.join(FE_DIR, "categorized_codelist", C.categorized_name(h, yr))
                   if getattr(C, "CATEGORIZED", False)
                   else os.path.join(FE_DIR, "codelist", C.codelist_name(h, yr)))
    up = [codelist_up, splits.split_path(h)]
    # CATEGORIZED: the per-(horizon) ZERO-FIT artifacts must exist for held-out/serve to reload them. If a
    # cached FE matrix is present but ANY of them is missing, force a rebuild so they are (re)written —
    # otherwise build(fit_split=False) hard-SystemExits later ("... not found"). Covers ALL zero-fit
    # artifacts — z-stats, valuecodes, moczero, and topcats (when CROSS_TOP_K>=2).
    force_fe = force
    if getattr(C, "CATEGORIZED", False):
        _zf = [f"zstats_categorized_{h}.parquet", f"valuecodes_categorized_{h}.parquet",
               f"moczero_categorized_{h}.parquet"]
        if int(getattr(C, "CROSS_TOP_K", 0) or 0) >= 2:
            _zf.append(f"topcats_categorized_{h}.parquet")
        # GCS-aware — ensure_local pulls a sidecar from GCS if only there (so a fresh VM with the
        # sidecars already on GCS does NOT needlessly rebuild FE); force a rebuild only if missing BOTH.
        _missing = [a for a in _zf if not ensure_local(os.path.join(_fe_dir(h, yr), a))]
        if _missing:
            print(f"[fe] {h}/{yr}yr: zero-fit artifact(s) missing (local+GCS) {_missing} -> force FE rebuild (re-persist)")
            force_fe = True
    if not need(out, force_fe, slc, up):
        print(f"[fe] {h}/{yr}yr: {os.path.basename(out)} present & fresh — skip")
        return
    # FE engine: pandas build_features.py is the ONLY engine (Polars removed — it never supported CATEGORIZED
    # mode and was always force-overridden to pandas here). No silent fallback: errors surface.
    fe_path = os.path.join(FE_DIR, "build_features.py")
    print(f"[fe] {h}/{yr}yr: building features (pandas) ...")
    mat, _ = _load(fe_path, "fe_engine").build(h, years=yr)
    artifacts.atomic_write(out, lambda t: mat.to_parquet(t, index=False))   # .tmp -> os.replace (no partial file)
    print(f"[fe] {h}/{yr}yr: {mat.shape[0]:,} patients x {mat.shape[1]:,} cols -> {out}")
    _done(out, slc, up)
    # explicitly publish the matrix AND the zero-fit sidecars to GCS — don't rely on build_features's
    # own best-effort mirror. held-out/serve on another VM RELOADS these sidecars from GCS (hard-fail if
    # absent), so an un-published sidecar = a later hard-fail; an un-published matrix = a needless rebuild.
    # publish() is a no-op when GCS is off or a file is missing.
    if getattr(C, "CATEGORIZED", False):
        _sc = [os.path.join(_fe_dir(h, yr), f"{n}_categorized_{h}.parquet")
               for n in ("zstats", "valuecodes", "moczero")]
        if int(getattr(C, "CROSS_TOP_K", 0) or 0) >= 2:
            _sc.append(os.path.join(_fe_dir(h, yr), f"topcats_categorized_{h}.parquet"))
        publish(out, *[p for p in _sc if os.path.exists(p)])
    else:
        publish(out)


def stage_stable(h, yr, force):
    """k-fold stability selection -> features_p005_{h}_stable.parquet (+ stable_features_{h}.csv)."""
    out_dir = _fe_dir(h, yr)
    stable = os.path.join(out_dir, f"features_p005_{h}_stable.parquet")
    flist = os.path.join(out_dir, f"stable_features_{h}.csv")
    fe_matrix = os.path.join(out_dir, f"features_p005_{h}.parquet")
    slc = _cfg("N_FOLDS", "MIN_FOLDS", "CUM_IMP", "REDFLAG_FE", horizon=h, years=yr)   # REDFLAG_FE force-keeps in stability_select
    up = [fe_matrix]
    if not need(stable, force, slc, up):
        print(f"[stable] {h}/{yr}yr: {os.path.basename(stable)} present & fresh — skip")
        return
    if not ensure_local(fe_matrix):
        print(f"[stable][skip] {h}/{yr}yr: feature matrix missing — run the fe stage first")
        return
    print(f"[stable] {h}/{yr}yr: k-fold stability selection ...")
    _load(os.path.join(FE_DIR, "stability_select.py"), "stability_select").select(
        h, in_path=fe_matrix, out_dir=out_dir)
    _done(stable, slc, up)
    publish(flist)


def stage_train(h, yr, force, Predictor):
    """Train on the stable matrix (split honored from the stamped column) -> model_{h}.joblib + png."""
    out_dir = _model_dir(h, yr)
    model = os.path.join(out_dir, f"model_{h}.joblib")
    png = os.path.join(out_dir, f"results_{h}.png")
    stable = os.path.join(_fe_dir(h, yr), f"features_p005_{h}_stable.parquet")
    slc = _cfg("RANDOM_STATE", "TEST_SIZE", "CALIB_SIZE", "TUNE", "DROP_THRESHOLD", "AGE_OFFSET_FE",
               horizon=h, years=yr)
    up = [stable]
    if not need(model, force, slc, up):
        print(f"[train] {h}/{yr}yr: model present & fresh — skip")
        return
    if not ensure_local(stable):
        print(f"[train][skip] {h}/{yr}yr: stable matrix missing — run the stable stage first")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n----- training {h}/{yr}yr: {stable} -----")
    p = Predictor(stable)
    (p.load_data()
       .explore_data()
       .preprocess_data(drop_threshold=C.DROP_THRESHOLD, horizon=h)   # explicit horizon -> split-stamp guard never skipped
       .split_data(test_size=C.TEST_SIZE, calib_size=C.CALIB_SIZE)   # honors the stamped canonical split
       .handle_imbalance(method="none")                              # cost-weighted (no SMOTE)
       .select_features(method="all")                                # stability_select already chose codes (on train)
       .train_and_evaluate(use_resampled=False)
       .hyperparameter_tuning()                                      # Optuna only if TUNE=1 (else untuned top-5)
       .create_ensemble()
       .calibrate(method="isotonic", by_age_band=True)
       .final_evaluation()                                           # Youden threshold on validation, scored on test
       .plot_results(save_path=png)
       .save_model(model))
    print(f"----- {h}/{yr}yr done -> {out_dir} -----")
    _done(model, slc, up)
    # Publish the results plot + the INTERNAL-test artifacts (metrics CSV + explainability_internal/) to GCS,
    # mirroring how evaluate_heldout publishes the held-out explainability/ — so internal results are not local-only.
    publish(png, os.path.join(out_dir, f"internal_{h}.csv"))
    _xint = os.path.join(out_dir, "explainability_internal")
    if os.path.isdir(_xint):
        publish(*[os.path.join(_xint, f) for f in sorted(os.listdir(_xint))
                  if os.path.isfile(os.path.join(_xint, f))])


def stage_parity_gate(h, yr):
    """Full-vs-keep-features FE PARITY GATE — HARD-FAIL before we trust held-out / inference. Proves the
    keep-features build (which held-out + serving use) is byte-identical to the full build for the model's
    features, so a cross-category/restricted family can't silently diverge at serve. Non-mutating
    (fe_keep_parity.py uses fit_split=False -> loads persisted train artifacts, never re-writes GCS).
    Gated by config.FE_PARITY_GATE (default on); exit 1 -> CalledProcessError -> aborts the run."""
    if not getattr(C, "FE_PARITY_GATE", True):
        print(f"[parity] {h}/{yr}yr: FE_PARITY_GATE off — skip")
        return
    if not ensure_local(os.path.join(_model_dir(h, yr), f"model_{h}.joblib")):
        print(f"[parity][skip] {h}/{yr}yr: trained model missing — run the train stage first")
        return
    print(f"\n----- FE parity gate {h}/{yr}yr (full vs keep-features; hard-fail) -----")
    _run("fe_keep_parity.py", [h, str(yr)], FE_DIR)   # exits 1 on ANY mismatch -> aborts before held-out


def stage_heldout(h, yr, force):
    """Touch-once labelled held-out evaluation + deployable Platt (disjoint 30/70 calib/test)."""
    out_dir = _model_dir(h, yr)
    model = os.path.join(out_dir, f"model_{h}.joblib")
    if not ensure_local(model):
        print(f"[heldout][skip] {h}/{yr}yr: trained model missing — run the train stage first")
        return
    report = os.path.join(out_dir, f"heldout_recalib_{h}.txt")
    platt = os.path.join(out_dir, f"platt_calib_{h}.joblib")
    slc = _cfg("TEST_SIZE", "CALIB_SIZE", horizon=h, years=yr)
    up = [model]
    if not need(report, force, slc, up):
        print(f"[heldout] {h}/{yr}yr: report present & fresh — skip")
        return
    _run("evaluate_heldout.py", [h, str(yr)] + (["--force"] if force else []), HELDOUT_DIR)  # --force rebuilds held-out FE
    _done(report, slc, up)
    # publish the report, the deployable Platt, the operating-threshold JSON (else a GCS-pulled deploy
    # would have the recalibrator but no threshold -> predict_unseen silently falls back to the 0.5 cut),
    # and the held-out matrix.
    publish(report, platt,
            os.path.join(out_dir, f"operating_threshold_{h}.json"),
            os.path.join(out_dir, f"operating_threshold_by_age_{h}.json"),   # per-age-band thresholds
            os.path.join(out_dir, f"operating_threshold_by_density_{h}.json"),  # per-density thresholds (publish no-ops if absent)
            os.path.join(out_dir, f"subgroup_audit_{h}.csv"),                # fairness audit
            os.path.join(out_dir, f"subgroup_disparities_{h}.csv"),          # (only if any flagged; publish no-ops if absent)
            os.path.join(_fe_dir(h, yr), f"heldout_features_{h}.parquet"))   # held-out FE matrix lives under 2_FE/output/
    # held-out explainability/ is uploaded by evaluate_heldout itself; nothing extra to publish here.


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description="Lung Cancer pipeline — one-command, cached, GCS-aware.")
    ap.add_argument("horizons", nargs="*", help="12mo and/or 1mo (default: config.HORIZONS)")
    ap.add_argument("--heldout", action="store_true", help="also run the held-out evaluation")
    ap.add_argument("--force", action="store_true", help="recompute every stage (ignore cache)")
    ap.add_argument("--no-gcs", action="store_true", help="local only — no GCS read/write")
    ap.add_argument("--engine", choices=["pandas"], default="pandas",
                    help="FE engine — pandas only (Polars backend removed)")
    ap.add_argument("--windows", nargs="*", type=int, default=None,
                    help="lookback windows in years to run (default: auto-discover from the category-map "
                         "files present in 2_FE/categorized_codelist/). No window is hardcoded.")
    a = ap.parse_args()

    # FAIL LOUD, FAIL EARLY: validate horizon tokens at the CLI boundary. Previously an unknown positional
    # (e.g. `12m`) was silently dropped -> coerced to "all horizons" -> crashed several stages later inside
    # build_features' config.horizons_from_argv. Reject it here with a clear message instead.
    _bad = [h for h in a.horizons if h not in C.HORIZONS]
    if _bad:
        raise SystemExit(f"[run_v3] unrecognized horizon argument(s) {_bad}; valid horizons are {list(C.HORIZONS)}.")
    horizons = list(a.horizons) or list(C.HORIZONS)

    _GCS["enabled"] = not a.no_gcs
    if a.no_gcs:
        # --no-gcs must also gate every artifact WRITE, including those in subprocesses (evaluate_heldout
        # mirrors its explainability folder via artifacts.upload_dir, which checks GCS_WRITE — not _GCS).
        # Setting the env switch propagates the "no writes" intent to children and unifies on one gate.
        os.environ["GCS_WRITE"] = "0"
    gcs = "off" if _fs() is None else C.GCS_ARTIFACTS    # variant artifact root (splits/raw_events stay at GCS_ROOT)
    print("=" * 70)
    print(f"PIPELINE (CATEGORIZED)  horizons={horizons}  engine={a.engine}  GCS={gcs}"
          f"{'  [FORCE]' if a.force else ''}{'  +heldout' if a.heldout else ''}")
    print("-" * 70)
    print("CONFIG (every tunable that shaped this run):")           # provenance: log the full config
    print(C.summary())
    print("=" * 70)

    # RUN PROVENANCE: write a run_metadata_{utc}.json (run_id + git_sha + UTC time + argv + full config +
    # horizons) under 3_Modeling_outputs/ so months later you can tell exactly what produced a model.
    try:
        import json, uuid, datetime
        meta = {"run_id": uuid.uuid4().hex[:12],
                "utc": datetime.datetime.utcnow().isoformat() + "Z",
                "git_sha": artifacts.git_sha(),
                "argv": sys.argv,
                "horizons": horizons, "windows": (list(a.windows) if a.windows else None),
                "engine": a.engine, "force": a.force, "heldout": a.heldout,
                "config": C.as_dict()}
        _meta_dir = os.path.join(HERE, "3_Modeling_outputs")
        os.makedirs(_meta_dir, exist_ok=True)
        _meta_path = os.path.join(_meta_dir, f"run_metadata_{meta['utc'].replace(':', '').replace('-', '')}.json")
        artifacts.atomic_write_text(_meta_path, json.dumps(meta, indent=2, default=str))
        print(f"[provenance] run {meta['run_id']} (git {meta['git_sha'][:8]}) -> {_meta_path}")
        print("=" * 70)
    except Exception as _pe:
        print(f"[provenance][warn] run metadata not written ({type(_pe).__name__}: {_pe})")

    Predictor = _load(os.path.join(MODEL_DIR, "lung_training.py"), "lung_training").LungCancerPredictor
    for h in horizons:                                   # both horizons (12mo=Astra then 1mo=Nova) by default
        # Windows for THIS horizon: an explicit --windows selection, else auto-discovered from the
        # category-map files actually present (config.categorized_windows). Each (horizon, window) flows to
        # its OWN {Astra|Nova}/{yr}yr_OFF/ folder via artifact_subdir — nothing is hardcoded to a year.
        yrs_for_h = list(a.windows) if a.windows else C.categorized_windows(h)
        print(f"\n{'#'*70}\n# {h}  (lookbacks: {yrs_for_h or 'NONE — no category-map files found'})\n{'#'*70}")
        if not yrs_for_h:
            print(f"[skip] {h}: no {C.horizon_label(h)}_*yr_categories.csv in {C.CATEGORIZED_DIR}")
            continue
        stage_split(h, a.force)                          # ONE split per horizon, shared across its lookbacks
        for yr in yrs_for_h:                             # each selected/discovered lookback, self-contained
            stage_codelist(h, yr, a.force)               # CATEGORIZED: validates the category map (Phase-1 skipped)
            stage_fe(h, yr, a.force, a.engine)           # FE groups by category from this window's map
            stage_stable(h, yr, a.force)
            stage_train(h, yr, a.force, Predictor)
            stage_parity_gate(h, yr)                     # hard-fail if keep-features build != full build
            if a.heldout:
                stage_heldout(h, yr, a.force)

    print("\n" + "=" * 70 + "\nPIPELINE COMPLETE\n" + "=" * 70)


if __name__ == "__main__":
    main()
