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
                                        data -> lung_codelist_{h}_{yr}yr.csv
  2. fe       2_FE/build_features      this lookback's codelist + cohort (BigQuery) -> tiered matrix
  3. stable   2_FE/stability_select    k-fold cumimp99 -> driving codes; stamps the split onto the matrix
  4. train    3_Modeling/lung_training  train on the stable matrix (split honored from the stamped column,
                                        validation-based model/threshold selection, isotonic calibration)
Optional:
  --heldout   run 4_Heldout/evaluate_heldout after training (touch-once labelled cohort)

Per-window artifacts (under V3/) are mirrored to GCS at GCS_ROOT/<relative path>:
  2_FE/codelist/lung_codelist_{h}_{yr}yr.csv
  2_FE/output/{h}/{yr}yr/features_p005_{h}.parquet, features_p005_{h}_stable.parquet, stable_features_{h}.csv
  output/{h}/{yr}yr/model_{h}.joblib, results_{h}.png, heldout_recalib_{h}.txt, platt_calib_{h}.joblib
The canonical split lives in SPLIT_DIR (already a gs:// path), written directly by the split stage.

Config: ../config.py + .env (GCS_ROOT, SPLIT_DIR, HORIZONS, FE_WINDOWS, FE_ENGINE, …).

Run (VM with BigQuery + the env; see requirements.txt):
    python run_v3.py                      # 12mo & 1mo x all FE_WINDOWS (5/10/20/100), GCS caching on
    python run_v3.py 12mo --windows 5     # one horizon, single 5yr lookback
    python run_v3.py --heldout            # full sweep + held-out eval
    python run_v3.py --force              # recompute every stage (ignore cache)
    python run_v3.py --no-gcs             # local only (no GCS read/write)
    python run_v3.py --engine polars      # use the Polars FE (once it passes fe_parity_check.py)
"""
import os
import sys
import argparse
import importlib.util
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # config + splits
import config as C
import splits

FE_DIR = os.path.join(HERE, "2_FE")
MODEL_DIR = os.path.join(HERE, "3_Modeling")
SNOMED_DIR = os.path.join(HERE, "1_Top_Snomed")
HELDOUT_DIR = os.path.join(HERE, "4_Heldout")

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
    """A gcsfs filesystem, or None if GCS is disabled / unavailable / GCS_ROOT unset."""
    if not _GCS["enabled"] or not getattr(C, "GCS_ROOT", ""):
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
    rel = os.path.relpath(local, HERE).replace(os.sep, "/")
    return C.GCS_ROOT.rstrip("/") + "/" + rel


def ensure_local(local):
    """Make `local` present by pulling it from GCS if it's only there. Return True if present after."""
    if os.path.exists(local):
        return True
    fs = _fs()
    if fs is None:
        return False
    key = _key(local)
    try:
        if fs.exists(key):
            os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
            fs.get(key, local)
            print(f"[gcs] pulled {key}")
            return True
    except Exception as e:
        print(f"[gcs][warn] pull failed for {key}: {e}")
    return False


def publish(*locals_):
    """Mirror produced artifacts to GCS (no-op if GCS disabled or the file is missing)."""
    fs = _fs()
    if fs is None:
        return
    for local in locals_:
        if not os.path.exists(local):
            continue
        key = _key(local)
        try:
            fs.put(local, key)
            print(f"[gcs] pushed -> {key}")
        except Exception as e:
            print(f"[gcs][warn] push failed for {key}: {e}")


def need(local, force):
    """True if the stage must (re)compute: forced, or not available locally and not pullable from GCS."""
    return True if force else not ensure_local(local)


# ----------------------------------------------------------------------------- stages
def stage_codelist(h, yr, force):
    """Phase 1 for THIS (horizon, lookback): per-code counts -> combined scoring -> codelist, scored on
    this lookback's TRAIN events. Each lookback gets its OWN codelist (lung_codelist_{h}_{yr}yr.csv) —
    no codelist is ever reused across lookbacks. Requires the canonical split (built first)."""
    codelist = os.path.join(FE_DIR, "codelist", f"lung_codelist_{h}_{yr}yr.csv")
    if not need(codelist, force):
        print(f"[codelist] {h}/{yr}yr: present — skip")
        return
    print(f"[codelist] {h}/{yr}yr: scoring codes on {yr}yr TRAIN data ...")
    _run("build_score_counts.py", [h, str(yr)], SNOMED_DIR)
    _run("Combined Scoring (ML+Stat).py", [h, str(yr)], SNOMED_DIR)
    _run("build_codelist.py", [h, str(yr)], SNOMED_DIR)
    publish(codelist)


def stage_split(h, force):
    """Create the canonical split for horizon `h` if it isn't already saved (in SPLIT_DIR / GCS)."""
    if not force and splits.exists(splits.split_path(h)):
        print(f"[split] {h}: canonical split present at {splits.split_path(h)} — skip")
        return
    _run("make_split.py", [h], HERE)        # writes directly to SPLIT_DIR (gs:// handled by splits)


def _fe_dir(h, yr):
    return os.path.join(FE_DIR, "output", h, f"{yr}yr")


def _model_dir(h, yr):
    return os.path.join(HERE, "output", h, f"{yr}yr")


def stage_fe(h, yr, force, engine):
    """Build the tiered feature matrix for horizon `h` at lookback `yr` years."""
    out = os.path.join(_fe_dir(h, yr), f"features_p005_{h}.parquet")
    if not need(out, force):
        print(f"[fe] {h}/{yr}yr: {os.path.basename(out)} present — skip")
        return
    fe_file = "build_features_polars.py" if engine == "polars" else "build_features.py"
    fe_path = os.path.join(FE_DIR, fe_file)
    if engine == "polars" and not os.path.exists(fe_path):
        print(f"[fe][warn] {fe_file} not found — falling back to pandas build_features.py")
        fe_path = os.path.join(FE_DIR, "build_features.py")
    print(f"[fe] {h}/{yr}yr: building features ({engine}) ...")
    fe = _load(fe_path, "fe_engine")
    mat, _ = fe.build(h, years=yr)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mat.to_parquet(out, index=False)
    print(f"[fe] {h}/{yr}yr: {mat.shape[0]:,} patients x {mat.shape[1]:,} cols -> {out}")
    publish(out)


def stage_stable(h, yr, force):
    """k-fold stability selection -> features_p005_{h}_stable.parquet (+ stable_features_{h}.csv)."""
    out_dir = _fe_dir(h, yr)
    stable = os.path.join(out_dir, f"features_p005_{h}_stable.parquet")
    flist = os.path.join(out_dir, f"stable_features_{h}.csv")
    if not need(stable, force):
        print(f"[stable] {h}/{yr}yr: {os.path.basename(stable)} present — skip")
        return
    fe_matrix = os.path.join(out_dir, f"features_p005_{h}.parquet")
    if not ensure_local(fe_matrix):
        print(f"[stable][skip] {h}/{yr}yr: feature matrix missing — run the fe stage first")
        return
    print(f"[stable] {h}/{yr}yr: k-fold stability selection ...")
    _load(os.path.join(FE_DIR, "stability_select.py"), "stability_select").select(
        h, in_path=fe_matrix, out_dir=out_dir)
    publish(stable, flist)


def stage_train(h, yr, force, Predictor):
    """Train on the stable matrix (split honored from the stamped column) -> model_{h}.joblib + png."""
    out_dir = _model_dir(h, yr)
    model = os.path.join(out_dir, f"model_{h}.joblib")
    png = os.path.join(out_dir, f"results_{h}.png")
    if not need(model, force):
        print(f"[train] {h}/{yr}yr: model present — skip")
        return
    stable = os.path.join(_fe_dir(h, yr), f"features_p005_{h}_stable.parquet")
    if not ensure_local(stable):
        print(f"[train][skip] {h}/{yr}yr: stable matrix missing — run the stable stage first")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n----- training {h}/{yr}yr: {stable} -----")
    p = Predictor(stable)
    (p.load_data()
       .explore_data()
       .preprocess_data(drop_threshold=C.DROP_THRESHOLD)
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
    publish(model, png)


def stage_heldout(h, yr, force):
    """Touch-once labelled held-out evaluation + deployable Platt (disjoint 30/70 calib/test)."""
    out_dir = _model_dir(h, yr)
    model = os.path.join(out_dir, f"model_{h}.joblib")
    if not ensure_local(model):
        print(f"[heldout][skip] {h}/{yr}yr: trained model missing — run the train stage first")
        return
    report = os.path.join(out_dir, f"heldout_recalib_{h}.txt")
    platt = os.path.join(out_dir, f"platt_calib_{h}.joblib")
    if not force and ensure_local(report):
        print(f"[heldout] {h}/{yr}yr: report present — skip")
        return
    _run("evaluate_heldout.py", [h, str(yr)], HELDOUT_DIR)
    publish(report, platt, os.path.join(out_dir, f"heldout_features_{h}.parquet"))


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description="Lung Cancer pipeline — one-command, cached, GCS-aware.")
    ap.add_argument("horizons", nargs="*", help="12mo and/or 1mo (default: config.HORIZONS)")
    ap.add_argument("--heldout", action="store_true", help="also run the held-out evaluation")
    ap.add_argument("--force", action="store_true", help="recompute every stage (ignore cache)")
    ap.add_argument("--no-gcs", action="store_true", help="local only — no GCS read/write")
    ap.add_argument("--engine", choices=["pandas", "polars"], default=getattr(C, "FE_ENGINE", "pandas"),
                    help="FE engine (default from config.FE_ENGINE)")
    ap.add_argument("--windows", nargs="*", type=int, default=None,
                    help="FE lookback windows in years (default: config.FE_WINDOWS, e.g. 5 10 20 100)")
    a = ap.parse_args()

    horizons = [h for h in a.horizons if h in ("12mo", "1mo")] or list(C.HORIZONS)
    windows = a.windows if a.windows else list(C.FE_WINDOWS)
    _GCS["enabled"] = not a.no_gcs
    gcs = "off" if _fs() is None else C.GCS_ROOT
    print("=" * 70)
    print(f"PIPELINE  horizons={horizons}  lookbacks={windows}yr  engine={a.engine}  GCS={gcs}"
          f"{'  [FORCE]' if a.force else ''}{'  +heldout' if a.heldout else ''}")
    print("=" * 70)

    Predictor = _load(os.path.join(MODEL_DIR, "lung_training.py"), "lung_training").LungCancerPredictor
    for h in horizons:                                   # both horizons (12mo then 1mo) by default
        print(f"\n{'#'*70}\n# {h}\n{'#'*70}")
        stage_split(h, a.force)                          # ONE split per horizon, shared across lookbacks
        for yr in windows:                               # each lookback window (5/10/20/100), self-contained
            stage_codelist(h, yr, a.force)               # codes RE-discovered on THIS lookback's train data
            stage_fe(h, yr, a.force, a.engine)           # FE uses this lookback's codelist + events
            stage_stable(h, yr, a.force)
            stage_train(h, yr, a.force, Predictor)
            if a.heldout:
                stage_heldout(h, yr, a.force)

    print("\n" + "=" * 70 + "\nPIPELINE COMPLETE\n" + "=" * 70)


if __name__ == "__main__":
    main()
