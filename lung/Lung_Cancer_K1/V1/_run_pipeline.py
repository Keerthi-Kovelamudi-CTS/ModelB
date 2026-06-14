"""
Lung B+C - shared pipeline driver: cohort SQL -> events -> feature matrix -> model.

Flow per config (window x ratio):
  FE   : 2_FE/SQL/{window}_{ratio}.sql  --(BigQuery)-->  events
         --> transform_features.extract_lung_risk_factors  -->  feature matrix
  MODEL: 3_Modeling/lung_training.LungCancerPredictor      -->  models + metrics

Outputs (features_*.csv, model_*.joblib, results_*.png, metrics) land in out_dir.

Modeling augmentations (scale_pos_weight instead of SMOTE, cumimp99/all+v2 feature
selection, isotonic per-age-band calibration) live in the shared LungCancerPredictor class.
"""
import os, sys, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
FE_DIR = os.path.join(ROOT, "2_FE")
MODEL_DIR = os.path.join(ROOT, "3_Modeling")
PROJECT_ID = "prj-cts-ai-dev-sp"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so the module's custom classes (CalibratedLungModel,
    # _MaskSelector) carry a resolvable __module__ and joblib.dump can pickle them.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export_query_to_csv(sql, out_csv, project_id=PROJECT_ID,
                        gcs_prefix="gs://cancer_transformed_ai/lung_bc", tag=None):
    """Run a SQL via the bq CLI (EXPORT DATA -> GCS -> one CSV at out_csv). SHARED export used by
    both run_fe and the lookback experiment, so the bq-export logic lives in exactly one place.
    Uses the VM's bq/gsutil auth (instance service account)."""
    import subprocess, glob
    sql = sql.rstrip().rstrip(";")
    tag = tag or os.path.splitext(os.path.basename(out_csv))[0]
    gcs = f"{gcs_prefix}/{tag}"
    raw = os.path.join(os.path.dirname(os.path.abspath(out_csv)), "raw_" + tag)
    os.makedirs(raw, exist_ok=True)
    for f in glob.glob(os.path.join(raw, "p_*.csv")):
        os.remove(f)
    print(f"[export] {tag}: bq EXPORT DATA -> {gcs} (project {project_id})")
    export = (f"EXPORT DATA OPTIONS(uri='{gcs}/p_*.csv', format='CSV', overwrite=true, header=true) AS {sql}")
    r = subprocess.run(["bq", "query", "--use_legacy_sql=false", "--format=none",
                        f"--project_id={project_id}", export], capture_output=True, text=True)
    assert r.returncode == 0, f"bq export failed:\n{r.stderr[-1200:]}"
    subprocess.run(f"gsutil -q -m cp '{gcs}/p_*.csv' '{raw}/'", shell=True, check=True)
    subprocess.run(f"gsutil -q -m rm -r '{gcs}/' 2>/dev/null", shell=True)
    shards = sorted(glob.glob(os.path.join(raw, "p_*.csv")))
    assert shards, "no CSV shards exported"
    subprocess.run(f"head -1 '{shards[0]}' > '{out_csv}'", shell=True, check=True)
    subprocess.run(f"for s in '{raw}'/p_*.csv; do tail -n +2 \"$s\" >> '{out_csv}'; done", shell=True, check=True)
    for s in shards:
        os.remove(s)
    print(f"[export] {out_csv}: {sum(1 for _ in open(out_csv))-1} rows")
    return out_csv


def run_fe(window, ratio, out_dir, project_id=PROJECT_ID, time_window_months=None,
           gcs_prefix="gs://cancer_transformed_ai/lung_bc"):
    """Run cohort SQL on our project -> build feature matrix -> save to out_dir. Returns matrix path."""
    sql_path = os.path.join(FE_DIR, "SQL", f"{window}_{ratio}.sql")
    assert os.path.exists(sql_path), f"missing SQL: {sql_path}"
    sys.path.insert(0, FE_DIR)
    import transform_features as fe  # type: ignore  # resolved at runtime via sys.path.insert
    csv_path = os.path.join(out_dir, f"events_{window}_{ratio}.csv")
    export_query_to_csv(open(sql_path).read(), csv_path, project_id=project_id,
                        gcs_prefix=gcs_prefix, tag=f"{window}_{ratio}")
    mat = fe.extract_lung_risk_factors(csv_path, time_window_months=time_window_months)
    drop = [c for c in mat.columns if "_SNOMED_CODES" in c or "_terms" in c.lower()]
    mat = mat.drop(columns=drop, errors="ignore")
    if "cancer_class" in mat.columns:
        mat["cancer_class"] = mat.pop("cancer_class")
    feat_path = os.path.join(out_dir, f"features_{window}_{ratio}.csv")
    mat.to_csv(feat_path, index=False)
    print(f"[FE] saved {mat.shape[0]} patients x {mat.shape[1]} features -> {feat_path}")
    return feat_path


def run_model(feat_path, out_dir, window, ratio):
    """Train on the feature matrix (shared training class) -> models + metrics in out_dir.

    Improved modeling chain:
      * scale_pos_weight instead of SMOTE  -> handle_imbalance('none') + use_resampled=False
        (models already carry class_weight / scale_pos_weight; true distribution kept so
        calibration is meaningful).
      * cumimp99 / all+v2 instead of top-150 mutual_info -> FEAT_METHOD env (default 'all').
      * isotonic per-age-band calibration -> calib_size split + calibrate().
    """
    LungCancerPredictor = _load_module(
        os.path.join(MODEL_DIR, "lung_training.py"), "lung_training"
    ).LungCancerPredictor
    tag = f"{window}_{ratio}"
    feat_method = os.environ.get("FEAT_METHOD", "cumimp")   # 'cumimp' (cumimp99, default) or 'all' (all+v2)
    p = LungCancerPredictor(feat_path)
    (p.load_data()
       .explore_data()
       .preprocess_data(drop_threshold=0.90)
       .split_data(test_size=0.10, calib_size=0.10)       # 80/10/10 train / calib-val / internal-test
       .handle_imbalance(method="none")                   # scale_pos_weight (no SMOTE)
       .select_features(method=feat_method, threshold=0.99)
       .train_and_evaluate(use_resampled=False))
    if os.environ.get("TUNE", "1") != "0":                # Optuna 100-trial tuning; set TUNE=0 to skip
        p.hyperparameter_tuning()                         # (slow; ~noise-level gain, often discarded by the ensemble)
    (p.create_ensemble()
       .calibrate(method="isotonic", by_age_band=True)
       .final_evaluation()
       .plot_results(save_path=os.path.join(out_dir, f"results_{tag}.png"))
       .save_model(os.path.join(out_dir, f"model_{tag}.joblib")))
    return p


def run_config(window, ratio, out_dir, project_id=PROJECT_ID, do_model=True):
    os.makedirs(out_dir, exist_ok=True)
    feat_path = run_fe(window, ratio, out_dir, project_id=project_id)
    if do_model:
        run_model(feat_path, out_dir, window, ratio)
    return feat_path
