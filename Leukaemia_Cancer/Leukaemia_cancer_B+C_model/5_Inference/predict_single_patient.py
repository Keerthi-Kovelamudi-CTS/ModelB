"""
Single-patient prediction for v3+ models (multi-seed ensemble + per-band calibration).

Loads saved artifacts from 3_Modeling/results/1_training/{window}/saved_models/:
  - config.json: selected_features, seeds, ensemble_models, calibrators_per_band,
    thresholds_per_band, tier90_thresholds_per_band, global_calibrator, global_threshold,
    band_bins, band_labels
  - seed_<N>/xgboost_model.pkl, lightgbm_model.pkl, catboost_model.pkl
  - seed_<N>/meta_learner.pkl

Produces calibrated cancer risk + tier decision for one patient feature vector.

NOTE: this script ASSUMES the feature vector is already computed (e.g., via the
FE pipeline run on the patient's history). Building features from raw events
JSON is the larger piece — see TODO in `predict_from_events()`.

Usage:
    python predict_single_patient.py --window 12mo --features-csv path/to/one_patient.csv
    python predict_single_patient.py --window 12mo --features-json '{"AGE_AT_INDEX": 67, ...}'
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_RESULTS = PROJECT_ROOT / "3_Modeling" / "results" / "1_training"

# Required so joblib can unpickle CalibratorWrapper from saved configs
sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
from _calibrator import CalibratorWrapper  # noqa: F401


def load_artifacts(window):
    """Load model config + per-seed base models + meta-learners."""
    model_dir = TRAIN_RESULTS / window / "saved_models"
    if not model_dir.exists():
        raise FileNotFoundError(f"No saved models for {window} at {model_dir}")

    cfg = joblib.load(model_dir / "config.json")
    seeds = cfg.get("seeds", [42])
    ensemble_models = cfg["ensemble_models"]

    seed_artifacts = {}
    for seed in seeds:
        seed_dir = model_dir / f"seed_{seed}"
        if not seed_dir.exists():
            raise FileNotFoundError(
                f"Expected multi-seed format with seed_{seed}/ subdir. "
                f"Legacy single-seed format not supported by this script."
            )
        models = {name: joblib.load(seed_dir / f"{name}_model.pkl") for name in ensemble_models}
        meta = joblib.load(seed_dir / "meta_learner.pkl")
        seed_artifacts[seed] = {"models": models, "meta": meta}

    return cfg, seed_artifacts


def predict(feature_vector, window):
    """Predict cancer risk for one patient.

    feature_vector: pandas Series indexed by feature names, OR dict {name: value}.
    window: '1mo', '3mo', '6mo', '12mo'.

    Returns dict with: risk_proba (calibrated), decision (0/1 at default threshold),
    age_band, threshold_used, tier90_decision, raw_proba (uncalibrated).
    """
    cfg, seed_artifacts = load_artifacts(window)
    selected = cfg["selected_features"]

    # Coerce to DataFrame row, align to selected features (fill missing with 0)
    if isinstance(feature_vector, dict):
        s = pd.Series(feature_vector)
    elif isinstance(feature_vector, pd.Series):
        s = feature_vector
    else:
        raise TypeError(f"feature_vector must be dict or Series, got {type(feature_vector)}")
    X = pd.DataFrame([{f: s.get(f, 0) for f in selected}], columns=selected).fillna(0)

    # Per-seed: predict with base models, stack via meta-learner
    per_seed_preds = []
    ensemble_models = cfg["ensemble_models"]
    for seed, art in seed_artifacts.items():
        base_preds = np.array([art["models"][n].predict_proba(X)[0, 1] for n in ensemble_models])
        meta = art["meta"]
        stacked = meta.predict_proba(base_preds.reshape(1, -1))[0, 1]
        per_seed_preds.append(stacked)
    raw_proba = float(np.mean(per_seed_preds))

    # Determine age band
    age = float(s.get("AGE_AT_INDEX", 65.0))
    band_bins = cfg["band_bins"]
    band_labels = cfg["band_labels"]
    band_idx = np.searchsorted(band_bins[1:], age, side="right")
    band_idx = min(band_idx, len(band_labels) - 1)
    age_band = band_labels[band_idx]

    # Apply per-band calibration + threshold (fall back to global if band missing)
    calibrators = cfg["calibrators_per_band"]
    thresholds = cfg["thresholds_per_band"]
    tier90_thresholds = cfg.get("tier90_thresholds_per_band", {})
    global_cal = cfg["global_calibrator"]
    global_thr = cfg["global_threshold"]

    cal = calibrators.get(age_band, global_cal)
    thr = thresholds.get(age_band, global_thr)
    t90 = tier90_thresholds.get(age_band)

    risk_proba = float(cal.transform(np.array([raw_proba]))[0])
    decision = int(risk_proba >= thr)
    tier90_decision = int(risk_proba >= t90) if t90 is not None and not np.isnan(t90) else None

    # Force zero risk if patient has no clinical features (matches training assumption).
    # Expose this as insufficient_data so the dual-horizon combiner can tell
    # "confident no-cancer" apart from "we never had enough signal to judge".
    clinical_cols = [c for c in selected if c not in ("AGE_AT_INDEX", "AGE_BAND")]
    insufficient_data = bool(clinical_cols and (X[clinical_cols].abs().sum(axis=1).iloc[0] == 0))
    if insufficient_data:
        risk_proba = 0.0
        decision = 0
        if tier90_decision is not None:
            tier90_decision = 0

    return {
        "window": window,
        "age": age,
        "age_band": age_band,
        "raw_proba": raw_proba,
        "risk_proba": risk_proba,
        "threshold_used": float(thr),
        "decision": decision,
        "tier90_threshold": float(t90) if t90 is not None and not np.isnan(t90) else None,
        "tier90_decision": tier90_decision,
        "insufficient_data": insufficient_data,
        "n_seeds": len(seed_artifacts),
    }


def build_feature_row_from_events(events, window):
    """Compute the model's feature vector for ONE patient from raw events.

    Option A (faithful): reuses the EXACT training FE functions from
    2_Feature_Engineering, so features match training by construction.

    `events`: list[dict] or DataFrame with the SQL-output schema per event row
    (PATIENT_GUID, EVENT_TYPE, EVENT_DATE, SNOMED_C_T_CONCEPT_ID/MED_CODE_ID,
    TERM/DRUG_TERM, VALUE, SEX, PATIENT_AGE, optional ANCHOR_DATE). For inference
    ANCHOR_DATE defaults to the patient's most-recent EVENT_DATE.

    Cohort-only steps are intentionally NOT applied (min_obs gating, NZV removal,
    5_cleanup column-drops) — `predict()` aligns this row to the model's
    selected_features, and 5_cleanup only drops columns (no value transforms).

    ⚠️ MUST be validated once on the VM: run on a real patient and diff this
    vector against that patient's row in feature_matrix_final — they must match.
    """
    import importlib
    fe_dir = str(PROJECT_ROOT / "2_Feature_Engineering")
    if fe_dir not in sys.path:
        sys.path.insert(0, fe_dir)
    cfg  = importlib.import_module("config")
    pre  = importlib.import_module("0_preprocess_to_fe")
    pipe = importlib.import_module("3_pipeline")
    canc = importlib.import_module("4_cancer_features")

    df = pd.DataFrame(events).copy()
    df.columns = df.columns.str.upper()
    df["EVENT_DATE"] = pd.to_datetime(df["EVENT_DATE"], errors="coerce")
    # anchor = MAX(event) for inference if not supplied (mirrors 0_preprocess_to_fe)
    if "ANCHOR_DATE" not in df.columns:
        df["ANCHOR_DATE"] = df.groupby("PATIENT_GUID")["EVENT_DATE"].transform("max")
    df["ANCHOR_DATE"] = pd.to_datetime(df["ANCHOR_DATE"], errors="coerce")
    if "MONTHS_BEFORE_ANCHOR" not in df.columns:
        df["MONTHS_BEFORE_ANCHOR"] = ((df["ANCHOR_DATE"] - df["EVENT_DATE"]).dt.days // 30).astype("Int64")
    df = df.rename(columns={"ANCHOR_DATE": "INDEX_DATE", "PATIENT_AGE": "AGE_AT_INDEX",
                            "MONTHS_BEFORE_ANCHOR": "MONTHS_BEFORE_INDEX", "CANCER_CLASS": "LABEL"})
    if "LABEL" not in df.columns:
        df["LABEL"] = 0
    is_obs = df["EVENT_TYPE"].astype(str).str.lower().str.startswith("obs")
    df["CODE_ID"] = np.where(is_obs, df.get("SNOMED_C_T_CONCEPT_ID"), df.get("MED_CODE_ID"))
    df["CODE_ID"] = pd.to_numeric(df["CODE_ID"], errors="coerce").astype("Int64")
    if "TERM" not in df.columns: df["TERM"] = ""
    if "DRUG_TERM" in df.columns:
        df["TERM"] = np.where(is_obs, df["TERM"], df["DRUG_TERM"])

    obs_map, med_map = pre.load_mapping(pre.DEFAULT_MAPPING)   # SAME mapping training used
    df["CATEGORY"] = None
    df.loc[is_obs,  "CATEGORY"] = pre.assign_category(df.loc[is_obs,  "CODE_ID"], obs_map).values
    df.loc[~is_obs, "CATEGORY"] = pre.assign_category(df.loc[~is_obs, "CODE_ID"], med_map).values
    # v3: use the SAME per-window A/B midpoint the training preprocess uses
    pre.TIME_WINDOW_MID_MONTHS = getattr(pre, "TIME_WINDOW_MID_BY_WINDOW", {}).get(window, pre.TIME_WINDOW_MID_MONTHS)
    months = pd.to_numeric(df["MONTHS_BEFORE_INDEX"], errors="coerce").fillna(-1).astype(int).values
    df["TIME_WINDOW"] = pre.assign_time_window(months)
    cat = df[df["CATEGORY"].notna() & df["TIME_WINDOW"].notna()].copy()
    # NB: no min_obs gate, no placeholder needed (single patient is always scored);
    # if cat is empty, predict()'s "all-zero clinical features ⇒ 0 risk" rule fires.

    clin = cat[cat["EVENT_TYPE"].astype(str).str.lower().str.startswith("obs")].copy()
    med  = cat[~cat["EVENT_TYPE"].astype(str).str.lower().str.startswith("obs")].copy()
    for fr in (clin, med):
        fr.columns = fr.columns.str.upper()
        if "VALUE" in fr.columns: fr["VALUE"] = pd.to_numeric(fr["VALUE"], errors="coerce")
    W = window.upper()
    # build_clinical_features reads a module-global `window_name` for the encoder
    # path (FE_RESULTS/<window>/encoder_mappings.json). At inference the TRAINING
    # encoder must already exist there so categorical indices match training
    # (it is loaded, not re-fit). Set the global the same way the batch run does.
    pipe.window_name = window

    # ── exact builder chain from 0_run_pipeline.run_feature_engineering ──
    # NB: the batch builders assume cohort-scale inputs; for a SINGLE patient a
    # degenerate group (e.g. zero medication events) can make a builder raise. At
    # inference that simply means "no events of that type", so those features are 0
    # (predict() fills any missing selected feature with 0). We guard each builder
    # so a degenerate group → its features default to 0, never a crash. The CORE
    # clinical builder is NOT guarded — if it fails that is a real error.
    clin_feats = pipe.build_clinical_features(clin, cfg)   # core; let it raise
    try:
        med_feats = pipe.build_medication_features(med, cfg)
    except Exception as e:
        logger.warning(f"  build_medication_features skipped (no/degenerate med events: {e})")
        med_feats = pd.DataFrame(index=clin_feats.index)
    fm = clin_feats.join(med_feats, how="left")
    med_cols = [c for c in fm.columns if c.startswith("MED_")]
    if med_cols:
        fm[med_cols] = fm[med_cols].fillna(0)
    try:
        fm = pipe.build_interaction_features(fm, cfg)
    except Exception as e:
        logger.warning(f"  build_interaction_features skipped ({e})")
    for fn in (pipe.build_advanced_features, pipe.extract_maximum_features,
               canc.build_cancer_specific_features, pipe.build_new_signal_features,
               pipe.build_trend_features, pipe.build_acceleration_features):
        try:
            add = fn(clin, med, fm, W, cfg)
            fm = fm.join(add, how="left", rsuffix="_x").fillna(0).replace([np.inf, -np.inf], 0)
            fm = fm.loc[:, ~fm.columns.duplicated()]
        except Exception as e:
            logger.warning(f"  {fn.__name__} skipped (degenerate single-patient input: {e})")
    if len(fm) != 1:
        logger.warning(f"expected 1 feature row, got {len(fm)} — using the first")
    return fm.iloc[0]


def predict_from_events(events, window):
    """Raw patient events → feature vector → calibrated risk + decision."""
    fv = build_feature_row_from_events(events, window)
    cfg, _ = load_artifacts(window)
    selected = cfg["selected_features"]
    covered = sum(1 for f in selected if f in fv.index and fv.get(f, 0) != 0)
    logger.info(f"  feature coverage: {covered}/{len(selected)} selected features non-zero "
                f"(rest default to 0 — verify against a training row before trusting)")
    return predict(fv, window)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", required=True, choices=["1mo", "12mo"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--features-csv", help="One-row CSV with feature columns (and AGE_AT_INDEX)")
    g.add_argument("--features-json", help="JSON string of {feature_name: value}")
    g.add_argument("--events-json", help="JSON list of raw event rows (SQL schema) — runs FE then predicts")
    g.add_argument("--events-csv", help="CSV of raw event rows for one patient — runs FE then predicts")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.events_json or args.events_csv:
        events = json.loads(args.events_json) if args.events_json else pd.read_csv(args.events_csv).to_dict("records")
        result = predict_from_events(events, args.window)
    else:
        if args.features_csv:
            df = pd.read_csv(args.features_csv)
            if len(df) != 1:
                raise ValueError(f"Expected 1 row, got {len(df)}")
            fv = df.iloc[0]
        else:
            fv = json.loads(args.features_json)
        result = predict(fv, args.window)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
