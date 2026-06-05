"""
Single-patient prediction (multi-seed ensemble + per-band calibration).

Loads saved artifacts from 3_Modeling/results/1_training/{window}/saved_models/:
  - config.json: selected_features, seeds, ensemble_models, calibrators_per_band,
    thresholds_per_band, tier90_thresholds_per_band, global_calibrator, global_threshold,
    band_bins, band_labels
  - seed_<N>/xgboost_model.pkl, lightgbm_model.pkl, catboost_model.pkl
  - seed_<N>/meta_learner.pkl

Accepts two input modes:
  1. Pre-computed flat feature vector (CSV row OR JSON dict) — fast path
  2. JSON-packed parquet/CSV from our prod FE (single-row) — auto-expanded via
     `_load_features.load_alex_features()` so the JSON `transformed_features`
     column is unpacked and dtypes are applied from the schema sidecar.

Usage:
    # Pre-flat (CSV with one row of feature columns)
    python predict_single_patient.py --window 12mo --features-csv one_patient.csv

    # Pre-flat (JSON dict of feature values)
    python predict_single_patient.py --window 12mo --features-json '{"AGE_AT_INDEX": 67, ...}'

    # JSON-packed FE output (with sidecar)
    python predict_single_patient.py --window 12mo \
        --features-file prostate_features_12mo.parquet \
        --schema-path  prostate_features_12mo_schema.json
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

# Required so joblib can unpickle CalibratorWrapper from saved configs.
# Also gives us `load_alex_features` for JSON-packed inputs.
sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
from _calibrator import CalibratorWrapper  # noqa: F401
from _load_features import load_alex_features  # noqa: E402


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
    # This is also the "not enough data" signal — we cannot make a confident call.
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


def predict_from_file(features_file, window, schema_path=None):
    """Predict from a JSON-packed FE output file (parquet/CSV).

    Loads via `load_alex_features` which auto-detects the JSON column
    (`transformed_features`/`snomed_features`/`json_features`) and applies
    the schema sidecar for clean dtypes. Expects exactly one row.
    """
    df = load_alex_features(features_file, schema_path=schema_path)
    if len(df) != 1:
        raise ValueError(
            f"{features_file}: expected 1 patient row, got {len(df)}. "
            "Pre-filter to the target patient before calling predict_from_file()."
        )
    return predict(df.iloc[0], window)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", required=True, choices=["1mo", "3mo", "6mo", "12mo"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--features-csv",  help="One-row CSV with FLAT feature columns")
    g.add_argument("--features-json", help="JSON string of FLAT {feature_name: value}")
    g.add_argument("--features-file", help="Parquet/CSV with JSON-packed `transformed_features` column (one patient row)")
    p.add_argument("--schema-path", help="Schema sidecar JSON (for --features-file with JSON column)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.features_file:
        result = predict_from_file(args.features_file, args.window, schema_path=args.schema_path)
    elif args.features_csv:
        df = pd.read_csv(args.features_csv)
        if len(df) != 1:
            raise ValueError(f"Expected 1 row, got {len(df)}")
        result = predict(df.iloc[0], args.window)
    else:
        fv = json.loads(args.features_json)
        result = predict(fv, args.window)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
