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

    # Force zero risk if patient has no clinical features (matches training assumption)
    clinical_cols = [c for c in selected if c not in ("AGE_AT_INDEX", "AGE_BAND")]
    if clinical_cols and (X[clinical_cols].abs().sum(axis=1).iloc[0] == 0):
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
        "n_seeds": len(seed_artifacts),
    }


def predict_from_events(patient_json, window):
    """Build features from raw events JSON, then predict.

    TODO (next session — biggest piece): replicate the FE pipeline (preprocess →
    sanity → clean → FE → cleanup) on a single patient's events list. The
    challenge is that 2_Feature_Engineering/3_pipeline.py is built for batch
    DataFrame ops. Options:
      A. Reuse the batch FE code with a 1-row DataFrame (slow but correct).
      B. Write a lean per-patient FE that mirrors the same feature definitions.
      C. Defer feature computation to the API layer; this script only predicts.

    For now this is a stub. Use predict() with a pre-computed feature vector.
    """
    raise NotImplementedError(
        "Building features from raw events is the larger TODO. "
        "Use predict(feature_vector, window) with already-computed features. "
        "See 2_Feature_Engineering/0_run_pipeline.py for the batch FE."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", required=True, choices=["1mo", "3mo", "6mo", "12mo"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--features-csv", help="One-row CSV with feature columns (and AGE_AT_INDEX)")
    g.add_argument("--features-json", help="JSON string of {feature_name: value}")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
