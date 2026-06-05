"""
Dual-horizon prediction: runs 12mo (early screening) + 1mo (acute alert) models
in parallel for one patient.

Top-level model output is one of three labels:
    "Cancer"         — at least one window fired (12mo OR 1mo)
    "No Cancer"      — at least one window had enough data and scored below threshold
    "Not enough data" — both windows lacked sufficient clinical features

Decision rule (per project_breast_dual_horizon.md):
    flagged_by_1mo  = score_1mo  >= threshold_1mo    # acute "in danger now"
    flagged_by_12mo = score_12mo >= threshold_12mo   # long-term risk

    if any fires:                  result = "Cancer"
    elif both insufficient data:   result = "Not enough data"
    else:                          result = "No Cancer"

Both raw scores are always returned so the receiving system can apply its own logic.

Reuses predict_single_patient.predict() under the hood — same calibration,
same per-band thresholds, same multi-seed ensemble logic. The only difference
is the OR-combine + 3-way label wrapper.

Usage:
    python predict_dual_horizon.py --features-csv path/to/one_patient.csv
    python predict_dual_horizon.py --features-json '{"AGE_AT_INDEX": 67, ...}'
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
from predict_single_patient import predict  # noqa: E402

logger = logging.getLogger(__name__)

WINDOWS = ("1mo", "12mo")  # order matters only for cosmetic output
RESULTS_ROOT = PROJECT_ROOT / "3_Modeling" / "results"


def _verdict_for_window(score):
    """Translate a single window's predict() output into its own 3-way label.

        decision == 1            → "Cancer"
        insufficient_data == True → "Not enough data"
        otherwise                → "No Cancer"
    """
    if score["decision"] == 1:
        return "Cancer"
    if score.get("insufficient_data", False):
        return "Not enough data"
    return "No Cancer"


def _classify(scores):
    """Combine per-window verdicts into one overall clinical label + which fired.

    Rule:
        any window says "Cancer"          → "Cancer"
        both windows say "Not enough data" → "Not enough data"
        otherwise                         → "No Cancer"

    Note the asymmetry: if 1mo has no recent data (insufficient) but 12mo has
    plenty of chronic data and scored low, the patient is a confident "No Cancer"
    — we don't need both windows to have data, just one.
    """
    fired_by = [w for w in WINDOWS if scores[w]["decision"] == 1]
    if fired_by:
        return "Cancer", fired_by
    if all(scores[w].get("insufficient_data", False) for w in WINDOWS):
        return "Not enough data", []
    return "No Cancer", []


def predict_dual_horizon(feature_vector):
    """Run both 12mo + 1mo models on one patient, return 3-way clinical label.

    feature_vector: dict or pd.Series of {feature_name: value} (matches what
        predict_single_patient.predict() expects).

    Returns:
        {
          "result": "Cancer" | "No Cancer" | "Not enough data",   # overall (OR of windows)
          "by_model": {
            "1mo":  {"verdict": "Cancer"|"No Cancer"|"Not enough data",
                     "risk_proba": float, "threshold_used": float,
                     "age_band": str, "insufficient_data": bool, ...},
            "12mo": {... same shape ...},
          },
          "fired_by": list[str],   # which windows said "Cancer"; subset of ["1mo","12mo"]
        }
    """
    scores = {w: predict(feature_vector, w, results_root=RESULTS_ROOT) for w in WINDOWS}
    # Tag each window's output with its own 3-way verdict so callers can see
    # exactly what each model said, side-by-side.
    by_model = {w: {"verdict": _verdict_for_window(scores[w]), **scores[w]} for w in WINDOWS}
    result, fired_by = _classify(scores)
    return {
        "result": result,
        "by_model": by_model,
        "fired_by": fired_by,
    }


def main():
    p = argparse.ArgumentParser(description="Dual-horizon (12mo + 1mo) prediction for one patient")
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

    result = predict_dual_horizon(fv)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
