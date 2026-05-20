"""
Dual-horizon prediction: runs 12mo (early screening) + 1mo (acute alert) models
in parallel for one patient.

Top-level model output is one of three labels (clinician-facing):
    "Cancer"          — at least one window fired (12mo OR 1mo)
    "No Cancer"       — at least one window had enough data and scored below threshold
    "Not enough data" — both windows lacked sufficient clinical features

Decision rule:
    flagged_by_1mo  = score_1mo  >= threshold_1mo    # acute "in danger now"
    flagged_by_12mo = score_12mo >= threshold_12mo   # long-term risk

    if any fires:                  result = "Cancer"
    elif both insufficient data:   result = "Not enough data"
    else:                          result = "No Cancer"

Both raw scores + per-window verdicts are always returned so the receiving
system can apply its own decision logic.

Accepts the same input modes as predict_single_patient:
  - FLAT one-row CSV  (--features-csv)
  - FLAT JSON dict    (--features-json)
  - JSON-packed FE output parquet/CSV (--features-file, optionally --schema-path)

The JSON-packed mode is the production path — it accepts the output of
2_Feature_Engineering/4_transform_features_json.py directly, with the
matching schema sidecar.

Usage:
    # Production: JSON-packed FE output + sidecar
    python predict_dual_horizon.py \
        --features-file prostate_features_unwindowed.parquet \
        --schema-path  prostate_features_unwindowed_schema.json

    # Quick test: already-flat feature dict
    python predict_dual_horizon.py --features-json '{"AGE_AT_INDEX": 67, ...}'
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
sys.path.insert(0, str(PROJECT_ROOT / "6_Explainability"))
from predict_single_patient import predict  # noqa: E402
from _load_features import load_alex_features  # noqa: E402

logger = logging.getLogger(__name__)

WINDOWS = ("1mo", "12mo")  # order matters only for cosmetic output


def _verdict_for_window(score):
    """Translate a single window's predict() output into its own 3-way label.

        decision == 1             → "Cancer"
        insufficient_data == True → "Not enough data"
        otherwise                 → "No Cancer"
    """
    if score["decision"] == 1:
        return "Cancer"
    if score.get("insufficient_data", False):
        return "Not enough data"
    return "No Cancer"


def _classify(scores):
    """Combine per-window verdicts into one overall clinical label + which fired.

    Rule:
        any window says "Cancer"           → "Cancer"
        both windows say "Not enough data" → "Not enough data"
        otherwise                          → "No Cancer"

    Asymmetry on "Not enough data": if only ONE window is sparse and the
    other has plenty of data + scored low, the patient is a confident
    "No Cancer" — don't downgrade to "Not enough data" because of one
    incomplete window.
    """
    fired_by = [w for w in WINDOWS if scores[w]["decision"] == 1]
    if fired_by:
        return "Cancer", fired_by
    if all(scores[w].get("insufficient_data", False) for w in WINDOWS):
        return "Not enough data", []
    return "No Cancer", []


def predict_dual_horizon(feature_vector, top_n=5):
    """Run both 12mo + 1mo models on one patient, return 3-way clinical label
    + SHAP top-N contributing factors per window (always included).

    feature_vector: dict or pd.Series of {feature_name: value} (FLAT — already
        expanded from any upstream JSON-packed format).
    top_n: number of top contributing features to return per window (default 5).

    Returns:
        {
          "result":   "Cancer" | "No Cancer" | "Not enough data",   # overall
          "by_model": {
            "1mo":  {"verdict": ..., "risk_proba": float, "decision": 0/1,
                     "threshold_used": float, "age_band": str,
                     "insufficient_data": bool,
                     "top_factors": [{"feature": str, "value": float, "shap": float}, ...]
                     ...},
            "12mo": {... same shape ...},
          },
          "fired_by": list[str],   # which windows said "Cancer"; subset of ["1mo","12mo"]
        }
    """
    from explain_single_patient import explain as _explain_one

    # Each window is independent (different model, different SHAP computation).
    # Run both in parallel on separate threads — XGBoost/LightGBM/CatBoost/SHAP
    # all release the GIL during their C++ work, so this is a real ~2× speedup.
    def _run_window(w):
        score = predict(feature_vector, w)
        try:
            top_factors = _explain_one(feature_vector, w, top_n=top_n)
        except Exception as e:
            logger.warning(f"SHAP explain failed for {w}: {e}")
            top_factors = None
        return w, score, top_factors

    with ThreadPoolExecutor(max_workers=len(WINDOWS)) as pool:
        results = list(pool.map(_run_window, WINDOWS))

    scores = {w: s for w, s, _ in results}
    by_model = {
        w: {
            "verdict": _verdict_for_window(scores[w]),
            **scores[w],
            "top_factors": tf,
        }
        for w, _, tf in results
    }
    result, fired_by = _classify(scores)
    return {
        "result": result,
        "by_model": by_model,
        "fired_by": fired_by,
    }


def predict_dual_horizon_from_file(features_file, schema_path=None, top_n=5):
    """Production entry point: predict from a JSON-packed FE output file.

    Loads via `load_alex_features` (auto-expands the JSON column, applies
    the schema sidecar). Expects exactly one patient row. Always returns
    SHAP top-N factors per window.
    """
    df = load_alex_features(features_file, schema_path=schema_path)
    if len(df) != 1:
        raise ValueError(
            f"{features_file}: expected 1 patient row, got {len(df)}. "
            "Pre-filter to the target patient before calling this."
        )
    return predict_dual_horizon(df.iloc[0], top_n=top_n)


def main():
    p = argparse.ArgumentParser(description="Dual-horizon (1mo + 12mo) prediction for one patient")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--features-csv",  help="One-row CSV with FLAT feature columns")
    g.add_argument("--features-json", help="JSON string of FLAT {feature_name: value}")
    g.add_argument("--features-file", help="Parquet/CSV with JSON-packed `transformed_features` column (one patient row)")
    p.add_argument("--schema-path", help="Schema sidecar JSON (for --features-file)")
    p.add_argument("--top-n", type=int, default=5, help="Number of top SHAP factors per window (default: 5)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.features_file:
        result = predict_dual_horizon_from_file(
            args.features_file, schema_path=args.schema_path, top_n=args.top_n,
        )
    elif args.features_csv:
        df = pd.read_csv(args.features_csv)
        if len(df) != 1:
            raise ValueError(f"Expected 1 row, got {len(df)}")
        result = predict_dual_horizon(df.iloc[0], top_n=args.top_n)
    else:
        fv = json.loads(args.features_json)
        result = predict_dual_horizon(fv, top_n=args.top_n)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
