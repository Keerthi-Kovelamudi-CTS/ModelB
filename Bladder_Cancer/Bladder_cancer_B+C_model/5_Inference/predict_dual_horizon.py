"""
Dual-horizon prediction for the EMIS cancers: runs the 12mo (early screening)
+ 1mo (acute alert) models together for ONE patient and combines them into a
single clinician-facing verdict.

Mirrors the Breast/Prostate B+C combiner so all cancers are deployment-uniform.
The EMIS difference: each window needs its OWN feature engineering (different
lookback), so the raw-events path runs predict_from_events() per window rather
than reusing one flat feature vector.

Top-level result (clinician-facing):
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
system can apply its own logic.

Usage:
    # Production: raw events JSON (runs window-specific FE for each horizon)
    python predict_dual_horizon.py --events-json '[{...event...}, ...]'
    python predict_dual_horizon.py --events-csv patient_events.csv

    # Already-computed flat features (same vector used for both windows)
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
sys.path.insert(0, str(SCRIPT_DIR))
from predict_single_patient import predict, predict_from_events  # noqa: E402

logger = logging.getLogger(__name__)

# v3-aligned dual horizon: 1mo (12-month lookback) + 12mo (20-year lookback).
WINDOWS = ("1mo", "12mo")  # order is cosmetic only


def _verdict_for_window(score):
    """Translate one window's predict() output into its own 3-way label."""
    if score["decision"] == 1:
        return "Cancer"
    if score.get("insufficient_data", False):
        return "Not enough data"
    return "No Cancer"


def _classify(scores):
    """Combine per-window verdicts into one overall label + which fired.

    Asymmetry on "Not enough data": if only ONE window is sparse and the other
    has plenty of data + scored low, the patient is a confident "No Cancer" —
    don't downgrade because of one incomplete window.
    """
    fired_by = [w for w in WINDOWS if scores[w]["decision"] == 1]
    if fired_by:
        return "Cancer", fired_by
    if all(scores[w].get("insufficient_data", False) for w in WINDOWS):
        return "Not enough data", []
    return "No Cancer", []


def _assemble(scores):
    by_model = {
        w: {"verdict": _verdict_for_window(scores[w]), **scores[w]}
        for w in WINDOWS
    }
    result, fired_by = _classify(scores)
    return {"result": result, "by_model": by_model, "fired_by": fired_by}


def predict_dual_horizon_from_events(events):
    """Run both horizons from ONE patient's raw events.

    Each window runs its own window-specific feature engineering via
    predict_from_events (1mo = 12-month lookback, 12mo = 20-year lookback),
    so the two scores see different histories — as intended.
    """
    def _run(w):
        return w, predict_from_events(events, w)

    with ThreadPoolExecutor(max_workers=len(WINDOWS)) as pool:
        results = dict(pool.map(_run, WINDOWS))
    return _assemble(results)


def predict_dual_horizon(feature_vector):
    """Run both horizons from one ALREADY-COMPUTED flat feature vector.

    feature_vector: dict or pd.Series of {feature_name: value}. The same vector
    is scored by both window models (each reindexes to its own selected_features).
    Prefer predict_dual_horizon_from_events when you have raw events, since the
    two horizons should really see different lookback windows.
    """
    def _run(w):
        return w, predict(feature_vector, w)

    with ThreadPoolExecutor(max_workers=len(WINDOWS)) as pool:
        results = dict(pool.map(_run, WINDOWS))
    return _assemble(results)


def main():
    p = argparse.ArgumentParser(description="Dual-horizon (1mo + 12mo) prediction for one patient")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--events-json", help="JSON list of raw event rows (SQL schema) — runs per-window FE then predicts")
    g.add_argument("--events-csv",  help="CSV of raw event rows for one patient — runs per-window FE then predicts")
    g.add_argument("--features-json", help="JSON dict of FLAT {feature_name: value} (scored by both windows)")
    g.add_argument("--features-csv",  help="One-row CSV with FLAT feature columns (scored by both windows)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.events_json or args.events_csv:
        events = (json.loads(args.events_json) if args.events_json
                  else pd.read_csv(args.events_csv).to_dict("records"))
        result = predict_dual_horizon_from_events(events)
    elif args.features_csv:
        df = pd.read_csv(args.features_csv)
        if len(df) != 1:
            raise ValueError(f"Expected 1 row, got {len(df)}")
        result = predict_dual_horizon(df.iloc[0])
    else:
        result = predict_dual_horizon(json.loads(args.features_json))

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
