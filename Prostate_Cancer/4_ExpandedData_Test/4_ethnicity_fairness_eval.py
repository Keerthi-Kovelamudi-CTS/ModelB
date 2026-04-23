"""
Ethnicity subgroup fairness evaluation.

Slices Framing A metrics (AUC, sens, spec) by ETHNICITY_GROUP so we can tell:
  - does the model perform uniformly across ethnicities?
  - is ETH_BLACK driving sensitivity up for Black patients specifically, or
    just globally?
  - are there groups where performance is materially worse (fairness concern)?

Inputs per window:
  ../3_Modeling/results/1_training/{w}/predictions_{w}.csv   (internal test)
  ./results/predictions_unseen_{w}.csv                       (holdout, patched)
  ../2_Feature_Engineering_Ethnicity/data/{w}/prostate_{w}_obs.csv
      (used to recover ethnicity per PATIENT_GUID; one value per patient)

Output per window (in ./results/):
  fairness_eval_{w}.csv    — one row per ethnicity group
  fairness_eval_{w}.txt    — readable summary

Usage:
  python 4_ethnicity_fairness_eval.py --window 12mo
  python 4_ethnicity_fairness_eval.py --window all
"""
import argparse
import json
import joblib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
TRAIN_RESULTS = BASE / "3_Modeling" / "results" / "1_training"
HOLDOUT_RESULTS = SCRIPT_DIR / "results"
FE_ETH_DATA = BASE / "2_Feature_Engineering_Ethnicity" / "data"


def _load_ethnicity_map(window: str) -> pd.Series:
    """PATIENT_GUID → ETHNICITY_GROUP (one value per patient)."""
    obs_path = FE_ETH_DATA / window / f"prostate_{window}_obs.csv"
    if not obs_path.exists():
        raise FileNotFoundError(f"Ethnicity source missing: {obs_path}")
    df = pd.read_csv(obs_path, usecols=["patient_guid", "ethnicity_group"],
                     low_memory=False)
    df.columns = df.columns.str.upper()
    df["PATIENT_GUID"] = df["PATIENT_GUID"].astype(str).str.strip("{}")
    eth = (df.drop_duplicates("PATIENT_GUID")
             .set_index("PATIENT_GUID")["ETHNICITY_GROUP"]
             .astype(str).str.strip())
    known = {"White", "Asian", "Black", "Chinese", "Mixed", "Other", "Not specified"}
    eth = eth.where(eth.isin(known), "Not specified")
    return eth


def _evaluate_group(y_true, y_score, threshold, group_name):
    """sens/spec/AUC for one subgroup."""
    n = len(y_true)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n == 0:
        return None
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan
    return {
        "group": group_name,
        "n_total": n, "n_positive": n_pos, "n_negative": n_neg,
        "prevalence_pct": round(100 * n_pos / n, 3) if n else 0.0,
        "auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "sensitivity": round(float(sens), 4) if not np.isnan(sens) else None,
        "specificity": round(float(spec), 4) if not np.isnan(spec) else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def evaluate_window(window: str) -> None:
    print(f"\n{'='*70}\n  FAIRNESS EVAL — {window}\n{'='*70}")

    train_pred = TRAIN_RESULTS / window / f"predictions_{window}.csv"
    unseen_pred = HOLDOUT_RESULTS / f"predictions_unseen_{window}.csv"
    config_path = TRAIN_RESULTS / window / "saved_models" / "config.json"

    for p in (train_pred, unseen_pred, config_path):
        if not p.exists():
            print(f"  MISSING: {p} — skip {window}")
            return

    # Threshold
    try:
        cfg = joblib.load(config_path)
    except Exception:
        with open(config_path) as f:
            cfg = json.load(f)
    threshold = float(cfg.get("threshold"))
    print(f"  Threshold (Tier-0 from config): {threshold:.4f}")

    # Combined Framing A (internal test + holdout)
    internal = pd.read_csv(train_pred)
    holdout = pd.read_csv(unseen_pred)
    if "y_true" not in holdout.columns:
        holdout["y_true"] = 0

    score_col = next((c for c in ("y_pred_proba", "y_pred_proba_cal")
                      if c in internal.columns), None)
    if score_col is None:
        raise RuntimeError(f"No score column in {train_pred}")
    combined = pd.concat(
        [internal[["PATIENT_GUID", "y_true", score_col]].assign(source="internal_test"),
         holdout[["PATIENT_GUID", "y_true", score_col]].assign(source="holdout")],
        ignore_index=True,
    )
    combined["PATIENT_GUID"] = combined["PATIENT_GUID"].astype(str).str.strip("{}")
    print(f"  Framing A pool: {len(combined):,} | cancer: {(combined['y_true']==1).sum():,}")

    # Ethnicity join
    eth_map = _load_ethnicity_map(window)
    combined["ethnicity"] = combined["PATIENT_GUID"].map(eth_map).fillna("Unmatched")
    matched_rate = (combined["ethnicity"] != "Unmatched").mean()
    print(f"  Ethnicity coverage: {matched_rate*100:.1f}%")

    rows = []
    rows.append(_evaluate_group(combined["y_true"].values,
                                 combined[score_col].values,
                                 threshold, "OVERALL"))
    for g in ["White", "Asian", "Black", "Chinese", "Mixed", "Other",
              "Not specified", "Unmatched"]:
        sub = combined[combined["ethnicity"] == g]
        r = _evaluate_group(sub["y_true"].values, sub[score_col].values,
                             threshold, g)
        if r is not None:
            rows.append(r)

    out_df = pd.DataFrame(rows)
    out_csv = HOLDOUT_RESULTS / f"fairness_eval_{window}.csv"
    out_df.to_csv(out_csv, index=False)

    # Text summary
    lines = [f"Prostate ethnicity fairness — window={window}",
             "=" * 70,
             f"Tier-0 threshold: {threshold:.4f}", ""]
    lines.append(f"{'Group':<18} {'N':>8} {'Cancer':>7} {'Prev%':>6} "
                 f"{'AUC':>6} {'Sens':>6} {'Spec':>6}")
    lines.append("-" * 70)
    for r in rows:
        auc = f"{r['auc']:.3f}" if r['auc'] is not None else "n/a"
        sens = f"{r['sensitivity']:.3f}" if r['sensitivity'] is not None else "n/a"
        spec = f"{r['specificity']:.3f}" if r['specificity'] is not None else "n/a"
        lines.append(f"{r['group']:<18} {r['n_total']:>8,} {r['n_positive']:>7,} "
                     f"{r['prevalence_pct']:>6.2f} {auc:>6} {sens:>6} {spec:>6}")
    out_txt = HOLDOUT_RESULTS / f"fairness_eval_{window}.txt"
    out_txt.write_text("\n".join(lines))

    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_txt}")
    print()
    print("\n".join(lines[3:]))  # print the table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["3mo", "6mo", "12mo", "all"],
                    default="all")
    args = ap.parse_args()
    windows = ["3mo", "6mo", "12mo"] if args.window == "all" else [args.window]
    HOLDOUT_RESULTS.mkdir(parents=True, exist_ok=True)
    for w in windows:
        evaluate_window(w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
