"""
Combined evaluation — Lymphoma-style.

Merges the internal test-set predictions (contains cancer cases at training
prevalence) with the unseen 300K-derived holdout predictions (all non-cancer)
and reports:
  - Run A: Non-cancer only (specificity / FP rate at realistic prevalence)
  - Run B: Combined (AUC + sens/spec at realistic ~0.1-0.2% prevalence)

At two operating points:
  - Tier-0 threshold (primary, from saved v3 config.json)
  - Tier-90 threshold (high-sensitivity alternative)
  - Plus a fresh ROC-derived threshold on the combined set

Inputs:
  ../3_Modeling/results/1_training/{w}/predictions_{w}.csv    (internal test)
  ./results/predictions_unseen_{w}.csv                         (holdout, from 2_predict_unseen.py)
  ../3_Modeling/results/1_training/{w}/saved_models/config.json (thresholds)

Outputs (per window, in ./results/):
  combined_eval_{w}.csv          — tabular metrics, one row per tier per run
  combined_eval_{w}.txt          — human-readable summary
  combined_eval_{w}.log          — detailed log with confusion matrices
  combined_eval_{w}_predictions.csv — merged prediction rows (for downstream analysis)
"""

import argparse
import json
import joblib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
TRAIN_RESULTS = BASE / "3_Modeling" / "results" / "1_training"
HOLDOUT_RESULTS = SCRIPT_DIR / "results"


def _score_col(df):
    """Pick the probability column (prefers calibrated)."""
    for c in ("y_pred_proba", "y_pred_proba_cal", "y_pred_proba_raw", "y_pred"):
        if c in df.columns:
            return c
    raise ValueError(f"No score column found. Columns: {list(df.columns)}")


def evaluate_at_threshold(y_true, y_score, threshold, label):
    """Compute sens/spec/AUC at a given threshold."""
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
        "label": label,
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "sensitivity": float(sens) if not np.isnan(sens) else None,
        "specificity": float(spec) if not np.isnan(spec) else None,
        "auc": float(auc) if not np.isnan(auc) else None,
        "n_total": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }


def find_operating_point(y_true, y_score, min_sens=None, min_spec=None):
    """Pick the threshold that maximises (sens+spec) subject to minimum constraints.
    Returns (threshold, sens, spec) or (None, nan, nan) if unreachable."""
    if len(np.unique(y_true)) < 2:
        return None, np.nan, np.nan
    fpr, tpr, thr = roc_curve(y_true, y_score)
    sens = tpr
    spec = 1 - fpr
    mask = np.ones_like(sens, dtype=bool)
    if min_sens is not None:
        mask &= (sens >= min_sens)
    if min_spec is not None:
        mask &= (spec >= min_spec)
    if mask.sum() == 0:
        return None, np.nan, np.nan
    scores = sens + spec
    scores = np.where(mask, scores, -np.inf)
    i = int(np.argmax(scores))
    return float(thr[i]), float(sens[i]), float(spec[i])


def main():
    parser = argparse.ArgumentParser(description="Combined holdout + internal-test evaluation")
    parser.add_argument("--window", required=True, choices=["3mo", "6mo", "12mo"])
    args = parser.parse_args()
    w = args.window

    HOLDOUT_RESULTS.mkdir(parents=True, exist_ok=True)

    # Logging (tee to file + stdout)
    log_path = HOLDOUT_RESULTS / f"combined_eval_{w}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info(f"  PROSTATE — COMBINED EVALUATION — {w}")
    log.info("=" * 70)

    # Paths
    train_pred_path   = TRAIN_RESULTS / w / f"predictions_{w}.csv"
    unseen_pred_path  = HOLDOUT_RESULTS / f"predictions_unseen_{w}.csv"
    config_path       = TRAIN_RESULTS / w / "saved_models" / "config.json"

    for p in (train_pred_path, unseen_pred_path, config_path):
        if not p.exists():
            log.error(f"MISSING: {p}")
            sys.exit(1)

    # Load
    log.info(f"\n  Train test predictions:  {train_pred_path}")
    train_df = pd.read_csv(train_pred_path)
    log.info(f"    {len(train_df):,} rows | cancer: {(train_df['y_true']==1).sum():,} | non-cancer: {(train_df['y_true']==0).sum():,}")

    log.info(f"\n  Unseen holdout predictions: {unseen_pred_path}")
    unseen_df = pd.read_csv(unseen_pred_path)
    if "y_true" not in unseen_df.columns:
        unseen_df["y_true"] = 0  # holdout is all non-cancer
    log.info(f"    {len(unseen_df):,} rows | cancer: {(unseen_df['y_true']==1).sum():,} | non-cancer: {(unseen_df['y_true']==0).sum():,}")

    # Thresholds from v3 saved config
    try:
        cfg = joblib.load(config_path)
    except Exception:
        with open(config_path) as f:
            cfg = json.load(f)
    tier0_thr = float(cfg.get("threshold"))
    tier90_thr = cfg.get("tier90_threshold")
    tier90_thr = float(tier90_thr) if tier90_thr is not None and not (isinstance(tier90_thr, float) and np.isnan(tier90_thr)) else None
    log.info(f"\n  Tier-0 threshold:   {tier0_thr:.4f}")
    log.info(f"  Tier-90 threshold:  {'n/a' if tier90_thr is None else f'{tier90_thr:.4f}'}")

    # Score column
    train_score = _score_col(train_df)
    unseen_score = _score_col(unseen_df)
    log.info(f"\n  Train score col:  {train_score}")
    log.info(f"  Unseen score col: {unseen_score}")

    # --- Run A: Non-cancer only ---
    log.info("\n" + "=" * 70)
    log.info(f"  RUN A: NON-CANCER ONLY ({len(unseen_df):,} patients)")
    log.info("=" * 70)
    yA, sA = unseen_df["y_true"].values, unseen_df[unseen_score].values
    rA_tier0  = evaluate_at_threshold(yA, sA, tier0_thr, f"RunA_Tier0_thr={tier0_thr:.4f}")
    log.info(f"\n  Tier-0 ({tier0_thr:.4f}):")
    log.info(f"    Confusion:  TP={rA_tier0['tp']} FP={rA_tier0['fp']} TN={rA_tier0['tn']} FN={rA_tier0['fn']}")
    log.info(f"    Specificity: {rA_tier0['specificity']:.4f}  (sens undefined — no positives)")
    rA_tier90 = None
    if tier90_thr is not None:
        rA_tier90 = evaluate_at_threshold(yA, sA, tier90_thr, f"RunA_Tier90_thr={tier90_thr:.4f}")
        log.info(f"\n  Tier-90 ({tier90_thr:.4f}):")
        log.info(f"    Confusion:  TP={rA_tier90['tp']} FP={rA_tier90['fp']} TN={rA_tier90['tn']} FN={rA_tier90['fn']}")
        log.info(f"    Specificity: {rA_tier90['specificity']:.4f}")

    # --- Run B: Combined ---
    log.info("\n" + "=" * 70)
    combined = pd.concat(
        [train_df[["y_true", train_score]].rename(columns={train_score: "score"}),
         unseen_df[["y_true", unseen_score]].rename(columns={unseen_score: "score"})],
        ignore_index=True,
    )
    n_combined = len(combined)
    n_pos = int((combined["y_true"] == 1).sum())
    n_neg = n_combined - n_pos
    log.info(f"  RUN B: COMBINED ({n_combined:,} patients | cancer: {n_pos:,} | non-cancer: {n_neg:,} | prevalence: {n_pos/n_combined*100:.3f}%)")
    log.info("=" * 70)
    yB, sB = combined["y_true"].values, combined["score"].values
    rB_tier0  = evaluate_at_threshold(yB, sB, tier0_thr, f"RunB_Tier0_thr={tier0_thr:.4f}")
    log.info(f"\n  Tier-0 ({tier0_thr:.4f}):")
    log.info(f"    Confusion:  TP={rB_tier0['tp']} FP={rB_tier0['fp']} TN={rB_tier0['tn']} FN={rB_tier0['fn']}")
    log.info(f"    Sensitivity: {rB_tier0['sensitivity']:.4f}")
    log.info(f"    Specificity: {rB_tier0['specificity']:.4f}")
    log.info(f"    AUC:         {rB_tier0['auc']:.4f}")
    rB_tier90 = None
    if tier90_thr is not None:
        rB_tier90 = evaluate_at_threshold(yB, sB, tier90_thr, f"RunB_Tier90_thr={tier90_thr:.4f}")
        log.info(f"\n  Tier-90 ({tier90_thr:.4f}):")
        log.info(f"    Confusion:  TP={rB_tier90['tp']} FP={rB_tier90['fp']} TN={rB_tier90['tn']} FN={rB_tier90['fn']}")
        log.info(f"    Sensitivity: {rB_tier90['sensitivity']:.4f}")
        log.info(f"    Specificity: {rB_tier90['specificity']:.4f}")

    # Fresh operating point on combined
    log.info(f"\n  Fresh operating point on combined (max sens+spec, no floor):")
    fresh_thr, fresh_sens, fresh_spec = find_operating_point(yB, sB)
    if fresh_thr is not None:
        rB_fresh = evaluate_at_threshold(yB, sB, fresh_thr, f"RunB_Fresh_thr={fresh_thr:.4f}")
        log.info(f"    Threshold:   {fresh_thr:.4f}")
        log.info(f"    Sensitivity: {fresh_sens:.4f}")
        log.info(f"    Specificity: {fresh_spec:.4f}")
    else:
        rB_fresh = None

    # Re-optimised Tier-0 and Tier-90 on combined
    log.info(f"\n  Re-optimised thresholds on combined:")
    rethr_t0 = find_operating_point(yB, sB, min_sens=0.80, min_spec=0.70)
    if rethr_t0[0] is not None:
        log.info(f"    Tier-0 (sens>=0.80, spec>=0.70): thr={rethr_t0[0]:.4f} sens={rethr_t0[1]:.4f} spec={rethr_t0[2]:.4f}")
    else:
        log.info(f"    Tier-0 target (sens>=0.80, spec>=0.70): NOT ACHIEVABLE on combined")
    rethr_t90 = find_operating_point(yB, sB, min_sens=0.90)
    if rethr_t90[0] is not None:
        log.info(f"    Tier-90 (sens>=0.90, max spec):   thr={rethr_t90[0]:.4f} sens={rethr_t90[1]:.4f} spec={rethr_t90[2]:.4f}")
    else:
        log.info(f"    Tier-90 target (sens>=0.90): NOT ACHIEVABLE on combined")

    # Write tabular CSV
    rows = [rA_tier0]
    if rA_tier90 is not None: rows.append(rA_tier90)
    rows.append(rB_tier0)
    if rB_tier90 is not None: rows.append(rB_tier90)
    if rB_fresh  is not None: rows.append(rB_fresh)
    if rethr_t0[0] is not None:
        rows.append({
            "label": "RunB_Tier0_reoptimised", "threshold": rethr_t0[0],
            "sensitivity": rethr_t0[1], "specificity": rethr_t0[2],
            "n_total": n_combined, "n_positive": n_pos, "n_negative": n_neg,
        })
    if rethr_t90[0] is not None:
        rows.append({
            "label": "RunB_Tier90_reoptimised", "threshold": rethr_t90[0],
            "sensitivity": rethr_t90[1], "specificity": rethr_t90[2],
            "n_total": n_combined, "n_positive": n_pos, "n_negative": n_neg,
        })

    out_csv = HOLDOUT_RESULTS / f"combined_eval_{w}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info(f"\n  Saved: {out_csv}")

    # Write merged predictions for downstream use
    combined_with_src = pd.concat(
        [train_df.assign(source="internal_test"),
         unseen_df.assign(source="holdout")],
        ignore_index=True,
    )
    combined_preds_path = HOLDOUT_RESULTS / f"combined_eval_{w}_predictions.csv"
    combined_with_src.to_csv(combined_preds_path, index=False)
    log.info(f"  Saved: {combined_preds_path}")

    # Write text summary
    out_txt = HOLDOUT_RESULTS / f"combined_eval_{w}.txt"
    with open(out_txt, "w") as f:
        f.write(f"Prostate combined evaluation — window={w}\n")
        f.write("=" * 70 + "\n")
        f.write(f"RunA (non-cancer only): {len(unseen_df):,} patients\n")
        f.write(f"  Tier-0 ({tier0_thr:.4f}): spec={rA_tier0['specificity']:.4f}\n")
        if rA_tier90:
            f.write(f"  Tier-90 ({tier90_thr:.4f}): spec={rA_tier90['specificity']:.4f}\n")
        f.write("\n")
        f.write(f"RunB (combined): {n_combined:,} patients ({n_pos:,} cancer, prevalence {n_pos/n_combined*100:.3f}%)\n")
        f.write(f"  Tier-0 ({tier0_thr:.4f}): AUC={rB_tier0['auc']:.4f} sens={rB_tier0['sensitivity']:.4f} spec={rB_tier0['specificity']:.4f}\n")
        if rB_tier90:
            f.write(f"  Tier-90 ({tier90_thr:.4f}): sens={rB_tier90['sensitivity']:.4f} spec={rB_tier90['specificity']:.4f}\n")
    log.info(f"  Saved: {out_txt}")

    log.info("\nDONE")


if __name__ == "__main__":
    main()
