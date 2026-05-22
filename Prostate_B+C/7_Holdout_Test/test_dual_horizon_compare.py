"""
Compare two dual-horizon setups on the shared test split:
  Setup A = (12mo, 1mo_5y)       — current production baseline
  Setup B = (12mo, 1mo_12mo)     — truncated-lookback experiment

Reads shared_split.json (the unified train/val/test assignment produced by
_shared_split.py) and runs both setups on the SAME test patients. The 12mo
model is shared between the two setups, so the comparison isolates the
effect of the 1mo lookback choice.

Outputs:
  - test_compare_per_patient.csv    — one row per test patient with both
                                       setups' scores, decisions, verdicts,
                                       and dual-horizon results.
  - Console: 2× per-setup metric blocks (AUC/Sens/Spec/CM) + a side-by-side
    summary saying which 1mo variant wins.

Usage:
    python3 test_dual_horizon_compare.py
    python3 test_dual_horizon_compare.py --out custom.csv

Pre-requisites (must be done in order):
  1. SQL extracts ran for all 3 windows (12mo, 1mo_5y, 1mo_12mo)
  2. FE pipeline produced cleanup parquets for all 3
  3. _shared_split.py ran → shared_split.json exists
  4. 1_training.py ran for all 3 windows → saved_models/ for each
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent
TRAIN_RESULTS = PROJECT_ROOT / "3_Modeling" / "results" / "1_training"
FE_CLEANUP    = PROJECT_ROOT / "2_Feature_Engineering" / "results" / "5_cleanup"
SHARED_SPLIT  = PROJECT_ROOT / "3_Modeling" / "shared_split.json"
DEFAULT_OUT   = SCRIPT_DIR / "test_compare_per_patient.csv"

sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
from _calibrator import CalibratorWrapper  # noqa: F401

SETUP_A = ("12mo", "1mo_5y")     # baseline
SETUP_B = ("12mo", "1mo_12mo")   # experiment


# ──────────────────────────────────────────────────────────────────────────
# Model artifact loading + single-patient scoring
# ──────────────────────────────────────────────────────────────────────────

def _load_artifacts(window):
    model_dir = TRAIN_RESULTS / window / "saved_models"
    cfg = joblib.load(model_dir / "config.json")
    seeds = cfg.get("seeds", [42])
    seed_artifacts = {}
    for seed in seeds:
        sd = model_dir / f"seed_{seed}"
        models = {n: joblib.load(sd / f"{n}_model.pkl") for n in cfg["ensemble_models"]}
        meta = joblib.load(sd / "meta_learner.pkl")
        seed_artifacts[seed] = {"models": models, "meta": meta}
    return cfg, seed_artifacts


def _score_batch(X, cfg, seed_artifacts):
    """Run base models + stacker + per-band calibrator on a DataFrame of patients.
    Returns DataFrame with risk_proba and decision per row (index = X.index).
    """
    ensemble = cfg["ensemble_models"]
    per_seed = []
    for seed, art in seed_artifacts.items():
        # Get base-model probabilities for each row × each model
        base_preds = np.column_stack([
            art["models"][n].predict_proba(X)[:, 1] for n in ensemble
        ])
        stacked = art["meta"].predict_proba(base_preds)[:, 1]
        per_seed.append(stacked)
    raw_proba = np.mean(np.column_stack(per_seed), axis=1)

    # Per-band calibration + threshold
    band_bins   = cfg["band_bins"]
    band_labels = cfg["band_labels"]
    ages = X.get("AGE_AT_INDEX", pd.Series(65.0, index=X.index)).fillna(65.0).to_numpy()
    band_idx = np.clip(np.searchsorted(band_bins[1:], ages, side="right"), 0, len(band_labels) - 1)
    bands = np.array([band_labels[i] for i in band_idx])

    cals  = cfg["calibrators_per_band"]
    thrs  = cfg["thresholds_per_band"]
    g_cal = cfg["global_calibrator"]
    g_thr = cfg["global_threshold"]

    risk = np.zeros(len(X))
    decision = np.zeros(len(X), dtype=int)
    for i, b in enumerate(bands):
        cal = cals.get(b, g_cal)
        thr = thrs.get(b, g_thr)
        rp = float(cal.transform(np.array([raw_proba[i]]))[0])
        risk[i] = rp
        decision[i] = int(rp >= thr)
    return pd.DataFrame({"risk_proba": risk, "decision": decision, "age_band": bands}, index=X.index)


def _verdict(decision, proba):
    if decision == 1:  return "Cancer"
    if proba == 0.0:   return "Not enough data"
    return "No Cancer"


def _classify(d_b, p_b, d_c, p_c):
    fired = []
    if d_c == 1: fired.append("1mo")
    if d_b == 1: fired.append("12mo")
    if fired:
        return "Cancer", fired
    if p_b == 0.0 and p_c == 0.0:
        return "Not enough data", []
    return "No Cancer", []


def _metrics_block(name, y_true, decisions, probas):
    y_true = np.asarray(y_true).astype(int)
    decisions = np.asarray(decisions).astype(int)
    probas = np.asarray(probas).astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, decisions, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = roc_auc_score(y_true, probas)
    except ValueError:
        auc = float("nan")
    print(f"  {name}")
    print(f"    AUC = {auc:.4f}   Sens = {sens*100:6.2f}%   Spec = {spec*100:6.2f}%")
    print(f"    CM:  TP={tp:>5}  FP={fp:>5}  FN={fn:>5}  TN={tn:>5}")
    print()
    return dict(auc=auc, sens=sens, spec=spec, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not SHARED_SPLIT.exists():
        raise FileNotFoundError(
            f"{SHARED_SPLIT} missing. Run 3_Modeling/_shared_split.py first."
        )
    split = json.load(open(SHARED_SPLIT))
    test_guids = split["test_guids"]
    logger.info(f"Shared test split: {len(test_guids):,} patients")

    # Load model artifacts for all 3 windows
    logger.info("Loading model artifacts...")
    cfg_12,  art_12  = _load_artifacts("12mo")
    cfg_5y,  art_5y  = _load_artifacts("1mo_5y")
    cfg_12m, art_12m = _load_artifacts("1mo_12mo")

    # Load feature matrices for all 3 windows, filtered to test patients
    logger.info("Loading + scoring per-window feature matrices on test set...")
    def _load_test(window, cfg):
        fm = pd.read_parquet(FE_CLEANUP / window / f"feature_matrix_clean_{window}.parquet")
        if "PATIENT_GUID" not in fm.columns:
            fm = fm.reset_index()
        fm["PATIENT_GUID"] = fm["PATIENT_GUID"].astype(str)
        fm = fm[fm["PATIENT_GUID"].isin(test_guids)].set_index("PATIENT_GUID")
        fm = fm.reindex(test_guids).dropna(how="all")
        # Align features to model's selected columns
        sel = cfg["selected_features"]
        X = fm.reindex(columns=sel, fill_value=0).fillna(0)
        y = fm["LABEL"] if "LABEL" in fm.columns else None
        return X, y

    X_12,  y_true = _load_test("12mo", cfg_12)
    X_5y,  _ = _load_test("1mo_5y", cfg_5y)
    X_12m, _ = _load_test("1mo_12mo", cfg_12m)

    # Score per window
    s_12  = _score_batch(X_12,  cfg_12,  art_12)
    s_5y  = _score_batch(X_5y,  cfg_5y,  art_5y)
    s_12m = _score_batch(X_12m, cfg_12m, art_12m)

    # Insufficient_data flag — derived from features summing to 0
    def _insuf(X, cfg):
        clinical_cols = [c for c in cfg["selected_features"] if c not in ("AGE_AT_INDEX", "AGE_BAND")]
        return (X[clinical_cols].abs().sum(axis=1) == 0).astype(int)

    s_12 ["insufficient_data"] = _insuf(X_12,  cfg_12)
    s_5y ["insufficient_data"] = _insuf(X_5y,  cfg_5y)
    s_12m["insufficient_data"] = _insuf(X_12m, cfg_12m)

    # Apply insufficient_data force-zero
    for s in (s_12, s_5y, s_12m):
        mask = s["insufficient_data"] == 1
        s.loc[mask, "risk_proba"] = 0.0
        s.loc[mask, "decision"]   = 0

    # ── Build per-patient comparison rows ──
    rows = []
    for pg in y_true.index:
        a_b_p, a_b_d = float(s_12.loc[pg, "risk_proba"]),  int(s_12.loc[pg, "decision"])
        # Setup A — uses 1mo_5y
        a_c_p, a_c_d = float(s_5y.loc[pg, "risk_proba"]),  int(s_5y.loc[pg, "decision"])
        result_a, fired_a = _classify(a_b_d, a_b_p, a_c_d, a_c_p)
        # Setup B — uses 1mo_12mo
        b_c_p, b_c_d = float(s_12m.loc[pg, "risk_proba"]), int(s_12m.loc[pg, "decision"])
        result_b, fired_b = _classify(a_b_d, a_b_p, b_c_d, b_c_p)
        rows.append({
            "patient_guid": pg,
            "y_true": int(y_true[pg]),
            "y_true_label": "Cancer" if y_true[pg] == 1 else "No Cancer",
            "age_band": s_12.loc[pg, "age_band"],
            # 12mo (shared between setups)
            "score_12mo": a_b_p, "decision_12mo": a_b_d, "verdict_12mo": _verdict(a_b_d, a_b_p),
            # 1mo variants
            "score_1mo_5y":   a_c_p, "decision_1mo_5y":   a_c_d, "verdict_1mo_5y":   _verdict(a_c_d, a_c_p),
            "score_1mo_12mo": b_c_p, "decision_1mo_12mo": b_c_d, "verdict_1mo_12mo": _verdict(b_c_d, b_c_p),
            # Dual-horizon outcomes
            "setupA_result":   result_a, "setupA_fired_by": "|".join(fired_a) or "",
            "setupB_result":   result_b, "setupB_fired_by": "|".join(fired_b) or "",
            "setupA_max_score": max(a_b_p, a_c_p),
            "setupB_max_score": max(a_b_p, b_c_p),
        })
    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # ── Summary ──
    y = out["y_true"].to_numpy()
    print()
    print("═" * 78)
    print(f"  DUAL-HORIZON COMPARISON — {len(out):,} test patients (shared split)")
    print(f"  cancer: {(y==1).sum():,}   non-cancer: {(y==0).sum():,}")
    print("═" * 78)
    print()
    a_dec = (out["setupA_result"] == "Cancer").astype(int)
    b_dec = (out["setupB_result"] == "Cancer").astype(int)
    a = _metrics_block("Setup A = 12mo + 1mo_5y  (baseline)",   y, a_dec, out["setupA_max_score"])
    b = _metrics_block("Setup B = 12mo + 1mo_12mo (experiment)", y, b_dec, out["setupB_max_score"])

    # Verdict — which variant wins?
    print("─" * 78)
    print(f"  Δ (B − A): AUC {b['auc']-a['auc']:+.4f}  Sens {(b['sens']-a['sens'])*100:+.2f}pp  Spec {(b['spec']-a['spec'])*100:+.2f}pp")
    winner = "1mo_12mo (truncated lookback)" if b["auc"] > a["auc"] else "1mo_5y (full lookback)"
    print(f"  → Winning 1mo variant by AUC: {winner}")
    print()
    print(f"  Per-patient CSV: {out_path}")


if __name__ == "__main__":
    main()
