"""
Full diagnostic suite for the dual-horizon test results.

Runs 5 analyses on the 844-patient test set:
  1. Missed cancers (FN) — what patterns do both models fail on?
  2. False positives (FP) — what patients get over-flagged?
  3. Dual-horizon score threshold tuning — sweep operating points.
  4. Age-band breakdown — Sens/Spec/AUC per age group.
  5. SHAP top-factors for cross-test cells (B + C) — adds top_factors_B and
     top_factors_C to the CSV (current CSV only has them for production
     cells A + D).

Inputs:
  - 4_Inference/test_dual_horizon_per_patient.csv  (must exist; produced by test_dual_horizon_local.py)
  - 3_Modeling/results/1_training/{1mo,12mo}/saved_models/   (for SHAP on B + C)
  - 2_Feature_Engineering/results/5_cleanup/{1mo,12mo}/feature_matrix_clean_*.parquet

Outputs:
  - 4_Inference/test_dual_horizon_per_patient_full.csv  (CSV + 2 new SHAP columns)
  - 4_Inference/dual_horizon_threshold_sweep.csv        (operating-point table)
  - 4_Inference/dual_horizon_age_band_metrics.csv       (per-band AUC/Sens/Spec)
  - Console: rich summary of all 5 analyses

Usage:
    python3 analyze_dual_horizon_results.py
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
INPUT_CSV     = SCRIPT_DIR / "test_dual_horizon_per_patient.csv"
OUT_FULL_CSV  = SCRIPT_DIR / "test_dual_horizon_per_patient_full.csv"
OUT_THR_CSV   = SCRIPT_DIR / "dual_horizon_threshold_sweep.csv"
OUT_BAND_CSV  = SCRIPT_DIR / "dual_horizon_age_band_metrics.csv"

sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
from _calibrator import CalibratorWrapper  # noqa: F401

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _print_section(title):
    print()
    print("═" * 78)
    print(f"  {title}")
    print("═" * 78)


def _metrics(y_true, decisions, probas):
    tn, fp, fn, tp = confusion_matrix(y_true, decisions, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = roc_auc_score(y_true, probas)
    except ValueError:
        auc = float("nan")
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), sens=sens, spec=spec, auc=auc)


# ─────────────────────────────────────────────────────────────────────────
# Analysis 1 + 2 — FN + FP patterns
# ─────────────────────────────────────────────────────────────────────────

def analyze_errors(df):
    _print_section("ANALYSIS 1+2 — Missed cancers (FN) and false positives (FP)")

    # FN: y_true=1, dual_horizon_result != Cancer
    fn = df[(df["y_true"] == 1) & (df["dual_horizon_result"] != "Cancer")].copy()
    # FP: y_true=0, dual_horizon_result == Cancer
    fp = df[(df["y_true"] == 0) & (df["dual_horizon_result"] == "Cancer")].copy()

    print(f"  Missed cancers (FN): {len(fn)}")
    print(f"  False positives (FP): {len(fp)}")
    print()

    print("  ── Missed cancers — age-band distribution ──")
    print(fn["age_band"].value_counts().to_string())
    print()
    print("  ── Missed cancers — both windows scored LOW (model uncertainty) ──")
    if len(fn) > 0:
        for _, r in fn.iterrows():
            print(f"    {r.patient_guid[:20]}…  band={r.age_band:<6}  "
                  f"1mo={r.D_score_1mo_on_1mo_data:.3f}({r.D_verdict})  "
                  f"12mo={r.A_score_12mo_on_12mo_data:.3f}({r.A_verdict})")
    print()
    print("  ── False positives — age-band distribution ──")
    print(fp["age_band"].value_counts().to_string())
    print()
    print("  ── False positives — fired_by breakdown (which window over-flagged?) ──")
    print(fp["fired_by"].value_counts().to_string())
    print()
    if len(fp) > 0:
        print("  ── Sample false positives (first 5) ──")
        for _, r in fp.head(5).iterrows():
            print(f"    {r.patient_guid[:20]}…  band={r.age_band:<6}  fired_by={r.fired_by:<10}  "
                  f"1mo={r.D_score_1mo_on_1mo_data:.3f}  12mo={r.A_score_12mo_on_12mo_data:.3f}")


# ─────────────────────────────────────────────────────────────────────────
# Analysis 3 — Threshold sweep on dual-horizon score
# ─────────────────────────────────────────────────────────────────────────

def threshold_sweep(df):
    _print_section("ANALYSIS 3 — Threshold sweep on max_production_score")

    y_true = df["y_true"].to_numpy()
    score = df["max_production_score"].to_numpy()
    thrs = np.linspace(0.05, 0.95, 19)

    rows = []
    for thr in thrs:
        y_pred = (score >= thr).astype(int)
        m = _metrics(y_true, y_pred, score)
        rows.append({"threshold": round(thr, 2), **m})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT_THR_CSV, index=False)

    # Show key operating points
    print(f"  {'thr':>6}  {'AUC':>7}  {'Sens':>8}  {'Spec':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")
    for _, r in sweep.iterrows():
        print(f"  {r.threshold:>6.2f}  {r.auc:>7.4f}  {r.sens*100:>7.2f}%  {r.spec*100:>7.2f}%  "
              f"{r.tp:>4}  {r.fp:>4}  {r.fn:>4}  {r.tn:>4}")

    print()
    print(f"  Saved threshold sweep → {OUT_THR_CSV.name}")

    # Find best Sens@≥90%, best Spec@≥90%, max Youden's J
    yj = sweep["sens"] + sweep["spec"] - 1
    best_yj = sweep.iloc[yj.idxmax()]
    sens90 = sweep[sweep["sens"] >= 0.90].sort_values("spec", ascending=False).head(1)
    spec90 = sweep[sweep["spec"] >= 0.90].sort_values("sens", ascending=False).head(1)
    print()
    print(f"  ┌─ Best operating points ─┐")
    print(f"  │ Max Youden's J:  thr={best_yj.threshold:.2f}  Sens={best_yj.sens*100:.1f}%  Spec={best_yj.spec*100:.1f}%")
    if len(sens90) > 0:
        r = sens90.iloc[0]
        print(f"  │ Sens≥90% maxSpec: thr={r.threshold:.2f}  Sens={r.sens*100:.1f}%  Spec={r.spec*100:.1f}%")
    if len(spec90) > 0:
        r = spec90.iloc[0]
        print(f"  │ Spec≥90% maxSens: thr={r.threshold:.2f}  Sens={r.sens*100:.1f}%  Spec={r.spec*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────
# Analysis 4 — Age-band breakdown
# ─────────────────────────────────────────────────────────────────────────

def age_band_breakdown(df):
    _print_section("ANALYSIS 4 — Performance by age band")
    rows = []
    for band, g in df.groupby("age_band"):
        y_true = g["y_true"].to_numpy()
        y_dual = (g["dual_horizon_result"] == "Cancer").astype(int).to_numpy()
        score = g["max_production_score"].to_numpy()
        m = _metrics(y_true, y_dual, score)
        rows.append({
            "age_band": band,
            "n_patients": int(len(g)),
            "n_cancer": int((y_true == 1).sum()),
            "n_non_cancer": int((y_true == 0).sum()),
            **m,
        })
    band_df = pd.DataFrame(rows).sort_values("age_band")
    band_df.to_csv(OUT_BAND_CSV, index=False)

    print(f"  {'band':<10}  {'n':>5}  {'pos':>5}  {'neg':>5}  {'AUC':>7}  {'Sens':>8}  {'Spec':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")
    for _, r in band_df.iterrows():
        print(f"  {r.age_band:<10}  {r.n_patients:>5}  {r.n_cancer:>5}  {r.n_non_cancer:>5}  "
              f"{r.auc:>7.4f}  {r.sens*100:>7.2f}%  {r.spec*100:>7.2f}%  "
              f"{r.tp:>4}  {r.fp:>4}  {r.fn:>4}  {r.tn:>4}")
    print()
    print(f"  Saved age-band metrics → {OUT_BAND_CSV.name}")


# ─────────────────────────────────────────────────────────────────────────
# Analysis 5 — SHAP for cross-test cells B + C
# ─────────────────────────────────────────────────────────────────────────

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


def _compute_shap_batch(X, cfg, seed_artifacts, top_n=5, label=""):
    if not HAS_SHAP:
        return [""] * len(X)
    n_rows, n_feats = X.shape
    feat_names = list(X.columns)
    agg = np.zeros((n_rows, n_feats), dtype=float)
    n_used = 0
    for seed, art in seed_artifacts.items():
        for name in cfg["ensemble_models"]:
            try:
                sv = shap.TreeExplainer(art["models"][name]).shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) >= 2 else sv[0]
                sv = np.asarray(sv)
                if sv.shape == agg.shape:
                    agg += sv
                    n_used += 1
                    logger.info(f"    SHAP ✓ {label} {name} seed={seed}")
            except Exception as e:
                logger.warning(f"    SHAP failed {name} seed={seed}: {e}")
    if n_used == 0:
        return [""] * n_rows
    agg /= n_used
    Xv = X.values
    out = []
    for i in range(n_rows):
        idx = np.argsort(-np.abs(agg[i]))[:top_n]
        out.append("; ".join(f"{feat_names[j]}={Xv[i,j]:.3g}({agg[i,j]:+.3f})" for j in idx))
    return out


def add_cross_cell_shap(df):
    _print_section("ANALYSIS 5 — SHAP for cross-test cells B + C")
    if not HAS_SHAP:
        print("  shap not installed — skipping")
        df["top_factors_B"] = ""
        df["top_factors_C"] = ""
        return df

    cfg_1mo,  art_1mo  = _load_artifacts("1mo")
    cfg_12mo, art_12mo = _load_artifacts("12mo")

    fm_1mo  = pd.read_parquet(FE_CLEANUP / "1mo"  / "feature_matrix_clean_1mo.parquet")
    fm_12mo = pd.read_parquet(FE_CLEANUP / "12mo" / "feature_matrix_clean_12mo.parquet")
    if "PATIENT_GUID" not in fm_1mo.columns:  fm_1mo  = fm_1mo.reset_index()
    if "PATIENT_GUID" not in fm_12mo.columns: fm_12mo = fm_12mo.reset_index()

    test_guids = df["patient_guid"].tolist()
    X_1mo  = (fm_1mo[fm_1mo["PATIENT_GUID"].isin(test_guids)]
              .set_index("PATIENT_GUID").reindex(test_guids))
    X_12mo = (fm_12mo[fm_12mo["PATIENT_GUID"].isin(test_guids)]
              .set_index("PATIENT_GUID").reindex(test_guids))

    # Cell B = 12mo model on 1mo data → SHAP on 12mo model with X_1mo
    X_B = X_1mo.reindex(columns=cfg_12mo["selected_features"], fill_value=0).fillna(0)
    # Cell C = 1mo model on 12mo data → SHAP on 1mo model with X_12mo
    X_C = X_12mo.reindex(columns=cfg_1mo["selected_features"], fill_value=0).fillna(0)

    print(f"  Computing SHAP for cell B (12mo model on 1mo data)...")
    df["top_factors_B"] = _compute_shap_batch(X_B, cfg_12mo, art_12mo, label="B-12mo")
    print(f"  Computing SHAP for cell C (1mo model on 12mo data)...")
    df["top_factors_C"] = _compute_shap_batch(X_C, cfg_1mo,  art_1mo,  label="C-1mo")
    print(f"  ✓ done")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-shap", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Run test_dual_horizon_local.py first to produce {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} test patients from {INPUT_CSV.name}")

    analyze_errors(df)
    threshold_sweep(df)
    age_band_breakdown(df)
    if not args.no_shap:
        df = add_cross_cell_shap(df)
    df.to_csv(OUT_FULL_CSV, index=False)

    print()
    print("═" * 78)
    print(f"  ✅ DONE. Enriched CSV: {OUT_FULL_CSV.name} ({len(df.columns)} cols)")
    print(f"     Threshold sweep:   {OUT_THR_CSV.name}")
    print(f"     Age-band metrics:  {OUT_BAND_CSV.name}")
    print("═" * 78)


if __name__ == "__main__":
    main()
