"""
Local dual-horizon test driver — full 2×2 model/cutoff comparison.

For each of 844 test patients (held out from BOTH 1mo and 12mo training/val):

  Compute ALL FOUR cells of the model × cutoff matrix:

                       │  Patient's 12mo-cutoff data  │  Patient's 1mo-cutoff data
    ───────────────────┼──────────────────────────────┼──────────────────────────────
    12mo model         │  cell A (training-matched)   │  cell B (cross-test)
    1mo  model         │  cell C (cross-test)         │  cell D (training-matched)
    ───────────────────┴──────────────────────────────┴──────────────────────────────

  Cells A + D are the "production" cells — each model gets the cutoff data it was
  trained on. Dual-horizon = OR(A.decision, D.decision).

  Cells B + C are diagnostic cross-tests — show how each model behaves on
  out-of-distribution temporal data (model robustness probe).

Outputs:
  1. test_dual_horizon_per_patient.csv  — 1 row/patient with all 4 cells + dual horizon
  2. Console: 5 metric blocks (A, B, C, D, E=dual_horizon) — AUC/Sens/Spec/confusion matrix

Usage:
    python3 test_dual_horizon_local.py
    python3 test_dual_horizon_local.py --no-shap         # skip SHAP for speed
    python3 test_dual_horizon_local.py --out custom.csv
"""

import argparse
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
FE_CLEANUP    = PROJECT_ROOT / "2_Feature_Engineering" / "data" / "features"
DEFAULT_OUT   = SCRIPT_DIR / "test_dual_horizon_per_patient.csv"

# Required so joblib can unpickle CalibratorWrapper
sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
from _calibrator import CalibratorWrapper  # noqa: F401

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _load_artifacts(window):
    """Load model config + all seeds' base models + meta-learners for `window`."""
    model_dir = TRAIN_RESULTS / window / "saved_models"
    cfg = joblib.load(model_dir / "config.json")
    seeds = cfg.get("seeds", [42])
    ensemble_models = cfg["ensemble_models"]
    seed_artifacts = {}
    for seed in seeds:
        seed_dir = model_dir / f"seed_{seed}"
        models = {n: joblib.load(seed_dir / f"{n}_model.pkl") for n in ensemble_models}
        meta = joblib.load(seed_dir / "meta_learner.pkl")
        seed_artifacts[seed] = {"models": models, "meta": meta}
    return cfg, seed_artifacts


def _align_X(feature_vector, selected):
    """Series/dict → 1-row DataFrame aligned to model's selected_features (fill 0)."""
    if isinstance(feature_vector, dict):
        s = pd.Series(feature_vector)
    else:
        s = feature_vector
    return pd.DataFrame([{f: s.get(f, 0) for f in selected}], columns=selected).fillna(0)


_AGE_FALLBACK_WARNED = False


def _get_patient_age(s):
    """Resolve patient age from a feature-vector Series, trying canonical names.

    Returns (age_float, source_name). source_name='__fallback_65__' if no column found.
    """
    global _AGE_FALLBACK_WARNED
    for col in ("patient_age", "PATIENT_AGE", "AGE_AT_INDEX", "age"):
        if col in s.index:
            v = s[col]
            try:
                fv = float(v)
                if np.isfinite(fv) and fv > 0:
                    return fv, col
            except (TypeError, ValueError):
                pass
    if not _AGE_FALLBACK_WARNED:
        logger.warning(
            "  ⚠ patient_age not found in feature vector — falling back to 65 "
            "(per-band calibration will be inaccurate). Check FE pipeline preserves the age column."
        )
        _AGE_FALLBACK_WARNED = True
    return 65.0, "__fallback_65__"


def _score(feature_vector, cfg, seed_artifacts):
    """Apply full predict pipeline: base models → meta → per-band calibration.

    Returns (risk_proba, decision, age_band, insufficient_data, age_used, age_source).
    Mirrors predict_single_patient.predict() exactly.
    """
    selected = cfg["selected_features"]
    ensemble_models = cfg["ensemble_models"]
    X = _align_X(feature_vector, selected)
    s = feature_vector if isinstance(feature_vector, pd.Series) else pd.Series(feature_vector)

    per_seed_preds = []
    for _, art in seed_artifacts.items():
        base_preds = np.array(
            [art["models"][n].predict_proba(X)[0, 1] for n in ensemble_models]
        )
        stacked = art["meta"].predict_proba(base_preds.reshape(1, -1))[0, 1]
        per_seed_preds.append(stacked)
    raw_proba = float(np.mean(per_seed_preds))

    age, age_source = _get_patient_age(s)
    band_idx = np.searchsorted(cfg["band_bins"][1:], age, side="right")
    band_idx = min(band_idx, len(cfg["band_labels"]) - 1)
    age_band = cfg["band_labels"][band_idx]

    cal = cfg["calibrators_per_band"].get(age_band, cfg["global_calibrator"])
    thr = cfg["thresholds_per_band"].get(age_band, cfg["global_threshold"])

    risk_proba = float(cal.transform(np.array([raw_proba]))[0])
    decision = int(risk_proba >= thr)

    clinical_cols = [c for c in selected if c not in ("patient_age", "AGE_BAND")]
    insuf = bool(clinical_cols and (X[clinical_cols].abs().sum(axis=1).iloc[0] == 0))
    if insuf:
        risk_proba = 0.0
        decision = 0
    return risk_proba, decision, age_band, insuf, age, age_source


def _verdict(decision, proba):
    if decision == 1: return "Cancer"
    if proba == 0.0:  return "Not enough data"
    return "No Cancer"


def _classify(d_a, p_a, d_d, p_d):
    """Dual-horizon OR — uses the training-matched cells A (12mo) + D (1mo)."""
    fired = []
    if d_d == 1: fired.append("1mo")
    if d_a == 1: fired.append("12mo")
    if fired:
        return "Cancer", fired
    if p_a == 0.0 and p_d == 0.0:
        return "Not enough data", []
    return "No Cancer", []


def _xgb_native_shap(xgb_model, X):
    """Bypass shap.TreeExplainer for XGBoost (incompatible with XGB 3.x JSON-array thresholds).
    Use XGBoost's built-in pred_contribs which returns (n_rows, n_feats + 1); last col is bias.
    """
    import xgboost as xgb
    booster = xgb_model.get_booster()
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dm, pred_contribs=True)
    return contribs[:, :-1]


def _compute_shap_batch(window, X, cfg, seed_artifacts, top_n=5):
    """Avg SHAP across all base models × seeds, return per-row top-N strings."""
    if not HAS_SHAP:
        return [""] * len(X)
    n_rows, n_feats = X.shape
    feat_names = list(X.columns)
    agg = np.zeros((n_rows, n_feats), dtype=float)
    n_used = 0
    for seed, art in seed_artifacts.items():
        for name in cfg["ensemble_models"]:
            try:
                model = art["models"][name]
                if name == "xgboost":
                    sv = _xgb_native_shap(model, X)
                else:
                    sv = shap.TreeExplainer(model).shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) >= 2 else sv[0]
                sv = np.asarray(sv)
                if sv.shape == agg.shape:
                    agg += sv
                    n_used += 1
                    logger.info(f"    SHAP ✓ {window} {name} seed={seed}")
                else:
                    logger.warning(f"    SHAP shape mismatch {name} seed={seed}: {sv.shape} vs {agg.shape}")
            except Exception as e:
                logger.warning(f"    SHAP failed {name} seed={seed}: {e}")
    if n_used == 0:
        return [""] * len(X)
    agg /= n_used
    Xv = X.values
    out = []
    for i in range(n_rows):
        idx = np.argsort(-np.abs(agg[i]))[:top_n]
        out.append("; ".join(
            f"{feat_names[j]}={Xv[i,j]:.3g}({agg[i,j]:+.3f})" for j in idx
        ))
    return out


def _metrics_block(name, y_true, decisions, probas):
    """Compute and print AUC/Sens/Spec/CM for one of the 5 metric blocks."""
    y_true = np.asarray(y_true).astype(int)
    decisions = np.asarray(decisions).astype(int)
    probas = np.asarray(probas).astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, decisions, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = roc_auc_score(y_true, probas)
    except ValueError:
        auc = float("nan")
    print(f"  {name}")
    print(f"    AUC = {auc:.4f}   Sens = {sens*100:6.2f}%   Spec = {spec*100:6.2f}%")
    print(f"    CM:  TP={tp:>4}  FP={fp:>4}  FN={fn:>4}  TN={tn:>4}")
    print()


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP top-factors")
    p.add_argument("--top-n", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── 1. Load per-window test-set predictions (used only for patient intersection + y_true)
    pred_1mo  = pd.read_csv(TRAIN_RESULTS / "1mo"  / "predictions_1mo.csv")
    pred_12mo = pd.read_csv(TRAIN_RESULTS / "12mo" / "predictions_12mo.csv")
    merged = pred_1mo.merge(
        pred_12mo, on="patient_id", suffixes=("_1mo", "_12mo"), validate="one_to_one",
    )
    logger.info(f"Test patients (in both windows' test splits): {len(merged):,}")

    test_guids = merged["patient_id"].tolist()
    y_true = merged["y_true_1mo"].astype(int).to_numpy()

    # ── 2. Load saved_models + feature matrices ──
    logger.info("\nLoading model artifacts + feature matrices...")
    cfg_1mo,  art_1mo  = _load_artifacts("1mo")
    cfg_12mo, art_12mo = _load_artifacts("12mo")
    fm_1mo  = pd.read_parquet(FE_CLEANUP / "1mo"  / "breast_feature_matrix.parquet")
    fm_12mo = pd.read_parquet(FE_CLEANUP / "12mo" / "breast_feature_matrix.parquet")
    if "patient_id" not in fm_1mo.columns:  fm_1mo  = fm_1mo.reset_index()
    if "patient_id" not in fm_12mo.columns: fm_12mo = fm_12mo.reset_index()

    # Filter to test patients, ordered same as `test_guids`
    fm_1mo_test  = (fm_1mo[fm_1mo["patient_id"].isin(test_guids)]
                    .set_index("patient_id").reindex(test_guids))
    fm_12mo_test = (fm_12mo[fm_12mo["patient_id"].isin(test_guids)]
                    .set_index("patient_id").reindex(test_guids))

    # ── 3. Score all 4 cells per patient (loop) ──
    logger.info(f"\nScoring 4 cells per patient ({len(test_guids):,} patients)...")
    cells_A = []  # 12mo model on 12mo data — training-matched
    cells_B = []  # 12mo model on 1mo  data — cross
    cells_C = []  # 1mo  model on 12mo data — cross
    cells_D = []  # 1mo  model on 1mo  data — training-matched
    for i, pg in enumerate(test_guids):
        fv_1mo  = fm_1mo_test.loc[pg]
        fv_12mo = fm_12mo_test.loc[pg]
        cells_A.append(_score(fv_12mo, cfg_12mo, art_12mo))
        cells_B.append(_score(fv_1mo,  cfg_12mo, art_12mo))
        cells_C.append(_score(fv_12mo, cfg_1mo,  art_1mo))
        cells_D.append(_score(fv_1mo,  cfg_1mo,  art_1mo))
        if (i+1) % 200 == 0:
            logger.info(f"  scored {i+1}/{len(test_guids)}")

    # ── 4. SHAP for the 2 production cells (A + D) ──
    if not args.no_shap and HAS_SHAP:
        logger.info("\nComputing SHAP top-factors (production cells A + D)...")
        X_A = (fm_12mo_test.reindex(columns=cfg_12mo["selected_features"], fill_value=0).fillna(0))
        X_D = (fm_1mo_test .reindex(columns=cfg_1mo ["selected_features"], fill_value=0).fillna(0))
        logger.info("  cell A (12mo model on 12mo data)...")
        shap_A = _compute_shap_batch("12mo", X_A, cfg_12mo, art_12mo, top_n=args.top_n)
        logger.info("  cell D (1mo model on 1mo data)...")
        shap_D = _compute_shap_batch("1mo",  X_D, cfg_1mo,  art_1mo,  top_n=args.top_n)
    else:
        if not HAS_SHAP and not args.no_shap:
            logger.warning("shap not installed — skipping SHAP")
        shap_A = [""] * len(test_guids)
        shap_D = [""] * len(test_guids)

    # ── 5. Per-patient CSV rows ──
    rows = []
    age_source_counts = {}
    for i, pg in enumerate(test_guids):
        (a_p, a_d, a_band, a_insuf, a_age, a_src) = cells_A[i]
        (b_p, b_d, b_band, b_insuf, b_age, b_src) = cells_B[i]
        (c_p, c_d, c_band, c_insuf, c_age, c_src) = cells_C[i]
        (d_p, d_d, d_band, d_insuf, d_age, d_src) = cells_D[i]
        result, fired_by = _classify(a_d, a_p, d_d, d_p)
        age_source_counts[a_src] = age_source_counts.get(a_src, 0) + 1
        rows.append({
            "patient_guid":             pg,
            "age_band":                 a_band,
            "age_used":                 a_age,
            "age_source":               a_src,
            "y_true":                   int(y_true[i]),
            "y_true_label":             "Cancer" if y_true[i] == 1 else "No Cancer",
            # CELL A — 12mo model on 12mo data (training-matched, production)
            "A_score_12mo_on_12mo_data":      a_p,
            "A_decision":                     a_d,
            "A_verdict":                      _verdict(a_d, a_p),
            "A_top_factors":                  shap_A[i],
            # CELL B — 12mo model on 1mo data (cross-test)
            "B_score_12mo_on_1mo_data":       b_p,
            "B_decision":                     b_d,
            "B_verdict":                      _verdict(b_d, b_p),
            # CELL C — 1mo model on 12mo data (cross-test)
            "C_score_1mo_on_12mo_data":       c_p,
            "C_decision":                     c_d,
            "C_verdict":                      _verdict(c_d, c_p),
            # CELL D — 1mo model on 1mo data (training-matched, production)
            "D_score_1mo_on_1mo_data":        d_p,
            "D_decision":                     d_d,
            "D_verdict":                      _verdict(d_d, d_p),
            "D_top_factors":                  shap_D[i],
            # DUAL-HORIZON (OR of A + D) — what production deploys
            "dual_horizon_result":            result,
            "fired_by":                       "|".join(fired_by) if fired_by else "",
            "max_production_score":           max(a_p, d_p),
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)

    # ── 5b. Age source + distribution diagnostics ──
    print()
    print("  Age source breakdown (resolved from):")
    for src, n in sorted(age_source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:24s}  {n:>6,}  ({100*n/len(out_df):.1f}%)")
    if "__fallback_65__" in age_source_counts:
        print("  ⚠ Some patients had no age column — per-band calibration unreliable for those.")
    print()
    print("  Age band distribution (actual, computed from resolved age):")
    print(out_df["age_band"].value_counts().sort_index().to_string())

    # ── 6. Metrics: 5 blocks ──
    print()
    print("═" * 76)
    print(f"  DUAL-HORIZON 2×2 EVALUATION — {len(out_df):,} test patients")
    print(f"  (held out from BOTH 1mo and 12mo training/val splits)")
    print(f"  cancer: {(y_true==1).sum():,}   non-cancer: {(y_true==0).sum():,}")
    print("═" * 76)
    print()
    _metrics_block(
        "A) 12mo model on 12mo-cutoff data  (training-matched, production)",
        y_true, out_df["A_decision"], out_df["A_score_12mo_on_12mo_data"],
    )
    _metrics_block(
        "B) 12mo model on 1mo-cutoff data   (CROSS-TEST — robustness probe)",
        y_true, out_df["B_decision"], out_df["B_score_12mo_on_1mo_data"],
    )
    _metrics_block(
        "C) 1mo  model on 12mo-cutoff data  (CROSS-TEST — robustness probe)",
        y_true, out_df["C_decision"], out_df["C_score_1mo_on_12mo_data"],
    )
    _metrics_block(
        "D) 1mo  model on 1mo-cutoff data   (training-matched, production)",
        y_true, out_df["D_decision"], out_df["D_score_1mo_on_1mo_data"],
    )
    dual_decision = (out_df["dual_horizon_result"] == "Cancer").astype(int)
    _metrics_block(
        "E) DUAL-HORIZON OR(A, D)            (= what production deploys)",
        y_true, dual_decision, out_df["max_production_score"],
    )
    print(f"  Full per-patient CSV: {out_path}")


if __name__ == "__main__":
    main()
