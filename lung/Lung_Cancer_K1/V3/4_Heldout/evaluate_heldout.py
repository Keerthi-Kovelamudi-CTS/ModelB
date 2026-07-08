"""
Held-out evaluation + Platt recalibration for the (curated + data-driven) Lung model — touch-once labelled cohort.

Runs the tiered FE + stable-feature model on the held-out cohort (the 500 cancer / 50k non-cancer):

  SQL/heldout_test_{GAP}mo.sql  --(BigQuery)-->  events
   -> build_features.build(h, sql_path=heldout SQL, fit_split=False)  (SAME codelist + families as
      training; lifetime FE window; nothing fit on held-out, fully leak-safe)
   -> transform_external (TRAIN-fitted encoders/median-impute/scaler from the saved model) -> X
   -> raw = model["model"].predict_proba(X)             (the uncalibrated ensemble's raw scores)

  Honest reporting via a DISJOINT split of the held-out (stratified, fixed seed) — fitting Platt and
  choosing the operating threshold on the SAME patients we then report on would bias the metrics:
   -> CALIB (30%): fit the Platt recalibrator + pick the Youden operating threshold here only.
   -> TEST  (70%): report AUROC/AUPRC + Sens/Spec/PPV/NPV @ that threshold, plus Brier/ECE — all on
      patients NOT used to fit Platt or choose the threshold, so the numbers are unbiased.
   -> the CALIB-fit Platt is saved as the deployable recalibration artifact.

Run (VM/env with BigQuery + the trained model at ../output/{h}/{years}yr/model_{h}.joblib):
    python evaluate_heldout.py 12mo 5      # horizon [years lookback]; default 12mo, config FE_YEARS_BEFORE
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
FE_DIR = os.path.join(HERE, "..", "2_FE")
sys.path.insert(0, os.path.dirname(HERE))      # V3 root, for config
import config as C

WINDOW = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("12mo", "1mo") else "12mo"
GAP = WINDOW.replace("mo", "")
# optional 2nd arg = FE lookback window in years (matches the model being evaluated); default config
YEARS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else getattr(C, "FE_YEARS_BEFORE", 5)
OUT_DIR = os.path.join(HERE, "..", "output", WINDOW, f"{YEARS}yr")
MODEL = os.path.join(OUT_DIR, f"model_{WINDOW}.joblib")
SQL = os.path.join(HERE, "SQL", f"heldout_test_{GAP}mo.sql")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _transform_external(df_raw, md):
    """Replicate LungCancerPredictor.transform_external using the SAVED model dict (no class instance):
    encoders -> reindex to TRAIN feature_names -> value cols filled w/ TRAIN median, rest -> 0 -> scale.
    Returns the scaled array the base ensemble expects (identical pipeline to training, no refit)."""
    d = df_raw.copy()
    for col, mp in (md.get("encoders") or {}).items():
        d[col] = (d[col].fillna("Unknown").astype(str).map(mp).fillna(-1) if col in d.columns else -1)
    d = d.reindex(columns=md["feature_names"])               # same columns/order as train; missing -> NaN
    d = d.apply(pd.to_numeric, errors="coerce").astype("float64")   # never nullable Int64 (parquet cache)
    vcols = set(md.get("value_cols") or [])
    val = [c for c in md["feature_names"] if c in vcols]
    if val and md.get("impute_medians") is not None:
        d[val] = d[val].fillna(md["impute_medians"])         # value -> TRAIN median
    d = d.replace([np.inf, -np.inf], np.nan)                 # inf -> NaN (then median/0 fill), scaler-safe
    return md["scaler"].transform(d.fillna(0.0).values)      # remaining (count) -> 0, then TRAIN scaler


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    assert os.path.exists(SQL), f"held-out SQL missing: {SQL}"
    assert os.path.exists(MODEL), f"trained model missing: {MODEL} (run run_v3.py first)"
    print(f"{'='*70}\nHELD-OUT EVAL + PLATT — {WINDOW} {YEARS}yr (gap {GAP}mo)\n{'='*70}")

    # 1) build the held-out feature matrix (SAME codelist + families as training; lifetime FE; no percentile)
    ho_matrix = os.path.join(OUT_DIR, f"heldout_features_{WINDOW}.parquet")
    if os.path.exists(ho_matrix):
        mat = pd.read_parquet(ho_matrix)
        print(f"[heldout] cached matrix: {mat.shape[0]:,} patients x {mat.shape[1]:,} cols")
    else:
        bf = _load(os.path.join(FE_DIR, "build_features.py"), "build_features")
        mat, _ = bf.build(WINDOW, sql_path=SQL, fit_split=False, years=YEARS)   # touch-once: nothing fit on held-out
        mat.to_parquet(ho_matrix, index=False)
        print(f"[heldout] built matrix: {mat.shape[0]:,} patients x {mat.shape[1]:,} cols -> {ho_matrix}")

    # 2) align/impute/scale with the TRAIN-fitted pipeline, then raw ensemble probabilities
    # Register the lung_training module first so joblib can unpickle CalibratedLungModel (it was
    # pickled with __module__ == "lung_training").
    _load(os.path.join(HERE, "..", "3_Modeling", "lung_training.py"), "lung_training")
    md = joblib.load(MODEL)
    y = mat["cancer_class"].astype(int).to_numpy()
    X = _transform_external(mat, md)
    raw = md["model"].predict_proba(X)[:, 1]
    print(f"[heldout] {len(y):,} patients ({int(y.sum())} cancer / {int((1-y).sum())} non-cancer)")

    # 3) DISJOINT calib/test split of the held-out (stratified, fixed seed). Fitting Platt and
    #    choosing the operating threshold on the same patients we report on would bias the metrics,
    #    so we carve a 30% CALIB slice for both of those, and report ONLY on the 70% TEST slice.
    cal_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=0.70, random_state=42, stratify=y)
    raw_cal, raw_te = raw[cal_idx], raw[te_idx]
    y_cal, y_te = y[cal_idx], y[te_idx]
    print(f"[heldout] calib {len(y_cal):,} ({int(y_cal.sum())} cancer) / "
          f"test {len(y_te):,} ({int(y_te.sum())} cancer)  (stratified, seed 42)")

    # 4) fit the deployable Platt recalibrator on CALIB ONLY (the saved deployment artifact), and pick
    #    the Youden operating threshold on CALIB. Both are then APPLIED to the unseen TEST slice.
    platt = LogisticRegression(max_iter=1000).fit(raw_cal.reshape(-1, 1), y_cal)
    joblib.dump(platt, os.path.join(OUT_DIR, f"platt_calib_{WINDOW}.joblib"))
    p_cal = platt.predict_proba(raw_cal.reshape(-1, 1))[:, 1]
    p_te = platt.predict_proba(raw_te.reshape(-1, 1))[:, 1]
    fpr, tpr, thr = roc_curve(y_cal, p_cal)            # threshold chosen on CALIB
    cut = thr[int(np.argmax(tpr - fpr))]

    # 5) report everything on the DISJOINT TEST slice (calibrated probabilities, calib-chosen threshold)
    au = roc_auc_score(y_te, p_te); ap = average_precision_score(y_te, p_te)
    pred = (p_te >= cut).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum()); fp = int(((pred == 1) & (y_te == 0)).sum())
    tn = int(((pred == 0) & (y_te == 0)).sum()); fn = int(((pred == 0) & (y_te == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0

    lines = [
        f"# Held-out — lung {WINDOW} (gap {GAP}mo) | reported on the 70% TEST slice n={len(y_te):,} "
        f"(pos {int(y_te.sum())}, prevalence {100*y_te.mean():.2f}%) | Platt + threshold fit on the disjoint 30% calib",
        f"AUROC {au:.4f}   AUPRC {ap:.4f}",
        f"@Youden (threshold chosen on calib, applied to test): Sens {sens*100:.1f}  Spec {spec*100:.1f}  "
        f"PPV {ppv*100:.1f}  NPV {npv*100:.1f}  (cut={cut:.4f})",
        f"-- calibration on the TEST slice (out-of-sample: Platt fit on calib) --",
        f"{'cal':9s} {'Brier':>8s} {'ECE':>8s}",
        f"{'raw':9s} {brier_score_loss(y_te, raw_te):>8.4f} {_ece(raw_te, y_te):>8.4f}   (uncalibrated)",
        f"{'Platt':9s} {brier_score_loss(y_te, p_te):>8.4f} {_ece(p_te, y_te):>8.4f}   (calib-fit, out-of-sample)",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(OUT_DIR, f"heldout_recalib_{WINDOW}.txt"), "w").write(report + "\n")
    print(f"\n-> heldout_recalib_{WINDOW}.txt + platt_calib_{WINDOW}.joblib (deployment artifact)  (in {OUT_DIR})")


if __name__ == "__main__":
    main()
