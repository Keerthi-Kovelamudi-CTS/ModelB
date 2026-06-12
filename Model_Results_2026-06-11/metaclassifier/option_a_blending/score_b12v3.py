"""Score b12_v3 VALID + TEST with our b12_v3-retrained lung models, SELECT the single best
window on VALID, and emit the ensemble contract as ONE model:
    xgboost_preds_valid.parquet  +  xgboost_preds_test.parquet   (model_name="xgboost")

Selection rule: best VALID AUROC (VALID is for tuning — allowed). TEST stays touch-once
(only scored once, with the already-chosen window). All-window preds are also written to
an internal/ subdir for the record (NOT delivered).

Runs from ~/Keerthii with EXP_SUFFIX=_b12v3 so it loads the b12_v3-trained models.
Hard rule: models trained on b12_v3 TRAIN only; VALID/TEST fully unseen here. ✅

Usage (cpu-02, after training):
  EXP_SUFFIX=_b12v3 GAP=12 NC_RATIO=1 ~/lungenv/bin/python Metaclassifer/score_b12v3.py \
      --valid ~/b12v3_run/valid_events.csv --test ~/b12v3_run/test_events.csv \
      --out ~/preds_b12v3 --model-version b12v3_12mo_1to1
"""
import argparse, os, sys, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/Keerthii
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "FE")); sys.path.insert(0, os.path.join(ROOT, "Modeling"))
def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
lb = _load(os.path.join(ROOT, "run_lookback_experiment.py"), "run_lookback_experiment")
fe = lb.fe
predmod = _load(os.path.join(ROOT, "Modeling", "predict_unseen_input_explainability.py"), "predict_unseen")

WINDOWS = [("5yr", 5), ("10yr", 10), ("20yr", 20), ("lifetime", 100)]


def score_window(split, ev, full, tag, L):
    """Return (patient_id[normalised], proba_1, y) for one window/split, or None if model missing."""
    od = os.path.join(lb.LOOKBACK_DIR, tag)
    model_path = os.path.join(od, f"model_{tag}_1to1.joblib")
    if not os.path.exists(model_path):
        print(f"  [skip] {split} {tag}: model missing"); return None
    feat_csv = os.path.join(od, f"b12v3_{split}_features.csv")
    lb.build_matrix(ev, L, feat_csv, full, xpoll_fit=False, xpoll_ref_path=os.path.join(od, f"xpoll_ref_{tag}.json"))
    feat = pd.read_csv(feat_csv, low_memory=False)
    pid = (fe.clean_patient_guid_series(feat["patient_guid"])
           .str.replace(r'[{}"]', "", regex=True).str.upper().str.strip())
    cp = predmod.CancerPredictor(model_path)
    X, y = cp.preprocess_data(feat)
    proba = np.asarray(cp.model.predict_proba(X))[:, 1].astype(float)
    return pid.values, proba, np.asarray(y).astype(int)


def emit(out_dir, fname, pid, proba, split, model_name, model_version):
    out = pd.DataFrame({"patient_id": pid, "split": split, "proba_1": proba, "proba_0": 1.0 - proba,
                        "model_name": model_name, "model_version": model_version,
                        "abstained": False}).drop_duplicates("patient_id")
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, fname); out.to_parquet(fp, index=False)
    print(f"  wrote {fp}  n={len(out)} mean_proba={out['proba_1'].mean():.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="~/preds_b12v3"); ap.add_argument("--model-version", default="b12v3_12mo_1to1")
    ap.add_argument("--window", default=None, help="fixed window (e.g. lifetime) -> skip VALID selection, emit this window as 'xgboost'")
    a = ap.parse_args(); out = os.path.expanduser(a.out); internal = os.path.join(out, "internal_all_windows")
    print(f"LOOKBACK_DIR = {lb.LOOKBACK_DIR}")
    valid_ev = pd.read_csv(os.path.expanduser(a.valid), low_memory=False); full_v = lb.full_patient_frame(valid_ev)

    # --- fixed-window mode (ensemble owner wants ONE window, e.g. lifetime) ---
    if a.window:
        L = {"5yr": 5, "10yr": 10, "20yr": 20, "lifetime": 100}[a.window]
        mv = f"{a.model_version}_{a.window}"
        pid_v, proba_v, y_v = score_window("valid", valid_ev, full_v, a.window, L)
        print(f"VALID AUROC ({a.window}) = {roc_auc_score(y_v, proba_v):.4f}")
        emit(out, "xgboost_preds_valid.parquet", pid_v, proba_v, "valid", "xgboost", mv)
        test_ev = pd.read_csv(os.path.expanduser(a.test), low_memory=False); full_t = lb.full_patient_frame(test_ev)
        pid_t, proba_t, _ = score_window("test", test_ev, full_t, a.window, L)
        emit(out, "xgboost_preds_test.parquet", pid_t, proba_t, "test", "xgboost", mv)
        print(f"\nDELIVER: {out}/xgboost_preds_valid.parquet + xgboost_preds_test.parquet  (window={a.window})")
        return

    # 1) score every window on VALID, record AUROC + keep preds
    vres = {}
    print("=== VALID per-window selection ===")
    for tag, L in WINDOWS:
        r = score_window("valid", valid_ev, full_v, tag, L)
        if r is None: continue
        pid, proba, y = r; auc = roc_auc_score(y, proba); vres[tag] = (auc, pid, proba, L)
        emit(internal, f"xgboost_{tag}_preds_valid.parquet", pid, proba, "valid", f"xgboost_{tag}", f"{a.model_version}_{tag}")
        print(f"  {tag}: VALID AUROC = {auc:.4f}")
    if not vres:
        sys.exit("no windows scored — models missing")

    # 2) pick best VALID AUROC
    best = max(vres, key=lambda t: vres[t][0]); auc_b, pid_v, proba_v, L_b = vres[best]
    print(f"\n>>> SELECTED window = {best} (VALID AUROC {auc_b:.4f}) <<<")

    # 3) emit the chosen window as the canonical 'xgboost' (2 contract files); TEST scored once here
    mv = f"{a.model_version}_{best}"
    emit(out, "xgboost_preds_valid.parquet", pid_v, proba_v, "valid", "xgboost", mv)
    test_ev = pd.read_csv(os.path.expanduser(a.test), low_memory=False); full_t = lb.full_patient_frame(test_ev)
    pid_t, proba_t, _ = score_window("test", test_ev, full_t, best, L_b)
    emit(out, "xgboost_preds_test.parquet", pid_t, proba_t, "test", "xgboost", mv)
    # also keep best-window test in internal for completeness
    emit(internal, f"xgboost_{best}_preds_test.parquet", pid_t, proba_t, "test", f"xgboost_{best}", mv)
    print(f"\nDELIVER: {out}/xgboost_preds_valid.parquet + xgboost_preds_test.parquet  (window={best})")


if __name__ == "__main__":
    main()
