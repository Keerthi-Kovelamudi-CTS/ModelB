"""
Real-world held-out (500/50k) inference for a CHOSEN lookback — run AFTER training/review.

This is a separate inference step (not part of the training sweep) so the held-out stays a true
one-shot real-world test, not something peeked at for model selection. It loads that lookback's
trained model, builds the held-out features at the same lookback (aligned to the full 500/50k
set, no-window-data patients zero-filled), predicts, and reports AUROC/AUPRC/Sens/Spec/PPV/NPV
with 95% bootstrap CIs at the operating threshold fixed on the internal test.

Usage:  cd ~/Keerthii && ~/lungenv/bin/python evaluate_holdout.py <5yr|10yr|20yr|lifetime>
"""
import os, sys, importlib.util
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "Modeling"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

lb = _load(os.path.join(ROOT, "run_lookback_experiment.py"), "run_lookback_experiment")
predmod = _load(os.path.join(ROOT, "Modeling", "predict_unseen_input_explainability_unified_v3.py"), "predict_unseen")
_load(os.path.join(ROOT, "Modeling", "training_v2_unified-query_data.py"), "lung_training")  # register LungCancerPredictor so the saved model unpickles


def main(arg, target_sens=0.90):
    L = {"5yr": 5, "10yr": 10, "20yr": 20, "lifetime": 100}.get(arg)
    assert L is not None, f"unknown lookback '{arg}' (use 5yr|10yr|20yr|lifetime)"
    tag = lb._tag(L); od = os.path.join(lb.LOOKBACK_DIR, tag)
    model_path = os.path.join(od, f"model_{tag}_1to1.joblib")
    assert os.path.exists(model_path), f"model missing — train first: {model_path}"

    print("=" * 70 + f"\nHELD-OUT (500/50k) inference — lookback {tag}\n" + "=" * 70)
    _, ho = lb.ensure_events()
    ho_ev = pd.read_csv(ho, low_memory=False)
    full_ho = lb.full_patient_frame(ho_ev)
    ho_feat = lb.build_matrix(ho_ev, L, os.path.join(od, "heldout_features.csv"), full_ho)

    cp = predmod.CancerPredictor(model_path)
    X, y = cp.preprocess_data(pd.read_csv(ho_feat, low_memory=False))
    s = cp.model.predict_proba(X)[:, 1]
    y = np.asarray(y).astype(int)

    # operating threshold = the one fixed on the internal test (from training metrics.csv)
    T = None
    mfile = os.path.join(od, "metrics.csv")
    if os.path.exists(mfile):
        m = pd.read_csv(mfile)
        if "threshold" in m.columns and len(m):
            T = float(m["threshold"].iloc[0])
    if T is None:
        T = lb.metrics.threshold_at_sensitivity(y, s, target_sens)
        print(f"(no internal threshold found; derived sens>={target_sens} threshold on held-out)")

    r = lb.metrics.eval_with_ci(y, s, threshold=T, label=f"{tag} heldout-500/50k")
    print("\n" + lb.metrics.format_result(r))
    out = os.path.join(od, "heldout_metrics.csv")
    pd.DataFrame([lb.metrics.results_to_row(r, tag, "heldout")]).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "usage: python evaluate_holdout.py <5yr|10yr|20yr|lifetime>"
    main(sys.argv[1])
