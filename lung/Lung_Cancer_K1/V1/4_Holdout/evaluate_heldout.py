"""
Held-out evaluation + Platt recalibration for the Lung model (touch-once labelled cohort).

Scores a TRAINED lookback model on the real 500/50k held-out, processed IDENTICALLY to training,
then calibrates with **Platt scaling on the held-out** (30% calib / 70% test) — the documented
approach (see Model_Results CALIBRATION_NOTES.md). NOTE: the King-Zeng true-prevalence offset is a
SEPARATE future *deployment* step and is intentionally NOT applied here.

Flow:
  heldout_test_{GAP}mo.sql --(BigQuery)--> events
   -> build_matrix(... xpoll_fit=False, xpoll_ref=<train ref>)   (FE + TRAIN xpoll, count->0/value->NaN)
   -> CancerPredictor(model).preprocess_data(matrix)             (TRAIN encoders/medians/scaler/select)
   -> RAW best-model proba
   -> stratified 30% calib / 70% test split (fixed seed)
   -> fit Platt (sigmoid) + isotonic (cross-check) on calib
   -> report on the disjoint test slice: AUROC/AUPRC (threshold-free), Brier+ECE (raw/Platt/iso),
      Sens/Spec/PPV/NPV at free/Youden, and save platt_calib_{window}.joblib + reliability curve.

Env:  GAP=12|1   NC_RATIO=1   WINDOW=5yr|10yr|20yr|lifetime   [CALIB_FRAC=0.30]
Run on the VM after the matching model exists:  GAP=12 WINDOW=5yr python 4_Holdout/evaluate_heldout.py
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                              # V1 root (parent of 4_Holdout)
GAP = os.environ.get("GAP", "12")
NC_RATIO = os.environ.get("NC_RATIO", "1")
WINDOW = os.environ.get("WINDOW", "5yr")
CALIB_FRAC = float(os.environ.get("CALIB_FRAC", "0.30"))
L = {"5yr": 5, "10yr": 10, "20yr": 20, "lifetime": 100}[WINDOW]

sys.path.insert(0, ROOT)                       # V1 root (for run_lookback_experiment)
sys.path.insert(0, os.path.join(ROOT, "2_FE"))
sys.path.insert(0, os.path.join(ROOT, "3_Modeling"))
import run_lookback_experiment as rl          # build_matrix, full_patient_frame, rp (export), GCS
from predict_unseen import CancerPredictor

OUT_DIR = os.path.join(ROOT, f"{GAP}mo_1to{NC_RATIO}", "lookback", WINDOW)
MODEL = os.path.join(OUT_DIR, f"model_{WINDOW}_1to{NC_RATIO}.joblib")
XREF = os.path.join(OUT_DIR, f"xpoll_ref_{WINDOW}.json")
SQL = os.path.join(HERE, "SQL", f"heldout_test_{GAP}mo.sql")


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            out += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return out


def main():
    assert os.path.exists(MODEL), f"trained model missing: {MODEL} (train the {WINDOW} window first)"
    assert os.path.exists(SQL), f"held-out SQL missing: {SQL}"
    print(f"{'='*70}\nHELD-OUT EVAL + PLATT — {GAP}mo 1:{NC_RATIO} {WINDOW}\n{'='*70}")

    # 1) export held-out cohort + 2) build FE matrix with the TRAIN xpoll reference (leakage-safe)
    ho_matrix = os.path.join(OUT_DIR, f"heldout_features_{WINDOW}.csv")
    if not os.path.exists(ho_matrix):
        ho_events = os.path.join(OUT_DIR, f"heldout_events_{WINDOW}.csv")
        rl.rp.export_query_to_csv(open(SQL).read(), ho_events, gcs_prefix=rl.GCS,
                                  tag=f"heldout_{GAP}mo_r{NC_RATIO}")
        ev = pd.read_csv(ho_events, low_memory=False)
        print(f"[heldout] {len(ev):,} events | {ev['patient_guid'].nunique():,} patients")
        rl.build_matrix(ev, L, ho_matrix, rl.full_patient_frame(ev), xpoll_fit=False,
                        xpoll_ref_path=XREF if os.path.exists(XREF) else None)

    # 3) RAW model probabilities (TRAIN-fitted transforms applied inside CancerPredictor)
    cp = CancerPredictor(MODEL)
    X, y = cp.preprocess_data(pd.read_csv(ho_matrix, low_memory=False))
    y = np.asarray(y).astype(int)
    raw = joblib.load(MODEL)["model"].predict_proba(X)[:, 1]

    # 4) stratified calib / test split (fixed seed)
    rng = np.random.RandomState(42)
    idx = np.arange(len(y)); cal = np.zeros(len(y), bool)
    for cls in (0, 1):
        c = idx[y == cls]; rng.shuffle(c); cal[c[:int(round(CALIB_FRAC * len(c)))]] = True
    rc, yc, rt, yt = raw[cal], y[cal], raw[~cal], y[~cal]

    # 5) fit Platt (sigmoid) + isotonic (cross-check) on the calib slice (held-out's own prevalence)
    platt = LogisticRegression(C=1e12, solver="lbfgs").fit(rc.reshape(-1, 1), yc)
    iso = IsotonicRegression(out_of_bounds="clip").fit(rc, yc)
    P = {"raw": rt, "platt": platt.predict_proba(rt.reshape(-1, 1))[:, 1], "isotonic": iso.predict(rt)}

    # 6) report on the disjoint TEST slice (AUROC/AUPRC threshold-free; Sens/Spec at free/Youden)
    au = roc_auc_score(yt, rt); ap = average_precision_score(yt, rt)
    fpr, tpr, thr = roc_curve(yt, rt); j = int(np.argmax(tpr - fpr)); T = thr[j]
    pred = rt >= T
    tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
    fn = int((~pred & (yt == 1)).sum()); tn = int((~pred & (yt == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0; spec = tn / (tn + fp) if tn + fp else 0
    ppv = tp / (tp + fp) if tp + fp else 0;  npv = tn / (tn + fn) if tn + fn else 0
    lines = [f"# Held-out — lung 1:{NC_RATIO} {GAP}mo {WINDOW} | calib n={cal.sum()} (pos {yc.sum()}) "
             f"test n={(~cal).sum()} (pos {yt.sum()}, prev {yt.mean()*100:.2f}%)",
             f"AUROC {au:.3f}  AUPRC {ap:.3f}  (threshold-free)",
             f"@free/Youden (test): Sens {sens*100:.1f}  Spec {spec*100:.1f}  PPV {ppv*100:.1f}  NPV {npv*100:.1f}",
             f"{'cal':9s} {'Brier':>8s} {'ECE':>8s}  (Platt = deployed calibrator)"]
    for name in ("raw", "platt", "isotonic"):
        lines.append(f"{name:9s} {brier_score_loss(yt, P[name]):8.4f} {_ece(P[name], yt):8.4f}")
    report = "\n".join(lines); print(report)
    open(os.path.join(OUT_DIR, f"heldout_recalib_{WINDOW}.txt"), "w").write(report + "\n")

    # 7) save the Platt calibrator + reliability curve
    joblib.dump(platt, os.path.join(OUT_DIR, f"platt_calib_{WINDOW}.joblib"))
    plt.figure(figsize=(7, 7)); bins = np.linspace(0, 1, 11)
    for name in ("raw", "platt", "isotonic"):
        pt = P[name]; bi = np.digitize(pt, bins[1:-1]); xs, ys = [], []
        for b in range(10):
            m = bi == b
            if m.sum() > 20:
                xs.append(pt[m].mean()); ys.append(yt[m].mean())
        plt.plot(xs, ys, "o-", label=f"{name} (Brier {brier_score_loss(yt, pt):.4f}, ECE {_ece(pt, yt):.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4, label="perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Lung 1:{NC_RATIO} {GAP}mo {WINDOW} — held-out reliability (test slice)")
    plt.legend(loc="upper left"); plt.grid(alpha=.3); plt.xlim(0, 1); plt.ylim(0, 1)
    plt.savefig(os.path.join(OUT_DIR, f"heldout_reliability_{WINDOW}.png"), dpi=140, bbox_inches="tight")
    print(f"-> platt_calib_{WINDOW}.joblib + heldout_recalib_{WINDOW}.txt + heldout_reliability_{WINDOW}.png")
    print("HELD-OUT EVAL + PLATT COMPLETE")


if __name__ == "__main__":
    main()
