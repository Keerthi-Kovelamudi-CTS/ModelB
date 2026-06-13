"""
Dual-horizon held-out evaluation: combine the 12mo and 1mo models on the SAME held-out patients.

The 12mo and 1mo held-out cohorts are the SAME 500/50k patients/anchors, differing ONLY by the
feature gap (months_before = 12 vs 1). So every held-out patient has two feature views; we score
each with its matching model and merge per patient. Reuses the artifacts the single-horizon runs
already produced (heldout_features_{window}.csv, model_{window}_1to{ratio}.joblib) — no re-FE.

On a shared stratified 70% test slice (Platt fit per horizon on the 30% calib slice, same split for
both horizons so the test patients align), reports:
  - 12mo only / 1mo only   : AUROC + Sens/Spec/PPV/NPV @Youden   (sanity vs the single-horizon runs)
  - DUAL max-prob          : score = max(calibrated p12, p1); AUROC + Sens/Spec/PPV/NPV @Youden
  - DUAL OR@thresholds     : flag if 12mo OR 1mo exceeds its OWN Youden threshold (deployment rule)
  - True-cancer coverage   : among real cancers, caught by both / 12mo-only / 1mo-only / neither

Noisy-OR is intentionally NOT reported (assumes the two models are independent; they are highly
correlated, so noisy-OR would over-combine and be miscalibrated).

Env:  NC_RATIO=1|5|10   WINDOW=5yr|10yr|20yr|lifetime   [CALIB_FRAC=0.30]
Run:  NC_RATIO=1 WINDOW=5yr python 4_Heldout/evaluate_dual_horizon.py
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # V1 root
NC_RATIO = os.environ.get("NC_RATIO", "1")
WINDOW = os.environ.get("WINDOW", "5yr")
CALIB_FRAC = float(os.environ.get("CALIB_FRAC", "0.30"))

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "2_FE"))
sys.path.insert(0, os.path.join(ROOT, "3_Modeling"))
from predict_unseen import CancerPredictor


def _score(gap):
    """Raw model probability per patient for one horizon (rows preserved -> positional guid)."""
    d = os.path.join(ROOT, f"{gap}mo_1to{NC_RATIO}", "lookback", WINDOW)
    feat = os.path.join(d, f"heldout_features_{WINDOW}.csv")
    model = os.path.join(d, f"model_{WINDOW}_1to{NC_RATIO}.joblib")
    assert os.path.exists(feat), f"missing {feat} (run the {gap}mo 1:{NC_RATIO} held-out first)"
    assert os.path.exists(model), f"missing {model}"
    df = pd.read_csv(feat, low_memory=False)
    guid = df["patient_guid"].astype(str).values
    cp = CancerPredictor(model)
    X, y = cp.preprocess_data(df)                # drops COLUMNS only -> rows/order preserved
    assert len(X) == len(guid), f"{gap}mo: row count changed in preprocess ({len(X)} vs {len(guid)})"
    raw = joblib.load(model)["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({"guid": guid, "y": np.asarray(y).astype(int), f"raw{gap}": raw})


def _youden_T(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def _oppoint(y, pred):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0; spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0;  npv = tn / (tn + fn) if tn + fn else 0.0
    return sens, spec, ppv, npv


def main():
    print(f"{'='*74}\nDUAL-HORIZON HELD-OUT — lung 1:{NC_RATIO} {WINDOW} (12mo + 1mo)\n{'='*74}")
    s12 = _score(12); s1 = _score(1)
    m = s12.merge(s1[["guid", "raw1", "y"]], on="guid", suffixes=("", "_b"))
    assert (m["y"] == m["y_b"]).all(), "label mismatch between horizons for some patients"
    m = m.drop(columns=["y_b"])
    print(f"matched patients: {len(m):,}  (12mo {len(s12):,}, 1mo {len(s1):,})  pos={int(m['y'].sum())}")

    # shared stratified 30% calib / 70% test split (same partition for both horizons)
    rng = np.random.RandomState(42)
    y = m["y"].values; idx = np.arange(len(m)); cal = np.zeros(len(m), bool)
    for c in (0, 1):
        ci = idx[y == c]; rng.shuffle(ci); cal[ci[:int(round(CALIB_FRAC * len(ci)))]] = True
    test = ~cal; yt = y[test]

    def platt(col):
        lr = LogisticRegression(C=1e12, solver="lbfgs").fit(m[col].values[cal].reshape(-1, 1), y[cal])
        return lr.predict_proba(m[col].values[test].reshape(-1, 1))[:, 1]
    p12 = platt("raw12"); p1 = platt("raw1"); dual = np.maximum(p12, p1)
    T12 = _youden_T(yt, p12); T1 = _youden_T(yt, p1); Td = _youden_T(yt, dual)

    lines = [f"# Dual-horizon held-out — lung 1:{NC_RATIO} {WINDOW} | test n={int(test.sum())} "
             f"(pos {int(yt.sum())}, prev {yt.mean()*100:.2f}%)  [Platt per horizon on 30% calib]",
             f"{'model':22s} {'AUROC':>7} {'AUPRC':>7} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6}"]

    def row(name, p, T):
        au = roc_auc_score(yt, p); ap = average_precision_score(yt, p)
        sens, spec, ppv, npv = _oppoint(yt, p >= T)
        return f"{name:22s} {au:7.3f} {ap:7.3f} {sens*100:6.1f} {spec*100:6.1f} {ppv*100:6.1f} {npv*100:6.1f}"

    lines.append(row("12mo only", p12, T12))
    lines.append(row("1mo only", p1, T1))
    lines.append(row("DUAL max(p12,p1)", dual, Td))
    sens, spec, ppv, npv = _oppoint(yt, (p12 >= T12) | (p1 >= T1))
    lines.append(f"{'DUAL OR@thresholds':22s} {'-':>7} {'-':>7} {sens*100:6.1f} {spec*100:6.1f} {ppv*100:6.1f} {npv*100:6.1f}")

    pos = yt == 1; c12 = p12 >= T12; c1 = p1 >= T1
    both = int((pos & c12 & c1).sum()); only12 = int((pos & c12 & ~c1).sum())
    only1 = int((pos & ~c12 & c1).sum()); neither = int((pos & ~c12 & ~c1).sum())
    lines += ["",
              f"# True-cancer coverage @each horizon's Youden threshold (of {int(pos.sum())} cancers in test):",
              f"  both={both}  12mo-only={only12}  1mo-only={only1}  missed-by-both={neither}",
              f"  -> OR-rule catches {both+only12+only1}/{int(pos.sum())} ({(both+only12+only1)/max(pos.sum(),1)*100:.1f}%); "
              f"1mo adds {only1} the 12mo misses, 12mo adds {only12} the 1mo misses."]

    report = "\n".join(lines); print("\n" + report)
    out = os.path.join(HERE, f"dual_heldout_{WINDOW}_1to{NC_RATIO}.txt")   # HERE = robust to folder name
    open(out, "w").write(report + "\n")
    print(f"\n-> saved {out}\nDUAL-HORIZON EVAL COMPLETE")


if __name__ == "__main__":
    main()
