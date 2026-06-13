"""
Dual-horizon held-out evaluation — per held-out DATASET, score it with BOTH models and combine.

For each held-out scenario we run BOTH the 12mo-trained and 1mo-trained models on the SAME dataset,
then merge per patient. This mirrors the single-horizon runs (12mo-on-12mo, 1mo-on-1mo) but adds
the other ("cross") model and the dual combination on each dataset:

  DATASET heldout_test_12mo.sql  (events cut off 12 months pre-dx):
     - 12mo model (matched) | 1mo model (cross) | DUAL = combine(both)
  DATASET heldout_test_1mo.sql   (events cut off  1 month  pre-dx):
     - 1mo model (matched)  | 12mo model (cross) | DUAL = combine(both)

Both models read the SAME dataset; each CancerPredictor reindexes it to its own trained feature set
(missing cols -> 0). Reuses the heldout_features_{window}.csv that the single-horizon runs cached
(no re-FE). Per dataset: shared 30% calib / 70% test split, Platt per model on calib, reported on test:
  - matched model / cross model : AUROC + Sens/Spec/PPV/NPV @Youden
  - DUAL max(p_matched,p_cross) : AUROC + Sens/Spec/PPV/NPV @Youden
  - DUAL OR@thresholds          : flag if either model exceeds its own Youden threshold
  - true-cancer coverage        : caught by both / matched-only / cross-only / neither
Noisy-OR omitted (models correlated -> would over-combine).

Env:  NC_RATIO=1|5|10   WINDOW=5yr|10yr|20yr|lifetime   [DATASET=both|12|1]   [CALIB_FRAC=0.30]
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
ROOT = os.path.dirname(HERE)
NC_RATIO = os.environ.get("NC_RATIO", "1")
WINDOW = os.environ.get("WINDOW", "5yr")
CALIB_FRAC = float(os.environ.get("CALIB_FRAC", "0.30"))
DATASET = os.environ.get("DATASET", "both")          # which held-out dataset(s) to evaluate on

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "2_FE"))
sys.path.insert(0, os.path.join(ROOT, "3_Modeling"))
from predict_unseen import CancerPredictor


def _model_path(gap):
    return os.path.join(ROOT, f"{gap}mo_1to{NC_RATIO}", "lookback", WINDOW, f"model_{WINDOW}_1to{NC_RATIO}.joblib")


def _dataset_path(gap):
    return os.path.join(ROOT, f"{gap}mo_1to{NC_RATIO}", "lookback", WINDOW, f"heldout_features_{WINDOW}.csv")


def _raw(df, model_gap):
    """Score the given held-out matrix with the model trained for model_gap (reindexes to its features)."""
    mp = _model_path(model_gap)
    assert os.path.exists(mp), f"missing model {mp}"
    cp = CancerPredictor(mp)
    X, y = cp.preprocess_data(df.copy())
    raw = joblib.load(mp)["model"].predict_proba(X)[:, 1]
    return raw, np.asarray(y).astype(int)


def _youden_T(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def _op(y, pred):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0; spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0;  npv = tn / (tn + fn) if tn + fn else 0.0
    return sens, spec, ppv, npv


def eval_on_dataset(ds_gap):
    """Score the ds_gap held-out dataset with BOTH the 12mo and 1mo models; combine."""
    feat = _dataset_path(ds_gap)
    assert os.path.exists(feat), f"missing {feat} (run the {ds_gap}mo 1:{NC_RATIO} held-out first)"
    df = pd.read_csv(feat, low_memory=False)
    raw12, y = _raw(df, 12)
    raw1, y1 = _raw(df, 1)
    assert (y == y1).all(), "label mismatch scoring the same dataset twice"

    rng = np.random.RandomState(42)
    idx = np.arange(len(y)); cal = np.zeros(len(y), bool)
    for c in (0, 1):
        ci = idx[y == c]; rng.shuffle(ci); cal[ci[:int(round(CALIB_FRAC * len(ci)))]] = True
    test = ~cal; yt = y[test]

    def platt(raw):
        lr = LogisticRegression(C=1e12, solver="lbfgs").fit(raw[cal].reshape(-1, 1), y[cal])
        return lr.predict_proba(raw[test].reshape(-1, 1))[:, 1]
    p12 = platt(raw12); p1 = platt(raw1); dual = np.maximum(p12, p1)
    T12 = _youden_T(yt, p12); T1 = _youden_T(yt, p1); Td = _youden_T(yt, dual)

    matched, cross = (12, 1) if ds_gap == 12 else (1, 12)
    pm, pc = (p12, p1) if ds_gap == 12 else (p1, p12)
    Tm, Tc = (T12, T1) if ds_gap == 12 else (T1, T12)

    L = [f"### DATASET = heldout_test_{ds_gap}mo.sql  | test n={int(test.sum())} (pos {int(yt.sum())}, prev {yt.mean()*100:.2f}%)",
         f"{'model on this data':24s} {'AUROC':>7} {'AUPRC':>7} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6}"]

    def row(name, p, T):
        s, sp, pv, nv = _op(yt, p >= T)
        return f"{name:24s} {roc_auc_score(yt,p):7.3f} {average_precision_score(yt,p):7.3f} {s*100:6.1f} {sp*100:6.1f} {pv*100:6.1f} {nv*100:6.1f}"

    L.append(row(f"{matched}mo model (matched)", pm, Tm))
    L.append(row(f"{cross}mo model (cross)", pc, Tc))
    L.append(row("DUAL max(p12,p1)", dual, Td))
    s, sp, pv, nv = _op(yt, (pm >= Tm) | (pc >= Tc))
    L.append(f"{'DUAL OR@thresholds':24s} {'-':>7} {'-':>7} {s*100:6.1f} {sp*100:6.1f} {pv*100:6.1f} {nv*100:6.1f}")

    pos = yt == 1; cm = pm >= Tm; cc = pc >= Tc
    both = int((pos & cm & cc).sum()); om = int((pos & cm & ~cc).sum())
    oc = int((pos & ~cm & cc).sum()); none = int((pos & ~cm & ~cc).sum())
    L += [f"# coverage of {int(pos.sum())} cancers: both={both}  {matched}mo-only={om}  {cross}mo-only={oc}  missed-by-both={none}",
          f"#   -> dual OR catches {both+om+oc}/{int(pos.sum())} ({(both+om+oc)/max(pos.sum(),1)*100:.1f}%); "
          f"the {cross}mo model adds {oc} that the matched {matched}mo misses."]
    return "\n".join(L)


def main():
    print(f"{'='*78}\nDUAL-HORIZON HELD-OUT (both models per dataset) — lung 1:{NC_RATIO} {WINDOW}\n{'='*78}")
    gaps = [12, 1] if DATASET == "both" else [int(DATASET)]
    blocks = [eval_on_dataset(g) for g in gaps]
    report = ("\n\n".join(blocks))
    print("\n" + report)
    out = os.path.join(HERE, f"dual_heldout_{WINDOW}_1to{NC_RATIO}.txt")
    open(out, "w").write(report + "\n")
    print(f"\n-> saved {out}\nDUAL-HORIZON EVAL COMPLETE")


if __name__ == "__main__":
    main()
