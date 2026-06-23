"""
Dual-horizon held-out evaluation — per held-out DATASET, score it with BOTH models and combine.

For each held-out scenario we run BOTH the 12mo-trained and 1mo-trained models on the SAME dataset,
then combine per patient. Primary dual decision = max(calibrated p12, p1) (a proper score -> one
controllable threshold), with OR@thresholds kept only as a reference (independent thresholds ->
uncontrolled operating point). Mirrors the single-horizon runs (12mo-on-12mo, 1mo-on-1mo) but adds
the other ("cross") model and the dual on each dataset:

  DATASET heldout_test_12mo.sql  (events cut off 12 months pre-dx):  12mo matched | 1mo cross | DUAL
  DATASET heldout_test_1mo.sql   (events cut off  1 month  pre-dx):  1mo matched  | 12mo cross | DUAL

Both models read the SAME dataset; each CancerPredictor reindexes it to its own trained feature set.
Reuses heldout_features_{window}.csv from the single-horizon runs (no re-FE). Per dataset: shared
30% calib / 70% test split, Platt per model on calib, reported on test:
  - matched / cross / DUAL max-prob : AUROC + Sens/Spec/PPV/NPV @Youden
  - DUAL OR@thresholds (reference)  : flag if either model exceeds its own Youden threshold
  - true-cancer coverage            : caught by both / matched-only / cross-only / neither
  - max-prob operating curve        : Spec/PPV/alerts-per-1000 at fixed Sens (80/85/90/95%), NO target
                                       fixed -> matched single vs DUAL, so the operating point is chosen later.
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
DATASET = os.environ.get("DATASET", "both")
SENS_TARGETS = [0.80, 0.85, 0.90, 0.95]

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "2_FE"))
sys.path.insert(0, os.path.join(ROOT, "3_Modeling"))
from predict_unseen import CancerPredictor


def _model_path(gap):
    return os.path.join(ROOT, f"{gap}mo_1to{NC_RATIO}", "lookback", WINDOW, f"model_{WINDOW}_1to{NC_RATIO}.joblib")


def _raw(df, model_gap):
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


def _op_at_sens(y, p, target):
    """Operating point with sensitivity >= target (highest threshold that still reaches it)."""
    fpr, tpr, thr = roc_curve(y, p)
    ix = np.where(tpr >= target)[0]
    if len(ix) == 0:
        return None
    T = thr[ix[0]]
    pred = p >= T
    sens, spec, ppv, npv = _op(y, pred)
    return dict(T=T, sens=sens, spec=spec, ppv=ppv, npv=npv, n_alert=int(pred.sum()))


def eval_on_dataset(ds_gap):
    feat = os.path.join(ROOT, f"{ds_gap}mo_1to{NC_RATIO}", "lookback", WINDOW, f"heldout_features_{WINDOW}.csv")
    assert os.path.exists(feat), f"missing {feat} (run the {ds_gap}mo 1:{NC_RATIO} held-out first)"
    df = pd.read_csv(feat, low_memory=False)
    raw12, y = _raw(df, 12)
    raw1, y1 = _raw(df, 1)
    assert (y == y1).all(), "label mismatch scoring the same dataset twice"

    rng = np.random.RandomState(42)
    idx = np.arange(len(y)); cal = np.zeros(len(y), bool)
    for c in (0, 1):
        ci = idx[y == c]; rng.shuffle(ci); cal[ci[:int(round(CALIB_FRAC * len(ci)))]] = True
    test = ~cal; yt = y[test]; n = int(test.sum())

    def platt(raw):
        lr = LogisticRegression(C=1e12, solver="lbfgs").fit(raw[cal].reshape(-1, 1), y[cal])
        return lr.predict_proba(raw[test].reshape(-1, 1))[:, 1]
    p12 = platt(raw12); p1 = platt(raw1)
    mx = np.maximum(p12, p1)                       # max-prob (recall-greedy)
    mn = 0.5 * (p12 + p1)                          # mean (tempered)
    # trained stacker: logistic regression on the two RAW scores -> learns weights + own calibration
    stk = LogisticRegression(C=1.0, solver="lbfgs").fit(np.column_stack([raw12[cal], raw1[cal]]), y[cal])
    ps = stk.predict_proba(np.column_stack([raw12[test], raw1[test]]))[:, 1]

    T12 = _youden_T(yt, p12); T1 = _youden_T(yt, p1)
    matched, cross = (12, 1) if ds_gap == 12 else (1, 12)
    pm, pc = (p12, p1) if ds_gap == 12 else (p1, p12)
    Tm, Tc = (T12, T1) if ds_gap == 12 else (T1, T12)
    w = stk.coef_[0]
    sw = f"[12mo={w[0]:+.2f}, 1mo={w[1]:+.2f}]"   # stacker weights -> which model it trusts on this data

    L = [f"### DATASET = heldout_test_{ds_gap}mo.sql  | test n={n} (pos {int(yt.sum())}, prev {yt.mean()*100:.2f}%)",
         f"{'model / combine':24s} {'AUROC':>7} {'AUPRC':>7} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6}  (@Youden)"]

    def row(name, p):
        T = _youden_T(yt, p); s, sp, pv, nv = _op(yt, p >= T)
        return f"{name:24s} {roc_auc_score(yt,p):7.3f} {average_precision_score(yt,p):7.3f} {s*100:6.1f} {sp*100:6.1f} {pv*100:6.1f} {nv*100:6.1f}"

    L.append(row(f"{matched}mo model (matched)", pm))
    L.append(row(f"{cross}mo model (cross)", pc))
    L.append(row("DUAL max(p12,p1)", mx))
    L.append(row("DUAL mean(p12,p1)", mn))
    L.append(row("DUAL stacker (LR)", ps))
    s, sp, pv, nv = _op(yt, (pm >= Tm) | (pc >= Tc))
    L.append(f"{'DUAL OR (reference)':24s} {'-':>7} {'-':>7} {s*100:6.1f} {sp*100:6.1f} {pv*100:6.1f} {nv*100:6.1f}")
    L.append(f"# stacker weights {sw} (larger = more trusted on this dataset)")

    pos = yt == 1; cm = pm >= Tm; cc = pc >= Tc
    both = int((pos & cm & cc).sum()); om = int((pos & cm & ~cc).sum())
    oc = int((pos & ~cm & cc).sum()); none = int((pos & ~cm & ~cc).sum())
    L += [f"# coverage of {int(pos.sum())} cancers: both={both}  {matched}mo-only={om}  {cross}mo-only={oc}  missed-by-both={none}"]

    # operating curve, NO target fixed: Spec at each recall, comparing the combine rules vs matched
    L += ["", "# Spec @ fixed Sens (test slice) — which combine wins at each recall (higher=better):",
          f"{'Sens':>5} | {'matched':>8} {'max':>7} {'mean':>7} {'stacker':>8}"]
    methods = [pm, mx, mn, ps]
    for ts in SENS_TARGETS:
        cells = []
        for p in methods:
            r = _op_at_sens(yt, p, ts)
            cells.append(f"{r['spec']*100:.1f}" if r else "-")
        L.append(f"{int(ts*100):>4}% | {cells[0]:>8} {cells[1]:>7} {cells[2]:>7} {cells[3]:>8}")
    return "\n".join(L)


def main():
    print(f"{'='*80}\nDUAL-HORIZON HELD-OUT (both models per dataset) — lung 1:{NC_RATIO} {WINDOW}\n{'='*80}")
    gaps = [12, 1] if DATASET == "both" else [int(DATASET)]
    report = "\n\n".join(eval_on_dataset(g) for g in gaps)
    print("\n" + report)
    out = os.path.join(HERE, f"dual_heldout_{WINDOW}_1to{NC_RATIO}.txt")
    open(out, "w").write(report + "\n")
    print(f"\n-> saved {out}\nDUAL-HORIZON EVAL COMPLETE")


if __name__ == "__main__":
    main()
