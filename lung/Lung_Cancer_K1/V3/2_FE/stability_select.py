"""
Stability selection (k-fold)  -  Lung Cancer model
=====================================================
Picks which features actually DRIVE predictions - data-driven, no manual top-N.

We do NOT choose a feature count by hand. Instead we run a tree model over k CV folds, and
keep features that are CONSISTENTLY in the cumulative-99%-importance set (cumimp99) across
folds. A feature selected in >= MIN_FOLDS folds is "stable" (robust to the split); features
that only look important in one lucky fold are dropped.

Input  : ./output/{horizon}/features_p005_{horizon}.parquet   (from build_features.py)
Outputs (./output/{horizon}/):
    stable_features_{horizon}.csv          feature, folds_selected, mean_importance
    features_p005_{horizon}_stable.parquet the reduced matrix (split + stable features + cancer_class)

Run (after build_features.py):  python stability_select.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier
    def _model():
        return LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=-1, verbose=-1)
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    def _model():
        return RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
                                      random_state=42, n_jobs=-1, class_weight="balanced")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # V3 root, for config + splits
import config as C
import splits
HORIZONS = list(C.HORIZONS)
if len(sys.argv) > 1:           # e.g. `python stability_select.py 12mo` to run a single horizon
    HORIZONS = [a for a in sys.argv[1:] if a in ("12mo", "1mo")] or HORIZONS
N_FOLDS = C.N_FOLDS
MIN_FOLDS = C.MIN_FOLDS         # keep a feature if selected (in cumimp99) in >= this many folds
CUM_IMP = C.CUM_IMP            # per-fold cumulative-importance cutoff


def cumimp99_features(importances, feat_names, cutoff=CUM_IMP):
    """Features whose cumulative (descending) importance reaches `cutoff`."""
    order = np.argsort(importances)[::-1]
    cum = np.cumsum(importances[order]) / (importances.sum() + 1e-12)
    k = int(np.searchsorted(cum, cutoff)) + 1            # smallest set covering `cutoff`
    return [feat_names[i] for i in order[:k]]


def select(h, in_path=None, out_dir=None):
    """Stability-select features for horizon `h`. `in_path`/`out_dir` override the default
    output/{h}/ location (the runner passes per-lookback-window dirs)."""
    fp = in_path or os.path.join(HERE, "output", h, f"features_p005_{h}.parquet")
    df = pd.read_parquet(fp)
    feat_cols = [c for c in df.columns if c not in ("patient_guid", "cancer_class")]
    # Leak-free: select features on the TRAIN patients only (the model's internal 10% test must not
    # influence which features are kept). Train guids = the SAME 80% the model trains on (build_features).
    sd = splits.load_or_make(h, df["patient_guid"], df["cancer_class"])   # canonical split (guid -> train/valid/test)
    _split_col = splits.clean_guid(df["patient_guid"]).map(splits.split_map(sd))   # per-row split label
    is_train = (_split_col == "train").to_numpy()
    Xall = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype("float64")  # never nullable Int64
    med = Xall[is_train].median(numeric_only=True)          # TRAIN medians for the selector's impute
    X = Xall[is_train].fillna(med).fillna(0.0).reset_index(drop=True)   # TRAIN ONLY
    y = df["cancer_class"].astype(int).to_numpy()[is_train]
    print(f"[{h}] {X.shape[0]:,} TRAIN patients (of {len(df):,}; test fit-excluded) x {X.shape[1]:,} "
          f"features -> {N_FOLDS}-fold stability")

    counts = Counter()
    imp_sum = np.zeros(len(feat_cols))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for i, (tr, _) in enumerate(cv.split(X, y), 1):
        m = _model()
        m.fit(X.iloc[tr].values, y[tr])
        imp = np.asarray(m.feature_importances_, dtype=float)
        imp_sum += imp
        sel = cumimp99_features(imp, feat_cols)
        counts.update(sel)
        print(f"  fold {i}: {len(sel):,} features in cumimp{int(CUM_IMP*100)}")

    mean_imp = dict(zip(feat_cols, imp_sum / N_FOLDS))
    stable = sorted([f for f, c in counts.items() if c >= MIN_FOLDS],
                    key=lambda f: -mean_imp[f])
    print(f"[{h}] stable (>= {MIN_FOLDS}/{N_FOLDS} folds): {len(stable):,} features")

    out_dir = out_dir or os.path.join(HERE, "output", h)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"feature": stable,
                  "folds_selected": [counts[f] for f in stable],
                  "mean_importance": [mean_imp[f] for f in stable]}
                 ).to_csv(os.path.join(out_dir, f"stable_features_{h}.csv"), index=False)
    out = df[["patient_guid"] + stable + ["cancer_class"]].copy()
    out.insert(1, "split", _split_col.to_numpy())               # stamp the canonical split (travels WITH the data)
    out.to_parquet(os.path.join(out_dir, f"features_p005_{h}_stable.parquet"), index=False)
    print(f"      -> stable_features_{h}.csv  +  features_p005_{h}_stable.parquet\n")


def main():
    for h in HORIZONS:
        select(h)


if __name__ == "__main__":
    main()
