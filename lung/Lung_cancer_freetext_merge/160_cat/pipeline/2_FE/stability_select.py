"""
Stability selection (k-fold)  -  Lung Cancer model
=====================================================
Picks which features actually DRIVE predictions - data-driven, no manual top-N.

We do NOT choose a feature count by hand. Instead we run a tree model over k CV folds, and
keep features that are CONSISTENTLY in the cumulative-99%-importance set (cumimp99) across
folds. A feature selected in >= MIN_FOLDS folds is "stable" (robust to the split); features
that only look important in one lucky fold are dropped.

Input  : ./output/{Astra|Nova}/{yr}yr_OFF/features_p005_{horizon}.parquet   (from build_features.py)
Outputs (same per-(horizon,window,arm) dir; run_v3.py passes it explicitly — see C.artifact_subdir):
    stable_features_{horizon}.csv          feature, folds_selected, mean_importance
    features_p005_{horizon}_stable.parquet the reduced matrix (split + stable features + cancer_class)

Run (after build_features.py):  python stability_select.py
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Stability selector is PINNED to LightGBM so the stable feature set is reproducible across machines.
# A silent LightGBM->RandomForest fallback would select a DIFFERENT feature set on a box without
# LightGBM, and (with name-only caching + GCS sharing) that wrong set could propagate to others.
# SELECTOR_NAME is recorded in select()'s output so the choice is always visible/auditable.
try:
    from lightgbm import LGBMClassifier
    SELECTOR_NAME = "LGBMClassifier"
    def _model():
        return LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=-1, verbose=-1)
except ImportError:
    # Allow an explicit opt-in to the RandomForest fallback; otherwise FAIL LOUD rather than silently
    # select on a different model than every other run.
    if os.environ.get("ALLOW_RF_SELECTOR", "0") not in ("1", "true", "yes", "on"):
        raise ImportError(
            "stability_select: LightGBM not installed and it is the PINNED selector. Install lightgbm, "
            "or set ALLOW_RF_SELECTOR=1 to deliberately use the RandomForest fallback (NOT comparable to "
            "LightGBM-selected runs — different stable feature set).")
    import warnings
    warnings.warn("stability_select: LightGBM unavailable -> RandomForest selector (ALLOW_RF_SELECTOR). "
                  "Stable set is NOT comparable to LightGBM-selected runs.", stacklevel=2)
    from sklearn.ensemble import RandomForestClassifier
    SELECTOR_NAME = "RandomForestClassifier(ALLOW_RF_SELECTOR)"
    def _model():
        return RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
                                      random_state=SEED, n_jobs=-1, class_weight="balanced")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # V3 root, for config + splits + artifacts
import config as C
import splits
import artifacts                            # atomic writes (no partial stable matrix on interruption)
HORIZONS = C.horizons_from_argv(sys.argv[1:])   # e.g. `stability_select.py 12mo`; hard-errors on a typo
N_FOLDS = C.N_FOLDS
MIN_FOLDS = C.MIN_FOLDS         # keep a feature if selected (in cumimp99) in >= this many folds
CUM_IMP = C.CUM_IMP            # per-fold cumulative-importance cutoff
SEED = C.RANDOM_STATE          # single source — the _model() factories + CV below reference this (resolved at call time)


def cumimp99_features(importances, feat_names, cutoff=CUM_IMP):
    """Features whose cumulative (descending) importance reaches `cutoff`."""
    order = np.argsort(importances)[::-1]
    cum = np.cumsum(importances[order]) / (importances.sum() + 1e-12)
    k = int(np.searchsorted(cum, cutoff)) + 1            # smallest set covering `cutoff`
    return [feat_names[i] for i in order[:k]]


def select(h, in_path=None, out_dir=None, yr=None):
    """Stability-select features for horizon `h`. `in_path`/`out_dir` override the default location;
    run_v3.py always passes the exact per-(horizon, window) dir. For a standalone call the lookback `yr`
    must be given (no year is hardcoded) — it resolves the default Astra/Nova/{yr}yr_OFF folder."""
    _sub = None
    if in_path is None or out_dir is None:                 # only needed to build a default path
        _yr = yr if yr is not None else int(getattr(C, "FE_YEARS_BEFORE", 0) or 0)
        if _yr <= 0:
            raise SystemExit("[stable] pass the lookback `yr` (or explicit in_path/out_dir); none is hardcoded.")
        _sub = C.artifact_subdir(h, _yr)
    fp = in_path or os.path.join(HERE, "output", _sub, f"features_p005_{h}.parquet")
    df = pd.read_parquet(fp)
    feat_cols = [c for c in df.columns if c not in ("patient_guid", "cancer_class")]
    # Leak-free: select features on the TRAIN patients only (the model's internal 10% test must not
    # influence which features are kept). Train guids = the SAME 80% the model trains on (build_features).
    sd = splits.load_required(h)   # canonical split (guid -> train/valid/test); must exist (make_split.py)
    _split_col = splits.clean_guid(df["patient_guid"]).map(splits.split_map(sd))   # per-row split label
    # Stamp-coverage guard: every matrix patient MUST be in the canonical split. A NaN here would later
    # map to neither train/valid/test and the patient would silently vanish from every split — fail loud.
    assert _split_col.notna().all(), (
        f"[{h}] stamp coverage gap: {int(_split_col.isna().sum())} of {len(_split_col)} matrix patients "
        f"are not in the canonical split — regenerate the split (make_split.py) or the FE matrix")
    is_train = (_split_col == "train").to_numpy()
    Xall = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype("float64")  # never nullable Int64
    med = Xall[is_train].median(numeric_only=True)          # TRAIN medians for the selector's impute
    X = Xall[is_train].fillna(med).fillna(0.0).reset_index(drop=True)   # TRAIN ONLY
    y = df["cancer_class"].astype(int).to_numpy()[is_train]
    print(f"[{h}] {X.shape[0]:,} TRAIN patients (of {len(df):,}; test fit-excluded) x {X.shape[1]:,} "
          f"features -> {N_FOLDS}-fold stability  [selector={SELECTOR_NAME}]")

    counts = Counter()
    imp_sum = np.zeros(len(feat_cols))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
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
    # RED-FLAG FORCE-KEEP (config.REDFLAG_FE): the individual lung-cancer red-flag categories (+ the lab-side
    # value categories Hb/platelets/CRP/Weight) are clinically critical but too sparse to clear cumimp99 — so
    # protect them from pruning. Union in any feature for a force-keep category (prefix "<cat>_") + the
    # g_redflag_* composite, even if selected in < MIN_FOLDS folds. (build_features is the single source of
    # the curated list; imported lazily only when the flag is on.)
    if getattr(C, "REDFLAG_FE", False):
        import build_features as _bf
        _keep_prefixes = tuple(f"{c}_" for c in _bf.REDFLAG_FORCE_KEEP_CATS)
        _forced = [f for f in feat_cols
                   if (f.startswith(_keep_prefixes) or f.startswith("g_redflag_")) and f not in set(stable)]
        if _forced:
            stable = stable + sorted(_forced, key=lambda f: -mean_imp.get(f, 0.0))
            print(f"[{h}] REDFLAG_FE: force-kept {len(_forced)} red-flag/lab-side feature(s) "
                  f"(would have been pruned) -> {len(stable):,} stable total")
    out_dir = out_dir or os.path.join(HERE, "output", _sub)
    os.makedirs(out_dir, exist_ok=True)
    _flist = pd.DataFrame({"feature": stable,
                           "folds_selected": [counts[f] for f in stable],
                           "mean_importance": [mean_imp[f] for f in stable]})
    artifacts.atomic_write(os.path.join(out_dir, f"stable_features_{h}.csv"),
                           lambda t: _flist.to_csv(t, index=False))
    out = df[["patient_guid"] + stable + ["cancer_class"]].copy()
    out.insert(1, "split", _split_col.to_numpy())               # stamp the canonical split (travels WITH the data)
    artifacts.atomic_write(os.path.join(out_dir, f"features_p005_{h}_stable.parquet"),
                           lambda t: out.to_parquet(t, index=False))
    print(f"      -> stable_features_{h}.csv  +  features_p005_{h}_stable.parquet\n")


def main():
    # Window-aware + no hardcode: explicit digits on argv (e.g. `stability_select.py 12mo 10`), else
    # auto-discover the windows that have a category map for this horizon. One stable set per (horizon, window).
    argv_years = [int(a) for a in sys.argv[1:] if a.isdigit()]
    for h in HORIZONS:
        windows = argv_years or (C.categorized_windows(h) if getattr(C, "CATEGORIZED", False) else [])
        if not windows:
            print(f"[stable] {h}: no lookback window selected/found — skip")
            continue
        for yr in windows:
            select(h, yr=yr)


if __name__ == "__main__":
    main()
