"""
Batch scorer — score a whole feature matrix with a trained B+C model (the batch
equivalent of predict_single_patient.predict). Used to score the 300K holdout (and
to regenerate test scores). Output feeds holdout_eval.py.

Replicates the training/inference path exactly:
  per seed: base models -> meta-learner (stacked) ; mean over seeds = raw_proba ;
  per-age-band isotonic calibration -> risk_proba.

verify mode: re-scores a model's OWN test patients and checks the result matches the
saved predictions_{window}.csv (proves the batch path == the training path).
"""
import sys, joblib
import numpy as np
import pandas as pd


def load_artifacts(results_window_dir):
    # joblib pickles reference _calibrator.CalibratorWrapper — that module lives in
    # 3_Modeling/ (3 levels up from results_*/1_training/{window}). Put it on the path.
    import os, sys
    mod3 = os.path.abspath(os.path.join(results_window_dir, '..', '..', '..'))
    if mod3 not in sys.path:
        sys.path.insert(0, mod3)
    md = f"{results_window_dir}/saved_models"
    cfg = joblib.load(f"{md}/config.json")
    seeds = cfg.get("seeds", [42]); ens = cfg["ensemble_models"]
    arts = {}
    for s in seeds:
        sd = f"{md}/seed_{s}"
        arts[s] = {
            "models": {n: joblib.load(f"{sd}/{n}_model.pkl") for n in ens},
            "meta": joblib.load(f"{sd}/meta_learner.pkl"),
        }
    return cfg, arts


def score_matrix(df, cfg, arts):
    """Return calibrated risk_proba for every row of df (reindexed to selected features)."""
    selected = cfg["selected_features"]
    X = df.reindex(columns=selected).fillna(0)
    ens = cfg["ensemble_models"]
    per_seed = []
    for s, a in arts.items():
        base = np.column_stack([a["models"][n].predict_proba(X)[:, 1] for n in ens])
        per_seed.append(a["meta"].predict_proba(base)[:, 1])
    raw = np.mean(per_seed, axis=0)
    # per-band isotonic calibration (vectorised by band group)
    bb = np.asarray(cfg["band_bins"]); bl = cfg["band_labels"]
    cals = cfg["calibrators_per_band"]; gcal = cfg["global_calibrator"]
    age = X["AGE_AT_INDEX"].values if "AGE_AT_INDEX" in X.columns else np.full(len(X), 65.0)
    bands = np.minimum(np.searchsorted(bb[1:], age, side="right"), len(bl) - 1)
    out = np.empty(len(X), dtype=float)
    for bi in np.unique(bands):
        m = bands == bi
        cal = cals.get(bl[bi], gcal)
        out[m] = cal.transform(raw[m])
    return out


def verify(results_window_dir, clean_matrix_path, predictions_csv):
    """Re-score the saved test patients; compare to predictions_{window}.csv."""
    cfg, arts = load_artifacts(results_window_dir)
    fm = pd.read_parquet(clean_matrix_path)
    pred = pd.read_csv(predictions_csv)
    # auto-detect guid + saved-score columns
    guid_col = next((c for c in pred.columns if 'guid' in c.lower() or c.lower() in ('patient_guid', 'index')), pred.columns[0])
    # prefer the CALIBRATED proba (score_matrix returns calibrated), not *_raw
    score_col = next((c for c in pred.columns if ('proba' in c.lower() or 'risk' in c.lower()) and 'raw' not in c.lower()), None) \
                or next((c for c in pred.columns if 'proba' in c.lower() or 'score' in c.lower()), None)
    g = pred[guid_col].astype(str).values
    sub = fm[fm.index.astype(str).isin(g)]
    sub = sub.loc[[x for x in g if x in set(sub.index.astype(str))]] if guid_col != 'index' else sub
    mine = score_matrix(sub, cfg, arts)
    saved = pred.set_index(pred[guid_col].astype(str)).loc[sub.index.astype(str)][score_col].values if score_col else None
    print(f"  scored {len(sub)} test patients; my range [{mine.min():.3f},{mine.max():.3f}]")
    if saved is not None:
        diff = np.abs(mine - saved)
        print(f"  vs saved {score_col}: max|Δ|={diff.max():.4f} mean|Δ|={diff.mean():.5f} corr={np.corrcoef(mine,saved)[0,1]:.4f}")
        print("  ✓ MATCHES saved predictions" if diff.max() < 1e-3 else "  ⚠ differs — investigate")


if __name__ == '__main__':
    # verify <results_window_dir> <clean_matrix.parquet> <predictions.csv>
    verify(sys.argv[1], sys.argv[2], sys.argv[3])
