"""
SHAP explainability (multi-seed + per-band calibration).

Runs SHAP on seed-42 XGBoost (single base model). Rankings transfer well to the
stacked ensemble (correlation ~0.95+); use that as the explanation surface.

Optional flags:
  --by-band        Compute SHAP separately per age band (<55, 55-65, 65-75, 75+) and
                   emit a Jaccard stability score across bands.
  --faithfulness   Perturbation test: replace each patient's top-K SHAP features with
                   the cohort median and measure change in predicted probability.
                   Higher mean |Δ| = more faithful attribution.

Outputs:
  results/shap_summary_{window}.csv             — per-feature mean |SHAP|
  results/shap_per_patient_{window}.csv         — per-patient top contributors
  results/shap_summary_{window}.png             — beeswarm
  results/shap_by_band_{window}.csv             — per-band feature ranking (--by-band)
  results/shap_band_stability_{window}.csv      — Jaccard top-K across bands (--by-band)
  results/shap_faithfulness_{window}.csv        — per-patient prediction shift (--faithfulness)

Usage:
    python 1_explain_predictions.py --window 12mo
    python 1_explain_predictions.py --window 12mo --by-band --faithfulness
    python 1_explain_predictions.py --window 12mo --n-samples 1000  # subsample
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_RESULTS = PROJECT_ROOT / "3_Modeling" / "results" / "1_training"
CLEANUP_RESULTS = PROJECT_ROOT / "2_Feature_Engineering" / "results" / "5_cleanup"
OUT_DIR = SCRIPT_DIR / "results"

# Required so joblib can unpickle CalibratorWrapper
sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
from _calibrator import CalibratorWrapper  # noqa: F401

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─── Data + model loading ──────────────────────────────────────────

def load_window(window, seed):
    """Load cleaned feature matrix, selected features, primary XGB model, and cfg.

    Returns age bands as labelled strings — derived from continuous AGE_AT_INDEX if
    present, otherwise reconstructed from the binary AGE_over_50/65/75 flags that
    survive 5_cleanup. Band edges match the modeling script's: <55, 55-65, 65-75, 75+.
    """
    saved_dir = TRAIN_RESULTS / window / "saved_models"
    seed_dir = saved_dir / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"No seed_{seed}/ in {saved_dir}. Multi-seed format expected.")

    cfg = joblib.load(saved_dir / "config.json")
    selected = cfg["selected_features"]
    xgb_model = joblib.load(seed_dir / "xgboost_model.pkl")

    fm_path = CLEANUP_RESULTS / window / f"feature_matrix_clean_{window}.parquet"
    if not fm_path.exists():
        fm_path = CLEANUP_RESULTS / window / f"feature_matrix_clean_{window}.csv"
    if not fm_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {fm_path}")
    fm = pd.read_parquet(fm_path) if fm_path.suffix == ".parquet" else pd.read_csv(fm_path, index_col=0)

    for f in selected:
        if f not in fm.columns:
            fm[f] = 0
    X = fm[selected].fillna(0)

    if "AGE_AT_INDEX" in fm.columns:
        ages_num = fm["AGE_AT_INDEX"].values
        bands = pd.cut(ages_num, bins=BAND_BINS, labels=BAND_LABELS, right=False).astype(str)
        bands = pd.Series(bands, index=fm.index)
    else:
        # Reconstruct from binary flags emitted by FE.
        a50 = fm.get("AGE_over_50", pd.Series(0, index=fm.index))
        a65 = fm.get("AGE_over_65", pd.Series(0, index=fm.index))
        a75 = fm.get("AGE_over_75", pd.Series(0, index=fm.index))
        bands = pd.Series(np.where(a75 == 1, "75+",
                          np.where(a65 == 1, "65-75",
                          np.where(a50 == 1, "55-65", "<55"))), index=fm.index)
    return X, bands, xgb_model, cfg, saved_dir


# ─── SHAP computation ──────────────────────────────────────────────

def shap_for_model(model, X):
    """SHAP values for XGBoost/LGBM/CatBoost. XGBoost 3.x has a JSON-array threshold
    that shap.TreeExplainer can't parse, so we use XGBoost's native pred_contribs.
    For others, shap.TreeExplainer works fine.
    """
    cls_name = type(model).__name__.lower()
    if 'xgb' in cls_name:
        import xgboost as xgb
        booster = model.get_booster()
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        contribs = booster.predict(dm, pred_contribs=True)
        # drop bias column (last)
        return np.asarray(contribs[:, :-1])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) >= 2 else sv[0]
    return np.asarray(sv)


# ─── Summaries + per-patient ───────────────────────────────────────

def write_summary(shap_values, X, window):
    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(OUT_DIR / f"shap_summary_{window}.csv", index=False)
    logger.info(f"  Saved {OUT_DIR}/shap_summary_{window}.csv")
    return summary


def write_per_patient(shap_values, X, window, top_k=10):
    feat_idx_top = np.argsort(-np.abs(shap_values), axis=1)[:, :top_k]
    rows = []
    for i, pid in enumerate(X.index):
        rec = {"PATIENT_GUID": pid}
        for k in range(top_k):
            f_idx = feat_idx_top[i, k]
            rec[f"feat_{k+1}"] = X.columns[f_idx]
            rec[f"shap_{k+1}"] = float(shap_values[i, f_idx])
            rec[f"value_{k+1}"] = float(X.iloc[i, f_idx])
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"shap_per_patient_{window}.csv", index=False)
    logger.info(f"  Saved {OUT_DIR}/shap_per_patient_{window}.csv")


def write_summary_plot(shap_values, X, window):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X, max_display=25, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"shap_summary_{window}.png", dpi=120)
        plt.close()
        logger.info(f"  Saved {OUT_DIR}/shap_summary_{window}.png")
    except Exception as e:
        logger.warning(f"  Plot skipped: {e}")


# ─── Per-band SHAP + stability metric ──────────────────────────────

BAND_BINS = [0, 55, 65, 75, 200]
BAND_LABELS = ["<55", "55-65", "65-75", "75+"]


def shap_by_age_band(shap_values, X, bands, window):
    """Per-band feature rankings + Jaccard stability across bands.

    `bands` is a Series of band labels (e.g., '<55', '55-65', ...) aligned with X.
    """
    bands = np.asarray(bands)

    per_band = {}
    for band in BAND_LABELS:
        mask = (bands == band)
        n = int(mask.sum())
        if n < 50:
            logger.warning(f"  Band {band}: only {n} patients — skipping")
            continue
        per_band[band] = pd.Series(np.abs(shap_values[mask]).mean(axis=0), index=X.columns)
        logger.info(f"  Band {band}: {n} patients ✓")

    if not per_band:
        logger.warning("  No bands had enough patients for per-band SHAP")
        return None

    band_df = pd.DataFrame(per_band)
    band_df.index.name = "feature"
    band_df["mean"] = band_df.mean(axis=1)
    band_df = band_df.sort_values("mean", ascending=False)
    band_df.to_csv(OUT_DIR / f"shap_by_band_{window}.csv")
    logger.info(f"  Saved {OUT_DIR}/shap_by_band_{window}.csv")

    bands_present = [b for b in BAND_LABELS if b in per_band]
    if len(bands_present) < 2:
        logger.warning(f"  Only {len(bands_present)} band(s) usable — stability metric skipped")
        return band_df

    # Jaccard stability across bands at top-K
    from itertools import combinations
    rows = []
    for k in (5, 10, 20):
        sets_k = {b: set(per_band[b].nlargest(k).index) for b in bands_present}
        for a, b in combinations(bands_present, 2):
            inter = len(sets_k[a] & sets_k[b])
            uni = len(sets_k[a] | sets_k[b])
            rows.append({
                "top_k": k, "band_a": a, "band_b": b,
                "intersection": inter, "union": uni,
                "jaccard": inter / uni if uni else np.nan,
            })
    stab_df = pd.DataFrame(rows)
    stab_df.to_csv(OUT_DIR / f"shap_band_stability_{window}.csv", index=False)
    logger.info(f"  Saved {OUT_DIR}/shap_band_stability_{window}.csv")

    logger.info("  Band stability (Jaccard, higher = more stable):")
    for k in (5, 10, 20):
        sub = stab_df[stab_df["top_k"] == k]
        if len(sub):
            logger.info(f"    top-{k:2d}: mean={sub['jaccard'].mean():.3f} "
                        f"min={sub['jaccard'].min():.3f}  ({len(sub)} band pairs)")
    return band_df


# ─── Faithfulness test (perturbation) ──────────────────────────────

def faithfulness_test(model, X, shap_values, window, k_list=(1, 3, 5, 10), n_eval=2000, seed=42):
    """For each patient, replace their top-K SHAP features with the cohort median
    and measure |Δ predicted_proba|.

    Median (not zero) is the right baseline because zero is itself a meaningful value
    in this schema (e.g., LAB_PSA_count=0 means "never tested", not "remove influence").
    Median ≈ "what would the prediction be if this feature carried no patient-specific info".

    Higher mean |Δproba| at small K = SHAP is identifying features that genuinely drive
    the prediction (faithful). Near-zero |Δproba| means SHAP is misattributing.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    if n_eval is not None and n > n_eval:
        idx = rng.choice(n, size=n_eval, replace=False)
        X_eval = X.iloc[idx]
        shap_eval = shap_values[idx]
    else:
        X_eval = X
        shap_eval = shap_values

    # Cohort medians (per feature) computed from the full X, not the eval subset.
    medians = X.median(axis=0).values

    base_proba = model.predict_proba(X_eval)[:, 1]

    rows = []
    for k in k_list:
        top_idx = np.argsort(-np.abs(shap_eval), axis=1)[:, :k]
        X_pert = X_eval.values.copy()
        for i in range(len(X_eval)):
            for j in top_idx[i]:
                X_pert[i, j] = medians[j]
        pert_proba = model.predict_proba(pd.DataFrame(X_pert, columns=X_eval.columns,
                                                     index=X_eval.index))[:, 1]
        delta = pert_proba - base_proba
        rows.append({
            "k_top_perturbed": k,
            "mean_abs_delta": float(np.abs(delta).mean()),
            "median_abs_delta": float(np.median(np.abs(delta))),
            "p95_abs_delta": float(np.quantile(np.abs(delta), 0.95)),
            "mean_signed_delta": float(delta.mean()),
            "n_patients": int(len(X_eval)),
        })
        logger.info(f"  Faithfulness top-{k:2d}: mean|Δproba|={rows[-1]['mean_abs_delta']:.4f} "
                    f"p95={rows[-1]['p95_abs_delta']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"shap_faithfulness_{window}.csv", index=False)
    logger.info(f"  Saved {OUT_DIR}/shap_faithfulness_{window}.csv")
    return df


# ─── Main entry ────────────────────────────────────────────────────

def explain(window, n_samples=None, seed=42, by_band=False, faithfulness=False):
    if not HAS_SHAP:
        raise RuntimeError("shap not installed. pip install shap")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, bands, xgb_model, cfg, saved_dir = load_window(window, seed)

    if n_samples and len(X) > n_samples:
        sample_idx = X.sample(n=n_samples, random_state=42).index
        X = X.loc[sample_idx]
        bands = bands.loc[sample_idx]
        logger.info(f"  Subsampled to {len(X):,} patients")

    logger.info(f"  SHAP on seed-{seed} XGBoost: {len(X):,} × {X.shape[1]} features")
    shap_values = shap_for_model(xgb_model, X)

    summary = write_summary(shap_values, X, window)
    write_per_patient(shap_values, X, window)
    write_summary_plot(shap_values, X, window)

    if by_band:
        logger.info("\n  ── Per-band SHAP analysis ──")
        shap_by_age_band(shap_values, X, bands, window)

    if faithfulness:
        logger.info("\n  ── Faithfulness (perturbation) test ──")
        faithfulness_test(xgb_model, X, shap_values, window)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", required=True, choices=["1mo", "12mo"])
    p.add_argument("--n-samples", type=int, default=None,
                    help="Subsample for speed (default: all patients)")
    p.add_argument("--seed", type=int, default=42, help="Which seed to explain (default: 42)")
    p.add_argument("--by-band", action="store_true",
                    help="Per-age-band SHAP + Jaccard stability metric")
    p.add_argument("--faithfulness", action="store_true",
                    help="Perturbation test: replace top-K features with median, measure |Δproba|")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = explain(
        args.window,
        n_samples=args.n_samples,
        seed=args.seed,
        by_band=args.by_band,
        faithfulness=args.faithfulness,
    )
    logger.info(f"\n  Top 15 features by mean |SHAP|:")
    for _, row in summary.head(15).iterrows():
        logger.info(f"    {row['feature']:50s}  {row['mean_abs_shap']:.4f}")


if __name__ == "__main__":
    main()
