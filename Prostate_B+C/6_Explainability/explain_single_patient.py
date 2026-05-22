"""
Per-patient SHAP explanation — single-row API for production inference.

Wraps the same logic as `3_Modeling/2_predict_unseen.py::compute_shap_top_factors`
but exposes a one-patient function that the production inference path can call.

Returns the top-N features (by |SHAP value| on logit scale) that drove the
model's score for this patient, with both the feature value and the signed
SHAP contribution.

Usage:
    from explain_single_patient import explain
    factors = explain(feature_vector_dict, window="12mo", top_n=5)
    # factors = [
    #   {"feature": "LAB_PSA_max", "value": 12.3, "shap": +0.41},
    #   {"feature": "PROST_AGEX_OVER_80", "value": 1.0, "shap": +0.28},
    #   ...
    # ]
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "3_Modeling" / "results"

# Required so joblib can unpickle CalibratorWrapper from saved configs.
sys.path.insert(0, str(PROJECT_ROOT / "3_Modeling"))
from _calibrator import CalibratorWrapper  # noqa: F401

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _align_feature_vector(feature_vector, selected_features):
    """Coerce dict/Series to a 1-row DataFrame aligned to the model's selected_features."""
    if isinstance(feature_vector, dict):
        s = pd.Series(feature_vector)
    elif isinstance(feature_vector, pd.Series):
        s = feature_vector
    else:
        raise TypeError(f"feature_vector must be dict or Series, got {type(feature_vector)}")
    return pd.DataFrame(
        [{f: s.get(f, 0) for f in selected_features}],
        columns=selected_features,
    ).fillna(0)


def explain(feature_vector, window, top_n=5, results_root=None):
    """Return top-N SHAP factors for ONE patient at the given window.

    Averages TreeExplainer SHAP values across all base models AND seeds. Values
    are on the model's logit scale — useful for RANKING features, not for
    interpreting as direct probability deltas.

    Parameters
    ----------
    feature_vector : dict or pd.Series
        Patient features keyed by feature name. Missing keys filled with 0.
    window : str
        '1mo' / '3mo' / '6mo' / '12mo' — which model to explain.
    top_n : int, default 5
        Number of top features to return.
    results_root : Path, optional
        Override for the modeling results dir. Defaults to
        PROJECT_ROOT/3_Modeling/results.

    Returns
    -------
    list[dict] | None
        [{"feature": str, "value": float, "shap": float}, ...] sorted by |shap| desc.
        Returns None if SHAP is unavailable.
    """
    if not HAS_SHAP:
        logger.warning("shap package not installed — no per-patient explanation available")
        return None

    if results_root is None:
        results_root = DEFAULT_RESULTS_ROOT
    model_dir = Path(results_root) / "1_training" / window / "saved_models"
    if not model_dir.exists():
        raise FileNotFoundError(f"No saved models for {window} at {model_dir}")

    cfg = joblib.load(model_dir / "config.json")
    seeds = cfg.get("seeds", [42])
    ensemble_models = cfg["ensemble_models"]
    selected = cfg["selected_features"]

    X = _align_feature_vector(feature_vector, selected)
    n_feats = X.shape[1]
    feature_names = list(X.columns)

    # Average SHAP across all base models AND all seeds (logit scale).
    agg = np.zeros((1, n_feats), dtype=float)
    n_used = 0
    for seed in seeds:
        seed_dir = model_dir / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        for name in ensemble_models:
            mp = seed_dir / f"{name}_model.pkl"
            if not mp.exists():
                continue
            try:
                m = joblib.load(mp)
                sv = shap.TreeExplainer(m).shap_values(X)
                # LightGBM (some versions) returns [neg, pos]; XGB/CatBoost return one array.
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) >= 2 else sv[0]
                sv = np.asarray(sv)
                if sv.shape == agg.shape:
                    agg += sv
                    n_used += 1
            except Exception as e:
                logger.warning(f"  SHAP failed for {name} seed={seed}: {e}")
    if n_used == 0:
        logger.warning(f"  SHAP: no base models contributed for {window}")
        return None
    agg /= n_used

    Xv = X.values
    idx = np.argsort(-np.abs(agg[0]))[:top_n]
    return [
        {
            "feature": feature_names[j],
            "value":   float(Xv[0, j]),
            "shap":    float(agg[0, j]),
        }
        for j in idx
    ]
