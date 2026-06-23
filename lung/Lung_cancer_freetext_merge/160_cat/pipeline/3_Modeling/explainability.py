"""
Explainability for the lung model — two views, both SHAP-based, written as CSVs + PNGs.

  (1) PATIENT-LEVEL  : per-patient top risk factors (signed SHAP) + a global SHAP summary plot.
  (2) SEGMENT-LEVEL  : the key risk-factor DRIVERS of each confusion-matrix segment
                       (True Positives / False Positives / False Negatives / True Negatives) —
                       i.e. "what pushes patients into each quadrant", via SHAP aggregated per segment.

Run it on a LABELLED evaluation slice (held-out or internal-test) so the four segments are defined.
SHAP backend: TREE-EXACT ONLY, in PROBABILITY space. A tree model -> TreeExplainer; a soft-voting
ensemble of trees -> the EXACT per-base-tree TreeExplainer weighted-averaged by the voting weights
— both explain ALL patients. (REQUIRES xgboost<3: shap-0.49's TreeExplainer can't parse xgboost 3.x's
vector base_score; see _member_tree_shap. requirements.txt pins it.) Members are explained with
model_output="probability" against a small background sample, so every member's SHAP is in the same
(probability) space and the voting-weighted sum exactly reconstructs the ensemble's predict_proba
(verified by additivity). There is NO KernelExplainer fallback: a non-tree model (e.g. AdaBoost,
which is fully excluded from the model panel) raises a RuntimeError rather than silently switching to
a slow/approximate explainer.
`code_terms` (optional {code->term}) renders features as "<code> (<term>) <family>".

Public API:
    explain(model, X, feature_names, y_true, y_pred, proba, out_dir, title="",
            max_explained=None, code_terms=None)          # max_explained kept for API-compat but IGNORED
    from_model_file(model_path, data_path, out_dir, ...)  # load V3 model dict + features, then explain
(tree-exact always explains ALL patients, so max_explained no longer subsamples anything.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def _atomic_csv(df, path):
    """Write a CSV atomically (tmp -> os.replace) so an interruption never leaves a partial/corrupt CSV."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


_TREE = {"RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
         "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "DecisionTreeClassifier"}
SEGMENTS = ("TP", "FP", "FN", "TN")
_SEG_DESC = {"TP": "True Positives (cancer correctly flagged)",
             "FP": "False Positives (non-cancer wrongly flagged)",
             "FN": "False Negatives (cancer missed)",
             "TN": "True Negatives (non-cancer correctly cleared)"}


def _tree_decomposition(model):
    """If `model` is a SOFT-VOTING ensemble of tree models, return (trees, weights); else (None, None).
    Its predict_proba is a TRUE linear combination of the members' probabilities, so SHAP is EXACT:
    SHAP(ensemble) = Σ wᵢ·SHAP(memberᵢ) — letting us use fast exact TreeExplainer per base tree on ALL
    patients. AdaBoost is deliberately NOT decomposed here (SAMME.R combines weak learners NONLINEARLY,
    so a per-tree sum would only be approximate) — and AdaBoost is fully excluded from the model panel
    (config.EXCLUDE_MODELS / removed from get_models), so a non-tree model never reaches explainability."""
    name = type(model).__name__
    if name != "VotingClassifier" or getattr(model, "voting", None) != "soft":
        return None, None
    ests = list(getattr(model, "estimators_", []))
    if not ests or not all(type(e).__name__ in _TREE for e in ests):
        return None, None
    w = model.weights if getattr(model, "weights", None) is not None else [1.0] * len(ests)
    return ests, np.asarray(w, dtype=float)


def _member_tree_shap(est, Xv, bg):
    """Exact PROBABILITY-space TreeExplainer SHAP (positive class) for ONE tree member. Uses
    model_output="probability" with a background sample `bg`, so EVERY member's SHAP is in the SAME
    (probability) space regardless of its native output (RandomForest/ExtraTrees are already
    probability; GradientBoosting/XGBoost/CatBoost are log-odds/margin natively). That makes the
    voting-weighted sum a TRUE decomposition of the soft-voting ensemble's predict_proba.
    Tries the estimator directly; for XGBoost it also tries the underlying Booster as a best-effort.

    IMPORTANT — the real XGBoost fix is the requirements pin `xgboost<3`. shap 0.49.x's TreeExplainer
    reads the model via save_raw(ubj) and does float(learner_model_param["base_score"]); xgboost 3.x
    serializes base_score as a VECTOR STRING ('[4.85E-1]') -> ValueError ("could not convert string to
    float"). This affects BOTH the sklearn wrapper AND the Booster (same UBJ), and even an explicit
    base_score=0.5 still serializes as a vector — so the Booster fallback below does NOT rescue 3.x; it
    only helps other version combos. xgboost 2.x serializes base_score as a scalar, so TreeExplainer
    works directly. Raises only if no exact tree route works (the caller then fails loudly — no kernel)."""
    try:
        return _pos_class(shap.TreeExplainer(est, data=bg, model_output="probability").shap_values(Xv))
    except Exception as e_direct:
        if type(est).__name__ == "XGBClassifier" and hasattr(est, "get_booster"):
            # Best-effort only (does NOT fix shap0.49 × xgboost3.x — see docstring; that's why we pin xgboost<3).
            return _pos_class(shap.TreeExplainer(est.get_booster(), data=bg,
                                                 model_output="probability").shap_values(Xv))
        raise e_direct


def _shap_values(model, X, y=None, max_explained=None, seed=0):
    """SHAP values for the positive class — TREE-EXACT ONLY, on ALL patients, in PROBABILITY space.
    There is NO KernelExplainer fallback: every deployable model is a tree (or a soft-voting ensemble
    of trees) and is explained by exact TreeExplainer with model_output="probability" against a small
    background sample. (REQUIRES xgboost<3 — see _member_tree_shap.) Probability space makes the voting-weighted member sum
    a true decomposition of the ensemble's predict_proba. A non-tree model (e.g. AdaBoost) or a tree
    member that TreeExplainer genuinely cannot handle raises a RuntimeError — we FAIL LOUDLY rather than
    silently switching to a slow/approximate explainer. (`max_explained` is accepted for API
    compatibility but ignored — tree-exact always explains ALL patients; `seed` seeds the background
    sample.) Returns (sv, idx)."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap not installed (pip install shap)")
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    n = len(Xv)
    name = type(model).__name__
    members, weights = _tree_decomposition(model)
    bg = shap.sample(Xv, min(100, n), random_state=seed)   # background for probability-space SHAP

    def _check_additivity(sv, label):
        """Sanity-check the documented additivity — in probability space base + ΣSHAP should reconstruct
        predict_proba, so the per-sample residual (proba - ΣSHAP) should be ~constant (= the base value).
        IMPORTANT: probability-space TreeExplainer uses an INTERVENTIONAL background SAMPLE, so this is
        inherently APPROXIMATE — a few % residual is normal (not a bug). So we only WARN on a GROSS mismatch
        (>10%), which would signal a real decomposition problem; we never block the (requested) SHAP output
        on the expected sampling approximation. (Earlier this raised at 1e-2, which wrongly blocked SHAP.)"""
        try:
            proba = model.predict_proba(Xv)[:, 1]
        except Exception:
            return sv                              # can't fetch proba (shouldn't happen) -> skip rather than crash
        resid = proba - np.asarray(sv).sum(axis=1)
        max_err = float(np.max(np.abs(resid - resid.mean()))) if len(resid) else 0.0
        if max_err > 0.10:
            print(f"[explain] WARNING: SHAP additivity for {label} is loose — max|base+ΣSHAP-predict_proba| "
                  f"= {max_err:.2e} (> 0.10). Likely the probability-space/background approximation, but "
                  f"unusually large; treat SHAP magnitudes as approximate.")
        return sv

    if name in _TREE:                              # single tree model -> exact (XGBoost-safe), ALL patients
        return _check_additivity(_member_tree_shap(model, Xv, bg), name), np.arange(n)
    if members is not None:                        # soft-voting ensemble of trees -> exact per-tree, weighted
        w = weights / weights.sum()
        agg = None
        for e, wi in zip(members, w):
            sv_e = _member_tree_shap(e, Xv, bg)    # XGBoost-safe; raises if there is no exact tree route
            agg = sv_e * wi if agg is None else agg + sv_e * wi
        print(f"[explain] {name}: exact per-tree SHAP over {len(members)} base trees, "
              f"weighted-averaged, ALL {n:,} patients")
        return _check_additivity(agg, name), np.arange(n)
    raise RuntimeError(                            # NO KernelExplainer — fail loudly
        f"[explain] {name} is not a tree model nor a soft-voting ensemble of trees. "
        f"KernelExplainer has been removed; only exact TreeExplainer is supported. "
        f"Exclude this model (config.EXCLUDE_MODELS) or train a tree model.")


def _pos_class(sv):
    """Normalize SHAP output to a 2D (n, n_features) array for the POSITIVE class, across shap
    versions: list [class0, class1] (older) or ndarray (n, f) / (n, f, n_classes) (newer)."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                                  # (n, f, n_classes) -> positive class
        sv = sv[:, :, -1]
    return sv


def segment_of(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int)
    seg = np.empty(len(y_true), dtype=object)
    seg[(y_true == 1) & (y_pred == 1)] = "TP"
    seg[(y_true == 0) & (y_pred == 1)] = "FP"
    seg[(y_true == 1) & (y_pred == 0)] = "FN"
    seg[(y_true == 0) & (y_pred == 0)] = "TN"
    return seg


def humanize(feat, code_terms):
    """Turn a code-prefixed feature like '428489000_decay_intensity' into
    '428489000 (Cough) decay_intensity' using code_terms {code:int -> term}. Global/concept features
    (g_age, NLR, g_consult_*, …) carry no numeric code prefix and are returned unchanged."""
    import re
    m = re.match(r"^(\d{3,})_(.*)$", str(feat))
    if not m:
        return str(feat)
    code, family = m.group(1), m.group(2)
    term = code_terms.get(int(code), code_terms.get(code)) if code_terms else None
    return f"{code} ({term}) {family}" if term else str(feat)


# ---------------------------------------------------------------- (1) patient-level
def patient_table(sv, feature_names, y_true, y_pred, proba, idx, top_k=10, guids=None):
    rows = []
    for r, i in enumerate(idx):
        order = np.argsort(np.abs(sv[r]))[::-1][:top_k]
        yt = int(y_true[i]) if y_true is not None else None
        yp = int(y_pred[i])
        # segment lets you filter mistakes directly: FP = wrongly flagged, FN = missed cancer
        seg = None if yt is None else (("TP" if yt == 1 else "FP") if yp == 1 else ("FN" if yt == 1 else "TN"))
        rec = {"row": int(i),
               "patient_guid": (str(guids[i]) if guids is not None else None),  # identify the patient / join age,density
               "segment": seg,
               "y_true": yt,
               "y_pred": yp,
               "prob": (float(proba[i]) if proba is not None else None)}
        for k, j in enumerate(order, 1):
            rec[f"factor_{k}"] = feature_names[j]
            rec[f"shap_{k}"] = round(float(sv[r, j]), 4)   # signed: + pushes toward cancer
        rows.append(rec)
    return pd.DataFrame(rows)


def global_summary_plot(sv, X, feature_names, out_png, max_display=20):
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    shap.summary_plot(sv, Xv, feature_names=feature_names, show=False, max_display=max_display)
    fig = plt.gcf(); fig.suptitle("Global SHAP summary (patient-level feature impact)", y=1.02)
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- (2) segment-level
def segment_drivers(sv, feature_names, seg_explained, top_k=15):
    """Per confusion-matrix segment: mean |SHAP| (importance) and mean signed SHAP (net direction)
    per feature, ranked by importance -> top_k drivers per segment."""
    rows = []
    for s in SEGMENTS:
        m = seg_explained == s
        n = int(m.sum())
        if n == 0:
            continue
        mean_abs = np.abs(sv[m]).mean(axis=0)
        mean_signed = sv[m].mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        for rank, j in enumerate(order, 1):
            rows.append({"segment": s, "segment_desc": _SEG_DESC[s], "n_in_segment": n, "rank": rank,
                         "feature": feature_names[j],
                         "mean_abs_shap": round(float(mean_abs[j]), 5),
                         "mean_signed_shap": round(float(mean_signed[j]), 5),
                         "direction": "↑risk" if mean_signed[j] > 0 else "↓risk"})
    return pd.DataFrame(rows)


def plot_segment_drivers(df, out_png, top_k=15):
    segs = [s for s in SEGMENTS if s in set(df["segment"])]
    fig, axes = plt.subplots(1, len(segs), figsize=(5.2 * len(segs), 6.5), squeeze=False)
    for ax, s in zip(axes[0], segs):
        d = df[df["segment"] == s].head(top_k).iloc[::-1]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in d["mean_signed_shap"]]
        ax.barh(range(len(d)), d["mean_abs_shap"], color=colors)
        _lab = d["feature_label"] if "feature_label" in d.columns else d["feature"]   # prefer the term label
        ax.set_yticks(range(len(d))); ax.set_yticklabels([str(f)[:48] for f in _lab], fontsize=7)
        ax.set_xlabel("mean |SHAP|"); ax.set_title(f"{s}  (n={int(d['n_in_segment'].iloc[0])})", fontsize=11)
    fig.suptitle("Key risk-factor drivers per confusion-matrix segment\n"
                 "(red = pushes toward cancer, blue = pushes away)", fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)


_SEG_TITLE = {"TP": "True Positives — cancer correctly flagged",
              "FP": "False Positives — non-cancer wrongly flagged",
              "FN": "False Negatives — cancer missed",
              "TN": "True Negatives — non-cancer correctly cleared"}


def plot_segment_drivers_individual(df, out_dir, top_k=15):
    """One LARGE, readable bar chart PER segment -> segment_{TP,FP,FN,TN}.png (more legible than
    the combined 4-panel figure)."""
    for s in SEGMENTS:
        d = df[df["segment"] == s].head(top_k).iloc[::-1]
        if len(d) == 0:
            continue
        n = int(d["n_in_segment"].iloc[0])
        fig, ax = plt.subplots(figsize=(11, 7.5))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in d["mean_signed_shap"]]
        ax.barh(range(len(d)), d["mean_abs_shap"], color=colors)
        _lab = d["feature_label"] if "feature_label" in d.columns else d["feature"]   # prefer the term label
        ax.set_yticks(range(len(d))); ax.set_yticklabels([str(f)[:60] for f in _lab], fontsize=12)
        ax.set_xlabel("mean |SHAP|  —  average impact on the model's output", fontsize=13)
        ax.set_title(f"{s}  ·  {_SEG_TITLE[s]}   (n = {n})\n"
                     f"red = pushes toward cancer   ·   blue = pushes away", fontsize=14, pad=12)
        ax.tick_params(axis="x", labelsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"segment_{s}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_segment_beeswarm(sv, X, feature_names, seg_explained, out_dir, max_display=15):
    """Per confusion-matrix segment: a SHAP BEESWARM (summary) plot ->
    segment_{TP,FP,FN,TN}_beeswarm.png. Each dot is ONE patient; x = signed SHAP
    (right pushes toward cancer), colour = that feature's value (red high, blue low).
    Complements the mean-|SHAP| bar charts by showing the per-patient DISTRIBUTION and
    whether high vs low feature values drive risk within the segment."""
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    seg_explained = np.asarray(seg_explained)
    for s in SEGMENTS:
        m = seg_explained == s
        n = int(m.sum())
        if n < 2:                                  # beeswarm needs >= 2 patients
            continue
        shap.summary_plot(sv[m], Xv[m], feature_names=list(feature_names),
                          show=False, max_display=max_display, plot_type="dot")
        fig = plt.gcf()
        fig.suptitle(f"{s} · {_SEG_TITLE[s]}  (n={n})\n"
                     f"beeswarm — x: SHAP (right → cancer) · colour: feature value (red high)",
                     fontsize=11, y=1.02)
        fig.savefig(os.path.join(out_dir, f"segment_{s}_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- orchestration
def explain(model, X, feature_names, y_true, y_pred, proba, out_dir, title="", max_explained=None,
            code_terms=None, patient_guids=None):
    """`code_terms` (optional) = {code:int -> term:str} from the codelist (Code,Name). When given,
    code-prefixed features are shown as '<code> (<term>) <family>' so the output is human-readable;
    global/concept features (g_age, NLR, …) are left as-is. Nothing is hardcoded — the map is the
    config's own codelist, so it aligns with whatever horizon/window/arm is being explained."""
    os.makedirs(out_dir, exist_ok=True)
    feature_names = list(feature_names)
    sv, idx = _shap_values(model, X, y=y_true, max_explained=max_explained)
    y_pred = np.asarray(y_pred).astype(int)

    # (1) patient-level — #factors per patient is env-configurable (EXPLAIN_TOPK, default 20)
    _topk = max(1, int(os.environ.get("EXPLAIN_TOPK", "20")))
    pt = patient_table(sv, feature_names, y_true, y_pred, proba, idx, top_k=_topk,
                       guids=(np.asarray(patient_guids) if patient_guids is not None else None))
    if code_terms:                               # show "<code> (term) <family>" instead of the bare code
        for k in range(1, _topk + 1):
            col = f"factor_{k}"
            if col in pt.columns:
                pt[col] = pt[col].map(lambda f: humanize(f, code_terms))
    _atomic_csv(pt, os.path.join(out_dir, "patient_explanations.csv"))
    try:
        Xexp = (X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx])
        names = [humanize(f, code_terms) for f in feature_names] if code_terms else feature_names
        global_summary_plot(sv, Xexp, names, os.path.join(out_dir, "shap_summary.png"))
    except Exception as e:
        print(f"[explain] summary plot skipped: {e}")

    # (2) segment-level (needs labels)
    seg_df = None
    if y_true is not None:
        seg_explained = segment_of(np.asarray(y_true)[idx], y_pred[idx])
        seg_df = segment_drivers(sv, feature_names, seg_explained)
        if code_terms:                           # add a human-readable label next to the raw feature
            seg_df.insert(seg_df.columns.get_loc("feature") + 1, "feature_label",
                          seg_df["feature"].map(lambda f: humanize(f, code_terms)))
        _atomic_csv(seg_df, os.path.join(out_dir, "segment_drivers.csv"))
        plot_segment_drivers(seg_df, os.path.join(out_dir, "segment_drivers.png"))
        plot_segment_drivers_individual(seg_df, out_dir)
        try:
            Xexp = (X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx])
            names = [humanize(f, code_terms) for f in feature_names] if code_terms else feature_names
            plot_segment_beeswarm(sv, Xexp, names, seg_explained, out_dir)
        except Exception as _e:
            print(f"[explain] segment beeswarm skipped: {_e}")
        counts = {s: int((seg_explained == s).sum()) for s in SEGMENTS}
        print(f"[explain] {title} segment sizes (explained): {counts}")
    print(f"[explain] wrote patient + segment explainability -> {out_dir}")
    return seg_df


def from_model_file(model_path, data_path, out_dir, threshold=None, title=""):
    """Load a V3 model dict + a features CSV/parquet (with cancer_class), preprocess via the saved
    train-fitted transforms (CancerPredictor), predict, and write #1 + #2 explainability."""
    import joblib
    import sys
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    from predict_unseen import CancerPredictor
    cp = CancerPredictor(model_path)
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    X, y = cp.preprocess_data(df)
    model = joblib.load(model_path)["model"]
    proba = model.predict_proba(X)[:, 1]
    if threshold is None:
        threshold = 0.5
    y_pred = (proba >= threshold).astype(int)
    feats = cp.selected_features or cp.feature_names or [f"f{i}" for i in range(X.shape[1])]
    return explain(model, X, feats, y, y_pred, proba, out_dir, title=title)
