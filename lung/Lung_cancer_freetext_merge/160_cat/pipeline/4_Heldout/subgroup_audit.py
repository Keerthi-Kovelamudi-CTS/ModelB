"""
Subgroup performance / fairness audit on the held-out TEST slice (Methodology §7.4/§8): Sens/Spec/PPV/NPV +
calibration (Brier/ECE) WITHIN age-band × sex × ethnicity (incl. Unknown) × record-density quartile, and a
disparity flag wherever a stratum's groups differ by > DISPARITY_PTS sensitivity points. Post-hoc on the
saved artifacts (cached held-out matrix + model + Platt + operating threshold) — no retrain.

Strata are recovered from the (raw, pre-scale) held-out FE matrix:
  sex        <- g_is_male (1=Male, 0=Female, NaN=Unknown)
  ethnicity  <- g_eth_* one-hots (the set column; all-zero => Unknown)
  age_band   <- ageband_* one-hots
  density_q  <- quartile of a record-density proxy (g_n_categories, else summed *_count)

Run on the box holding the artifacts (cpu-02=12mo, gpu-14=1mo):
    python 4_Heldout/subgroup_audit.py 12mo 10
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C

SEED = int(getattr(C, "RANDOM_STATE", 42))
DISPARITY_PTS = 5.0
MIN_N = 30


def _reg(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
_reg(os.path.join(os.path.dirname(HERE), "3_Modeling", "lung_training.py"), "lung_training")
lm = _reg(os.path.join(os.path.dirname(HERE), "3_Modeling", "lung_metrics.py"), "lung_metrics")


def _transform_external(df, md):
    d = df.copy()
    for col, mp in (md.get("encoders") or {}).items():
        d[col] = (d[col].fillna("Unknown").astype(str).map(mp).fillna(-1) if col in d.columns else -1)
    d = d.reindex(columns=md["feature_names"]).apply(pd.to_numeric, errors="coerce").astype("float64")
    vc = set(md.get("value_cols") or []); val = [c for c in md["feature_names"] if c in vc]
    if val and md.get("impute_medians") is not None:
        d[val] = d[val].fillna(md["impute_medians"])
    return md["scaler"].transform(d.replace([np.inf, -np.inf], np.nan).fillna(0.0).values)


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def _recover_strata(mat_te):
    """One label per patient for each stratum, from the raw FE matrix rows of the test slice."""
    n = len(mat_te); cols = mat_te.columns
    out = {}
    # sex
    if "g_is_male" in cols:
        s = pd.to_numeric(mat_te["g_is_male"], errors="coerce")
        out["sex"] = np.where(s > 0.5, "Male", np.where(s < 0.5, "Female", "Unknown")).astype(object)
    # ethnicity (argmax of g_eth_*; all-zero -> Unknown)
    eth_cols = [c for c in cols if c.startswith("g_eth_")]
    if eth_cols:
        E = mat_te[eth_cols].fillna(0.0).to_numpy()
        lab = np.array([c[len("g_eth_"):] for c in eth_cols])
        out["eth"] = np.where(E.sum(1) > 0, lab[E.argmax(1)], "Unknown").astype(object)
    # age band (argmax of ageband_*)
    ab_cols = [c for c in cols if c.startswith("ageband_")]
    if ab_cols:
        A = mat_te[ab_cols].fillna(0.0).to_numpy()
        lab = np.array([c[len("ageband_"):] for c in ab_cols])
        out["age_band"] = np.where(A.sum(1) > 0, lab[A.argmax(1)], "Unknown").astype(object)
    # record-density quartile
    dens_col = "g_n_categories" if "g_n_categories" in cols else None
    if dens_col is None:
        cnt = [c for c in cols if c.endswith("_count") or c == "g_n_categories_6mo"]
        dens = mat_te[cnt].fillna(0.0).sum(1) if cnt else pd.Series(np.zeros(n))
    else:
        dens = pd.to_numeric(mat_te[dens_col], errors="coerce").fillna(0.0)
    try:
        out["density_q"] = pd.qcut(dens.rank(method="first"), 4, labels=["Q1_sparse", "Q2", "Q3", "Q4_rich"]).astype(object).to_numpy()
    except Exception:
        out["density_q"] = np.array(["all"] * n, dtype=object)
    return out


def run_audit(mat_te, y_te, p_te, cut, out_dir, window, years):
    """Compute + write the subgroup audit for one horizon's TEST slice. Reusable: called both by this
    script's main() (standalone) and by 4_Heldout/evaluate_heldout.py (so every held-out run emits it).
    `mat_te` = the (raw, pre-scale) held-out FE-matrix rows for the test slice."""
    strata = _recover_strata(mat_te.reset_index(drop=True))
    rows, flags = [], []
    print(f"\nSUBGROUP AUDIT — {window} {years}yr | test n={len(y_te):,} pos={int(y_te.sum())} | cut={cut:.4f} | "
          f"flag if Sens range > {DISPARITY_PTS} pts (groups n>={MIN_N})")
    for col, g in strata.items():
        g = np.asarray(g); sens_by = {}
        print(f"\n  === {col} ===")
        print(f"    {'group':<28}{'n':>6}{'pos':>5}{'AUROC':>7}{'Sens':>7}{'Spec':>7}{'PPV':>6}{'Brier':>7}{'ECE':>6}")
        for val in sorted(pd.unique(g), key=str):
            m = g == val; n = int(m.sum())
            if n < MIN_N:
                continue
            r = lm.eval_with_ci(y_te[m], p_te[m], threshold=float(cut), label=f"{col}={val}")
            row = lm.results_to_row(r, years, f"{col}={val}")
            row["brier"] = round(float(brier_score_loss(y_te[m], p_te[m])) if len(np.unique(y_te[m])) > 1 else float("nan"), 4)
            row["ece"] = round(float(_ece(p_te[m], y_te[m])), 4)
            rows.append(row); sens_by[val] = r["sensitivity"]
            au = r["auroc"] if not np.isnan(r["auroc"]) else float("nan")
            print(f"    {str(val):<28}{n:>6}{int(y_te[m].sum()):>5}{au*100:>6.1f}%{r['sensitivity']*100:>6.1f}%"
                  f"{r['specificity']*100:>6.1f}%{r['ppv']*100:>5.1f}%{row['brier']:>7}{row['ece']:>6}")
        if len(sens_by) >= 2:
            spread = (max(sens_by.values()) - min(sens_by.values())) * 100
            mark = "  *** DISPARITY ***" if spread > DISPARITY_PTS else ""
            lo = min(sens_by, key=sens_by.get); hi = max(sens_by, key=sens_by.get)
            print(f"    -> Sens range {spread:.1f} pts (low={lo} {sens_by[lo]*100:.1f}% / high={hi} {sens_by[hi]*100:.1f}%){mark}")
            if spread > DISPARITY_PTS:
                flags.append({"stratum": col, "sens_spread_pts": round(spread, 1),
                              "lowest_group": lo, "lowest_sens": round(sens_by[lo], 4),
                              "highest_group": hi, "highest_sens": round(sens_by[hi], 4)})
    out_csv = os.path.join(out_dir, f"subgroup_audit_{window}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if flags:
        pd.DataFrame(flags).to_csv(os.path.join(out_dir, f"subgroup_disparities_{window}.csv"), index=False)
    print(f"\n  -> wrote {out_csv} ({len(rows)} subgroups); disparities flagged: {len(flags)}")
    return flags


def main():
    """Standalone: load the saved artifacts, reproduce the held-out test slice, run the audit."""
    win = sys.argv[1] if len(sys.argv) > 1 else "12mo"
    years = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else int(getattr(C, "FE_YEARS_BEFORE", 0) or 0)
    out_dir = os.path.join(HERE, "..", "3_Modeling_outputs", C.artifact_subdir(win, years))
    mat = pd.read_parquet(os.path.join(HERE, "..", "2_FE", "output", C.artifact_subdir(win, years), f"heldout_features_{win}.parquet"))
    md = joblib.load(os.path.join(out_dir, f"model_{win}.joblib"))
    platt = joblib.load(os.path.join(out_dir, f"platt_calib_{win}.joblib"))
    thr_p = os.path.join(out_dir, f"operating_threshold_{win}.json")
    cut = json.load(open(thr_p))["threshold"] if os.path.exists(thr_p) else 0.5
    y = mat["cancer_class"].astype(int).to_numpy()
    raw = md["model"].predict_proba(_transform_external(mat, md))[:, 1]
    _, te = train_test_split(np.arange(len(y)), test_size=0.70, random_state=SEED, stratify=y)
    p_te = platt.predict_proba(raw[te].reshape(-1, 1))[:, 1]
    run_audit(mat.iloc[te], y[te], p_te, float(cut), out_dir, win, years)


if __name__ == "__main__":
    main()
