"""
Full-vs-keep-features PARITY GATE (automated; hard-fail, no fallback).
=====================================================================
The held-out / inference path builds ONLY the model's features (`keep_features=...`) to avoid the ~280k-wide
OOM. That shortcut must be byte-identical to the full build for every model feature — a cross-category or
restricted family computed differently under keep-features would silently corrupt serving. This gate proves
it on a patient sample: build both ways, compare the model's columns, exit 1 on ANY mismatch.

Runs per (horizon, years); same contract for both horizons. Uses the SAME shared fit for both builds so the
only difference under test is the keep-features restriction itself.

  python fe_keep_parity.py 12mo 10 [--n 200] [--heldout]      # --heldout: validate the serve cohort
"""
import os, sys, importlib.util, numpy as np, pandas as pd, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)   # config.py (root) + build_features.py (here)
ATOL = 1e-9                                  # byte-ish; the build is deterministic so the real bar is 0


def _load(path, name):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m


def main():
    win = sys.argv[1] if len(sys.argv) > 1 else "12mo"
    years = next((int(a) for a in sys.argv[2:] if a.isdigit()), 0)
    N = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--n"), 200)
    heldout = "--heldout" in sys.argv
    sys.argv = sys.argv[:1]                  # config.py parses argv on import
    import config as C
    _load(os.path.join(ROOT, "3_Modeling", "lung_training.py"), "lung_training")   # for joblib unpickle
    fe = _load(os.path.join(HERE, "build_features.py"), "fe_engine")

    arm = C.artifact_subdir(win, years)
    md = joblib.load(os.path.join(ROOT, "3_Modeling_outputs", arm, f"model_{win}.joblib"))
    feats = [f for f in md["feature_names"] if f != "cancer_class"]
    sql = None
    if heldout:
        # The actual serve/held-out cohort SQL (NOT the train cohort). Previously pointed at a non-existent
        # 0_SQL/holdout_lung.sql and silently fell back to sql=None (the TRAIN cohort), so --heldout never
        # validated the serve cohort's keep-features build. Hard-fail if missing rather than silently degrade.
        sql = os.path.join(ROOT, "4_Heldout", "SQL", f"heldout_test_{win}.sql")
        if not os.path.exists(sql):
            raise SystemExit(f"[parity] --heldout: serve-cohort SQL not found: {sql}")

    print(f"[parity] {win} {years}yr {'HELDOUT' if heldout else 'TRAIN'} cohort | model feats={len(feats):,} | sample={N}")
    # NON-MUTATING: fit_split=False ALWAYS — load the persisted TRAIN zero-fit artifacts (zstats/value_codes/
    # top_cats/moc_zero), never re-fit/re-persist. This is exactly the serve path we want to validate, and it
    # guarantees the gate never overwrites the production FE artifacts on GCS.
    shared = fe.build(win, sql_path=sql, fit_split=False, years=years, schema_only=True)
    pids = shared["roster"]["patient_guid"].drop_duplicates().to_numpy()
    rng = np.random.RandomState(int(getattr(C, "RANDOM_STATE", 42)))
    sample = set(pids[rng.choice(len(pids), size=min(N, len(pids)), replace=False)])

    full, _ = fe.build(win, sql_path=sql, fit_split=False, years=years,
                       patient_ids=sample, shared=shared, keep_features=None)        # ALL families/cols
    keep, _ = fe.build(win, sql_path=sql, fit_split=False, years=years,
                       patient_ids=sample, shared=shared, keep_features=md["feature_names"])  # model cols only

    full = full.set_index("patient_guid").sort_index()
    keep = keep.set_index("patient_guid").sort_index()
    if list(full.index) != list(keep.index):
        raise SystemExit(f"[parity] FAIL: patient sets differ (full {len(full)} vs keep {len(keep)}) -> abort")

    # KEEP must carry every model column (it reindexes to feature_names); a missing one is a real bug.
    miss_keep = [f for f in feats if f not in keep.columns]
    if miss_keep:
        raise SystemExit(f"[parity] FAIL: KEEP build missing {len(miss_keep)} model col(s) e.g. {miss_keep[:5]} -> abort")
    # FULL legitimately omits a column when NO sampled patient has that category's events (genuine sparsity;
    # on the cohort it would be 0). Those are not parity violations -> compare the INTERSECTION both produced.
    cols = [f for f in feats if f in full.columns]
    skipped = len(feats) - len(cols)

    a = full[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    b = keep[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    close = np.isclose(a, b, rtol=0.0, atol=ATOL, equal_nan=True)
    if not close.all():
        n_bad = int((~close).sum()); i, j = np.argwhere(~close)[0]
        raise SystemExit(f"[parity] FAIL: {n_bad} value(s) differ (atol={ATOL}); first at patient "
                         f"{full.index[i]} col '{cols[j]}': full={a[i, j]} keep={b[i, j]} -> abort (no fallback)")
    maxdiff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    print(f"[parity] PASS: {len(full):,} patients x {len(cols):,} model features identical "
          f"(max abs diff {maxdiff:.3e}, atol {ATOL:.0e}); {skipped} col(s) absent in sample (sparse, skipped)")
    sys.exit(0)


if __name__ == "__main__":
    main()
