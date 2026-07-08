"""
Parity harness — Polars FE backend vs the pandas reference (run locally, no BigQuery).
========================================================================================
Builds a synthetic `ev` event frame matching build_features.load_events' schema (varied histories,
value-bearing + partial-numeric + code-only codes, NaN ages, tied days, single-event groups), then
runs each of the six per-code families both ways and asserts the outputs are equal (same columns,
same order, NaN-aware values within tolerance). Exits 0 on PASS, 1 on FAIL.

Run:  python fe_parity_check.py
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_features as bf
import build_features_polars as pf


def synthetic_ev(seed=0):
    rng = np.random.RandomState(seed)
    value_codes = [1001, 1002]      # mostly numeric (labs/vitals)
    partial_codes = [2001]          # some numeric, some not
    code_only = [3001, 3002]        # never numeric
    all_codes = value_codes + partial_codes + code_only
    rows = []
    for p in range(45):
        pg = "{P%03d}" % p
        for _ in range(rng.randint(1, 13)):
            code = int(rng.choice(all_codes))
            days = int(rng.randint(0, 1800))
            if code in value_codes:
                val = float(rng.normal(10, 2))
            elif code in partial_codes:
                val = float(rng.normal(5, 1)) if rng.rand() < 0.4 else np.nan
            else:
                val = np.nan
            age = float(rng.randint(40, 86)) if rng.rand() < 0.95 else np.nan
            rows.append((pg, code, days, val, age, int(rng.rand() < 0.3), int(rng.rand() < 0.2)))
    # deterministic edge cases
    rows += [
        ("{PX01}", 1001, 100, 12.0, 60.0, 1, 0),                    # single value event
        ("{PX02}", 3001, 50, np.nan, np.nan, 0, 0),                 # single code-only, null age
        ("{PX03}", 1002, 200, 9.0, 70.0, 0, 1),                     # tied days, differing value/age
        ("{PX03}", 1002, 200, 11.0, 71.0, 0, 1),
        ("{PX04}", 1001, 300, 8.0, np.nan, 0, 0),                   # 4 value events -> exercises val_accel
        ("{PX04}", 1001, 250, 9.0, 55.0, 0, 0),
        ("{PX04}", 1001, 150, 10.0, 55.0, 0, 0),
        ("{PX04}", 1001, 40, 13.0, 55.0, 0, 0),
    ]
    ev = pd.DataFrame(rows, columns=["patient_guid", "code", "days", "value_num",
                                     "event_age", "active", "sig"])
    ev["code"] = ev["code"].astype("int64")
    ev["months"] = ev["days"] / 30.44
    ev["event_type"] = "observation"
    return ev


def _cmp(name, a, b, rtol=1e-5, atol=1e-6):
    """Compare two (patient_guid)-indexed feature frames. Returns (ok, message)."""
    a = a.sort_index(); b = b.sort_index()
    if list(a.columns) != list(b.columns):
        only_a = [c for c in a.columns if c not in set(b.columns)]
        only_b = [c for c in b.columns if c not in set(a.columns)]
        if set(a.columns) != set(b.columns):
            return False, f"column SET differs (+pandas {only_a[:5]} / +polars {only_b[:5]})"
        return False, "column ORDER differs"
    if list(a.index) != list(b.index):
        return False, f"index differs ({len(a)} vs {len(b)} patients)"
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    av = a.to_numpy(dtype=float); bv = b.to_numpy(dtype=float)
    close = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
    if not close.all():
        i, j = np.argwhere(~close)[0]
        return False, (f"{int((~close).sum())} value(s) differ; first at "
                       f"row {a.index[i]} col '{a.columns[j]}': pandas={av[i, j]} polars={bv[i, j]}")
    return True, f"{a.shape[0]} patients x {a.shape[1]} cols match"


def main():
    ev = synthetic_ev()
    value_codes = set(ev.dropna(subset=["value_num"])["code"].unique())
    relative = bf.ANCHOR_MODE == "patient_last"
    print(f"synthetic ev: {len(ev):,} events, {ev.patient_guid.nunique()} patients, "
          f"{ev.code.nunique()} codes, value_codes={sorted(value_codes)}\n")

    cases = [
        ("occurrence_features", lambda m: m.occurrence_features(ev)),
        ("flag_features",       lambda m: m.flag_features(ev)),
        ("age_features",        lambda m: m.age_features(ev)),
        ("value_features",      lambda m: m.value_features(ev, value_codes)),
        ("band_features",       lambda m: m.band_features(ev, bf.TIME_BANDS, value_codes, relative)),
        ("cumulative_features", lambda m: m.cumulative_features(ev, bf.CUMULATIVE_WINDOWS, value_codes, relative)),
    ]
    all_ok = True
    for name, fn in cases:
        try:
            ok, msg = _cmp(name, fn(bf), fn(pf))
        except Exception as e:
            ok, msg = False, f"EXCEPTION: {type(e).__name__}: {e}"
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} {msg}")

    print("\n" + ("ALL FAMILIES PARITY PASS" if all_ok else "PARITY FAILURES — see above"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
