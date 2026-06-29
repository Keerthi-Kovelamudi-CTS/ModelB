"""Cohort events — the LIFETIME events are pulled from BigQuery ONCE per horizon and cached as parquet
(GCS + local); every lookback window is then derived by filtering that one cached frame LOCALLY, with
the exact same calendar-year rule BigQuery uses.

So BigQuery is scanned **once per horizon** for the whole pipeline (all windows 5/10/20/100, code
scoring, value-bearing, FE, AND the split membership), shared across machines /
reruns / teammates via GCS. Held-out is the same: one lifetime pull per horizon.

Cache key:  {GCS_ROOT}/raw_events/{tag}{Astra|Nova}_lifetime.parquet   (Astra=12mo, Nova=1mo; tag="" training, "heldout_" held-out)

Window filtering is BIT-IDENTICAL to the per-window BigQuery pull: the lookback is a calendar-year
window (effective_date >= DATE_SUB(anchor, INTERVAL N YEAR)). We reconstruct anchor_date locally as
event_date + days_before_anchor and apply pandas DateOffset(years=N) — same calendar arithmetic.
Codes are CAST AS STRING and parsed with to_int64 (18-digit DMD/SNOMED codes > 2**53 stay exact).
"""
import os
import re
import sys
import pandas as pd
try:
    import db_dtypes  # noqa: F401  — registers the BQ DATE ('dbdate') extension dtype so the SELECT-*
    # lifetime parquet (which now carries native date columns, e.g. event_date) reads back with pd.read_parquet.
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
import artifacts                          # atomic cache writes + SQL-hash freshness manifest

# Widest lookback we ever filter to -> the lifetime pull uses this many years_before (covers all windows).
LIFETIME_YEARS = max(list(getattr(C, "FE_WINDOWS", [100])) + [getattr(C, "SCORING_YEARS_BEFORE", 5), 100])

# SELECT * — cache the cohort SQL's FULL final output, NOTHING trimmed (so the raw cache is complete for
# any downstream use, incl. associated_text / free-text / LLM). event_date + days_before_anchor let us
# reconstruct anchor_date and window locally. Downstream `to_int64` parses the (BQ-native) 18-digit DM+D /
# SNOMED codes exactly, so no CAST is needed here. NOTE: this also carries patient_age (a snapshot age the
# pipeline deliberately does NOT use — age_at_anchor is the leak-safe one); it's present but unused.
_QUERY = """
    SELECT * FROM (
{sql}
    )
"""


def to_int64(series):
    """Exact string/number -> nullable Int64 WITHOUT a float64 hop. `pd.to_numeric(...).astype('Int64')`
    upcasts to float64 when NaNs are present, which ROUNDS 17-18-digit DMD/SNOMED codes > 2**53 (e.g.
    39115611000001103 -> ...104). Parsing each value to a Python int keeps them exact."""
    def _one(x):
        if x is None:
            return pd.NA
        s = str(x).strip()
        if s in ("", "nan", "NaN", "None", "<NA>"):
            return pd.NA
        try:
            return int(s) if s.lstrip("-").isdigit() else int(float(s))
        except Exception:
            return pd.NA
    return pd.array([_one(x) for x in series], dtype="Int64")


def _sql_path(horizon):
    return os.path.join(HERE, "0_SQL", f"{horizon}_1to1.sql")


def cache_path(horizon, tag=""):
    root = C.GCS_ROOT.rstrip("/") if getattr(C, "GCS_ROOT", "") else os.path.join(HERE, "cache")
    return f"{root}/raw_events/{tag}{C.horizon_label(horizon)}_lifetime.parquet"   # Astra=12mo, Nova=1mo


def _exists(path):
    if path.startswith("gs://"):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().exists(path)
        except Exception:
            return False
    return os.path.exists(path)


def _save(df, path):
    # ATOMIC: write to a temp then os.replace (local) / single-object write (gs://) so a crash or a
    # concurrent run never leaves a PARTIAL parquet that a later run would treat as a valid cache.
    artifacts.atomic_write(path, lambda t: df.to_parquet(t, index=False))


def _pull_lifetime(horizon, sql_path):
    sql_file = sql_path or _sql_path(horizon)
    with open(sql_file, encoding="utf-8") as _f:
        sql = _f.read().rstrip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    sql = re.sub(r"ORDER\s+BY\s+patient_guid\s*$", "", sql, flags=re.I).rstrip()
    # Set the lifetime lookback window. GUARD: if the 'N AS years_before' alias isn't found the cohort
    # SQL changed shape — fail LOUD rather than silently pull the SQL's hardcoded (possibly small) window,
    # which would build a too-short cache and make longer lookbacks silently miss events.
    sql, _n = re.subn(r"\d+(\s+AS years_before\b)", rf"{LIFETIME_YEARS}\1", sql, count=1)
    if _n == 0:
        raise RuntimeError(f"[events] could not set the lifetime window: no 'N AS years_before' found in "
                           f"{sql_file} — the cohort SQL structure changed; refusing to pull the wrong window.")
    print(f"[events] BigQuery LIFETIME pull {horizon} (years_before={LIFETIME_YEARS}; one-time) ...")
    from google.cloud import bigquery
    ev = bigquery.Client(project=getattr(C, "GCP_PROJECT", None)).query(_QUERY.format(sql=sql)).to_dataframe()
    ev["cancer_class"] = ev["cancer_class"].astype(int)
    return ev


def _window(ev, years):
    """Filter lifetime events to a `years` lookback using the SAME calendar rule as BigQuery
    (effective_date >= DATE_SUB(anchor, INTERVAL years YEAR)); anchor = event_date + days_before_anchor."""
    if years >= LIFETIME_YEARS:
        return ev.copy()
    ed = pd.to_datetime(ev["event_date"], errors="coerce")
    anchor = ed + pd.to_timedelta(pd.to_numeric(ev["days_before_anchor"], errors="coerce"), unit="D")
    cutoff = anchor - pd.DateOffset(years=int(years))         # calendar-year subtraction (matches INTERVAL N YEAR)
    return ev[ed >= cutoff].copy()


def _cache_fresh(path, sql_file):
    """Is the cached lifetime parquet still valid for the CURRENT cohort SQL + window? Compares ONLY the
    SQL content hash + LIFETIME_YEARS recorded in the cache's manifest — deliberately NOT git_sha, so an
    unrelated code commit does not trigger an (expensive) BigQuery re-pull, but a cohort-SQL edit does.
    Returns True (fresh), False (stale -> re-pull), or None (legacy cache w/o manifest -> trust, warn)."""
    m = artifacts.read_manifest(path)
    if m is None:
        return None                                          # unversioned legacy cache -> back-compat
    want = artifacts.file_hash(sql_file)
    got = (m.get("upstream") or {}).get(os.path.basename(sql_file))
    return (got == want) and (m.get("config", {}).get("LIFETIME_YEARS") == LIFETIME_YEARS)


# Single-entry in-process memo of the last-read LIFETIME frame (keyed by cache path). Within one build()
# the windowed stream AND the un-windowed patient roster both call load_cohort_events for the SAME cache
# path — this lets the 2nd call reuse the in-memory frame instead of re-reading the (multi-GB) parquet.
# Single entry => bounded to ONE resident frame (a different path evicts the old). _window() returns a
# COPY, so the memoized frame is never mutated by callers.
_FRAME_MEMO = {"path": None, "ev": None}


def load_cohort_events(horizon, years=None, sql_path=None, tag=""):
    """Cohort events for (horizon, lookback). LIFETIME events are pulled from BigQuery once and cached;
    this returns them filtered to `years` locally. `sql_path` overrides the cohort (held-out: tag='heldout_').
    The cache is keyed to the cohort SQL: if the SQL (or LIFETIME_YEARS) changed, it is re-pulled — so a
    cohort-definition edit never silently propagates a stale cohort through the whole pipeline."""
    yrs = int(years if years is not None else getattr(C, "FE_YEARS_BEFORE", 5))
    path = cache_path(horizon, tag)
    sql_file = sql_path or _sql_path(horizon)
    fresh = _cache_fresh(path, sql_file) if _exists(path) else False
    if _exists(path) and fresh is not False:                 # fresh (True) or legacy-no-manifest (None)
        if fresh is None:
            print(f"[events][warn] cached lifetime {tag}{horizon} has NO manifest (unversioned legacy cache) "
                  f"-> trusting it. If the cohort SQL changed, clear it to force a fresh pull.")
        if _FRAME_MEMO["path"] == path and _FRAME_MEMO["ev"] is not None:
            ev = _FRAME_MEMO["ev"]                            # in-process reuse: no second disk read of the same cache
            print(f"[events] cached lifetime {tag}{horizon} (in-memory reuse) ({len(ev):,}); window {yrs}yr")
        else:
            ev = pd.read_parquet(path)
            _FRAME_MEMO["path"], _FRAME_MEMO["ev"] = path, ev
            print(f"[events] cached lifetime {tag}{horizon} -> {path} ({len(ev):,}); window {yrs}yr")
    else:
        if _exists(path):                                    # exists but STALE (SQL or window changed)
            print(f"[events][STALE] {path}: cohort SQL or LIFETIME_YEARS changed since this cache was built "
                  f"-> RE-PULLING from BigQuery (one-time).")
        ev = _pull_lifetime(horizon, sql_path)
        _save(ev, path)
        _FRAME_MEMO["path"], _FRAME_MEMO["ev"] = path, ev    # memo the freshly-pulled frame (roster reuse)
        try:                                                 # SQL-hash + window manifest (provenance + staleness)
            artifacts.write_manifest(path, {"LIFETIME_YEARS": LIFETIME_YEARS}, [sql_file],
                                     extra={"rows": int(len(ev)), "patients": int(ev.patient_guid.nunique())})
        except Exception as _me:
            print(f"[events][warn] cache manifest not written ({type(_me).__name__}: {_me})")
        print(f"[events] pulled {len(ev):,} | {ev.patient_guid.nunique():,} patients -> cached {path}")
    return _window(ev, yrs)
