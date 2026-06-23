"""Step 0 (split-first) — create the canonical train/valid/test split ONCE per horizon.

Derives the cohort MEMBERSHIP (distinct patient_guid + cancer_class) from the shared LIFETIME cohort
events (events.load_cohort_events; pulled from BigQuery once per horizon and cached — see events.py)
and writes `{SPLIT_DIR}/lung_{h}_split.parquet` via splits.load_or_make. No separate BigQuery query.

Membership is taken at the LIFETIME window (the widest), so the split is a SUPERSET covering every
smaller lookback's FE patients — a patient appears in FE only if they have an event in the window, so a
longer lookback has MORE patients. (A patient in the split but absent from a given window's matrix is fine.)
Running this first also triggers the one-time lifetime pull that every later step then reuses.

Run once (VM w/ BigQuery):  python make_split.py            # both horizons
                            python make_split.py 12mo       # one horizon
                            python make_split.py --force    # REBUILD — but if a split ALREADY exists this
                                                            # is REFUSED unless ALLOW_SPLIT_OVERWRITE=1 is
                                                            # also set (the existing split is backed up
                                                            # timestamped first). This protects the shared
                                                            # canonical split (and the free-text pipeline
                                                            # aligned to it) from an accidental --force.
                            ALLOW_SPLIT_OVERWRITE=1 python make_split.py --force   # deliberate re-create
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # for config + splits + events
import config as C
import splits
import events


def main():
    horizons = C.horizons_from_argv(sys.argv[1:])   # hard-errors on an unrecognized token (no silent default)
    force = "--force" in sys.argv[1:]               # rebuild even if a split already exists (cohort changed)
    for h in horizons:
        print(f"\n=== {h}: building canonical split{' (--force rebuild)' if force else ''} ===")
        ev = events.load_cohort_events(h, events.LIFETIME_YEARS)   # lifetime (one-time BQ pull, cached)
        pat = ev[["patient_guid", "cancer_class"]].drop_duplicates("patient_guid").copy()
        pat["cancer_class"] = pat["cancer_class"].astype(int)
        print(f"  cohort: {len(pat):,} patients ({int(pat['cancer_class'].sum())} cancer)")
        splits.load_or_make(h, pat["patient_guid"], pat["cancer_class"], force=force)
    print(f"\nDone. Canonical splits in {C.SPLIT_DIR}/ — every step now loads these (split-first).")


if __name__ == "__main__":
    main()
