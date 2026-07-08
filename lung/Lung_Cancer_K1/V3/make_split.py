"""Step 0 (split-first) — create the canonical train/valid/test split ONCE per horizon.

Pulls the cohort MEMBERSHIP (distinct patient_guid + cancer_class) from the cohort SQL and writes
`{SPLIT_DIR}/lung_{h}_{FE_YEARS}yr_split.parquet` via splits.load_or_make. Every later step (code
scoring, codelist, FE, stability-selection, model training) loads this file and filters/assigns by
patient_guid — guaranteeing identical, reproducible train/valid/test patients and no leakage.

The patient SET is independent of the lookback window (years_before filters events, not the cohort),
so one split per horizon serves every lookback experiment.

Run once (VM w/ BigQuery):  python make_split.py            # both horizons
                            python make_split.py 12mo       # one horizon
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # for config + splits
import config as C
import splits
from google.cloud import bigquery


# Build membership over the WIDEST lookback we will ever run, so the split is a SUPERSET that covers
# every (smaller) window's FE patients — a patient appears in FE only if they have an event in the
# window, so a longer lookback has MORE patients. Using the max window guarantees no FE patient is ever
# missing a split assignment. (A patient in the split but absent from a given window's matrix is fine.)
MEMBERSHIP_YEARS = max(list(C.FE_WINDOWS) + [C.SCORING_YEARS_BEFORE])


def cohort_membership_sql(h):
    """DISTINCT (patient_guid, cancer_class) over the horizon's cohort SQL at the widest lookback."""
    p = os.path.join(HERE, "0_SQL", f"{h}_1to1.sql")
    sql = open(p, encoding="utf-8").read().rstrip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    sql = re.sub(r"ORDER\s+BY\s+patient_guid\s*$", "", sql, flags=re.I).rstrip()
    sql = re.sub(r"\d+(\s+AS years_before\b)", rf"{MEMBERSHIP_YEARS}\1", sql, count=1)
    return f"SELECT DISTINCT patient_guid, cancer_class FROM (\n{sql}\n)"


def main():
    horizons = [a for a in sys.argv[1:] if a in ("12mo", "1mo")] or list(C.HORIZONS)
    client = bigquery.Client(project=C.GCP_PROJECT)
    for h in horizons:
        print(f"\n=== {h}: building canonical split ===")
        df = client.query(cohort_membership_sql(h)).to_dataframe()
        df["cancer_class"] = df["cancer_class"].astype(int)
        print(f"  cohort: {len(df):,} patients ({int(df['cancer_class'].sum())} cancer)")
        splits.load_or_make(h, df["patient_guid"], df["cancer_class"])
    print(f"\nDone. Canonical splits in {C.SPLIT_DIR}/ — every step now loads these (split-first).")


if __name__ == "__main__":
    main()
