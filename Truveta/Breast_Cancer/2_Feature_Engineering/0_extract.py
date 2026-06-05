"""
Extract per-(patient, code) aggregates from BigQuery.

Reads `truveta_gold.breast_cohort_events` joined to the curated codelist and
aggregates event-level data into one row per (patient_id, code_id) pair, with
count / value_mean / value_last / duration_sum. This is the "long format" used
by 1_build_features.py to pivot into the wide patient × category feature matrix.

Aggregating in BQ (~10s) is much cheaper than pulling 47M event rows to local.

Usage:
    python 0_extract.py
"""

import argparse
import sys
import pandas as pd

from google.cloud import bigquery
from google.auth import compute_engine

import config
from io_utils import write_table


def _bq_client(project: str) -> bigquery.Client:
    # Metadata-server creds work from any subprocess context (nohup, cron, etc).
    # `bq` CLI auth is brittle under non-interactive shells, so we bypass it.
    return bigquery.Client(project=project, credentials=compute_engine.Credentials())


def main(window: str, cohort_table: str = None):
    # Allow override (e.g. holdout workflow passes the holdout cohort table).
    if cohort_table is None:
        cohort_table = config.COHORT_TABLES[window]
    # Layer-G trend split is patient-relative (midpoint of each patient's own
    # event timeline, computed in BQ) — no fixed-window split parameter needed.
    raw_dir = config.RAW_DIR / window
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load curated category codelist (TSV: code_type, category, code_id, term).
    if not config.CODELIST_PATH.exists():
        sys.exit(f"Codelist not found: {config.CODELIST_PATH}")

    sep = '\t' if config.CODELIST_PATH.suffix in ('.tsv', '.txt') else ','
    codelist = pd.read_csv(config.CODELIST_PATH, sep=sep)

    required = {'code_type', 'category', 'code_id', 'term'}
    missing = required - set(codelist.columns)
    if missing:
        sys.exit(f"Codelist missing required columns: {missing}\n"
                 f"Has: {list(codelist.columns)}")

    codelist['code_id'] = codelist['code_id'].astype('Int64')
    codelist = codelist.dropna(subset=['code_id'])

    # Derive feature_type from code_type (rxnorm → medication, else observation).
    codelist['feature_type'] = codelist['code_type'].str.lower().map(
        lambda v: 'medication' if v == 'rxnorm' else 'observation'
    )

    print(f"Loaded codelist: {len(codelist)} codes across "
          f"{codelist['category'].nunique()} categories")
    print(f"  vocabularies: {codelist['code_type'].value_counts().to_dict()}")

    code_ids_sql = ','.join(str(int(c)) for c in codelist['code_id'].unique())

    sql = f"""
    WITH codelist AS (
      SELECT code_id
      FROM UNNEST([{code_ids_sql}]) AS code_id
    ),
    events AS (
      SELECT
        ce.patient_id,
        COALESCE(ce.concept_id, ce.med_concept_id) AS code_id,
        ce.event_type,
        ce.event_date,
        ce.value_numeric,
        ce.duration,
        -- patient-relative timeline bounds (lung-style): split each patient's OWN
        -- event span in half, so short-history patients still get a within-history
        -- trend rather than a false "all-recent ⇒ worsening" artifact.
        MIN(ce.event_date) OVER (PARTITION BY ce.patient_id) AS pt_first,
        MAX(ce.event_date) OVER (PARTITION BY ce.patient_id) AS pt_last
      FROM `{cohort_table}` ce
      JOIN codelist cl
        ON COALESCE(ce.concept_id, ce.med_concept_id) = cl.code_id
    )
    SELECT
      patient_id,
      code_id,
      ANY_VALUE(event_type)                                            AS event_type,
      COUNT(*)                                                         AS count,
      -- recent vs older half of the patient's OWN event timeline (midpoint of pt_first..pt_last)
      COUNTIF(event_date >= DATE_ADD(pt_first, INTERVAL DIV(DATE_DIFF(pt_last, pt_first, DAY), 2) DAY)) AS count_h2,
      COUNTIF(event_date <  DATE_ADD(pt_first, INTERVAL DIV(DATE_DIFF(pt_last, pt_first, DAY), 2) DAY)) AS count_h1,
      AVG(value_numeric)                                               AS value_mean,
      ARRAY_AGG(value_numeric IGNORE NULLS ORDER BY event_date DESC LIMIT 1)[SAFE_OFFSET(0)]
                                                                       AS value_last,
      SUM(IFNULL(duration, 0))                                         AS duration_sum,
      MIN(event_date)                                                  AS first_event,
      MAX(event_date)                                                  AS last_event
    FROM events
    GROUP BY patient_id, code_id
    """

    raw_csv = raw_dir / 'breast_patient_code_aggregates.csv'
    client = _bq_client(config.BQ_PROJECT)

    print(f"[{window}] Running BQ aggregation -> {raw_csv}")
    try:
        df = client.query(sql).to_dataframe()
    except Exception as e:
        sys.exit(f"BQ aggregation failed: {e!r}")
    df.to_csv(raw_csv, index=False)
    print(f"  wrote {len(df):,} (patient, code) aggregates")
    print(f"  unique patients: {df['patient_id'].nunique():,}")
    print(f"  unique codes:    {df['code_id'].nunique():,}")

    # Spine: per-patient demographics, anchor, label.
    spine_sql = f"""
    SELECT DISTINCT
      patient_id,
      sex,
      ethnicity,
      state_or_province,
      patient_age,
      anchor_date,
      cancer_class AS label,
      cancer_id
    FROM `{cohort_table}`
    """
    spine_csv = raw_dir / 'breast_spine.csv'
    print(f"[{window}] Running BQ spine extract -> {spine_csv}")
    try:
        spine = client.query(spine_sql).to_dataframe()
    except Exception as e:
        sys.exit(f"BQ spine extract failed: {e!r}")
    spine.to_csv(spine_csv, index=False)
    print(f"  wrote spine for {len(spine):,} patients")

    # Cache codelist (with category) for 1_build_features.py
    write_table(codelist, raw_dir / 'breast_codelist.parquet')
    print(f"[{window}] Done.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', required=True,
                    help=f"Which cohort window to extract (e.g. {config.WINDOWS}, or a custom label like 'holdout_1mo')")
    ap.add_argument('--cohort-table', default=None,
                    help="Override config.COHORT_TABLES[window] — useful for holdout cohorts. "
                         "Format: 'project.dataset.table'")
    args = ap.parse_args()
    main(args.window, cohort_table=args.cohort_table)
