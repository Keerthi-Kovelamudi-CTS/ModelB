# ═══════════════════════════════════════════════════════════════
# STEP 6 — WRITE FE FEATURES TO BIGQUERY FEATURE STORE
#
# Mirrors lung's write_to_bigQ() pattern from
#   lung/SNOMED_json_v3/transform_features_json.py
#
# Reads the cleaned feature matrix produced by step 5 (cleanup) and
# uploads each window to a day-partitioned BigQuery table:
#
#   {PROJECT_ID}.{DATASET_ID}.{cancer}_features_{window}
#
# Day-partitioning + WRITE_APPEND keep each daily snapshot as a
# separate partition. The live API queries the latest partition.
# Apply a partition expiration TTL (e.g. 30 days) on the BQ table
# console-side to bound storage growth.
#
# Usage (from 0_run_pipeline.py):
#   python 0_run_pipeline.py --step bq
# ═══════════════════════════════════════════════════════════════

import logging
from datetime import datetime, timezone

import pandas as pd
from google.cloud import bigquery

import config
from io_utils import read_table

logger = logging.getLogger(__name__)

# ─── BigQuery destination ────────────────────────────────────
# Override via env vars or by passing args to write_features_to_bq().
PROJECT_ID = 'prj-cts-ai-dev-sp'
DATASET_ID = 'prediction_emis'


def write_features_to_bq(
    df: pd.DataFrame,
    cancer: str,
    window: str,
    project_id: str = PROJECT_ID,
    dataset_id: str = DATASET_ID,
    write_disposition: str = 'WRITE_APPEND',
) -> str:
    """Upload one window's feature matrix to BigQuery.

    Adds a ``partition_date`` column with today's UTC date. The destination
    table is created on first write (day-partitioned on ``partition_date``).

    Args:
        df: Feature matrix, one row per patient.
        cancer: Cancer name (lowercased for the table name).
        window: Prediction window string (``1mo`` / ``3mo`` / ``6mo`` / ``12mo``).
        project_id: GCP project containing the destination dataset.
        dataset_id: BigQuery dataset name (the dataset must already exist).
        write_disposition: ``'WRITE_APPEND'`` keeps daily history (default —
            recommended for a feature store). ``'WRITE_TRUNCATE'`` overwrites.

    Returns:
        Fully-qualified table id that was written.
    """
    df = df.copy()
    df['partition_date'] = datetime.now(timezone.utc).date()

    # parquet saves PATIENT_GUID as the index — promote to a column
    if df.index.name and str(df.index.name).upper() in ('PATIENT_GUID', 'PATIENTID'):
        df = df.reset_index()

    table_name = f'{cancer.lower()}_features_{window}'
    table_id = f'{project_id}.{dataset_id}.{table_name}'

    client = bigquery.Client(project=project_id)

    job_config = bigquery.LoadJobConfig(
        write_disposition=getattr(bigquery.WriteDisposition, write_disposition),
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field='partition_date',
        ),
        # FE emits hundreds of columns of varying types; let BQ infer the
        # schema from the DataFrame rather than hand-writing it. A future
        # iteration may freeze the schema once feature names are stable.
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logger.info(
        f"  [BQ] {table_id}: loaded {job.output_rows:,} rows × "
        f"{len(df.columns)} cols (partition_date={df['partition_date'].iloc[0]})"
    )
    return table_id


def _read_window_matrix(cfg, window: str):
    """Return the cleaned feature matrix for one window, or None if missing."""
    candidates = [
        cfg.CLEANUP_RESULTS / window / f'feature_matrix_clean_{window}.parquet',
        cfg.FE_RESULTS     / window / f'feature_matrix_final_{window}.parquet',
    ]
    for path in candidates:
        if path.exists():
            return read_table(path, low_memory=False), path
    return None, None


def run_write_to_bq(cfg) -> dict:
    """Step 6 entry point — write each window's matrix to BigQuery.

    Returns ``{window: table_id}`` for the windows successfully uploaded.
    Missing matrices and per-window failures are logged but don't stop the
    other windows.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"  STEP 6: WRITE TO BIGQUERY FEATURE STORE — {cfg.CANCER_NAME.upper()}")
    logger.info(f"{'='*70}")

    written: dict[str, str] = {}
    for window in cfg.WINDOWS:
        df, path = _read_window_matrix(cfg, window)
        if df is None:
            logger.warning(f"  [{window}] No feature matrix found — skip")
            continue
        logger.info(f"  [{window}] Loaded {df.shape[0]:,} × {df.shape[1]} from {path.name}")
        try:
            table_id = write_features_to_bq(
                df,
                cancer=cfg.CANCER_NAME,
                window=window,
            )
            written[window] = table_id
        except Exception as e:
            logger.error(f"  [{window}] BQ write FAILED: {e}")

    logger.info(f"\n  STEP 6 COMPLETE: {len(written)}/{len(cfg.WINDOWS)} windows uploaded")
    return written


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_write_to_bq(config)
