"""
Prostate Cancer Risk Factors — JSON-Packed FE Output
====================================================

Production-ready FE pipeline for the Prostate cancer model. Mirrors lung's
``SNOMED_json_v3/transform_features_json.py`` interface (BQ-in → JSON-packed
DataFrame → BQ-out) and uses the prostate-specific feature engineering
blocks ported from ``Prostate_2.0_1to1/2_Feature_Engineering/``.

Output layout (7 columns — identical to lung):

    patient_guid, sex, patient_ethnicity_16, patient_ethnicity_6,
    patient_age, transformed_features, cancer_class

All ~1300 prostate features (PSA trend, vitals, symptoms, comorbidities,
medication patterns, interactions, acceleration, etc.) are packed into the
single ``transformed_features`` column as a JSON object per patient row.

The output table is day-partitioned in BigQuery; the live API queries the
latest partition for each patient_guid.

Usage
-----
Lifetime / full lookback:
    python transform_features_json.py

Lifetime + windowed (e.g. last 1 month gap = "1mo" model variant):
    python transform_features_json.py --window 1mo

Programmatic:
    from transform_features_json import extract_prostate_features_json
    df = extract_prostate_features_json(csv_or_bq_query, window="1mo")
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timezone
from functools import reduce
from importlib import import_module
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Lung-style analytics helpers (used by FE blocks for value-trend features)
# Our prostate-specific FE blocks (ported from Prostate_2.0_1to1).
# Numbered modules can't be imported with `import` (Python identifier rules),
# so use importlib for stage modules + plain imports for the unnumbered helpers.
from importlib import import_module as _imp
import config
from io_utils import read_table

_prostate_preprocess = _imp('0_preprocess')
_prostate_pipeline   = _imp('1_pipeline_blocks')
_prostate_specific   = _imp('2_prostate_features')
_prostate_cleanup    = _imp('3_cleanup')

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Output layout — lung's 7-col spine + Alex-aligned anchor + cancer_id
# ----------------------------------------------------------------------
IDENTIFIER_COLUMN = 'patient_guid'
DEMOGRAPHIC_COLUMNS = [
    'sex',
    'patient_ethnicity_16',
    'patient_ethnicity_6',
    'patient_age',
]
# Anchor metadata column (semantic: latest event date in the patient's
# available data; for inference = today's most recent event, for training
# cancer cohort ≈ diagnosis_date - months_before). Mirrors Alex's
# `max_event_date` naming.
ANCHOR_COLUMN = 'max_event_date'
# Trailing label columns. `cancer_class` is binary (0/1); `cancer_id` is
# the matched SNOMED cancer_id (NULL for non-cancer) — useful for audit
# trail + cross-cancer joins.
TRAILING_COLUMNS = ['cancer_class', 'cancer_id']
TRAILING_COLUMN  = TRAILING_COLUMNS[0]   # back-compat alias
JSON_COLUMN_NAME = 'transformed_features'

# ----------------------------------------------------------------------
# BigQuery destination
# ----------------------------------------------------------------------
PROJECT_ID = 'prj-cts-ai-dev-sp'
DATASET_ID = 'prediction_emis'
TABLE_PREFIX = 'prostate_features'   # final table is {prefix}_{window} (or just {prefix})


# ======================================================================
# Data ingestion
# ======================================================================
def load_data_from_bigquery(sql_file_path: str, project_id: str = PROJECT_ID) -> pd.DataFrame:
    """
    Run a SQL file against BigQuery and return the result as a DataFrame.

    Mirrors lung/SNOMED_json_v3/transform_features_json.load_data_from_bigquery.
    """
    from google.cloud import bigquery  # type: ignore

    print(f"Reading SQL query from: {sql_file_path}")
    with open(sql_file_path, 'r', encoding='utf-8') as fh:
        query = fh.read()

    client = bigquery.Client(project=project_id)
    print(f"Submitting query to BigQuery (project={project_id})...")
    df = client.query(query).to_dataframe()
    print(f"  loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ======================================================================
# FE orchestration
# ======================================================================
def _preprocess_inmem(raw_df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    """
    Run the prostate preprocess step on an in-memory DataFrame.

    Wraps the file-based ``preprocess.main()`` so we can pipe BQ -> DataFrame
    -> preprocessed without round-tripping through parquet.

    Includes the defensive anchor fallback (derive anchor_date from
    MAX(event_date) per patient if SQL doesn't emit it) — the lung base SQL
    we're adapting from doesn't include anchor metadata.
    """
    df = raw_df.copy()

    # --- Defensive: derive anchor metadata if upstream SQL didn't emit it ---
    if 'ANCHOR_DATE' not in df.columns and 'anchor_date' not in df.columns:
        logger.warning(
            "ANCHOR_DATE missing from BQ result — deriving from MAX(event_date) per patient"
        )
        df.columns = [c.upper() for c in df.columns]
        df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
        df['ANCHOR_DATE'] = df.groupby('PATIENT_GUID')['EVENT_DATE'].transform('max')

    df.columns = [c.upper() for c in df.columns]

    if 'MONTHS_BEFORE_ANCHOR' not in df.columns:
        df['EVENT_DATE']  = pd.to_datetime(df['EVENT_DATE'],  errors='coerce')
        df['ANCHOR_DATE'] = pd.to_datetime(df['ANCHOR_DATE'], errors='coerce')
        df['MONTHS_BEFORE_ANCHOR'] = ((df['ANCHOR_DATE'] - df['EVENT_DATE']).dt.days // 30).astype('Int64')

    if 'DAYS_BEFORE_ANCHOR' not in df.columns:
        df['EVENT_DATE']  = pd.to_datetime(df['EVENT_DATE'],  errors='coerce')
        df['ANCHOR_DATE'] = pd.to_datetime(df['ANCHOR_DATE'], errors='coerce')
        df['DAYS_BEFORE_ANCHOR'] = (df['ANCHOR_DATE'] - df['EVENT_DATE']).dt.days.astype('Int64')

    # Apply the same column-renaming + CATEGORY assignment + TIME_WINDOW
    # logic the file-based preprocess does. Reusing the helpers from
    # preprocess.py directly so logic stays in lockstep.
    obs_map, med_map = _prostate_preprocess.load_mapping(mapping_path)

    df = df.rename(columns={
        'ANCHOR_DATE':          'INDEX_DATE',
        'PATIENT_AGE':          'AGE_AT_INDEX',
        'MONTHS_BEFORE_ANCHOR': 'MONTHS_BEFORE_INDEX',
        'CANCER_CLASS':         'LABEL',
    })

    is_obs = df['EVENT_TYPE'] == 'observation'
    df['CODE_ID'] = np.where(is_obs, df.get('SNOMED_C_T_CONCEPT_ID'), df.get('MED_CODE_ID'))
    df['CODE_ID'] = pd.to_numeric(df['CODE_ID'], errors='coerce').astype('Int64')
    df['TERM']    = np.where(is_obs, df['TERM'], df.get('DRUG_TERM'))

    df['CATEGORY'] = None
    df.loc[is_obs,  'CATEGORY'] = _prostate_preprocess.assign_category(df.loc[is_obs,  'CODE_ID'], obs_map).values
    df.loc[~is_obs, 'CATEGORY'] = _prostate_preprocess.assign_category(df.loc[~is_obs, 'CODE_ID'], med_map).values

    months = pd.to_numeric(df['MONTHS_BEFORE_INDEX'], errors='coerce').fillna(-1).astype(int).values
    df['TIME_WINDOW'] = _prostate_preprocess.assign_time_window(months)

    df = df[df['CATEGORY'].notna() & df['TIME_WINDOW'].notna()].copy()
    return df


def _run_fe_blocks(preprocessed_df: pd.DataFrame, window_label: str, cfg) -> pd.DataFrame:
    """
    Run the prostate-specific FE blocks (4a–4g) and return the feature matrix.

    Mirrors the inner loop of ``Prostate_2.0_1to1/2_Feature_Engineering/0_run_pipeline.run_feature_engineering``
    but in-memory, single-window, no disk round-trip.
    """
    obs_df = preprocessed_df[preprocessed_df['EVENT_TYPE'] == 'observation'].copy()
    med_df = preprocessed_df[preprocessed_df['EVENT_TYPE'] == 'medication'].copy()
    for col in ('VALUE',):
        if col in obs_df.columns:
            obs_df[col] = pd.to_numeric(obs_df[col], errors='coerce')
        if col in med_df.columns:
            med_df[col] = pd.to_numeric(med_df[col], errors='coerce')

    fm = _prostate_pipeline.build_clinical_features(obs_df, med_df, window_label.upper(), cfg)
    fm = fm.join(_prostate_pipeline.build_medication_features(med_df, window_label.upper(), cfg), how='left')
    fm = fm.join(_prostate_pipeline.build_interaction_features(fm, cfg),                    how='left')
    fm = fm.join(_prostate_pipeline.build_advanced_features(obs_df, med_df, fm, window_label.upper(), cfg), how='left')
    fm = fm.join(_prostate_pipeline.extract_maximum_features(obs_df, med_df, fm, window_label.upper(), cfg), how='left')
    fm = fm.join(_prostate_specific.build_cancer_specific_features(obs_df, med_df, fm, window_label.upper(), cfg), how='left')
    fm = fm.join(_prostate_pipeline.build_new_signal_features(obs_df, med_df, fm, window_label.upper(), cfg), how='left')
    fm = fm.join(_prostate_pipeline.build_trend_features(obs_df, med_df, fm, window_label.upper(), cfg),       how='left')
    fm = fm.join(_prostate_pipeline.build_acceleration_features(obs_df, med_df, fm, window_label.upper(), cfg), how='left')

    fm = fm.fillna(0).replace([np.inf, -np.inf], 0)
    fm = fm.loc[:, ~fm.columns.duplicated()]
    return fm


# ======================================================================
# JSON packing — IDENTICAL to lung's pack_features_as_json
# ======================================================================
def _to_json_safe(value):
    """Coerce a single cell to a JSON-serialisable Python primitive."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _row_to_json(row, feature_columns):
    payload = {col: _to_json_safe(row[col]) for col in feature_columns}
    return json.dumps(payload, ensure_ascii=False, default=str)


def _build_feature_schema_from_wide(
    wide_df: pd.DataFrame,
    json_column_name: str = JSON_COLUMN_NAME,
    cancer: str = 'prostate',
    window: str = '1mo',
    pipeline_version: str = 'prod_ready_v1',
) -> dict:
    """Build the schema-sidecar dict from a wide feature DataFrame.

    Schema is captured BEFORE JSON packing (where per-column dtypes are
    still preserved). Output matches the format used by the prostate
    JSON pilot at ``Prostate_2.0_1to1/3_Modeling/json_pilot/schemas/``.
    """
    # All spine columns that go alongside the JSON payload — must match what
    # pack_features_as_json() emits, otherwise the sidecar's spine_columns
    # list won't match the actual parquet schema.
    spine_candidates = [IDENTIFIER_COLUMN, *DEMOGRAPHIC_COLUMNS,
                        ANCHOR_COLUMN, *TRAILING_COLUMNS]
    spine_cols = [c for c in spine_candidates if c in wide_df.columns]
    feature_cols = [c for c in wide_df.columns if c not in set(spine_cols)]

    def _infer_dtype(series: pd.Series) -> str:
        # Detect category BEFORE dropna — pandas drops the .cat accessor on subsets.
        if isinstance(series.dtype, pd.CategoricalDtype):
            return 'category'
        s = series.dropna()
        if len(s) == 0:
            return 'float'
        if pd.api.types.is_bool_dtype(s):
            return 'bool'
        if pd.api.types.is_integer_dtype(s):
            return 'int'
        if pd.api.types.is_float_dtype(s):
            return 'float'
        if pd.api.types.is_datetime64_any_dtype(s):
            return 'date'
        return 'string'

    features = {}
    for c in feature_cols:
        dt = _infer_dtype(wide_df[c])
        entry = {'dtype': dt}
        # For categoricals, include the declared categories so downstream
        # consumers (and label encoders) have stable, deterministic class lists
        # — matches Alex's lung sidecar shape exactly.
        if dt == 'category':
            entry['categories'] = list(wide_df[c].cat.categories)
        features[c] = entry
    return {
        'cancer':           cancer,
        'window':           window,
        'pipeline_version': pipeline_version,
        'generated_at':     datetime.now(timezone.utc).isoformat() + 'Z',
        'n_patients':       int(len(wide_df)),
        'n_features':       len(features),
        'spine_columns':    spine_cols,
        'json_column':      json_column_name,
        'features':         features,
    }


def write_schema_sidecar(schema_dict: dict, out_path: 'str | Path') -> Path:
    """Write the schema dict to a JSON sidecar file (UTF-8, pretty-printed)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(schema_dict, fh, indent=2, ensure_ascii=False)
    return out_path


def pack_features_as_json(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide feature DataFrame to the JSON-packed BQ layout.

    Output column order:
        patient_guid, sex, patient_ethnicity_16, patient_ethnicity_6,
        patient_age, max_event_date, transformed_features, cancer_class, cancer_id

    (partition_date is appended at BQ-write time, not here.)
    """
    spine_lower = [IDENTIFIER_COLUMN, *DEMOGRAPHIC_COLUMNS, ANCHOR_COLUMN, *TRAILING_COLUMNS]
    if wide_df.empty:
        return pd.DataFrame(columns=[IDENTIFIER_COLUMN, *DEMOGRAPHIC_COLUMNS,
                                     ANCHOR_COLUMN, JSON_COLUMN_NAME, *TRAILING_COLUMNS])

    # Lowercase column names so the spine columns match the contract.
    df = wide_df.copy()
    spine_upper = {c.upper() for c in spine_lower}
    df.columns = [c.lower() if c.upper() in spine_upper else c for c in df.columns]

    front_cols    = [c for c in [IDENTIFIER_COLUMN, *DEMOGRAPHIC_COLUMNS, ANCHOR_COLUMN] if c in df.columns]
    trailing_cols = [c for c in TRAILING_COLUMNS if c in df.columns]
    reserved      = set(front_cols) | set(trailing_cols)
    feature_cols  = [c for c in df.columns if c not in reserved]

    packed = df[front_cols].copy()
    packed[JSON_COLUMN_NAME] = df.apply(lambda row: _row_to_json(row, feature_cols), axis=1)
    for c in trailing_cols:
        packed[c] = df[c].values
    return packed[[*front_cols, JSON_COLUMN_NAME, *trailing_cols]]


# ======================================================================
# BigQuery sink — IDENTICAL pattern to lung's write_to_bigQ
# ======================================================================
def write_to_bigQ(df: pd.DataFrame, table_suffix: str = '') -> str:
    """Upload the JSON-packed DataFrame to BigQuery."""
    from google.cloud import bigquery  # type: ignore

    df = df.copy()
    df['partition_date'] = datetime.now(timezone.utc).date()

    table_name = f'{TABLE_PREFIX}{("_" + table_suffix) if table_suffix else ""}'
    table_id   = f'{PROJECT_ID}.{DATASET_ID}.{table_name}'

    # `transformed_features` uses BigQuery native JSON type (not STRING) —
    # validates JSON at write time, supports JSON_VALUE / JSON_QUERY pushdown
    # for any future cohort / drift / audit queries, and matches the contract
    # used by Alex's lung feature table.
    schema = [
        bigquery.SchemaField('patient_guid',          'STRING'),
        bigquery.SchemaField('sex',                   'STRING'),
        bigquery.SchemaField('patient_ethnicity_16',  'STRING'),
        bigquery.SchemaField('patient_ethnicity_6',   'STRING'),
        bigquery.SchemaField('patient_age',           'INT64'),
        bigquery.SchemaField('max_event_date',        'DATE'),
        bigquery.SchemaField('transformed_features',  'JSON'),
        bigquery.SchemaField('cancer_class',          'INT64'),
        bigquery.SchemaField('cancer_id',             'STRING'),
        bigquery.SchemaField('partition_date',        'DATE'),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field='partition_date',
        ),
    )

    client = bigquery.Client(project=PROJECT_ID)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {job.output_rows} rows into {table_id}")
    return table_id


# ======================================================================
# High-level entry points
# ======================================================================
def extract_prostate_features_json(
    csv_or_bq: 'pd.DataFrame | str',
    window: str = '1mo',
    mapping_path: 'Path | None' = None,
    schema_sidecar_path: 'str | Path | None' = None,
) -> pd.DataFrame:
    """
    Run the full prostate FE pipeline end-to-end and return the JSON-packed
    DataFrame (one row per patient).

    Args:
        csv_or_bq: Either an in-memory DataFrame (already loaded from BQ),
            a path to a CSV file, or a path to a .sql file (executed against BQ).
        window: Prediction window label ('1mo' / '3mo' / '6mo' / '12mo') — used
            in TIME_WINDOW assignment and the BQ table suffix.
        mapping_path: Path to code_category_mapping_v2.json. Defaults to the
            file in this package's codelists/ directory.
        schema_sidecar_path: If provided, write a schema sidecar JSON file
            describing every feature key + dtype (per the proposal's
            Condition #1). Captured from the wide DataFrame BEFORE JSON
            packing so dtypes are reliable.
    """
    if mapping_path is None:
        mapping_path = Path(__file__).parent / 'codelists' / 'code_category_mapping_v2.json'

    # --- Load raw events -------------------------------------------------
    if isinstance(csv_or_bq, pd.DataFrame):
        raw = csv_or_bq
    elif isinstance(csv_or_bq, (str, Path)) and str(csv_or_bq).endswith('.sql'):
        raw = load_data_from_bigquery(str(csv_or_bq))
    else:
        raw = read_table(csv_or_bq, low_memory=False)

    # --- Preprocess (defensive anchor + categorize + time-window) --------
    preprocessed = _preprocess_inmem(raw, mapping_path)

    # --- Run FE blocks ---------------------------------------------------
    fm = _run_fe_blocks(preprocessed, window_label=window, cfg=config)

    # --- Cleanup ---------------------------------------------------------
    cleaned = _prostate_cleanup.cleanup_features(fm, window.upper(), config)

    # --- Re-attach demographics + anchor + label that cleanup dropped (non-numeric) ---
    # cleanup.py step 5a strips non-numeric columns (sex / ethnicity / dates).
    # For the JSON-packed output we need them back as front + trailing columns.
    # Carries: SEX, PATIENT_ETHNICITY_16/_6, AGE_AT_INDEX, INDEX_DATE
    # (renamed → max_event_date), LABEL (→ cancer_class), CANCER_ID.
    demo_cols = ['PATIENT_GUID', 'SEX', 'PATIENT_ETHNICITY_16', 'PATIENT_ETHNICITY_6',
                 'AGE_AT_INDEX', 'INDEX_DATE', 'LABEL', 'CANCER_ID']
    available = [c for c in demo_cols if c in preprocessed.columns]
    demo = (preprocessed[available]
            .drop_duplicates(subset=['PATIENT_GUID'])
            .set_index('PATIENT_GUID'))
    cleaned_with_demo = cleaned.join(demo, how='left').reset_index().rename(columns={
        'index':                'patient_guid',
        'PATIENT_GUID':         'patient_guid',
        'SEX':                  'sex',
        'PATIENT_ETHNICITY_16': 'patient_ethnicity_16',
        'PATIENT_ETHNICITY_6':  'patient_ethnicity_6',
        'AGE_AT_INDEX':         'patient_age',
        'INDEX_DATE':           'max_event_date',   # NEW — Alex-aligned naming
        'LABEL':                'cancer_class',
        'CANCER_ID':            'cancer_id',        # NEW — audit trail
    })

    # --- Build + persist schema sidecar (BEFORE packing — keeps dtypes intact) ---
    if schema_sidecar_path is not None:
        schema = _build_feature_schema_from_wide(
            cleaned_with_demo,
            json_column_name=JSON_COLUMN_NAME,
            window=window,
        )
        written = write_schema_sidecar(schema, schema_sidecar_path)
        print(f"Wrote schema sidecar: {written}  ({schema['n_features']} features)")

    # --- Pack to JSON ----------------------------------------------------
    return pack_features_as_json(cleaned_with_demo)


def main(sql_file: 'str | None' = None, window: str = '1mo',
         project_id: str = PROJECT_ID, upload_to_bq: bool = True) -> str:
    """End-to-end entry point: SQL -> FE -> JSON pack -> BQ upload + local CSV."""
    if sql_file is None:
        sql_file = str(Path(__file__).resolve().parent.parent / '1_SQL' / 'unified_prostate_extraction.sql')

    print(f"\n{'='*80}")
    print(f"PROSTATE FE — JSON-PACKED OUTPUT")
    print(f"  SQL:    {sql_file}")
    print(f"  Window: {window}")
    print(f"{'='*80}\n")

    sidecar_path = f'prostate_features_{window}_schema.json'
    packed = extract_prostate_features_json(
        sql_file,
        window=window,
        schema_sidecar_path=sidecar_path,
    )

    out_path = f'prostate_features_{window}_json.csv'
    packed.to_csv(out_path, index=False)
    print(f"\nWrote local CSV:      {out_path}  ({len(packed):,} rows)")
    print(f"Wrote schema sidecar: {sidecar_path}")

    table_id = ''
    if upload_to_bq:
        table_id = write_to_bigQ(packed, table_suffix=window)

    print(f"\n{'='*80}")
    print(f"DONE — {len(packed)} patients packed")
    print(f"  local CSV: {out_path}")
    if table_id:
        print(f"  BQ table:  {table_id}")
    print(f"{'='*80}\n")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prostate FE — JSON-packed output')
    parser.add_argument('--sql',    type=str, default=None,
                        help='Path to .sql file (default: sql/unified_prostate_extraction.sql)')
    parser.add_argument('--window', type=str, default='1mo',
                        choices=['1mo', '3mo', '6mo', '12mo'],
                        help='Prediction window (default: 1mo)')
    parser.add_argument('--no-bq',  action='store_true', help='Skip BigQuery upload (write only local CSV)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main(sql_file=args.sql, window=args.window, upload_to_bq=not args.no_bq)
