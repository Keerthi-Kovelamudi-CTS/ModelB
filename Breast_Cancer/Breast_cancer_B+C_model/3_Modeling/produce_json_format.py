"""
produce_json_format.py — convert a flat feature matrix into JSON-column format.

This is a PILOT script for migrating to Alex's proposed JSON-column architecture.
It takes a flat parquet/CSV (the output of our FE pipeline) and produces:

  1. A JSON-column CSV  (one row per patient: spine cols + 1 'snomed_features' JSON col)
  2. A JSON-column parquet  (same structure, much smaller)
  3. A schema sidecar (lists every feature key + dtype)
  4. A small comparison report (sizes, dtype counts, etc.)

The output mirrors what Alex's BigQuery output WOULD look like post-ADR.
We can then load via _load_features.py to verify modeling still works.

USAGE
=====

    python3 produce_json_format.py \\
        path/to/feature_matrix_final_1mo.parquet \\
        --out-dir path/to/json_pilot/ \\
        --cancer breast \\
        --window 1mo

Outputs in --out-dir:
    breast_1mo_jsonformat.csv       # post-ADR style CSV
    breast_1mo_jsonformat.parquet   # post-ADR style parquet (recommended)
    breast_1mo_schema.json          # feature schema sidecar
    breast_1mo_conversion_report.md # before/after comparison

WHAT GOES IN JSON vs STAYS FLAT
================================

Spine columns (kept FLAT):
    PATIENT_GUID, AGE_AT_INDEX, SEX, INDEX_DATE, EVENT_DATE,
    LABEL, ANCHOR_DATE, ANCHOR_YEAR, PATIENT_ETHNICITY, CANCER_ID,
    CANCER_CLASS, max_event_date, patient_shard

All other columns → packed into the snomed_features JSON column.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that should ALWAYS stay as flat top-level columns (not in JSON)
DEFAULT_SPINE_COLS = {
    "PATIENT_GUID", "patient_guid",
    "AGE_AT_INDEX", "patient_age", "PATIENT_AGE",
    "SEX", "sex",
    "INDEX_DATE", "EVENT_DATE", "ANCHOR_DATE", "ANCHOR_YEAR", "anchor_date",
    "max_event_date",
    "PATIENT_ETHNICITY", "patient_ethnicity",
    "CANCER_ID", "cancer_id",
    "CANCER_CLASS", "cancer_class",
    "LABEL",
    "patient_shard",
}

JSON_COLUMN_NAME = "snomed_features"


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def produce_json_format(
    flat_path,
    out_dir,
    cancer: str = "breast",
    window: str = "1mo",
    spine_cols: Optional[set] = None,
    write_csv: bool = True,
    write_parquet: bool = True,
    pipeline_version: str = "v4_1to1",
):
    """Convert a flat feature matrix to JSON-column format.

    Parameters
    ----------
    flat_path : str | Path
        Input parquet or CSV file (the FE pipeline's flat output).
    out_dir : str | Path
        Output directory for all generated artifacts.
    cancer : str
        Cancer name (used in filenames + manifest).
    window : str
        Window label (used in filenames + manifest).
    spine_cols : set, optional
        Columns to keep flat (not put in JSON). Default: DEFAULT_SPINE_COLS.

    Returns
    -------
    dict — paths to generated artifacts + size stats.
    """
    flat_path = Path(flat_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spine_cols = spine_cols if spine_cols is not None else DEFAULT_SPINE_COLS

    # ─── Load the flat input ───
    logger.info(f"Loading flat input: {flat_path}")
    if flat_path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(flat_path)
    elif flat_path.suffix == ".csv":
        df = pd.read_csv(flat_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported input format: {flat_path.suffix}")

    # If PATIENT_GUID was the index, restore as a column
    if df.index.name == "PATIENT_GUID":
        df = df.reset_index()

    n_rows, n_cols = df.shape
    logger.info(f"  loaded: {n_rows:,} rows × {n_cols} columns")

    # ─── Partition columns into spine vs features ───
    spine = [c for c in df.columns if c in spine_cols]
    feature_cols = [c for c in df.columns if c not in spine_cols]
    logger.info(f"  spine columns ({len(spine)}): {spine}")
    logger.info(f"  feature columns ({len(feature_cols)}): packed into JSON")

    # ─── Build the JSON column row-by-row ───
    logger.info(f"  packing {len(feature_cols)} features into JSON column...")
    features_only = df[feature_cols]

    def _to_native(v):
        """Convert any numpy/pandas scalar to a JSON-serializable native Python type."""
        if pd.isna(v):
            return None
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return v

    json_strings = features_only.apply(
        lambda row: json.dumps({k: _to_native(v) for k, v in row.items()
                                if not pd.isna(v)},      # skip NULL keys → smaller JSON
                                separators=(',', ':')),  # compact format
        axis=1
    )

    # Assemble the JSON-column DataFrame
    out_df = pd.concat(
        [df[spine].reset_index(drop=True),
         pd.Series(json_strings.values, name=JSON_COLUMN_NAME)],
        axis=1
    )

    # ─── Write outputs ───
    csv_path = parquet_path = None
    if write_csv:
        csv_path = out_dir / f"{cancer}_{window}_jsonformat.csv"
        logger.info(f"  writing CSV: {csv_path}")
        out_df.to_csv(csv_path, index=False)

    if write_parquet:
        parquet_path = out_dir / f"{cancer}_{window}_jsonformat.parquet"
        logger.info(f"  writing Parquet: {parquet_path}")
        out_df.to_parquet(parquet_path, index=False)

    # ─── Build schema sidecar ───
    schema = {
        "cancer":           cancer,
        "window":           window,
        "pipeline_version": pipeline_version,
        "generated_at":     datetime.utcnow().isoformat() + "Z",
        "n_patients":       int(n_rows),
        "n_features":       len(feature_cols),
        "spine_columns":    spine,
        "json_column":      JSON_COLUMN_NAME,
        "features":         {},
    }
    for col in feature_cols:
        dtype_str = str(df[col].dtype)
        if dtype_str.startswith(("int", "Int")):
            dt = "int"
        elif dtype_str.startswith("float"):
            dt = "float"
        elif dtype_str == "bool" or dtype_str == "boolean":
            dt = "bool"
        elif "datetime" in dtype_str:
            dt = "date"
        else:
            dt = "string"
        schema["features"][col] = {"dtype": dt}

    schema_path = out_dir / f"{cancer}_{window}_schema.json"
    logger.info(f"  writing schema sidecar: {schema_path}")
    schema_path.write_text(json.dumps(schema, indent=2))

    # ─── Build comparison report ───
    flat_size = flat_path.stat().st_size
    csv_size = csv_path.stat().st_size if csv_path else 0
    parq_size = parquet_path.stat().st_size if parquet_path else 0
    schema_size = schema_path.stat().st_size

    dtype_counts_in = df.dtypes.value_counts().to_dict()
    dtype_counts_out_schema = pd.Series(
        [v["dtype"] for v in schema["features"].values()]
    ).value_counts().to_dict()

    report_path = out_dir / f"{cancer}_{window}_conversion_report.md"
    report = f"""# JSON-format conversion report — {cancer} {window}

Generated: {schema['generated_at']}

## Input
- File: `{flat_path.name}`
- Size: **{flat_size/1e6:.2f} MB**
- Shape: **{n_rows:,} rows × {n_cols} columns**
- Dtype distribution: {dtype_counts_in}

## Output
| Artifact | Path | Size |
|---|---|---|
| JSON-column CSV     | `{csv_path.name if csv_path else '—'}`     | {csv_size/1e6:.2f} MB |
| JSON-column Parquet | `{parquet_path.name if parquet_path else '—'}` | {parq_size/1e6:.2f} MB |
| Schema sidecar      | `{schema_path.name}`                        | {schema_size/1024:.1f} KB |

## Column split
- **Spine** ({len(spine)} flat columns): `{', '.join(spine)}`
- **Features** ({len(feature_cols)} columns) → packed into single `{JSON_COLUMN_NAME}` JSON column

## Schema sidecar dtype distribution
{dtype_counts_out_schema}

## Storage comparison
| Format | Size | Ratio vs flat input |
|---|---|---|
| Flat parquet (input) | {flat_size/1e6:.2f} MB | 1.00× |
| JSON-column CSV      | {csv_size/1e6:.2f} MB  | {csv_size/flat_size:.2f}× |
| JSON-column Parquet  | {parq_size/1e6:.2f} MB | {parq_size/flat_size:.2f}× |

**Recommendation**: ship the Parquet version to Alex. CSV is human-readable
but ~5× larger than parquet for the same content.

## Validation
Run `_load_features.py` against the JSON-column file and confirm the
loaded DataFrame matches the original flat DataFrame value-by-value.
"""
    report_path.write_text(report)
    logger.info(f"  wrote report: {report_path}")

    return {
        "csv_path":       csv_path,
        "parquet_path":   parquet_path,
        "schema_path":    schema_path,
        "report_path":    report_path,
        "input_size_mb":  flat_size/1e6,
        "csv_size_mb":    csv_size/1e6,
        "parquet_size_mb":parq_size/1e6,
        "n_features":     len(feature_cols),
        "n_patients":     n_rows,
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description="Convert flat feature matrix → JSON-column format")
    ap.add_argument("input", help="Path to flat parquet/CSV (FE output)")
    ap.add_argument("--out-dir", required=True, help="Output directory for all artifacts")
    ap.add_argument("--cancer", default="breast", help="Cancer name (default: breast)")
    ap.add_argument("--window", default="1mo", help="Window label (default: 1mo)")
    ap.add_argument("--no-csv", action="store_true", help="Skip CSV output (parquet-only)")
    ap.add_argument("--no-parquet", action="store_true", help="Skip parquet output (CSV-only)")
    args = ap.parse_args()

    result = produce_json_format(
        args.input, args.out_dir,
        cancer=args.cancer, window=args.window,
        write_csv=not args.no_csv,
        write_parquet=not args.no_parquet,
    )
    print(f"\n✓ Conversion complete:")
    for k, v in result.items():
        print(f"    {k:18s}: {v}")
