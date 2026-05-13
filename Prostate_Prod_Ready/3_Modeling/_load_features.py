"""
_load_alex_features.py — consume Alex's BQ-productionized feature tables.

Use this in modeling scripts to load features regardless of whether Alex's
output is the legacy flat-column format OR the JSON-column format from his
ADR. One helper handles both — modeling code stays unchanged across the
migration.

USAGE
=====

    from _load_alex_features import load_alex_features

    # Whether the CSV is flat (today) or JSON-column (post-ADR), this works:
    df = load_alex_features("data/prostate_features.csv")

    # Optional: pass a schema sidecar so JSON values cast to correct dtype
    # (Alex emits this alongside the table — see ADR Condition #1)
    df = load_alex_features(
        "data/prostate_features.csv",
        schema_path="data/snomed_features_schema.json",
    )

    # Train as usual:
    X = df.drop(columns=['LABEL', 'PATIENT_GUID'])
    y = df['LABEL']

WHY THIS EXISTS
===============

Alex's pre-ADR table:           flat — 500 columns directly named
Alex's post-ADR table:          8 spine columns + 1 'snomed_features' JSON column

Without this helper, modeling code that does `df['NUM_SMOKING_OCCURRENCES']`
would fail post-ADR because that column lives inside the JSON object.

This helper:
  1. Detects which format the file is in
  2. If JSON-column: expands `snomed_features` back to flat columns
  3. If a schema sidecar exists: casts each feature to its declared dtype
     (otherwise pd.json_normalize would mis-type sparse-NULL columns as object)
  4. Returns a clean flat DataFrame ready for `model.fit(X, y)`

Cancer-agnostic — works for every cancer. Just import and call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Known JSON-column names this loader can recognise.
# Order matters: when auto-detecting, the loader tries these in sequence.
#   - "transformed_features": lung's existing table + our Prostate_Prod_Ready
#     producer (transform_features_json.py)
#   - "snomed_features":      Alex's ADR convention
#   - "json_features":        defensive fallback
JSON_COLUMN_NAMES = ("transformed_features", "snomed_features", "json_features")

# Back-compat constant (kept for external callers that import this name)
JSON_COLUMN_NAME = JSON_COLUMN_NAMES[0]


def _detect_json_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    """Return the first known JSON column name found in df, or None."""
    candidates = [preferred] if preferred else []
    candidates.extend(c for c in JSON_COLUMN_NAMES if c not in candidates)
    for name in candidates:
        if name and name in df.columns:
            return name
    return None


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def load_alex_features(
    path: PathLike,
    schema_path: Optional[PathLike] = None,
    *,
    expand_json: bool = True,
    json_column: Optional[str] = None,
    require_schema: bool = False,
) -> pd.DataFrame:
    """Load a feature table produced by Alex's pipeline OR our prod-ready FE.

    Handles both the legacy flat-column format AND the post-ADR JSON-column
    format transparently. Caller doesn't need to know which one they got, OR
    which JSON column name the producer used (auto-detects across the known
    set: ``transformed_features``, ``snomed_features``, ``json_features``).

    Parameters
    ----------
    path : str | Path
        Path to a .csv, .csv.gz, .parquet, or BQ-export directory.
    schema_path : str | Path, optional
        Path to a schema sidecar (per ADR Condition #1). If provided, expanded
        JSON columns are cast to declared dtypes. If absent, dtypes are
        inferred (best-effort) — works but less reliable.
    expand_json : bool, default True
        If True and a JSON column exists, expand it to flat columns.
        Set False if you want to keep the JSON column intact (rare).
    json_column : str, optional
        Explicit JSON column name to use. If None (default), auto-detects
        across the known names. Override only if the producer uses a name
        not in JSON_COLUMN_NAMES.
    require_schema : bool, default False
        If True and the JSON column exists but no schema_path provided,
        raise ValueError. Use this in production to catch silent dtype drift.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame ready for modeling.
        - Spine columns preserved as-is (PATIENT_GUID, SEX, AGE, LABEL, etc.)
        - JSON column expanded into per-feature columns
        - Dtypes coerced per schema (if provided)

    Examples
    --------
    >>> df = load_alex_features("prostate_v4.csv")                       # auto-detects format + column
    >>> df = load_alex_features("prostate_v4.parquet", "schema.json")    # bulletproof dtypes
    >>> df = load_alex_features("prostate_v4.csv", require_schema=True)  # production strict mode
    >>> df = load_alex_features("custom.csv", json_column="my_features") # explicit override
    """
    path = Path(path)
    df = _read_table(path)
    n_cols_before = df.shape[1]

    detected = _detect_json_column(df, preferred=json_column) if expand_json else None

    if expand_json and detected:
        if schema_path is None and require_schema:
            raise ValueError(
                f"JSON column '{detected}' present but no schema_path provided "
                f"(require_schema=True). Pass schema_path or set require_schema=False."
            )
        schema = _load_schema(schema_path) if schema_path else None
        df = _expand_json_column(df, detected, schema)
        n_cols_after = df.shape[1]
        logger.info(
            f"  expanded '{detected}' JSON: "
            f"{n_cols_before} → {n_cols_after} columns "
            f"({n_cols_after - n_cols_before + 1} features extracted)"
        )
    elif (not expand_json) and _detect_json_column(df):
        kept = _detect_json_column(df)
        logger.info(f"  '{kept}' present but expand_json=False — left intact")
    else:
        logger.info(f"  flat-column format detected ({n_cols_before} cols) — no expansion needed")

    return df


def validate_against_manifest(
    df: pd.DataFrame,
    manifest_path: PathLike,
    *,
    raise_on_missing: bool = True,
) -> None:
    """Verify the DataFrame's columns match what the manifest expects.

    Use this between FE and modeling to catch silent column drops/renames.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded feature matrix.
    manifest_path : str | Path
        Path to a feature manifest JSON (lists expected columns).
    raise_on_missing : bool, default True
        If True, raise ValueError when expected columns are missing.
        If False, just log a warning.
    """
    manifest = json.load(open(manifest_path))
    expected = set(manifest.get("features", {}).keys())
    if not expected:
        # Older format: maybe a flat list under different key
        expected = set(manifest.get("columns", []))
    actual = set(df.columns) - {"LABEL"}
    missing = expected - actual
    extra = actual - expected
    if missing:
        msg = f"  {len(missing)} expected columns missing from DataFrame (first 5: {sorted(missing)[:5]})"
        if raise_on_missing:
            raise ValueError(msg)
        logger.warning(msg)
    if extra:
        logger.warning(f"  {len(extra)} unexpected columns in DataFrame (first 5: {sorted(extra)[:5]})")


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────

def _read_table(path: Path) -> pd.DataFrame:
    """Auto-detect file type (CSV / Parquet) and load."""
    s = str(path)
    if s.endswith(".parquet") or s.endswith(".pq"):
        return pd.read_parquet(path)
    if s.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False, compression="gzip")
    if s.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if path.is_dir():
        # BQ exports often go to a directory of shards — load all CSVs concatenated
        shards = sorted(path.glob("*.csv")) or sorted(path.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No CSV/Parquet shards in directory: {path}")
        return pd.concat([_read_table(p) for p in shards], ignore_index=True)
    raise ValueError(f"Unsupported file type: {path} (expected .csv, .csv.gz, .parquet, or directory)")


def _load_schema(schema_path: PathLike) -> dict:
    """Load Alex's schema sidecar.

    Expected format:
        {
          "features": {
            "NUM_SMOKING_OCCURRENCES":   {"dtype": "int", "domain": "smoking", ...},
            "FEV_TREND_SLOPE":           {"dtype": "float", ...},
            "smoking_IS_WORSENING":      {"dtype": "bool", ...},
            "FIRST_SMOKING_DATE":        {"dtype": "date", ...},
            ...
          }
        }
    """
    return json.load(open(schema_path))


def _expand_json_column(df: pd.DataFrame, col: str, schema: Optional[dict]) -> pd.DataFrame:
    """Expand a JSON-string column into separate flat columns.

    If `schema` is provided (with 'features' key), cast each expanded column
    to the declared dtype. Otherwise, best-effort numeric coercion.
    """
    raw = df[col]

    # Robust JSON parse: each row may already be a dict (BQ-pandas) or a JSON string (CSV)
    parsed = raw.apply(lambda v: v if isinstance(v, dict) else (json.loads(v) if isinstance(v, str) and v else {}))
    flat = pd.json_normalize(parsed)

    # Cast dtypes if schema available
    if schema is not None:
        feature_meta = schema.get("features", {})
        for col_name, meta in feature_meta.items():
            if col_name not in flat.columns:
                continue
            dtype = meta.get("dtype", "")
            try:
                if dtype.startswith("int"):
                    flat[col_name] = pd.to_numeric(flat[col_name], errors="coerce").astype("Int64")
                elif dtype.startswith("float"):
                    flat[col_name] = pd.to_numeric(flat[col_name], errors="coerce")
                elif dtype == "bool":
                    flat[col_name] = flat[col_name].map(
                        {"true": True, "false": False, "True": True, "False": False, True: True, False: False}
                    )
                elif dtype == "date":
                    flat[col_name] = pd.to_datetime(flat[col_name], errors="coerce")
                # else: leave as string
            except Exception as e:
                logger.warning(f"  dtype-cast failed for {col_name} ({dtype}): {e}")
    else:
        # Best-effort: numeric coercion for anything that looks numeric
        for col_name in flat.columns:
            sample = flat[col_name].dropna().head(50)
            if len(sample) == 0:
                continue
            # Try numeric conversion — keep if it works for >90% of non-null values
            coerced = pd.to_numeric(sample, errors="coerce")
            if coerced.notna().mean() >= 0.9:
                flat[col_name] = pd.to_numeric(flat[col_name], errors="coerce")

    return pd.concat([df.drop(columns=[col]), flat], axis=1)


# ────────────────────────────────────────────────────────────────────────────
# CLI: quick smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description="Quick smoke test for load_alex_features")
    ap.add_argument("path", help="Path to feature table (CSV / Parquet / BQ-export dir)")
    ap.add_argument("--schema", help="Optional path to schema sidecar JSON")
    ap.add_argument("--require-schema", action="store_true", help="Fail if JSON column has no schema")
    ap.add_argument("--head", type=int, default=3, help="Print N preview rows")
    args = ap.parse_args()

    df = load_alex_features(args.path, args.schema, require_schema=args.require_schema)
    print(f"\nLoaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nDtypes summary:")
    print(df.dtypes.value_counts())
    print(f"\nFirst {args.head} rows:")
    print(df.head(args.head).to_string())
