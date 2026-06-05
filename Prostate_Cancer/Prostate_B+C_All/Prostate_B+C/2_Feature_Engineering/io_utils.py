"""
Centralized table I/O — Parquet by default, CSV fallback.

Migration helpers used across the FE + modeling pipeline so we can switch
from CSV to Parquet incrementally without breaking anything.

Usage:
    from io_utils import read_table, write_table

    # Read: tries the path as-given, then auto-falls back to the sibling .parquet
    # or .csv file. Pass standard pandas kwargs (low_memory, index_col, etc.).
    df = read_table(path)

    # Write: extension determines format. Default = parquet.
    write_table(df, path)               # path.csv → writes parquet to path.parquet
    write_table(df, path, format='csv') # force csv

Why parquet:
  - 5-10× faster I/O on the multi-GB intermediate files
  - 50-80% smaller on disk (gzip-like compression)
  - Type-safe schema (no '123' vs '123.0' float-string ambiguity)

Backward compat:
  - read_table opens .csv if .parquet missing (for legacy intermediate files)
  - write_table writes parquet by default but honors explicit .csv
"""

from pathlib import Path
import pandas as pd

# pandas kwargs that are CSV-only and should be dropped when reading parquet
_CSV_ONLY_READ = {'low_memory', 'on_bad_lines', 'sep', 'encoding', 'lineterminator',
                  'skip_blank_lines', 'chunksize', 'dtype', 'na_values', 'header',
                  'parse_dates', 'date_parser', 'date_format', 'thousands', 'decimal'}
_CSV_ONLY_WRITE = {'sep', 'encoding', 'lineterminator', 'header', 'na_rep', 'mode'}

PARQUET_ENGINE = 'pyarrow'


def _strip_csv_kwargs(kwargs, csv_only):
    return {k: v for k, v in kwargs.items() if k not in csv_only}


def read_table(path, **kwargs):
    """Read a CSV or Parquet table. Tries given path; falls back to sibling
    with the other extension if not found."""
    p = Path(path)
    # If path exists, dispatch by extension
    if p.exists():
        if p.suffix == '.parquet':
            return pd.read_parquet(p, engine=PARQUET_ENGINE,
                                   **_strip_csv_kwargs(kwargs, _CSV_ONLY_READ))
        return pd.read_csv(p, **kwargs)
    # Auto-fallback: try the other extension
    if p.suffix == '.csv':
        alt = p.with_suffix('.parquet')
        if alt.exists():
            return pd.read_parquet(alt, engine=PARQUET_ENGINE,
                                   **_strip_csv_kwargs(kwargs, _CSV_ONLY_READ))
    elif p.suffix == '.parquet':
        alt = p.with_suffix('.csv')
        if alt.exists():
            return pd.read_csv(alt, **kwargs)
    raise FileNotFoundError(f"Neither {p} nor sibling format found")


def write_table(df, path, format=None, **kwargs):
    """Write a DataFrame to CSV or Parquet.

    If `format` is given ('csv' or 'parquet'), it overrides the extension.
    Otherwise extension is honored (.csv → CSV, .parquet → Parquet, anything
    else → defaults to Parquet).
    """
    p = Path(path)
    fmt = format or ({'csv': 'csv', '.csv': 'csv',
                      'parquet': 'parquet', '.parquet': 'parquet'}.get(p.suffix, 'parquet'))
    if fmt == 'csv':
        # Ensure .csv extension
        if p.suffix != '.csv':
            p = p.with_suffix('.csv')
        df.to_csv(p, **kwargs)
    else:
        if p.suffix != '.parquet':
            p = p.with_suffix('.parquet')
        # Strip CSV-only kwargs; keep index handling
        pq_kwargs = _strip_csv_kwargs(kwargs, _CSV_ONLY_WRITE)
        # Pandas to_parquet doesn't accept the kwargs CSV uses; standardize
        # `index` keyword (default False unless user wants it)
        df.to_parquet(p, engine=PARQUET_ENGINE,
                      index=pq_kwargs.get('index', False))


def parquet_path(path):
    """Coerce a path to its .parquet variant (preserves directory + stem)."""
    return Path(path).with_suffix('.parquet')


def csv_path(path):
    """Coerce a path to its .csv variant."""
    return Path(path).with_suffix('.csv')
