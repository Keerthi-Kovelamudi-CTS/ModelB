"""Light I/O helpers — parquet preferred, csv supported as fallback."""

import pandas as pd
from pathlib import Path


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV or parquet file. Auto-detects from extension."""
    p = Path(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(p, **{k: v for k, v in kwargs.items() if k != 'low_memory'})
    if p.suffix == '.csv':
        return pd.read_csv(p, low_memory=False, **kwargs)
    if p.suffix == '.gz' and p.name.endswith('.csv.gz'):
        return pd.read_csv(p, low_memory=False, compression='gzip', **kwargs)
    raise ValueError(f"Unsupported file type: {p}")


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write to CSV or parquet based on extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == '.parquet':
        df.to_parquet(p, engine='pyarrow', index=index, compression='snappy')
    elif p.suffix == '.csv':
        df.to_csv(p, index=index)
    else:
        raise ValueError(f"Unsupported file type: {p}")
