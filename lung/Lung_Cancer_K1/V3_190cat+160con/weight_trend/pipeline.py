"""
Sliding window weight trend features from Astra 10yr event table.

For each patient:
  - Bins weight measurements into 3-month windows over 10yr lookback
  - w00 = most recent window, w39 = oldest window
  - Computes per-window mean, window-to-window rate of change, and trend summary stats
  - Outputs one row per patient with cancer_class label

Usage: python pipeline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import PARQUET_PATH, TERMS, WINDOW_DAYS, LOOKBACK_DAYS, N_WINDOWS, OUTPUT_PATH, TERM_VALUE_RANGES


def load_weight_events(path: str, terms: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    df = df[df["term"].isin(terms)].copy()
    df = df[df["days_before_anchor"] <= LOOKBACK_DAYS]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Apply per-term plausible range filter
    keep = pd.Series(True, index=df.index)
    for term, (lo, hi) in TERM_VALUE_RANGES.items():
        if lo is None and hi is None:
            continue
        mask = df["term"] == term
        if lo is not None:
            keep &= ~(mask & (df["value"] < lo))
        if hi is not None:
            keep &= ~(mask & (df["value"] > hi))
    df = df[keep]

    return df[["patient_guid", "cancer_class", "days_before_anchor", "term", "value"]]


def assign_window(days_before_anchor: pd.Series) -> pd.Series:
    # w00 = most recent (0-91 days), w39 = oldest (3559-3650 days)
    return (days_before_anchor // WINDOW_DAYS).clip(upper=N_WINDOWS - 1).astype(int)


def build_window_cols(events: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Mean weight per patient per window. One row per patient, columns w00..w{N-1}."""
    events = events.copy()
    events["win"] = assign_window(events["days_before_anchor"])

    agg = (
        events.groupby(["patient_guid", "win"])["value"]
        .mean()
        .unstack(level="win")
    )

    # Rename columns: 0 → w00, 39 → w39
    agg.columns = [f"{prefix}_w{int(c):02d}" for c in agg.columns]

    # Ensure all 40 windows exist
    all_cols = [f"{prefix}_w{i:02d}" for i in range(N_WINDOWS)]
    agg = agg.reindex(columns=all_cols)

    return agg


def build_roc_cols(wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rate of change per day from window i+1 → window i (older → newer).
    Positive roc = weight increased toward anchor date."""
    roc_cols = {}
    for i in range(N_WINDOWS - 1):
        col_new = f"{prefix}_w{i:02d}"
        col_old = f"{prefix}_w{i+1:02d}"
        if col_new in wide.columns and col_old in wide.columns:
            roc_cols[f"{prefix}_roc_w{i:02d}"] = (wide[col_new] - wide[col_old]) / WINDOW_DAYS
    return pd.DataFrame(roc_cols, index=wide.index)


def build_summary_cols(wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Trend summary per patient across all windows."""
    w_cols = [c for c in wide.columns if c.startswith(f"{prefix}_w")]
    vals = wide[w_cols]

    summary = {}

    # n observations is computed from events separately, so here use window count
    summary[f"{prefix}_n_windows"] = vals.notna().sum(axis=1)

    # Linear slope of weight vs window index (positive = increasing toward anchor)
    def _slope(row):
        mask = row.notna()
        if mask.sum() < 2:
            return np.nan
        x = np.array([int(c[-2:]) for c in w_cols])[mask.values]
        y = row.values[mask.values]
        return np.polyfit(x, y, 1)[0]

    summary[f"{prefix}_slope"] = vals.apply(_slope, axis=1)

    # Total change: most_recent - oldest (across non-null windows)
    summary[f"{prefix}_total_change"] = vals.apply(
        lambda r: r[r.notna()].iloc[0] - r[r.notna()].iloc[-1] if r.notna().sum() >= 2 else np.nan,
        axis=1,
    )

    # Pct change relative to oldest observed weight (NaN if oldest weight is 0)
    def _pct_change(r):
        valid = r[r.notna()]
        if len(valid) < 2 or valid.iloc[-1] == 0:
            return np.nan
        return (valid.iloc[0] - valid.iloc[-1]) / valid.iloc[-1]

    summary[f"{prefix}_pct_change"] = vals.apply(_pct_change, axis=1)

    return pd.DataFrame(summary, index=wide.index)


def build_n_obs_col(events: pd.DataFrame, prefix: str) -> pd.Series:
    return events.groupby("patient_guid").size().rename(f"{prefix}_n_obs")


def run():
    print("Loading events...")
    events = load_weight_events(PARQUET_PATH, TERMS)
    print(f"  {len(events):,} weight events, {events['patient_guid'].nunique():,} patients")

    # One label per patient
    labels = events.groupby("patient_guid")["cancer_class"].first()

    results = []
    for term in TERMS:
        prefix = term.lower().replace(" ", "_")
        term_events = events[events["term"] == term]

        print(f"Building windows for '{term}' (prefix={prefix})...")
        wide = build_window_cols(term_events, prefix)
        roc = build_roc_cols(wide, prefix)
        summary = build_summary_cols(wide, prefix)
        n_obs = build_n_obs_col(term_events, prefix)

        term_df = pd.concat([wide, roc, summary, n_obs], axis=1)
        results.append(term_df)

    features = pd.concat(results, axis=1)
    features = features.join(labels)
    features.index.name = "patient_guid"
    features = features.reset_index()

    out_path = Path(__file__).parent / OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Shape: {features.shape}")
    print(f"Cancer: {features['cancer_class'].value_counts().to_dict()}")
    win_cols = [c for c in features.columns if len(c) >= 4 and c[-4] == "_" and c[-3] == "w" and c[-2:].isdigit() and "_roc_" not in c]
    roc_cols = [c for c in features.columns if "_roc_" in c]
    sum_cols = [c for c in features.columns if any(c.endswith(x) for x in ["_slope", "_total_change", "_pct_change", "_n_windows", "_n_obs"])]
    print(f"\nFeature groups:")
    print(f"  Window cols   : {len(win_cols):>4}")
    print(f"  ROC cols      : {len(roc_cols):>4}")
    print(f"  Summary cols  : {len(sum_cols):>4}")


if __name__ == "__main__":
    run()
