"""
Option 2: Summary-stats-only sliding window pipeline.
For each term produces 5 features per patient (not 84):
  slope, total_change, pct_change, recent_roc, n_windows

Edit TERMS / TERM_VALUE_RANGES below to add/remove features.
Run coverage_scan.py first to check patient coverage before adding a term.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import PARQUET_PATH, WINDOW_DAYS, LOOKBACK_DAYS, N_WINDOWS

# ── Edit here to add terms ────────────────────────────────────────────────
TERMS = [
    "Body weight",
    "Body mass index",
    "Systolic arterial pressure",
    "Diastolic arterial pressure",
    "Haemoglobin estimation",
    "Serum creatinine level",
    "Serum albumin level",
    "Serum TSH (thyroid stimulating hormone) level",
    "Pulse rate",
    "Haemoglobin A1c level - International Federation of Clinical Chemistry and Laboratory Medicine standardised",
]

# Plausible value ranges — applied AFTER unit normalization, rows outside dropped
TERM_VALUE_RANGES = {
    "Body weight":               (20, 300),
    "Body mass index":           (10, 80),
    "Systolic arterial pressure":(60, 250),
    "Diastolic arterial pressure":(30, 150),
    "Haemoglobin estimation":    (3, 25),    # g/dL after normalization
    "Serum creatinine level":    (10, 2000),
    "Serum albumin level":       (10, 70),
    "Serum TSH (thyroid stimulating hormone) level": (0.001, 100),
    "Pulse rate":                (20, 250),
    "Haemoglobin A1c level - International Federation of Clinical Chemistry and Laboratory Medicine standardised": (15, 200),
}

# Unit normalization: if value > threshold, divide by divisor to convert to canonical unit
# Haemoglobin: g/L → g/dL (values >30 divide by 10)
TERM_UNIT_NORMALIZE = {
    "Haemoglobin estimation": {"threshold": 30, "divisor": 10},
}

OUTPUT_PATH = "output/summary_features.parquet"

# Recent window bucket used for recent_roc: w04-w08 (12-24 months before anchor)
RECENT_WIN_START = 4
RECENT_WIN_END   = 9   # exclusive


def load_events():
    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    df = df[df["term"].isin(TERMS)].copy()
    df = df[df["days_before_anchor"] <= LOOKBACK_DAYS]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Unit normalization (e.g. haemoglobin g/L → g/dL)
    for term, rule in TERM_UNIT_NORMALIZE.items():
        mask = (df["term"] == term) & (df["value"] > rule["threshold"])
        df.loc[mask, "value"] = df.loc[mask, "value"] / rule["divisor"]

    # Plausible range filter
    keep = pd.Series(True, index=df.index)
    for term, (lo, hi) in TERM_VALUE_RANGES.items():
        if lo is None and hi is None:
            continue
        mask = df["term"] == term
        if lo is not None:
            keep &= ~(mask & (df["value"] < lo))
        if hi is not None:
            keep &= ~(mask & (df["value"] > hi))
    return df[keep][["patient_guid", "cancer_class", "days_before_anchor", "term", "value"]]


def term_prefix(term: str) -> str:
    return term.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("/", "_")[:30]


def build_summary(events: pd.DataFrame, term: str) -> pd.DataFrame:
    prefix = term_prefix(term)
    sub = events[events["term"] == term].copy()
    sub["win"] = (sub["days_before_anchor"] // WINDOW_DAYS).clip(upper=N_WINDOWS - 1).astype(int)

    # Per patient per window: mean value
    wide = sub.groupby(["patient_guid", "win"])["value"].mean().unstack("win")

    all_wins = list(range(N_WINDOWS))
    wide = wide.reindex(columns=all_wins)

    records = {}

    # slope: linear fit over (window_idx, mean_value) — negative slope = declining toward anchor
    def _slope(row):
        mask = row.notna()
        if mask.sum() < 2:
            return np.nan
        x = np.array(all_wins)[mask.values]
        y = row.values[mask.values]
        return np.polyfit(x, y, 1)[0]

    records[f"{prefix}_slope"] = wide.apply(_slope, axis=1)

    # total_change: most_recent_window − oldest_window
    def _total_change(row):
        v = row[row.notna()]
        return (v.iloc[0] - v.iloc[-1]) if len(v) >= 2 else np.nan

    records[f"{prefix}_total_change"] = wide.apply(_total_change, axis=1)

    # pct_change: total_change / oldest value
    def _pct_change(row):
        v = row[row.notna()]
        if len(v) < 2 or v.iloc[-1] == 0:
            return np.nan
        return (v.iloc[0] - v.iloc[-1]) / v.iloc[-1]

    records[f"{prefix}_pct_change"] = wide.apply(_pct_change, axis=1)

    # recent_roc: mean rate of change across w04–w08 (12–24mo before anchor)
    recent_cols = [w for w in range(RECENT_WIN_START, RECENT_WIN_END) if w in wide.columns]

    def _recent_roc(row):
        vals = row[recent_cols].dropna()
        if len(vals) < 2:
            return np.nan
        x = np.array(vals.index)
        return np.polyfit(x, vals.values, 1)[0] / WINDOW_DAYS  # per day

    records[f"{prefix}_recent_roc"] = wide[recent_cols].apply(_recent_roc, axis=1)

    # n_windows: count of windows with data
    records[f"{prefix}_n_windows"] = wide.notna().sum(axis=1)

    return pd.DataFrame(records, index=wide.index)


def run():
    print("Loading events...")
    events = load_events()
    print(f"  {len(events):,} events, {events['patient_guid'].nunique():,} patients")

    labels = events.groupby("patient_guid")["cancer_class"].first()

    results = []
    for term in TERMS:
        prefix = term_prefix(term)
        n_pts = events[events["term"] == term]["patient_guid"].nunique()
        pct = n_pts / events["patient_guid"].nunique() * 100
        print(f"  {prefix[:35]:<35} {n_pts:>5} patients ({pct:.1f}%)")
        summary = build_summary(events, term)
        results.append(summary)

    features = pd.concat(results, axis=1)
    features = features.join(labels)
    features.index.name = "patient_guid"
    features = features.reset_index()

    out = Path(__file__).parent / OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)

    feat_cols = [c for c in features.columns if c not in ("patient_guid", "cancer_class")]
    print(f"\nSaved: {out}")
    print(f"Shape: {features.shape}  ({len(feat_cols)} features, {len(TERMS)} terms × 5 summary stats)")
    print(f"Cancer: {features['cancer_class'].value_counts().to_dict()}")


if __name__ == "__main__":
    run()
