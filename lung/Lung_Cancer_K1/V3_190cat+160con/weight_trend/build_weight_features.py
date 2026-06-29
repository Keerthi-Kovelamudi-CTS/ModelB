"""
Builds 40-window body weight feature matrix — one row per patient.

Windows: w00 (most recent, 0-90 days before anchor) → w39 (oldest, 3549-3639 days)
Each window = mean body weight (kg) in that 3-month period.

Imputation:
  - Patients with ≥1 reading : linear interpolate between known windows, then
                                ffill/bfill to fill edges (incl. w00-w03 which
                                are empty by design — prediction horizon)
  - Patients with 0 readings  : cohort median weight per cancer_class

Output:
  output/weight_window_features.parquet
  output/weight_window_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

PARQUET_PATH = "../data/Astra_10yr_160_withtext.parquet"
TERM         = "Body weight"
VAL_RANGE    = (20, 300)     # kg — drop dirty values
WINDOW_DAYS  = 91            # ~3 months
LOOKBACK     = 3650          # 10 years
N_WINDOWS    = LOOKBACK // WINDOW_DAYS   # 40

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN_COLS = [f"bw_w{i:02d}" for i in range(N_WINDOWS)]   # bw_w00 … bw_w39


def load_events():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow",
                         columns=["patient_guid", "cancer_class",
                                  "days_before_anchor", "term", "value"])
    df = df[df["term"] == TERM].copy()
    df = df[df["days_before_anchor"] <= LOOKBACK]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[(df["value"] >= VAL_RANGE[0]) & (df["value"] <= VAL_RANGE[1])]
    df["win"] = (df["days_before_anchor"] // WINDOW_DAYS).clip(upper=N_WINDOWS - 1).astype(int)
    print(f"  {len(df):,} events | {df['patient_guid'].nunique():,} patients with weight data")
    return df


def build_raw_wide(events):
    """One row per patient, 40 window cols (NaN where no measurement)."""
    wide = (events.groupby(["patient_guid", "win"])["value"]
            .mean()
            .unstack("win")
            .reindex(columns=range(N_WINDOWS)))
    wide.columns = WIN_COLS
    return wide


def get_all_patients(events):
    """All patients in the cohort (incl. those with no weight data)."""
    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow",
                         columns=["patient_guid", "cancer_class"])
    return df.groupby("patient_guid")["cancer_class"].first()


def impute(wide, all_labels):
    """
    1. Align to full patient list (adds patients with no weight data as all-NaN rows).
    2. Per patient with ≥1 reading: linear interpolate → ffill → bfill.
    3. Per patient with 0 readings: fill with cohort median per cancer_class.
    Returns wide df with bw_imputed_count column.
    """
    # Align to all patients
    wide = wide.reindex(all_labels.index)
    wide["cancer_class"] = all_labels

    # Track which cells were originally NaN (will be imputed)
    was_nan = wide[WIN_COLS].isna()

    # Step 1: per-patient interpolation + edge fill
    print("  Interpolating per patient...")
    wide[WIN_COLS] = (wide[WIN_COLS]
                      .interpolate(method="linear", axis=1, limit_direction="both"))

    # Step 2: patients with ZERO weight readings -> LEFT AS NaN (no label-based fill).
    # LEAKAGE FIX: the prior version filled these with the cohort median PER cancer_class, which uses
    # the label (cancer median 74.6 vs non-cancer 71.9 kg) -> leaks the class and is not deployable at
    # inference (label unknown). Trees handle NaN natively; bw_imputed_count == N_WINDOWS flags them as
    # fully-missing. (For an LSTM, impute with a label-AGNOSTIC value + a missingness mask downstream.)
    no_data_mask = wide[WIN_COLS].isna().all(axis=1)
    print(f"  Patients with zero weight readings (left NaN, no label-based fill): {int(no_data_mask.sum()):,}")

    # Count imputed windows per patient
    wide["bw_imputed_count"] = was_nan.reindex(wide.index).sum(axis=1)

    return wide


def run():
    events    = load_events()
    all_labels = get_all_patients(events)

    print("Building raw window matrix...")
    wide = build_raw_wide(events)

    print("Imputing missing windows...")
    wide = impute(wide, all_labels)

    # Final column order: patient_guid, cancer_class, bw_w00..w39, bw_imputed_count
    wide = wide.reset_index().rename(columns={"patient_guid": "patient_guid"})
    cols = ["patient_guid", "cancer_class"] + WIN_COLS + ["bw_imputed_count"]
    wide = wide[cols]

    # Sanity check: patients with >=1 reading are fully interpolated; zero-reading patients stay all-NaN
    # (intentional, leak-free) — so every row is either fully filled or fully NaN, never partial.
    _nan_per_pat = wide[WIN_COLS].isna().sum(axis=1)
    assert ((_nan_per_pat == 0) | (_nan_per_pat == N_WINDOWS)).all(), "unexpected partial-NaN rows after interpolation"

    print(f"\nOutput shape: {wide.shape}")
    print(f"  Patients          : {len(wide):,}")
    print(f"  Window cols       : {len(WIN_COLS)}  ({WIN_COLS[0]} → {WIN_COLS[-1]})")
    print(f"  Fully observed    : {(wide['bw_imputed_count'] == 0).sum():,}")
    print(f"  Partially imputed : {((wide['bw_imputed_count'] > 0) & (wide['bw_imputed_count'] < N_WINDOWS)).sum():,}")
    print(f"  Fully imputed     : {(wide['bw_imputed_count'] == N_WINDOWS).sum():,}")
    print(f"\nWeight range across all windows: {wide[WIN_COLS].min().min():.1f} – {wide[WIN_COLS].max().max():.1f} kg")
    print(f"Mean weight by class:")
    for cls in [0, 1]:
        m = wide[wide["cancer_class"] == cls][WIN_COLS].mean().mean()
        print(f"  cancer_class={cls}: {m:.1f} kg")

    parquet_path = OUT_DIR / "weight_window_features.parquet"
    csv_path     = OUT_DIR / "weight_window_features.csv"

    wide.to_parquet(parquet_path, index=False)
    wide.to_csv(csv_path, index=False)

    print(f"\nSaved:")
    print(f"  {parquet_path}")
    print(f"  {csv_path}")

    return wide


if __name__ == "__main__":
    df = run()
    print("\nSample (first 3 rows, first 8 window cols):")
    print(df[["patient_guid", "cancer_class"] + WIN_COLS[:8] + ["bw_imputed_count"]].head(3).to_string())
