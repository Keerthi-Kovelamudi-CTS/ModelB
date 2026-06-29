"""
Scans all numeric observation terms in Astra 10yr parquet.
Plots patient coverage per term (cancer vs non-cancer) before any filtering.
Run this to decide which terms to include in the sliding window pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET_PATH = "../data/Astra_10yr_withtext.parquet"
OUT_DIR = Path(__file__).parent / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COVERAGE = 10   # % — draw threshold line here


def compute_coverage(df):
    total     = df["patient_guid"].nunique()
    cancer    = df[df["cancer_class"] == 1]["patient_guid"].nunique()
    noncancer = df[df["cancer_class"] == 0]["patient_guid"].nunique()

    obs = df[df["event_type"] == "observation"].copy()
    obs["val"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs[obs["val"].notna()]

    cov = pd.DataFrame({
        "all":      obs.groupby("term")["patient_guid"].nunique().div(total)      * 100,
        "cancer":   obs[obs["cancer_class"] == 1].groupby("term")["patient_guid"].nunique().div(cancer)    * 100,
        "noncancer":obs[obs["cancer_class"] == 0].groupby("term")["patient_guid"].nunique().div(noncancer) * 100,
    }).fillna(0).sort_values("all", ascending=False)

    return cov


def plot_coverage_all(cov):
    """Horizontal bar chart: all terms >= 1% coverage, sorted. Mark 10% threshold."""
    plot_df = cov[cov["all"] >= 1].copy()

    # Shorten long names for readability
    plot_df.index = [t[:55] + "…" if len(t) > 55 else t for t in plot_df.index]
    plot_df = plot_df.sort_values("all")

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.28)))

    y = np.arange(n)
    ax.barh(y - 0.22, plot_df["noncancer"], height=0.4, color="#2196F3", alpha=0.8, label="Non-cancer")
    ax.barh(y + 0.22, plot_df["cancer"],    height=0.4, color="#E53935", alpha=0.8, label="Cancer")

    ax.axvline(MIN_COVERAGE, color="black", linewidth=1.2, linestyle="--",
               label=f"{MIN_COVERAGE}% threshold")
    ax.axvline(50, color="gray", linewidth=0.8, linestyle=":", label="50% threshold")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.index, fontsize=7.5)
    ax.set_xlabel("% Patients with ≥1 Numeric Measurement", fontsize=10)
    ax.set_title("Patient Coverage per Observation Term — Astra 10yr Cohort\n(cancer vs non-cancer, sorted by overall coverage)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "term_coverage_all.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'term_coverage_all.png'}")


def plot_coverage_top(cov, top_n=30):
    """Zoomed view of top N terms — easier to read."""
    plot_df = cov.head(top_n).copy()
    plot_df.index = [t[:55] + "…" if len(t) > 55 else t for t in plot_df.index]
    plot_df = plot_df.sort_values("all")

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(12, n * 0.4))

    y = np.arange(n)
    ax.barh(y - 0.22, plot_df["noncancer"], height=0.4, color="#2196F3", alpha=0.8, label="Non-cancer")
    ax.barh(y + 0.22, plot_df["cancer"],    height=0.4, color="#E53935", alpha=0.8, label="Cancer")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["all"] + 0.5, i, f"{row['all']:.0f}%", va="center", fontsize=7.5, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.index, fontsize=9)
    ax.set_xlabel("% Patients with ≥1 Numeric Measurement", fontsize=10)
    ax.set_title(f"Top {top_n} Terms by Patient Coverage — Cancer vs Non-Cancer", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "term_coverage_top30.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'term_coverage_top30.png'}")


def run():
    print("Loading data...")
    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    print(f"  {df['patient_guid'].nunique():,} total patients")

    print("Computing coverage...")
    cov = compute_coverage(df)

    above_threshold = cov[cov["all"] >= MIN_COVERAGE]
    print(f"  {len(cov)} terms with any numeric data")
    print(f"  {len(above_threshold)} terms >= {MIN_COVERAGE}% patient coverage")

    print("\nTop 20 terms:")
    print(cov.head(20).round(1).to_string())

    plot_coverage_all(cov)
    plot_coverage_top(cov, top_n=30)

    # Save coverage table for reference
    out_csv = Path(__file__).parent / "output" / "term_coverage.csv"
    cov.to_csv(out_csv)
    print(f"Coverage table: {out_csv}")

    return cov


if __name__ == "__main__":
    run()
