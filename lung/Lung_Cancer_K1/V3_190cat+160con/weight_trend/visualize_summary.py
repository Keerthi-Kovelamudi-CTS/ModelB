"""
Visualizations for the 10-term summary feature set (Option 2 pipeline).
Generates: signal ranking, total change grid, slope grid, coverage comparison.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

OUT_DIR = Path(__file__).parent / "output" / "plots" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {0: "#2196F3", 1: "#E53935"}
LABELS = {0: "Non-cancer", 1: "Cancer"}

TERMS = [
    ("body_weight",                   "Body Weight",       "kg"),
    ("body_mass_index",               "BMI",               "kg/m²"),
    ("systolic_arterial_pressure",    "Systolic BP",       "mmHg"),
    ("diastolic_arterial_pressure",   "Diastolic BP",      "mmHg"),
    ("haemoglobin_estimation",        "Haemoglobin",       "g/dL"),
    ("serum_creatinine_level",        "Creatinine",        "µmol/L"),
    ("serum_albumin_level",           "Albumin",           "g/L"),
    ("serum_tsh_thyroid_stimulating_","TSH",               "mU/L"),
    ("pulse_rate",                    "Pulse Rate",        "bpm"),
    ("haemoglobin_a1c_level___intern","HbA1c",             "mmol/mol"),
]
STATS = ["slope", "total_change", "pct_change", "recent_roc"]


def load():
    return pd.read_parquet(Path(__file__).parent / "output" / "summary_features.parquet")


def ttest(df, col):
    a = df[df["cancer_class"] == 1][col].dropna()
    b = df[df["cancer_class"] == 0][col].dropna()
    t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return t, p


# ── Plot 1: Signal ranking heatmap (−log10 p per term × stat) ─────────────

def plot_signal_heatmap(df):
    n_terms = len(TERMS)
    n_stats = len(STATS)
    log_p   = np.zeros((n_terms, n_stats))
    t_signs = np.zeros((n_terms, n_stats))

    for i, (prefix, _, _) in enumerate(TERMS):
        for j, stat in enumerate(STATS):
            col = f"{prefix}_{stat}"
            if col not in df.columns:
                continue
            t, p = ttest(df, col)
            log_p[i, j]   = -np.log10(max(p, 1e-20))
            t_signs[i, j] = np.sign(t)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(log_p, cmap="YlOrRd", aspect="auto", vmin=0, vmax=15)

    ax.set_xticks(range(n_stats))
    ax.set_xticklabels(["Slope", "Total Change", "% Change", "Recent RoC"], fontsize=10)
    ax.set_yticks(range(n_terms))
    ax.set_yticklabels([f"{label} ({unit})" for _, label, unit in TERMS], fontsize=9)

    for i in range(n_terms):
        for j in range(n_stats):
            col = f"{TERMS[i][0]}_{STATS[j]}"
            if col not in df.columns:
                continue
            t, p = ttest(df, col)
            sign = "↑C" if t > 0 else "↓C"
            txt = f"{sign}\np={p:.3f}" if p >= 0.001 else f"{sign}\np<0.001"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if log_p[i, j] > 7 else "black")

    plt.colorbar(im, ax=ax, label="−log₁₀(p-value)", shrink=0.8)
    ax.set_title("Signal Strength per Feature: Cancer vs Non-Cancer\n(↑C = higher in cancer, ↓C = lower in cancer)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_signal_heatmap.png", dpi=150)
    plt.close(fig)


# ── Plot 2: Total change distributions — 2×5 grid ─────────────────────────

def plot_total_change_grid(df):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    clip_pcts = {
        "body_weight": (-30, 30),
        "body_mass_index": (-8, 8),
        "systolic_arterial_pressure": (-40, 40),
        "diastolic_arterial_pressure": (-30, 30),
        "haemoglobin_estimation": (-3, 3),
        "serum_creatinine_level": (-100, 100),
        "serum_albumin_level": (-10, 10),
        "serum_tsh_thyroid_stimulating_": (-3, 3),
        "pulse_rate": (-30, 30),
        "haemoglobin_a1c_level___intern": (-20, 20),
    }

    for ax, (prefix, label, unit) in zip(axes, TERMS):
        col = f"{prefix}_total_change"
        lo, hi = clip_pcts.get(prefix, (-50, 50))
        _, p = ttest(df, col)
        sig = f"p<0.001" if p < 0.001 else f"p={p:.3f}"

        for cls in [0, 1]:
            vals = df[df["cancer_class"] == cls][col].dropna().clip(lo, hi)
            ax.hist(vals, bins=40, alpha=0.55, color=COLORS[cls],
                    label=f"{LABELS[cls]} μ={vals.mean():.2f}", density=True)
            ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.2)
        ax.axvline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_title(f"{label}\n({sig})", fontsize=9)
        ax.set_xlabel(unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Total Change (oldest → most recent window) — Cancer vs Non-Cancer", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_total_change_grid.png", dpi=150)
    plt.close(fig)


# ── Plot 3: Slope distributions — 2×5 grid ────────────────────────────────

def plot_slope_grid(df):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for ax, (prefix, label, unit) in zip(axes, TERMS):
        col = f"{prefix}_slope"
        _, p = ttest(df, col)
        sig = "p<0.001" if p < 0.001 else f"p={p:.3f}"

        for cls in [0, 1]:
            vals = df[df["cancer_class"] == cls][col].dropna()
            p5, p95 = vals.quantile(0.05), vals.quantile(0.95)
            vals = vals.clip(p5, p95)
            ax.hist(vals, bins=40, alpha=0.55, color=COLORS[cls],
                    label=f"{LABELS[cls]} μ={vals.mean():.3f}", density=True)
            ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.2)
        ax.axvline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_title(f"{label}\n({sig})", fontsize=9)
        ax.set_xlabel(f"{unit}/window", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Trend Slope (10yr sliding window) — Cancer vs Non-Cancer  [clipped p5–p95]", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_slope_grid.png", dpi=150)
    plt.close(fig)


# ── Plot 4: n_windows (coverage) — grouped bar ────────────────────────────

def plot_n_windows(df):
    labels_short = [label for _, label, _ in TERMS]
    c_med  = [df[df["cancer_class"] == 1][f"{p}_n_windows"].median() for p, _, _ in TERMS]
    nc_med = [df[df["cancer_class"] == 0][f"{p}_n_windows"].median() for p, _, _ in TERMS]

    x = np.arange(len(TERMS))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, nc_med, width=0.4, color=COLORS[0], alpha=0.85, label="Non-cancer (median)")
    ax.bar(x + 0.2, c_med,  width=0.4, color=COLORS[1], alpha=0.85, label="Cancer (median)")
    ax.axhline(10, color="gray", linewidth=0.8, linestyle="--", label="10 windows")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Median windows with data (out of 40)", fontsize=10)
    ax.set_title("Median Data Coverage per Term — Cancer vs Non-Cancer", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_n_windows_comparison.png", dpi=150)
    plt.close(fig)


# ── Plot 5: p-value ranking bar chart ─────────────────────────────────────

def plot_pvalue_ranking(df):
    rows = []
    for prefix, label, unit in TERMS:
        for stat in STATS:
            col = f"{prefix}_{stat}"
            if col not in df.columns:
                continue
            t, p = ttest(df, col)
            rows.append({"term": label, "stat": stat, "p": p, "t": t,
                         "log_p": -np.log10(max(p, 1e-20)),
                         "label": f"{label} — {stat.replace('_',' ')}"})

    rank = pd.DataFrame(rows).sort_values("log_p", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors_bar = [COLORS[1] if r["t"] < 0 else "#FF9800" for _, r in rank.iterrows()]
    y = np.arange(len(rank))
    ax.barh(y, rank["log_p"], color=colors_bar, alpha=0.85)
    ax.axvline(-np.log10(0.05), color="black", linewidth=1, linestyle="--", label="p=0.05")
    ax.axvline(-np.log10(0.001), color="gray", linewidth=0.8, linestyle=":", label="p=0.001")
    ax.set_yticks(y)
    ax.set_yticklabels(rank["label"], fontsize=8)
    ax.set_xlabel("−log₁₀(p-value)", fontsize=10)
    ax.set_title("Feature Signal Ranking — All Terms × Stats\n(red = lower in cancer, orange = higher in cancer)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_pvalue_ranking.png", dpi=150)
    plt.close(fig)


def run():
    print("Loading summary features...")
    df = load()
    print(f"  {len(df):,} patients, {df['cancer_class'].value_counts().to_dict()}")
    print("Generating plots...")
    plot_signal_heatmap(df)
    plot_total_change_grid(df)
    plot_slope_grid(df)
    plot_n_windows(df)
    plot_pvalue_ranking(df)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
