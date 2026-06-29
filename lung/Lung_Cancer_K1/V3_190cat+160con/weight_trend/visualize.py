"""
Sliding window trend visualization: cancer vs non-cancer.
Loops over all TERMS defined in config.py — no hardcoded feature names.
Outputs per-term plots to output/plots/<prefix>/ and returns stats dict.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

from config import TERMS, N_WINDOWS, OUTPUT_PATH

PLOT_DIR = Path(__file__).parent / "output" / "plots"
COLORS = {0: "#2196F3", 1: "#E53935"}
LABELS = {0: "Non-cancer", 1: "Cancer"}


def term_to_prefix(term: str) -> str:
    return term.lower().replace(" ", "_")


def win_cols_for(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([c for c in df.columns if len(c) >= 4 and c[-4] == "_" and c[-3] == "w" and c[-2:].isdigit() and "_roc_" not in c and c.startswith(prefix)])


def roc_cols_for(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([c for c in df.columns if f"{prefix}_roc_" in c])


def load():
    df = pd.read_parquet(Path(__file__).parent / OUTPUT_PATH)
    return df


# ── per-term plots ──────────────────────────────────────────────────────────

def plot_mean_trend(df, win_cols, prefix, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(win_cols))
    for cls in [0, 1]:
        sub = df[df["cancer_class"] == cls][win_cols]
        mean, sem = sub.mean(), sub.sem()
        ax.plot(x, mean.values, color=COLORS[cls], label=LABELS[cls], linewidth=2)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.15, color=COLORS[cls])
    ax.invert_xaxis()
    ax.set_xticks(x[::4])
    ax.set_xticklabels([f"w{i:02d}" for i in range(0, len(win_cols), 4)], rotation=45, fontsize=8)
    ax.set_xlabel("Window (w39=oldest → w00=most recent)", fontsize=10)
    ax.set_ylabel(f"Mean {prefix.replace('_', ' ').title()}", fontsize=10)
    ax.set_title(f"Mean Trend: {prefix.replace('_', ' ').title()} — Cancer vs Non-Cancer", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_mean_trend.png", dpi=150)
    plt.close(fig)


def plot_coverage(df, win_cols, prefix, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(win_cols))
    for cls in [0, 1]:
        sub = df[df["cancer_class"] == cls][win_cols]
        cov = sub.notna().mean() * 100
        ax.bar(x + (0.4 if cls == 1 else 0), cov.values, width=0.4,
               color=COLORS[cls], alpha=0.8, label=LABELS[cls])
    ax.invert_xaxis()
    ax.set_xticks(x[::4] + 0.2)
    ax.set_xticklabels([f"w{i:02d}" for i in range(0, len(win_cols), 4)], rotation=45, fontsize=8)
    ax.set_xlabel("Window (w39=oldest → w00=most recent)", fontsize=10)
    ax.set_ylabel("% Patients with Data", fontsize=10)
    ax.set_title(f"Data Coverage: {prefix.replace('_', ' ').title()}", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "02_coverage.png", dpi=150)
    plt.close(fig)


def plot_slope_dist(df, prefix, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    col = f"{prefix}_slope"
    for cls in [0, 1]:
        vals = df[df["cancer_class"] == cls][col].dropna().clip(-5, 5)
        ax.hist(vals, bins=60, alpha=0.5, color=COLORS[cls],
                label=f"{LABELS[cls]} mean={vals.mean():.3f}", density=True)
        ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"Trend Slope (per window)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Slope Distribution: {prefix.replace('_', ' ').title()} (clipped ±5)", fontsize=12)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "03_slope_dist.png", dpi=150)
    plt.close(fig)


def plot_change_dist(df, prefix, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [
        (axes[0], f"{prefix}_total_change", "Total Change", (-50, 50)),
        (axes[1], f"{prefix}_pct_change",   "% Change",     (-0.5, 0.5)),
    ]
    for ax, col, label, clip in pairs:
        if col not in df.columns:
            continue
        for cls in [0, 1]:
            vals = df[df["cancer_class"] == cls][col].dropna().clip(*clip)
            ax.hist(vals, bins=60, alpha=0.5, color=COLORS[cls], label=LABELS[cls], density=True)
            ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel(label, fontsize=10); ax.set_ylabel("Density", fontsize=10)
        ax.set_title(label, fontsize=11); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Change Distribution: {prefix.replace('_', ' ').title()}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "04_change_dist.png", dpi=150)
    plt.close(fig)


def plot_roc_trend(df, roc_cols, prefix, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(roc_cols))
    for cls in [0, 1]:
        mean = df[df["cancer_class"] == cls][roc_cols].mean()
        ax.plot(x, mean.values, color=COLORS[cls], label=LABELS[cls], linewidth=2)
    ax.invert_xaxis()
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([c.split("_roc_")[1] for c in roc_cols][::4], rotation=45, fontsize=8)
    ax.set_xlabel("Window transition (oldest → most recent)", fontsize=10)
    ax.set_ylabel("Mean Rate of Change (per day)", fontsize=10)
    ax.set_title(f"Rate of Change: {prefix.replace('_', ' ').title()}", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "05_roc_trend.png", dpi=150)
    plt.close(fig)


def plot_n_windows(df, prefix, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    col = f"{prefix}_n_windows"
    for cls in [0, 1]:
        vals = df[df["cancer_class"] == cls][col]
        ax.hist(vals, bins=range(0, N_WINDOWS + 2), alpha=0.5,
                color=COLORS[cls], label=LABELS[cls], density=True)
    ax.set_xlabel("Windows with Data", fontsize=10); ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Coverage per Patient: {prefix.replace('_', ' ').title()}", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "06_n_windows.png", dpi=150)
    plt.close(fig)


# ── stats ───────────────────────────────────────────────────────────────────

def compute_stats(df, prefix) -> dict:
    out = {"prefix": prefix}
    for cls in [0, 1]:
        sub = df[df["cancer_class"] == cls]
        lbl = LABELS[cls].lower().replace("-", "_")
        out[f"{lbl}_n"]                  = len(sub)
        out[f"{lbl}_slope_mean"]         = sub[f"{prefix}_slope"].mean()
        out[f"{lbl}_slope_median"]       = sub[f"{prefix}_slope"].median()
        out[f"{lbl}_total_change_mean"]  = sub[f"{prefix}_total_change"].mean()
        out[f"{lbl}_pct_change_mean"]    = sub[f"{prefix}_pct_change"].mean() * 100
        out[f"{lbl}_n_windows_median"]   = sub[f"{prefix}_n_windows"].median()
        out[f"{lbl}_n_obs_median"]       = sub[f"{prefix}_n_obs"].median()

    for col_suffix, key in [("_slope", "slope"), ("_total_change", "total_change")]:
        col = f"{prefix}{col_suffix}"
        a = df[df["cancer_class"] == 1][col].dropna()
        b = df[df["cancer_class"] == 0][col].dropna()
        t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        out[f"{key}_ttest_t"] = t
        out[f"{key}_ttest_p"] = p

    return out


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading features...")
    df = load()
    all_stats = {}

    for term in TERMS:
        prefix = term_to_prefix(term)
        wc = win_cols_for(df, prefix)
        rc = roc_cols_for(df, prefix)

        if not wc:
            print(f"  Skipping '{term}' — no columns found in parquet.")
            continue

        out_dir = PLOT_DIR / prefix
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Plotting '{term}' ({prefix}) — {len(wc)} windows, {len(rc)} roc cols...")
        plot_mean_trend(df, wc, prefix, out_dir)
        plot_coverage(df, wc, prefix, out_dir)
        plot_slope_dist(df, prefix, out_dir)
        plot_change_dist(df, prefix, out_dir)
        plot_roc_trend(df, rc, prefix, out_dir)
        plot_n_windows(df, prefix, out_dir)

        all_stats[prefix] = compute_stats(df, prefix)

    print("Done. Plots saved to output/plots/<prefix>/")
    return df, all_stats


if __name__ == "__main__":
    _, all_stats = run()
    for prefix, s in all_stats.items():
        print(f"\n── {prefix} ──")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
