"""
Combined visualization: Body weight + BMI side-by-side sliding window plots.
Trims structurally empty w00-w03 (12mo prediction horizon excluded from raw data).
Outputs to output/plots/combined/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

OUT_DIR = Path(__file__).parent / "output" / "plots" / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {0: "#2196F3", 1: "#E53935"}
LABELS = {0: "Non-cancer", 1: "Cancer"}

TERMS = {
    "body_weight":     "Body Weight (kg)",
    "body_mass_index": "BMI (kg/m²)",
}

# w00-w03 are always empty (min days_before_anchor=366 → 12mo horizon excluded)
SKIP_WINDOWS = 4


def load():
    return pd.read_parquet(Path(__file__).parent / "output" / "weight_features.parquet")


def active_win_cols(df, prefix):
    all_w = sorted([c for c in df.columns
                    if c.startswith(f"{prefix}_w") and "_roc_" not in c
                    and c[-2:].isdigit()])
    return [c for c in all_w if df[c].notna().any()]


def active_roc_cols(df, prefix):
    return sorted([c for c in df.columns if f"{prefix}_roc_" in c
                   and df[c].notna().any()])


# ── Plot 1: Mean trend — BW and BMI side by side ───────────────────────────

def plot_mean_trend_combined(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (prefix, ylabel) in zip(axes, TERMS.items()):
        wc = active_win_cols(df, prefix)
        x = np.arange(len(wc))
        x_labels = [c.split("_w")[1] for c in wc]
        for cls in [0, 1]:
            sub = df[df["cancer_class"] == cls][wc]
            mean, sem = sub.mean(), sub.sem()
            ax.plot(x, mean.values, color=COLORS[cls], label=LABELS[cls], linewidth=2)
            ax.fill_between(x, mean - sem, mean + sem, alpha=0.15, color=COLORS[cls])
        ax.invert_xaxis()
        step = max(1, len(wc) // 9)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(x_labels[::step], rotation=45, fontsize=8)
        ax.set_xlabel("Window (oldest → most recent)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Mean {ylabel} Trend", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Sliding Window Trend — Cancer vs Non-Cancer (10yr lookback, 3mo windows)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_mean_trend_combined.png", dpi=150)
    plt.close(fig)


# ── Plot 2: Coverage — BW and BMI side by side ────────────────────────────

def plot_coverage_combined(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, (prefix, title) in zip(axes, TERMS.items()):
        wc = active_win_cols(df, prefix)
        x = np.arange(len(wc))
        for cls in [0, 1]:
            cov = df[df["cancer_class"] == cls][wc].notna().mean() * 100
            ax.bar(x + (0.4 if cls == 1 else 0), cov.values, width=0.4,
                   color=COLORS[cls], alpha=0.8, label=LABELS[cls])
        ax.invert_xaxis()
        step = max(1, len(wc) // 9)
        x_labels = [c.split("_w")[1] for c in wc]
        ax.set_xticks(x[::step] + 0.2)
        ax.set_xticklabels(x_labels[::step], rotation=45, fontsize=8)
        ax.set_xlabel("Window (oldest → most recent)", fontsize=10)
        ax.set_ylabel("% Patients with Data", fontsize=10)
        ax.set_title(f"Coverage — {title}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Data Coverage Per Window", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_coverage_combined.png", dpi=150)
    plt.close(fig)


# ── Plot 3: Slope distribution — BW and BMI side by side ──────────────────

def plot_slope_combined(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (prefix, title) in zip(axes, TERMS.items()):
        col = f"{prefix}_slope"
        for cls in [0, 1]:
            vals = df[df["cancer_class"] == cls][col].dropna().clip(-5, 5)
            ax.hist(vals, bins=60, alpha=0.5, color=COLORS[cls],
                    label=f"{LABELS[cls]} μ={vals.mean():.3f}", density=True)
            ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.5)
        ax.set_xlabel("Trend slope (per window, clipped ±5)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Slope Distribution — {title}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Weight Trend Slope: Cancer vs Non-Cancer", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_slope_combined.png", dpi=150)
    plt.close(fig)


# ── Plot 4: Total change distribution — BW and BMI side by side ───────────

def plot_total_change_combined(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    clips = {"body_weight": (-50, 50), "body_mass_index": (-15, 15)}
    for ax, (prefix, title) in zip(axes, TERMS.items()):
        col = f"{prefix}_total_change"
        lo, hi = clips[prefix]
        for cls in [0, 1]:
            vals = df[df["cancer_class"] == cls][col].dropna().clip(lo, hi)
            ax.hist(vals, bins=60, alpha=0.5, color=COLORS[cls],
                    label=f"{LABELS[cls]} μ={vals.mean():.2f}", density=True)
            ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel(f"Total change ({title})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Total Change — {title}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Total Change (oldest → most recent window): Cancer vs Non-Cancer", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_total_change_combined.png", dpi=150)
    plt.close(fig)


# ── Plot 5: Rate of change — BW and BMI side by side ──────────────────────

def plot_roc_combined(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (prefix, title) in zip(axes, TERMS.items()):
        rc = active_roc_cols(df, prefix)
        x = np.arange(len(rc))
        for cls in [0, 1]:
            mean = df[df["cancer_class"] == cls][rc].mean()
            ax.plot(x, mean.values, color=COLORS[cls], label=LABELS[cls], linewidth=2)
        ax.invert_xaxis()
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        step = max(1, len(rc) // 9)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([c.split("_roc_")[1] for c in rc][::step], rotation=45, fontsize=8)
        ax.set_xlabel("Window transition (oldest → most recent)", fontsize=10)
        ax.set_ylabel("Mean rate of change (per day)", fontsize=10)
        ax.set_title(f"Rate of Change — {title}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Rate of Change Per Window Transition: Cancer vs Non-Cancer", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_roc_combined.png", dpi=150)
    plt.close(fig)


# ── Plot 6: n_windows distribution — BW and BMI side by side ──────────────

def plot_n_windows_combined(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (prefix, title) in zip(axes, TERMS.items()):
        col = f"{prefix}_n_windows"
        for cls in [0, 1]:
            ax.hist(df[df["cancer_class"] == cls][col],
                    bins=range(0, 42), alpha=0.5, color=COLORS[cls],
                    label=LABELS[cls], density=True)
        ax.set_xlabel("Windows with data", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Coverage per Patient — {title}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Number of 3-Month Windows with Data per Patient", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_n_windows_combined.png", dpi=150)
    plt.close(fig)


# ── Stats ──────────────────────────────────────────────────────────────────

def compute_stats(df):
    all_stats = {}
    for prefix in TERMS:
        s = {}
        for cls in [0, 1]:
            sub = df[df["cancer_class"] == cls]
            lbl = LABELS[cls].lower().replace("-", "_")
            s[f"{lbl}_n"]                 = len(sub)
            s[f"{lbl}_slope_mean"]        = sub[f"{prefix}_slope"].mean()
            s[f"{lbl}_slope_median"]      = sub[f"{prefix}_slope"].median()
            s[f"{lbl}_total_change_mean"] = sub[f"{prefix}_total_change"].mean()
            s[f"{lbl}_pct_change_mean"]   = sub[f"{prefix}_pct_change"].mean() * 100
            s[f"{lbl}_n_windows_median"]  = sub[f"{prefix}_n_windows"].median()
            s[f"{lbl}_n_obs_median"]      = sub[f"{prefix}_n_obs"].median()
        for col_sfx, key in [("_slope", "slope"), ("_total_change", "total_change")]:
            a = df[df["cancer_class"] == 1][f"{prefix}{col_sfx}"].dropna()
            b = df[df["cancer_class"] == 0][f"{prefix}{col_sfx}"].dropna()
            t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
            s[f"{key}_t"] = t
            s[f"{key}_p"] = p
        all_stats[prefix] = s
    return all_stats


def run():
    print("Loading features...")
    df = load()
    print("Generating combined plots...")
    plot_mean_trend_combined(df)
    plot_coverage_combined(df)
    plot_slope_combined(df)
    plot_total_change_combined(df)
    plot_roc_combined(df)
    plot_n_windows_combined(df)
    stats = compute_stats(df)
    print(f"Saved to {OUT_DIR}")
    return stats


if __name__ == "__main__":
    stats = run()
    for prefix, s in stats.items():
        print(f"\n── {prefix} ──")
        for k, v in s.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
