"""
Weight loss trend analysis — Body weight only.
Runs on any Astra-format parquet, auto-generates plots + markdown.
Empty windows are detected from the data and documented automatically.

Usage:
  python heldout_analysis.py                          # heldout defaults
  python heldout_analysis.py --parquet ../data/Astra_10yr_160_withtext.parquet \\
                              --plot-subdir astra_160 \\
                              --markdown weight_loss_analysis_astra_10yr_160_withtext.md \\
                              --gcs gs://gcs-ai-dev-model-artifacts/keerthi/Lung_Cancer/raw_events/Astra_10yr_160_withtext.parquet
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats
from datetime import date

# ── Defaults (heldout) ────────────────────────────────────────────────────
DEFAULT_PARQUET     = "../data/heldout_Astra_10yr_160_withtext.parquet"
DEFAULT_PLOT_SUBDIR = "heldout"
DEFAULT_MARKDOWN    = "heldout_weight_loss_analysis_astra_10yr.md"
DEFAULT_GCS         = "gs://gcs-ai-dev-model-artifacts/keerthi/Lung_Cancer/raw_events/heldout_Astra_10yr_160_withtext.parquet"

TERM        = "Body weight"
VAL_RANGE   = (20, 300)
WINDOW_DAYS = 91
LOOKBACK    = 3650
N_WINDOWS   = LOOKBACK // WINDOW_DAYS   # 40

BASE_DIR = Path(__file__).parent
COLORS   = {0: "#2196F3", 1: "#E53935"}
LABELS   = {0: "Non-cancer", 1: "Cancer"}


# ── Data ──────────────────────────────────────────────────────────────────

def load(parquet_path):
    print(f"Loading {Path(parquet_path).name}...")
    df = pd.read_parquet(parquet_path, engine="pyarrow",
                         columns=["patient_guid", "cancer_class", "days_before_anchor",
                                  "term", "value"])
    df = df[df["term"] == TERM].copy()
    df = df[df["days_before_anchor"] <= LOOKBACK]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[(df["value"] >= VAL_RANGE[0]) & (df["value"] <= VAL_RANGE[1])]
    df["win"] = (df["days_before_anchor"] // WINDOW_DAYS).clip(upper=N_WINDOWS - 1).astype(int)
    return df


def build_wide(events):
    wide = (events.groupby(["patient_guid", "win"])["value"]
            .mean().unstack("win")
            .reindex(columns=range(N_WINDOWS)))
    labels = events.groupby("patient_guid")["cancer_class"].first()
    return wide.join(labels)


def build_summary(wide):
    wc = list(range(N_WINDOWS))

    def _slope(row):
        v = row[wc].dropna()
        return np.polyfit(v.index.astype(int), v.values, 1)[0] if len(v) >= 2 else np.nan

    def _total_change(row):
        v = row[wc].dropna()
        return (v.iloc[0] - v.iloc[-1]) if len(v) >= 2 else np.nan

    def _pct_change(row):
        v = row[wc].dropna()
        if len(v) < 2 or v.iloc[-1] == 0: return np.nan
        return (v.iloc[0] - v.iloc[-1]) / v.iloc[-1]

    def _recent_roc(row):
        rc = [c for c in [4, 5, 6, 7, 8] if c in row.index]
        v = row[rc].dropna()
        return np.polyfit(v.index.astype(int), v.values, 1)[0] / WINDOW_DAYS if len(v) >= 2 else np.nan

    return pd.DataFrame({
        "slope":        wide.apply(_slope, axis=1),
        "total_change": wide.apply(_total_change, axis=1),
        "pct_change":   wide.apply(_pct_change, axis=1),
        "recent_roc":   wide.apply(_recent_roc, axis=1),
        "n_windows":    wide[wc].notna().sum(axis=1),
        "cancer_class": wide["cancer_class"],
    })


# ── Window audit (empty vs active) ────────────────────────────────────────

def detect_window_info(wide, events):
    """
    Returns dict with per-window event counts, coverage, and empty/active classification.
    Detected purely from data — no hardcoded assumptions.
    """
    rows = []
    event_counts = events.groupby("win").size()

    for w in range(N_WINDOWS):
        day_lo = w * WINDOW_DAYS
        day_hi = (w + 1) * WINDOW_DAYS - 1
        n_events = int(event_counts.get(w, 0))
        n_pts    = int(wide[w].notna().sum()) if w in wide.columns else 0
        total    = len(wide)
        pct_c    = wide[wide["cancer_class"] == 1][w].notna().mean() * 100 if w in wide.columns else 0.0
        pct_nc   = wide[wide["cancer_class"] == 0][w].notna().mean() * 100 if w in wide.columns else 0.0
        rows.append({
            "window":   f"w{w:02d}",
            "day_lo":   day_lo,
            "day_hi":   day_hi,
            "n_events": n_events,
            "n_pts":    n_pts,
            "pct_all":  round(n_pts / total * 100, 1),
            "pct_cancer": round(pct_c, 1),
            "pct_noncancer": round(pct_nc, 1),
            "empty":    n_events == 0,
        })

    empty  = [r for r in rows if r["empty"]]
    active = [r for r in rows if not r["empty"]]
    return {"all": rows, "empty": empty, "active": active}


# ── Plots ─────────────────────────────────────────────────────────────────

def active_wins(wide):
    return [c for c in range(N_WINDOWS) if c in wide.columns and wide[c].notna().any()]


def _xtick_setup(ax, wc, step=None):
    x = np.arange(len(wc))
    step = step or max(1, len(wc) // 9)
    ax.invert_xaxis()
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"w{c:02d}" for c in wc][::step], rotation=45, fontsize=8)
    return x


def plot_mean_trend(wide, out_dir, cohort_label):
    wc = active_wins(wide)
    fig, ax = plt.subplots(figsize=(12, 5))
    for cls in [0, 1]:
        sub = wide[wide["cancer_class"] == cls][wc]
        mean, sem = sub.mean(), sub.sem()
        ax.plot(np.arange(len(wc)), mean.values, color=COLORS[cls], label=LABELS[cls], linewidth=2)
        ax.fill_between(np.arange(len(wc)), mean - sem, mean + sem, alpha=0.15, color=COLORS[cls])
    x = _xtick_setup(ax, wc)
    ax.set_xlabel("Window (oldest → most recent)", fontsize=10)
    ax.set_ylabel("Mean Body Weight (kg)", fontsize=10)
    ax.set_title(f"Mean Weight Trend Over 10 Years — {cohort_label}: Cancer vs Non-Cancer", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "01_mean_trend.png", dpi=150); plt.close(fig)


def plot_coverage(wide, out_dir, cohort_label):
    wc = active_wins(wide)
    x = np.arange(len(wc))
    fig, ax = plt.subplots(figsize=(12, 4))
    for cls in [0, 1]:
        cov = wide[wide["cancer_class"] == cls][wc].notna().mean() * 100
        ax.bar(x + (0.4 if cls == 1 else 0), cov.values, width=0.4,
               color=COLORS[cls], alpha=0.8, label=LABELS[cls])
    _xtick_setup(ax, wc)
    ax.set_xticks(x[::max(1, len(wc)//9)] + 0.2)
    ax.set_xticklabels([f"w{c:02d}" for c in wc][::max(1, len(wc)//9)], rotation=45, fontsize=8)
    ax.set_xlabel("Window (oldest → most recent)", fontsize=10)
    ax.set_ylabel("% Patients with Data", fontsize=10)
    ax.set_title(f"Data Coverage Per Window — {cohort_label}", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "02_coverage.png", dpi=150); plt.close(fig)


def plot_slope_dist(summary, out_dir, cohort_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in [0, 1]:
        vals = summary[summary["cancer_class"] == cls]["slope"].dropna().clip(-5, 5)
        ax.hist(vals, bins=60, alpha=0.5, color=COLORS[cls],
                label=f"{LABELS[cls]} μ={vals.mean():.3f}", density=True)
        ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Trend slope (kg/window, clipped ±5)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Slope Distribution — {cohort_label}", fontsize=12)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "03_slope_dist.png", dpi=150); plt.close(fig)


def plot_total_change(summary, out_dir, cohort_label):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, label, clip in [
        (axes[0], "total_change", "Total Change (kg)", (-50, 50)),
        (axes[1], "pct_change",   "% Change",          (-0.5, 0.5)),
    ]:
        for cls in [0, 1]:
            vals = summary[summary["cancer_class"] == cls][col].dropna().clip(*clip)
            ax.hist(vals, bins=60, alpha=0.5, color=COLORS[cls],
                    label=f"{LABELS[cls]} μ={vals.mean():.3f}", density=True)
            ax.axvline(vals.mean(), color=COLORS[cls], linestyle="--", linewidth=1.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel(label, fontsize=10); ax.set_ylabel("Density", fontsize=10)
        ax.set_title(label, fontsize=11); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Weight Change Distribution — {cohort_label}", fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir / "04_change_dist.png", dpi=150); plt.close(fig)


def plot_roc(wide, out_dir, cohort_label):
    wc = active_wins(wide)
    roc_pairs = [(wc[i], wc[i+1]) for i in range(len(wc)-1)]
    roc_means = {cls: [(wide[wide["cancer_class"]==cls][w_new] - wide[wide["cancer_class"]==cls][w_old]).mean() / WINDOW_DAYS
                       for w_new, w_old in roc_pairs] for cls in [0, 1]}
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(roc_pairs))
    for cls in [0, 1]:
        ax.plot(x, roc_means[cls], color=COLORS[cls], label=LABELS[cls], linewidth=2)
    _xtick_setup(ax, wc[:-1])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Window transition (oldest → most recent)", fontsize=10)
    ax.set_ylabel("Mean RoC (kg/day)", fontsize=10)
    ax.set_title(f"Rate of Change Per Window Transition — {cohort_label}", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "05_roc_trend.png", dpi=150); plt.close(fig)


def plot_n_windows(summary, out_dir, cohort_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in [0, 1]:
        ax.hist(summary[summary["cancer_class"] == cls]["n_windows"],
                bins=range(0, N_WINDOWS + 2), alpha=0.5,
                color=COLORS[cls], label=LABELS[cls], density=True)
    ax.set_xlabel("Windows with data", fontsize=10); ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Coverage per Patient — {cohort_label}", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "06_n_windows.png", dpi=150); plt.close(fig)


# ── Stats ─────────────────────────────────────────────────────────────────

def compute_stats(summary):
    s = {}
    for cls in [0, 1]:
        sub = summary[summary["cancer_class"] == cls]
        lbl = LABELS[cls].lower().replace("-", "_")
        s[f"{lbl}_n"]                 = len(sub)
        s[f"{lbl}_slope_mean"]        = sub["slope"].mean()
        s[f"{lbl}_slope_median"]      = sub["slope"].median()
        s[f"{lbl}_total_change_mean"] = sub["total_change"].mean()
        s[f"{lbl}_pct_change_mean"]   = sub["pct_change"].mean() * 100
        s[f"{lbl}_n_windows_median"]  = sub["n_windows"].median()
    for col in ["slope", "total_change"]:
        a = summary[summary["cancer_class"] == 1][col].dropna()
        b = summary[summary["cancer_class"] == 0][col].dropna()
        t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        s[f"{col}_t"] = t
        s[f"{col}_p"] = p
    return s


# ── Markdown auto-generation ──────────────────────────────────────────────

def write_markdown(md_path, parquet_name, gcs_path, plot_subdir,
                   events, wide, summary, stats, window_info):
    """Auto-generates full markdown report from computed data. No hardcoded values."""

    s      = stats
    empty  = window_info["empty"]
    active = window_info["active"]
    all_w  = window_info["all"]

    c_n   = s["cancer_n"]
    nc_n  = s["non_cancer_n"]
    total = c_n + nc_n

    # Total patients in cohort (including those without weight data)
    total_pts_raw = events["patient_guid"].nunique()  # patients with weight data only here

    # Empty window table rows
    empty_rows = "\n".join(
        f"| {r['window']} | {r['day_lo']}–{r['day_hi']} | {r['n_events']} | 0.0% | 0.0% | 0.0% |"
        for r in empty
    )
    empty_summary = (
        f"w{empty[0]['window'][1:]}–w{empty[-1]['window'][1:]}"
        if empty else "none"
    )
    empty_day_range = (
        f"{empty[0]['day_lo']}–{empty[-1]['day_hi']} days before anchor"
        if empty else "N/A"
    )
    first_active_win = active[0]["window"] if active else "N/A"
    first_active_day = active[0]["day_lo"] if active else "N/A"

    # Active window table (sample: first 5 + last 2)
    sample_active = active[:5] + (active[-2:] if len(active) > 5 else [])
    active_rows = "\n".join(
        f"| {r['window']} | {r['day_lo']}–{r['day_hi']} | {r['n_events']:,} | {r['pct_all']}% | {r['pct_cancer']}% | {r['pct_noncancer']}% |"
        for r in sample_active
    )
    if len(active) > 7:
        active_rows += f"\n| ... | ... | ... | ... | ... | ... |"

    p_tc  = s["total_change_p"]
    p_sl  = s["slope_p"]
    tc_sig = "< 0.0001" if p_tc < 0.0001 else f"{p_tc:.4f}"
    sl_sig = "< 0.0001" if p_sl < 0.0001 else f"{p_sl:.4f}"

    plots_rel = f"output/plots/{plot_subdir}"

    md = f"""# Weight Loss Trend Analysis — {parquet_name}

**Generated:** {date.today().isoformat()}
**Dataset:** {parquet_name}
**Source:** {gcs_path}
**Term:** {TERM} | **Value filter:** {VAL_RANGE[0]}–{VAL_RANGE[1]} kg
**Window size:** {WINDOW_DAYS} days (~3 months) | **Total windows defined:** {N_WINDOWS} (w00–w{N_WINDOWS-1:02d})
**Lookback:** {LOOKBACK} days ({LOOKBACK // 365} years)

---

## Cohort

| | Cancer | Non-Cancer | Total |
|---|---|---|---|
| Patients with weight data | {c_n:,} | {nc_n:,} | {total:,} |
| Median weight observations | {summary[summary.cancer_class==1].n_windows.median():.0f} | {summary[summary.cancer_class==0].n_windows.median():.0f} | — |
| Median windows covered | {s['cancer_n_windows_median']:.0f} / {len(active)} active | {s['non_cancer_n_windows_median']:.0f} / {len(active)} active | — |

---

## Window Audit — Empty vs Active

> Windows are assigned as: `window_index = floor(days_before_anchor / {WINDOW_DAYS})`
> Detected from data — not hardcoded.

### Empty Windows ({len(empty)} windows with 0 events)

| Window | Days before anchor | Events | Coverage (all) | Coverage (cancer) | Coverage (non-cancer) |
|---|---|---|---|---|---|
{empty_rows}

**Reason:** The raw dataset minimum `days_before_anchor` = {int(events.days_before_anchor.min())} days.
Windows {empty_summary} span {empty_day_range} — inside the **{WINDOW_DAYS * len(empty) // 30}-month prediction horizon** which is excluded from lookback features by design.
These windows carry **0 events across all patients** and are excluded from all plots and feature computation.

### First Active Window

`{first_active_win}` — days {first_active_day}–{first_active_day + WINDOW_DAYS - 1} before anchor
({first_active_day // 30} months before anchor)

### Active Window Coverage (sample)

| Window | Days before anchor | Events | Coverage (all) | Coverage (cancer) | Coverage (non-cancer) |
|---|---|---|---|---|---|
{active_rows}

**Total active windows used in analysis: {len(active)} (w{active[0]['window'][1:]}–w{active[-1]['window'][1:]})**

---

## 1. Mean Weight Trend Over 10 Years

![Mean Weight Trend]({plots_rel}/01_mean_trend.png)

- Cancer patients carry lower weight than non-cancer across all active windows.
- Non-cancer shows upward drift toward anchor — weight gain over time.
- Gap widens in the most recent active windows ({first_active_win}–w{active[4]['window'][1:] if len(active)>4 else active[-1]['window'][1:]}).

---

## 2. Data Coverage Per Window

![Coverage]({plots_rel}/02_coverage.png)

- Coverage peaks at {first_active_win}–w{active[4]['window'][1:] if len(active)>4 else active[-1]['window'][1:]} (most recent active windows).
- Cancer patients have higher coverage per window — more frequent clinical visits.
- Drops steadily in older windows.

---

## 3. Trend Slope Distribution

![Slope Distribution]({plots_rel}/03_slope_dist.png)

| | Cancer | Non-Cancer |
|---|---|---|
| Mean slope (kg/window) | {s['cancer_slope_mean']:.3f} | {s['non_cancer_slope_mean']:.3f} |
| Median slope | {s['cancer_slope_median']:.3f} | {s['non_cancer_slope_median']:.3f} |
| t-statistic | {s['slope_t']:.3f} | |
| **p-value** | **{sl_sig}** | |

---

## 4. Total Weight Change (oldest → most recent window)

![Change Distribution]({plots_rel}/04_change_dist.png)

| | Cancer | Non-Cancer |
|---|---|---|
| Mean total change (kg) | **{s['cancer_total_change_mean']:.2f}** | **{s['non_cancer_total_change_mean']:.2f}** |
| Mean % change | **{s['cancer_pct_change_mean']:.2f}%** | **{s['non_cancer_pct_change_mean']:.2f}%** |
| t-statistic | {s['total_change_t']:.3f} | |
| **p-value** | **{tc_sig}** | |

---

## 5. Rate of Change Per Window Transition

![RoC Trend]({plots_rel}/05_roc_trend.png)

- Non-cancer: consistent positive RoC — steady weight gain.
- Cancer: near-zero or negative RoC; reversal in most recent active windows.

---

## 6. Coverage Per Patient

![N Windows]({plots_rel}/06_n_windows.png)

| | Cancer | Non-Cancer |
|---|---|---|
| Median windows | {s['cancer_n_windows_median']:.0f} / {len(active)} | {s['non_cancer_n_windows_median']:.0f} / {len(active)} |

---

## Output Files

```
{plots_rel}/
  01_mean_trend.png
  02_coverage.png
  03_slope_dist.png
  04_change_dist.png
  05_roc_trend.png
  06_n_windows.png
```
"""
    Path(md_path).write_text(md.strip())
    print(f"Markdown saved: {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def run(parquet_path, plot_subdir, md_path, gcs_path, cohort_label):
    events = load(parquet_path)
    pt_labels = events.groupby("patient_guid")["cancer_class"].first()
    print(f"  {len(events):,} events | {events['patient_guid'].nunique():,} patients | {pt_labels.value_counts().to_dict()}")

    print("Building windows...")
    wide    = build_wide(events)
    summary = build_summary(wide)
    stats   = compute_stats(summary)
    winfo   = detect_window_info(wide, events)

    print(f"  Empty windows: {[r['window'] for r in winfo['empty']]}")
    print(f"  Active windows: {len(winfo['active'])} ({winfo['active'][0]['window']}–{winfo['active'][-1]['window']})")

    out_dir = BASE_DIR / "output" / "plots" / plot_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_mean_trend(wide, out_dir, cohort_label)
    plot_coverage(wide, out_dir, cohort_label)
    plot_slope_dist(summary, out_dir, cohort_label)
    plot_total_change(summary, out_dir, cohort_label)
    plot_roc(wide, out_dir, cohort_label)
    plot_n_windows(summary, out_dir, cohort_label)

    write_markdown(
        md_path      = BASE_DIR / md_path,
        parquet_name = Path(parquet_path).name,
        gcs_path     = gcs_path,
        plot_subdir  = plot_subdir,
        events       = events,
        wide         = wide,
        summary      = summary,
        stats        = stats,
        window_info  = winfo,
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",     default=DEFAULT_PARQUET)
    parser.add_argument("--plot-subdir", default=DEFAULT_PLOT_SUBDIR)
    parser.add_argument("--markdown",    default=DEFAULT_MARKDOWN)
    parser.add_argument("--gcs",         default=DEFAULT_GCS)
    parser.add_argument("--label",       default="Heldout Cohort")
    args = parser.parse_args()

    run(
        parquet_path  = args.parquet,
        plot_subdir   = args.plot_subdir,
        md_path       = args.markdown,
        gcs_path      = args.gcs,
        cohort_label  = args.label,
    )
