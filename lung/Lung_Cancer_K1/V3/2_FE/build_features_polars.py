"""
Polars-accelerated per-code feature engineering — drop-in backend for build_features.
=====================================================================================
Selected with FE_ENGINE=polars (default is pandas). This module re-uses ALL of build_features —
load_events, the codelist/tiered partition, the concept families, the global/comment/derangement/
blood-ratio families, the fill rules, and `build()` itself — and ONLY swaps the six heavy per-code
families (occurrence / flag / age / value / band / cumulative) for Polars implementations.

Each Polars family does the expensive per-(patient_guid, code) reduction in Polars, then re-uses the
exact pandas helpers for the parts where float/ordering parity matters most:
  - `bf._wide`         — identical wide-reshape + column naming,
  - `bf._group_linreg` — identical summed-moment OLS slope/correlation,
  - `bf._halves`       — identical chronological-halves frequency / worsening.
So the output is column-for-column and value-for-value equivalent to the pandas engine (verified by
`fe_parity_check.py`); only the grouping is faster. Use this engine only once that check passes.
"""
import os
import sys
import numpy as np
import pandas as pd
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_features as bf

KEYS = ["patient_guid", "code"]


def _idx(pldf):
    """Polars frame -> pandas, indexed by (patient_guid, code) for bf._wide."""
    return pldf.to_pandas().set_index(KEYS)


# --------------------------------------------------------------------------- the six per-code families
def flag_features(ev):
    if ev.empty:
        return bf.flag_features(ev)
    a = _idx(pl.from_pandas(ev[["patient_guid", "code", "active", "sig"]])
             .group_by(KEYS).agg(pl.col("active").max(), pl.col("sig").max()))
    return pd.concat([bf._wide(a["active"], "has_active"),
                      bf._wide(a["sig"], "has_significant")], axis=1)


def age_features(ev):
    if ev.empty:
        return bf.age_features(ev)
    df = pl.from_pandas(ev[["patient_guid", "code", "days", "event_age"]])
    # pandas groupby.first()/last() skip NaN, so rank on the non-null rows, days ascending
    # (smallest days = most recent = age_last; largest days = earliest = age_first).
    nn = df.filter(pl.col("event_age").is_not_null()).sort(KEYS + ["days"])
    fl = nn.group_by(KEYS, maintain_order=True).agg(
        age_last=pl.col("event_age").first(), age_first=pl.col("event_age").last())
    med = df.group_by(KEYS).agg(age_median=pl.col("event_age").median())
    a = _idx(fl.join(med, on=KEYS, how="full", coalesce=True))
    return pd.concat([bf._wide(a["age_first"], "age_first"),
                      bf._wide(a["age_last"], "age_last"),
                      bf._wide(a["age_median"], "age_median")], axis=1)


def occurrence_features(ev):
    if ev.empty:
        return bf.occurrence_features(ev)
    e = ev.copy()
    e["_moc"] = e["months"] - e["months"].min()          # months since data cutoff (pandas, exact)
    df = pl.from_pandas(e[["patient_guid", "code", "days", "_moc"]]).with_columns([
        (-pl.col("_moc") / bf.DECAY_TAU_MONTHS).exp().alias("_w"),
        (pl.col("_moc") <= bf.RECENT_RATIO_CUTOFF).cast(pl.Float64).alias("_recent"),
    ])
    base = _idx(df.group_by(KEYS).agg([
        pl.len().alias("count"),
        pl.col("_moc").min().alias("recency_months"),
        pl.col("days").max().alias("_dmax"), pl.col("days").min().alias("_dmin"),
        pl.col("_w").sum().alias("decay_intensity"),
        pl.col("_recent").sum().alias("_recsum"),
    ]))

    def _binct(lo, hi):
        sub = df.filter((pl.col("_moc") >= lo) & (pl.col("_moc") < hi)).group_by(KEYS).agg(pl.len().alias("c"))
        # cast to SIGNED int64: pl.len() is uint32, and the 2nd-difference accel can go negative
        return _idx(sub)["c"].reindex(base.index, fill_value=0).astype("int64")

    count = base["count"]
    timespan_years = (base["_dmax"] - base["_dmin"]) / 365.25
    recency = base["recency_months"]
    decay = base["decay_intensity"]
    recent_ratio = base["_recsum"] / count
    accel = _binct(*bf.ACCEL_BINS["recent"]) - 2 * _binct(*bf.ACCEL_BINS["mid"]) + _binct(*bf.ACCEL_BINS["old"])
    freq_per_year = count / timespan_years.replace(0, np.nan)

    # intervals / frequency-trend / halves: event-level, re-use the exact pandas helpers
    tev = e if bf.TREND_MAX_MONTHS is None else e[e["_moc"] <= bf.TREND_MAX_MONTHS]
    s = tev[KEYS + ["days"]].sort_values(KEYS + ["days"])
    s["_iv"] = s.groupby(KEYS)["days"].diff().abs()
    iv = s.dropna(subset=["_iv"])
    gi = iv.groupby(KEYS)["_iv"]
    interval_median, interval_min, interval_max = gi.median(), gi.min(), gi.max()
    iv = iv.assign(_ord=iv.groupby(KEYS).cumcount())
    freq_trend_slope, _ = bf._group_linreg(iv, "_ord", "_iv")
    first_half_freq, second_half_freq, is_worsening = bf._halves(tev, KEYS)

    parts = {
        "count": count, "present": (count > 0).astype(int), "recency_months": recency,
        "decay_intensity": decay, "accel": accel, "recent_ratio": recent_ratio,
        "freq_per_year": freq_per_year, "timespan_years": timespan_years,
        "interval_median": interval_median, "interval_min": interval_min,
        "interval_max": interval_max, "freq_trend_slope": freq_trend_slope,
        "first_half_freq": first_half_freq, "second_half_freq": second_half_freq,
        "is_worsening": is_worsening,
    }
    return pd.concat([bf._wide(s_, sfx) for sfx, s_ in parts.items()], axis=1)


def value_features(ev, value_codes):
    val = ev[ev["code"].isin(value_codes)].dropna(subset=["value_num"]).copy()
    if val.empty:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    df = pl.from_pandas(val[["patient_guid", "code", "days", "value_num"]])
    st = _idx(df.group_by(KEYS).agg([
        pl.col("value_num").mean().alias("val_mean"),
        pl.col("value_num").median().alias("val_median"),
        pl.col("value_num").min().alias("val_min"),
        pl.col("value_num").max().alias("val_max"),
        pl.col("value_num").std().alias("val_std"),           # ddof=1, matches pandas
    ]))
    vs = _idx(df.sort(KEYS + ["days"]).group_by(KEYS, maintain_order=True).agg([
        pl.col("value_num").first().alias("val_latest"),       # asc days -> recent first
        pl.col("value_num").last().alias("val_first"),
    ]))
    mean, vmax, vmin, vstd = st["val_mean"], st["val_max"], st["val_min"], st["val_std"]
    latest, first = vs["val_latest"], vs["val_first"]
    stats = {"val_mean": mean, "val_median": st["val_median"], "val_min": vmin, "val_max": vmax,
             "val_std": vstd, "val_range": vmax - vmin, "val_first": first, "val_latest": latest,
             "val_abs_change": latest - first,
             "val_pct_change": (latest - first) / first.replace(0, np.nan) * 100.0,
             "val_latest_z": (latest - mean) / vstd.replace(0, np.nan)}
    # trend + acceleration: re-use the exact pandas summed-moment OLS
    cutoff = ev["months"].min()
    tval = val if bf.TREND_MAX_MONTHS is None else val[(val["months"] - cutoff) <= bf.TREND_MAX_MONTHS]
    tval = tval.assign(_x=-tval["days"])
    slope, corr = bf._group_linreg(tval, "_x", "value_num")
    stats["val_trend_slope"], stats["val_trend_corr"] = slope, corr
    vsrt = val.sort_values(KEYS + ["days"], ascending=[True, True, False])      # oldest first
    vsrt["_n"] = vsrt.groupby(KEYS)["days"].transform("size")
    vsrt["_r"] = vsrt.groupby(KEYS).cumcount()
    h = vsrt[vsrt["_n"] >= 4].copy()
    if not h.empty:
        h["_half"] = np.where(h["_r"] < h["_n"] / 2.0, "old", "new")
        h["_x"] = -h["days"]
        s_old, _ = bf._group_linreg(h[h["_half"] == "old"], "_x", "value_num")
        s_new, _ = bf._group_linreg(h[h["_half"] == "new"], "_x", "value_num")
        stats["val_accel"] = s_new - s_old
    return pd.concat([bf._wide(s_, sfx) for sfx, s_ in stats.items()], axis=1)


def band_features(ev, bands, value_codes, relative):
    e = ev.copy()
    if relative:
        e["_mo"] = e["months"] - e.groupby("patient_guid")["months"].transform("min")
    else:
        e["_mo"] = e["months"] - e["months"].min()
    df = pl.from_pandas(e[["patient_guid", "code", "days", "value_num", "_mo"]])
    vc = list(value_codes)
    blocks = []
    for lo, hi in bands:
        tag = f"w{int(lo)}_{int(hi)}"
        msub = df.filter((pl.col("_mo") >= lo) & (pl.col("_mo") < hi))
        cnt = _idx(msub.group_by(KEYS).agg(pl.len().alias("c")))["c"]
        blocks.append(bf._wide(cnt, f"count_{tag}"))
        blocks.append(bf._wide((cnt > 0).astype(int), f"present_{tag}"))
        vsub = msub.filter(pl.col("code").is_in(vc) & pl.col("value_num").is_not_null())
        if vsub.height:
            blocks.append(bf._wide(_idx(vsub.group_by(KEYS).agg(
                pl.col("value_num").mean().alias("m")))["m"], f"val_mean_{tag}"))
            blocks.append(bf._wide(_idx(vsub.sort(KEYS + ["days"]).group_by(KEYS, maintain_order=True).agg(
                pl.col("value_num").first().alias("l")))["l"], f"val_latest_{tag}"))
            vpd = (e[(e["_mo"] >= lo) & (e["_mo"] < hi) & e["code"].isin(value_codes)]
                   .dropna(subset=["value_num"]).assign(_x=lambda d: -d["days"]))
            slope, _ = bf._group_linreg(vpd, "_x", "value_num")
            blocks.append(bf._wide(slope, f"val_slope_{tag}"))
    if not blocks:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    return pd.concat(blocks, axis=1)


def cumulative_features(ev, windows, value_codes, relative):
    e = ev.copy()
    if relative:
        e["_mo"] = e["months"] - e.groupby("patient_guid")["months"].transform("min")
    else:
        e["_mo"] = e["months"] - e["months"].min()
    df = pl.from_pandas(e[["patient_guid", "code", "days", "value_num", "_mo"]])
    vc = list(value_codes)
    blocks = []
    for n in windows:
        tag = f"last{int(n)}"
        sub = df.filter(pl.col("_mo") < n)
        cnt = _idx(sub.group_by(KEYS).agg(pl.len().alias("c")))["c"]
        blocks.append(bf._wide(cnt, f"count_{tag}"))
        vsub = sub.filter(pl.col("code").is_in(vc) & pl.col("value_num").is_not_null())
        if vsub.height:
            blocks.append(bf._wide(_idx(vsub.group_by(KEYS).agg(
                pl.col("value_num").mean().alias("m")))["m"], f"val_mean_{tag}"))
            blocks.append(bf._wide(_idx(vsub.sort(KEYS + ["days"]).group_by(KEYS, maintain_order=True).agg(
                pl.col("value_num").first().alias("l")))["l"], f"val_latest_{tag}"))
    if not blocks:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    return pd.concat(blocks, axis=1)


# --------------------------------------------------------------------------- engine swap
_FAMILIES = ["occurrence_features", "flag_features", "age_features",
             "value_features", "band_features", "cumulative_features"]


def _swap():
    """Point build_features' six per-code families at the Polars versions; return the originals."""
    saved = {n: getattr(bf, n) for n in _FAMILIES}
    for n in _FAMILIES:
        setattr(bf, n, globals()[n])
    return saved


def _restore(saved):
    for n, f in saved.items():
        setattr(bf, n, f)


def build(h, sql_path=None, fit_split=True, years=None):
    """Identical to build_features.build, but the six per-code families run in Polars."""
    saved = _swap()
    try:
        return bf.build(h, sql_path=sql_path, fit_split=fit_split, years=years)
    finally:
        _restore(saved)


def main():
    saved = _swap()
    try:
        bf.main()
    finally:
        _restore(saved)


if __name__ == "__main__":
    main()
