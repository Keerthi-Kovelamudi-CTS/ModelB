"""
Codelist-driven FE - Lung Cancer model  (TIERED: concept-level + per-code)
====================================================================
TIERED feature engineering over the codelist (curated U data-driven):
  - the CURATED clinical-concept codes get concept-level FE (concept_features.py)
  - every OTHER (data-driven) code gets the generic PER-CODE feature vocabulary below
The two sets are DISJOINT — no code is featurized twice.

Reads the codelist + cohort SQL:
  - codelist : ./codelist/lung_codelist_{horizon}.csv        (curated U data-driven codes)
  - cohort   : ./SQL/{horizon}_1to1.sql                       (run on BigQuery -> events)

PER-CODE FEATURES (the generic per-code vocabulary, applied to the data-driven codes):
  OCCURRENCE (every code):
    _count _present _recency_months _decay_intensity _accel _recent_ratio
    _freq_per_year _timespan_years _interval_median _interval_min _interval_max _freq_trend_slope
    _first_half_freq _second_half_freq _is_worsening
  PROBLEM-LIST flags (every code; populated for observations):
    _has_active _has_significant      (from problem_status_description / significance_description)
  AGE at event (every code):
    _age_first _age_last _age_median
  VALUE / TREND (any per-code code carrying numeric values, e.g. labs/vitals — NOT only the >= MIN_VALUE_FRAC set):
    _val_first _val_latest _val_mean _val_median _val_min _val_max _val_std _val_range
    _val_abs_change _val_pct_change _val_latest_z _val_trend_slope _val_trend_corr
  PER-TIME-BAND (windowed; bands + anchoring from ANCHOR_MODE / TIME_BANDS_*):
    every code:   _count_w{lo}_{hi} _present_w{lo}_{hi}
    value codes:  _val_mean_w{lo}_{hi} _val_latest_w{lo}_{hi} _val_slope_w{lo}_{hi}
    Default ANCHOR_MODE="patient_last": windows count back from each patient's most-recent event
    (gap-agnostic, inference-safe), SAME bands for every horizon.
  CUMULATIVE last-N (windows from CUMULATIVE_WINDOWS):
    every code:   _count_last{n}      value codes: _val_mean_last{n} _val_latest_last{n}
  GLOBAL per-PATIENT (not per code): g_total_events g_distinct_codes g_distinct_obs_codes
    g_distinct_med_codes g_value_measured g_active_problems g_significant_problems
    g_consult_total g_consult_recency_months g_consult_accel g_consult_recent_rate

WHY value features only where numeric values exist: every per-code code that carries ANY numeric
value gets the full value vocabulary (rich per-code FE). A code with no numeric `value` (e.g. "Cough")
has nothing to summarize, so those columns are simply not emitted (not a real omission). Every
NON-value family is applied to EVERY code.

FILL RULES: count/present/decay/accel/recent_ratio/flags -> 0 when a patient never has the code
(genuine 0). recency/timespan/interval/value/age/trend -> NaN (not measurable; impute downstream,
never fake-0).

DIMENSIONALITY: ~28 features x thousands of codes = a VERY wide matrix (p>>n). Output is PARQUET
(a dense CSV this wide would not fit in memory). The model never trains on this raw matrix -
run `stability_select.py` next: it keeps only the codes/features that consistently drive
predictions across folds, and writes the small matrix the model actually trains on.

OUTPUT: ./output/{horizon}/features_p005_{horizon}.parquet   (one row per patient + cancer_class)

Run (VM w/ BigQuery, decent RAM):  python build_features.py        # both horizons

NOTE: statistical eligibility is NOT a leakage filter - curate pathway/post-suspicion codes out of
the codelist before using this for an early-prediction model.

Family switches (set any to False to drop that family): FEATURE_FAMILIES below.
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd
from google.cloud import bigquery

# Add this file's dir to sys.path so `import concept_features` works even when build_features is
# loaded via importlib (run_v3.py) rather than run from inside 2_FE/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import concept_features as cf   # tiered FE: concept-level FE on the curated codes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # V3 root, for `import config` / `import splits`
import config as C
import splits as _splits

# TIERED FE: curated codes get CONCEPT-level FE (concept_features); every OTHER codelist code
# gets the generic per-code families below. The two sets are disjoint — no code is featurized twice.
# Curated clinical-concept codes get concept-level FE; loaded only when USE_CURATED is on (else every
# code gets generic per-code FE and the concept tier is skipped — matches build_codelist's behaviour).
CURATED = (set(json.load(open(os.path.join(HERE, "..", "1_Top_Snomed", "curated_codes.json"))))
           if getattr(C, "USE_CURATED", True) else set())
HORIZONS = ["12mo", "1mo"]
if len(sys.argv) > 1:           # e.g. `python build_features.py 12mo` to build a single horizon
    HORIZONS = [a for a in sys.argv[1:] if a in ("12mo", "1mo")] or HORIZONS
# Leak-free FE: stability selection is fit on the SAME 80% TRAIN the model uses
# (replicating lung_training's seed-42 stratified 80/10/10), so the model's internal 10% test never
# informs the features. Clean 80/10/10 — no separate carved-out test set.

MIN_VALUE_FRAC = 0.30     # a code is "value-bearing" if >= this fraction of its events carry a number
DECAY_TAU_MONTHS = 12.0   # recency-weighted decay constant (matches concept_features)
RECENT_RATIO_CUTOFF = 24.0  # months (from data cutoff): "recent" band for recent_ratio
# Acceleration bins, measured from the DATA CUTOFF (= start of available data), so band 0 is the
# freshest events for BOTH horizons. (Previously hardcoded 12-18/18-30/30-54 = the 12mo gap, which
# left the 1mo horizon's fresh 1-12mo events in no bin.) 2nd-diff recent-2*mid+old > 0 => ramping up.
ACCEL_BINS = {"recent": (0.0, 6.0), "mid": (6.0, 18.0), "old": (18.0, 42.0)}
# Trend cap (OPTIONAL, default OFF). The lifetime trend + the per-band/cumulative windowed trends
# already let the model pick the relevant horizon, so we do NOT throw away long-term signal by
# default. Set to a number (e.g. 60) only if you want to also force the overall slope/intervals to
# a recent window. None = use full lifetime (recommended: don't lose signal).
TREND_MAX_MONTHS = None
# Cumulative "last-N months" windows (from start of available data) -> count + value mean/latest.
CUMULATIVE_WINDOWS = [6, 12, 24, 60]

# Per-time-band windows. Each band gets per-code count/present (+ value mean/latest for value
# codes), so the model can split on a specific window instead of only the collapsed accel/slope.
#
# Bands ALWAYS start at 0 and are the SAME for every horizon: the cohort SQL already applied the
# gap cutoff (12mo / 1mo), so the most-recent available event is the start of the data -> band 0.
# We do NOT re-offset by the gap (that would double-count it and leave band w0_6 empty).
TIME_BANDS = [(0, 6), (6, 18), (18, 36), (36, 72), (72, 999)]   # months from START OF AVAILABLE DATA
#   final (72, 999) is the open-ended catch-all so no event is dropped from the band view
#
# ANCHOR_MODE only decides what "month 0" is referenced to:
#   "patient_last"  -> each patient's OWN most-recent event (gap-agnostic, inference-safe).
#   "cohort_anchor" -> the cohort's data cutoff (global earliest days-before-anchor), diagnosis-aligned.
# Either way band 0 = start of available data. recency/decay/recent_ratio/accel stay anchor-relative
# (they measure staleness vs the prediction point, which is what we want and transfers to inference).
ANCHOR_MODE = "patient_last"

# Even 6-month bands for the GENERIC per-patient activity-rate trajectory (escalation signal):
# event/consult counts in 0-6, 6-12, 12-18 ... months before the prediction point. Rising counts
# toward month 0 = ramping presentation (prodromal). Patient-relative (from each patient's last event).
RATE_BANDS = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 36), (36, 60), (60, 999)]
LAB_Z_EXTREME = 2.0       # |latest value z vs patient's own baseline| >= this = an "extreme/deranged" lab

# Per-code problem-list value strings (CareRecord_Problem).
ACTIVE_STATUS_VALUE = "active problem"
SIGNIFICANT_VALUE = "significant problem"

FEATURE_FAMILIES = {
    "occurrence": True,   # count, present, recency, decay, accel, recent_ratio, freq, timespan, intervals, freq_trend, halves
    "flags":      True,   # has_active, has_significant
    "age":        True,   # age_first, age_last, age_median
    "value":      True,   # value/trend stats (value-bearing codes only)
    "bands":      True,   # disjoint per-time-band count/present (+ value mean/latest/slope), TIME_BANDS
    "cumulative": True,   # cumulative last-N count (+ value mean/latest), CUMULATIVE_WINDOWS
    "percentile": False,  # OFF: redundant w/ raw value for trees + only cross-patient-fit family
    "global":     True,   # per-PATIENT cross-code aggregates (total events, distinct codes, consults, burden)
    "comment":    True,   # per-PATIENT free-text problem-list comment keyword/presence features
    "derangement": True,  # GENERIC cross-code burden: # extreme labs / mean|z| / # rising / # worsening (no hardcoded codes)
    "blood_ratios": True, # NLR / PLR / LMR / CRP-albumin (mGPS) + trend slopes (the only lightly-hardcoded block)
}

# The ONLY hardcoded codes — the analytes for the 4 clinical blood ratios (all confirmed present
# in the codelist). Everything else in FE is generic per-code.
NEUTROPHIL_CODE = [1022551000000104]
LYMPHOCYTE_CODE = [1022581000000105]
PLATELET_CODE   = [1022651000000100]
MONOCYTE_CODE   = [1022591000000107]
CRP_CODES       = [1001371000000100, 999651000000107]
ALBUMIN_CODES   = [1000821000000103]

# Free-text problem-list COMMENT keywords. LEAKAGE-SAFE: prodromal lung symptoms + risk factors ONLY
# (NO cancer/malignant/tumour/mass/nodule/lesion/2WW/referral/"suspected" — those leak the outcome).
COMMENT_KEYWORDS = {
    "haemoptysis":     ["haemopt", "hemopt", "coughing up blood", "blood in sputum", "bloody sputum"],
    "cough":           ["cough"],
    "dyspnoea":        ["dyspnoea", "dyspnea", "breathless", "shortness of breath", "short of breath", " sob"],
    "chest_pain":      ["chest pain"],
    "weight_loss":     ["weight loss", "wt loss", "losing weight", "lost weight"],
    "fatigue":         ["fatigue", "lethargy", "malaise"],
    "hoarseness":      ["hoarse", "voice change"],
    "chest_infection": ["chest infection", "pneumonia", "lrti"],
    "appetite":        ["poor appetite", "loss of appetite", "anorexia"],
    "night_sweats":    ["night sweat"],
    "clubbing":        ["clubbing"],
    "smoking":         ["smok", "cigarette", "tobacco", "pack year"],
}


def _codelist_path(h, years=None):
    """Per-(horizon, lookback) codelist — each FE lookback uses the codelist discovered on the SAME
    lookback's train data (build_codelist writes lung_codelist_{h}_{years}yr.csv)."""
    yrs = int(years if years is not None else getattr(C, "FE_YEARS_BEFORE", 5))
    return os.path.join(HERE, "codelist", f"lung_codelist_{h}_{yrs}yr.csv")


def _cohort_sql_path(h):
    return os.path.join(HERE, "..", "0_SQL", f"{h}_1to1.sql")   # shared single-source SQL


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_events(h, sql_path=None, years=None):
    """Pull the {horizon}_1to1 cohort events from BigQuery (only the columns FE needs).
    `sql_path` overrides the default 0_SQL cohort (e.g. the held-out SQL in 4_Heldout).
    `years` sets the FE lookback window (default config.FE_YEARS_BEFORE; e.g. 5/10/20/100=lifetime)."""
    yrs = int(years if years is not None else getattr(C, "FE_YEARS_BEFORE", 5))
    sql = open(sql_path or _cohort_sql_path(h), encoding="utf-8").read().rstrip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    sql = re.sub(r"ORDER\s+BY\s+patient_guid\s*$", "", sql, flags=re.I).rstrip()
    sql = re.sub(r"\d+(\s+AS years_before\b)", rf"{yrs}\1", sql, count=1)   # FE lookback window (years)
    query = f"""
        SELECT patient_guid, cancer_class, event_type,
               CAST(snomed_c_t_concept_id AS STRING) AS snomed_c_t_concept_id,
               CAST(med_code_id AS STRING) AS med_code_id, value, days_before_anchor,
               event_age, age_at_anchor, sex, patient_ethnicity_6,
               problem_status_description, significance_description, problem_comment
        FROM (
{sql}
        )
    """
    print(f"[{h}] pulling cohort events from BigQuery ...")
    ev = bigquery.Client().query(query).to_dataframe()
    ev["cancer_class"] = ev["cancer_class"].astype(int)
    # Build the unified code as nullable Int64 from STRING (NOT via np.where on numerics, which
    # upcasts to float64 and rounds 18-digit DMD med codes > 2**53 -> they'd silently fail to match
    # the codelist; ~90 COPD-inhaler codes were being dropped). String -> to_numeric stays exact.
    _sn = pd.to_numeric(ev["snomed_c_t_concept_id"], errors="coerce").astype("Int64")
    _md = pd.to_numeric(ev["med_code_id"], errors="coerce").astype("Int64")
    ev["code"] = _sn.where(ev["event_type"].eq("observation"), _md)
    ev["days"] = pd.to_numeric(ev["days_before_anchor"], errors="coerce")
    ev["months"] = ev["days"] / 30.44
    ev["value_num"] = pd.to_numeric(ev["value"], errors="coerce")
    ev["event_age"] = pd.to_numeric(ev["event_age"], errors="coerce")
    ev["active"] = (ev["problem_status_description"].astype(str).str.strip().str.lower()
                    == ACTIVE_STATUS_VALUE).astype(int)
    ev["sig"] = (ev["significance_description"].astype(str).str.strip().str.lower()
                 == SIGNIFICANT_VALUE).astype(int)
    ev = ev.dropna(subset=["code"])
    ev["code"] = ev["code"].astype("int64")
    print(f"  {len(ev):,} events | {ev.patient_guid.nunique():,} patients")
    return ev


# ---------------------------------------------------------------------------
# Helpers (all vectorized over the (patient_guid, code) grouping)
# ---------------------------------------------------------------------------
def _wide(series, suffix):
    """(patient_guid, code)-indexed Series -> wide df: columns '<code>_<suffix>'. Empty-safe:
    a group helper that finds no qualifying rows returns an empty/unnamed Series -> emit no columns."""
    names = list(series.index.names) if series.index.nlevels > 1 else []
    if series.empty or "code" not in names:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    w = series.unstack("code")
    w.columns = [f"{int(c)}_{suffix}" for c in w.columns]
    return w


def _group_linreg(df, xcol, ycol, keys=("patient_guid", "code")):
    """Vectorized per-group OLS slope + Pearson r via summed moments (no per-group apply)."""
    d = df[list(keys) + [xcol, ycol]].copy()
    d["_xx"] = d[xcol] * d[xcol]
    d["_xy"] = d[xcol] * d[ycol]
    d["_yy"] = d[ycol] * d[ycol]
    a = d.groupby(list(keys)).agg(n=(xcol, "size"), Sx=(xcol, "sum"), Sy=(ycol, "sum"),
                                  Sxx=("_xx", "sum"), Sxy=("_xy", "sum"), Syy=("_yy", "sum"))
    den = (a["n"] * a["Sxx"] - a["Sx"] ** 2).replace(0, np.nan)
    num = a["n"] * a["Sxy"] - a["Sx"] * a["Sy"]
    slope = num / den
    corr = num / np.sqrt(den * (a["n"] * a["Syy"] - a["Sy"] ** 2))
    corr = corr.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)   # a correlation must be in [-1,1];
    return slope, corr                                               # near-constant y can make the sqrt underflow -> inf


def occurrence_features(ev):
    """All occurrence/temporal-dynamics features, per (patient, code).
    Recency/decay/accel/recent_ratio are measured from the DATA CUTOFF (start of available data)
    so they start at 0 and are horizon-consistent. Trends/intervals/halves use only the most-recent
    TREND_MAX_MONTHS. Count/present/timespan/freq are lifetime."""
    keys = ["patient_guid", "code"]
    ev = ev.copy()
    ev["_moc"] = ev["months"] - ev["months"].min()       # months since data cutoff (starts at 0)
    g = ev.groupby(keys)
    count = g.size()
    recency = g["_moc"].min()
    timespan_years = (g["days"].max() - g["days"].min()) / 365.25
    freq_per_year = count / timespan_years.replace(0, np.nan)

    ev["_w"] = np.exp(-ev["_moc"] / DECAY_TAU_MONTHS)
    ev["_recent"] = (ev["_moc"] <= RECENT_RATIO_CUTOFF).astype(float)
    decay = ev.groupby(keys)["_w"].sum()
    recent_ratio = ev.groupby(keys)["_recent"].sum() / count

    def _binct(lo, hi):
        return (ev[(ev["_moc"] >= lo) & (ev["_moc"] < hi)]
                .groupby(keys).size().reindex(count.index, fill_value=0))
    c_r = _binct(*ACCEL_BINS["recent"])
    c_m = _binct(*ACCEL_BINS["mid"])
    c_o = _binct(*ACCEL_BINS["old"])
    accel = c_r - 2 * c_m + c_o

    # --- trend-capped frame: only the most-recent TREND_MAX_MONTHS of available data ---
    tev = ev if TREND_MAX_MONTHS is None else ev[ev["_moc"] <= TREND_MAX_MONTHS]
    # intervals between consecutive events of the same code (sorted recent->old)
    s = tev[keys + ["days"]].sort_values(keys + ["days"])
    s["_iv"] = s.groupby(keys)["days"].diff().abs()
    iv = s.dropna(subset=["_iv"])
    gi = iv.groupby(keys)["_iv"]
    interval_median, interval_min, interval_max = gi.median(), gi.min(), gi.max()
    # frequency trend: slope of interval vs its order index (shrinking gaps => worsening)
    iv = iv.assign(_ord=iv.groupby(keys).cumcount())
    freq_trend_slope, _ = _group_linreg(iv, "_ord", "_iv")
    # first-half vs second-half frequency + is_worsening (within the trend window)
    first_half_freq, second_half_freq, is_worsening = _halves(tev, keys)

    present = (count > 0).astype(int)
    parts = {
        "count": count, "present": present, "recency_months": recency,
        "decay_intensity": decay, "accel": accel, "recent_ratio": recent_ratio,
        "freq_per_year": freq_per_year, "timespan_years": timespan_years,
        "interval_median": interval_median, "interval_min": interval_min,
        "interval_max": interval_max, "freq_trend_slope": freq_trend_slope,
        "first_half_freq": first_half_freq, "second_half_freq": second_half_freq,
        "is_worsening": is_worsening,
    }
    return pd.concat([_wide(s_, sfx) for sfx, s_ in parts.items()], axis=1)


def _halves(ev, keys):
    """Per (patient, code): split events chronologically into older/newer halves, return
    first-half freq, second-half freq, and is_worsening (newer freq > 1.1x older). Vectorized."""
    s = ev[keys + ["days"]].sort_values(keys + ["days"], ascending=[True, True, False])  # oldest first
    s["_n"] = s.groupby(keys)["days"].transform("size")
    s["_r"] = s.groupby(keys).cumcount()
    s = s[s["_n"] >= 2]
    if s.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    s["_half"] = np.where(s["_r"] < s["_n"] / 2.0, "h1", "h2")    # h1 = older, h2 = newer
    gh = s.groupby(keys + ["_half"])
    cnt = gh.size()
    span = (gh["days"].max() - gh["days"].min()) / 365.25
    freq = (cnt / span.replace(0, np.nan)).unstack("_half")
    fh = freq["h1"] if "h1" in freq.columns else pd.Series(np.nan, index=freq.index)
    sh = freq["h2"] if "h2" in freq.columns else pd.Series(np.nan, index=freq.index)
    is_wors = (sh > 1.1 * fh).astype(float)
    return fh, sh, is_wors


def flag_features(ev):
    """Per-code problem-list flags: ever an active / significant problem."""
    g = ev.groupby(["patient_guid", "code"])
    return pd.concat([_wide(g["active"].max(), "has_active"),
                      _wide(g["sig"].max(), "has_significant")], axis=1)


def age_features(ev):
    """Per-code patient age at first / last occurrence + median."""
    keys = ["patient_guid", "code"]
    s = ev[keys + ["days", "event_age"]].sort_values(keys + ["days"])   # asc days = recent first
    g = s.groupby(keys)["event_age"]
    age_last = g.first()     # most recent (smallest days)
    age_first = g.last()     # earliest (largest days)
    age_median = ev.groupby(keys)["event_age"].median()
    return pd.concat([_wide(age_first, "age_first"), _wide(age_last, "age_last"),
                      _wide(age_median, "age_median")], axis=1)


def value_bearing_codes(ev):
    """Codes where >= MIN_VALUE_FRAC of events carry a numeric value (labs/vitals)."""
    val = ev.dropna(subset=["value_num"])
    if val.empty:
        return set()
    frac = val.groupby("code").size() / ev.groupby("code").size()
    return set(frac[frac >= MIN_VALUE_FRAC].index)


def value_features(ev, value_codes):
    """Value/trend stats per (patient, code) for value-bearing codes only."""
    keys = ["patient_guid", "code"]
    val = ev[ev["code"].isin(value_codes)].dropna(subset=["value_num"]).copy()
    if val.empty:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))

    g = val.groupby(keys)["value_num"]
    mean, median, vmin, vmax, vstd = g.mean(), g.median(), g.min(), g.max(), g.std()
    stats = {"val_mean": mean, "val_median": median, "val_min": vmin, "val_max": vmax,
             "val_std": vstd, "val_range": vmax - vmin}
    vs = val.sort_values(keys + ["days"])                                # asc days = recent first
    gv = vs.groupby(keys)["value_num"]
    latest, first = gv.first(), gv.last()                                # latest=recent, first=earliest
    stats["val_first"], stats["val_latest"] = first, latest
    stats["val_abs_change"] = latest - first
    stats["val_pct_change"] = (latest - first) / first.replace(0, np.nan) * 100.0
    stats["val_latest_z"] = (latest - mean) / vstd.replace(0, np.nan)    # latest vs patient's own baseline
    # trend over time, capped to the most-recent TREND_MAX_MONTHS: x = -days (later => larger x)
    cutoff = ev["months"].min()
    tval = val if TREND_MAX_MONTHS is None else val[(val["months"] - cutoff) <= TREND_MAX_MONTHS]
    tval = tval.assign(_x=-tval["days"])
    slope, corr = _group_linreg(tval, "_x", "value_num")
    stats["val_trend_slope"], stats["val_trend_corr"] = slope, corr
    # value acceleration: slope(recent half) - slope(older half) (>=4 measurements). +ve = steepening
    vsrt = val.sort_values(keys + ["days"], ascending=[True, True, False])    # oldest first
    vsrt["_n"] = vsrt.groupby(keys)["days"].transform("size")
    vsrt["_r"] = vsrt.groupby(keys).cumcount()
    h = vsrt[vsrt["_n"] >= 4].copy()
    if not h.empty:
        h["_half"] = np.where(h["_r"] < h["_n"] / 2.0, "old", "new")
        h["_x"] = -h["days"]
        s_old, _ = _group_linreg(h[h["_half"] == "old"], "_x", "value_num")
        s_new, _ = _group_linreg(h[h["_half"] == "new"], "_x", "value_num")
        stats["val_accel"] = s_new - s_old
    return pd.concat([_wide(s_, sfx) for sfx, s_ in stats.items()], axis=1)


def band_features(ev, bands, value_codes, relative):
    """Per-time-band occurrence (every code) + value mean/latest + value trend slope (value codes).
    `bands` = list of [lo, hi) month windows. `relative=True` -> windows count back from each
    patient's most-recent event (gap-agnostic, inference-safe); False -> fixed months-before-anchor.
    Lets the model split on a specific window rather than only the collapsed accel / overall slope."""
    keys = ["patient_guid", "code"]
    ev = ev.copy()
    if relative:                                                 # band 0 = each patient's last event
        ev["_mo"] = ev["months"] - ev.groupby("patient_guid")["months"].transform("min")
    else:                                                        # band 0 = cohort data cutoff (gap)
        ev["_mo"] = ev["months"] - ev["months"].min()
    is_val = ev["code"].isin(value_codes)
    blocks = []
    for lo, hi in bands:
        m = (ev["_mo"] >= lo) & (ev["_mo"] < hi)
        tag = f"w{int(lo)}_{int(hi)}"
        cnt = ev[m].groupby(keys).size()
        blocks.append(_wide(cnt, f"count_{tag}"))
        blocks.append(_wide((cnt > 0).astype(int), f"present_{tag}"))
        vsub = ev[m & is_val].dropna(subset=["value_num"])
        if not vsub.empty:
            blocks.append(_wide(vsub.groupby(keys)["value_num"].mean(), f"val_mean_{tag}"))
            vs = vsub.sort_values(keys + ["days"])                        # asc days = recent first
            blocks.append(_wide(vs.groupby(keys)["value_num"].first(), f"val_latest_{tag}"))
            vsub = vsub.assign(_x=-vsub["days"])                          # within-band value trend
            slope, _ = _group_linreg(vsub, "_x", "value_num")
            blocks.append(_wide(slope, f"val_slope_{tag}"))
    if not blocks:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    return pd.concat(blocks, axis=1)


def percentile_features(ev, value_codes, train_guids):
    """Per value-code: latest value's percentile vs the COHORT and vs the patient's AGE-BAND.
    Scale-free 'where does this patient sit in the population' signal. LEAKAGE-SAFE: the reference
    distribution is built from TRAIN patients only; every patient is then ranked against that
    train reference (so test patients never inform their own percentiles)."""
    keys = ["patient_guid", "code"]
    val = ev[ev["code"].isin(value_codes)].dropna(subset=["value_num"])
    if val.empty:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    vs = val.sort_values(keys + ["days"])
    latest = vs.groupby(keys)["value_num"].first().reset_index()         # patient, code, latest value
    age = ev.groupby("patient_guid")["event_age"].max()
    latest["band"] = latest["patient_guid"].map(age // 10 * 10)
    latest["_train"] = latest["patient_guid"].isin(train_guids)

    def _fill_pct(col, group_keys):
        """Rank each patient's value within the TRAIN sub-distribution of its group (searchsorted on
        the sorted train values); NaN where the group has no train reference (imputed downstream)."""
        latest[col] = np.nan
        for _, idx in latest.groupby(group_keys).groups.items():
            sub = latest.loc[idx]
            ref = np.sort(sub.loc[sub["_train"], "value_num"].to_numpy())
            if ref.size:
                latest.loc[idx, col] = np.searchsorted(ref, sub["value_num"].to_numpy(), side="right") / ref.size

    _fill_pct("pctile", "code")
    _fill_pct("ageband_pctile", ["code", "band"])
    out = []
    for col, sfx in [("pctile", "val_pctile"), ("ageband_pctile", "val_ageband_pctile")]:
        w = latest.pivot(index="patient_guid", columns="code", values=col)
        w.columns = [f"{int(c)}_{sfx}" for c in w.columns]
        out.append(w)
    return pd.concat(out, axis=1)


# train_guids() removed — the CANONICAL split (splits.load_or_make / make_split.py) is now the single
# source of train/valid/test; no per-module train_test_split re-derivation. build() reads train guids
# via _splits.guids_for(...); stability_select reads the same canonical split.


def cumulative_features(ev, windows, value_codes, relative):
    """Cumulative 'last-N months' windows (from start of available data): count (+ value mean/latest
    for value codes). Overlapping windows let the model read accumulation over time, alongside the
    disjoint bands."""
    keys = ["patient_guid", "code"]
    ev = ev.copy()
    if relative:
        ev["_mo"] = ev["months"] - ev.groupby("patient_guid")["months"].transform("min")
    else:
        ev["_mo"] = ev["months"] - ev["months"].min()
    is_val = ev["code"].isin(value_codes)
    blocks = []
    for n in windows:
        m = ev["_mo"] < n
        tag = f"last{int(n)}"
        cnt = ev[m].groupby(keys).size()
        blocks.append(_wide(cnt, f"count_{tag}"))
        vsub = ev[m & is_val].dropna(subset=["value_num"])
        if not vsub.empty:
            blocks.append(_wide(vsub.groupby(keys)["value_num"].mean(), f"val_mean_{tag}"))
            vs = vsub.sort_values(keys + ["days"])
            blocks.append(_wide(vs.groupby(keys)["value_num"].first(), f"val_latest_{tag}"))
    if not blocks:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    return pd.concat(blocks, axis=1)


def global_features(ev):
    """Per-PATIENT cross-code aggregates (not per code): event/code volume, consultation cadence,
    problem-list burden. Counts/rates -> 0 when absent; recency -> NaN.
    Fed the FULL event stream (every code, not just the codelist) so the volume counts reflect
    the patient's TRUE healthcare utilisation, not just their significant-code events."""
    ev = ev.copy()
    ev["_moc"] = ev["months"] - ev["months"].min()
    g = ev.groupby("patient_guid")
    out = pd.DataFrame(index=pd.Index(ev["patient_guid"].unique(), name="patient_guid"))
    out["g_total_events"] = g.size()
    out["g_distinct_codes"] = g["code"].nunique()
    out["g_distinct_obs_codes"] = ev[ev["event_type"].eq("observation")].groupby("patient_guid")["code"].nunique()
    out["g_distinct_med_codes"] = ev[ev["event_type"].eq("medication")].groupby("patient_guid")["code"].nunique()
    out["g_value_measured"] = ev.dropna(subset=["value_num"]).groupby("patient_guid").size()
    out["g_active_problems"] = g["active"].sum()
    out["g_significant_problems"] = g["sig"].sum()
    enc = ev.drop_duplicates(["patient_guid", "days"])                   # one encounter per patient-day
    ge = enc.groupby("patient_guid")
    out["g_consult_total"] = ge.size()
    out["g_consult_recency_months"] = ge["_moc"].min()

    def _b(lo, hi):
        return (enc[(enc["_moc"] >= lo) & (enc["_moc"] < hi)]
                .groupby("patient_guid").size().reindex(out.index, fill_value=0))
    out["g_consult_accel"] = _b(*ACCEL_BINS["recent"]) - 2 * _b(*ACCEL_BINS["mid"]) + _b(*ACCEL_BINS["old"])
    recent = (enc[enc["_moc"] <= RECENT_RATIO_CUTOFF]
              .groupby("patient_guid").size().reindex(out.index, fill_value=0))
    out["g_consult_recent_rate"] = recent / out["g_consult_total"]
    zero = ["g_total_events", "g_distinct_codes", "g_distinct_obs_codes", "g_distinct_med_codes",
            "g_value_measured", "g_active_problems", "g_significant_problems", "g_consult_total",
            "g_consult_accel", "g_consult_recent_rate"]
    out[zero] = out[zero].fillna(0)

    # --- patient demographics: age at the prediction point, sex, ethnicity ---
    # Kept numeric so they flow through stability-selection: age/sex -> NaN where unknown (imputed);
    # ethnicity one-hot -> 0/1 (a missing ethnicity = all-zero, correct). age is a top lung-ca risk factor.
    if "age_at_anchor" in ev.columns:
        out["g_age"] = pd.to_numeric(g["age_at_anchor"].first(), errors="coerce")   # age at anchor_date
    if "sex" in ev.columns:
        sx = g["sex"].first().astype(str).str.strip().str.lower()
        out["g_is_male"] = pd.Series(np.where(sx.str.startswith("m"), 1.0,
                                     np.where(sx.str.startswith("f"), 0.0, np.nan)), index=sx.index)
    if "patient_ethnicity_6" in ev.columns:
        eth = g["patient_ethnicity_6"].first().astype(str).str.strip()
        eth = eth.where(~eth.str.lower().isin(["nan", "none", "", "null"]))          # NaN -> all-zero dummies
        dummies = pd.get_dummies(eth, prefix="g_eth").astype(float)
        out = out.join(dummies)                                                      # 0/1, one col per ethnicity
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def comment_features(ev):
    """Per-PATIENT free-text problem-list comment features (`problem_comment`): presence, volume,
    prodromal-keyword counts (COMMENT_KEYWORDS — leak-safe symptoms/risk only), distinct-symptom-group
    burden, and red-flag recency. Counts/flags -> 0 when absent; recency stays NaN (imputed).
    Fed the FULL event stream (not just the codelist) so comment burden is the patient's true
    free-text history, not only comments attached to significant codes."""
    idx = pd.Index(ev["patient_guid"].unique(), name="patient_guid")
    out = pd.DataFrame(index=idx)
    if "problem_comment" not in ev.columns:
        return out
    w = ev[ev["event_type"].eq("observation")].copy()
    w["_c"] = w["problem_comment"].astype(str).str.strip().str.lower()
    w = w[~w["_c"].isin(["", "nan", "none", "null"])]
    if w.empty:
        return out
    out["g_comment_count"] = w.groupby("patient_guid").size()
    out["g_has_comment"] = (out["g_comment_count"] > 0).astype(float)
    grp_present = pd.DataFrame(index=out.index)
    redflag = pd.Series(False, index=w.index)
    for grp, terms in COMMENT_KEYWORDS.items():
        pat = "|".join(re.escape(t) for t in terms)
        m = w["_c"].str.contains(pat, regex=True, na=False)
        redflag = redflag | m
        cnt = w[m].groupby("patient_guid").size()
        out[f"g_comment_kw_{grp}"] = cnt
        grp_present[grp] = (cnt > 0)
    out["g_comment_symptom_groups"] = grp_present.reindex(out.index).fillna(False).astype(int).sum(axis=1)
    rf = w[redflag]
    if not rf.empty:
        out["g_comment_redflag_recency_months"] = rf.groupby("patient_guid")["months"].min()
    zcols = [c for c in out.columns if not c.endswith("_months")]
    out[zcols] = out[zcols].fillna(0)
    return out


def derangement_features(ev, ev_full, value_codes):
    """GENERIC cross-code burden / escalation (NO hardcoded codes) — generalizes the SPIRIT of the
    curated composites (NLR/clusters/mGPS) as per-patient aggregates over the per-code data:
      * activity-rate trajectory: event + consult counts in even 6-mo bands (RATE_BANDS) -> ramping?
      * lab derangement: # value-codes whose LATEST is extreme vs the patient's OWN baseline + mean|z|
      * # value-codes rising (positive within-patient value slope)
      * # codes worsening (rising event freq) / accelerating (2nd-diff > 0)
      * recent distinct-code burden (last 6 / 12 months, full stream)
    Counts -> 0 when absent; mean|z| -> NaN (imputed)."""
    keys = ["patient_guid", "code"]
    out = pd.DataFrame(index=pd.Index(ev_full["patient_guid"].unique(), name="patient_guid"))

    # activity-rate trajectory (even 6-mo bands, patient-relative; full stream)
    ef = ev_full.copy()
    ef["_mo"] = ef["months"] - ef.groupby("patient_guid")["months"].transform("min")   # 0 = patient's last event
    enc = ef.drop_duplicates(["patient_guid", "days"])                                  # one consult per patient-day
    for lo, hi in RATE_BANDS:
        tag = f"w{int(lo)}_{int(hi)}"
        out[f"g_events_{tag}"]   = ef[(ef["_mo"] >= lo) & (ef["_mo"] < hi)].groupby("patient_guid").size()
        out[f"g_consults_{tag}"] = enc[(enc["_mo"] >= lo) & (enc["_mo"] < hi)].groupby("patient_guid").size()

    # lab derangement (within-patient z of latest value; value codes only)
    val = ev[ev["code"].isin(value_codes)].dropna(subset=["value_num"])
    if not val.empty:
        g = val.groupby(keys)["value_num"]
        mean, std = g.mean(), g.std()
        latest = val.sort_values(keys + ["days"]).groupby(keys)["value_num"].first()
        za = ((latest - mean) / std.replace(0, np.nan)).abs()
        out["g_n_labs_extreme"] = (za >= LAB_Z_EXTREME).groupby("patient_guid").sum()
        out["g_mean_abs_lab_z"] = za.groupby("patient_guid").mean()
        slope, _ = _group_linreg(val.assign(_x=-val["days"]), "_x", "value_num")
        out["g_n_labs_rising"] = (slope > 0).groupby("patient_guid").sum()

    # code escalation (frequency worsening / accelerating; codelist codes)
    e = ev.copy(); e["_moc"] = e["months"] - e["months"].min()
    cnt = e.groupby(keys).size()
    def _binct(lo, hi):
        return e[(e["_moc"] >= lo) & (e["_moc"] < hi)].groupby(keys).size().reindex(cnt.index, fill_value=0)
    accel = _binct(*ACCEL_BINS["recent"]) - 2 * _binct(*ACCEL_BINS["mid"]) + _binct(*ACCEL_BINS["old"])
    out["g_n_codes_accelerating"] = (accel > 0).groupby("patient_guid").sum()
    _, _, is_w = _halves(e, keys)
    if not is_w.empty:
        out["g_n_codes_worsening"] = (is_w > 0).groupby("patient_guid").sum()

    # recent distinct-code burden (full stream)
    for w in (6, 12):
        out[f"g_distinct_codes_last{w}"] = ef[ef["_mo"] < w].groupby("patient_guid")["code"].nunique()

    zc = [c for c in out.columns if c != "g_mean_abs_lab_z"]      # counts -> 0; mean|z| stays NaN
    out[zc] = out[zc].fillna(0)
    return out


def blood_ratio_features(ev):
    """The ONLY lightly-hardcoded block: NLR / PLR / LMR / CRP-albumin (mGPS) from each analyte's LATEST
    value + ratio TREND slopes. Per-patient, inference-safe. Analyte codes are all present in the codelist;
    a tree could partly learn these from the separate analyte value features, but the explicit ratio is cleaner."""
    obs = ev[ev["event_type"].eq("observation")].dropna(subset=["value_num"])
    out = pd.DataFrame(index=pd.Index(ev["patient_guid"].unique(), name="patient_guid"))
    if obs.empty:
        return out
    def last_val(codes):
        s = obs[obs["code"].isin(codes)].sort_values(["patient_guid", "days"])
        return s.groupby("patient_guid")["value_num"].first()        # smallest days = most recent
    neut, lymph, plat = last_val(NEUTROPHIL_CODE), last_val(LYMPHOCYTE_CODE), last_val(PLATELET_CODE)
    mono, crp, alb = last_val(MONOCYTE_CODE), last_val(CRP_CODES), last_val(ALBUMIN_CODES)
    ls = lymph.replace(0, np.nan)
    out["NLR"] = neut / ls
    out["PLR"] = plat / ls
    out["LMR"] = lymph / mono.replace(0, np.nan)
    out["CRP_ALBUMIN_RATIO"] = crp / alb.replace(0, np.nan)
    def ratio_trend(num_codes, den_codes):
        num = obs[obs["code"].isin(num_codes)].groupby(["patient_guid", "days"])["value_num"].mean().rename("num")
        den = obs[obs["code"].isin(den_codes)].groupby(["patient_guid", "days"])["value_num"].mean().rename("den")
        m = pd.concat([num, den], axis=1).dropna(); m = m[m["den"] != 0]
        if m.empty:
            return pd.Series(dtype=float)
        m = m.reset_index(); m["_r"] = m["num"] / m["den"]; m["_x"] = -m["days"]
        slope, _ = _group_linreg(m, "_x", "_r", keys=("patient_guid",))
        return slope
    out["NLR_TREND_SLOPE"] = ratio_trend(NEUTROPHIL_CODE, LYMPHOCYTE_CODE)
    out["PLR_TREND_SLOPE"] = ratio_trend(PLATELET_CODE, LYMPHOCYTE_CODE)
    out["LMR_TREND_SLOPE"] = ratio_trend(LYMPHOCYTE_CODE, MONOCYTE_CODE)
    return out


def build(h, sql_path=None, fit_split=True, years=None):
    """Build the per-code feature matrix for horizon `h`.
    `sql_path`  : override the default 0_SQL cohort (e.g. 4_Heldout's held-out SQL).
    `fit_split` : True for the training cohort (compute the 80% train guids that stability-selection
                  reuses); False for held-out (touch-once eval — no internal split, nothing fit here).
    `years`     : FE lookback window in years (default config.FE_YEARS_BEFORE)."""
    codes = pd.read_csv(_codelist_path(h, years))
    # Codelist is in curation format (Code, Name, Value). FE only needs the numeric Code column;
    # Name/Value are for human review and are ignored here.
    code_col = "Code" if "Code" in codes.columns else "code_id"   # back-compat with the old code_id format
    keep = set(pd.to_numeric(codes[code_col], errors="coerce").dropna().astype("int64"))
    print(f"[{h}] codelist: {len(keep):,} codes (curated U data-driven)")

    ev_full = load_events(h, sql_path, years)   # full event stream (every code) — true patient volume
    ev = ev_full[ev_full["code"].isin(keep)]    # codelist-restricted

    # --- TIERED partition: curated codes -> CONCEPT FE; remaining codes -> generic PER-CODE FE ---
    curated_in = keep & CURATED                 # curated codes that are actually in this codelist
    remaining = keep - CURATED                  # everything else -> per-code families
    assert not (curated_in & remaining), "tiered FE: curated_in and remaining must be disjoint"
    print(f"[{h}] tiered FE: {len(curated_in)} curated (concept FE) + {len(remaining)} remaining (per-code FE)")
    ev_remaining = ev[ev["code"].isin(remaining)]    # per-code families run on the NON-curated codes
    ev_cur = ev[ev["code"].isin(curated_in)]         # concept families run on the curated codes

    # B: keep EVERY cohort patient in ALL phases (training + held-out). No-hit patients still carry
    # global/demographic/comment/derangement signal; uniform across phases + deployment realism.
    patients = (ev_full[["patient_guid", "cancer_class"]].drop_duplicates("patient_guid")
                  .set_index("patient_guid"))
    # TRAIN guids from the CANONICAL saved split (split-first) — the SAME patients as scoring + model.
    # Held-out (fit_split=False) does no split. (guids_for returns CLEANED guids.)
    tr_guids = (_splits.guids_for(
        _splits.load_or_make(h, patients.index.to_numpy(), patients["cancer_class"].to_numpy()), "train")
        if fit_split else set())

    # Which codes emit value features is determined on the TRAIN patients only (strict split-first;
    # clean-match both sides) so the internal test never influences the feature schema. Held-out uses all.
    if fit_split and tr_guids:
        ev_vb = ev_remaining[_splits.clean_guid(ev_remaining["patient_guid"]).isin(tr_guids)]
    else:
        ev_vb = ev_remaining
    # RICH per-code value FE: emit value/trend features for EVERY per-code code that carries ANY numeric
    # value, not just the strict >= MIN_VALUE_FRAC "value-bearing" set. Codes with no numeric value
    # contribute nothing (value_features/band/cumulative dropna value_num) -> no all-NaN dead columns.
    value_codes = set(ev_vb.dropna(subset=["value_num"])["code"].unique())
    vb_codes = value_bearing_codes(ev_vb)        # strict (>= MIN_VALUE_FRAC) set — used by the lab-derangement family
    print(f"[{h}] per-code value FE on {len(value_codes):,} codes with any numeric value"
          f"{' (TRAIN-only)' if (fit_split and tr_guids) else ''}; "
          f"{len(vb_codes):,} strict value-bearing (>= {int(MIN_VALUE_FRAC*100)}%) for derangement")
    if fit_split:
        print(f"[{h}] split-first: stability-selection fit on {len(tr_guids):,}/{len(patients):,} train patients")
    blocks = []
    # --- generic PER-CODE families: run on ev_remaining (curated codes excluded -> no duplication) ---
    if FEATURE_FAMILIES["occurrence"]:
        print(f"[{h}]   occurrence/dynamics ...");  blocks.append(occurrence_features(ev_remaining))
    if FEATURE_FAMILIES["flags"]:
        print(f"[{h}]   problem-list flags ...");   blocks.append(flag_features(ev_remaining))
    if FEATURE_FAMILIES["age"]:
        print(f"[{h}]   age-at-event ...");         blocks.append(age_features(ev_remaining))
    if FEATURE_FAMILIES["value"]:
        print(f"[{h}]   value/trend ...");          blocks.append(value_features(ev_remaining, value_codes))
    if FEATURE_FAMILIES["bands"]:
        relative = ANCHOR_MODE == "patient_last"
        print(f"[{h}]   per-time-band ({ANCHOR_MODE}) {TIME_BANDS} ...")
        blocks.append(band_features(ev_remaining, TIME_BANDS, value_codes, relative))
    if FEATURE_FAMILIES["cumulative"]:
        relative = ANCHOR_MODE == "patient_last"
        print(f"[{h}]   cumulative last-N {CUMULATIVE_WINDOWS} ...")
        blocks.append(cumulative_features(ev_remaining, CUMULATIVE_WINDOWS, value_codes, relative))
    if FEATURE_FAMILIES["percentile"]:
        print(f"[{h}]   value percentile (cohort + age-band) ...")
        blocks.append(percentile_features(ev_remaining, value_codes, tr_guids))

    # --- CONCEPT-level families: run on ev_cur (the curated codes) — skipped when USE_CURATED is off ---
    if curated_in:
        print(f"[{h}]   concept: symptom dynamics ...");  blocks.append(cf.compute_symptom_dynamics(ev_cur))
        print(f"[{h}]   concept: problem-list flags ..."); blocks.append(cf.compute_problem_flags(ev_cur))
        print(f"[{h}]   concept: clusters ...");           blocks.append(cf.compute_clusters(ev_cur))
        print(f"[{h}]   concept: interactions ...");       blocks.append(cf.interaction_features(ev_cur))
        print(f"[{h}]   concept: smoking dose ...");       blocks.append(cf.compute_smoking_dose(ev_cur))
        print(f"[{h}]   concept: lab level-stats ...");    blocks.append(cf.compute_lab_stats(ev_cur))

    if FEATURE_FAMILIES["global"]:
        print(f"[{h}]   global cross-code aggregates (FULL stream) ...")
        blocks.append(global_features(ev_full))      # true patient volume, not just codelist events
    if FEATURE_FAMILIES["comment"]:
        print(f"[{h}]   problem-comment keyword/presence (FULL stream) ...")
        blocks.append(comment_features(ev_full))     # true comment burden, not just codelist events
    if FEATURE_FAMILIES["derangement"]:
        print(f"[{h}]   derangement + rate-trajectory (generic, no hardcode) ...")
        blocks.append(derangement_features(ev, ev_full, vb_codes))
    if FEATURE_FAMILIES["blood_ratios"]:
        print(f"[{h}]   blood ratios NLR/PLR/LMR/CRP-alb (lightly hardcoded analytes) ...")
        blocks.append(blood_ratio_features(ev_full))

    mat = patients.join(blocks, how="left")
    # 0-fill genuine-absence families (incl. per-band count/present); leave the rest NaN
    # (impute downstream, never fake-0). Substring match so '<code>_count_w12_18' is caught too.
    zero_substr = ("_count", "_present", "_decay_intensity", "_accel", "_recent_ratio",
                   "_has_active", "_has_significant")
    # Concept-layer genuine-absence cols are UPPERCASE (no-curated-event patients -> 0, not median).
    # Value/recency concept cols (months / lab levels / pack-years) must stay NaN for imputation.
    _CONCEPT_ZERO = ("_PRESENT", "_CLUSTER", "_HAS_ACTIVE", "_HAS_SIGNIFICANT", "_MEASURED",
                     "_BURDEN", "INT_", "NUM_ACTIVE", "_ACCEL", "_DECAY_INTENSITY", "_RECENT_RATIO")
    _NAN_KEEP = ("_MONTHS", "_LATEST", "_VMAX", "_VMIN", "_VMEAN", "_VALUE_ACCEL",
                 "PACK_YEARS_MAX", "CIGS_PER_DAY_MAX")
    zcols = [c for c in mat.columns
             if any(s in c for s in zero_substr)
             or (any(s in c for s in _CONCEPT_ZERO) and not any(k in c for k in _NAN_KEEP))]
    mat[zcols] = mat[zcols].fillna(0)
    # Force features to plain float64 (never pandas nullable Int64): parquet round-trips integer
    # columns as Int64, and filling their NaNs with a float median later raises TypeError.
    feat_cols = [c for c in mat.columns if c != "cancer_class"]
    mat[feat_cols] = mat[feat_cols].astype("float64").replace([np.inf, -np.inf], np.nan)  # no inf -> scaler-safe
    mat["cancer_class"] = mat.pop("cancer_class").astype("int64")    # label last, kept integer 0/1
    return mat.reset_index(), tr_guids


def main():
    for h in HORIZONS:
        out_dir = os.path.join(HERE, "output", h)
        os.makedirs(out_dir, exist_ok=True)
        mat, tr_guids = build(h)
        out = os.path.join(out_dir, f"features_p005_{h}.parquet")
        mat.to_parquet(out, index=False)
        # (train guids are NOT persisted here anymore — stability_select reads the CANONICAL split
        #  directly via splits.load_or_make, the single source of train/valid/test.)
        print(f"[{h}] wrote {mat.shape[0]:,} patients x {mat.shape[1]:,} cols -> {out}")
        print(f"      next: python stability_select.py   (k-fold importance -> stable feature set)\n")


if __name__ == "__main__":
    main()
