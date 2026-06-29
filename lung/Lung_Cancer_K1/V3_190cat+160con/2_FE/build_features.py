"""
Codelist-driven FE - Lung Cancer model  (per-code / per-category)
====================================================================
Feature engineering over the data-driven codelist: every codelist code gets the generic PER-CODE
feature vocabulary below (in CATEGORIZED mode the grouping key is the hand-assigned CATEGORY instead
of the individual SNOMED). There is NO curated clinical-concept tier.

Reads the codelist + cohort SQL:
  - codelist : ./codelist/{Astra|Nova}_{yr}yr_OFF_codelist.csv   (data-driven codes)
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
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # V3 root, for `import config` / `import splits`
import config as C
import splits as _splits
import events as _events            # shared GCS-cached cohort events (BigQuery scanned once per horizon×lookback)
import artifacts                     # atomic_write + GCS-aware exists (for persisting categorized z-stats)

# Every codelist code gets the generic per-code (or per-category) feature vocabulary below — there is
# no curated clinical-concept tier.
HORIZONS = C.horizons_from_argv(sys.argv[1:])   # e.g. `build_features.py 12mo 5`; hard-errors on a typo
# Leak-free FE: stability selection is fit on the SAME 80% TRAIN the model uses
# (replicating lung_training's seed-42 stratified 80/10/10), so the model's internal 10% test never
# informs the features. Clean 80/10/10 — no separate carved-out test set.

# Tunables sourced from the central config.py (single source of truth; logged in the run manifest) —
# NOT shadowed by local literals. getattr keeps the literal as a fallback for true standalone use.
MIN_VALUE_FRAC = getattr(C, "MIN_VALUE_FRAC", 0.10)     # a code is "value-bearing" if >= this fraction of events carry a number (config default 0.10)
DECAY_TAU_MONTHS = getattr(C, "DECAY_TAU_MONTHS", 12.0)  # recency-weighted decay constant
# WEIGHT-TREND native FE (port of weight_trend/build_weight_features.py): 40 quarterly windows of mean body
# weight (bw_w00=most-recent .. bw_w39=oldest) + bw_imputed_count. Off by default; flows through the normal
# p<0.05 -> stability -> corr-prune funnel like every other feature (NOT a bolt-on). Leak-free + parity-safe.
WEIGHT_TREND_FE   = getattr(C, "WEIGHT_TREND_FE", False)
WEIGHT_TREND_CODE = int(getattr(C, "WEIGHT_TREND_CODE", 27113001))   # SNOMED "Body weight"
RECENT_RATIO_CUTOFF = 24.0  # months (from data cutoff): "recent" band for recent_ratio (FE-internal, not in config)
# Acceleration bins, measured from the DATA CUTOFF (= start of available data), so band 0 is the
# freshest events for BOTH horizons. (Previously hardcoded 12-18/18-30/30-54 = the 12mo gap, which
# left the 1mo horizon's fresh 1-12mo events in no bin.) 2nd-diff recent-2*mid+old > 0 => ramping up.
ACCEL_BINS = {"recent": (0.0, 6.0), "mid": (6.0, 18.0), "old": (18.0, 42.0)}
# CHRONICITY (config.CHRONICITY_FE) — months thresholds for the tenure x recent-activity interaction
# (stable-chronic vs new/escalating disease). FE-internal, config-overridable.
CHRONICITY_FE = getattr(C, "CHRONICITY_FE", False)
CHRONICITY_NEW_ONSET_MO = getattr(C, "CHRONICITY_NEW_ONSET_MO", 18.0)
CHRONICITY_TENURE_MO = getattr(C, "CHRONICITY_TENURE_MO", 36.0)
CHRONICITY_RECENT_MO = getattr(C, "CHRONICITY_RECENT_MO", 12.0)
# RED-FLAG BURDEN (config.REDFLAG_FE) — curated lung-cancer red-flag CATEGORY names (must match the map).
# Each is too sparse to survive stability selection alone; redflag_features() aggregates them into one
# high-coverage composite (the non-smoker detection channel). Negation/"absent" codes are moved OUT of these
# categories by split_risk_categories.py so the burden counts POSITIVE findings only.
REDFLAG_FE = getattr(C, "REDFLAG_FE", False)
REDFLAG_CATEGORIES = frozenset({
    "Haemoptysis", "Weight loss", "Hoarseness", "Finger clubbing", "Anaemia", "Chest pain",
    "Night sweats", "Cough", "Anorexia / appetite loss", "Altered appetite", "Dysphagia", "Pleural effusion",
})
# individual red-flag + lab-side value categories that stability_select FORCE-KEEPS when REDFLAG_FE=1
REDFLAG_FORCE_KEEP_CATS = REDFLAG_CATEGORIES | {
    "Blood test - haemoglobin", "Blood test - platelets", "Blood test - CRP", "Weight",
}
# Trend cap (OPTIONAL, default OFF). The lifetime trend + the per-band/cumulative windowed trends
# already let the model pick the relevant horizon, so we do NOT throw away long-term signal by
# default. Set to a number (e.g. 60) only if you want to also force the overall slope/intervals to
# a recent window. None = use full lifetime (recommended: don't lose signal).
TREND_MAX_MONTHS = getattr(C, "TREND_MAX_MONTHS", None)
# Cumulative "last-N months" windows (from start of available data) -> count + value mean/latest.
CUMULATIVE_WINDOWS = [6, 12, 24, 60]

# Per-time-band windows. Each band gets per-code count/present (+ value mean/latest for value
# codes), so the model can split on a specific window instead of only the collapsed accel/slope.
#
# Bands ALWAYS start at 0 and are the SAME for every horizon: the cohort SQL already applied the
# gap cutoff (12mo / 1mo), so the most-recent available event is the start of the data -> band 0.
# We do NOT re-offset by the gap (that would double-count it and leave band w0_6 empty).
TIME_BANDS = [(0, 6), (6, 12), (12, 24), (24, 60), (60, 999)]   # finer, clinically-aligned bands (months);
# concentrates resolution in the prodromal window (0-24mo) where cancer signal lives. Per category each band
# emits count/present (+ value mean/latest/slope for value categories).
#   final (60, 999) is the open-ended catch-all so no event is dropped from the band view
#
# ANCHOR_MODE only decides what "month 0" is referenced to:
#   "patient_last"  -> each patient's OWN most-recent event (gap-agnostic, inference-safe).
#   "cohort_anchor" -> the cohort's data cutoff (global earliest days-before-anchor), diagnosis-aligned.
# Either way band 0 = start of available data. recency/decay/recent_ratio/accel stay anchor-relative
# (they measure staleness vs the prediction point, which is what we want and transfers to inference).
ANCHOR_MODE = getattr(C, "ANCHOR_MODE", "patient_last")

# Even 6-month bands for the GENERIC per-patient activity-rate trajectory (escalation signal):
# event/consult counts in 0-6, 6-12, 12-18 ... months before the prediction point. Rising counts
# toward month 0 = ramping presentation (prodromal). Patient-relative (from each patient's last event).
RATE_BANDS = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 36), (36, 60), (60, 999)]
LAB_Z_EXTREME = getattr(C, "LAB_Z_EXTREME", 2.0)   # |latest value z vs patient's own baseline| >= this = "extreme/deranged" lab

# Per-code problem-list value strings (CareRecord_Problem).
ACTIVE_STATUS_VALUE = "active problem"
SIGNIFICANT_VALUE = "significant problem"

FEATURE_FAMILIES = {
    "occurrence": True,   # count, present, recency, decay, accel, recent_ratio, freq, timespan, intervals, freq_trend, halves
    "flags":      True,   # has_active, has_significant
    "age":        True,   # age_first, age_last, age_median
    "value":      True,   # value/trend stats (any code carrying numeric values)
    "bands":      True,   # disjoint per-time-band count/present (+ value mean/latest/slope), TIME_BANDS
    "cumulative": True,   # cumulative last-N count (+ value mean/latest), CUMULATIVE_WINDOWS
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
    lookback's train data (build_codelist writes {Astra|Nova}_{years}yr_OFF_codelist.csv)."""
    yrs = int(years if years is not None else getattr(C, "FE_YEARS_BEFORE", 5))
    return os.path.join(HERE, "codelist", C.codelist_name(h, yrs))


def _cohort_sql_path(h):
    return os.path.join(HERE, "..", "0_SQL", f"{h}_1to1.sql")   # shared single-source SQL


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_events(h, sql_path=None, years=None):
    """Cohort events for FE — from the shared GCS-cached events (pulled from BigQuery once per
    (horizon, lookback); see ../events.py), then the FE-derived columns built locally. No re-scan.
    `sql_path` overrides the default cohort (held-out); `years` is the FE lookback window."""
    tag = "heldout_" if sql_path else ""
    ev = _events.load_cohort_events(h, years, sql_path=sql_path, tag=tag).copy()
    # Free-text findings are snomed-coded OBSERVATIONS, but some merge caches tag them event_type="text".
    # Normalize that tag to "observation" ONCE here so they (a) route to snomed at line 209 instead of
    # falling back to med_code_id (null for text -> the row would be silently dropped), and (b) are counted
    # by every observation-keyed stat below (g_distinct_obs_codes, problem-list flags, value derangement...).
    ev.loc[ev["event_type"].eq("text"), "event_type"] = "observation"
    # Build the unified code via EXACT string->Int64 (events.to_int64), NOT pd.to_numeric: the latter
    # upcasts to float64 when the column has NaNs and rounds 18-digit DMD codes > 2**53 (e.g.
    # 39115611000001103 -> ...104) -> they'd mismatch the codelist. Parsing to Python int keeps exact.
    _sn = pd.Series(_events.to_int64(ev["snomed_c_t_concept_id"]), index=ev.index)
    _md = pd.Series(_events.to_int64(ev["med_code_id"]), index=ev.index)
    ev["code"] = _sn.where(ev["event_type"].eq("observation"), _md)
    # Force PLAIN float64 (not pandas nullable Float64): when the source column (BQ/parquet) is a nullable
    # extension type, to_numeric preserves Float64, whose masked-array math raises spurious "invalid value
    # in sqrt" RuntimeWarnings (_group_linreg) and the "dtype incompatible with float64" FutureWarning in
    # the val_accel assignment. astype('float64') maps <NA>->NaN (identical downstream semantics) and is
    # value-neutral, but keeps every numeric path on numpy float64.
    ev["days"] = pd.to_numeric(ev["days_before_anchor"], errors="coerce").astype("float64")
    ev["months"] = ev["days"] / 30.44
    ev["value_num"] = pd.to_numeric(ev["value"], errors="coerce").astype("float64")
    ev["event_age"] = pd.to_numeric(ev["event_age"], errors="coerce").astype("float64")
    ev["active"] = (ev["problem_status_description"].astype(str).str.strip().str.lower()
                    == ACTIVE_STATUS_VALUE).astype(int)
    ev["sig"] = (ev["significance_description"].astype(str).str.strip().str.lower()
                 == SIGNIFICANT_VALUE).astype(int)
    # Free-text enrichment fields (ASSERTION_FE). Absent in a baseline cache -> safe defaults
    # (every event treated as a positive finding; no duration/frequency) so baseline FE is unchanged.
    ev["ft_assertion"] = (ev["ft_assertion"].astype(str).str.strip().str.lower()
                          if "ft_assertion" in ev.columns else "")
    ev["ft_dur"]  = pd.to_numeric(ev["ft_dur"],  errors="coerce") if "ft_dur"  in ev.columns else np.nan
    ev["ft_freq"] = pd.to_numeric(ev["ft_freq"], errors="coerce") if "ft_freq" in ev.columns else np.nan
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
    # Per-SNOMED mode: the key is a numeric code -> "<code>_<suffix>". CATEGORIZED mode: the key is a
    # string category label (e.g. "respiratory_clinic") -> "<category>_<suffix>" (no int cast).
    def _name(c):
        try:
            return f"{int(c)}_{suffix}"
        except (TypeError, ValueError):
            return f"{c}_{suffix}"
    w.columns = [_name(c) for c in w.columns]
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


def occurrence_features(ev, moc_zero=None):
    """All occurrence/temporal-dynamics features, per (patient, code).
    Recency/decay/accel/recent_ratio are measured from the DATA CUTOFF (start of available data)
    so they start at 0 and are horizon-consistent. Trends/intervals/halves use only the most-recent
    TREND_MAX_MONTHS. Count/present/timespan/freq are lifetime."""
    keys = ["patient_guid", "code"]
    ev = ev.copy()
    # Anchored to the COHORT-GLOBAL months.min() (not per-patient). This relies on the cohort SQL's hard
    # floor on days_before_anchor (the 12mo/1mo gap): months.min() ~= gap_months in BOTH train and held-out,
    # so the zero-point is the same constant across cohorts (sub-month drift; monotonic -> trees unaffected;
    # negligible vs DECAY_TAU=12). IF a future cohort SQL drops that gap floor, train/serve decay scales
    # would diverge — keep the floor, or switch this to a per-patient/explicit-anchor reference.
    ev["_moc"] = ev["months"] - (ev["months"].min() if moc_zero is None else moc_zero)  # months since cohort data cutoff
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
    # CHRONICITY (config.CHRONICITY_FE): tenure x recent-activity interaction — separates LONG-STANDING
    # STABLE disease (benign; raises TN / cuts elderly-COPD FPs) from NEW-ONSET / RE-ESCALATING disease
    # (cancer signal). _moc is months since the cohort data cutoff (0 = most recent): recency = min _moc
    # (most-recent event), onset = max _moc (first/oldest event = tenure). All per-(patient,category),
    # deterministic from event timing -> LEAK-SAFE (no train-fit).
    if CHRONICITY_FE:
        onset = g["_moc"].max()                                              # tenure (oldest event distance)
        recent_active = recency <= CHRONICITY_RECENT_MO                      # has an event in the recent window
        chronic = onset >= CHRONICITY_TENURE_MO                              # long-standing disease
        parts.update({
            "onset_months": onset,
            "new_onset": (onset <= CHRONICITY_NEW_ONSET_MO).astype(float),   # first appearance recent = new
            "chronic_stable": (chronic & ~recent_active).astype(float),      # old + quiet = BENIGN (raises TN)
            "escalating_chronic": (chronic & recent_active & (accel > 0)).astype(float),  # old + recent flare = SIGNAL
        })
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
    """Value/trend stats per (patient, code) for the value-carrying codes passed in `value_codes`."""
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
    # ALWAYS emit val_accel (NaN where a (patient,code) has <4 measurements) so the matrix column set is
    # data-density-independent — the column exists for every value-bearing code regardless of measurement counts.
    accel = pd.Series(np.nan, index=mean.index)
    h = vsrt[vsrt["_n"] >= 4].copy()
    if not h.empty:
        h["_half"] = np.where(h["_r"] < h["_n"] / 2.0, "old", "new")
        h["_x"] = -h["days"]
        s_old, _ = _group_linreg(h[h["_half"] == "old"], "_x", "value_num")
        s_new, _ = _group_linreg(h[h["_half"] == "new"], "_x", "value_num")
        _d = (s_new - s_old)
        accel.loc[_d.index] = _d.values
    stats["val_accel"] = accel
    return pd.concat([_wide(s_, sfx) for sfx, s_ in stats.items()], axis=1)


# Medications carry no value_num, so the value/derangement families skip them entirely. They DO carry the
# prescribing course length (`duration` = course_duration_in_days) and the dispensed drug name (`drug_term`).
# These features recover the treatment-INTENSITY / chronicity signal that pooling meds as bare occurrences
# throws away — per (patient, category): how long the courses were, total exposure, the most-recent course,
# whether use is chronic (>=28d), and how many distinct drugs in the class. NOTE: dose and quantity are NOT
# in the source data (the cohort SQL pulls only course_duration_in_days + drug term for meds), so they
# cannot be engineered here without a SQL change + cache re-pull.
MED_DUR_CAP_DAYS = getattr(C, "MED_DUR_CAP_DAYS", 365)   # course_duration_in_days has junk outliers (max ~37k) -> cap to a sane year
def med_features(ev):
    """Medication course-duration + drug-variety features, per (patient, category). Observation rows and
    meds with no duration contribute nothing; absence is zero-filled downstream like every other family."""
    keys = ["patient_guid", "code"]
    m = ev[ev["event_type"].eq("medication")]
    if m.empty or "duration" not in m.columns:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    m = m.copy()
    m["_dur"] = pd.to_numeric(m["duration"], errors="coerce").clip(lower=0, upper=MED_DUR_CAP_DAYS)
    blocks = []
    d = m.dropna(subset=["_dur"])
    if not d.empty:
        gd = d.groupby(keys)["_dur"]
        blocks += [_wide(gd.mean(), "med_dur_mean"),
                   _wide(gd.max(),  "med_dur_max"),
                   _wide(gd.sum(),  "med_dur_total"),       # cumulative exposure days in the drug class
                   _wide(gd.std(),  "med_dur_std")]
        latest = d.sort_values(keys + ["days"]).groupby(keys)["_dur"].first()   # smallest days = most recent issue
        blocks.append(_wide(latest, "med_dur_latest"))
        chronic = d.assign(_c=(d["_dur"] >= 28).astype(float)).groupby(keys)["_c"].mean()  # repeat/long-course fraction
        blocks.append(_wide(chronic, "med_chronic_frac"))
    if "drug_term" in m.columns:                            # distinct named drugs in the class (variety beyond the dm+d-code count)
        ndr = m.dropna(subset=["drug_term"]).groupby(keys)["drug_term"].nunique()
        blocks.append(_wide(ndr, "med_n_drugs"))
    blocks = [b for b in blocks if not b.empty]
    if not blocks:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    return pd.concat(blocks, axis=1)


def band_features(ev, bands, value_codes, relative, pmin=None, moc_zero=None):
    """Per-time-band occurrence (every code) + value mean/latest + value trend slope (value codes).
    `bands` = list of [lo, hi) month windows. `relative=True` -> windows count back from each
    patient's most-recent event (gap-agnostic, inference-safe); False -> fixed months-before-anchor.
    Lets the model split on a specific window rather than only the collapsed accel / overall slope."""
    keys = ["patient_guid", "code"]
    ev = ev.copy()
    if relative:                                                 # band 0 = each patient's last event
        # INVARIANT: in patient_last/relative mode the per-patient anchor (pmin) MUST come from the
        # FULL stream. Passing the restricted stream's min shifts the anchor for patients whose latest
        # event is a non-kept category -> the keep-features value-mismatch bug. build() passes _pmin from ev_remaining.
        ev["_mo"] = ev["months"] - (ev["patient_guid"].map(pmin) if pmin is not None else ev.groupby("patient_guid")["months"].transform("min"))
    else:                                                        # band 0 = cohort data cutoff (gap)
        # use the ZERO-FIT TRAIN anchor (moc_zero) when given, so the band origin is the
        # same constant at train and serve (recomputing the cohort min per-cohort would shift it).
        ev["_mo"] = ev["months"] - (moc_zero if moc_zero is not None else ev["months"].min())
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


# train_guids() removed — the CANONICAL split (splits.load_or_make / make_split.py) is now the single
# source of train/valid/test; no per-module train_test_split re-derivation. build() reads train guids
# via _splits.guids_for(...); stability_select reads the same canonical split.


def cumulative_features(ev, windows, value_codes, relative, pmin=None, moc_zero=None):
    """Cumulative 'last-N months' windows (from start of available data): count (+ value mean/latest
    for value codes). Overlapping windows let the model read accumulation over time, alongside the
    disjoint bands."""
    keys = ["patient_guid", "code"]
    ev = ev.copy()
    if relative:
        # INVARIANT: in patient_last/relative mode the per-patient anchor (pmin) MUST come from the
        # FULL stream. Passing the restricted stream's min shifts the anchor for patients whose latest
        # event is a non-kept category -> the keep-features value-mismatch bug. build() passes _pmin from ev_remaining.
        ev["_mo"] = ev["months"] - (ev["patient_guid"].map(pmin) if pmin is not None else ev.groupby("patient_guid")["months"].transform("min"))
    else:
        # zero-fit TRAIN anchor when given (same constant train↔serve).
        ev["_mo"] = ev["months"] - (moc_zero if moc_zero is not None else ev["months"].min())
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


def global_features(ev, demographics_only=False):
    """Per-PATIENT features. By default: cross-code utilisation VOLUME (event/code counts, consultation
    cadence, problem-list burden) over the FULL stream + demographics. With `demographics_only=True`
    (STRICT category-only mode) it emits ONLY demographics (age/sex/ethnicity/age-bands) — no volume
    features, which would otherwise be a care-seeking/utilisation proxy from non-map codes."""
    ev = ev.copy()
    ev["_moc"] = ev["months"] - ev["months"].min()
    g = ev.groupby("patient_guid")
    out = pd.DataFrame(index=pd.Index(ev["patient_guid"].unique(), name="patient_guid"))
    if demographics_only:
        return _demographics_block(ev, g, out)
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

    return _demographics_block(ev, g, out)


def redflag_features(ev, moc_zero=None):
    """COMPOSITE lung-cancer RED-FLAG BURDEN (config.REDFLAG_FE) — the non-smoker detection channel.
    Aggregates the individually-too-sparse red-flag categories (REDFLAG_CATEGORIES) into one per-patient
    feature family with enough coverage to survive stability selection. All count/decay-type, so the
    downstream fill (0 = absent) is the correct neutral. Per-patient deterministic over event timing vs
    the cohort cutoff -> LEAK-SAFE (no train-fit). Mirrors occurrence/global decay/accel conventions."""
    out = pd.DataFrame(index=pd.Index(ev["patient_guid"].unique(), name="patient_guid"))
    cols = ["g_redflag_n_distinct", "g_redflag_n_events", "g_redflag_present",
            "g_redflag_recent6_distinct", "g_redflag_recent12_distinct", "g_redflag_decay", "g_redflag_accel"]
    rf = ev[ev["code"].isin(REDFLAG_CATEGORIES)].copy()
    if rf.empty:
        out[cols] = 0.0
        return out
    rf["_moc"] = rf["months"] - (rf["months"].min() if moc_zero is None else moc_zero)
    g = rf.groupby("patient_guid")
    out["g_redflag_n_distinct"] = g["code"].nunique()
    out["g_redflag_n_events"] = g.size()
    out["g_redflag_present"] = out["g_redflag_n_events"].notna().astype(float)   # 1 only for patients WITH a red flag
    out["g_redflag_recent6_distinct"] = rf[rf["_moc"] <= 6].groupby("patient_guid")["code"].nunique()
    out["g_redflag_recent12_distinct"] = rf[rf["_moc"] <= 12].groupby("patient_guid")["code"].nunique()
    rf["_w"] = np.exp(-rf["_moc"] / DECAY_TAU_MONTHS)
    out["g_redflag_decay"] = rf.groupby("patient_guid")["_w"].sum()

    def _b(lo, hi):
        return (rf[(rf["_moc"] >= lo) & (rf["_moc"] < hi)]
                .groupby("patient_guid").size().reindex(out.index, fill_value=0))
    out["g_redflag_accel"] = _b(*ACCEL_BINS["recent"]) - 2 * _b(*ACCEL_BINS["mid"]) + _b(*ACCEL_BINS["old"])
    out[cols] = out[cols].fillna(0)   # patients with no red flag -> 0 (correct neutral)
    return out


def _demographics_block(ev, g, out):
    """Patient demographics: age (+ age-band one-hots), sex, ethnicity one-hots. Used by both the full
    global_features and the STRICT demographics_only path. Numeric so they flow through stability-selection
    (age/sex -> NaN where unknown -> imputed; ethnicity/age-band one-hots 0/1 -> missing = all-zero).
    age is the single biggest lung-cancer risk factor."""
    # L9 guard: age_at_prediction is the model's #1 feature. If age_at_anchor ever silently drops from the
    # schema we would emit NO age columns and median-impute the top feature with no error. In CATEGORIZED
    # mode (the prod path) fail loud instead.
    if getattr(C, "CATEGORIZED", False):
        assert "age_at_anchor" in ev.columns, ("[fe] age_at_anchor missing from the stream -> age_at_prediction "
                                               "would be silently dropped (it is the top feature). Check the cohort SQL/schema.")
    if "age_at_anchor" in ev.columns:
        # age at the anchor = the prediction point (NOT snapshot age, which would leak time-since-anchor).
        out["age_at_prediction"] = pd.to_numeric(g["age_at_anchor"].first(), errors="coerce")
        # age-band one-hots (lung-ca risk rises steeply with age). Edges: <50, 50-59, 60-69, 70-79, 80+.
        _bands = pd.cut(out["age_at_prediction"], bins=[0, 50, 60, 70, 80, 200],
                        right=False, labels=["u50", "50_59", "60_69", "70_79", "80p"])
        out = out.join(pd.get_dummies(_bands, prefix="ageband").astype(float))
    if "sex" in ev.columns:
        sx = g["sex"].first().astype(str).str.strip().str.lower()
        out["g_is_male"] = pd.Series(np.where(sx.str.startswith("m"), 1.0,
                                     np.where(sx.str.startswith("f"), 0.0, np.nan)), index=sx.index)
    if "patient_ethnicity_6" in ev.columns:
        # FIXED ethnicity vocabulary so the g_eth_* column set is IDENTICAL across every cohort (no schema
        # drift): pd.get_dummies alone emits only the values present in THIS cohort, so an ethnicity seen
        # only at deployment would be silently dropped (that patient loses its ethnicity signal). These are
        # the authoritative LABEL_6 groups from EMIS_BULK_DATA_temp.Patient_Ethnicity; NULL / unseen -> the
        # explicit "Unknown" level (never imputed to White). Reindex to the fixed columns at write time.
        ETHNICITY_LEVELS = list(getattr(C, "ETHNICITY_LEVELS",
            ["White", "Asian or Asian British", "Black or Black British",
             "Chinese or Other Ethnic Groups", "Mixed", "Unknown"]))
        eth = g["patient_ethnicity_6"].first().astype(str).str.strip()
        eth = eth.where(~eth.str.lower().isin(["nan", "none", "", "null"])).fillna("Unknown")
        eth = eth.where(eth.isin(ETHNICITY_LEVELS), "Unknown")                       # any unseen value -> Unknown
        _ed = pd.get_dummies(eth, prefix="g_eth").astype(float)
        out = out.join(_ed.reindex(columns=[f"g_eth_{e}" for e in ETHNICITY_LEVELS], fill_value=0.0))
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
    clinical blood-ratio composites (NLR/clusters/mGPS) as per-patient aggregates over the per-code data:
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


def clinical_concept_features(ev):
    """Arm C — ABSOLUTE clinical-threshold lab flags from RAW values (NOT the per-code z used in categorized
    mode), so they encode 'clinically abnormal' (population scale), which the z-scoring structurally cannot.
    Runs on the FULL raw stream (like blood_ratio_features) -> keep-features-safe. Lung-relevant markers:
      clin_thrombocytosis_* : platelets > THROMBOCYTOSIS_PLT (paraneoplastic marker; validated p95~408 here)
      clin_raised_crp_*     : CRP > RAISED_CRP_MGL (systemic inflammation)
      clin_hypoalbuminemia_*: albumin < HYPOALBUMINEMIA_GL (cachexia/cancer; DENSITY-INDEPENDENT — one low reading is signal)
    For each: _ever (any abnormal), _latest (most-recent reading abnormal), _count (# abnormal), _extremeval (peak/nadir).
    Counts/flags -> 0 when absent; _extremeval stays NaN (imputed). Hb/anaemia + LDH omitted (no usable / too-sparse values)."""
    obs = ev[ev["event_type"].eq("observation")].dropna(subset=["value_num"])
    out = pd.DataFrame(index=pd.Index(ev["patient_guid"].unique(), name="patient_guid"))
    if obs.empty:
        return out
    def flags(codes, thr, tag, hi=True):
        s = obs[obs["code"].isin(codes)].sort_values(["patient_guid", "days"])   # asc days = most recent first
        if s.empty:
            return
        ab = (s["value_num"] > thr) if hi else (s["value_num"] < thr)
        s = s.assign(_ab=ab.astype(float))
        g = s.groupby("patient_guid")
        out[f"clin_{tag}_ever"]    = g["_ab"].max()        # ever abnormal
        out[f"clin_{tag}_latest"]  = g["_ab"].first()      # most-recent reading abnormal (smallest days)
        out[f"clin_{tag}_count"]   = g["_ab"].sum()        # # abnormal readings
        out[f"clin_{tag}_extremeval"] = (g["value_num"].max() if hi else g["value_num"].min())  # peak (hi) / nadir (lo), absolute scale
    flags(PLATELET_CODE, float(getattr(C, "THROMBOCYTOSIS_PLT", 400.0)), "thrombocytosis",  hi=True)
    flags(CRP_CODES,     float(getattr(C, "RAISED_CRP_MGL", 10.0)),      "raised_crp",      hi=True)
    flags(ALBUMIN_CODES, float(getattr(C, "HYPOALBUMINEMIA_GL", 35.0)),  "hypoalbuminemia", hi=False)
    zc = [c for c in out.columns if not c.endswith("_extremeval")]   # flags/counts -> genuine 0; _extremeval -> NaN (imputed)
    out[zc] = out[zc].fillna(0)
    return out


# ---------------------------------------------------------------- CATEGORIZED-only category-structure FE
def category_diversity_features(ev, z_extreme=2.0):
    """Within-category richness (CATEGORIZED), all from the user's categories only. `ev` has the original
    SNOMED in '_snomed', the category in 'code', 'months' (before anchor), and 'value_num' (= the per-code
    z in categorized mode). Per (patient, category):
        <cat>_n_distinct_codes / _distinct_ratio  : SNOMED diversity within the category
        <cat>_count_share                         : category events / patient's TOTAL category events
                                                    (utilisation-normalized using MAP events only)
        <cat>_first_months                        : months-before-anchor of the FIRST occurrence (tenure)
        <cat>_n_extreme / _max_abs_z              : value derangement within the category (from the z)
    Patient-level:
        g_n_categories                            : # distinct categories (breadth)
        g_category_entropy / g_category_gini      : how spread vs concentrated the category profile is
    """
    keys = ["patient_guid", "code"]
    g = ev.groupby(keys)
    n_events = g.size()
    n_distinct = g["_snomed"].nunique()
    ratio = n_distinct / n_events.replace(0, np.nan)
    first = g["months"].max()                                            # earliest = largest months-before-anchor
    tot = ev.groupby("patient_guid").size()                              # patient's TOTAL category events
    share = n_events / pd.Series(n_events.index.get_level_values(0), index=n_events.index).map(tot)
    # value derangement within the category (value_num is the unit-robust z in categorized mode)
    # build _az from zz's OWN value_num (copy first) — .assign(_az=ev[...]) sourced .abs() from the
    # parent ev and relied on a unique index to align; a future concat without reset_index would misalign.
    zz = ev.dropna(subset=["value_num"]).copy()
    zz["_az"] = zz["value_num"].abs()
    n_ext = zz[zz["_az"] >= z_extreme].groupby(keys).size().reindex(n_events.index, fill_value=0)
    max_z = zz.groupby(keys)["_az"].max()
    out = pd.concat([_wide(n_distinct, "n_distinct_codes"), _wide(ratio, "distinct_ratio"),
                     _wide(share, "count_share"), _wide(first, "first_months"),
                     _wide(n_ext, "n_extreme"), _wide(max_z, "max_abs_z")], axis=1)
    # patient-level breadth + concentration (entropy / Gini of the per-category event shares)
    def _ent(s):
        p = (s / s.sum()).values
        return float(-(p * np.log(p)).sum())
    def _gini(s):
        a = np.sort(s.values.astype(float)); n = len(a); tot_ = a.sum()
        return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * tot_)) if tot_ > 0 else 0.0
    pc = n_events.groupby("patient_guid")
    pat = pd.DataFrame({"g_n_categories": ev.groupby("patient_guid")["code"].nunique(),
                        "g_category_entropy": pc.apply(_ent),
                        "g_category_gini": pc.apply(_gini)})
    return out.join(pat, how="outer")


def cross_category_features(ev, top_cats):
    """Cross-category structure (CATEGORIZED mode) over the TOP-K categories `top_cats` (kept tractable):
        cooc_<A>__<B>        = 1.0 if the patient has >=1 event in BOTH categories (unordered presence)
        seq_<A>__before__<B> = 1.0 if A's FIRST event is earlier than B's first (A precedes B; directional)
    Only the top_cats are paired (K*(K-1)/2 co-occurrence + ordered sequence), so the column count stays
    bounded regardless of how many categories exist. Returns a patient_guid-indexed frame."""
    import itertools
    cats = sorted(c for c in set(top_cats) if c is not None)
    sub = ev[ev["code"].isin(cats)]
    if sub.empty or len(cats) < 2:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    pres = sub.assign(_one=1).pivot_table(index="patient_guid", columns="code", values="_one",
                                          aggfunc="max", fill_value=0)
    cnt = sub.pivot_table(index="patient_guid", columns="code", values="days", aggfunc="size", fill_value=0)  # event counts
    first = sub.pivot_table(index="patient_guid", columns="code", values="days", aggfunc="max")  # max days = earliest
    cats = [c for c in cats if c in pres.columns]
    out = {}
    for a, b in itertools.combinations(cats, 2):
        both = (pres[a] * pres[b]).astype(float)
        out[f"cooc_{a}__{b}"] = both                                          # presence co-occurrence (0/1)
        out[f"coocn_{a}__{b}"] = (np.minimum(cnt[a], cnt[b]) * both).astype(float)   # co-occurrence STRENGTH
        out[f"seq_{a}__before__{b}"] = ((first[a] > first[b]) & (both > 0)).astype(float)
        out[f"seq_{b}__before__{a}"] = ((first[b] > first[a]) & (both > 0)).astype(float)
    return pd.DataFrame(out, index=pres.index)


def category_age_interactions(ev, top_cats, age):
    """CATEGORIZED #6: for the top-K categories, INT_<cat>_x_age = (category event count) * patient age.
    Lets the model learn that a category's signal scales with age (a top lung-ca risk factor) without a
    separate concept layer. `age` is a patient_guid-indexed age Series; uses only category events + age."""
    cats = sorted(c for c in set(top_cats) if c is not None)
    sub = ev[ev["code"].isin(cats)]
    if sub.empty or age is None or not len(age):
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    cnt = sub.pivot_table(index="patient_guid", columns="code", values="months", aggfunc="size", fill_value=0)
    a = pd.to_numeric(age.reindex(cnt.index), errors="coerce")
    out = {f"INT_{c}_x_age": (cnt[c].astype(float) * a) for c in cnt.columns}
    return pd.DataFrame(out, index=cnt.index)


def category_temporal_features(ev):
    """Inter-category temporal structure (CATEGORIZED). `ev` has the category in 'code' and 'months'
    (= months before the anchor; smaller = more recent). Emits:
      <category>_recency_rank  : rank of the category by recency within the patient (1 = most recent)
      g_last_category_months   : months-before-anchor of the patient's MOST-RECENT mapped-category event
      g_category_span_months   : span (oldest -> newest) of the patient's mapped-category events
      g_n_categories_6mo/_12mo : # distinct categories with an event in the last 6 / 12 months
    Richer than the lifetime breadth g_n_categories: captures *when* and *how concentrated* category activity is."""
    rec = ev.groupby(["patient_guid", "code"])["months"].min()          # most-recent occurrence per category
    rank = rec.groupby("patient_guid").rank(method="min", ascending=True)
    out = _wide(rank, "recency_rank")                                   # per-category recency rank
    pg = rec.groupby("patient_guid")
    n6 = rec[rec <= 6].reset_index().groupby("patient_guid")["code"].nunique()
    n12 = rec[rec <= 12].reset_index().groupby("patient_guid")["code"].nunique()
    pat = pd.DataFrame({"g_last_category_months": pg.min(),
                        "g_category_span_months": pg.max() - pg.min(),
                        "g_n_categories_6mo": n6, "g_n_categories_12mo": n12})
    return out.join(pat, how="outer")


def _zero_fill_genuine_absence(mat):
    """0-fill the genuine-absence feature families (counts / flags / category-structure / interactions /
    one-hots); value & recency columns stay NaN for downstream median-impute. Factored helper used by build()."""
    zero_substr = ("_count", "_present", "_decay_intensity", "_accel", "_recent_ratio",
                   "_has_active", "_has_significant",
                   "_n_distinct_codes", "_distinct_ratio", "g_n_categories", "cooc_", "coocn_", "seq_",
                   "_n_extreme",
                   "clin_",                      # clinical-threshold flags (clin_*_ever/_latest/_count) -> genuine 0
                   "g_eth_", "ageband_")
    _CONCEPT_ZERO = ("_PRESENT", "_CLUSTER", "_HAS_ACTIVE", "_HAS_SIGNIFICANT", "_MEASURED",
                     "_BURDEN", "INT_", "NUM_ACTIVE", "_ACCEL", "_DECAY_INTENSITY", "_RECENT_RATIO")
    _NAN_KEEP = ("_MONTHS", "_LATEST", "_VMAX", "_VMIN", "_VMEAN", "PACK_YEARS_MAX", "CIGS_PER_DAY_MAX")
    zcols = [c for c in mat.columns
             if c != "cancer_class" and "_val_" not in c and "_extremeval" not in c   # clin_*_extremeval stays NaN -> imputed
             and (any(s in c for s in zero_substr)
                  or (any(s in c for s in _CONCEPT_ZERO) and not any(k in c for k in _NAN_KEEP)))]
    mat[zcols] = mat[zcols].fillna(0)
    return mat


def assertion_attribute_features(ev):
    """Free-text assertion-split + attribute features, per (patient, category). ASSERTION_FE only.
    `ev` is the all-assertion mapped stream (`code` = category). Negated ("absent") and family-history
    mentions are kept SEPARATE from the positive-finding burden (which the base families already count),
    so "no cough" never inflates the "cough" count. Duration/frequency stats are over positive findings."""
    keys = ["patient_guid", "code"]
    a = ev["ft_assertion"].astype(str)
    out = []
    for tag, mask in (("absent", a.eq("absent")), ("fhx", a.eq("family"))):
        sub = ev[mask]
        if len(sub):
            cnt = sub.groupby(keys).size()
            out.append(_wide(cnt, f"{tag}_count"))
            out.append(_wide((cnt > 0).astype(int), f"{tag}_present"))
    pos = ev[a.isin(["present", "historical", ""])]
    for col, sfx in (("ft_dur", "dur"), ("ft_freq", "freq")):
        sub = pos.dropna(subset=[col])
        if len(sub):
            g = sub.groupby(keys)[col]
            out.append(_wide(g.mean(), f"{sfx}_mean"))
            out.append(_wide(g.max(),  f"{sfx}_max"))
            out.append(_wide(g.last(), f"{sfx}_last"))
    if not out:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    return pd.concat(out, axis=1)


def weight_trend_features(ev):
    """Native port of weight_trend/build_weight_features.py. 40 quarterly windows of MEAN body weight
    (bw_w00 = most recent 0-90d before anchor .. bw_w39 = oldest) + bw_imputed_count. Per-patient LINEAR
    interpolation across windows (ffill/bfill the edges). LEAK-FREE: patients with zero weight readings are
    left ALL-NaN (NO cohort/class median fill). Sourced from ev_full RAW kg values under SNOMED
    WEIGHT_TREND_CODE over the FULL stream -> parity-safe (pure per-patient transform, no cohort fit).
    Returned as a patient_guid-indexed block so it flows through the normal selection funnel."""
    NW, WD, LO, HI = 40, 91, 20.0, 300.0
    cols = [f"bw_w{i:02d}" for i in range(NW)]
    sub = ev[ev["code"] == WEIGHT_TREND_CODE].dropna(subset=["value_num"]).copy()
    sub = sub[(sub["value_num"] >= LO) & (sub["value_num"] <= HI) & sub["days"].notna() & (sub["days"] >= 0)]
    if sub.empty:
        return pd.DataFrame(index=pd.Index([], name="patient_guid"))
    sub["_win"] = (sub["days"] // WD).clip(upper=NW - 1).astype(int)
    wide = (sub.groupby(["patient_guid", "_win"])["value_num"].mean()
               .unstack("_win").reindex(columns=range(NW)))
    wide.columns = cols
    was_nan = wide.isna()
    wide = wide.interpolate(method="linear", axis=1, limit_direction="both")   # per-patient; all-NaN rows stay NaN
    wide["bw_imputed_count"] = was_nan.sum(axis=1).astype("float64")
    wide.index.name = "patient_guid"
    return wide


def build(h, sql_path=None, fit_split=True, years=None, patient_ids=None, shared=None, schema_only=False, keep_features=None):
    """Build the per-code feature matrix for horizon `h`.
    `sql_path`  : override the default 0_SQL cohort (e.g. 4_Heldout's held-out SQL).
    `fit_split` : True for the training cohort (compute the 80% train guids that stability-selection
                  reuses); False for held-out (touch-once eval — no internal split, nothing fit here).
    `years`     : FE lookback window in years — REQUIRED (the run selects it; no year is hardcoded)."""
    categorized = getattr(C, "CATEGORIZED", False)
    # The lookback window must be selected by the caller (run_v3 passes the discovered/selected window).
    # No year is hardcoded; a missing window is a hard error rather than a silent fallback.
    yrs = int(years) if years is not None else int(getattr(C, "FE_YEARS_BEFORE", 0) or 0)
    if categorized and yrs <= 0:
        raise SystemExit("[fe] CATEGORIZED: no lookback window given. Pass `years` (e.g. build(h, years=10)) "
                         "or run via run_v3 (auto-discovers from 2_FE/categorized_codelist/).")
    if shared is None:
        ev_full = load_events(h, sql_path, years)   # full event stream (every code) — true patient volume
        # LEAKAGE GUARD: drop post-diagnosis codes (lung resection/surgery) from the FULL stream up front, so
        # they can NEVER enter any feature (category map, volume, global) in either mode. These occur only
        # after a confirmed cancer diagnosis -> not available at a genuine pre-dx prediction point (config.LEAKY_CODES).
        _leaky = set(getattr(C, "LEAKY_CODES", ()) or ())
        if _leaky:
            _before = len(ev_full)
            ev_full = ev_full[~ev_full["code"].isin(_leaky)].copy()
            _dropped = _before - len(ev_full)
            if _dropped:
                print(f"[{h}] LEAKAGE guard: dropped {_dropped:,} event(s) for {len(_leaky)} post-dx surgical code(s)")

        ev_assert = None   # ASSERTION_FE: the all-assertion mapped stream (absent/family), set in the categorized branch
        if categorized:
            # --- CATEGORIZED FE: the grouping key flips from per-SNOMED `code` to a hand-assigned CATEGORY.
            # Read the SNOMED->category map for THIS (horizon, lookback), restrict the stream to mapped codes,
            # then OVERWRITE the `code` column with the category label so every downstream per-"code" family
            # (occurrence/flags/age/bands/cumulative) emits "<category>_<family>" with no other changes. ---
            cat_path = os.path.join(HERE, "categorized_codelist", C.categorized_name(h, yrs))
            assert os.path.exists(cat_path), (f"[fe] CATEGORIZED: category map not found for {h} {yrs}yr:\n  {cat_path}\n"
                                              f"  expected 2_FE/categorized_codelist/{C.categorized_name(h, yrs)} "
                                              f"(cols Code,Name,Category).")
            # Read Code as STRING + parse with events.to_int64 (EXACT) — NOT pd.to_numeric: the latter upcasts
            # to float64 and ROUNDS the 17-18 digit DM+D medication codes (> 2**53) in their last digits, so
            # they'd never match the event stream (which is parsed with to_int64). Reading as str also stops
            # pandas float-rounding the big codes at read time. (Same exact-int discipline as events.py.)
            cmap = pd.read_csv(cat_path, dtype=str)                          # cols: Code, Name, Category
            ccol = "Code" if "Code" in cmap.columns else "code_id"
            cmap = cmap[[ccol, "Category"]].dropna()
            cmap["_code"] = pd.Series(_events.to_int64(cmap[ccol]), index=cmap.index)   # exact int64 (no >2**53 rounding)
            cmap = cmap.dropna(subset=["_code"])
            code2cat = dict(zip(cmap["_code"].astype("int64"), cmap["Category"].astype(str).str.strip()))
            keep = set(code2cat)
            ev = ev_full[ev_full["code"].isin(keep)].copy()
            ev["_snomed"] = ev["code"]                                      # keep original SNOMED for within-category diversity
            ev["code"] = ev["code"].map(code2cat)                           # code -> category (grouping key)
            print(f"[{h}] CATEGORIZED FE: {len(keep):,} SNOMEDs -> {ev['code'].nunique():,} categories "
                  f"({os.path.basename(cat_path)})")
            # --- category-map QC: coverage + category sizes + map codes absent from the cohort ---
            _cat_sizes = pd.Series(code2cat).value_counts()                 # # codes mapped into each category
            _cov = 100.0 * len(ev) / max(len(ev_full), 1)                   # % of cohort events that are categorized
            _absent = sorted(keep - set(pd.to_numeric(ev_full["code"], errors="coerce").dropna().astype("int64")))
            _singletons = int((_cat_sizes == 1).sum())
            print(f"[{h}] category-map QC: event coverage {_cov:.1f}% ({len(ev):,}/{len(ev_full):,}); "
                  f"category sizes median {int(_cat_sizes.median())} / max {int(_cat_sizes.max())} "
                  f"({_cat_sizes.idxmax()}); {_singletons} singleton categories")
            if _absent:
                print(f"[{h}]   [QC warn] {len(_absent)} mapped codes have NO events in the cohort (e.g. {_absent[:5]})")
            if _cov < 50:
                print(f"[{h}]   [QC warn] only {_cov:.1f}% of coded events are categorized — much signal falls "
                      f"outside the map (consider widening the category map)")
            # ASSERTION_FE: the base per-category families count POSITIVE findings only. Absent/family events
            # (free-text) are held aside in ev_assert for assertion_attribute_features so "no cough" never
            # inflates the "cough" count. Structured events (ft_assertion="") are positive by default.
            if getattr(C, "ASSERTION_FE", False):
                _a = ev["ft_assertion"]
                _pos = _a.isin(["present", "historical", ""]) | _a.isna()
                ev_assert = ev.copy()
                ev = ev[_pos].copy()
                print(f"[{h}] ASSERTION_FE: base families on {len(ev):,} positive findings; "
                      f"{int((~_pos).sum()):,} absent/family events routed to the assertion block")
            ev_remaining = ev                            # all category events get the generic per-category families
            # STRICT category-only FE: every feature derives from the user's categorized_codelist (categories)
            # + category-structure + patient demographics. No hardcoded clinical codes, no concept tier.
        else:
            codes = pd.read_csv(_codelist_path(h, years))
            # Codelist columns: Code, Name, Value + selection scores (combined_rank/odds_ratio/p_value/prevalence).
            # FE only needs the numeric Code column; the rest are for human review and are ignored here.
            code_col = "Code" if "Code" in codes.columns else "code_id"   # back-compat with the old code_id format
            keep = set(pd.to_numeric(codes[code_col], errors="coerce").dropna().astype("int64"))
            print(f"[{h}] codelist: {len(keep):,} data-driven codes")

            ev = ev_full[ev_full["code"].isin(keep)]    # codelist-restricted
            ev_remaining = ev                            # every codelist code gets the generic per-code families

        # B: NEVER drop an extracted cohort patient — keep EVERY patient in ALL phases (training + held-out),
        # even those with ZERO events in THIS lookback window. Such patients are absent from the WINDOWED
        # stream (ev_full) but ARE present in the UN-WINDOWED lifetime cache (they have older events) with
        # their REAL demographics + label. Build the patient roster from the lifetime cache so (a) nobody is
        # silently dropped, and (b) the zero-window rows get REAL age/sex/ethnicity (not imputed). Event-based
        # families still come from the windowed stream (left-join -> genuine-absence 0 / impute); only the
        # demographics block is fed this roster so every cohort patient is covered. Works for BOTH cohorts:
        # training uses 0_SQL (== canonical split), held-out passes its own sql_path (== held-out roster).
        _rtag = "heldout_" if sql_path else ""
        _roster = (_events.load_cohort_events(h, years=_events.LIFETIME_YEARS, sql_path=sql_path, tag=_rtag)
                     .drop_duplicates("patient_guid").copy())
        _roster = _roster[[c for c in ("patient_guid", "cancer_class", "age_at_anchor", "sex",
                                       "patient_ethnicity_6") if c in _roster.columns]]
        _roster["months"] = 0.0     # global_features(demographics_only) reads a `months` col (value unused here)
        patients = _roster.set_index("patient_guid")[["cancer_class"]]
        _zero_win = patients.index.difference(pd.Index(ev_full["patient_guid"].unique()))
        if len(_zero_win):
            print(f"[{h}] preserve-all: {len(_zero_win):,} cohort patient(s) have 0 events in the {yrs}yr window "
                  f"-> kept as demographic-only rows (real demographics from lifetime cache; total {len(patients):,})")
        # TRAIN guids from the CANONICAL saved split (split-first) — the SAME patients as scoring + model.
        # Held-out (fit_split=False) does no split. (guids_for returns CLEANED guids.)
        tr_guids = (_splits.guids_for(_splits.load_required(h), "train")   # canonical split must exist (make_split.py)
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
        # value_codes = codes/categories that carry ANY numeric value, determined on TRAIN rows only
        # (strict split-first; clean-match) so the internal test never influences the schema. Held-out: all.
        # CATEGORIZED mode does CATEGORY VALUE POOLING: the value/trend family runs at the CATEGORY level
        # (numeric values pooled within each category — the hand-curated category groupings are assumed
        # same-unit). vb_codes (strict, for the lab-derangement family) is left empty: derangement +
        # blood-ratios stay off in categorized (cross-code lab burden / hardcoded analytes don't transfer
        # to pooled categories).
        if categorized:
            # UNIT-ROBUST value pooling: standardize each ORIGINAL SNOMED's values to a z-score vs its OWN
            # TRAIN distribution BEFORE the value family pools them per category — so pooling values within a
            # (possibly mixed-unit) category is meaningful (e.g. platelets + Hb in one category no longer mix
            # raw units; each contributes its standardized deviation). value_num is replaced in-place by the z.
            # ZERO-FIT held-out: the per-SNOMED train mean/std are PERSISTED when fit_split=True and RELOADED
            # when fit_split=False, so held-out FE uses the identical transform (no train/serve skew, no refit).
            _zs_local = os.path.join(HERE, "output", C.artifact_subdir(h, yrs), f"zstats_categorized_{h}.parquet")
            _zs_gcs = (f"{C.GCS_ARTIFACTS.rstrip('/')}/2_FE/output/{C.artifact_subdir(h, yrs)}/zstats_categorized_{h}.parquet"
                       if getattr(C, "GCS_ARTIFACTS", "") else None)
            if fit_split and tr_guids:
                # FIT on TRAIN only, then PERSIST (local + GCS mirror) so held-out reuses the identical transform.
                _vsrc = ev_remaining.dropna(subset=["value_num"])
                _base = _vsrc[_splits.clean_guid(_vsrc["patient_guid"]).isin(tr_guids)]
                _st = _base.groupby("_snomed")["value_num"].agg(["mean", "std"])
                _out = _st.reset_index()
                artifacts.atomic_write(_zs_local, lambda t: _out.to_parquet(t, index=False))
                if _zs_gcs:
                    try:
                        artifacts.atomic_write(_zs_gcs, lambda t: _out.to_parquet(t, index=False))
                    except Exception as _ze:
                        raise SystemExit(f"[{h}] FE ABORT: zstats GCS mirror FAILED ({_ze}) — refusing to report FE "
                                         f"success with a half-published sidecar set (held-out/serve on another VM would "
                                         f"hard-fail on zero-fit reload). Fix GCS write access and re-run.")
                _src = "TRAIN stats (persisted)"
            else:
                # HELD-OUT / no-split: REUSE saved train stats — NEVER refit on held-out (zero-fit).
                if os.path.exists(_zs_local):
                    _st = pd.read_parquet(_zs_local).set_index("_snomed")[["mean", "std"]]; _src = "TRAIN stats (loaded local)"
                elif _zs_gcs and artifacts.exists(_zs_gcs):
                    _st = pd.read_parquet(_zs_gcs).set_index("_snomed")[["mean", "std"]]; _src = "TRAIN stats (loaded GCS)"
                else:
                    raise SystemExit(
                        f"[fe] CATEGORIZED held-out: train z-stats not found (local or GCS):\n  {_zs_local}\n"
                        f"  Run the training FE for {h} {yrs}yr first (run_v3.py) so the stats are persisted.")
            _m = ev_remaining["_snomed"].map(_st["mean"])
            _sd = ev_remaining["_snomed"].map(_st["std"]).replace(0, np.nan)
            ev_remaining["value_num"] = (ev_remaining["value_num"] - _m) / _sd   # per-code z (TRAIN basis) -> pooled by category
            # value_codes (which CATEGORIES emit value/band/cumulative sub-columns) is ZERO-FIT like the
            # z-stats + top_cats above — FIT on TRAIN ONLY + PERSIST when fit_split, RELOAD when held-out/serve.
            # Previously it was taken from ev_remaining (full cohort incl. valid/test at train; whole held-out
            # at serve) and never persisted -> the value-column VOCABULARY leaked valid/test presence into the
            # training matrix and could drift train-vs-serve (the score gate can't see it: it runs full-vs-keep-features
            # on the SAME cohort). The values themselves are already TRAIN-based (persisted z-stats).
            _vc_local = os.path.join(HERE, "output", C.artifact_subdir(h, yrs), f"valuecodes_categorized_{h}.parquet")
            _vc_gcs = (f"{C.GCS_ARTIFACTS.rstrip('/')}/2_FE/output/{C.artifact_subdir(h, yrs)}/valuecodes_categorized_{h}.parquet"
                       if getattr(C, "GCS_ARTIFACTS", "") else None)
            if fit_split and tr_guids:
                _vc_tr = ev_remaining[_splits.clean_guid(ev_remaining["patient_guid"]).isin(tr_guids)]
                value_codes = set(_vc_tr.dropna(subset=["value_num"])["code"].unique())
                _vcdf = pd.DataFrame({"code": pd.Series(sorted(map(str, value_codes)), dtype=str)})
                artifacts.atomic_write(_vc_local, lambda t: _vcdf.to_parquet(t, index=False))
                if _vc_gcs:
                    try:
                        artifacts.atomic_write(_vc_gcs, lambda t: _vcdf.to_parquet(t, index=False))
                    except Exception as _ve:
                        raise SystemExit(f"[{h}] FE ABORT: valuecodes GCS mirror FAILED ({_ve}) — refusing to report FE "
                                         f"success with a half-published sidecar set (held-out/serve on another VM would "
                                         f"hard-fail on zero-fit reload). Fix GCS write access and re-run.")
            else:
                if os.path.exists(_vc_local):
                    value_codes = set(pd.read_parquet(_vc_local)["code"].astype(str).tolist())
                elif _vc_gcs and artifacts.exists(_vc_gcs):
                    value_codes = set(pd.read_parquet(_vc_gcs)["code"].astype(str).tolist())
                else:
                    raise SystemExit(f"[fe] CATEGORIZED held-out: train value_codes not found ({_vc_local}); "
                                     f"run the training FE for {h} {yrs}yr first to persist them.")
            vb_codes = set()    # derangement is OFF in strict category-only mode
            print(f"[{h}] CATEGORIZED unit-robust value pooling: per-code z over {_st.shape[0]:,} value-bearing "
                  f"SNOMEDs -> {len(value_codes):,} categories carry pooled values ({_src})")
        else:
            value_codes = set(ev_vb.dropna(subset=["value_num"])["code"].unique())
            vb_codes = value_bearing_codes(ev_vb)    # strict (>= MIN_VALUE_FRAC) set — used by the lab-derangement family
            print(f"[{h}] per-code value FE on {len(value_codes):,} codes with any numeric value"
                  f"{' (TRAIN-only)' if (fit_split and tr_guids) else ''}; "
                  f"{len(vb_codes):,} strict value-bearing (>= {int(MIN_VALUE_FRAC*100)}%) for derangement")
        if fit_split:
            print(f"[{h}] split-first: stability-selection fit on {len(tr_guids):,}/{len(patients):,} train patients")
        # STRICT category-only FE: per-category families + category-structure (diversity/cross/temporal) +
        # patient DEMOGRAPHICS only. Turn OFF everything that uses non-map codes / full-stream volume:
        # comment, derangement, blood-ratios; and `global` runs in DEMOGRAPHICS-ONLY mode
        # (age/sex/ethnicity/age-bands — no utilisation-volume features).
        # moc_zero (the cohort data-cutoff anchor that occurrence_features uses for decay/recency/accel/
        # recent_ratio, build_features.py:259) is the LAST fit-per-cohort quantity — make it ZERO-FIT like the
        # z-stats / top_cats / value_codes. Recomputing the cohort min independently on train vs held-out is a
        # small constant train↔serve scale shift on those features, invisible to the same-cohort parity gate.
        # FIT on TRAIN + persist when fit_split; RELOAD when held-out/serve (hard-fail if absent, like the rest).
        moc_zero = ev_remaining["months"].min()
        if categorized:
            _mz_local = os.path.join(HERE, "output", C.artifact_subdir(h, yrs), f"moczero_categorized_{h}.parquet")
            _mz_gcs = (f"{C.GCS_ARTIFACTS.rstrip('/')}/2_FE/output/{C.artifact_subdir(h, yrs)}/moczero_categorized_{h}.parquet"
                       if getattr(C, "GCS_ARTIFACTS", "") else None)
            if fit_split and tr_guids:
                moc_zero = float(ev_remaining[_splits.clean_guid(ev_remaining["patient_guid"]).isin(tr_guids)]["months"].min())
                _mzdf = pd.DataFrame({"moc_zero": [moc_zero]})
                artifacts.atomic_write(_mz_local, lambda t: _mzdf.to_parquet(t, index=False))
                if _mz_gcs:
                    try:
                        artifacts.atomic_write(_mz_gcs, lambda t: _mzdf.to_parquet(t, index=False))
                    except Exception as _mze:
                        raise SystemExit(f"[{h}] FE ABORT: moczero GCS mirror FAILED ({_mze}) — refusing to report FE "
                                         f"success with a half-published sidecar set (held-out/serve on another VM would "
                                         f"hard-fail on zero-fit reload). Fix GCS write access and re-run.")
            else:
                if os.path.exists(_mz_local):
                    moc_zero = float(pd.read_parquet(_mz_local)["moc_zero"].iloc[0])
                elif _mz_gcs and artifacts.exists(_mz_gcs):
                    moc_zero = float(pd.read_parquet(_mz_gcs)["moc_zero"].iloc[0])
                else:
                    raise SystemExit(f"[fe] CATEGORIZED held-out: train moc_zero not found ({_mz_local}); "
                                     f"run the training FE for {h} {yrs}yr first to persist it.")
        top_cats = []
        if categorized and int(getattr(C, "CROSS_TOP_K", 0) or 0) >= 2:
            # top_cats (cross-category top-K) is ZERO-FIT like the z-stats: FIT on TRAIN + PERSIST when
            # fit_split, RELOAD when held-out/serve — so cooc/INT features use the SAME categories the model
            # was trained on (no train/serve skew from per-cohort recomputation).
            _tc_local = os.path.join(HERE, "output", C.artifact_subdir(h, yrs), f"topcats_categorized_{h}.parquet")
            _tc_gcs = (f"{C.GCS_ARTIFACTS.rstrip('/')}/2_FE/output/{C.artifact_subdir(h, yrs)}/topcats_categorized_{h}.parquet"
                       if getattr(C, "GCS_ARTIFACTS", "") else None)
            if fit_split and tr_guids:
                _prev = ev_vb.groupby("code")["patient_guid"].nunique().sort_values(ascending=False)
                # ADMIN_NOISE_FE=0: keep admin/process categories OUT of the cross-category top-K too, else
                # cooc_/seq_/INT_ features form on the 90%+-prevalent admin cats (which the per-category column
                # prune can't catch by name). Excluding here -> persisted top_cats -> serve stays consistent.
                if not getattr(C, "ADMIN_NOISE_FE", True):
                    _anp = os.path.join(HERE, "categorized_codelist", "admin_noise_categories.txt")
                    if not os.path.exists(_anp):   # FAIL LOUD: missing list would silently let admin into top-K
                        raise RuntimeError(f"[fe] ADMIN_NOISE_FE=0 but admin-noise list missing: {_anp}. "
                                           f"Refusing to build cross-category top-K including admin categories "
                                           f"(no silent fallback) — sync admin_noise_categories.txt.")
                    with open(_anp, encoding="utf-8") as _af:
                        _noiseset = {ln.strip() for ln in _af if ln.strip()}
                    _prev = _prev[~_prev.index.astype(str).isin(_noiseset)]
                top_cats = list(_prev.head(int(C.CROSS_TOP_K)).index)
                _tcdf = pd.DataFrame({"category": pd.Series(top_cats, dtype=str)})
                artifacts.atomic_write(_tc_local, lambda t: _tcdf.to_parquet(t, index=False))
                if _tc_gcs:
                    try:
                        artifacts.atomic_write(_tc_gcs, lambda t: _tcdf.to_parquet(t, index=False))
                    except Exception as _te:
                        raise SystemExit(f"[{h}] FE ABORT: topcats GCS mirror FAILED ({_te}) — refusing to report FE "
                                         f"success with a half-published sidecar set (held-out/serve on another VM would "
                                         f"hard-fail on zero-fit reload). Fix GCS write access and re-run.")
            else:
                # HELD-OUT / serve: REUSE saved TRAIN top_cats (never recompute per-cohort).
                if os.path.exists(_tc_local):
                    top_cats = pd.read_parquet(_tc_local)["category"].astype(str).tolist()
                elif _tc_gcs and artifacts.exists(_tc_gcs):
                    top_cats = pd.read_parquet(_tc_gcs)["category"].astype(str).tolist()
                else:
                    raise SystemExit(f"[fe] CATEGORIZED held-out: train top_cats not found ({_tc_local}); "
                                     f"run the training FE (or a fit_split pass) for {h} {yrs}yr first to persist them.")
        if schema_only:
            return {"ev_full": ev_full, "ev_remaining": ev_remaining, "roster": _roster,
                    "value_codes": value_codes, "vb_codes": vb_codes, "top_cats": top_cats, "moc_zero": moc_zero,
                    "ev_assert": ev_assert}
    else:
        ev_full = shared["ev_full"]; ev_remaining = shared["ev_remaining"]; _roster = shared["roster"]
        value_codes = shared["value_codes"]; vb_codes = shared["vb_codes"]
        top_cats = shared["top_cats"]; moc_zero = shared["moc_zero"]; tr_guids = set(); ev = ev_remaining
        ev_assert = shared.get("ev_assert")
        if patient_ids is not None:
            _pid = set(patient_ids)
            ev_full = ev_full[ev_full["patient_guid"].isin(_pid)]
            ev_remaining = ev_remaining[ev_remaining["patient_guid"].isin(_pid)]
            _roster = _roster[_roster["patient_guid"].isin(_pid)]
            if ev_assert is not None:
                ev_assert = ev_assert[ev_assert["patient_guid"].isin(_pid)]
        patients = _roster.set_index("patient_guid")[["cancer_class"]]
    fam = dict(FEATURE_FAMILIES)
    if categorized:
        _force_off = ["comment", "derangement"]
        if not getattr(C, "BLOOD_RATIOS_FE", False):   # Arm R: BLOOD_RATIOS_FE=1 keeps NLR/PLR/LMR/mGPS on in categorized
            _force_off.append("blood_ratios")
        for off in _force_off:
            fam[off] = False
    blocks = []
    # build-only-model-features: when keep_features is given (held-out / inference), restrict the
    # EXPENSIVE per-category WIDE families (occurrence/flags/age/value/bands/cumulative) to ONLY the
    # categories the model actually uses (~28x fewer). The cross-category / category-structure /
    # demographics families keep running on the FULL stream below — they aggregate ACROSS categories,
    # so restricting would change their values. The final matrix is reindexed to keep_features. This is
    # equivalent to the full build then column-select, but far cheaper (no ~290k-wide build/OOM).
    ev_wide = ev_remaining
    if keep_features is not None and categorized:
        _cats = sorted(set(ev_remaining["code"].dropna().astype(str).unique()), key=len, reverse=True)
        _needed = set(map(str, top_cats))                      # cross-category needs its top_cats
        for _f in keep_features:
            for _c in _cats:
                if _f.startswith(_c + "_") or _f == _c:
                    _needed.add(_c)        # add ALL prefix-matching categories (over-include is safe; never miss a feature's true category)
        ev_wide = ev_remaining[ev_remaining["code"].astype(str).isin(_needed)]
        print(f"[{h}]   keep-features: {len(list(keep_features)):,} model feats -> {len(_needed):,} categories "
              f"(wide families restricted; aggregates/cross/demographics stay FULL)")
    # --- generic PER-(code|category) families: WIDE families on ev_wide (restricted under keep-features) ---
    if fam["occurrence"]:
        print(f"[{h}]   occurrence/dynamics ...");  blocks.append(occurrence_features(ev_wide, moc_zero=moc_zero))
    if fam["flags"]:
        print(f"[{h}]   problem-list flags ...");   blocks.append(flag_features(ev_wide))
    if fam["age"] and getattr(C, "AGE_PROXY_FE", True):
        print(f"[{h}]   age-at-event ...");         blocks.append(age_features(ev_wide))
    elif fam["age"]:
        print(f"[{h}]   age-at-event ... SKIPPED (AGE_PROXY_FE=0: per-category age proxies pruned)")
    if fam["value"]:
        print(f"[{h}]   value/trend ...");          blocks.append(value_features(ev_wide, value_codes))
    if getattr(C, "MED_FE", True):
        print(f"[{h}]   medication duration / drug-variety ..."); blocks.append(med_features(ev_wide))
    _pmin = ev_remaining.groupby("patient_guid")["months"].min()   # per-patient anchor from FULL stream -> keep-features-safe relative bands/cumulative
    if fam["bands"]:
        relative = ANCHOR_MODE == "patient_last"
        print(f"[{h}]   per-time-band ({ANCHOR_MODE}) {TIME_BANDS} ...")
        blocks.append(band_features(ev_wide, TIME_BANDS, value_codes, relative, pmin=_pmin, moc_zero=moc_zero))
    if fam["cumulative"]:
        relative = ANCHOR_MODE == "patient_last"
        print(f"[{h}]   cumulative last-N {CUMULATIVE_WINDOWS} ...")
        blocks.append(cumulative_features(ev_wide, CUMULATIVE_WINDOWS, value_codes, relative, pmin=_pmin, moc_zero=moc_zero))
    # --- CATEGORIZED-only category-structure FE (recovers signal that pooling SNOMEDs loses) ---
    if categorized:
        print(f"[{h}]   category intra (diversity / count_share / first_months / value-extremes / entropy) ...")
        blocks.append(category_diversity_features(ev_remaining, z_extreme=getattr(C, "LAB_Z_EXTREME", 2.0)))
        K = int(getattr(C, "CROSS_TOP_K", 0) or 0)
        if K >= 2 and top_cats:
            print(f"[{h}]   cross-category co-occurrence(+strength) + ordering over top-{K} categories ...")
            blocks.append(cross_category_features(ev_remaining, top_cats))
            if getattr(C, "AGE_PROXY_FE", True):     # INT_<cat>_x_age are age proxies (top FP drivers) -> pruned with the rest under AGE_PROXY_FE=0
                print(f"[{h}]   category x age interactions (top-{K}) ...")
                _age = pd.to_numeric(ev_full.groupby("patient_guid")["age_at_anchor"].first(), errors="coerce")
                blocks.append(category_age_interactions(ev_remaining, top_cats, _age))
            else:
                print(f"[{h}]   category x age interactions ... SKIPPED (AGE_PROXY_FE=0)")
        print(f"[{h}]   inter-category temporal (recency rank + months-since-last + n_categories 6/12mo) ...")
        blocks.append(category_temporal_features(ev_remaining))

    # RED-FLAG BURDEN composite (config.REDFLAG_FE) — runs on the FULL categorized frame (ev_remaining),
    # NOT keep-restricted ev_wide, so it aggregates ALL red-flag categories even when the individual ones
    # aren't model features (else held-out/serve would compute 0 -> train/serve skew). Per-patient aggregate.
    if REDFLAG_FE:
        print(f"[{h}]   red-flag burden composite (non-smoker channel) ...")
        blocks.append(redflag_features(ev_remaining, moc_zero=moc_zero))

    if fam["global"]:
        print(f"[{h}]   {'patient demographics only (age/sex/ethnicity/age-bands)' if categorized else 'global cross-code aggregates (FULL stream)'} ...")
        # CATEGORIZED: demographics come from the full roster (incl. zero-window patients -> real age/sex/
        # ethnicity). Non-categorized volume features still use the windowed full stream (ev_full).
        blocks.append(global_features(_roster if categorized else ev_full, demographics_only=categorized))
    if fam["comment"]:
        print(f"[{h}]   problem-comment keyword/presence (FULL stream) ...")
        blocks.append(comment_features(ev_full))     # true comment burden, not just codelist events
    if fam["derangement"]:
        print(f"[{h}]   derangement + rate-trajectory (generic, no hardcode) ...")
        # CATEGORIZED: run on the RAW full stream (ev is category-remapped + z-pooled there, so per-lab
        # derangement must use ev_full's raw codes/values + the raw value-bearing set vb_codes).
        _de_ev = ev_full if categorized else ev
        blocks.append(derangement_features(_de_ev, ev_full, vb_codes))
    if fam["blood_ratios"]:
        print(f"[{h}]   blood ratios NLR/PLR/LMR/CRP-alb (lightly hardcoded analytes) ...")
        blocks.append(blood_ratio_features(ev_full))
    if getattr(C, "CLINICAL_FE", False):   # Arm C: absolute clinical-threshold flags on RAW values (full stream -> parity-safe)
        print(f"[{h}]   clinical-threshold flags (thrombocytosis / raised-CRP, raw values) ...")
        blocks.append(clinical_concept_features(ev_full))
    if getattr(C, "ASSERTION_FE", False) and ev_assert is not None:   # free-text assertion-split + duration/frequency per category
        print(f"[{h}]   assertion-split (absent/family counts) + duration/frequency stats per category ...")
        blocks.append(assertion_attribute_features(ev_assert))
    if WEIGHT_TREND_FE:   # native weight-trend 40-window block (bw_w*) — full stream (ev_full), parity-safe, leak-free
        print(f"[{h}]   weight-trend 40-window (bw_w00..bw_w39 + imputed_count, SNOMED {WEIGHT_TREND_CODE}) ...")
        blocks.append(weight_trend_features(ev_full))

    mat = patients.join(blocks, how="left")
    # DENSITY-PROXY prune (mirror of AGE_PROXY_FE): drop the pure utilisation-volume / lifetime-breadth
    # globals that drive the care-seeking density confound (Sens 45% sparse -> 95% rich) without adding
    # cancer-specific signal. Kept: recent-breadth (6/12mo), consult accel/rate/recency, problem counts,
    # and the density-safe lab z-scores. Reversible via DENSITY_PROXY_FE=1 (default ON).
    if not getattr(C, "DENSITY_PROXY_FE", True):
        _dens_cols = ["g_total_events", "g_distinct_codes", "g_distinct_obs_codes", "g_distinct_med_codes",
                      "g_value_measured", "g_consult_total", "g_n_categories", "g_category_entropy",
                      "g_category_gini", "g_category_span_months",
                      # recent-breadth + recency-of-ANY-contact: count ALL categories (incl admin) -> visit/
                      # density proxies (a frequent attender scores high regardless of clinical signal). The
                      # per-CLINICAL-category recency/accel already carry the prodromal timing.
                      "g_n_categories_6mo", "g_n_categories_12mo", "g_last_category_months"]
        _drop = [c for c in _dens_cols if c in mat.columns]
        mat = mat.drop(columns=_drop)
        print(f"[{h}]   DENSITY_PROXY_FE=0: pruned {len(_drop)} density-proxy global(s): {_drop}")
    # ADMIN-NOISE prune: drop ALL feature columns for the curated pure-process/clerical categories
    # (appointments, notes, consent, DNA, leaflets, billing, enhanced-services-admin) listed in
    # categorized_codelist/admin_noise_categories.txt — they carry no cancer signal and ride the
    # care-utilisation confound. Column "<category>_<family>" is dropped when its category is in the list.
    # Map rows are NOT deleted (codes kept for serve coverage); only the FEATURES are pruned. ADMIN_NOISE_FE=1
    # (default) keeps them; set =0 to prune. Reviewable/editable list file.
    if not getattr(C, "ADMIN_NOISE_FE", True):
        _an_path = os.path.join(HERE, "categorized_codelist", "admin_noise_categories.txt")
        if os.path.exists(_an_path):
            with open(_an_path, encoding="utf-8") as _af:
                _noise = [ln.strip() for ln in _af if ln.strip()]
            _pref = tuple(f"{c}_" for c in _noise)
            _drop = [c for c in mat.columns if c.startswith(_pref)]
            mat = mat.drop(columns=_drop)
            print(f"[{h}]   ADMIN_NOISE_FE=0: pruned {len(_drop)} feature col(s) across {len(_noise)} admin/process categories")
        elif keep_features is None:
            # TRAIN build: a missing list would SILENTLY train WITH admin noise. FAIL LOUD (no soft fallback).
            raise RuntimeError(f"[fe] ADMIN_NOISE_FE=0 but admin-noise list missing: {_an_path}. "
                               f"Refusing to train with un-pruned admin noise — sync admin_noise_categories.txt.")
        else:
            # SERVE/held-out: keep_features reindex is the source of truth for presence (admin cols aren't in
            # feature_names), so a missing list cannot introduce admin features. Warn, don't abort.
            print(f"[{h}]   ADMIN_NOISE_FE=0, list missing -> relying on keep_features reindex (serve-safe)")
    # CATEGORY PRUNE (config.CATEGORY_PRUNE_FE): drop ALL feature columns for the per-horizon prune-list
    # categories (admin/process + comorbidity-burden/frailty proxies, no lung-specific signal). Same
    # mechanism as ADMIN_NOISE but a much larger, data-driven, per-horizon list (de-bloats the ~93%-singleton
    # map + de-noises stability selection). List mirrors the map filename: <Astra|Nova>_<yr>yr_droplist.txt.
    # Map rows kept (serve coverage); only FEATURES pruned. FAIL-LOUD at train if the list is missing.
    if getattr(C, "CATEGORY_PRUNE_FE", False):
        _dl_path = os.path.join(HERE, "categorized_codelist",
                                C.categorized_name(h, yrs).replace("_categories.csv", "_droplist.txt"))
        if os.path.exists(_dl_path):
            with open(_dl_path, encoding="utf-8") as _df:
                _pcats = [ln.strip() for ln in _df if ln.strip()]
            _ppref = tuple(f"{c}_" for c in _pcats)
            _pdrop = [c for c in mat.columns if c.startswith(_ppref)]
            mat = mat.drop(columns=_pdrop)
            print(f"[{h}]   CATEGORY_PRUNE_FE=1: pruned {len(_pdrop)} feature col(s) across {len(_pcats)} "
                  f"low-value categories ({os.path.basename(_dl_path)})")
        elif keep_features is None:
            raise RuntimeError(f"[fe] CATEGORY_PRUNE_FE=1 but prune list missing: {_dl_path}. Refusing to "
                               f"train with the un-pruned (bloated/noisy) map — run build_category_droplist.py "
                               f"or sync the *_droplist.txt (no silent fallback).")
        else:
            print(f"[{h}]   CATEGORY_PRUNE_FE=1, list missing -> relying on keep_features reindex (serve-safe)")
    if keep_features is not None:
        _kf = [c for c in keep_features if c != "cancer_class"]
        # Note: a model feature absent here is LEGITIMATE when no patient in this cohort has the
        # underlying events (value/interaction features for sparse categories) -> reindex NaN -> the
        # model's TRAIN-median/0 impute handles it (identical to the full build). The restriction-miss
        # bug is caught by the full-vs-keep-features score gate (<1e-9), not by requiring every feature here.
        mat = mat.reindex(columns=_kf + ["cancer_class"])     # keep-features exactly the model's features (+ label), in order
    # 0-fill genuine-absence families (counts/flags/structure/one-hots); value/recency stay NaN -> imputed
    # downstream (factored into _zero_fill_genuine_absence).
    mat = _zero_fill_genuine_absence(mat)
    # Force features to plain float64 (never pandas nullable Int64): parquet round-trips integer
    # columns as Int64, and filling their NaNs with a float median later raises TypeError.
    feat_cols = [c for c in mat.columns if c != "cancer_class"]
    mat[feat_cols] = mat[feat_cols].astype("float64").replace([np.inf, -np.inf], np.nan)  # no inf -> scaler-safe
    mat["cancer_class"] = mat.pop("cancer_class").astype("int64")    # label last, kept integer 0/1
    return mat.reset_index(), tr_guids


def main():
    # Windows are selected, never hardcoded: explicit digits on argv (e.g. `build_features.py 12mo 10`),
    # else auto-discover the windows that have a category map for this horizon. Each (horizon, window)
    # writes to its OWN output/{Astra|Nova}/{yr}yr_OFF/ folder (artifact_subdir) — no folder is hardcoded.
    argv_years = [int(a) for a in sys.argv[1:] if a.isdigit()]
    for h in HORIZONS:
        windows = argv_years or (C.categorized_windows(h) if getattr(C, "CATEGORIZED", False) else [])
        if not windows:
            print(f"[{h}] no lookback window selected/found "
                  f"({'no category-map files in ' + C.CATEGORIZED_DIR if getattr(C,'CATEGORIZED',False) else 'pass a year, e.g. ' + h + ' 5'}) — skip")
            continue
        for yr in windows:
            out_dir = os.path.join(HERE, "output", C.artifact_subdir(h, yr))
            os.makedirs(out_dir, exist_ok=True)
            mat, _ = build(h, years=yr)
            out = os.path.join(out_dir, f"features_p005_{h}.parquet")
            artifacts.atomic_write(out, lambda t: mat.to_parquet(t, index=False))   # .tmp -> os.replace (no partial matrix)
            # (train guids are NOT persisted here — stability_select reads the CANONICAL split directly.)
            print(f"[{h} {yr}yr] wrote {mat.shape[0]:,} patients x {mat.shape[1]:,} cols -> {out}")
            print(f"      next: python stability_select.py {h}   (k-fold importance -> stable feature set)\n")


if __name__ == "__main__":
    main()
