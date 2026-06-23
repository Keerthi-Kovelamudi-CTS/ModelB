"""Central configuration for the lung pipeline — the SINGLE source of settings.

Every setting has its real value as the default below; edit them right here. A real environment
variable still overrides any setting per-run (e.g. `TUNE=1 TOP_N=300 python run_v3.py`), and if an
optional `.env` file happens to sit next to this file it is loaded too — but NO `.env` is required:
this file alone is fully sufficient.

Usage (scripts add the V3 root to sys.path, then):
    import config as C
    C.TOP_N, C.RANDOM_STATE, C.FE_YEARS_BEFORE, ...
"""
import os
from pathlib import Path

# Optional: load a `.env` next to this file IF one exists (no-op otherwise). Not required — every
# value already has its real default below. Real env vars still win over both.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass


def _s(key, default):
    return os.environ.get(key, default)


def _i(key, default):
    return int(os.environ.get(key, default))


def _f(key, default):
    return float(os.environ.get(key, default))


def _b(key, default=False):
    return str(os.environ.get(key, str(default))).strip().lower() in ("1", "true", "yes", "on")


def _opt_i(key, default):
    """int or None (the literal 'none'/'' -> None). For e.g. TREND_MAX_MONTHS."""
    v = str(os.environ.get(key, default)).strip().lower()
    return None if v in ("none", "", "null") else int(v)


# --- GCP / data ---
GCP_PROJECT          = _s("GCP_PROJECT", "prj-cts-ai-dev-sp")
HORIZONS             = [h.strip() for h in _s("HORIZONS", "12mo,1mo").split(",") if h.strip()]
# Lookback windows are NOT hardcoded — the run selects them. FE_WINDOWS defaults to EMPTY, which makes
# run_v3 auto-discover the available windows per horizon from the category-map files actually present in
# 2_FE/categorized_codelist/ (config.categorized_windows). Pass `--windows 10 ...` to run a chosen subset,
# or place more {Astra|Nova}_{N}yr_categories.csv files to add windows. Each (horizon, window) flows to its
# own {Astra|Nova}/{N}yr_OFF/ output folder via artifact_subdir.
FE_WINDOWS           = [int(x) for x in _s("FE_WINDOWS", "").split(",") if x.strip()]   # "" -> auto-discover from files
FE_YEARS_BEFORE      = _i("FE_YEARS_BEFORE", 0)        # single-step fallback only; 0 = "unset" (the run must pass a window)
SCORING_YEARS_BEFORE = _i("SCORING_YEARS_BEFORE", 0)  # (Phase-1 scoring is skipped in categorized mode)

# --- Phase 1a: code scoring (1_Top_Snomed/Combined Scoring) — only used in per-SNOMED scoring; CATEGORIZED
# skips Phase 1, so these are inert here but kept central so the per-SNOMED path has ONE source of truth
# (no duplicated literals in Combined Scoring) and every knob is logged in the run manifest via summary(). ---
NEG_POS_RATIO            = _i("NEG_POS_RATIO", 5)        # ML negative:positive downsampling (statistical ranking still sees the FULL cohort)
SCORING_TOP_N            = _i("SCORING_TOP_N", 150)      # # of top codes written to the scores CSV
SCORING_MAX_ML_FEATURES  = _i("SCORING_MAX_ML_FEATURES", 500)
SCORING_CV_FOLDS         = _i("SCORING_CV_FOLDS", 3)
MIN_PATIENTS_PER_FEATURE = _i("MIN_PATIENTS_PER_FEATURE", 3)
SCORING_COMBINED_WEIGHTS = {"stat_score": 0.30, "chi2_score": 0.20, "ml_importance": 0.50}
SCORING_ML_WEIGHTS       = {"lr": 0.25, "rf": 0.35, "gb": 0.40}

# --- Phase 1b: codelist construction ---
TOP_N          = _i("TOP_N", 500)
OR_MIN         = _f("OR_MIN", 2.0)
PREV_MIN       = _f("PREV_MIN", 0.01)
MIN_VALUE_FRAC = _f("MIN_VALUE_FRAC", 0.10)   # a code is "value-bearing" if >= this fraction of events carry a number.
# Lowered 0.30 -> 0.10 (2026-06-20): captures ~70 more lab codes (365 -> 435) so the model leans more on clinical
# VALUES and less on raw occurrence/counts. 0.05 was rejected (the extra codes carry values <10% of the time -> noisy).
# Data-driven pipeline: the codelist is the top-N scored codes only and FE is purely generic /
# category-structure / demographic. There is NO curated clinical-concept tier (removed).
# CATEGORIZED feature engineering: FE groups events by a hand-assigned CATEGORY rather than per individual
# SNOMED. The SNOMED->category map is read from 2_FE/categorized_codelist/{Astra|Nova}_{yr}yr_categories.csv
# (cols Code,Name,Category), one file per (horizon, lookback) the run selects. Feature columns become
# "<category>_<family>". STRICT category-only scope — every feature is list-based or a demographic:
#   per-category temporal + CATEGORY-POOLED value/trend (UNIT-ROBUST: each code's values z-scored vs its own
#   TRAIN distribution before pooling, so mixed-unit categories are fine) + within-category diversity
#   (n_distinct/ratio/count_share/first_months/n_extreme/max_abs_z) + cross/inter-category structure
#   (cooc/coocn/seq/INT_x_age/recency-rank/entropy/gini) + DEMOGRAPHICS ONLY (age_at_prediction/bands/sex/eth).
# OFF (not list-based / skew/leak risk): derangement, blood-ratios, full-stream
# utilisation-VOLUME (g_total_events/distinct-codes/consults/problems) and comment features.
# This is the pipeline's mode (leave ON).
CATEGORIZED    = _b("CATEGORIZED", True)
# CATEGORIZED-only extra FE depth (on top of the per-category temporal vocabulary):
#  * within-category diversity: <cat>_n_distinct_codes + <cat>_distinct_ratio (recovers structure that
#    pooling SNOMEDs into a category loses) + patient breadth g_n_categories. (always on in categorized)
#  * cross-category structure over the TOP-K categories by TRAIN prevalence (kept tractable): pairwise
#    co-occurrence cooc_<A>__<B> + temporal ordering seq_<A>__before__<B>. CROSS_TOP_K=0 disables it.
CROSS_TOP_K    = _i("CROSS_TOP_K", 25)
# Post-diagnosis LEAKAGE codes — excluded from ALL feature engineering (build_features filters them out of
# the event stream before mapping). These are lung-resection / surgical-excision SNOMEDs that only occur
# AFTER a confirmed cancer diagnosis, so they are NOT available at a genuine pre-diagnosis prediction point
# and would dishonestly inflate metrics (and were partly mis-mapped into "Skin lesion removal"). 2026-06-20.
LEAKY_CODES    = frozenset({
    173171007,          # Lobectomy of lung
    359615001,          # Partial lobectomy of lung
    173170008,          # Bilobectomy of lung
    1773021000006100,   # Total lobectomy of left upper lobe
    1773041000006110,   # Total lobectomy of right middle lobe
    1773051000006110,   # Total lobectomy of right upper lobe
    49795001,           # Total pneumonectomy
    119746007,          # Lung excision
    232639000,          # Excision of segment of left lower lobe
    232638008,          # Excision of segment of left upper lobe
    173172000,          # Excision of segment of lung
    232637003,          # Excision of segment of right lower lobe
    232635006,          # Excision of segment of right upper lobe
})
# NOTE (2026-06-20): post-diagnosis TREATMENT categories (oncology / chemo / radio / palliative) are NOT
# excluded. The cohort SQL enforces a strict gap (feature window ends 12mo/1mo BEFORE diagnosis), so these
# post-dx codes are structurally outside the window for genuine cases; and at 1mo (Nova diagnostic tool)
# the pre-dx oncology-WORKUP signals are legitimate and predictive. Only the surgical-resection SNOMEDs above
# (LEAKY_CODES) are dropped — they were also mis-mapped (lung excision -> "Skin lesion removal"), and this
# removes that pollution while leaving the genuine skin-lesion codes in that category intact.
# Per-category AGE features (<cat>_age_first/_age_last/_age_median) are age PROXIES — they re-encode the
# patient's age at routine events and (per the FP SHAP dig, 2026-06) fire for elderly non-cancer patients,
# hurting specificity without adding cancer-specific signal. Default ON = retrain-2 behaviour; set AGE_PROXY_FE=0
# to PRUNE the whole per-category age family AND the INT_<cat>_x_age interactions (both are age proxies / top
# FP drivers). The global age_at_prediction + age-bands are kept regardless — age once, not 20 times.
AGE_PROXY_FE   = _b("AGE_PROXY_FE", False)   # lung-clean prod default 2026-06-20
# Record-DENSITY proxies: pure utilisation-volume / lifetime-breadth globals (g_total_events,
# g_distinct_*_codes, g_value_measured, g_consult_total, g_n_categories, g_category_entropy/gini,
# g_category_span_months). Per the segment dig (2026-06) these drive the care-seeking density confound
# (Sens 45% sparse -> 95% rich) without adding cancer-specific signal. Default ON = current behaviour;
# set DENSITY_PROXY_FE=0 to PRUNE them (the model then leans on clinical content + density-safe lab
# z-scores). Recent-breadth (g_n_categories_6mo/_12mo), consult ACCEL/rate/recency and active/significant
# problem counts are KEPT (temporal/clinical, not pure volume).
DENSITY_PROXY_FE = _b("DENSITY_PROXY_FE", False)   # lung-clean prod default 2026-06-20
# ADMIN/PROCESS noise: ~99 curated pure-clerical categories (appointments, notes, consent, DNA, leaflets,
# billing, enhanced-services-admin) in categorized_codelist/admin_noise_categories.txt — no cancer signal,
# ride the care-utilisation confound (same family as the age/density proxies). Default ON keeps them; set
# ADMIN_NOISE_FE=0 to PRUNE every feature column for those categories (map rows kept for serve coverage).
# The list file is reviewable/editable; clinical "administration of …"/oxygen/medication/specialty-referral/
# disease-monitoring categories are deliberately EXCLUDED from it (kept).
ADMIN_NOISE_FE = _b("ADMIN_NOISE_FE", False)   # lung-clean prod default 2026-06-20
# CHRONICITY features (additive FE family; default OFF). Per-category tenure x recent-activity interaction
# to separate LONG-STANDING STABLE disease (benign — raises TN / cuts the elderly-COPD false positives)
# from NEW-ONSET or RE-ESCALATING disease (the cancer signal). Emits per category: _onset_months (tenure),
# _new_onset (first occurrence within CHRONICITY_NEW_ONSET_MO), _chronic_stable (tenure >= CHRONICITY_TENURE_MO
# AND no event within CHRONICITY_RECENT_MO = old+quiet), _escalating_chronic (chronic AND recent event AND
# accel>0 = old+flaring). Per-patient deterministic (event timing vs the cohort cutoff) -> LEAK-SAFE, no
# train-fit. Set CHRONICITY_FE=1 to enable. Months thresholds below are FE-internal but config-overridable.
CHRONICITY_FE        = _b("CHRONICITY_FE", False)
# RED-FLAG BURDEN (additive FE family; default OFF). The non-smoker lung-cancer detection channel: the
# model detects elderly cancer almost only via smoking->COPD->respiratory, and MISSES never-smoker/non-COPD
# cancers. Their signals (haemoptysis, weight-loss, hoarseness, finger-clubbing, anaemia, chest-pain,
# night-sweats, appetite-loss, dysphagia, pleural-effusion) EXIST + are cancer-enriched but each is too
# sparse to survive stability selection -> pruned -> invisible. REDFLAG_FE=1 emits a COMPOSITE per-patient
# burden (g_redflag_n_distinct / _n_events / _present / _recent6_distinct / _recent12_distinct / _decay /
# _accel) over the curated red-flag category set (build_features.REDFLAG_CATEGORIES) — high enough coverage
# (~30-40% of elderly cancers) to survive. Stability-select also FORCE-KEEPS the individual red-flag
# categories + the lab-side value categories (Hb / platelets / CRP / Weight, for the anaemia/paraneoplastic/
# weight-loss value-trends). Per-patient deterministic -> LEAK-SAFE.
REDFLAG_FE           = _b("REDFLAG_FE", False)
# CATEGORY PRUNE (default OFF). The map is ~93% ungrouped singletons — mostly admin/process/clerical +
# comorbidity-burden/frailty proxies (the same density confound) with NO lung-specific signal. They bloat FE
# ~6x and inject noise into stability selection. CATEGORY_PRUNE_FE=1 drops every feature column for the
# categories in the per-horizon prune list (2_FE/categorized_codelist/{Astra|Nova}_{yr}yr_droplist.txt,
# generated by build_category_droplist.py: KEEP = model-used OR clinically-lung-relevant OR discriminates
# with >=30 patients; DROP = everything else). Map rows are NOT deleted (codes kept for serve coverage);
# only FEATURES are pruned. FAIL-LOUD at train if the list is missing. Reviewable CSVs alongside.
CATEGORY_PRUNE_FE    = _b("CATEGORY_PRUNE_FE", False)
CHRONICITY_NEW_ONSET_MO = _f("CHRONICITY_NEW_ONSET_MO", 18.0)   # first occurrence within this -> "new onset"
CHRONICITY_TENURE_MO    = _f("CHRONICITY_TENURE_MO", 36.0)      # tenure (onset) >= this -> "chronic"
CHRONICITY_RECENT_MO    = _f("CHRONICITY_RECENT_MO", 12.0)      # most-recent event within this -> "recently active"
# Operating-point refinement at INFERENCE (read by predict_unseen via env so the deploy script stays
# config-free): 'age' (default; per-age-band cut — recovers elderly specificity), 'density' (per-record-
# density-quartile cut), or 'none' (single global cut). evaluate_heldout writes BOTH artifacts regardless;
# this only selects which one predict_unseen applies. NOT stacked (joint age x density needs joint calib).
THRESHOLD_REFINE = _s("THRESHOLD_REFINE", "age")
# Blood-ratio features (NLR / PLR / LMR / CRP-albumin mGPS + ratio trend slopes) computed on the RAW full
# stream from hardcoded analyte codes — validated systemic-inflammation cancer markers. Force-disabled in
# CATEGORIZED mode by default (historical: skew/leak caution). Set BLOOD_RATIOS_FE=1 to re-enable and test.
BLOOD_RATIOS_FE = _b("BLOOD_RATIOS_FE", False)  # OFF — feature-importance audit (12mo, 2026-06-20): DEAD (0.44% total imp, 5 feats). Set =1 to re-test.
# Arm C — absolute clinical-threshold lab flags on RAW values (NOT z-scored), the one thing the unit-robust
# per-code z-scoring structurally cannot express ("clinically abnormal" vs "high for this patient"). Lung-relevant
# paraneoplastic/inflammatory markers validated against this cohort's value distributions (2026-06-20):
# thrombocytosis (platelets >400 ×10^9/L; p95=408) and raised CRP (>10 mg/L; med 4). Hb/anaemia omitted —
# Hb values are absent/mixed-unit here (anaemia is already captured as the "Anaemia" diagnosis category).
CLINICAL_FE        = _b("CLINICAL_FE", False)  # OFF — feature-importance audit (12mo, 2026-06-20): DEAD (0.32% total imp, 3 feats). Set =1 to re-test.
THROMBOCYTOSIS_PLT = _f("THROMBOCYTOSIS_PLT", 400.0)   # platelet count upper-normal (×10^9/L; cohort p95=408)
RAISED_CRP_MGL     = _f("RAISED_CRP_MGL", 10.0)        # CRP upper-normal (mg/L; cohort med=4)
HYPOALBUMINEMIA_GL = _f("HYPOALBUMINEMIA_GL", 35.0)    # serum albumin lower-normal (g/L; cohort p5=35) — cachexia/cancer marker,
#                                                        density-INDEPENDENT signal (one low reading = signal, regardless of visit count).
#                                                        (LDH considered but dropped: only ~120 values in the cohort — too sparse.)
# AGE-OFFSET (de-confound by reweighting). The model's discrimination collapses to AGE: global AUROC 0.93
# is inflated by easy young-vs-old separation while WITHIN-age AUROC is only ~0.76-0.83, and ~97% of FPs
# are age-driven. AGE_OFFSET_FE=1 reweights the TRAIN rows with inverse-propensity weights that make
# AGE-BAND ⊥ class (within each class the band distribution is matched to the marginal P(band)) — so age
# carries ~no marginal class information and EVERY learner is forced onto the within-age clinical signal.
# Age's legitimate epidemiological PRIOR is re-added downstream by the per-age-band isotonic calibration
# (calibrate(by_age_band=True)) + the per-age operating threshold, so calibrated probabilities stay
# prevalence/age-correct. MODELING-only (does NOT change feature columns -> no FE rebuild; gates the model
# stage only). Reversible; default OFF. The ensemble-compatible equivalent of a base_margin age offset
# (a literal init_score offset is XGB/LGBM-only and would break the soft-voting tree ensemble).
AGE_OFFSET_FE  = _b("AGE_OFFSET_FE", False)
# Candidate models to EXCLUDE from training/selection/ensemble. AdaBoost is demoted off by default: it's
# competitive but rarely the single best, and it's the ONLY non-tree-exact model. SHAP is now TREE-EXACT
# ONLY (explainability._shap_values; KernelExplainer removed), so an AdaBoost member would make SHAP RAISE
# and poison the fast exact per-tree path of any soft-voting ensemble it joins. Keeping it excluded makes
# every deployable model (single or ensemble) tree-exact -> fast, exact explainability on ALL patients.
EXCLUDE_MODELS = [m.strip() for m in _s("EXCLUDE_MODELS", "AdaBoost").split(",") if m.strip()]
# Output naming: Astra = 12mo horizon, Nova = 1mo horizon. The fixed "OFF" suffix below is the artifact
# namespace token (data-driven pipeline) kept for stable GCS/local artifact paths — see _OUTPUT_TAG.
# Artifacts land under 3_Modeling_outputs/{Astra|Nova}/{yr}yr_OFF/ ; the SNOMED->category maps live in
# 2_FE/categorized_codelist/{Astra|Nova}_{yr}yr_categories.csv.
HORIZON_LABELS = {"12mo": "Astra", "1mo": "Nova"}
_OUTPUT_TAG    = "OFF"   # fixed artifact-path namespace token (data-driven); retained for path stability


def horizon_label(h):
    """Friendly horizon name for outputs (Astra/Nova); falls back to the raw key."""
    return HORIZON_LABELS.get(h, h)


def artifact_subdir(h, yr):
    """Per-(horizon, lookback) output subfolder, e.g. 'Astra/5yr_OFF'."""
    return f"{horizon_label(h)}/{yr}yr_{_OUTPUT_TAG}"


def codelist_name(h, yr):
    """Codelist filename, e.g. 'Astra_5yr_OFF_codelist.csv'."""
    return f"{horizon_label(h)}_{yr}yr_{_OUTPUT_TAG}_codelist.csv"


def categorized_name(h, yr):
    """Hand-curated SNOMED->category mapping filename (CATEGORIZED mode), e.g. 'Astra_10yr_categories.csv'.
    Lives in 2_FE/categorized_codelist/. Columns: Code, Name, Category."""
    return f"{horizon_label(h)}_{yr}yr_categories.csv"


# Directory holding the per-(horizon, lookback) category maps (resolved relative to this config file).
CATEGORIZED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_FE", "categorized_codelist")


def categorized_windows(h):
    """Auto-discover which lookback windows (years) have a category map for horizon `h`, by scanning
    CATEGORIZED_DIR for '{label}_{N}yr_categories.csv'. Returns a sorted list of ints (e.g. [10]).
    No year is hardcoded anywhere — the available windows are whatever files you've placed in the folder."""
    import re, glob
    label = horizon_label(h)
    pat = re.compile(rf"^{re.escape(label)}_(\d+)yr_categories\.csv$")
    yrs = []
    for p in glob.glob(os.path.join(CATEGORIZED_DIR, f"{label}_*yr_categories.csv")):
        m = pat.match(os.path.basename(p))
        if m:
            yrs.append(int(m.group(1)))
    return sorted(yrs)

# --- Phase 2b: stability selection (slightly INCLUSIVE in this variant: keep features that are consistent
# in >=2 of 5 folds; CUM_IMP at the validated 0.99 default — the 0.99->0.999 sliver is near-noise) ---
N_FOLDS   = _i("N_FOLDS", 5)
MIN_FOLDS = _i("MIN_FOLDS", 2)     # keep a feature selected in >=2 of 5 folds (was 3) -> less pruning, main inclusivity lever
CUM_IMP   = _f("CUM_IMP", 0.99)    # 99% cumulative importance (validated default; matches V3_snomed)

# --- Phase 2: feature engineering ---
FE_ENGINE        = _s("FE_ENGINE", "pandas")   # pandas is the ONLY engine in this variant — the Polars backend was removed (it never supported CATEGORIZED mode). Kept as a fixed "pandas" knob for the cache-key/provenance.
ANCHOR_MODE      = _s("ANCHOR_MODE", "patient_last")
TREND_MAX_MONTHS = _opt_i("TREND_MAX_MONTHS", "none")
DECAY_TAU_MONTHS = _f("DECAY_TAU_MONTHS", 12.0)
LAB_Z_EXTREME    = _f("LAB_Z_EXTREME", 2.0)
# Full-vs-keep-features FE parity gate: after each train build, prove the keep-features build (held-out /
# inference) is byte-identical to the full build for the model's features. HARD-FAILS the run on mismatch.
FE_PARITY_GATE   = _b("FE_PARITY_GATE", True)
# --- Phase 3: modeling ---
RANDOM_STATE    = _i("RANDOM_STATE", 42)
TEST_SIZE       = _f("TEST_SIZE", 0.10)
CALIB_SIZE      = _f("CALIB_SIZE", 0.10)
DROP_THRESHOLD  = _f("DROP_THRESHOLD", 0.90)
TUNE            = _b("TUNE", False)
TUNE_TOP_N      = _i("TUNE_TOP_N", 5)
N_TUNING_TRIALS = _i("N_TUNING_TRIALS", 100)
TUNING_CV_FOLDS = _i("TUNING_CV_FOLDS", 5)

# --- Phase 4: held-out evaluation ---
HELDOUT_CHUNK = _i("HELDOUT_CHUNK", 0)

# --- canonical split (split-first; created once by make_split.py, loaded by every step) ---
# SPLIT_SEED is INTENTIONALLY decoupled from RANDOM_STATE: the canonical train/valid/test partition must
# stay STABLE across model-seed experiments (changing RANDOM_STATE must NOT reshuffle which patients are
# in test). Change SPLIT_SEED only when you deliberately want a different cohort partition (rare).
SPLIT_SEED      = _i("SPLIT_SEED", 42)
# Shared GCS location so the split is identical for everyone, every run. Point these at a local dir
# (e.g. "splits") and set GCS_ROOT="" to run fully local with no GCS.
SPLIT_DIR       = _s("SPLIT_DIR", "gs://gcs-ai-dev-model-artifacts/keerthi/Lung_Cancer/splits")
USE_SAVED_SPLIT = _b("USE_SAVED_SPLIT", True)   # True = load the saved canonical split everywhere
# GCS_ROOT = COMMON root shared by BOTH pipeline variants (V3_snomed / V3_categorized): the canonical
# splits/ and the raw_events/ cache live here and are reused across both. ("" = local only.)
GCS_ROOT        = _s("GCS_ROOT", "gs://gcs-ai-dev-model-artifacts/keerthi/Lung_Cancer")
# GCS_VARIANT = this pipeline's own subfolder under GCS_ROOT for its PER-VARIANT artifacts (2_FE +
# 3_Modeling_outputs). This is the CATEGORIZED variant -> "V3_categorized".
GCS_VARIANT     = _s("GCS_VARIANT", "V3_categorized")
# GCS_ARTIFACTS = where 2_FE + 3_Modeling_outputs are mirrored (variant-scoped). splits + raw_events
# deliberately stay at GCS_ROOT (common), NOT here.
GCS_ARTIFACTS   = (GCS_ROOT.rstrip("/") + "/" + GCS_VARIANT) if GCS_ROOT else ""


def summary():
    """One-line-per-setting dump (handy at the top of a run for the log)."""
    keys = [k for k in globals() if k.isupper()]
    return "\n".join(f"  {k} = {globals()[k]!r}" for k in keys)


def as_dict():
    """All UPPER-CASE settings as a JSON-able dict — for the run-provenance manifest."""
    out = {}
    for k in globals():
        if k.isupper():
            v = globals()[k]
            out[k] = v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else repr(v)
    return out


def horizons_from_argv(argv):
    """Horizons requested on the command line, HARD-ERRORING on an unrecognized token — so a typo like
    `1m` fails loudly instead of silently defaulting to all horizons. Digit tokens (year lookbacks) and
    flags (leading '-') are ignored. Returns the requested horizons, or all HORIZONS if none given."""
    bad = [a for a in argv if not a.isdigit() and not str(a).startswith("-") and a not in HORIZONS]
    if bad:
        raise SystemExit(f"[config] unrecognized argument(s) {bad}; valid horizons are {list(HORIZONS)} "
                         f"(year lookbacks are digits; flags start with '-').")
    return [a for a in argv if a in HORIZONS] or list(HORIZONS)


if __name__ == "__main__":
    print("V3 config:\n" + summary())
