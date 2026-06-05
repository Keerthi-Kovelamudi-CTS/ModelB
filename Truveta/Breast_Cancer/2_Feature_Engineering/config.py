"""
Configuration for Truveta breast-cancer Feature Engineering — heavy-FE design.

Three layers of features, all driven by the curated category codelist:

    Layer A — Category aggregations (per patient × ~57 curated categories × 6 aggs)
              count / present / value_mean / duration_sum / recency_days / event_density
              → ~400 features

    Layer B — Breast-specific hand-engineered (only built where supporting codes exist)
              Currently: HRT thresholds, alcohol-heavy flag, postmenopausal-age proxy
              → ~5 features

    Layer C — Engagement / code-type-aware features (vocabulary breadth, density,
              recency, lab-orders-without-value). Strong signal not captured by
              any single category aggregation.
              → ~6 features

    Layer D — Cross-feature interactions (start lean — 3 pairs).
              → ~3 features

Total: ~414 features + 8 spine columns.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).resolve().parent
DATA_DIR          = SCRIPT_DIR / 'data'
RAW_DIR           = DATA_DIR / 'raw'           # event-level extract from BQ — {window}/ subdirs
FEATURES_DIR      = DATA_DIR / 'features'      # wide patient × feature matrix — {window}/ subdirs

# Curated codelist with category column. Columns: code_type, category, code_id, term.
CODELIST_PATH     = (SCRIPT_DIR.parent / 'codelists' / 'breast_curated_categories.tsv')

# ── Dual-horizon windows (matches EMIS Breast pipeline) ──────────────────────
WINDOWS           = ['1mo', '12mo']

# ── BQ sources (per window) ──────────────────────────────────────────────────
BQ_PROJECT        = 'prj-cts-ai-dev-sp'
COHORT_TABLES     = {
    '1mo':  'prj-cts-ai-dev-sp.truveta_gold.breast_cohort_events_1mo',
    '12mo': 'prj-cts-ai-dev-sp.truveta_gold.breast_cohort_events_12mo',
}

# ── Cohort thresholds applied at FE time ──────────────────────────────────────
# Approach B (matches EMIS Breast_2.0_1to1): keep EVERY cohort patient in the
# feature matrix, including those with no/few curated events. Sparse patients
# fall through the left-join as all-zero rows (the "__PLACEHOLDER__" analog) so
# the model trains on the "no curated signal → no cancer" boundary and does not
# go out-of-distribution on sparse patients at deployment.
#
# When KEEP_SPARSE_PATIENTS is True the MIN_OBS_EVENTS_PER_PATIENT eligibility
# filter is NOT applied (it is retained only for the Approach-A / reporting path).
KEEP_SPARSE_PATIENTS       = True
MIN_OBS_EVENTS_PER_PATIENT = 10

# ── Shared train/val/test split (consumed by 3_Modeling/_shared_split.py) ─────
SHARED_SPLIT_PATH       = SCRIPT_DIR.parent / '3_Modeling' / 'shared_split.json'
MIN_EVENTS_PER_WINDOW   = 5

# ── Layer G: lung-style temporal trend / worsening (per category) ─────────────
# Mirrors the lung pipeline (SNOMED_json_v3) and EMIS's SEQ_*/ACCEL_* features,
# which Truveta previously lacked because 0_extract.py aggregated event dates
# away. 0_extract.py now also emits per-(patient, code) event counts split into
# the OLDER half (h1) and RECENT half (h2) of the patient's OWN event timeline
# (midpoint of that patient's first..last event — lung-style, patient-relative,
# NOT a fixed calendar point). This means short-history patients still get a
# meaningful within-history trend instead of a false "all-recent ⇒ worsening".
# Layer G aggregates h1/h2 per category and emits trend features ONLY for the
# categories that mattered to the model (STRONG + MEDIUM tiers from
# codelists/category_importance_tiers.csv) — temporal resolution where it helps,
# without exploding the feature space.

TREND_CATEGORIES = [
    'Comorbidity - Mental Health', 'Screening - Mammography', 'Lab - Lipid Panel',
    'Comorbidity - Cardiovascular', 'Medication - Allergy / Respiratory',
    'Lifestyle - Alcohol Use', 'Medication - Antihypertensives', 'General Imaging',
    'Medication - Antibiotics', 'Lifestyle - Tobacco Use',
    'Medication - Supplements / Vitamins', 'Vital Signs / Anthropometrics',
    'Comorbidity - Respiratory / ENT', 'Lifestyle - SDOH', 'Lab - Cardiac Enzymes',
    'Medication - Cardiac', 'Comorbidity - Skin / Dermatology',
    'Medication - Mental Health / Sleep', 'Medication - Statins (Lipid Lowering)',
    'Lab - Coagulation', 'Symptoms - Pain', 'Medication - GI / Acid Suppression',
    'Lifestyle - Social Determinants of Health', 'Medication - Thyroid',
    'Symptoms - General', 'Symptoms - Breast Lump', 'Comorbidity - GI / GU',
    'Post-Acute Markers', 'Lifestyle - Substance Use',
    'Medication - Respiratory (Chronic)', 'History - Radiation Exposure',
]

# Per gated category, Layer G emits: __trend_h2_over_h1, __is_worsening, __recent_frac
TREND_AGGREGATIONS = ['trend_h2_over_h1', 'is_worsening', 'recent_frac']
TREND_FILL_VALUES  = {'trend_h2_over_h1': 0.0, 'is_worsening': 0, 'recent_frac': 0.0}

# ── Layer A: category aggregations ────────────────────────────────────────────
# Each (patient × category) emits these aggregations. Adding `recency_days` and
# `event_density` lifts the feature space from 4 to 6 aggs per category.
CATEGORY_AGGREGATIONS = [
    'count',         # total events
    'present',       # 0/1 binary
    'value_mean',    # mean numeric value (NaN where labs absent)
    'duration_sum',  # total duration days (mostly meds)
    'recency_days',  # days from anchor_date to most-recent event in category
    'event_density', # events per active month (count / active_months)
]

# Fills applied to each agg kind when patient has zero events in a category.
# Recency: a sentinel large number means "no event ever". value_mean stays NaN.
AGG_FILL_VALUES = {
    'count':         0,
    'present':       0,
    'duration_sum':  0,
    'event_density': 0.0,
    'recency_days':  9999,   # sentinel — "no event ever"
    # 'value_mean' intentionally not filled — NaN encodes "no measurement"
}

# ── Layer B: breast-specific hand-engineered features ─────────────────────────
# Defined as (feature_name, kind, params). The breast_features module
# resolves these against the joined (patient × category) data.
#
# Categories referenced here MUST exist in the curated TSV — if a category has
# no codes, the feature silently becomes all-zero (acceptable, but flagged in
# the run log).
HRT_CATEGORY = 'Medication - Hormone Therapy (HRT)'
ALCOHOL_CATEGORY = 'Lifestyle - Alcohol Use'

# 5 years in days (clinical threshold for "long-term HRT")
HRT_LONG_TERM_DAYS = 5 * 365

# ≥4 standard drinks/day → "heavy alcohol use" (NIAAA threshold for women)
ALCOHOL_HEAVY_DRINKS_PER_DAY = 4.0

# Postmenopausal age proxy (no menopause codes available; age ≥55 is the
# conventional EHR proxy when menopause status is unknown).
POSTMENOPAUSAL_AGE = 55

# High-risk age (NHS England breast screening starts at 50)
HIGH_RISK_AGE = 50

BREAST_FEATURES = [
    # Long-term HRT exposure — strong known risk factor.
    ('BRCA_HRT_DURATION_YEARS', 'category_duration_years',
        {'category': HRT_CATEGORY}),
    ('BRCA_HRT_LONG_TERM_5Y',   'category_duration_threshold',
        {'category': HRT_CATEGORY, 'threshold_days': HRT_LONG_TERM_DAYS}),

    # Heavy alcohol — IARC group-1 carcinogen for breast.
    ('BRCA_ALCOHOL_HEAVY',      'category_value_threshold',
        {'category': ALCOHOL_CATEGORY, 'threshold': ALCOHOL_HEAVY_DRINKS_PER_DAY}),

    # Age-band proxies (no menopause codes available).
    ('BRCA_AGE_GE_50',          'spine_threshold',
        {'column': 'patient_age', 'threshold': HIGH_RISK_AGE}),
    ('BRCA_POSTMENOPAUSAL_AGE', 'spine_threshold',
        {'column': 'patient_age', 'threshold': POSTMENOPAUSAL_AGE}),
]

# ── Layer C: engagement / code-type-aware features ────────────────────────────
# Computed once per patient from the long-format aggregates (not per-category).
# These capture the "high-engagement" signal that simple category counts miss.
ENGAGEMENT_FEATURES = [
    'ENG_N_DISTINCT_VOCABS',       # how many code_types (snomed/loinc/rxnorm/...) does this patient have?
    'ENG_N_DISTINCT_CATEGORIES',   # how many curated categories does this patient touch?
    'ENG_TOTAL_EVENTS',            # sum of count across all codes/categories
    'ENG_EVENT_DENSITY_OVERALL',   # total_events / months_of_coverage
    'ENG_RECENCY_MIN_DAYS',        # most-recent event across all categories
    'ENG_LAB_ORDERS_NO_VALUE',     # observation events with NULL value_mean (ordered but no result)
    'ENG_ACCEL_H2_OVER_H1',        # patient-level acceleration ratio (recent half / older half) — EMIS ACCEL parity
    'ENG_ACCEL_H2_MINUS_H1',       # patient-level net acceleration (recent − older events)
]

# ── Layer D: cross-feature interactions ───────────────────────────────────────
# Each interaction is (feature_name, left_feature, right_feature). The product
# is computed at the end of feature building, after Layers A/B/C are populated.
# Left/right feature names must match exactly the columns produced above.
#
# Keep this list short (≤5). More interactions = more multicollinearity and
# diminishing returns; ML models can learn most pairwise effects from raw
# Layer A/B features given enough data.
INTERACTION_PAIRS = [
    # ── Original 3 ─────────────────────────────────────────────────────
    ('INT_HRT_X_POSTMENO',                'BRCA_HRT_LONG_TERM_5Y', 'BRCA_POSTMENOPAUSAL_AGE'),
    ('INT_ALCOHOL_X_HRT',                 'BRCA_ALCOHOL_HEAVY',    'BRCA_HRT_LONG_TERM_5Y'),
    ('INT_ENGAGEMENT_X_AGE50',            'ENG_TOTAL_EVENTS',      'BRCA_AGE_GE_50'),

    # ── v6: Top-feature-driven additions (informed by v6 importances) ──
    # patient_age × top-importance Layer F features
    ('INT_AGE_X_RISK',                    'BRCA_AGE_GE_50',        'BRCA_E_RISK_score'),
    ('INT_AGE_X_LUMP',                    'BRCA_AGE_GE_50',        'BRCA_E_has_lump'),
    ('INT_AGE_X_FAMHX',                   'BRCA_AGE_GE_50',        'BRCA_E_has_famhx_breast'),
    ('INT_AGE_X_BREAST_INDICATORS',       'BRCA_AGE_GE_50',        'BRCA_E_breast_indicators'),
    # Risk-score × symptom presence (compound risk)
    ('INT_RISK_X_LUMP',                   'BRCA_E_RISK_score',     'BRCA_E_has_lump'),
    ('INT_RISK_X_FAMHX',                  'BRCA_E_RISK_score',     'BRCA_E_has_famhx_breast'),
    ('INT_RISK_X_MULTI_SX',               'BRCA_E_RISK_score',     'BRCA_E_multi_breast_sx'),
    # Family-history × hereditary-load
    ('INT_FAMHX_X_BRCA',                  'BRCA_E_has_famhx_breast', 'BRCA_E_has_brca_test'),
    ('INT_FAMHX_X_ATYP',                  'BRCA_E_has_famhx_breast', 'BRCA_E_has_atypical'),
    # Postmenopausal × symptom (asymmetric risk window)
    ('INT_POSTMENO_X_LUMP',               'BRCA_POSTMENOPAUSAL_AGE', 'BRCA_E_has_lump'),
    ('INT_POSTMENO_X_BREAST_INDICATORS',  'BRCA_POSTMENOPAUSAL_AGE', 'BRCA_E_breast_indicators'),
    # Hereditary × age (BRCA carrier high-risk window)
    ('INT_BRCA_X_AGE50',                  'BRCA_E_has_brca_test',  'BRCA_AGE_GE_50'),
    # Symptom burden × hereditary load
    ('INT_MULTI_SX_X_HEREDITARY',         'BRCA_E_multi_breast_sx', 'BRCA_E_strong_hereditary'),
]

# ── Cohort window ─────────────────────────────────────────────────────────────
# The cohort table provides anchor_date per patient. Recency is computed as
# (anchor_date - last_event_date) in days. Active months for density is the
# span between the first and last event in the cohort window, capped at the
# below upper bound.
COHORT_WINDOW_MONTHS = 15   # 15mo→1mo before anchor_date (matches Top_Snomed)

# ── Spine column contract (mirrors Prostate_Prod_Ready) ───────────────────────
IDENTIFIER_COLUMN    = 'patient_id'
DEMOGRAPHIC_COLUMNS  = ['sex', 'ethnicity', 'state_or_province', 'patient_age']
ANCHOR_COLUMN        = 'anchor_date'
TRAILING_COLUMNS     = ['label', 'cancer_id']  # label = cancer_class (1/0)

SPINE_COLUMNS = [IDENTIFIER_COLUMN, *DEMOGRAPHIC_COLUMNS, ANCHOR_COLUMN, *TRAILING_COLUMNS]


# ═══════════════════════════════════════════════════════════════
# MODELING CONFIGURATION (consumed by 3_Modeling/1_training.py)
# Mirrors EMIS Breast modeling config for parity.
# ═══════════════════════════════════════════════════════════════
CANCER_NAME = 'breast'
PREFIX      = 'BRCA_'
SEEDS = [42]
N_SELECT_FEATURES = 500
N_SELECT_FEATURES_PER_WINDOW = {
    '1mo':  500,
    '12mo': 500,
}
OPTUNA_TRIALS = 150
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.10
TEST_RATIO  = 0.15
