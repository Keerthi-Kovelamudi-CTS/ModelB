
# ═══════════════════════════════════════════════════════════════
# LEUKAEMIA CANCER — 1:1 COHORT VARIANT
# ─────────────────────────────────────────────────────────────
# This folder was scaffolded from Prostate_2.0_1to1. The pipeline
# code (3_pipeline.py, modeling, explainability) is generic and
# works as-is. The cancer-specific bits below are PROSTATE
# PLACEHOLDERS — replace them before running:
#
#   ☐ OBS_CATEGORIES     — leukaemia symptom/observation categories
#   ☐ MED_CATEGORIES     — leukaemia medication categories
#   ☐ LAB_CATEGORIES     — leukaemia relevant labs
#   ☐ LAB_BAD_DIRECTION  — clinical direction-of-bad per lab
#   ☐ LAB_WORSENING_RULES— optional vectorised worsening rules
#   ☐ LAB_RANGES         — physiologic bounds for outlier filter
#   ☐ CLUSTER_DEFINITIONS— leukaemia symptom clusters
#   ☐ INTERACTION_PAIRS  — leukaemia-relevant interactions
#   ☐ DECAY_CATEGORIES   — categories to exp-decay
#   ☐ SYMPTOM_CATEGORIES — aggregate symptom set
#
# Also replace 4_cancer_features.py (currently PSA-specific).
# Sex filter: both (apply in the SQL WHERE clause).
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PIPELINE CONFIGURATION
# Single source of truth for all cancer-specific settings.
# To port to a new cancer: copy this file and update the values.
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

# ─── Identity ────────────────────────────────────────────────
CANCER_NAME = 'leukaemia'
DATA_PREFIX = 'leukaemia'          # CSV filename prefix: prostate_3mo_obs.csv
PREFIX = 'LEUK_'                 # Feature prefix for cancer-specific columns
LABEL_COL = 'LABEL'
WINDOWS = ["1mo", "3mo", "6mo", "12mo"]

# ─── Paths (auto-resolved from this file's location) ────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'
FE_RESULTS = SCRIPT_DIR / 'results' / '3_feature_engineering'
CLEANUP_RESULTS = SCRIPT_DIR / 'results' / '5_cleanup'
SANITY_RESULTS = SCRIPT_DIR / 'results' / '1_sanity_check'

# Whether to save intermediate CSVs (feature_matrix.csv, enhanced.csv, mega.csv)
# Set False for production (only saves final); True for debugging
SAVE_INTERMEDIATES = False


# ═══════════════════════════════════════════════════════════════
# OBSERVATION CATEGORIES (from actual prostate SQL data)
# ═══════════════════════════════════════════════════════════════
OBS_CATEGORIES = [
    'BLOOD_COUNT_ABNORMALITIES',  # PRIMARY signal — lymphocytosis, leucocytosis, anaemia, thrombocytopenia
    'B_SYMPTOMS',                 # constitutional: fatigue, fever, night sweats, weight loss
    'LYMPHADENOPATHY',            # KEY signal — swollen glands, splenomegaly, neck/groin masses
    'BLEEDING_BRUISING',          # thrombocytopenia / coagulopathy presentation
    'INFECTION',                  # recurrent / opportunistic infections (immunocompromise)
    'SKIN_LYMPHOPROLIFERATIVE',   # pruritus, rash (B-cell / mycosis fungoides patterns)
    'BONE_PAIN',                  # marrow infiltration / myeloma-like
    'GI_SYMPTOMS',                # bloating, abdominal mass, constitutional GI
    'IMAGING',                    # chest X-ray abnormal, US neck
    'LYMPH_NODE_PROCEDURE',       # lymph node biopsy / ACE level
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (from actual prostate SQL data)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'ANTIBIOTICS',
    'ANTIBIOTIC_IMMUNOCOMPROMISE',  # co-trimoxazole — PCP prophylaxis
    'ANTIVIRALS_SYSTEMIC',          # aciclovir — herpes/zoster prophylaxis
    'ANTIFUNGALS_ORAL_SYSTEMIC',    # fluconazole/itraconazole/nystatin
    'CORTICOSTEROIDS_SYSTEMIC',     # prednisolone — steroid as treatment proxy
    'IRON_SUPPLEMENTS',
    'ORAL_ULCER_INFECTION_TREATMENT',
    'TRANEXAMIC_ACID',
    'ANTIHISTAMINES_PRURITUS',
    'NSAIDS_BONE_PAIN',
    'ANTIEMETICS_GI',
    'ALLOPURINOL_URIC_ACID',        # tumour-lysis-syndrome-relevant
    'OTHER_PREDICTIVE_MED',
]
# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'KEY_BLOOD_TESTS',              # FBC, reticulocyte count, WBC, differential
    'HAEMATOLOGY_INVESTIGATIONS',   # LDH, B2M, paraproteins, immunoglobulins, ESR, plasma viscosity, CRP, clotting
    'AUTOIMMUNE_SCREEN',            # RF, ANA, autoantibodies, complement
    'BMI',  # body mass index — value trend (falling BMI flagged by worsening rule)
    'BODY_WEIGHT',  # body weight in kg — falling weight is a cancer red flag
    'IDEAL_BODY_WEIGHT',  # reference value; tracks weight goal recorded by clinician
    'SYSTOLIC_BP',  # systolic blood pressure (mmHg)
    'DIASTOLIC_BP',  # diastolic blood pressure (mmHg)
    'HEART_RATE',  # heart/pulse rate (bpm)
]
# ═══════════════════════════════════════════════════════════════
# LAB DIRECTION OF BADNESS
# Per-category sign convention for slope → "worsening" boolean.
# 'up' = higher values are clinically worse (slope > 0 = worsening)
# 'down' = lower values are worse (slope < 0 = worsening)
# Categories omitted → no _worsening flag emitted.
# ═══════════════════════════════════════════════════════════════
LAB_BAD_DIRECTION = {
    # Leukaemia labs are mixed-direction within each bucket — set to None and
    # rely on LAB_WORSENING_RULES below for term-specific rules where needed.
    'KEY_BLOOD_TESTS':            None,
    'HAEMATOLOGY_INVESTIGATIONS': None,
    'AUTOIMMUNE_SCREEN':          None,
    'BMI':         None,
    'BODY_WEIGHT':         None,
    'IDEAL_BODY_WEIGHT':         None,
    'SYSTOLIC_BP':         'up',
    'DIASTOLIC_BP':         'up',
    'HEART_RATE':         None,
}

# Optional pluggable worsening rules per category (overrides LAB_BAD_DIRECTION).
# Vectorised callable: rule(slope, first, last, mean, max_val, count) -> 0/1 Series.
LAB_WORSENING_RULES = {
    # NOTE: leukaemia labs are heterogeneous within categories. Useful rules
    # need term-level filtering (e.g., LDH rising AND >250 = bad). Until then
    # we leave this empty; the pipeline still computes slope and pct_change for
    # every lab term, so signals are preserved without binary worsening flags.
    # Body weight: ≥2 measurements AND falling slope (cachexia / unexplained weight loss).
    'BODY_WEIGHT': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) < 0)
    ).astype(int),
    # BMI: ≥2 measurements AND falling slope (red-flag drop, even from obese baseline).
    'BMI': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) < 0)
    ).astype(int),

}
# ═══════════════════════════════════════════════════════════════
# INVESTIGATION CATEGORIES (imaging / specialist tests)
# ═══════════════════════════════════════════════════════════════
INVESTIGATION_CATEGORIES = [
    'IMAGING',                      # chest X-ray, neck US
    'LYMPH_NODE_PROCEDURE',         # lymph node biopsy
    'HAEMATOLOGY_INVESTIGATIONS',   # full haem workup
    'AUTOIMMUNE_SCREEN',
]
# ═══════════════════════════════════════════════════════════════
# SYMPTOM CLUSTER DEFINITIONS
# Each entry: (group_name, [list of obs categories])
# Used by build_advanced_features() for symptom clustering
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    ('BLOOD_COUNT',        ['BLOOD_COUNT_ABNORMALITIES', 'KEY_BLOOD_TESTS']),
    ('LYMPHATIC',          ['LYMPHADENOPATHY', 'LYMPH_NODE_PROCEDURE']),
    ('CONSTITUTIONAL',     ['B_SYMPTOMS', 'BONE_PAIN']),
    ('BLEEDING',           ['BLEEDING_BRUISING']),
    ('INFECTIONS',         ['INFECTION']),  # recurrent infections = immunocompromise
    ('SKIN',               ['SKIN_LYMPHOPROLIFERATIVE']),
    ('GI',                 ['GI_SYMPTOMS']),
    ('INVESTIGATION',      ['IMAGING', 'HAEMATOLOGY_INVESTIGATIONS', 'AUTOIMMUNE_SCREEN', 'LYMPH_NODE_PROCEDURE']),
]
# Categories considered "symptoms" for aggregate features
SYMPTOM_CATEGORIES = [
    'B_SYMPTOMS', 'LYMPHADENOPATHY', 'BLEEDING_BRUISING',
    'INFECTION', 'SKIN_LYMPHOPROLIFERATIVE', 'BONE_PAIN', 'GI_SYMPTOMS',
]
# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# Each entry: (feature_name, col_A, col_B)
# Creates INT_{feature_name} = col_A * col_B
# Columns must exist as has_ever flags in the merged feature matrix
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # Classic leukaemia presentation patterns
    ('count_abn_plus_lymphadenopathy',
     'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever', 'OBS_LYMPHADENOPATHY_has_ever'),
    ('count_abn_plus_b_symptoms',
     'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever', 'OBS_B_SYMPTOMS_has_ever'),
    ('count_abn_plus_bleeding',
     'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever', 'OBS_BLEEDING_BRUISING_has_ever'),
    ('count_abn_plus_infection',
     'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever', 'OBS_INFECTION_has_ever'),
    ('infection_plus_repeat_antibiotics',
     'OBS_INFECTION_has_ever', 'MED_ANTIBIOTICS_has_ever'),
    ('lymphadenopathy_plus_imaging',
     'OBS_LYMPHADENOPATHY_has_ever', 'OBS_IMAGING_has_ever'),
    ('count_abn_plus_haem_investigation',
     'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever', 'OBS_HAEMATOLOGY_INVESTIGATIONS_has_ever'),
    ('pruritus_plus_count_abn',
     'OBS_SKIN_LYMPHOPROLIFERATIVE_has_ever', 'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever'),
    ('bone_pain_plus_count_abn',
     'OBS_BONE_PAIN_has_ever', 'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever'),
    ('lymph_node_biopsy_plus_lymphadenopathy',
     'OBS_LYMPH_NODE_PROCEDURE_has_ever', 'OBS_LYMPHADENOPATHY_has_ever'),
    ('immunocompromise_prophylaxis',
     'OBS_INFECTION_has_ever', 'MED_ANTIBIOTIC_IMMUNOCOMPROMISE_has_ever'),
    ('steroid_treatment_proxy',
     'OBS_B_SYMPTOMS_has_ever', 'MED_CORTICOSTEROIDS_SYSTEMIC_has_ever'),
    ('bruising_plus_tranexamic',
     'OBS_BLEEDING_BRUISING_has_ever', 'MED_TRANEXAMIC_ACID_has_ever'),
    ('uric_acid_plus_count_abn',
     'OBS_BLOOD_COUNT_ABNORMALITIES_has_ever', 'MED_ALLOPURINOL_URIC_ACID_has_ever'),
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['NSAIDS_BONE_PAIN']
GI_MED_CATEGORIES = ['ANTIEMETICS_GI']
REPEAT_PRESCRIPTION_CATEGORY = 'ANTIBIOTICS'  # recurrent infections need repeat prescriptions
STEROID_CATEGORY = 'CORTICOSTEROIDS_SYSTEMIC'  # actual systemic steroids in this cancerGI_MED_CATEGORIES = ['ANTICHOLINERGICS']

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# Categories whose events get exponential time-decay weighting
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'BLOOD_COUNT_ABNORMALITIES', 'B_SYMPTOMS', 'LYMPHADENOPATHY',
    'BLEEDING_BRUISING', 'INFECTION', 'BONE_PAIN', 'SKIN_LYMPHOPROLIFERATIVE',
]
# ═══════════════════════════════════════════════════════════════
# CROSS-DOMAIN INTERACTION CONFIG
# Named boolean combinations for CROSS_ features
# ═══════════════════════════════════════════════════════════════
# (built programmatically in pipeline.py using cluster/visit/inv features)

# ═══════════════════════════════════════════════════════════════
# LAB RANGES — physiologically plausible bounds
# Values outside range → VALUE set to NaN (row preserved)
# Used by clean_data.py
# Format: { category: { term_string: (low, high) } }
# ═══════════════════════════════════════════════════════════════
LAB_RANGES = {
    # TODO: leukaemia-specific lab term names + plausible bounds.
    'BMI': {
        'Body mass index': (10.0, 80.0),
    },
    'BODY_WEIGHT': {
        'Body weight': (20.0, 300.0),
    },
    'IDEAL_BODY_WEIGHT': {
        'Ideal body weight': (30.0, 150.0),
    },
    'SYSTOLIC_BP': {
        'Systolic arterial pressure': (50.0, 280.0),
    },
    'DIASTOLIC_BP': {
        'Diastolic arterial pressure': (20.0, 180.0),
    },
    'HEART_RATE': {
        'Heart rate': (25.0, 250.0),
        'Pulse rate': (25.0, 250.0),
    },
}
# ═══════════════════════════════════════════════════════════════
# MODELING CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SEEDS = [42]                     # single-seed pilot (multi-seed code still wired; just loops once)
N_SELECT_FEATURES = 265           # fallback global; window-specific dict below takes precedence
N_SELECT_FEATURES_PER_WINDOW = {  # tuned per window: dense signal early, sparse signal late
    '1mo':  300,
    '3mo':  275,
    '6mo':  250,
    '12mo': 225,
}
OPTUNA_TRIALS = 75
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

# ═══════════════════════════════════════════════════════════════
# FEATURE GROUP DISPLAY (for cleanup summary)
# (prefix, display_name)
# ═══════════════════════════════════════════════════════════════
FEATURE_GROUP_DISPLAY = [
    (f'{PREFIX}LAB_',     f'{CANCER_NAME.title()} lab thresholds'),
    (f'{PREFIX}DX_',      f'{CANCER_NAME.title()} diagnostic pathway'),
    (f'{PREFIX}TX_',      f'{CANCER_NAME.title()} treatment'),
    (f'{PREFIX}RF_',      f'{CANCER_NAME.title()} risk factors'),
    (f'{PREFIX}',         f'{CANCER_NAME.title()}-specific'),
    ('OBS_',              'Observation'),
    ('LAB_TRAJ_',         'Lab trajectory'),
    ('LABTERM_',          'Lab (per-term)'),
    ('LAB_',              'Lab (basic)'),
    ('INV_PATTERN_',      'Inv patterns'),
    ('INV_',              'Investigation'),
    ('AGG_',              'Aggregate'),
    ('TEMP_',             'Temporal'),
    ('MED_ESC_',          'Med escalation'),
    ('MED_AGG_',          'Med aggregate'),
    ('MEDCAT_',           'Med (per-cat)'),
    ('MEDREC_',           'Med recurrence'),
    ('MED_',              'Medication'),
    ('INT_',              'Interactions'),
    ('CROSS_',            'Cross-domain'),
    ('CLUSTER_',          'Symptom clusters'),
    ('VISIT_',            'Visit patterns'),
    ('TRAJ_',             'Trajectory'),
    ('MONTHLY_',          'Monthly bins'),
    ('ROLLING3M_',        'Rolling 3-month'),
    ('CAT_',              'Per-category granular'),
    ('PAIR_',             'Obs co-occurrence'),
    ('CMPAIR_',           'Clin-Med pairs'),
    ('SEQ_',              'Sequence'),
    ('RECUR_',            'Recurrence (obs)'),
    ('RATE_',             'Rates & ratios'),
    ('AGE_',              'Age features'),
    ('AGEX_',             'Age interactions'),
    ('ENTROPY_',          'Entropy'),
    ('GINI_',             'Gini'),
    ('DECAY_',            'Time-decay'),
]
