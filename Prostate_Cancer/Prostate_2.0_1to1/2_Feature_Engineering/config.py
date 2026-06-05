# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PIPELINE CONFIGURATION
# Single source of truth for all cancer-specific settings.
# To port to a new cancer: copy this file and update the values.
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

# ─── Identity ────────────────────────────────────────────────
CANCER_NAME = 'prostate'
DATA_PREFIX = 'prostate'          # CSV filename prefix: prostate_3mo_obs.csv
PREFIX = 'PROST_'                 # Feature prefix for cancer-specific columns
LABEL_COL = 'LABEL'
WINDOWS = ["1mo", "3mo", "6mo", "12mo"]

# ─── Paths (auto-resolved from this file's location) ────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'
_RESULTS_ROOT   = SCRIPT_DIR / 'results'
FE_RESULTS      = _RESULTS_ROOT / '3_feature_engineering'
CLEANUP_RESULTS = _RESULTS_ROOT / '5_cleanup'
SANITY_RESULTS  = _RESULTS_ROOT / '1_sanity_check'

# ─── Codelist mapping JSON path (consumed by 0_preprocess_to_fe.py) ─────
CODELIST_MAPPING_JSON = SCRIPT_DIR.parent / 'codelists' / 'code_category_mapping_v2.json'

# Whether to save intermediate CSVs (feature_matrix.csv, enhanced.csv, mega.csv)
# Set False for production (only saves final); True for debugging
SAVE_INTERMEDIATES = False


# ═══════════════════════════════════════════════════════════════
# OBSERVATION CATEGORIES
# ═══════════════════════════════════════════════════════════════
OBS_CATEGORIES = [
    'ALBUMIN_PROTEIN',
    'ALP_BONE_MARKER',
    'BONE_MUSCLE',
    'CALCIUM',
    'CATHETER',
    'COAGULATION',
    'CONSTITUTIONAL',
    'CYSTOSCOPY',
    'DRE',
    'ELECTROLYTES',
    'ERECTILE_DYSFUNCTION',
    'EXAMINATION',
    'FAMILY_HISTORY',
    'FBC_HAEMATOLOGY',
    'HAEMATOLOGY',
    'HAEMATURIA',
    'HORMONAL',
    'INFLAMMATORY',
    'IPSS',
    'LAB_FLAGS',
    'LIVER_FUNCTION',
    'LUTS',
    'PAIN_PELVIC_BONE',
    'PROSTATIC_CONDITIONS',
    'PSA',
    'PSA_FREE',
    'PSA_RATIO',
    'RENAL_FUNCTION',
    'SCREENING_PATHWAY',
    'SEXUAL_REPRODUCTIVE',
    'SMOKING',
    'URINARY_RETENTION',
    'URINE_MARKERS',
    'UROLOGY_IMAGING',
    'UROLOGY_PATHWAY',
    'UTI',
    'VITAMIN_D',
    'WEIGHT',
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    '5ARI',                  # finasteride, dutasteride
    'ALPHA_BLOCKERS',        # tamsulosin, alfuzosin
    'ANTICHOLINERGICS',      # overactive bladder
    'ANTICOAGULANT',
    'CATHETER_SUPPLIES',
    'ED_MEDICATIONS',        # erectile dysfunction
    'PAIN_ESCALATION',
    'UTI_ANTIBIOTICS',       # UTI-specific antibiotics
    'UTI_TREATMENT',         # other UTI treatments
]

# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'PSA',
    'PSA_FREE',     # free PSA (lower in cancer relative to total)
    'PSA_RATIO',    # free:total PSA ratio (low ratio suggests cancer)
    'RENAL_FUNCTION',
    'ELECTROLYTES',
    'HAEMATOLOGY',
    'FBC_HAEMATOLOGY',
    'ALP_BONE_MARKER',
    'ALBUMIN_PROTEIN',
    'LIVER_FUNCTION',
    'CALCIUM',
    'COAGULATION',
    'HORMONAL',
    'INFLAMMATORY',
    'URINE_MARKERS',
    'VITAMIN_D',
    'BMI',
    'BODY_WEIGHT',
    'IDEAL_BODY_WEIGHT',
    'SYSTOLIC_BP',
    'DIASTOLIC_BP',
    'HEART_RATE',
]

# ═══════════════════════════════════════════════════════════════
# LAB DIRECTION OF BADNESS
# Per-category sign convention for slope → "worsening" boolean.
# 'up' = higher values are clinically worse (slope > 0 = worsening)
# 'down' = lower values are worse (slope < 0 = worsening)
# Categories omitted → no _worsening flag emitted.
# ═══════════════════════════════════════════════════════════════
LAB_BAD_DIRECTION = {
    'PSA':             'up',
    'PSA_FREE':        'down',   # cancer makes free-PSA fraction shrink
    'PSA_RATIO':       'down',   # low free:total ratio is cancer-like
    'ALP_BONE_MARKER': 'up',
    'RENAL_FUNCTION':  'down',   # eGFR low = bad
    'FBC_HAEMATOLOGY': 'down',   # Hb low = bad
    'HAEMATOLOGY':     'up',     # CRP high = bad
    'ALBUMIN_PROTEIN': 'down',   # low albumin = advanced disease
    'LIVER_FUNCTION':  'up',     # bilirubin/ALT high = bad
    'CALCIUM':         'up',     # hypercalcaemia = mets marker
    'ELECTROLYTES':    None,     # both directions matter → skip
    'COAGULATION':     None,     # context-dependent
    'HORMONAL':        'down',   # low testosterone = treatment failure / advanced
    'URINE_MARKERS':   None,
    'VITAMIN_D':       'down',
    'BMI':             None,
    'BODY_WEIGHT':     None,
    'IDEAL_BODY_WEIGHT': None,
    'SYSTOLIC_BP':     'up',
    'DIASTOLIC_BP':    'up',
    'HEART_RATE':      None,
}

# Optional pluggable worsening rules per category (overrides LAB_BAD_DIRECTION).
# Vectorised callable: rule(slope, first, last, mean, max_val, count) -> 0/1 Series.
# Use when slope-sign alone is insufficient (e.g. PSA where absolute level matters too).
LAB_WORSENING_RULES = {
    # PSA worsening: count>=2 AND any of
    #   (a) rising slope AND last value >= 4 ng/mL (clinical threshold)
    #   (b) ever exceeded 10 ng/mL
    #   (c) >=50% jump from first to last value.
    'PSA': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (
            ((slope.fillna(0) > 0) & (last.fillna(0) >= 4.0))
            | (mx.fillna(0) >= 10.0)
            | ((first.fillna(0) > 0) &
               ((last.fillna(0) - first.fillna(0)) / first.fillna(0).clip(lower=0.01) >= 0.5))
        )
    ).astype(int),
    # ALP bone marker: rising AND last above upper-normal (~130 U/L) is suspicious for bone mets.
    'ALP_BONE_MARKER': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) > 0) & (last.fillna(0) >= 130.0)
    ).astype(int),
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
    'UROLOGY_IMAGING',
    'CYSTOSCOPY',
    'DRE',
    'SCREENING_PATHWAY',
]

# ═══════════════════════════════════════════════════════════════
# SYMPTOM CLUSTER DEFINITIONS
# Each entry: (group_name, [list of obs categories])
# Used by build_advanced_features() for symptom clustering
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    ('URINARY',        ['LUTS', 'URINARY_RETENTION', 'HAEMATURIA', 'UTI']),
    ('PSA',            ['PSA', 'PSA_FREE', 'PSA_RATIO']),
    ('BONE',           ['PAIN_PELVIC_BONE', 'BONE_MUSCLE', 'ALP_BONE_MARKER']),
    ('CONSTITUTIONAL', ['CONSTITUTIONAL', 'WEIGHT']),
    ('INVESTIGATION',  ['UROLOGY_IMAGING', 'CYSTOSCOPY', 'DRE', 'SCREENING_PATHWAY']),
    ('SEXUAL',         ['ERECTILE_DYSFUNCTION', 'SEXUAL_REPRODUCTIVE']),
    ('PROSTATIC',      ['PROSTATIC_CONDITIONS', 'IPSS', 'CATHETER']),
    ('UROLOGY',        ['UROLOGY_PATHWAY']),
]

# Categories considered "symptoms" for aggregate features
SYMPTOM_CATEGORIES = [
    'LUTS', 'URINARY_RETENTION', 'HAEMATURIA', 'UTI',
    'PAIN_PELVIC_BONE', 'BONE_MUSCLE', 'CONSTITUTIONAL',
    'ERECTILE_DYSFUNCTION', 'PROSTATIC_CONDITIONS',
    'CATHETER', 'WEIGHT',
]

# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# Each entry: (feature_name, col_A, col_B)
# Creates INT_{feature_name} = col_A * col_B
# Columns must exist as has_ever flags in the merged feature matrix
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    ('luts_plus_psa',
     'OBS_LUTS_has_ever', 'OBS_PSA_has_ever'),
    ('haematuria_plus_luts',
     'OBS_HAEMATURIA_has_ever', 'OBS_LUTS_has_ever'),
    ('bone_pain_plus_imaging',
     'OBS_PAIN_PELVIC_BONE_has_ever', 'OBS_UROLOGY_IMAGING_has_ever'),
    ('luts_plus_alpha_blockers',
     'OBS_LUTS_has_ever', 'MED_ALPHA_BLOCKERS_has_ever'),
    ('retention_plus_catheter',
     'OBS_URINARY_RETENTION_has_ever', 'MED_CATHETER_SUPPLIES_has_ever'),
    ('psa_plus_dre',
     'OBS_PSA_has_ever', 'OBS_DRE_has_ever'),
    ('luts_plus_5ari',
     'OBS_LUTS_has_ever', 'MED_5ARI_has_ever'),
    ('uti_plus_antibiotics',
     'OBS_UTI_has_ever', 'MED_UTI_ANTIBIOTICS_has_ever'),
    ('pain_plus_pain_meds',
     'OBS_PAIN_PELVIC_BONE_has_ever', 'MED_PAIN_ESCALATION_has_ever'),
    ('ed_plus_ed_meds',
     'OBS_ERECTILE_DYSFUNCTION_has_ever', 'MED_ED_MEDICATIONS_has_ever'),
    ('prostatic_plus_ipss',
     'OBS_PROSTATIC_CONDITIONS_has_ever', 'OBS_IPSS_has_ever'),
    ('psa_plus_urology_pathway',
     'OBS_PSA_has_ever', 'OBS_UROLOGY_PATHWAY_has_ever'),
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['PAIN_ESCALATION']
GI_MED_CATEGORIES = ['ANTICHOLINERGICS']
REPEAT_PRESCRIPTION_CATEGORY = 'UTI_ANTIBIOTICS'
STEROID_CATEGORY = 'PAIN_ESCALATION'  # closest to escalation tracking

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# Categories whose events get exponential time-decay weighting
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'LUTS', 'HAEMATURIA', 'PAIN_PELVIC_BONE', 'PSA',
    'URINARY_RETENTION', 'CONSTITUTIONAL',
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
    'PSA': {
        'Serum PSA (prostate specific antigen) level': (0, 10000),   # ng/mL
        'Prostate specific antigen level':             (0, 10000),
        'Total PSA (prostate specific antigen) level': (0, 10000),
    },
    'PSA_FREE': {
        'Free PSA (prostate specific antigen) level':         (0, 10000),  # ng/mL
        'Serum free PSA (prostate specific antigen) level':   (0, 10000),
    },
    'PSA_RATIO': {
        'Free:total PSA (prostate specific antigen) ratio':         (0, 100),  # 0-1 fraction or 0-100 percent
        'Serum free:total PSA (prostate specific antigen) ratio':   (0, 100),
    },
    'RENAL_FUNCTION': {
        'Glomerular filtration rate':               (0, 200),    # mL/min/1.73m2
        'Serum creatinine level':                   (0, 2000),   # umol/L
    },
    'ALP_BONE_MARKER': {
        'Serum alkaline phosphatase level':         (0, 2000),   # U/L (bone mets marker)
    },
    'ELECTROLYTES': {
        'Serum sodium level':                       (100, 170),  # mmol/L
        'Serum potassium level':                    (2.0, 8.0),  # mmol/L
    },
    'CALCIUM': {
        'Serum adjusted calcium concentration':     (1.0, 4.0),  # mmol/L
        'Plasma calcium level':                     (1.0, 4.0),
    },
    'FBC_HAEMATOLOGY': {
        'Haemoglobin concentration':                (20, 250),   # g/L
        'Platelet count':                           (0, 2000),   # x10^9/L
        'White blood cell count':                   (0, 500),    # x10^9/L
    },
    'HORMONAL': {
        'Serum testosterone level':                 (0, 60),     # nmol/L
    },
    'ALBUMIN_PROTEIN': {
        'Serum albumin level':                      (10, 60),    # g/L
    },
    'LIVER_FUNCTION': {
        'Plasma total bilirubin level':             (0, 500),    # umol/L
        'Serum alanine aminotransferase level':     (0, 2000),   # U/L
    },
    'VITAMIN_D': {
        'Total 25-hydroxyvitamin D level':          (0, 300),    # nmol/L
    },
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
