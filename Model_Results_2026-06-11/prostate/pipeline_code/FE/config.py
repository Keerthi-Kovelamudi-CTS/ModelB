# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PIPELINE CONFIGURATION
# Single source of truth for all cancer-specific settings.
# Categories keyed to codelist2.0/code_category_mapping_2.0.json
# (43 obs categories, 7 med categories).
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

# ─── Identity ────────────────────────────────────────────────
CANCER_NAME = 'prostate'
DATA_PREFIX = 'prostate'
PREFIX = 'PROS_'
LABEL_COL = 'LABEL'
WINDOWS = ["12mo"]

# ─── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'
_RESULTS_ROOT   = SCRIPT_DIR / 'results'
FE_RESULTS      = _RESULTS_ROOT / '3_feature_engineering'
CLEANUP_RESULTS = _RESULTS_ROOT / '5_cleanup'
SANITY_RESULTS  = _RESULTS_ROOT / '1_sanity_check'
SHARED_SPLIT_PATH       = SCRIPT_DIR.parent / 'Modeling' / 'shared_split.json'
MIN_EVENTS_PER_WINDOW   = 5

CODELIST_MAPPING_JSON = SCRIPT_DIR.parent / 'codelist2.0' / 'code_category_mapping_2.0.json'
SAVE_INTERMEDIATES = False


# ═══════════════════════════════════════════════════════════════
# OBSERVATION CATEGORIES (43) — from the prostate codelist
# ═══════════════════════════════════════════════════════════════
OBS_CATEGORIES = [
    'ALBUMIN_PROTEIN',
    'ALP_BONE_MARKER',
    'BMI',
    'BODY_WEIGHT',
    'BONE_MUSCLE',
    'CALCIUM',
    'CATHETER',
    'COAGULATION',
    'CONSTITUTIONAL',
    'DIASTOLIC_BP',
    'DRE',
    'ELECTROLYTES',
    'ERECTILE_DYSFUNCTION',
    'EXAMINATION',
    'FAMILY_HISTORY',
    'FBC_HAEMATOLOGY',
    'HAEMATOLOGY',
    'HAEMATURIA',
    'HEART_RATE',
    'HORMONAL',
    'HYPERTENSION',
    'IDEAL_BODY_WEIGHT',
    'INFLAMMATORY',
    'IPSS',
    'LAB_FLAGS',
    'LIVER_FUNCTION',
    'LUTS',
    'OBESITY',
    'PAIN_PELVIC_BONE',
    'PROSTATIC_CONDITIONS',
    'PSA',
    'PSA_FREE',
    'PSA_RATIO',
    'RENAL_FUNCTION',
    'SEXUAL_REPRODUCTIVE',
    'SMOKING',
    'SYSTOLIC_BP',
    'URINARY_RETENTION',
    'URINE_MARKERS',
    'UROLOGY_PATHWAY',
    'UTI',
    'VITAMIN_D',
    'WEIGHT',
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (7)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    '5ARI',
    'ALPHA_BLOCKERS',
    'ANTICHOLINERGICS',
    'ED_MEDICATIONS',
    'PAIN_ESCALATION',
    'UTI_ANTIBIOTICS',
    'UTI_TREATMENT',
]

# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# NOTE: PSA presence/test-counts are LEAKY (ordering a PSA test is part of the
# diagnostic workup). We use PSA VALUE-derived features only — never has/count.
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'PSA', 'PSA_FREE', 'PSA_RATIO',
    'ALP_BONE_MARKER', 'CALCIUM', 'ALBUMIN_PROTEIN',
    'RENAL_FUNCTION', 'LIVER_FUNCTION', 'ELECTROLYTES',
    'FBC_HAEMATOLOGY', 'HAEMATOLOGY', 'INFLAMMATORY', 'VITAMIN_D',
    'BMI', 'SYSTOLIC_BP', 'DIASTOLIC_BP', 'HEART_RATE', 'IPSS',
]

# ═══════════════════════════════════════════════════════════════
# LAB DIRECTION OF BADNESS
# ═══════════════════════════════════════════════════════════════
LAB_BAD_DIRECTION = {
    'PSA':              'up',    # rising PSA = bad
    'PSA_FREE':         'down',  # low free PSA fraction = bad
    'PSA_RATIO':        'down',  # low free/total ratio (<0.15) = bad
    'ALP_BONE_MARKER':  'up',    # high ALP = bone-met marker
    'CALCIUM':          'up',    # hypercalcaemia = bone-met marker
    'RENAL_FUNCTION':   None,    # mixed (creatinine up / eGFR down)
    'ALBUMIN_PROTEIN':  'down',  # low albumin = constitutional decline
    'IPSS':             'up',    # higher score = worse urinary symptoms
}

LAB_WORSENING_RULES = {
    # PSA rising AND last >= 4 ng/mL (the standard referral threshold)
    'PSA': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) > 0) & (last.fillna(0) >= 4.0)
    ).astype(int),
    # ALP rising AND last elevated (bone metastasis signal)
    'ALP_BONE_MARKER': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) > 0) & (last.fillna(0) >= 130.0)
    ).astype(int),
}

# ═══════════════════════════════════════════════════════════════
# INVESTIGATION CATEGORIES
# ═══════════════════════════════════════════════════════════════
INVESTIGATION_CATEGORIES = [
    'DRE', 'IPSS', 'ALP_BONE_MARKER', 'RENAL_FUNCTION',
    'FBC_HAEMATOLOGY', 'CALCIUM', 'LIVER_FUNCTION', 'URINE_MARKERS',
]

# ═══════════════════════════════════════════════════════════════
# CLUSTER DEFINITIONS
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    # Lower-urinary-tract symptom system (the dominant prostate presentation)
    ('LOWER_URINARY_TRACT', ['LUTS', 'IPSS', 'URINARY_RETENTION', 'CATHETER',
                              'PROSTATIC_CONDITIONS']),
    # Urinary-tract pathology / infection system
    ('URINARY_PATHOLOGY', ['HAEMATURIA', 'UTI', 'URINE_MARKERS', 'UROLOGY_PATHWAY']),
    # PSA / tumour-marker system (VALUE-driven, presence excluded as leaky)
    ('PSA_SYSTEM', ['PSA', 'PSA_FREE', 'PSA_RATIO']),
    # Bone-metastasis signal system
    ('BONE_MET_SIGNAL', ['ALP_BONE_MARKER', 'CALCIUM', 'PAIN_PELVIC_BONE', 'BONE_MUSCLE']),
    # Constitutional / systemic decline
    ('CONSTITUTIONAL_SYS', ['CONSTITUTIONAL', 'WEIGHT', 'BODY_WEIGHT',
                            'ALBUMIN_PROTEIN', 'INFLAMMATORY']),
    # Sexual / hormonal system
    ('SEXUAL_HORMONAL', ['ERECTILE_DYSFUNCTION', 'SEXUAL_REPRODUCTIVE', 'HORMONAL']),
    # Cardiometabolic background
    ('CARDIOMETABOLIC', ['HYPERTENSION', 'OBESITY', 'BMI']),
    # Renal / electrolyte system (obstructive uropathy proxy)
    ('RENAL_SYSTEM', ['RENAL_FUNCTION', 'ELECTROLYTES', 'CALCIUM']),
]

# Symptom / system categories for aggregate features
SYMPTOM_CATEGORIES = [
    'LUTS', 'IPSS', 'URINARY_RETENTION', 'HAEMATURIA',
    'PAIN_PELVIC_BONE', 'ERECTILE_DYSFUNCTION', 'CONSTITUTIONAL',
    'UTI', 'BONE_MUSCLE', 'WEIGHT',
]

# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# (format: name, 'OBS_<CAT>_has_ever' / 'MED_<CAT>_has_ever')
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # LUTS treated with an alpha-blocker (symptomatic BPH/obstruction pathway)
    ('luts_plus_alpha_blocker',
     'OBS_LUTS_has_ever', 'MED_ALPHA_BLOCKERS_has_ever'),
    # LUTS treated with a 5-alpha-reductase inhibitor (large prostate)
    ('luts_plus_5ari',
     'OBS_LUTS_has_ever', 'MED_5ARI_has_ever'),
    # Prostatic condition on a 5ARI
    ('prostatic_cond_plus_5ari',
     'OBS_PROSTATIC_CONDITIONS_has_ever', 'MED_5ARI_has_ever'),
    # Urinary retention requiring catheter
    ('retention_plus_catheter',
     'OBS_URINARY_RETENTION_has_ever', 'OBS_CATHETER_has_ever'),
    # DRE performed in a man with LUTS (active assessment)
    ('dre_plus_luts',
     'OBS_DRE_has_ever', 'OBS_LUTS_has_ever'),
    # Haematuria with LUTS (urological-pathway flag)
    ('haematuria_plus_luts',
     'OBS_HAEMATURIA_has_ever', 'OBS_LUTS_has_ever'),
    # Erectile dysfunction treated (sexual-health engagement)
    ('ed_plus_ed_meds',
     'OBS_ERECTILE_DYSFUNCTION_has_ever', 'MED_ED_MEDICATIONS_has_ever'),
    # Bone pain with raised ALP (metastasis co-signal)
    ('bone_pain_plus_alp',
     'OBS_PAIN_PELVIC_BONE_has_ever', 'OBS_ALP_BONE_MARKER_has_ever'),
    # Recurrent UTI on antibiotics
    ('uti_plus_antibiotic',
     'OBS_UTI_has_ever', 'MED_UTI_ANTIBIOTICS_has_ever'),
    # Overactive-bladder symptoms on anticholinergics
    ('luts_plus_anticholinergic',
     'OBS_LUTS_has_ever', 'MED_ANTICHOLINERGICS_has_ever'),
    # Family history with urinary symptoms
    ('famhx_plus_luts',
     'OBS_FAMILY_HISTORY_has_ever', 'OBS_LUTS_has_ever'),
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['PAIN_ESCALATION']
GI_MED_CATEGORIES = []
REPEAT_PRESCRIPTION_CATEGORY = 'ALPHA_BLOCKERS'
STEROID_CATEGORY = None

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES (recency-weighted event load)
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'LUTS',
    'URINARY_RETENTION',
    'HAEMATURIA',
    'PAIN_PELVIC_BONE',
    'PROSTATIC_CONDITIONS',
    'DRE',
    'PSA',
    'ERECTILE_DYSFUNCTION',
]

# ═══════════════════════════════════════════════════════════════
# LAB RANGES (outlier clamping; (lo, hi) per recognised TERM substring)
# ═══════════════════════════════════════════════════════════════
LAB_RANGES = {
    'PSA': {
        'prostate specific antigen': (0, 10000),
        'Serum PSA':                 (0, 10000),
        'PSA':                       (0, 10000),
    },
    'PSA_FREE': {
        'free PSA':                  (0, 5000),
        'Free prostate specific antigen': (0, 5000),
    },
    'PSA_RATIO': {
        'free/total':                (0, 1.5),
        'PSA ratio':                 (0, 1.5),
        'percent free':              (0, 100),
    },
    'ALP_BONE_MARKER': {
        'alkaline phosphatase':      (0, 3000),
        'ALP':                       (0, 3000),
    },
    'CALCIUM': {
        'calcium':                   (0, 5),       # mmol/L
        'corrected calcium':         (0, 5),
    },
    'RENAL_FUNCTION': {
        'creatinine':                (0, 2000),    # umol/L
        'eGFR':                      (0, 200),
        'estimated glomerular':      (0, 200),
        'urea':                      (0, 100),
    },
    'ALBUMIN_PROTEIN': {
        'albumin':                   (0, 100),     # g/L
        'total protein':             (0, 200),
    },
    'INFLAMMATORY': {
        'C reactive protein':        (0, 1000),
        'CRP':                       (0, 1000),
        'ESR':                       (0, 200),
    },
    'VITAMIN_D': {
        'vitamin D':                 (0, 500),
    },
    'IPSS': {
        'International Prostate Symptom Score': (0, 35),
        'IPSS':                      (0, 35),
    },
}

# ═══════════════════════════════════════════════════════════════
# MODELING CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SEEDS = [42]
N_SELECT_FEATURES = 500
N_SELECT_FEATURES_PER_WINDOW = {
    '12mo': 500,
}
OPTUNA_TRIALS = 75
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

# ═══════════════════════════════════════════════════════════════
# FEATURE GROUP DISPLAY (for cleanup summary)
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
