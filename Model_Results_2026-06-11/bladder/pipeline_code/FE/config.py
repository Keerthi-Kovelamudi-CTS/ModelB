# ═══════════════════════════════════════════════════════════════
# BLADDER CANCER — PIPELINE CONFIGURATION
# Single source of truth for all cancer-specific settings.
# Categories keyed to codelist2.0/code_category_mapping_2.0.json
# (48 obs categories, 21 med categories).
# No sex filter (bladder cancer affects both sexes; sex carried as a feature).
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

# ─── Identity ────────────────────────────────────────────────
CANCER_NAME = 'bladder'
DATA_PREFIX = 'bladder'
PREFIX = 'BLAD_'
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
# OBSERVATION CATEGORIES (48) — from the bladder codelist
# ═══════════════════════════════════════════════════════════════
OBS_CATEGORIES = [
    'ALCOHOL',
    'ANAEMIA_DX',
    'APPETITE_LOSS',
    'ATRIAL_FIBRILLATION',
    'BACK_PAIN',
    'BMI',
    'BODY_WEIGHT',
    'BPH',
    'CATHETER_PROCEDURES',
    'CKD',
    'COPD',
    'CURRENT_SMOKER',
    'DIABETES',
    'DIASTOLIC_BP',
    'DVT',
    'DYSURIA',
    'ELECTROLYTES',
    'EX_SMOKER',
    'FATIGUE',
    'FRAILTY',
    'GYNAECOLOGICAL_BLEEDING',
    'HAEMATOLOGY',
    'HAEMATURIA',
    'HEART_FAILURE',
    'HEART_RATE',
    'HYPERTENSION',
    'IDEAL_BODY_WEIGHT',
    'IMAGING',
    'INFLAMMATORY',
    'LIVER',
    'LOIN_PAIN',
    'LUTS',
    'METABOLIC',
    'NIGHT_SWEATS',
    'OBESITY',
    'OTHERS',
    'PREVIOUS_CANCER',
    'PULMONARY_EMBOLISM',
    'RECURRENT_UTI',
    'RENAL',
    'SMOKING_CESSATION_REFUSED',
    'SUPRAPUBIC_PAIN',
    'SYSTOLIC_BP',
    'URINE_INVESTIGATIONS',
    'URINE_LAB_ABNORMALITIES',
    'UROLOGICAL_CONDITIONS',
    'WEIGHT_LOSS',
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (21)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'ANTICOAGULANTS',
    'BLADDER_ANTISPASMODICS',
    'CATHETER_MAINTENANCE',
    'CATHETER_SUPPLIES_CATHETERS',
    'CATHETER_SUPPLIES_LEG_BAGS',
    'CATHETER_SUPPLIES_NIGHT_BAGS',
    'CATHETER_SUPPLIES_STRAPS',
    'CATHETER_SUPPLIES_VALVES',
    'COPD_RESPIRATORY',
    'GI_ANTISPASMODICS',
    'HAEMOSTATIC',
    'IRON_SUPPLEMENTS',
    'LAXATIVES',
    'LUTS_MEDICATIONS',
    'NORETHISTERONE',
    'OPIOID_ANALGESICS',
    'PIOGLITAZONE',
    'SMOKING_CESSATION_MEDS',
    'URINARY_RETENTION_DRUGS',
    'UTI_ANTIBIOTICS',
    'WOUND_STOMA_CARE',
]

# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# Haemoglobin (HAEMATOLOGY) is the key value lab — anaemia from haematuria.
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'HAEMATOLOGY', 'RENAL', 'LIVER', 'ELECTROLYTES', 'INFLAMMATORY',
    'METABOLIC', 'URINE_LAB_ABNORMALITIES', 'BMI',
    'SYSTOLIC_BP', 'DIASTOLIC_BP', 'HEART_RATE',
]

# ═══════════════════════════════════════════════════════════════
# LAB DIRECTION OF BADNESS
# ═══════════════════════════════════════════════════════════════
LAB_BAD_DIRECTION = {
    'HAEMATOLOGY':   'down',   # low Hb (anaemia) = bad
    'RENAL':         None,     # mixed
    'INFLAMMATORY':  'up',     # high CRP/ESR = bad
    'METABOLIC':     None,
}

LAB_WORSENING_RULES = {
    # Haemoglobin falling AND last < 110 g/L (anaemia from chronic blood loss)
    'HAEMATOLOGY': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) < 0) & (last.fillna(999) < 110.0)
    ).astype(int),
}

# ═══════════════════════════════════════════════════════════════
# INVESTIGATION CATEGORIES
# ═══════════════════════════════════════════════════════════════
INVESTIGATION_CATEGORIES = [
    'URINE_INVESTIGATIONS', 'URINE_LAB_ABNORMALITIES', 'IMAGING',
    'HAEMATOLOGY', 'RENAL', 'INFLAMMATORY',
]

# ═══════════════════════════════════════════════════════════════
# CLUSTER DEFINITIONS
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    # Haematuria / urinary-pathology system (the dominant bladder presentation)
    ('URINARY_PATHOLOGY', ['HAEMATURIA', 'DYSURIA', 'RECURRENT_UTI',
                           'URINE_INVESTIGATIONS', 'URINE_LAB_ABNORMALITIES',
                           'UROLOGICAL_CONDITIONS']),
    # Lower-urinary-tract symptom system
    ('LOWER_URINARY_TRACT', ['LUTS', 'BPH', 'CATHETER_PROCEDURES']),
    # Smoking system (dominant bladder-cancer risk factor)
    ('SMOKING_SYSTEM', ['CURRENT_SMOKER', 'EX_SMOKER', 'SMOKING_CESSATION_REFUSED']),
    # Constitutional / systemic decline
    ('CONSTITUTIONAL_SYS', ['WEIGHT_LOSS', 'FATIGUE', 'NIGHT_SWEATS',
                            'APPETITE_LOSS', 'FRAILTY', 'ANAEMIA_DX']),
    # Pain system (loin / suprapubic / back)
    ('PAIN_SYSTEM', ['LOIN_PAIN', 'SUPRAPUBIC_PAIN', 'BACK_PAIN']),
    # Anaemia / haematology system
    ('ANAEMIA_SYSTEM', ['ANAEMIA_DX', 'HAEMATOLOGY']),
    # Cardiometabolic background
    ('CARDIOMETABOLIC', ['HYPERTENSION', 'DIABETES', 'OBESITY', 'BMI', 'METABOLIC']),
    # Renal / electrolyte system
    ('RENAL_SYSTEM', ['CKD', 'RENAL', 'ELECTROLYTES']),
]

# Symptom / system categories for aggregate features
SYMPTOM_CATEGORIES = [
    'HAEMATURIA', 'DYSURIA', 'LUTS', 'RECURRENT_UTI', 'LOIN_PAIN',
    'SUPRAPUBIC_PAIN', 'BACK_PAIN', 'WEIGHT_LOSS', 'FATIGUE',
    'NIGHT_SWEATS', 'APPETITE_LOSS',
]

# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# (format: name, 'OBS_<CAT>_has_ever' / 'MED_<CAT>_has_ever')
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # Haematuria in a smoker (highest-risk bladder presentation)
    ('haematuria_plus_current_smoker',
     'OBS_HAEMATURIA_has_ever', 'OBS_CURRENT_SMOKER_has_ever'),
    ('haematuria_plus_ex_smoker',
     'OBS_HAEMATURIA_has_ever', 'OBS_EX_SMOKER_has_ever'),
    # Haematuria with anaemia (chronic blood loss)
    ('haematuria_plus_anaemia',
     'OBS_HAEMATURIA_has_ever', 'OBS_ANAEMIA_DX_has_ever'),
    # Haematuria investigated
    ('haematuria_plus_urine_invest',
     'OBS_HAEMATURIA_has_ever', 'OBS_URINE_INVESTIGATIONS_has_ever'),
    # Recurrent UTI on antibiotics
    ('uti_plus_antibiotic',
     'OBS_RECURRENT_UTI_has_ever', 'MED_UTI_ANTIBIOTICS_has_ever'),
    # Dysuria with recurrent UTI
    ('dysuria_plus_uti',
     'OBS_DYSURIA_has_ever', 'OBS_RECURRENT_UTI_has_ever'),
    # Anaemia treated with iron (chronic blood loss management)
    ('anaemia_plus_iron',
     'OBS_ANAEMIA_DX_has_ever', 'MED_IRON_SUPPLEMENTS_has_ever'),
    # Pioglitazone exposure with urinary symptoms (drug-associated bladder risk)
    ('pioglitazone_plus_haematuria',
     'OBS_HAEMATURIA_has_ever', 'MED_PIOGLITAZONE_has_ever'),
    # LUTS treated
    ('luts_plus_luts_med',
     'OBS_LUTS_has_ever', 'MED_LUTS_MEDICATIONS_has_ever'),
    # Smoking with weight loss (constitutional red flag)
    ('smoker_plus_weight_loss',
     'OBS_CURRENT_SMOKER_has_ever', 'OBS_WEIGHT_LOSS_has_ever'),
    # Previous cancer with haematuria (field effect)
    ('prevcancer_plus_haematuria',
     'OBS_PREVIOUS_CANCER_has_ever', 'OBS_HAEMATURIA_has_ever'),
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['OPIOID_ANALGESICS']
GI_MED_CATEGORIES = ['GI_ANTISPASMODICS', 'LAXATIVES']
REPEAT_PRESCRIPTION_CATEGORY = 'UTI_ANTIBIOTICS'
STEROID_CATEGORY = None

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES (recency-weighted event load)
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'HAEMATURIA',
    'DYSURIA',
    'RECURRENT_UTI',
    'LUTS',
    'LOIN_PAIN',
    'SUPRAPUBIC_PAIN',
    'WEIGHT_LOSS',
    'ANAEMIA_DX',
]

# ═══════════════════════════════════════════════════════════════
# LAB RANGES (outlier clamping; (lo, hi) per recognised TERM substring)
# ═══════════════════════════════════════════════════════════════
LAB_RANGES = {
    'HAEMATOLOGY': {
        'haemoglobin':       (0, 250),     # g/L
        'Hb':                (0, 250),
        'haematocrit':       (0, 1),
        'white cell':        (0, 200),
        'platelet':          (0, 2000),
        'MCV':               (0, 200),
    },
    'RENAL': {
        'creatinine':        (0, 2000),
        'eGFR':              (0, 200),
        'estimated glomerular': (0, 200),
        'urea':              (0, 100),
    },
    'LIVER': {
        'alkaline phosphatase': (0, 3000),
        'ALT':               (0, 3000),
        'bilirubin':         (0, 1000),
        'albumin':           (0, 100),
    },
    'INFLAMMATORY': {
        'C reactive protein': (0, 1000),
        'CRP':               (0, 1000),
        'ESR':               (0, 200),
    },
    'METABOLIC': {
        'HbA1c':             (10, 200),
        'glucose':           (0, 50),
        'cholesterol':       (0, 20),
    },
    'URINE_LAB_ABNORMALITIES': {
        'red blood cell':    (0, 100000),
        'urine':             (0, 100000),
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
