
# ═══════════════════════════════════════════════════════════════
# BLADDER CANCER — 1:1 COHORT VARIANT
# ─────────────────────────────────────────────────────────────
# This folder was scaffolded from Prostate_2.0_1to1. The pipeline
# code (3_pipeline.py, modeling, explainability) is generic and
# works as-is. The cancer-specific bits below are PROSTATE
# PLACEHOLDERS — replace them before running:
#
#   ☐ OBS_CATEGORIES     — bladder symptom/observation categories
#   ☐ MED_CATEGORIES     — bladder medication categories
#   ☐ LAB_CATEGORIES     — bladder relevant labs
#   ☐ LAB_BAD_DIRECTION  — clinical direction-of-bad per lab
#   ☐ LAB_WORSENING_RULES— optional vectorised worsening rules
#   ☐ LAB_RANGES         — physiologic bounds for outlier filter
#   ☐ CLUSTER_DEFINITIONS— bladder symptom clusters
#   ☐ INTERACTION_PAIRS  — bladder-relevant interactions
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
CANCER_NAME = 'bladder'
DATA_PREFIX = 'bladder'          # CSV filename prefix: prostate_3mo_obs.csv
PREFIX = 'BLAD_'                 # Feature prefix for cancer-specific columns
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
    # Symptoms / clinical observations specific to bladder cancer presentation
    'HAEMATURIA',                # PRIMARY signal — visible/microscopic blood in urine
    'URINE_LAB_ABNORMALITIES',   # urine dip findings (blood/protein/cells)
    'URINE_INVESTIGATIONS',      # urine culture / microscopy / cytology orders
    'LUTS',                      # storage/voiding symptoms
    'DYSURIA',                   # painful urination
    'SUPRAPUBIC_PAIN',
    'BACK_PAIN',
    'LOIN_PAIN',
    'UROLOGICAL_CONDITIONS',     # bladder stones, hydronephrosis, AKI etc.
    'CATHETER_PROCEDURES',
    'IMAGING',                   # renal/bladder ultrasound
    'GYNAECOLOGICAL_BLEEDING',   # important DDx for haematuria in females
    'WEIGHT_LOSS',
    'FATIGUE',
    'APPETITE_LOSS',
    'NIGHT_SWEATS',
    'DVT', 'PULMONARY_EMBOLISM',
    'ANAEMIA_DX',                # diagnosed anaemia (not lab)
    'FRAILTY',
    'CURRENT_SMOKER', 'EX_SMOKER', 'SMOKING_CESSATION_REFUSED',  # smoking is the #1 bladder cancer risk factor
    'ALCOHOL',
    'DIABETES', 'CKD', 'HYPERTENSION', 'OBESITY', 'COPD',
    'HEART_FAILURE', 'ATRIAL_FIBRILLATION',
    'RECURRENT_UTI',             # often misdiagnosed bladder cancer in women
    'BPH',                       # DDx for LUTS in older men
    'PREVIOUS_CANCER',
    'OTHERS',
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (from actual prostate SQL data)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'UTI_ANTIBIOTICS',
    'CATHETER_SUPPLIES_NIGHT_BAGS',
    'CATHETER_SUPPLIES_LEG_BAGS',
    'CATHETER_SUPPLIES_STRAPS',
    'CATHETER_SUPPLIES_CATHETERS',
    'CATHETER_SUPPLIES_VALVES',
    'CATHETER_MAINTENANCE',
    'LUTS_MEDICATIONS',
    'BLADDER_ANTISPASMODICS',
    'URINARY_RETENTION_DRUGS',
    'HAEMOSTATIC',               # tranexamic acid — bleeding control
    'ANTICOAGULANTS',            # haematuria DDx (anticoag-related vs cancer)
    'WOUND_STOMA_CARE',
    'LAXATIVES',
    'SMOKING_CESSATION_MEDS',
    'COPD_RESPIRATORY',
    'NORETHISTERONE',
    'IRON_SUPPLEMENTS',          # iron-deficiency anaemia treatment
    'OPIOID_ANALGESICS',         # pain escalation
    'GI_ANTISPASMODICS',
]

# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'RENAL',           # eGFR, creatinine — obstruction can raise creatinine
    'HAEMATOLOGY',     # Hb, platelets, ferritin, WBC — anaemia of malignancy
    'INFLAMMATORY',    # CRP, ESR
    'LIVER',           # ALT, ALP — distant mets
    'METABOLIC',       # albumin, glucose, HbA1c
    'ELECTROLYTES',    # calcium — paraneoplastic hypercalcaemia
    'PSA',             # incidental in older men; helpful for prostate-vs-bladder DDx
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
    'RENAL':         'down',  # eGFR low = bad
    'HAEMATOLOGY':   'down',  # Hb low = anaemia of malignancy / chronic blood loss
    'INFLAMMATORY':  'up',    # CRP/ESR high = bad
    'LIVER':         'up',    # ALT/ALP high = liver mets
    'METABOLIC':     'down',  # low albumin = advanced disease
    'ELECTROLYTES':  'up',    # hypercalcaemia = paraneoplastic / bone mets
    'PSA':           'up',    # if clinically rising
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
    # Hb falling AND last value below normal = anaemia of malignancy / chronic blood loss
    # (clinical Hb threshold ~13 g/dL for adult males, ~12 for females; using 12 as conservative cutoff)
    'HAEMATOLOGY': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) < 0) & (last.fillna(15) < 12.0)
    ).astype(int),
    # CRP rising AND last value above normal = inflammatory burden (often paraneoplastic in bladder ca)
    'INFLAMMATORY': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) > 0) & (last.fillna(0) > 10.0)
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
    'IMAGING',                  # renal/bladder US, KUB
    'URINE_INVESTIGATIONS',     # urine culture / microscopy / cytology
    'URINE_LAB_ABNORMALITIES',  # urine dip findings (proxy for AnyU&E investigation)
]

# ═══════════════════════════════════════════════════════════════
# SYMPTOM CLUSTER DEFINITIONS
# Each entry: (group_name, [list of obs categories])
# Used by build_advanced_features() for symptom clustering
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    ('HAEMATURIA',      ['HAEMATURIA', 'URINE_LAB_ABNORMALITIES']),
    ('URINARY',         ['LUTS', 'DYSURIA', 'SUPRAPUBIC_PAIN', 'RECURRENT_UTI', 'CATHETER_PROCEDURES']),
    ('PAIN',            ['BACK_PAIN', 'LOIN_PAIN', 'SUPRAPUBIC_PAIN']),
    ('SYSTEMIC',        ['WEIGHT_LOSS', 'FATIGUE', 'APPETITE_LOSS', 'NIGHT_SWEATS', 'ANAEMIA_DX']),
    ('INVESTIGATION',   ['IMAGING', 'URINE_INVESTIGATIONS']),
    ('THROMBOEMBOLIC',  ['DVT', 'PULMONARY_EMBOLISM']),
    ('SMOKING',         ['CURRENT_SMOKER', 'EX_SMOKER', 'SMOKING_CESSATION_REFUSED']),
    ('COMORBIDITY',     ['DIABETES', 'CKD', 'HYPERTENSION', 'COPD', 'HEART_FAILURE', 'ATRIAL_FIBRILLATION']),
]

# Categories considered "symptoms" for aggregate features
SYMPTOM_CATEGORIES = [
    'HAEMATURIA', 'LUTS', 'DYSURIA', 'SUPRAPUBIC_PAIN',
    'BACK_PAIN', 'LOIN_PAIN',
    'WEIGHT_LOSS', 'FATIGUE', 'APPETITE_LOSS', 'NIGHT_SWEATS',
    'ANAEMIA_DX', 'GYNAECOLOGICAL_BLEEDING',
]

# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# Each entry: (feature_name, col_A, col_B)
# Creates INT_{feature_name} = col_A * col_B
# Columns must exist as has_ever flags in the merged feature matrix
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # Haematuria DDx + risk-factor combos
    ('haematuria_plus_smoking',
     'OBS_HAEMATURIA_has_ever', 'OBS_CURRENT_SMOKER_has_ever'),
    ('haematuria_plus_smoking_history',
     'OBS_HAEMATURIA_has_ever', 'OBS_EX_SMOKER_has_ever'),
    ('haematuria_plus_uti',
     'OBS_HAEMATURIA_has_ever', 'OBS_RECURRENT_UTI_has_ever'),
    ('haematuria_plus_imaging',
     'OBS_HAEMATURIA_has_ever', 'OBS_IMAGING_has_ever'),
    ('haematuria_plus_anticoag',
     'OBS_HAEMATURIA_has_ever', 'MED_ANTICOAGULANTS_has_ever'),
    ('haematuria_plus_haemostatic',
     'OBS_HAEMATURIA_has_ever', 'MED_HAEMOSTATIC_has_ever'),
    ('haematuria_plus_anaemia',
     'OBS_HAEMATURIA_has_ever', 'OBS_ANAEMIA_DX_has_ever'),
    # LUTS / urinary tract combos
    ('luts_plus_haematuria',
     'OBS_LUTS_has_ever', 'OBS_HAEMATURIA_has_ever'),
    ('luts_plus_bladder_antispasm',
     'OBS_LUTS_has_ever', 'MED_BLADDER_ANTISPASMODICS_has_ever'),
    ('recurrent_uti_plus_antibiotics',
     'OBS_RECURRENT_UTI_has_ever', 'MED_UTI_ANTIBIOTICS_has_ever'),
    ('catheter_plus_haematuria',
     'OBS_CATHETER_PROCEDURES_has_ever', 'OBS_HAEMATURIA_has_ever'),
    # Systemic + local
    ('weight_loss_plus_anaemia',
     'OBS_WEIGHT_LOSS_has_ever', 'OBS_ANAEMIA_DX_has_ever'),
    ('loin_pain_plus_imaging',
     'OBS_LOIN_PAIN_has_ever', 'OBS_IMAGING_has_ever'),
    ('haematuria_plus_iron',
     'OBS_HAEMATURIA_has_ever', 'MED_IRON_SUPPLEMENTS_has_ever'),
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['OPIOID_ANALGESICS']
GI_MED_CATEGORIES = ['GI_ANTISPASMODICS']
REPEAT_PRESCRIPTION_CATEGORY = 'UTI_ANTIBIOTICS'
STEROID_CATEGORY = 'OPIOID_ANALGESICS'  # bladder analogue: pain escalation as severity proxy

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# Categories whose events get exponential time-decay weighting
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'HAEMATURIA', 'URINE_LAB_ABNORMALITIES', 'LUTS', 'DYSURIA',
    'WEIGHT_LOSS', 'FATIGUE', 'ANAEMIA_DX',
    'RECURRENT_UTI', 'LOIN_PAIN', 'SUPRAPUBIC_PAIN',
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
    # TODO: bladder-specific lab term names + plausible bounds.
    # Schema: { CATEGORY: { exact term string from data: (low, high) } }.
    # Leave empty until you've inspected actual term values from the SQL extract.
    # Pipeline tolerates an empty dict — outliers won't be filtered, but values
    # outside reason will still be visible in the model.
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
