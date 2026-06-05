
# ═══════════════════════════════════════════════════════════════
# OVARIAN CANCER — 1:1 COHORT VARIANT
# ─────────────────────────────────────────────────────────────
# This folder was scaffolded from Prostate_2.0_1to1. The pipeline
# code (3_pipeline.py, modeling, explainability) is generic and
# works as-is. The cancer-specific bits below are PROSTATE
# PLACEHOLDERS — replace them before running:
#
#   ☐ OBS_CATEGORIES     — ovarian symptom/observation categories
#   ☐ MED_CATEGORIES     — ovarian medication categories
#   ☐ LAB_CATEGORIES     — ovarian relevant labs
#   ☐ LAB_BAD_DIRECTION  — clinical direction-of-bad per lab
#   ☐ LAB_WORSENING_RULES— optional vectorised worsening rules
#   ☐ LAB_RANGES         — physiologic bounds for outlier filter
#   ☐ CLUSTER_DEFINITIONS— ovarian symptom clusters
#   ☐ INTERACTION_PAIRS  — ovarian-relevant interactions
#   ☐ DECAY_CATEGORIES   — categories to exp-decay
#   ☐ SYMPTOM_CATEGORIES — aggregate symptom set
#
# Also replace 4_cancer_features.py (currently PSA-specific).
# Sex filter: F (apply in the SQL WHERE clause).
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — PIPELINE CONFIGURATION
# Single source of truth for all cancer-specific settings.
# To port to a new cancer: copy this file and update the values.
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

# ─── Identity ────────────────────────────────────────────────
CANCER_NAME = 'ovarian'
DATA_PREFIX = 'ovarian'          # CSV filename prefix: prostate_3mo_obs.csv
PREFIX = 'OVAR_'                 # Feature prefix for cancer-specific columns
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
    # Symptoms / clinical observations specific to ovarian cancer presentation
    'ABDOMINAL_MASS',            # PRIMARY signal — pelvic/abdominal mass on exam
    'ASCITES',                   # PRIMARY signal — late-stage ovarian
    'ABDOMINAL_BLOATING',        # KEY symptom (NICE recognises persistent bloating as red flag)
    'EARLY_SATIETY',             # KEY symptom (one of the four classic ovarian symptoms)
    'ABDOMINAL_PAIN',
    'GI_SYMPTOMS',
    'GYNAECOLOGICAL_BLEEDING',
    'VAGINAL_DISCHARGE',
    'BREAST_LUMP',
    'GYNAECOLOGICAL_HISTORY',
    'URINARY_SYMPTOMS',
    'WEIGHT_LOSS',
    'FATIGUE',
    'SYSTEMIC',                  # pleural effusion, oedema
    'OVARIAN_CYST',              # ovarian cyst on imaging — high-risk DDx
    'GYNAECOLOGICAL_DX',
    'POSTMENOPAUSAL',
    'DVT', 'PULMONARY_EMBOLISM', # paraneoplastic / Trousseau-like
    'MICROBIOLOGY',
    'IMAGING',
    'GYNAE_PROCEDURES',
    'OTHER_PROCEDURES',
    'REPRODUCTIVE',
    'CONTRACEPTION',
    'CLINICAL_ASSESSMENT',
    'EXAMINATION',
    'MONITORING',
    'OTHER_CLINICAL',
    'SCREENING',
    'ANAEMIA',
    'ATRIAL_FIBRILLATION', 'CKD', 'HYDRONEPHROSIS', 'DIVERTICULITIS',
    'RENAL_STONE', 'CYSTOCELE', 'HYPONATRAEMIA', 'PNEUMONIA', 'OBESITY',
    'CANDIDA',
    'FAMILY_HISTORY',
    'HRT',
    'REPRODUCTIVE_HISTORY',
    'SOCIAL',
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (from actual prostate SQL data)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'OPIOID_ANALGESICS', 'NSAIDS', 'NEUROPATHIC_PAIN',
    'PPI',
    'GI_ANTISPASMODICS', 'ANTIEMETICS',
    'LAXATIVES',
    'UTI_ANTIBIOTICS', 'GENERAL_ANTIBIOTICS', 'ANTIFUNGALS',
    'CARDIOVASCULAR', 'DIURETICS',
    'ANTICOAGULANTS', 'HAEMOSTATIC',
    'HRT', 'HORMONAL', 'ORAL_CONTRACEPTIVE',
    'IRON_SUPPLEMENTS',
    'THYROID',
    'ANTIDEPRESSANTS',
    'ANTIHISTAMINES',
    'BLADDER_ANTISPASMODICS',
]
# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'HAEMATOLOGY',          # Hb, platelets, ferritin, iron — anaemia of malignancy
    'RENAL',                # eGFR, creatinine
    'INFLAMMATORY',         # CRP, ESR
    'METABOLIC',            # albumin
    'LIVER',                # ALP, ALT — distant mets
    'HORMONAL',             # LH, FSH, oestradiol, testosterone, SHBG
    'CANCER_MARKER',        # LDH (CA-125 not in current codelist; add when available)
    'ELECTROLYTES',         # corrected calcium
    'COAGULATION',          # APTT, clotting screen
    'GI_INVESTIGATION',     # amylase, calprotectin, FIT
    'OTHER_LAB',            # BNP, lab abnormal flags
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
    'HAEMATOLOGY':       'down',  # Hb low = anaemia of malignancy / chronic blood loss
    'RENAL':             'down',  # eGFR low = bad
    'INFLAMMATORY':      'up',    # CRP/ESR high = bad
    'METABOLIC':         'down',  # low albumin = advanced disease
    'LIVER':             'up',    # ALP/ALT high = liver mets
    'HORMONAL':          None,    # context-dependent (premenopausal vs postmenopausal)
    'CANCER_MARKER':     'up',    # LDH high = paraneoplastic / mass burden
    'ELECTROLYTES':      'up',    # hypercalcaemia = paraneoplastic
    'COAGULATION':       None,    # context
    'GI_INVESTIGATION':  'up',    # calprotectin / amylase high = bad
    'OTHER_LAB':         None,
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
    # (using 12 g/dL as conservative cutoff for women)
    'HAEMATOLOGY': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) < 0) & (last.fillna(15) < 12.0)
    ).astype(int),
    # CRP rising AND last value above normal = inflammatory burden / paraneoplastic
    'INFLAMMATORY': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) > 0) & (last.fillna(0) > 10.0)
    ).astype(int),
    # LDH rising AND last value above 250 U/L = paraneoplastic / mass burden
    'CANCER_MARKER': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) > 0) & (last.fillna(0) > 250.0)
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
    'IMAGING',          # transvaginal US, pelvic US, abdominal US, chest X-ray
    'GYNAE_PROCEDURES', # endometrial sampling, hysteroscopy, biopsy
    'MICROBIOLOGY',     # vaginal swabs, chlamydia testing
]
# ═══════════════════════════════════════════════════════════════
# SYMPTOM CLUSTER DEFINITIONS
# Each entry: (group_name, [list of obs categories])
# Used by build_advanced_features() for symptom clustering
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    ('ABDOMINAL',       ['ABDOMINAL_MASS', 'ASCITES', 'ABDOMINAL_BLOATING', 'EARLY_SATIETY', 'ABDOMINAL_PAIN']),
    ('GYNAECOLOGICAL',  ['GYNAECOLOGICAL_BLEEDING', 'VAGINAL_DISCHARGE', 'GYNAECOLOGICAL_DX', 'OVARIAN_CYST', 'POSTMENOPAUSAL']),
    ('GI_DIGESTIVE',    ['GI_SYMPTOMS', 'EARLY_SATIETY']),
    ('URINARY',         ['URINARY_SYMPTOMS']),
    ('SYSTEMIC',        ['WEIGHT_LOSS', 'FATIGUE', 'SYSTEMIC', 'ANAEMIA']),
    ('INVESTIGATION',   ['IMAGING', 'GYNAE_PROCEDURES', 'MICROBIOLOGY']),
    ('THROMBOEMBOLIC',  ['DVT', 'PULMONARY_EMBOLISM']),
    ('COMORBIDITY',     ['ATRIAL_FIBRILLATION', 'CKD', 'HYDRONEPHROSIS', 'DIVERTICULITIS', 'OBESITY']),
]
# Categories considered "symptoms" for aggregate features
SYMPTOM_CATEGORIES = [
    'ABDOMINAL_MASS', 'ASCITES', 'ABDOMINAL_BLOATING', 'EARLY_SATIETY',
    'ABDOMINAL_PAIN', 'GI_SYMPTOMS',
    'GYNAECOLOGICAL_BLEEDING', 'VAGINAL_DISCHARGE',
    'URINARY_SYMPTOMS',
    'WEIGHT_LOSS', 'FATIGUE', 'SYSTEMIC',
]
# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# Each entry: (feature_name, col_A, col_B)
# Creates INT_{feature_name} = col_A * col_B
# Columns must exist as has_ever flags in the merged feature matrix
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # The classic ovarian-cancer pattern: mass + ascites + systemic
    ('mass_plus_ascites',
     'OBS_ABDOMINAL_MASS_has_ever', 'OBS_ASCITES_has_ever'),
    ('bloating_plus_pain',
     'OBS_ABDOMINAL_BLOATING_has_ever', 'OBS_ABDOMINAL_PAIN_has_ever'),
    ('bloating_plus_imaging',
     'OBS_ABDOMINAL_BLOATING_has_ever', 'OBS_IMAGING_has_ever'),
    ('early_satiety_plus_weight_loss',
     'OBS_EARLY_SATIETY_has_ever', 'OBS_WEIGHT_LOSS_has_ever'),
    ('mass_plus_imaging',
     'OBS_ABDOMINAL_MASS_has_ever', 'OBS_IMAGING_has_ever'),
    ('ascites_plus_weight_loss',
     'OBS_ASCITES_has_ever', 'OBS_WEIGHT_LOSS_has_ever'),
    ('ovarian_cyst_plus_imaging',
     'OBS_OVARIAN_CYST_has_ever', 'OBS_IMAGING_has_ever'),
    ('postmenopausal_bleeding',
     'OBS_GYNAECOLOGICAL_BLEEDING_has_ever', 'OBS_POSTMENOPAUSAL_has_ever'),
    ('bloating_plus_antispasm',
     'OBS_ABDOMINAL_BLOATING_has_ever', 'MED_GI_ANTISPASMODICS_has_ever'),
    ('gi_symptoms_plus_antiemetics',
     'OBS_GI_SYMPTOMS_has_ever', 'MED_ANTIEMETICS_has_ever'),
    ('bleeding_plus_haemostatic',
     'OBS_GYNAECOLOGICAL_BLEEDING_has_ever', 'MED_HAEMOSTATIC_has_ever'),
    ('weight_loss_plus_anaemia',
     'OBS_WEIGHT_LOSS_has_ever', 'OBS_ANAEMIA_has_ever'),
    ('breast_lump_plus_family_history',
     'OBS_BREAST_LUMP_has_ever', 'OBS_FAMILY_HISTORY_has_ever'),
    ('mass_plus_ldh_high',
     'OBS_ABDOMINAL_MASS_has_ever', 'LAB_CANCER_MARKER_worsening'),
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['OPIOID_ANALGESICS', 'NSAIDS', 'NEUROPATHIC_PAIN']
GI_MED_CATEGORIES = ['GI_ANTISPASMODICS', 'ANTIEMETICS']
REPEAT_PRESCRIPTION_CATEGORY = 'HRT'  # often a long-term repeat in this cohort
STEROID_CATEGORY = 'OPIOID_ANALGESICS'  # ovarian analogue: pain escalation as severity proxyGI_MED_CATEGORIES = ['ANTICHOLINERGICS']

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# Categories whose events get exponential time-decay weighting
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'ABDOMINAL_MASS', 'ASCITES', 'ABDOMINAL_BLOATING', 'EARLY_SATIETY',
    'ABDOMINAL_PAIN', 'GI_SYMPTOMS', 'GYNAECOLOGICAL_BLEEDING',
    'WEIGHT_LOSS', 'FATIGUE', 'URINARY_SYMPTOMS',
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
    # TODO: ovarian-specific lab term names + plausible bounds.
    # Schema: { CATEGORY: { exact term string from data: (low, high) } }.
    # Leave empty until you've inspected actual term values from the SQL extract.
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
