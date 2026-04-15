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
WINDOWS = ['3mo', '6mo', '12mo']

# ─── Paths (auto-resolved from this file's location) ────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'
FE_RESULTS = SCRIPT_DIR / 'results' / '3_feature_engineering'
CLEANUP_RESULTS = SCRIPT_DIR / 'results' / '5_cleanup'
TEXT_RESULTS = SCRIPT_DIR / 'results' / '6_text_features' / 'keywords'
EMB_RESULTS = SCRIPT_DIR / 'results' / '6_text_features' / 'tfidf_embeddings'
BERT_RESULTS = SCRIPT_DIR / 'results' / '6_text_features' / 'bert_embeddings'
SANITY_RESULTS = SCRIPT_DIR / 'results' / '1_sanity_check'

# Whether to save intermediate CSVs (feature_matrix.csv, enhanced.csv, mega.csv)
# Set False for production (only saves final); True for debugging
SAVE_INTERMEDIATES = False


# ═══════════════════════════════════════════════════════════════
# OBSERVATION CATEGORIES
# UPDATE these based on your SQL/SNOMED groupings for prostate
# ═══════════════════════════════════════════════════════════════
OBS_CATEGORIES = [
    'URINARY_SYMPTOMS',
    'PSA_MONITORING',
    'HAEMATURIA',
    'BONE_PAIN',
    'ERECTILE_DYSFUNCTION',
    'LOWER_BACK_PAIN',
    'WEIGHT_LOSS',
    'FATIGUE',
    'ABNORMAL_IMAGING',
    'LAB_MARKERS',
    'HAEMATOLOGICAL_ABNORMALITIES',
    'PAIN_SYMPTOMS',
    'ADDITIONAL_SYMPTOMS',
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'ALPHA_BLOCKERS',                 # tamsulosin, alfuzosin
    'FIVE_ALPHA_REDUCTASE_INHIBITORS', # finasteride, dutasteride
    'ANTIBIOTICS',                     # UTI / prostatitis treatment
    'PAIN_MEDICATIONS',
    'CORTICOSTEROIDS',
    'GI_MEDICATIONS',
    'NUTRITIONAL_SUPPLEMENTS',
]

# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'PSA_MONITORING',
    'LAB_MARKERS',
    'HAEMATOLOGICAL_ABNORMALITIES',
]

# ═══════════════════════════════════════════════════════════════
# INVESTIGATION CATEGORIES (imaging / specialist tests)
# ═══════════════════════════════════════════════════════════════
INVESTIGATION_CATEGORIES = [
    'ABNORMAL_IMAGING',
]

# ═══════════════════════════════════════════════════════════════
# SYMPTOM CLUSTER DEFINITIONS
# Each entry: (group_name, [list of obs categories])
# Used by build_advanced_features() for symptom clustering
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    ('URINARY',     ['URINARY_SYMPTOMS', 'HAEMATURIA']),
    ('PSA',         ['PSA_MONITORING']),
    ('BONE',        ['BONE_PAIN', 'LOWER_BACK_PAIN']),
    ('CONSTITUTIONAL', ['WEIGHT_LOSS', 'FATIGUE']),
    ('IMAGING',     ['ABNORMAL_IMAGING']),
    ('SEXUAL',      ['ERECTILE_DYSFUNCTION']),
]

# Categories considered "symptoms" for aggregate features
SYMPTOM_CATEGORIES = [
    'URINARY_SYMPTOMS', 'HAEMATURIA', 'BONE_PAIN',
    'LOWER_BACK_PAIN', 'WEIGHT_LOSS', 'FATIGUE',
    'ERECTILE_DYSFUNCTION', 'PAIN_SYMPTOMS',
    'ADDITIONAL_SYMPTOMS',
]

# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# Each entry: (feature_name, col_A, col_B)
# Creates INT_{feature_name} = col_A * col_B
# Columns must exist as has_ever flags in the merged feature matrix
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    ('urinary_plus_psa',
     'OBS_URINARY_SYMPTOMS_has_ever', 'OBS_PSA_MONITORING_has_ever'),
    ('haematuria_plus_urinary',
     'OBS_HAEMATURIA_has_ever', 'OBS_URINARY_SYMPTOMS_has_ever'),
    ('bone_pain_plus_imaging',
     'OBS_BONE_PAIN_has_ever', 'OBS_ABNORMAL_IMAGING_has_ever'),
    ('urinary_plus_alpha_blockers',
     'OBS_URINARY_SYMPTOMS_has_ever', 'MED_ALPHA_BLOCKERS_has_ever'),
    ('pain_plus_pain_meds',
     'OBS_PAIN_SYMPTOMS_has_ever', 'MED_PAIN_MEDICATIONS_has_ever'),
    ('fatigue_plus_weight_loss',
     'OBS_FATIGUE_has_ever', 'OBS_WEIGHT_LOSS_has_ever'),
    ('bone_pain_plus_back_pain',
     'OBS_BONE_PAIN_has_ever', 'OBS_LOWER_BACK_PAIN_has_ever'),
    ('urinary_plus_antibiotics',
     'OBS_URINARY_SYMPTOMS_has_ever', 'MED_ANTIBIOTICS_has_ever'),
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['PAIN_MEDICATIONS', 'CORTICOSTEROIDS']
GI_MED_CATEGORIES = ['GI_MEDICATIONS']
REPEAT_PRESCRIPTION_CATEGORY = 'ANTIBIOTICS'  # track repeat prescriptions
STEROID_CATEGORY = 'CORTICOSTEROIDS'

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# Categories whose events get exponential time-decay weighting
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'URINARY_SYMPTOMS', 'HAEMATURIA', 'BONE_PAIN', 'PSA_MONITORING',
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
    'PSA_MONITORING': {
        'Prostate specific antigen':                (0, 10000),   # ng/mL
        'Prostate specific antigen level':          (0, 10000),
    },
    'LAB_MARKERS': {
        'Serum CRP (C reactive protein) level':     (0, 600),    # mg/L
        'Plasma C reactive protein':                (0, 600),
        'Glomerular filtration rate':               (0, 200),    # mL/min/1.73m2
        'Serum alkaline phosphatase level':         (0, 2000),   # U/L (bone mets)
        'Plasma total bilirubin level':             (0, 500),    # umol/L
        'Serum testosterone level':                 (0, 60),     # nmol/L
        'Serum creatinine level':                   (0, 2000),   # umol/L
    },
    'HAEMATOLOGICAL_ABNORMALITIES': {
        'Haemoglobin concentration':                (20, 250),   # g/L
        'Platelet count':                           (0, 2000),   # x10^9/L
        'White blood cell count':                   (0, 500),    # x10^9/L
    },
}

# ═══════════════════════════════════════════════════════════════
# TEXT / NLP CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Observation categories with clinically meaningful text
TEXT_CLINICAL_CATEGORIES = [
    'URINARY_SYMPTOMS', 'PSA_MONITORING', 'BONE_PAIN',
    'ABNORMAL_IMAGING', 'HAEMATURIA', 'ERECTILE_DYSFUNCTION',
    'LAB_MARKERS', 'HAEMATOLOGICAL_ABNORMALITIES',
    'PAIN_SYMPTOMS', 'ADDITIONAL_SYMPTOMS',
]

# Regex for prostate-related terms (used to pull text from any category)
TEXT_SEARCH_PATTERN = (
    r'prostat|psa|adenocarcinoma|gleason|urinar|bladder|retention'
    r'|nocturia|hesitanc|dribbl|haematuria|bone\s*scan|metasta'
    r'|bph|benign\s*prostatic|turp|biops|digital\s*rectal'
)

# Boilerplate text to filter out
TEXT_BOILERPLATE = [
    r'normal[\s,-]*no\s*action',
    r'no\s*significant\s*pathology',
    r'within\s*normal\s*limits',
    r'unremarkable',
    r'please\s*note\s*method\s*changed',
    r'reference\s*range',
    r'adjusting\s*egfr',
    r'ckd\s*guidelines',
    r'efi\s*score',
    r'frailty',
]

# Text keyword dictionaries — {feature_name: regex_pattern}
TEXT_IMAGING_KEYWORDS = {
    'TEXT_IMG_psa_elevated':       r'psa.{0,20}(?:elevat|raised|high|abnormal|increas)',
    'TEXT_IMG_bone_lesion':        r'bone.{0,20}(?:lesion|metasta|destruction|scan|abnormal)',
    'TEXT_IMG_prostate_enlargement': r'prostat.{0,15}(?:enlarg|hypertrop|nodular|firm|irregular)',
    'TEXT_IMG_urinary_obstruction': r'(?:urinar|bladder).{0,20}(?:obstruct|retain|residual|incomplete)',
    'TEXT_IMG_biopsy_finding':     r'biops|histolog|cytolog|gleason|adenocarcinoma',
    'TEXT_IMG_dre_abnormal':       r'digital\s*rectal.{0,20}(?:abnormal|hard|nodule|irregular)|dre.{0,10}abnormal',
    'TEXT_IMG_mri_finding':        r'mri.{0,20}(?:prostat|lesion|pirads|suspicious)',
    'TEXT_IMG_normal':             r'(?:normal|no\s*abnormal|unremark|no\s*significant|within\s*normal)',
}

TEXT_SEVERITY_KEYWORDS = {
    'TEXT_SEV_worsening':    r'worsen|getting\s*worse|deteriorat|increas.{0,10}(pain|symptom|psa|frequency)',
    'TEXT_SEV_persistent':   r'persistent\s*(pain|symptom|urinar|haematuria|psa)',
    'TEXT_SEV_severe':       r'severe\s*(pain|symptom|urinar|obstruct|retention)',
    'TEXT_SEV_recurrent':    r'recurrent\s*(pain|infection|uti|haematuria|symptom)',
    'TEXT_SEV_new':          r'new\s*(symptom|pain|haematuria|lump|finding)|recently\s*(develop|start|notic)',
}

TEXT_CLINICAL_KEYWORDS = {
    'TEXT_CLIN_biopsy':         r'biops|histolog|cytolog|trus|transrectal|template',
    'TEXT_CLIN_prostate_term':  r'prostat|adenocarcinoma|gleason|pirads|turp',
    'TEXT_CLIN_bone_mets':      r'bone\s*(?:metasta|scan|lesion)|skeletal\s*survey',
    'TEXT_CLIN_psa_rising':     r'psa\s*(?:ris|increas|doubl|veloc|elevat)',
    'TEXT_CLIN_urinary':        r'(?:urinar|luts|nocturia|hesitanc|dribbl|frequen|urgenc)',
    'TEXT_CLIN_haematuria':     r'haematuria|blood\s*(?:in|urine)',
    'TEXT_CLIN_retention':      r'retention|catheter|residual\s*volume',
    'TEXT_CLIN_weight_loss':    r'weight\s*(loss|lost|losing)|appetite\s*(loss|poor|reduc)|cachex',
    'TEXT_CLIN_fatigue':        r'fatigu|tir|lethar|malaise|exhausti',
    'TEXT_CLIN_family_history': r'family\s*histor',
}

# Severity categories for text extraction (non-lab)
TEXT_SEVERITY_CATEGORIES = [
    'URINARY_SYMPTOMS', 'HAEMATURIA', 'BONE_PAIN',
    'LOWER_BACK_PAIN', 'WEIGHT_LOSS', 'FATIGUE',
    'ABNORMAL_IMAGING', 'PAIN_SYMPTOMS',
    'ERECTILE_DYSFUNCTION', 'ADDITIONAL_SYMPTOMS',
]

# BERT model
BERT_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'
N_EMBEDDING_COMPONENTS = 15

# ═══════════════════════════════════════════════════════════════
# MODELING CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SEEDS = [13, 42, 45, 77]         # 4 seeds for robust evaluation
N_SELECT_FEATURES = 265           # 225 base + 10 text + 15 TF-IDF + 15 BERT
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
