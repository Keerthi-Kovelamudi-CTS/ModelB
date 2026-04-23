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

# Downsample ratio for training — if set, sanity check keeps
#   N_neg_kept = round(DOWNSAMPLE_NEG_TO_POS_RATIO × N_pos) negative patients
# Set to None (or remove) for natural prevalence.
DOWNSAMPLE_NEG_TO_POS_RATIO = 1.0
DOWNSAMPLE_SEED = 42

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
# OBSERVATION CATEGORIES (from actual prostate SQL data)
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
    'IPSS',
    'LAB_FLAGS',
    'LIVER_FUNCTION',
    'LUTS',
    'PAIN_PELVIC_BONE',
    'PROSTATIC_CONDITIONS',
    'PSA',
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
# MEDICATION CATEGORIES (from actual prostate SQL data)
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
    'URINE_MARKERS',
    'VITAMIN_D',
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
    ('PSA',            ['PSA']),
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
}

# ═══════════════════════════════════════════════════════════════
# TEXT / NLP CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Observation categories with clinically meaningful text
TEXT_CLINICAL_CATEGORIES = [
    'LUTS', 'PSA', 'PAIN_PELVIC_BONE', 'UROLOGY_IMAGING',
    'HAEMATURIA', 'ERECTILE_DYSFUNCTION', 'DRE',
    'PROSTATIC_CONDITIONS', 'URINARY_RETENTION',
    'CONSTITUTIONAL', 'UROLOGY_PATHWAY', 'CYSTOSCOPY',
    'FAMILY_HISTORY', 'IPSS', 'CATHETER',
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
    'LUTS', 'HAEMATURIA', 'PAIN_PELVIC_BONE',
    'BONE_MUSCLE', 'CONSTITUTIONAL', 'WEIGHT',
    'UROLOGY_IMAGING', 'URINARY_RETENTION',
    'ERECTILE_DYSFUNCTION', 'PROSTATIC_CONDITIONS',
]

# BERT model
# NeuML/pubmedbert-base-embeddings is a PubMedBERT checkpoint purpose-built
# for sentence embeddings, shipped as safetensors (no torch.load CVE block)
# and drop-in compatible with sentence-transformers. Previous choice
# (pritamdeka/S-PubMedBert-MS-MARCO) failed to load under torch >= 2.6 and
# silently fell back to MiniLM during training.
BERT_MODEL_NAME = 'NeuML/pubmedbert-base-embeddings'
BERT_FALLBACK_MODELS = [
    'pritamdeka/S-PubMedBert-MS-MARCO',   # old choice, try as backup
    'all-MiniLM-L6-v2',                   # last-resort general-purpose
]
N_EMBEDDING_COMPONENTS = 15

# ═══════════════════════════════════════════════════════════════
# MODELING CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SEEDS = [42]                     # single best seed for v3 pilot
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
