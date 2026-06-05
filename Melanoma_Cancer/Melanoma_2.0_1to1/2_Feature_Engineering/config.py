
# ═══════════════════════════════════════════════════════════════
# MELANOMA CANCER — 1:1 COHORT VARIANT
# ─────────────────────────────────────────────────────────────
# This folder was scaffolded from Prostate_2.0_1to1. The pipeline
# code (3_pipeline.py, modeling, explainability) is generic and
# works as-is. The cancer-specific bits below are PROSTATE
# PLACEHOLDERS — replace them before running:
#
#   ☐ OBS_CATEGORIES     — melanoma symptom/observation categories
#   ☐ MED_CATEGORIES     — melanoma medication categories
#   ☐ LAB_CATEGORIES     — melanoma relevant labs
#   ☐ LAB_BAD_DIRECTION  — clinical direction-of-bad per lab
#   ☐ LAB_WORSENING_RULES— optional vectorised worsening rules
#   ☐ LAB_RANGES         — physiologic bounds for outlier filter
#   ☐ CLUSTER_DEFINITIONS— melanoma symptom clusters
#   ☐ INTERACTION_PAIRS  — melanoma-relevant interactions
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
CANCER_NAME = 'melanoma'
DATA_PREFIX = 'melanoma'          # CSV filename prefix: prostate_3mo_obs.csv
PREFIX = 'MELA_'                 # Feature prefix for cancer-specific columns
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
    # Melanoma is largely procedural/observational — these are the visible-skin signals
    'MOLE_NAEVUS_PIGMENTED_LESION',  # PRIMARY signal — naevi, lentigos, dysplastic naevi
    'NICE_7PCL_FEATURES',            # KEY signal — NICE 7-point checklist (irregular pigmentation/border, change in size, etc.)
    'SKIN_LESION_PRESENTATION',      # general skin lesion observations
    'RISK_FACTOR',                   # Fitzpatrick skin type
    'SUN_DAMAGE_PRE_MALIGNANT',      # solar keratosis, Bowen's disease (pre-malignant)
    'PRIOR_SKIN_CANCER_BCC_SCC',     # prior BCC/SCC raises risk
    'OTHER_SKIN_LESIONS',            # benign DDx
    'DERMATOSCOPY_PHOTOGRAPHY',      # diagnostic dermoscopy / lesion photography
    'PRIOR_SKIN_PROCEDURES_BIOPSY',  # excision / biopsy events
    'WIDE_EXCISION_GRAFTING',        # wide local excision (definitive treatment / strong signal)
    'CRYOTHERAPY',                   # cryo for benign / pre-malignant lesions
    'SKIN_SYMPTOMS',                 # rash, pruritus
    'SKIN_TREATMENT_WOUND',          # post-procedure care
    'SUTURE_POST_OP_MINOR_SURGERY',
    'MINOR_SURGERY_ADMIN',
    'HISTOLOGY',                     # histology-laboratory orders / abnormal flag
    'FAMILY_HISTORY',                # family history of melanoma / skin cancer
    'PERSONAL_HISTORY',              # H/O malignant melanoma (prior diagnosis flag)
    'DERMATOLOGY_REFERRAL_CLINIC',   # referral or seen-in-clinic events
    'PLASTIC_SURGERY',               # plastic surgery referrals
    'CLINICAL_SIGNS',                # palpable lumps, lymph node masses
    'SKIN_INFECTION_WORKUP',         # wound swabs, infection signs
    'VIROLOGY_SKIN_RELATED',         # HSV testing
    'CRYOTHERAPY_WART_TREATMENT',    # wart treatment (often co-occurs in dermatology)
    'SKIN_CONDITIONS',               # benign skin conditions (DDx)
    'INSECT_BITES',
    'OTHERS',                        # uncategorised but predictive codes (kept so they get base FE features)
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (from actual prostate SQL data)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'ACTINIC_PRE_MALIGNANT_TREATMENT', # 5FU, imiquimod, ingenol — pre-malignant treatment
    'ANTIFUNGAL_SYSTEMIC',
    'SKIN_ANTIBIOTICS_ORAL',
    'LOCAL_ANAESTHETICS_SKIN',         # lidocaine — strong signal for skin procedure
    'WOUND_DRESSINGS',
    'SUTURES_WOUND_TAPE',
    'WOUND_CARE',
    'IMMUNOSUPPRESSANT_SYSTEMIC',      # azathioprine, methotrexate (raises melanoma risk)
    'IMMUNOSUPPRESSANT_CORTICOSTEROIDS', # systemic prednisolone
    'ANTIVIRAL_SYSTEMIC',
    'ANTIBIOTIC_IMMUNOCOMPROMISE',
    'TOPICAL_STEROIDS_SKIN',
    'TOPICAL_ANTIBIOTICS_SKIN',
    'TOPICAL_STEROID_SCALP',
    'TOPICAL_SKIN_OTHER',
    'ANTIVIRAL_SKIN',
    'OTHER_PREDICTIVE_MED',
]
# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    # Melanoma cohort has minimal lab content in this codelist — most signal is
    # procedural/observational. Leave empty until lab markers (LDH, S100, eosinophils)
    # are added.
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
    # No labs configured for melanoma in this codelist. See note above.
    'BMI':         None,
    'BODY_WEIGHT':         None,
    'IDEAL_BODY_WEIGHT':         None,
    'SYSTOLIC_BP':         'up',
    'DIASTOLIC_BP':         'up',
    'HEART_RATE':         None,
}

# Optional pluggable worsening rules per category (overrides LAB_BAD_DIRECTION).
LAB_WORSENING_RULES = {
    # No labs configured for melanoma. Add LDH/S100 rules when lab codes are added.
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
    'DERMATOSCOPY_PHOTOGRAPHY',     # diagnostic dermoscopy
    'HISTOLOGY',                    # histopath orders
    'PRIOR_SKIN_PROCEDURES_BIOPSY', # biopsy / excision events
]
# ═══════════════════════════════════════════════════════════════
# SYMPTOM CLUSTER DEFINITIONS
# Each entry: (group_name, [list of obs categories])
# Used by build_advanced_features() for symptom clustering
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    ('MOLE_LESION',         ['MOLE_NAEVUS_PIGMENTED_LESION', 'NICE_7PCL_FEATURES', 'SKIN_LESION_PRESENTATION', 'OTHER_SKIN_LESIONS']),
    ('DERMATOLOGY',         ['DERMATOSCOPY_PHOTOGRAPHY', 'DERMATOLOGY_REFERRAL_CLINIC', 'HISTOLOGY']),
    ('PROCEDURE',           ['PRIOR_SKIN_PROCEDURES_BIOPSY', 'WIDE_EXCISION_GRAFTING', 'CRYOTHERAPY', 'SKIN_TREATMENT_WOUND', 'SUTURE_POST_OP_MINOR_SURGERY', 'MINOR_SURGERY_ADMIN']),
    ('RISK',                ['RISK_FACTOR', 'SUN_DAMAGE_PRE_MALIGNANT', 'FAMILY_HISTORY', 'PERSONAL_HISTORY', 'PRIOR_SKIN_CANCER_BCC_SCC']),
    ('INFECTION_WORKUP',    ['SKIN_INFECTION_WORKUP', 'VIROLOGY_SKIN_RELATED']),
    ('SYMPTOMS',            ['SKIN_SYMPTOMS', 'CLINICAL_SIGNS']),
    ('OTHER_DERMATOLOGY',   ['SKIN_CONDITIONS', 'CRYOTHERAPY_WART_TREATMENT', 'INSECT_BITES']),
    ('PLASTIC',             ['PLASTIC_SURGERY']),
]
# Categories considered "symptoms" for aggregate features
SYMPTOM_CATEGORIES = [
    'MOLE_NAEVUS_PIGMENTED_LESION', 'NICE_7PCL_FEATURES',
    'SKIN_LESION_PRESENTATION', 'SKIN_SYMPTOMS', 'CLINICAL_SIGNS',
]
# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# Each entry: (feature_name, col_A, col_B)
# Creates INT_{feature_name} = col_A * col_B
# Columns must exist as has_ever flags in the merged feature matrix
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # Classic melanoma diagnostic pathway
    ('mole_plus_dermoscopy',
     'OBS_MOLE_NAEVUS_PIGMENTED_LESION_has_ever', 'OBS_DERMATOSCOPY_PHOTOGRAPHY_has_ever'),
    ('mole_plus_dermatology_referral',
     'OBS_MOLE_NAEVUS_PIGMENTED_LESION_has_ever', 'OBS_DERMATOLOGY_REFERRAL_CLINIC_has_ever'),
    ('7pcl_plus_dermatology',
     'OBS_NICE_7PCL_FEATURES_has_ever', 'OBS_DERMATOLOGY_REFERRAL_CLINIC_has_ever'),
    ('mole_plus_excision',
     'OBS_MOLE_NAEVUS_PIGMENTED_LESION_has_ever', 'OBS_PRIOR_SKIN_PROCEDURES_BIOPSY_has_ever'),
    ('skin_lesion_change_plus_excision',
     'OBS_SKIN_LESION_PRESENTATION_has_ever', 'OBS_PRIOR_SKIN_PROCEDURES_BIOPSY_has_ever'),
    ('biopsy_plus_histology',
     'OBS_PRIOR_SKIN_PROCEDURES_BIOPSY_has_ever', 'OBS_HISTOLOGY_has_ever'),
    ('mole_plus_histology',
     'OBS_MOLE_NAEVUS_PIGMENTED_LESION_has_ever', 'OBS_HISTOLOGY_has_ever'),
    ('prior_skin_cancer_plus_personal_history',
     'OBS_PRIOR_SKIN_CANCER_BCC_SCC_has_ever', 'OBS_PERSONAL_HISTORY_has_ever'),
    ('family_history_plus_dermatology',
     'OBS_FAMILY_HISTORY_has_ever', 'OBS_DERMATOLOGY_REFERRAL_CLINIC_has_ever'),
    ('skin_lesion_plus_wide_excision',
     'OBS_SKIN_LESION_PRESENTATION_has_ever', 'OBS_WIDE_EXCISION_GRAFTING_has_ever'),
    ('mole_plus_plastic_surgery',
     'OBS_MOLE_NAEVUS_PIGMENTED_LESION_has_ever', 'OBS_PLASTIC_SURGERY_has_ever'),
    ('sun_damage_plus_dermatology',
     'OBS_SUN_DAMAGE_PRE_MALIGNANT_has_ever', 'OBS_DERMATOLOGY_REFERRAL_CLINIC_has_ever'),
    ('mole_plus_local_anaesthetic',
     'OBS_MOLE_NAEVUS_PIGMENTED_LESION_has_ever', 'MED_LOCAL_ANAESTHETICS_SKIN_has_ever'),
    ('immunosuppression_plus_skin_lesion',
     'MED_IMMUNOSUPPRESSANT_SYSTEMIC_has_ever', 'OBS_SKIN_LESION_PRESENTATION_has_ever'),
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = []  # no opioid/NSAID category in melanoma codelist
GI_MED_CATEGORIES = []
REPEAT_PRESCRIPTION_CATEGORY = 'TOPICAL_STEROIDS_SKIN'  # often long-term repeat for chronic skin
STEROID_CATEGORY = 'IMMUNOSUPPRESSANT_CORTICOSTEROIDS'  # systemic steroids if usedGI_MED_CATEGORIES = ['ANTICHOLINERGICS']

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# Categories whose events get exponential time-decay weighting
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'MOLE_NAEVUS_PIGMENTED_LESION', 'NICE_7PCL_FEATURES',
    'SKIN_LESION_PRESENTATION', 'SKIN_SYMPTOMS', 'CLINICAL_SIGNS',
    'DERMATOLOGY_REFERRAL_CLINIC',
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
    # No labs in melanoma codelist; LDH/S100 ranges TBD when added.
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
