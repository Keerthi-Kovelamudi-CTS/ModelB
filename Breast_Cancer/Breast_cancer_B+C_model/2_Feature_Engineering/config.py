# ═══════════════════════════════════════════════════════════════
# BREAST CANCER — PIPELINE CONFIGURATION
# Single source of truth for all cancer-specific settings.
# Trimmed + restructured 2026-05-31: 116 obs + 19 med categories.
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

# ─── Identity ────────────────────────────────────────────────
CANCER_NAME = 'breast'
DATA_PREFIX = 'breast'
PREFIX = 'BRCA_'
LABEL_COL = 'LABEL'
WINDOWS = ["1mo", "12mo"]

# ─── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'
_RESULTS_ROOT   = SCRIPT_DIR / 'results'
FE_RESULTS      = _RESULTS_ROOT / '3_feature_engineering'
CLEANUP_RESULTS = _RESULTS_ROOT / '5_cleanup'
SANITY_RESULTS  = _RESULTS_ROOT / '1_sanity_check'
SHARED_SPLIT_PATH       = SCRIPT_DIR.parent / '3_Modeling' / 'shared_split.json'
MIN_EVENTS_PER_WINDOW   = 5

CODELIST_MAPPING_JSON = SCRIPT_DIR.parent / 'codelist2.0' / 'code_category_mapping_2.0.json'
SAVE_INTERMEDIATES = False


# ═══════════════════════════════════════════════════════════════
# OBSERVATION CATEGORIES (116) — granular curation
# ═══════════════════════════════════════════════════════════════
OBS_CATEGORIES = [
    'BREAST_BENIGN',
    'COMORBID_ASTHMA',
    'COMORBID_ASTHMA_CONTROL',
    'COMORBID_CHRONIC_RHINITIS',
    'COMORBID_COPD',
    'COMORBID_CVD',
    'COMORBID_CVD_FH_IHD_GT60',
    'COMORBID_CVD_FH_IHD_LT60',
    'COMORBID_CVD_STATIN_PROPHYL',
    'COMORBID_CVD_VARICOSE',
    'COMORBID_DEPRESSION',
    'COMORBID_DEPRESSION_SAD',
    'COMORBID_DEPRESSION_SCREEN',
    'COMORBID_DM',
    'COMORBID_DM_EYE',
    'COMORBID_DM_FOOT',
    'COMORBID_HAYFEVER',
    'COMORBID_HTN',
    'COMORBID_HYPERCHOL',
    'COMORBID_HYPOTHYROID',
    'COMORBID_IHD',
    'COMORBID_PSYCH_BEREAVEMENT',
    'COMORBID_PSYCH_STRESS',
    'COMORBID_PSYCH_STRESS_HOME',
    'FAMHX_BREAST_OVARIAN',
    'FAMHX_OTHER_CANCERS',
    'HEREDITARY_BRCA',
    'HISTORY_OOPHORECTOMY',
    'LAB_FSH',
    'LAB_FT3',
    'LAB_FT4',
    'LAB_FT4_SERUM',
    'LAB_GONADOTROPHIN',
    'LAB_HBA1C',
    'LAB_LIPIDS',
    'LAB_SPIRO_FLOW',
    'LAB_SPIRO_VOLUMES',
    'LAB_TFT',
    'LAB_TFT_TEST',
    'LAB_TPO_AB',
    'LAB_TSH',
    'LAB_LH',
    'LAB_OESTRADIOL',
    'LAB_PROLACTIN',
    'LAB_SEX_HORMONES_OTHER',

    'LAB_TSH_PLASMA',
    'LIFE_ALCOHOL',
    'LIFE_ALC_EDU',
    'LIFE_ALC_SCREEN',
    'LIFE_BMI',
    'LIFE_DIET',
    'LIFE_PHYSACT',
    'LIFE_SMOKE_CO',
    'LIFE_SMOKING',
    'OVARIAN_CONDITIONS',
    'REPRO_ATROPHIC_VAG',
    'REPRO_MENOPAUSE',
    'REPRO_PARITY',
    'REPRO_PMS',
    'REPRO_TAH',
    'REPRO_TAH_BSO',
    'SYMPTOM_BREAST_LUMP',
    'SYMPTOM_BREAST_PAIN',
    'SYMPTOM_NIPPLE',
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (19)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'ANXIOLYTICS_HYPNOTICS',
    'DM_MONITORING',
    'HRT',
    'OC',
    'RESPIRATORY_INHALERS',
]

# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'LAB_HBA1C',
    'LAB_LIPIDS',
    'LAB_FT3', 'LAB_FT4', 'LAB_FT4_PLASMA', 'LAB_FT4_SERUM',
    'LAB_T4', 'LAB_T4_SERUM', 'LAB_TSH', 'LAB_TSH_PLASMA',
    'LAB_FSH', 'LAB_LH', 'LAB_PROLACTIN',
    'LAB_OESTRADIOL', 'LAB_SEX_HORMONES_OTHER',
    'LAB_SPIRO_VOLUMES', 'LAB_SPIRO_FLOW',
]

# ═══════════════════════════════════════════════════════════════
# LAB DIRECTION OF BADNESS
# ═══════════════════════════════════════════════════════════════
LAB_BAD_DIRECTION = {
    'LAB_HBA1C':         'up',
    'LAB_LIPIDS':        None,   # mixed
    'LAB_TSH':           None,
    'LAB_FT4':           None,
    'LAB_SPIRO_VOLUMES': 'down',  # low FEV1/FVC = bad
    'LAB_SPIRO_FLOW':    'down',  # low PEF = bad
}

LAB_WORSENING_RULES = {
    # HbA1c rising AND last >= 48 mmol/mol (diabetes threshold)
    'LAB_HBA1C': lambda slope, first, last, mean, mx, n: (
        (n.fillna(0) >= 2) & (slope.fillna(0) > 0) & (last.fillna(0) >= 48.0)
    ).astype(int),
}

# ═══════════════════════════════════════════════════════════════
# INVESTIGATION CATEGORIES
# ═══════════════════════════════════════════════════════════════
INVESTIGATION_CATEGORIES = [
    'LAB_SPIRO_FLOW', 'LAB_HBA1C', 'LAB_TFT_TEST',
    'COMORBID_DEPRESSION_SCREEN', 'COMORBID_DM_FOOT',
    'LAB_BP_NORMAL', 'LAB_BP_BORDERLINE', 'LAB_BP_BORDERLINE_RAISED',
]

# ═══════════════════════════════════════════════════════════════
# CLUSTER DEFINITIONS
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    # Cardiometabolic (BC risk via insulin resistance / oestrogen pathway)
    ('CARDIOMETABOLIC', ['COMORBID_CVD', 'COMORBID_IHD', 'COMORBID_HTN',
                          'COMORBID_HYPERCHOL', 'COMORBID_DM']),
    # Diabetes monitoring engagement (high-utilisation proxy)
    ('DM_MONITORING_ENG', ['COMORBID_DM', 'COMORBID_DM_FOOT', 'COMORBID_DM_EYE',
                            'LAB_HBA1C', 'DM_MONITORING']),
    # Mental health system
    ('MENTAL_HEALTH', ['COMORBID_DEPRESSION', 'COMORBID_DEPRESSION_MILD',
                        'COMORBID_DEPRESSION_MOD', 'COMORBID_DEPRESSION_SAD',
                        'COMORBID_DEPRESSION_SX', 'COMORBID_INSOMNIA',
                        'COMORBID_HADS', 'COMORBID_MH_CPA', 'COMORBID_MH_CARE_PLAN',
                        'COMORBID_PSYCH_STRESS', 'COMORBID_PSYCH_STRESS_HOME',
                        'COMORBID_PSYCH_BEREAVEMENT',
                        'ANTIDEPRESSANTS', 'ANTIPSYCHOTICS', 'ANXIOLYTICS_HYPNOTICS']),
    # CV medication intensity
    ('CV_MEDS', ['ACE_INHIBITORS', 'ARBS', 'BETA_BLOCKERS', 'CCB',
                  'DIURETICS', 'STATINS', 'ANTIPLATELETS',
                  'ALPHA_BLOCKERS_CENTRAL']),
    # Reproductive + HRT (strong BC risk modifier system)
    ('REPRODUCTIVE_HRT', ['REPRO_TAH', 'REPRO_TAH_BSO', 'REPRO_PMS',
                           'REPRO_ATROPHIC_VAG', 'REPRO_MENOPAUSE',
                           'REPRO_MENSTRUAL_FLOW', 'REPRO_PARITY',
                           'HRT', 'OC']),
    # Hereditary BC risk
    ('HEREDITARY_RT_RISK', ['FAMHX_BREAST_OVARIAN', 'HISTORY_HODGKIN_RT']),
    # Asthma system
    ('ASTHMA_SYSTEM', ['COMORBID_ASTHMA', 'COMORBID_ASTHMA_CONTROL',
                        'RESPIRATORY_INHALERS']),
    # Lifestyle composite
    ('LIFESTYLE', ['LIFE_BMI', 'LIFE_WEIGHT_SX', 'LIFE_ALCOHOL',
                    'LIFE_ALC_EDU', 'LIFE_ALC_SCREEN', 'LIFE_DIET',
                    'LIFE_PHYSACT', 'LIFE_SMOKING', 'LIFE_SMOKE_CO']),
]

# Symptom / system categories for aggregate features
SYMPTOM_CATEGORIES = [
    'COMORBID_DEPRESSION', 'COMORBID_INSOMNIA', 'COMORBID_MEMORY_LOSS',
    'COMORBID_HADS', 'COMORBID_FLU_LIKE', 'COMORBID_HAYFEVER',
    'COMORBID_PSYCH_STRESS', 'COMORBID_PSYCH_BEREAVEMENT',
    'REPRO_MENOPAUSE', 'REPRO_PMS', 'REPRO_MENSTRUAL_FLOW',
    'LIFE_BMI', 'LIFE_WEIGHT_SX',
]

# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # Cardiometabolic burden treated
    ('htn_plus_statins',
     'OBS_COMORBID_HTN_has_ever', 'MED_STATINS_has_ever'),
    ('htn_plus_acei',
     'OBS_COMORBID_HTN_has_ever', 'MED_ACE_INHIBITORS_has_ever'),
    ('htn_plus_arb',
     'OBS_COMORBID_HTN_has_ever', 'MED_ARBS_has_ever'),
    ('hyperchol_plus_statins',
     'OBS_COMORBID_HYPERCHOL_has_ever', 'MED_STATINS_has_ever'),
    # Diabetes engagement
    ('dm_obs_plus_monitoring',
     'OBS_COMORBID_DM_has_ever', 'MED_DM_MONITORING_has_ever'),
    # Mental health treated
    ('depression_plus_ssri',
     'OBS_COMORBID_DEPRESSION_has_ever', 'MED_ANTIDEPRESSANTS_has_ever'),
    ('anxiety_plus_anxiolytic',
     'OBS_COMORBID_DEPRESSION_has_ever', 'MED_ANXIOLYTICS_HYPNOTICS_has_ever'),
    # Smoking + cessation
    ('smoking_plus_cessation',
     'OBS_LIFE_SMOKING_has_ever', 'MED_SMOKING_CESSATION_has_ever'),
    # Reproductive × HRT (highest-leverage BC interaction)
    ('hrt_plus_menopause',
     'MED_HRT_has_ever', 'OBS_REPRO_MENOPAUSE_has_ever'),
    ('hrt_plus_atrophic_vag',
     'MED_HRT_has_ever', 'OBS_REPRO_ATROPHIC_VAG_has_ever'),
    ('tah_bso_plus_menopause',
     'OBS_REPRO_TAH_BSO_has_ever', 'OBS_REPRO_MENOPAUSE_has_ever'),
    # Family hx breast × HRT exposure (multiplicative BC risk)
    ('famhx_breast_plus_hrt',
     'OBS_FAMHX_BREAST_OVARIAN_has_ever', 'MED_HRT_has_ever'),
    ('famhx_breast_plus_oc',
     'OBS_FAMHX_BREAST_OVARIAN_has_ever', 'MED_OC_has_ever'),
    # Respiratory burden treated
    ('asthma_plus_inhaler',
     'OBS_COMORBID_ASTHMA_has_ever', 'MED_RESPIRATORY_INHALERS_has_ever'),
    # VTE marker (LMWH ~ phlebitis hx)
    ('phlebitis_plus_lmwh',
     'OBS_COMORBID_CVD_PHLEBITIS_has_ever', 'MED_LMWH_has_ever'),
]

# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = []  # no pain category in EMIS breast curation
GI_MED_CATEGORIES = ['PPI_H2_CHRONIC_GI']
REPEAT_PRESCRIPTION_CATEGORY = 'PPI_H2_CHRONIC_GI'
STEROID_CATEGORY = 'RESPIRATORY_INHALERS'

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'LIFE_BMI',
    'LIFE_SMOKING',
    'LIFE_ALCOHOL',
    'COMORBID_DEPRESSION',
    'COMORBID_INSOMNIA',
    'REPRO_MENOPAUSE',
    'HRT',                # HRT exposure recency matters
    'OC',                 # OC use recency matters
]

# ═══════════════════════════════════════════════════════════════
# LAB RANGES
# ═══════════════════════════════════════════════════════════════
LAB_RANGES = {
    'LAB_HBA1C': {
        'A1C':                  (10, 200),  # mmol/mol IFCC
        'Hb. A1C':              (10, 200),
    },
    'LAB_TSH': {
        'TSH':                  (0.001, 100),
        'thyroid stimulating hormone': (0.001, 100),
    },
    'LAB_LIPIDS': {
        'cholesterol':          (0, 20),
        'HDL':                  (0, 5),
        'LDL':                  (0, 15),
        'triglyceride':         (0, 30),
    },
    'LAB_SPIRO_FLOW': {
        'Peak expiratory flow rate':  (0, 1200),
    },
    'LAB_SPIRO_VOLUMES': {
        'Forced expired volume':      (0, 10),
        'Forced vital capacity':      (0, 15),
        'FEV1/FVC':                   (0, 1.5),
    },
    'LAB_FT4': {
        'Free thyroxine':       (0, 80),
        'Free T4':              (0, 80),
        'T4':                   (0, 200),
    },
}

# ═══════════════════════════════════════════════════════════════
# MODELING CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SEEDS = [42]
N_SELECT_FEATURES = 500
N_SELECT_FEATURES_PER_WINDOW = {
    '1mo':  500,
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
