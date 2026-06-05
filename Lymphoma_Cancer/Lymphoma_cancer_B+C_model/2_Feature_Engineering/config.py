
# ═══════════════════════════════════════════════════════════════
# LYMPHOMA CANCER — 1:1 COHORT VARIANT
# ─────────────────────────────────────────────────────────────
# This folder was scaffolded from Prostate_2.0_1to1. The pipeline
# code (3_pipeline.py, modeling, explainability) is generic and
# works as-is. The cancer-specific bits below are PROSTATE
# PLACEHOLDERS — replace them before running:
#
#   ☐ OBS_CATEGORIES     — lymphoma symptom/observation categories
#   ☐ MED_CATEGORIES     — lymphoma medication categories
#   ☐ LAB_CATEGORIES     — lymphoma relevant labs
#   ☐ LAB_BAD_DIRECTION  — clinical direction-of-bad per lab
#   ☐ LAB_WORSENING_RULES— optional vectorised worsening rules
#   ☐ LAB_RANGES         — physiologic bounds for outlier filter
#   ☐ CLUSTER_DEFINITIONS— lymphoma symptom clusters
#   ☐ INTERACTION_PAIRS  — lymphoma-relevant interactions
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
CANCER_NAME = 'lymphoma'
DATA_PREFIX = 'lymphoma'          # CSV filename prefix: prostate_3mo_obs.csv
PREFIX = 'LYMP_'                 # Feature prefix for cancer-specific columns
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
    'B_SYMPTOMS',                  # PRIMARY signal — night sweats, fever, weight loss, fatigue
    'LYMPHADENOPATHY',             # PRIMARY signal — peripheral / cervical / mediastinal nodes, neck/groin/axillary masses
    'SPLENOMEGALY',                # KEY signal — Hodgkin / NHL hallmark
    'SKIN_SYMPTOMS',               # pruritus (Hodgkin signature) + rashes (cutaneous T-cell lymphoma)
    'HAEMATOLOGICAL_ABNORMALITIES',# lymphocytosis, lymphopenia, anaemia, paraproteinaemia, MGUS
    'INFECTION_MARKERS',           # EBV, CMV, glandular fever — viral lymphoma associations
    'AUTOIMMUNE_IMMUNE',           # RA, vasculitis — lymphoma risk factors
    'CONSTITUTIONAL_SYMPTOMS',     # general malaise, GI, respiratory, ascites
    'RECURRENT_INFECTIONS',        # immunoparesis (especially CLL/NHL)
    'UNEXPLAINED_MASSES',          # lumps, testicular masses, pulmonary nodules, tonsillar enlargement
    'ABNORMAL_IMAGING',            # CXR / CT / MRI / US flagged abnormal
    'RISK_SCORES',                 # QAdmissions, PARR-30
    'PAIN_SYMPTOMS',               # spinal cord compression, abdominal/loin pain
    'HIGH_RISK_CONDITIONS',        # COVID-19 high-risk flags (proxy for immunocompromise)
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION CATEGORIES (from actual prostate SQL data)
# ═══════════════════════════════════════════════════════════════
MED_CATEGORIES = [
    'ANTIBIOTICS',                 # broad — used heavily for recurrent infections in lymphoma cohort
    'ANTIFUNGALS',
    'ANTIVIRALS',                  # aciclovir prophylaxis
    'GI_MEDICATIONS',              # PPIs, H2 blockers, antispasmodics
    'PAIN_ANTIHISTAMINES',         # mixed bag — NSAIDs, codeine, antihistamines (pruritus)
    'CORTICOSTEROIDS',             # prednisolone, dexamethasone — actual lymphoma treatment
    'COUGH_SUPPRESSANTS',
    'ANAEMIA_TREATMENT',           # iron, B12, folate
    'NUTRITIONAL_SUPPLEMENTS',     # cachexia / weight loss
    'OTHER_PREDICTIVE_MED',
]
# ═══════════════════════════════════════════════════════════════
# LAB CATEGORIES (obs rows that carry numeric VALUE)
# ═══════════════════════════════════════════════════════════════
LAB_CATEGORIES = [
    'LAB_MARKERS',                 # LDH, B2M, CEA, vitamin D, GFR, bilirubin, paraproteins
                                    # immunoglobulins (G/M/A), light chains, CRP, immunofixation
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
    # LAB_MARKERS bucket is heterogeneous: LDH/B2M/paraproteins UP=bad, immunoglobulins DOWN=bad
    # (immunoparesis), GFR DOWN=bad. Use LAB_WORSENING_RULES below for term-specific logic
    # once term names are inspected from data.
    'LAB_MARKERS': None,
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
    # NOTE: lymphoma labs are heterogeneous within LAB_MARKERS. Useful rules need
    # term-level filtering (e.g., LDH rising AND >250). Add when terms are known.
    # The pipeline still computes slope + pct_change + trend_r per term so signals
    # are preserved without binary worsening flags here.
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
    'ABNORMAL_IMAGING',            # CXR / CT / MRI abnormal flags
    'LAB_MARKERS',                 # extensive lymphoma workup labs
    'INFECTION_MARKERS',           # viral serology (EBV/CMV)
]
# ═══════════════════════════════════════════════════════════════
# SYMPTOM CLUSTER DEFINITIONS
# Each entry: (group_name, [list of obs categories])
# Used by build_advanced_features() for symptom clustering
# ═══════════════════════════════════════════════════════════════
CLUSTER_DEFINITIONS = [
    ('B_CLUSTER',           ['B_SYMPTOMS', 'CONSTITUTIONAL_SYMPTOMS']),
    ('LYMPHATIC',           ['LYMPHADENOPATHY', 'SPLENOMEGALY', 'UNEXPLAINED_MASSES']),
    ('HAEMATOLOGICAL',      ['HAEMATOLOGICAL_ABNORMALITIES', 'LAB_MARKERS']),
    ('INFECTION_PRONE',     ['INFECTION_MARKERS', 'RECURRENT_INFECTIONS']),
    ('IMAGING',             ['ABNORMAL_IMAGING']),
    ('SKIN',                ['SKIN_SYMPTOMS']),
    ('PAIN',                ['PAIN_SYMPTOMS']),
    ('RISK',                ['RISK_SCORES', 'HIGH_RISK_CONDITIONS', 'AUTOIMMUNE_IMMUNE']),
]
# Categories considered "symptoms" for aggregate features
SYMPTOM_CATEGORIES = [
    'B_SYMPTOMS', 'LYMPHADENOPATHY', 'SPLENOMEGALY',
    'SKIN_SYMPTOMS', 'CONSTITUTIONAL_SYMPTOMS',
    'UNEXPLAINED_MASSES', 'PAIN_SYMPTOMS',
]
# ═══════════════════════════════════════════════════════════════
# INTERACTION FEATURE PAIRS
# Each entry: (feature_name, col_A, col_B)
# Creates INT_{feature_name} = col_A * col_B
# Columns must exist as has_ever flags in the merged feature matrix
# ═══════════════════════════════════════════════════════════════
INTERACTION_PAIRS = [
    # Classic lymphoma diagnostic patterns
    ('lymphadenopathy_plus_b_symptoms',
     'OBS_LYMPHADENOPATHY_has_ever', 'OBS_B_SYMPTOMS_has_ever'),
    ('lymphadenopathy_plus_imaging',
     'OBS_LYMPHADENOPATHY_has_ever', 'OBS_ABNORMAL_IMAGING_has_ever'),
    ('splenomegaly_plus_lymphadenopathy',
     'OBS_SPLENOMEGALY_has_ever', 'OBS_LYMPHADENOPATHY_has_ever'),
    ('splenomegaly_plus_b_symptoms',
     'OBS_SPLENOMEGALY_has_ever', 'OBS_B_SYMPTOMS_has_ever'),
    ('haem_abn_plus_lab_markers',
     'OBS_HAEMATOLOGICAL_ABNORMALITIES_has_ever', 'LAB_LAB_MARKERS_has_ever'),
    ('recurrent_infections_plus_antibiotics',
     'OBS_RECURRENT_INFECTIONS_has_ever', 'MED_ANTIBIOTICS_has_ever'),
    ('ebv_plus_b_symptoms',
     'OBS_INFECTION_MARKERS_has_ever', 'OBS_B_SYMPTOMS_has_ever'),
    ('mass_plus_imaging',
     'OBS_UNEXPLAINED_MASSES_has_ever', 'OBS_ABNORMAL_IMAGING_has_ever'),
    ('skin_plus_lymphadenopathy',
     'OBS_SKIN_SYMPTOMS_has_ever', 'OBS_LYMPHADENOPATHY_has_ever'),
    ('pruritus_plus_b_symptoms',
     'OBS_SKIN_SYMPTOMS_has_ever', 'OBS_B_SYMPTOMS_has_ever'),
    ('corticosteroids_plus_b_symptoms',
     'MED_CORTICOSTEROIDS_has_ever', 'OBS_B_SYMPTOMS_has_ever'),
    ('lymphadenopathy_plus_recurrent_infections',
     'OBS_LYMPHADENOPATHY_has_ever', 'OBS_RECURRENT_INFECTIONS_has_ever'),
    ('autoimmune_plus_lymphadenopathy',
     'OBS_AUTOIMMUNE_IMMUNE_has_ever', 'OBS_LYMPHADENOPATHY_has_ever'),
    ('weight_loss_plus_anaemia_treatment',
     'OBS_B_SYMPTOMS_has_ever', 'MED_ANAEMIA_TREATMENT_has_ever'),
]
# ═══════════════════════════════════════════════════════════════
# MEDICATION ESCALATION CONFIG
# ═══════════════════════════════════════════════════════════════
PAIN_MED_CATEGORIES = ['PAIN_ANTIHISTAMINES']
GI_MED_CATEGORIES = ['GI_MEDICATIONS']
REPEAT_PRESCRIPTION_CATEGORY = 'ANTIBIOTICS'  # recurrent infections drive repeat prescriptions
STEROID_CATEGORY = 'CORTICOSTEROIDS'  # systemic steroids — actual lymphoma treatment proxyGI_MED_CATEGORIES = ['ANTICHOLINERGICS']

# ═══════════════════════════════════════════════════════════════
# TIME-DECAY CATEGORIES
# Categories whose events get exponential time-decay weighting
# ═══════════════════════════════════════════════════════════════
DECAY_CATEGORIES = [
    'B_SYMPTOMS', 'LYMPHADENOPATHY', 'SPLENOMEGALY',
    'HAEMATOLOGICAL_ABNORMALITIES', 'RECURRENT_INFECTIONS',
    'UNEXPLAINED_MASSES', 'ABNORMAL_IMAGING', 'SKIN_SYMPTOMS',
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
    # TODO: lymphoma-specific lab term names + plausible bounds.
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
