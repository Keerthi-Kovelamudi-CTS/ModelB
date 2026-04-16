# ═══════════════════════════════════════════════════════════════
# FEATURE DICTIONARY — Maps every feature to a clinical explanation
#
# Three explainability classes:
#   'direct'   — Directly maps to a clinical concept a GP would understand
#   'indirect'  — Needs a sentence but is clinically meaningful
#   'opaque'    — Model internal / statistical artifact, not explainable
#
# The enhancement loop uses this to decide what to keep/remove.
# Production inference uses this to generate clinician-friendly reports.
#
# HOW TO MAINTAIN:
#   - When you add features to the FE pipeline, add them here
#   - Run 2_audit_features.py to find features missing from this dict
# ═══════════════════════════════════════════════════════════════

# ─── Explainability class definitions ────────────────────────
# 'direct':   keep always — these ARE the clinical signal
# 'indirect': keep if important — need a good sentence for the report
# 'opaque':   candidates for removal if important — can't explain to clinician

EXPLAINABILITY_CLASS = {
    # ══════════════════════════════════════════════════════════
    # DEMOGRAPHICS — always direct
    # ══════════════════════════════════════════════════════════
    'AGE_AT_INDEX':     'direct',
    'AGE_BAND':         'direct',
    'AGE_over_50':      'direct',
    'AGE_over_65':      'direct',
    'AGE_over_75':      'direct',

    # ══════════════════════════════════════════════════════════
    # PROSTATE-SPECIFIC — always direct
    # ══════════════════════════════════════════════════════════
    'PROST_':           'direct',

    # ══════════════════════════════════════════════════════════
    # OBSERVATION FEATURES — direct (per-category symptom data)
    # ══════════════════════════════════════════════════════════
    'OBS_':             'direct',

    # ══════════════════════════════════════════════════════════
    # LAB VALUES — direct (actual test results)
    # ══════════════════════════════════════════════════════════
    'LABTERM_':         'direct',
    'LAB_TRAJ_':        'indirect',   # trajectory needs a sentence

    # ══════════════════════════════════════════════════════════
    # MEDICATION — direct (what drugs prescribed)
    # ══════════════════════════════════════════════════════════
    'MED_ESC_':         'indirect',   # escalation pattern
    'MEDCAT_':          'direct',
    'MEDREC_':          'indirect',

    # ══════════════════════════════════════════════════════════
    # CLINICAL PATTERNS — indirect (meaningful but need explanation)
    # ══════════════════════════════════════════════════════════
    'CLUSTER_':         'indirect',
    'CROSS_':           'indirect',
    'INT_':             'indirect',
    'INV_PATTERN_':     'indirect',
    'DECAY_':           'indirect',
    'SEQ_':             'indirect',
    'RECUR_':           'indirect',

    # ══════════════════════════════════════════════════════════
    # VISIT/TEMPORAL — indirect
    # ══════════════════════════════════════════════════════════
    'VISIT_':           'indirect',
    'TEMP_':            'indirect',
    'TRAJ_':            'indirect',

    # ══════════════════════════════════════════════════════════
    # TEXT KEYWORDS — direct (clinical language found in notes)
    # ══════════════════════════════════════════════════════════
    'TEXT_IMG_':         'direct',
    'TEXT_SEV_':         'direct',
    'TEXT_CLIN_':        'direct',
    'TEXT_COMP_':        'indirect',

    # ══════════════════════════════════════════════════════════
    # OPAQUE — statistical features, can't explain to clinician
    # ══════════════════════════════════════════════════════════
    'EMB_':             'opaque',    # TF-IDF embedding dimensions
    'BERT_':            'opaque',    # BERT embedding dimensions
    'ENTROPY_':         'opaque',    # Shannon entropy of categories
    'GINI_':            'opaque',    # Gini concentration measure
    'MONTHLY_':         'opaque',    # raw monthly event bins
    'ROLLING3M_':       'opaque',    # rolling window counts
    'RATE_':            'opaque',    # events-per-category ratio
    'PAIR_':            'opaque',    # co-occurrence pairs (statistical)
    'CMPAIR_':          'opaque',    # clinical-med pairs (statistical)
    'CAT_':             'indirect',  # per-category granular (borderline)
    'AGEX_':            'indirect',  # age interactions

    # ══════════════════════════════════════════════════════════
    # AGGREGATES — mostly opaque (generic utilization)
    # ══════════════════════════════════════════════════════════
    'AGG_total_events':             'opaque',
    'AGG_unique_categories':        'opaque',
    'AGG_unique_code_ids':          'opaque',
    'AGG_events_A':                 'opaque',
    'AGG_events_B':                 'opaque',
    'AGG_symptom_count_A':          'indirect',
    'AGG_symptom_count_B':          'indirect',
    'AGG_symptom_count_total':      'indirect',
    'AGG_symptom_unique_categories': 'indirect',
    'AGG_symptom_acceleration':     'indirect',
    'AGG_new_symptom_cats_in_B':    'indirect',
    'MED_AGG_':                     'opaque',
    'LAB_':                         'indirect',   # fallback for LAB_ not caught above
    'MED_':                         'direct',     # fallback for MED_ not caught above
    'INV_':                         'indirect',   # fallback for INV_
}


# ═══════════════════════════════════════════════════════════════
# FEATURE → CLINICAL DESCRIPTION
# Used to generate clinician-friendly explanations.
# Format: prefix or exact name → template string
# {value} is replaced with the actual feature value at inference time.
# ═══════════════════════════════════════════════════════════════

CLINICAL_DESCRIPTIONS = {
    # ── Demographics ──────────────────────────────────────────
    'AGE_AT_INDEX':     'Patient age: {value} years',
    'AGE_over_65':      'Patient is over 65 (higher prostate cancer risk age group)',
    'AGE_over_75':      'Patient is over 75',

    # ── PSA ───────────────────────────────────────────────────
    'PROST_LAB_psa_latest':         'Most recent PSA level: {value} ng/mL',
    'PROST_LAB_psa_first':          'First PSA level in window: {value} ng/mL',
    'PROST_LAB_psa_elevated_4':     'PSA above 4 ng/mL (above normal threshold)',
    'PROST_LAB_psa_elevated_10':    'PSA above 10 ng/mL (significantly elevated)',
    'PROST_LAB_psa_elevated_20':    'PSA above 20 ng/mL (high-risk level)',
    'PROST_LAB_psa_very_high':      'PSA above 100 ng/mL (very high — suggestive of advanced disease)',
    'PROST_LAB_psa_delta':          'PSA change: {value} ng/mL over the observation period',
    'PROST_LAB_psa_rising':         'PSA is rising over time',
    'PROST_LAB_psa_rapid_rise':     'PSA rising rapidly (>2 ng/mL change)',
    'PROST_psa_acceleration':       'PSA tests becoming more frequent (more monitoring recently)',
    'PROST_has_psa_monitoring':     'Patient has had PSA tests',
    'PROST_psa_count_total':        '{value} PSA tests recorded',

    # ── Urinary ───────────────────────────────────────────────
    'PROST_has_luts':               'Lower urinary tract symptoms present (frequency, hesitancy, weak stream)',
    'PROST_urinary_acceleration':   'LUTS worsening over time',
    'PROST_urinary_count_B':        '{value} LUTS events in recent period',
    'PROST_has_haematuria':         'Blood in urine (haematuria) recorded',
    'PROST_has_erectile_dysfunction': 'Erectile dysfunction recorded',
    'PROST_has_urinary_retention':  'Urinary retention recorded',
    'PROST_has_prostatic_conditions': 'Prostatic condition recorded',
    'PROST_has_ipss':               'IPSS (International Prostate Symptom Score) recorded',

    # ── Bone / Metastatic ─────────────────────────────────────
    'PROST_has_pelvic_bone_pain':   'Pelvic or bone pain reported',
    'PROST_has_bone_muscle':        'Bone/muscle symptoms reported',
    'PROST_has_any_bone_symptom':   'Bone, pelvic, or musculoskeletal pain present (possible metastatic signal)',
    'PROST_bone_acceleration':      'Bone/pelvic pain worsening over time',
    'PROST_LAB_alp_elevated':       'Alkaline phosphatase elevated (possible bone involvement)',
    'PROST_LAB_alp_latest':         'Alkaline phosphatase level: {value} U/L',

    # ── Other Labs ────────────────────────────────────────────
    'PROST_LAB_crp_elevated':       'CRP elevated >10 mg/L (inflammation marker)',
    'PROST_LAB_hb_low':             'Haemoglobin low <120 g/L (anaemia — possible advanced disease)',
    'PROST_LAB_testosterone_low':   'Testosterone low <8 nmol/L',

    # ── Treatment ─────────────────────────────────────────────
    'PROST_has_alpha_blockers':     'Prescribed alpha blockers (tamsulosin — for LUTS)',
    'PROST_has_5ari':               'Prescribed 5-alpha reductase inhibitor (finasteride/dutasteride)',
    'PROST_bph_combo_treatment':    'On combination BPH treatment (alpha blocker + 5-ARI)',
    'PROST_uti_abx_count_A':        '{value} UTI antibiotic prescriptions in earlier period',
    'PROST_uti_abx_count_B':        '{value} UTI antibiotic prescriptions recently',
    'PROST_pain_med_count_B':       '{value} pain escalation prescriptions recently',
    'PROST_has_anticholinergics':   'Prescribed anticholinergics (overactive bladder)',
    'PROST_has_catheter_supplies':  'Catheter supplies prescribed',
    'PROST_has_ed_meds':            'Erectile dysfunction medication prescribed',

    # ── Diagnostic Pathway ────────────────────────────────────
    'PROST_has_urology_imaging':    'Urology imaging performed',
    'PROST_has_dre':                'Digital rectal examination (DRE) performed',
    'PROST_has_cystoscopy':         'Cystoscopy performed',
    'PROST_has_urology_pathway':    'Patient on urology referral pathway',
    'PROST_has_family_history':     'Family history recorded',
    'PROST_DX_symptom_to_imaging_days': 'Time from first urinary symptom to investigation: {value} days',

    # ── Risk Score ────────────────────────────────────────────
    'PROST_risk_score':             'Composite risk score: {value}/5 (age + PSA + LUTS + bone + imaging)',
    'PROST_RF_peak_age':            'Patient in peak prostate cancer age range (50-80)',
    'PROST_RF_elderly':             'Patient over 70 (age is a significant risk factor)',
    'PROST_RF_over65_with_psa':     'Over 65 with PSA monitoring (targeted screening)',
    'PROST_RF_over65_with_luts':    'Over 65 with lower urinary tract symptoms',
    'PROST_RF_psa_plus_bone':       'Elevated PSA combined with bone symptoms (high-risk combination)',

    # ── Observation symptoms (generic pattern) ────────────────
    'OBS_{cat}_count_total':        '{value} {cat_readable} events recorded',
    'OBS_{cat}_count_B':            '{value} {cat_readable} events in recent period',
    'OBS_{cat}_has_ever':           '{cat_readable} recorded in patient history',
    'OBS_{cat}_acceleration':       '{cat_readable} becoming more frequent recently',
    'OBS_{cat}_new_in_B':           '{cat_readable} appeared recently (new symptom)',

    # ── Medication (generic pattern) ──────────────────────────
    'MED_{cat}_count_total':        '{value} {cat_readable} prescriptions',
    'MED_{cat}_has_ever':           'Has been prescribed {cat_readable}',
    'MED_{cat}_acceleration':       '{cat_readable} prescriptions increasing',

    # ── Clusters ──────────────────────────────────────────────
    'CLUSTER_URINARY_any':          'Urinary symptom cluster present',
    'CLUSTER_BONE_any':             'Bone/back pain cluster present',
    'CLUSTER_CONSTITUTIONAL_any':   'Constitutional symptoms present (weight loss, fatigue)',
    'CLUSTER_PSA_any':              'PSA monitoring cluster active',
    'CLUSTER_multi_system_count':   '{value} different symptom systems involved',
    'CLUSTER_3plus_systems':        '3+ symptom systems involved (multi-system presentation)',

    # ── Cross-domain ──────────────────────────────────────────
    'CROSS_diagnostic_odyssey':     'Multiple visits across different symptom types (diagnostic journey)',

    # ── Lab trajectories ──────────────────────────────────────
    'LAB_TRAJ_{cat}_slope':         '{cat_readable} lab values trending: slope = {value}',
    'LAB_TRAJ_{cat}_declining':     '{cat_readable} lab values declining',

    # ── Medication escalation ─────────────────────────────────
    'MED_ESC_pain_acceleration':    'Pain medication prescriptions increasing',
    'MED_ESC_polypharmacy_increase': 'Number of different medication types increasing',

    # ── Text features ─────────────────────────────────────────
    'TEXT_IMG_psa_elevated':         'Clinical notes mention elevated PSA',
    'TEXT_IMG_bone_lesion':          'Clinical notes mention bone lesion',
    'TEXT_IMG_prostate_enlargement': 'Clinical notes mention prostate enlargement',
    'TEXT_IMG_biopsy_finding':       'Clinical notes mention biopsy or histology',
    'TEXT_IMG_dre_abnormal':         'Clinical notes mention abnormal digital rectal exam',
    'TEXT_IMG_mri_finding':          'Clinical notes mention MRI finding',
    'TEXT_IMG_urinary_obstruction':  'Clinical notes mention urinary obstruction',
    'TEXT_SEV_worsening':            'Clinical notes describe worsening symptoms',
    'TEXT_SEV_persistent':           'Clinical notes describe persistent symptoms',
    'TEXT_SEV_recurrent':            'Clinical notes describe recurrent symptoms',
    'TEXT_SEV_new':                  'Clinical notes describe new symptoms',
    'TEXT_CLIN_biopsy':              'Biopsy or histology mentioned in notes',
    'TEXT_CLIN_prostate_term':       'Prostate-specific terms found in clinical notes',
    'TEXT_CLIN_bone_mets':           'Bone metastasis terms found in notes',
    'TEXT_CLIN_psa_rising':          'Rising PSA mentioned in notes',
    'TEXT_CLIN_urinary':             'Urinary symptoms described in notes',
    'TEXT_CLIN_haematuria':          'Haematuria mentioned in notes',
    'TEXT_CLIN_weight_loss':         'Weight loss mentioned in notes',
    'TEXT_CLIN_family_history':      'Family history mentioned in notes',
    'TEXT_COMP_prostate_cluster':    'Clinical notes show pattern of PSA + urinary/bone findings',

    # ── Opaque (should NOT appear in clinician reports) ───────
    'EMB_text_dim_':                '[Text embedding dimension — statistical, not clinically interpretable]',
    'BERT_dim_':                    '[BERT embedding dimension — statistical, not clinically interpretable]',
    'ENTROPY_':                     '[Category diversity measure — statistical]',
    'GINI_':                        '[Category concentration measure — statistical]',
    'MONTHLY_':                     '[Monthly event count — generic utilization]',
    'ROLLING3M_':                   '[Rolling window count — generic utilization]',
    'RATE_':                        '[Event rate ratio — statistical]',
    'PAIR_':                        '[Feature co-occurrence — statistical]',
    'CMPAIR_':                      '[Clinical-medication pair — statistical]',
}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_explainability_class(feature_name):
    """Return 'direct', 'indirect', or 'opaque' for a feature."""
    # Check exact match first
    if feature_name in EXPLAINABILITY_CLASS:
        return EXPLAINABILITY_CLASS[feature_name]
    # Check prefix match (longer prefixes first for specificity)
    sorted_prefixes = sorted(EXPLAINABILITY_CLASS.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if feature_name.startswith(prefix):
            return EXPLAINABILITY_CLASS[prefix]
    return 'opaque'  # unknown features default to opaque


def get_clinical_description(feature_name, value=None):
    """Return a clinician-friendly description for a feature."""
    # Exact match
    if feature_name in CLINICAL_DESCRIPTIONS:
        template = CLINICAL_DESCRIPTIONS[feature_name]
        if value is not None:
            return template.replace('{value}', str(round(value, 2)))
        return template.replace('{value}', '?')

    # Prefix match
    sorted_keys = sorted(CLINICAL_DESCRIPTIONS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if feature_name.startswith(key):
            template = CLINICAL_DESCRIPTIONS[key]
            if value is not None:
                template = template.replace('{value}', str(round(value, 2)))

            # Try to extract category from feature name for {cat_readable}
            cat = feature_name.replace(key, '').split('_')[0] if '{cat' in key else ''
            cat_readable = cat.replace('_', ' ').lower()
            template = template.replace('{cat_readable}', cat_readable)
            template = template.replace('{cat}', cat)
            return template

    return f'{feature_name} = {value}' if value is not None else feature_name


def classify_all_features(feature_list):
    """Classify a list of features into direct/indirect/opaque."""
    result = {'direct': [], 'indirect': [], 'opaque': []}
    for f in feature_list:
        cls = get_explainability_class(f)
        result[cls].append(f)
    return result
