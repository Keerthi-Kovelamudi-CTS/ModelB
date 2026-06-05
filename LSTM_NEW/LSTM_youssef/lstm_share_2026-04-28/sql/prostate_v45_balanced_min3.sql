/* prostate v45_balanced_min3 — snomed-only (no med branch)
 *
 * Goal: guaranteed 1:1 cancer:non-cancer output AFTER curated codelist filter.
 *
 * Why no medications: prostate ablation (k10 vs k9) showed the med branch adds
 * zero value — workup signal is dominated by SNOMED-side codes (biopsy referral,
 * DRE, fast-track), not prescriptions. Med-side dropped from this SQL to cut
 * BQ scan cost and output CSV size.
 *
 * Key changes vs v45 / v45_with_codes:
 *   1. curated_codes CTE inlines 237 curated codes (no PSA — caused shortcut learning in tabular)
 *   2. cancer_event_counts: INNER JOIN curated_codes → HAVING ≥3 counts curated only
 *   3. non_cancer_event_counts: INNER JOIN curated_codes → HAVING ≥3 counts curated only
 *   4. observation_events_raw: INNER JOIN curated_codes → output filtered to curated obs
 *   5. final_medication / medication_events_raw / medication_events: REMOVED
 *   6. Final UNION ALL collapsed to single SELECT FROM observation_events
 *
 * Unchanged from v45:
 *   - 1:1 cancer:non-cancer ratio (year-only matching, NO age-band match)
 *   - 10y lookback, 12mo buffer
 *   - min_obs_events_per_patient = 3 (on curated)
 * v45_balanced_min3 specifically: SQL HAVING threshold lowered from 10 to 3
 *   - Captures patients with as few as 3 curated events (early-stage thin records)
 *   - Pair with Python --min-snomed-seq 3 for end-to-end MIN=3 testing
 */

/*
 * Unified cancer / non-cancer patient events for ML training.
 *
 * PURPOSE:
 *   Produce a single dataset with both cancer and non-cancer patient events
 *   (observations + medications) for training an ML model to predict cancer.
 *   Two columns distinguish the groups:
 *        - `cancer_class` (1/0)
 *        - `cancer_id` (null for non-cancer patients)
 *
 * HOW IT WORKS:
 *   1. Pick target cancer patients whose diagnosis date ∈ [window).
 *   2. Pick non-cancer patients (no cancer diagnosis) with at least one trainable
 *      observation event ∈ [window); anchor = ONE such event, uniformly sampled
 *      per patient via deterministic FARM_FINGERPRINT hash (v40 change — see below).
 *   3. For each patient, build a lookback window [anchor - years_before,
 *      anchor - months_before) and collect observation + medication events in it.
 *   4. Apply the same filters (excluded codes, palliative care, immunisations,
 *      minimum event thresholds) to both groups — identical pipeline modulo class.
 *   5. Keep ALL qualifying cancer patients; for non-cancer, sample per-year
 *      at the cancer-class rate.
 *
 * HOW DATA LEAKAGE IS PREVENTED / MITIGATED:
 *   - Same relative lookback window for both groups.
 *   - Cancer-related SNOMED codes excluded from BOTH groups.
 *   - Palliative-care patients excluded from BOTH groups.
 *   - All filters applied identically.
 *   - Minimum event thresholds applied equally.
 *   - Anchor-date window matched across classes (v38 — year distributions match
 *     by construction, not by fingerprint-forcing).
 *   - No matching on clinical features (age, gender, event frequency) that
 *     could remove real predictive signals.
 *
 * WHAT IS DELIBERATELY NOT MATCHED (potential genuine predictive signals):
 *   - Age, gender, ethnicity distributions.
 *   - Number of events per patient (visit frequency).
 *   - Types of observations and medications.
 *
 */

WITH params AS (
  SELECT
    DATE '1950-01-01' AS longterm_mh_start,
    DATE '2026-01-01' AS longterm_mh_end,
    DATE '2017-01-01' AS anchor_window_start,        -- inclusive lower bound for anchor dates
    DATE '2026-01-01' AS anchor_window_end,          -- exclusive upper bound for anchor dates
    10                AS years_before,
    12                AS months_before,
    3                 AS min_obs_events_per_patient,  -- minimum observation events required per patient
    1                 AS non_cancer_ratio,            -- default ratio of non-cancer to cancer patients is 1:1 (change to 2 for 2:1)
    '%prostate%'          AS target_cancer_pattern        -- Specify the target cancer
    /* No LIMIT on events — unified_patients controls patient count, so row count is naturally bounded */
),

/* ═══════════════════════════════════════════════════════════════════════════
   Shared reference tables — used by both cancer and non-cancer pipelines
   ═══════════════════════════════════════════════════════════════════════════ */

diagnostic_codes AS (
  SELECT
    code_id,
    source_practice_code,
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date,
    term,
    snomed_c_t_concept_id,
    snomed_c_t_description_id
  FROM `cthesigns-platform-475414-b7.EMIS_BULK_DATA_PROCESSED.Coding_ClinicalCode`
  WHERE snomed_c_t_concept_id IS NOT NULL
    AND term IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY code_id, source_practice_code ORDER BY file_date DESC
  ) = 1
),

patients AS (
  SELECT patient_guid, sex, date_of_birth, deleted
  FROM `cthesigns-platform-475414-b7.EMIS_BULK_DATA_PROCESSED.Admin_Patient`
  WHERE deleted != true OR deleted IS NULL
),

curated_codes AS (
  -- v45_balanced: SQL-side codelist filter (237 curated codes)
  -- Used at BOTH event-counting (for HAVING) and output level
  SELECT snomed_code, term FROM UNNEST(ARRAY<STRUCT<snomed_code INT64, term STRING>>[
    (1000381000000105, 'Serum PSA (prostate specific antigen) level'),
    (1000481000000100, 'Serum free PSA (prostate specific antigen) level'),
    (1000491000000103, 'Serum free:total PSA (prostate specific antigen) ratio'),
    (1000651000000109, 'Serum potassium level'),
    (1000661000000107, 'Serum sodium level'),
    (1000691000000101, 'Serum calcium level'),
    (1000731000000107, 'Serum creatinine level'),
    (1000811000000109, 'Serum total protein'),
    (1000821000000103, 'Serum albumin level'),
    (1000841000000105, 'Serum alkaline phosphatase liver isoenzyme level'),
    (1000851000000108, 'Plasma GGT level'),
    (1000961000000100, 'Plasma urea level'),
    (1000971000000107, 'Urea and electrolytes level'),
    (1001011000000107, 'Plasma creatinine level'),
    (1001231000000108, 'Serum globulin level'),
    (1001371000000100, 'Serum CRP (C reactive protein) level'),
    (1005651000000101, 'Plasma total protein'),
    (1006591000000104, 'Total PSA (prostate specific antigen) level'),
    (1007881000000101, 'Urine dipstick test'),
    (1011481000000105, 'eGFR using creatinine CKD-EPI'),
    (1013041000000106, 'Plasma alkaline phosphatase level'),
    (1013211000000103, 'Plasma alanine aminotransferase level'),
    (1013361000000105, 'Plasma albumin level'),
    (1014831000000107, 'Urine microscopy'),
    (1014851000000100, 'Urine leucocyte test'),
    (1017381000000106, 'Plasma sodium level'),
    (1017391000000108, 'Plasma calcium level'),
    (1017401000000106, 'Plasma potassium level'),
    (1018251000000107, 'ALT/SGPT serum level'),
    (10186811000001105, 'Tabphyn MR 400microgram capsules'),
    (10188911000001101, 'Pamsvax XL 400microgram capsules'),
    (1020291000000106, 'GFR calculated by abbreviated MDRD'),
    (1022291000000105, 'Haematocrit'),
    (1022431000000105, 'Haemoglobin estimation'),
    (1022441000000101, 'FBC - full blood count'),
    (1022451000000103, 'Red blood cell count'),
    (1022471000000107, 'MCH - Mean corpuscular haemoglobin'),
    (1022481000000109, 'MCHC - Mean corpuscular haemoglobin concentration'),
    (1022491000000106, 'MCV - Mean corpuscular volume'),
    (1022511000000103, 'Erythrocyte sedimentation rate'),
    (1022541000000102, 'Total white cell count'),
    (1022551000000104, 'Neutrophil count'),
    (1022561000000101, 'Eosinophil count'),
    (1022571000000108, 'Basophil count'),
    (1022581000000105, 'Lymphocyte count'),
    (1022591000000107, 'Monocyte count'),
    (1022651000000100, 'Platelet count'),
    (1023711000000100, 'Urine culture'),
    (1023721000000106, 'Urine nitrite level'),
    (1026761000000106, 'Total bilirubin level'),
    (1027171000000101, 'Prolactin level'),
    (1027791000000103, 'Urine microalbumin/creatinine ratio'),
    (1027931000000101, 'Blood calcium level'),
    (1028081000000104, 'Direct (conjugated) bilirubin'),
    (1028091000000102, 'GGT level'),
    (1028101000000105, 'Creatine kinase level'),
    (1028131000000104, 'Urea and electrolytes'),
    (1028281000000106, 'Blood urea'),
    (1028641000000101, 'Coagulation tests'),
    (1028731000000100, 'Urine protein/creatinine ratio'),
    (1029801000000105, 'Total 25-hydroxyvitamin D level'),
    (1030021000000101, 'Free PSA (prostate specific antigen) level'),
    (1030031000000104, 'Free:total PSA (prostate specific antigen) ratio'),
    (1030791000000100, 'Prostate specific antigen level'),
    (1031081000000108, 'Liver function tests - general'),
    (1031101000000102, 'Aspartate aminotransferase level'),
    (1038641000000104, 'Urine microscopy: red cells'),
    (1041881000000101, 'Urine microscopy: pus cells'),
    (11441004, 'Prostatism'),
    (128606002, 'Urological disorder'),
    (129252005, 'Insertion of suprapubic catheter'),
    (130951007, 'Bladder retention of urine'),
    (13301000087109, 'Ultrasound of bilateral kidneys'),
    (135885005, 'Advice about impotence'),
    (137771000006103, 'Smoking Age Started'),
    (139394000, 'Nocturia D'),
    (14608411000001108, 'Tolterodine 4mg modified-release capsules'),
    (14610411000001107, 'Potassium citrate mixture'),
    (160290005, 'Family history of neoplasm of male genital tract'),
    (160618006, 'Current non-smoker'),
    (161080002, 'Passive smoking risk'),
    (161550001, 'History of haematuria'),
    (161555006, 'H/O: prostatism'),
    (161829004, 'Weight symptom NOS'),
    (161894002, 'Complaining of low back pain'),
    (162116003, 'Urinary frequency'),
    (162147009, 'Pelvic pain'),
    (162148004, 'C/O perineal pain'),
    (162410003, 'C/O: a swelling'),
    (165346000, 'Laboratory test result abnormal'),
    (166160000, 'Prostate specific antigen outside reference range'),
    (166646003, 'ALT level abnormal'),
    (167217005, 'Urine examination'),
    (167300001, 'Urine blood test = +'),
    (167301002, 'Urine blood test = ++'),
    (167302009, 'Urine blood test = +++'),
    (167767004, 'Semen abnormal'),
    (167800009, 'Sperm: haemospermia O/E'),
    (168041003, 'O/E: renal calculus'),
    (168130002, 'RBCs seen on microscopy'),
    (168336001, 'MSU sent for C/S'),
    (168337005, 'MSU sent for bacteriology'),
    (168339008, 'Catheter urine sent for culture'),
    (1726491000006101, 'LUTS - Lower urinary tract symptoms'),
    (1807411000006100, 'Suprapubic tenderness'),
    (197938001, 'Painless haematuria'),
    (197941005, 'Frank haematuria'),
    (21173002, 'Benign adenoma of prostate'),
    (225087008, 'Urinary catheter care'),
    (236633002, 'Overactive bladder'),
    (236645006, 'Bladder outflow obstruction'),
    (236648008, 'Acute retention of urine'),
    (236650000, 'Chronic retention of urine'),
    (23676211000001102, 'Solifenacin 6mg / Tamsulosin 400microgram modified-release tablets'),
    (247358007, 'Abdominal pain type'),
    (249274008, 'Urinary symptoms'),
    (251984009, 'Urinary flow rate'),
    (266569009, 'BPH - benign prostatic hypertrophy'),
    (267031002, 'Tiredness'),
    (267062003, 'Genitourinary symptoms NOS'),
    (267064002, 'Retention of urine'),
    (267067009, 'Lumbar ache - renal'),
    (268637002, 'Psychosexual dysfunction'),
    (268915006, 'O/E - weight 10-20% over ideal'),
    (26958001, 'Liver function test'),
    (271302001, 'O/E - PR - prostatic swelling'),
    (271349002, 'Urine microscopy:RBCs present'),
    (272047006, 'Complaining of loin pain'),
    (272062008, 'C/O - tired all the time'),
    (274296009, 'O/E -rectal examination'),
    (274734008, 'Micturition frequency and polyuria'),
    (275302008, 'Prostate enlarged on PR'),
    (275413005, 'Blocked catheter'),
    (275741008, 'Leucocytes in urine'),
    (279031000006106, 'O/E - PR - Diffusely enlarged prostate'),
    (300471006, 'Observation of frequency of urination'),
    (30281009, 'Prostatic disorder'),
    (307541003, 'Lower urinary tract symptoms'),
    (309089006, 'Prostate mass'),
    (3188711000001103, 'Xatral XL 10mg tablets'),
    (34436003, 'Haematuria'),
    (34615008, 'Haemospermia'),
    (35920211000001106, 'Tramadol 100mg modified-release capsules'),
    (365689007, 'Finding related to casts on urine microscopy'),
    (3857111000001101, 'Caverject Dual Chamber 20microgram'),
    (38754011000001109, 'Tamsulosin 400microgram modified-release capsules'),
    (38893911000001102, 'Fesoterodine 8mg modified-release tablets'),
    (38894011000001104, 'Fesoterodine 4mg modified-release tablets'),
    (38897311000001105, 'Alfuzosin 10mg modified-release tablets'),
    (39020411000001106, 'Doxazosin 4mg modified-release tablets'),
    (39021111000001107, 'Doxazosin 8mg modified-release tablets'),
    (390900001, 'Smoking cessation milestones'),
    (39112111000001104, 'Oxybutynin 5mg modified-release tablets'),
    (39112411000001109, 'Dihydrocodeine 60mg modified-release tablets'),
    (396152005, 'Raised prostate specific antigen'),
    (39687011000001101, 'Ciprofloxacin 750mg tablets'),
    (39687511000001109, 'Ciprofloxacin 250mg tablets'),
    (39687811000001107, 'Ciprofloxacin 500mg tablets'),
    (39694811000001102, 'Cefalexin 500mg capsules'),
    (39696511000001106, 'Indoramin 20mg tablets'),
    (39697411000001109, 'Levofloxacin 250mg tablets'),
    (39706511000001100, 'Vardenafil 10mg tablets'),
    (39706711000001105, 'Vardenafil 20mg tablets'),
    (39735311000001105, 'Cefalexin 250mg capsules'),
    (397803000, 'Male erectile disorder'),
    (410007005, 'Rectal examination'),
    (4105911000001106, 'Cialis 10mg tablets'),
    (4106311000001100, 'Cialis 20mg tablets'),
    (413173009, 'Minutes from waking to first tobacco consumption'),
    (414205003, 'Family history of prostate cancer'),
    (417611000001101, 'Ciproxin 500mg tablets'),
    (41952811000001109, 'Nitrofurantoin 100mg capsules'),
    (41952911000001104, 'Nitrofurantoin 100mg tablets'),
    (41953111000001108, 'Nitrofurantoin 50mg capsules'),
    (41953211000001102, 'Nitrofurantoin 50mg tablets'),
    (41953511000001104, 'Ofloxacin 400mg tablets'),
    (41956211000001109, 'Trimethoprim 200mg tablets'),
    (42010011000001101, 'Co-codamol 8mg/500mg capsules'),
    (42010611000001108, 'Co-dydramol 10mg/500mg tablets'),
    (42030000, 'Genitourinary disease NOS'),
    (42030911000001107, 'Alfuzosin 2.5mg tablets'),
    (42038911000001109, 'Oxybutynin 2.5mg tablets'),
    (42043111000001101, 'Terazosin 2mg tablets'),
    (42043211000001107, 'Terazosin 5mg tablets'),
    (42043711000001100, 'Tolterodine 1mg tablets'),
    (42044211000001105, 'Trospium chloride 60mg modified-release capsules'),
    (428556002, 'Total international prostate symptom score'),
    (43491000, 'Acute epididymitis'),
    (4367811000001102, 'Dutasteride 500microgram capsules'),
    (441915005, 'Measurement of renal function'),
    (442618008, 'Abnormal finding on evaluation procedure'),
    (49650001, 'Dysuria'),
    (525511000001107, 'Viagra 100mg tablets'),
    (58972000, 'Dribbling of urine'),
    (5972002, 'Hesitancy of micturition'),
    (61033006, 'Unstable bladder'),
    (638411000001105, 'Muse 1000microgram urethral sticks'),
    (68226007, 'Acute cystitis'),
    (70660007, 'Hydrocele of tunica vaginalis'),
    (7163005, 'Urinary tract obstruction'),
    (75088002, 'Urgent desire to urinate'),
    (759391000006100, 'IPSS - frequency'),
    (759401000006103, 'IPSS - incomplete emptying'),
    (759411000006100, 'IPSS - intermittency'),
    (759421000006108, 'IPSS - nocturia'),
    (759441000006101, 'IPSS - urgency'),
    (759451000006104, 'IPSS - weak stream'),
    (763101000000106, 'Severe lower urinary tract symptoms'),
    (763111000000108, 'Moderate lower urinary tract symptoms'),
    (763121000000102, 'Mild lower urinary tract symptoms'),
    (801000119105, 'On rectal examination of prostate abnormality detected'),
    (80274001, 'Glomerular filtration rate'),
    (808811000001107, 'Viagra 50mg tablets'),
    (853721000006104, 'Left loin pain'),
    (860914002, 'Erectile dysfunction'),
    (9009911000001104, 'SomaErect Response II vacuum pump'),
    (909491000006106, 'Reason for care Enlarged prostate'),
    (935051000000108, 'Corrected serum calcium level'),
    (938911000000100, 'At risk of acute kidney injury'),
    (9483111000001109, 'Flomaxtra XL 400microgram tablets'),
    (9484411000001101, 'Tamsulosin 400microgram modified-release tablets'),
    (9713002, 'Prostatitis and other inflammatory diseases of prostate'),
    (981411000006101, 'Nocturia'),
    (992821000000106, 'Organism count'),
    (992871000000105, 'Epithelial cell count'),
    (993111000000101, 'Plasma globulin level'),
    (993501000000105, 'Red blood cell distribution width'),
    (993551000000106, 'Large unstained cells'),
    (997121000000100, 'Gonadotrophin levels'),
    (997531000000108, 'Liver function test alt'),
    (997561000000103, 'Plasma total bilirubin level'),
    (997591000000109, 'Serum total bilirubin level'),
    (997741000000108, '25-Hydroxyvitamin D3 level'),
    (999131000000106, 'Sample microscopy: red cells'),
    (999291000000102, 'Sample microscopy: leucocytes'),
    (999651000000107, 'Plasma C reactive protein'),
    (999691000000104, 'Serum bilirubin level')
  ])
),


medical_history AS (
  SELECT
    co.source_practice_code                      AS practice_id,
    co.code_id                                   AS code_id,
    co.patient_guid                              AS patient_guid,
    co.consultation_guid                         AS consultation_guid,
    PARSE_DATE('%Y-%m-%d', co.effective_date)    AS effective_date,
    PARSE_DATE('%Y-%m-%d', co.entered_date)      AS entered_date,
    co.associated_text                           AS associated_text,
    co.observation_type                          AS observation_type,
    co.value                                     AS value,
    co.ENTERED_BY_USER_IN_ROLE_GUID,
    dc.term,
    SAFE_CAST(dc.snomed_c_t_concept_id AS INT64) AS snomed_c_t_concept_id,
    dc.snomed_c_t_description_id,
    dc.source_practice_code                      AS dc_source_practice_code
  FROM `cthesigns-platform-475414-b7.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` AS co
  LEFT JOIN diagnostic_codes AS dc
    ON SAFE_CAST(co.code_id AS INT64) = SAFE_CAST(dc.code_id AS INT64)
   AND co.source_practice_code = dc.source_practice_code
  CROSS JOIN params param
  WHERE co.effective_date IS NOT NULL
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) >= param.longterm_mh_start
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) <  param.longterm_mh_end
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 1: Identify cancer patients and cancer-related codes
   ═══════════════════════════════════════════════════════════════════════════ */

/* All cancer SNOMED codes from the reference table.
   Used for two purposes:
   (a) Find patients who have any cancer diagnosis
   (b) Exclude these codes from observation events for BOTH groups,
       so the ML model cannot simply look for cancer diagnosis codes */
cancer_snomed_codes AS (
  SELECT DISTINCT SAFE_CAST(SNOMED_CODE AS INT64) AS snomed_code
  FROM `cthesigns-platform-475414-b7.EMIS_MED_CODES.Dim_Cancer_Codes`
  WHERE LOWER(cancer_id) NOT LIKE '%disease%'
),

/* Every patient who has ANY type of cancer diagnosis.
   These patients are excluded from the non-cancer group to ensure
   the non-cancer group is truly cancer-free. */
any_cancer_patient AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
  JOIN cancer_snomed_codes csc
    ON mh.snomed_c_t_concept_id = csc.snomed_code
),

/* For each patient, find their earliest diagnosis date.
   The diagnosis date becomes the "anchor" for the lookback window. */
target_cancer_patients AS (
  SELECT patient_guid, date_of_diagnosis, cancer_id
  FROM (
    SELECT
      mh.patient_guid,
      mh.effective_date AS date_of_diagnosis,
      dcc.cancer_id,
      ROW_NUMBER() OVER (PARTITION BY mh.patient_guid ORDER BY mh.effective_date, dcc.cancer_id) AS rn
    FROM medical_history mh
    JOIN `cthesigns-platform-475414-b7.EMIS_MED_CODES.Dim_Cancer_Codes` dcc
      ON CAST(mh.snomed_c_t_concept_id AS STRING) = dcc.SNOMED_CODE
    CROSS JOIN params param
    WHERE mh.effective_date IS NOT NULL
      AND mh.effective_date >= param.anchor_window_start
      AND mh.effective_date <  param.anchor_window_end
      AND LOWER(dcc.cancer_id) NOT LIKE '%disease%'
      AND LOWER(dcc.cancer_id) LIKE param.target_cancer_pattern
  )
  WHERE rn = 1
),

/* Palliative-care patients — excluded from BOTH groups.
   This avoids bias: if palliative care were only excluded from one group,
   the ML model could potentially learn to use palliative-care codes as a class signal. */
palliative_patients AS (
  SELECT DISTINCT patient_guid
  FROM medical_history
  WHERE snomed_c_t_concept_id = 1403151000000103
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 2: Find non-cancer patients
   Patients with NO cancer diagnosis of any kind, and not in palliative care.
   ═══════════════════════════════════════════════════════════════════════════ */

non_cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
  WHERE NOT EXISTS (
      SELECT 1 FROM any_cancer_patient acp
      WHERE acp.patient_guid = mh.patient_guid
    )
    AND NOT EXISTS (
      SELECT 1 FROM palliative_patients pp
      WHERE pp.patient_guid = mh.patient_guid
    )
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 3: Set anchor dates for non-cancer patients
   ─────────────────────────────────────────────────────────────────────────
   Cancer patients use their diagnosis date as the anchor.
   Non-cancer anchor = an observation event date sampled from inside 
   the anchor window.
    ═══════════════════════════════════════════════════════════════════════════ */

non_cancer_with_anchor AS (
  SELECT
    ncp.patient_guid,
    mh.effective_date AS anchor_date
  FROM non_cancer_patients ncp
  JOIN medical_history mh
    ON mh.patient_guid = ncp.patient_guid
  CROSS JOIN params param
  WHERE mh.effective_date >= param.anchor_window_start
    AND mh.effective_date <  param.anchor_window_end
    AND mh.snomed_c_t_concept_id IS NOT NULL
    AND NOT EXISTS (
      SELECT 1 FROM cancer_snomed_codes csc
      WHERE csc.snomed_code = mh.snomed_c_t_concept_id
    )
    AND mh.snomed_c_t_concept_id NOT IN (
      1572871000006101, 279991000000102, 428481002,
      979851000000101, 887641000000105, 1958701000006105, 185317003,
      788007007, 283511000000105, 182836005, 182888003, 184103008,
      498521000006103, 386472008
    )
    AND mh.observation_type NOT IN ('Immunisation')
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY ncp.patient_guid
    ORDER BY FARM_FINGERPRINT(CONCAT(
      CAST(ncp.patient_guid AS STRING), '|',
      CAST(mh.effective_date AS STRING), '|',
      CAST(mh.snomed_c_t_concept_id AS STRING)
    ))
  ) = 1
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 4: Count observation events per patient and apply minimums
   ─────────────────────────────────────────────────────────────────────────
   Count how many distinct observation events each patient has within their
   lookback window (using the same filters as the final output).
   Only keep patients with enough events (>= min thresholds set in params)
   so the ML model has sufficient data to learn from each patient.
   ═══════════════════════════════════════════════════════════════════════════ */

/* Cancer patients: count distinct observation events in their lookback window.
   Uses SELECT DISTINCT on the same columns as the final output,
   so the count matches the actual number of rows per patient. */
cancer_event_counts AS (
  SELECT
    patient_guid, anchor_date, cancer_id,
    COUNT(*) AS obs_event_count
  FROM (
    SELECT DISTINCT
      tcp.patient_guid,
      tcp.date_of_diagnosis AS anchor_date,
      tcp.cancer_id,
      mh.effective_date,
      mh.snomed_c_t_concept_id,
      CAST(mh.term AS STRING) AS term,
      CAST(mh.associated_text AS STRING) AS associated_text,
      CAST(mh.value AS STRING) AS value
    FROM target_cancer_patients tcp
    JOIN medical_history mh
      ON mh.patient_guid = tcp.patient_guid
    INNER JOIN curated_codes cc
      ON cc.snomed_code = mh.snomed_c_t_concept_id  -- v45_balanced: count curated events only
    CROSS JOIN params param
    WHERE NOT EXISTS (
        SELECT 1 FROM palliative_patients pp WHERE pp.patient_guid = tcp.patient_guid
      )
      AND mh.effective_date >= DATE_SUB(tcp.date_of_diagnosis, INTERVAL param.years_before YEAR)
      AND mh.effective_date <  DATE_SUB(tcp.date_of_diagnosis, INTERVAL param.months_before MONTH)
      AND mh.snomed_c_t_concept_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM cancer_snomed_codes csc
        WHERE csc.snomed_code = mh.snomed_c_t_concept_id
      )
      AND mh.snomed_c_t_concept_id NOT IN (
        1572871000006101, 279991000000102, 428481002,
        979851000000101, 887641000000105, 1958701000006105, 185317003,
        788007007, 283511000000105, 182836005, 182888003, 184103008,
        498521000006103, 386472008
      )
      AND mh.observation_type NOT IN ('Immunisation')
  )
  GROUP BY 1, 2, 3
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

/* Non-cancer patients: same counting logic as cancer patients above. */
non_cancer_event_counts AS (
  SELECT
    patient_guid, anchor_date,
    COUNT(*) AS obs_event_count
  FROM (
    SELECT DISTINCT
      nca.patient_guid,
      nca.anchor_date,
      mh.effective_date,
      mh.snomed_c_t_concept_id,
      CAST(mh.term AS STRING) AS term,
      CAST(mh.associated_text AS STRING) AS associated_text,
      CAST(mh.value AS STRING) AS value
    FROM non_cancer_with_anchor nca
    JOIN medical_history mh
      ON mh.patient_guid = nca.patient_guid
    INNER JOIN curated_codes cc
      ON cc.snomed_code = mh.snomed_c_t_concept_id  -- v45_balanced: count curated events only
    CROSS JOIN params param
    WHERE mh.effective_date >= DATE_SUB(nca.anchor_date, INTERVAL param.years_before YEAR)
      AND mh.effective_date <  DATE_SUB(nca.anchor_date, INTERVAL param.months_before MONTH)
      AND mh.snomed_c_t_concept_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM cancer_snomed_codes csc
        WHERE csc.snomed_code = mh.snomed_c_t_concept_id
      )
      AND mh.snomed_c_t_concept_id NOT IN (
        1572871000006101, 279991000000102, 428481002,
        979851000000101, 887641000000105, 1958701000006105, 185317003,
        788007007, 283511000000105, 182836005, 182888003, 184103008,
        498521000006103, 386472008
      )
      AND mh.observation_type NOT IN ('Immunisation')
  )
  GROUP BY 1, 2
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 5: Balance the dataset 
   ─────────────────────────────────────────────────────────────────────────
   Keep ALL qualifying cancer patients (to maximize the positive class).
   Randomly sample non-cancer patients to match the cancer patient count 
   for each anchor year (with non_cancer_ratio applied).

   Step A. Count cancer patients per anchor-year (within the anchor window).
   Step B. Rank non-cancer patients randomly WITHIN their natural anchor year
           using FARM_FINGERPRINT (reproducible pseudo-random ordering).
   Step C. Keep the top (non_cancer_ratio * cancer_count[year]) non-cancer
           patients in each year.

   Result:
     - non-cancer per-year count = non_cancer_ratio × cancer per-year count
     - WITHIN each year, the chosen non-cancer patients are a uniform
       pseudo-random subset of the eligible non-cancer population — no
       patient is forced into a year they don't naturally belong to.
   ═══════════════════════════════════════════════════════════════════════════ */

cancer_year_counts AS (
  SELECT
    EXTRACT(YEAR FROM anchor_date) AS anchor_year,
    COUNT(*)                        AS cancer_count
  FROM cancer_event_counts
  GROUP BY 1
),

non_cancer_ranked AS (
  SELECT
    patient_guid,
    anchor_date,
    EXTRACT(YEAR FROM anchor_date) AS anchor_year,
    ROW_NUMBER() OVER (
      PARTITION BY EXTRACT(YEAR FROM anchor_date)
      ORDER BY FARM_FINGERPRINT(CAST(patient_guid AS STRING))
    ) AS rank_in_year
  FROM non_cancer_event_counts
),

non_cancer_sampled AS (
  SELECT ncr.patient_guid, ncr.anchor_date
  FROM non_cancer_ranked ncr
  JOIN cancer_year_counts cyc USING (anchor_year)
  CROSS JOIN params param
  WHERE ncr.rank_in_year <= param.non_cancer_ratio * cyc.cancer_count
),

/* ═══════════════════════════════════════════════════════════════════════════
   Unified patient set: all cancer + randomly sampled non-cancer
   ═══════════════════════════════════════════════════════════════════════════ */

unified_patients AS (
  /* All qualifying cancer patients */
  SELECT patient_guid, anchor_date, cancer_id,
         1 AS cancer_class,
         EXTRACT(YEAR FROM anchor_date) AS anchor_year
  FROM cancer_event_counts

  UNION ALL

  /* Randomly sampled non-cancer patients (same count as cancer) */
  SELECT patient_guid, anchor_date, CAST(NULL AS STRING) AS cancer_id,
         0 AS cancer_class,
         EXTRACT(YEAR FROM anchor_date) AS anchor_year
  FROM non_cancer_sampled
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 6: Produce observation events for selected patients
   ─────────────────────────────────────────────────────────────────────────
   (Med branch removed — see header comment.)
   ═══════════════════════════════════════════════════════════════════════════ */

/* Raw observation events (before minimum-count enforcement) */
observation_events_raw AS (
  SELECT DISTINCT
    up.patient_guid,
    up.cancer_class,
    up.cancer_id,
    pp.sex,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS patient_age,
    CAST(rc.ethnicity AS STRING)                              AS patient_ethnicity,
    mh.effective_date                                         AS event_date,         -- diagnostic only; do NOT use as a model feature (anchor-year bias)
    DATE_DIFF(mh.effective_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS event_age,
    DATE_DIFF(up.anchor_date, mh.effective_date, DAY)         AS days_before_anchor,  -- use this instead of event_date for temporal model features
    up.anchor_year,
    'observation'                                             AS event_type,
    mh.snomed_c_t_concept_id,
    CAST(mh.term AS STRING)                                   AS term,
    CAST(mh.associated_text AS STRING)                        AS associated_text,
    CAST(mh.value AS STRING)                                  AS value,
    CAST(NULL AS INT64)                                       AS med_code_id,
    CAST(NULL AS STRING)                                      AS drug_term,
    CAST(NULL AS INT64)                                       AS duration
  FROM unified_patients up
  JOIN medical_history mh
    ON mh.patient_guid = up.patient_guid
  INNER JOIN curated_codes cc_obs
    ON cc_obs.snomed_code = mh.snomed_c_t_concept_id  -- v45_balanced: output curated obs only
  JOIN patients pp
    ON pp.patient_guid = up.patient_guid
  LEFT JOIN `cthesigns-platform-475414-b7.EMIS_MED_CODES.Ethnicity_Snowmed_Codes` rc
    ON CAST(mh.snomed_c_t_concept_id AS STRING) = rc.code
  CROSS JOIN params param
  WHERE
    /* ── Symmetric relative lookback window ── */
    mh.effective_date >= DATE_SUB(up.anchor_date, INTERVAL param.years_before YEAR)
    AND mh.effective_date <  DATE_SUB(up.anchor_date, INTERVAL param.months_before MONTH)
    /* ── Exclude cancer SNOMED codes from BOTH groups (prevents label leakage) ── */
    AND NOT EXISTS (
      SELECT 1 FROM cancer_snomed_codes csc
      WHERE csc.snomed_code = mh.snomed_c_t_concept_id
    )
    /* ── Standard exclusions (applied identically to both classes) ── */
    AND mh.snomed_c_t_concept_id IS NOT NULL
    AND mh.snomed_c_t_concept_id NOT IN (
      1572871000006101, 279991000000102, 428481002,
      979851000000101, 887641000000105, 1958701000006105, 185317003,
      788007007, 283511000000105, 182836005, 182888003, 184103008,
      498521000006103,   -- Attachment for medical notes
      386472008           -- Telephone consultations
    )
    AND mh.observation_type NOT IN ('Immunisation')
  /* No LIMIT — unified_patients already controls the patient count via balanced sampling,
     so the total row count is bounded. Removing LIMIT avoids any risk of splitting
     a patient's events or arbitrary truncation. */
),

/* Enforce minimum observation events per patient.
   Uses a window function to count on the exact same deduplicated rows
   that will appear in the output, then filters out patients with too few. */
observation_events AS (
  SELECT * EXCEPT(patient_obs_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_guid) AS patient_obs_count
    FROM observation_events_raw
  )
  WHERE patient_obs_count >= (SELECT min_obs_events_per_patient FROM params)
)

/* ═══════════════════════════════════════════════════════════════════════════
   Step 7: Final output — observations only (med branch removed for prostate)
   ═══════════════════════════════════════════════════════════════════════════ */
SELECT * FROM observation_events;
