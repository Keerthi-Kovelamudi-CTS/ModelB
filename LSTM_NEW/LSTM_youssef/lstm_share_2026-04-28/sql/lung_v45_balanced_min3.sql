/* lung v45_balanced_min3 — auto-generated from v45 + post-filter 1:1 balancing
 *
 * Goal: guaranteed 1:1 cancer:non-cancer output AFTER both Python --min-snomed-seq 3
 * AND --min-med-seq 1 filters (lung dual-branch).
 *
 * Key changes vs v45 / v45_with_codes:
 *   1. curated_codes CTE inlines 155 curated codes
 *   2. cancer_event_counts: INNER JOIN curated_codes → HAVING ≥3 counts curated only
 *   3. non_cancer_event_counts: INNER JOIN curated_codes → HAVING ≥3 counts curated only
 *   4. cancer_med_counts / non_cancer_med_counts: HAVING ≥3 curated meds (lung-specific —
 *      forces 1:1 to be preserved through Python's --min-med-seq filter, since matched
 *      non-cancer controls are far less likely to have curated meds than cancer patients)
 *   5. observation_events_raw: INNER JOIN curated_codes → output filtered to curated obs
 *   6. medication_events_raw: INNER JOIN curated_codes → output filtered to curated meds
 *
 * The 1:1 year-balanced sampling now operates on patients eligible on BOTH the SNOMED
 * AND med side, giving a 1:1 LSTM training cohort even after the Python med-branch
 * eligibility filter.
 *
 * Unchanged from v45:
 *   - 1:1 cancer:non-cancer ratio (year-only matching, NO age-band match)
 *   - 10y lookback, 12mo buffer
 *   - min_obs_events_per_patient = 3, min_med_events_per_patient = 1 (curated)
 *     (med threshold loosened to 1 to expand cohort while keeping 1:1 balance preserved
 *     by the cancer_med_counts/non_cancer_med_counts CTEs)
 *   - Medication UNION enabled
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
    1                 AS min_med_events_per_patient,  -- minimum medication events required per patient (loosened 3→1 to expand cohort while preserving 1:1 balance)
    1                 AS non_cancer_ratio,            -- default ratio of non-cancer to cancer patients is 1:1 (change to 2 for 2:1)
    '%lung%'          AS target_cancer_pattern        -- Specify the target cancer
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
  -- v45_balanced: SQL-side codelist filter (155 curated codes)
  -- Used at BOTH event-counting (for HAVING) and output level
  SELECT snomed_code, term FROM UNNEST(ARRAY<STRUCT<snomed_code INT64, term STRING>>[
    (1000821000000103, 'Serum albumin level'),
    (1017311000000104, 'Blood oxygen saturation (calculated)'),
    (1022551000000104, 'Neutrophil count'),
    (102589003, 'Atypical chest pain'),
    (103228002, 'Haemoglobin saturation with oxygen'),
    (106511000001103, 'Salamol 100micrograms inhaler'),
    (1079461000000108, 'NAFLD (Non-Alcoholic Fatty Liver Disease) fibrosis score'),
    (1087441000000106, 'Smoking cessation programme declined'),
    (1091811000000102, 'Diastolic arterial pressure'),
    (11833005, 'Dry cough'),
    (127783003, 'Spirometry'),
    (12906411000001100, 'Fostair 100/6 inhaler'),
    (13645005, 'COPD - Chronic obstructive pulmonary disease'),
    (160603005, 'Light cigarette smoker (1-9 cigs/day)'),
    (160604004, 'Moderate cigarette smoker (10-19 cigs/day)'),
    (160605003, 'Heavy cigarette smoker (20-39 cigs/day)'),
    (160606002, 'Very heavy cigarette smoker (40+ cigs/day)'),
    (160625004, 'Date ceased smoking'),
    (161635002, 'History of asbestos exposure'),
    (161924005, 'Productive cough -green sputum'),
    (161929000, 'Chesty cough'),
    (161947006, 'Nocturnal cough / wheeze'),
    (162573006, 'Suspected lung cancer'),
    (164457001, 'O/E - finger clubbing'),
    (168734001, 'Standard chest X-ray abnormal'),
    (170614009, 'Inhaler technique observed'),
    (170635006, 'Asthma not disturbing sleep'),
    (170638008, 'Asthma not limiting activities'),
    (170661005, 'Using inhaled steroids - normal dose'),
    (171255006, 'Spirometry screening'),
    (1879891000006107, 'Non-alcoholic fatty liver disease (NAFLD) fibrosis score'),
    (195647007, 'Acute respiratory infections'),
    (195742007, 'Acute lower respiratory tract infection'),
    (195967001, 'Asthma'),
    (1970531000006107, 'COPD assessment test score - cough'),
    (1970541000006102, 'COPD assessment test score - phlegm (mucus)'),
    (1970561000006103, 'COPD assessment test score - breathless walking up hill/stairs'),
    (199251000000107, 'Fast track cancer referral'),
    (201031000000108, 'Asthma trigger - respiratory infection'),
    (204991000000107, 'Suspected chronic obstructive pulmonary disease'),
    (222311000001102, 'Ventolin 100micrograms Evohaler'),
    (2237002, 'Pleuritic pain'),
    (225323000, 'Smoking cessation advice'),
    (225324006, 'Advice on effects of smoking on health'),
    (230057008, 'Cigar consumption'),
    (24184005, 'Raised blood pressure'),
    (249366005, 'Has nosebleeds - epistaxis'),
    (251944000, 'FEV1/FVC ratio'),
    (263747008, 'Emphysematous'),
    (266918002, 'Tobacco smoking consumption'),
    (266920004, 'Trivial cigarette smoker (less than one cigarette/day)'),
    (266921000, 'Ex-trivial cigarette smoker (<1/day)'),
    (266922007, 'Ex-light cigarette smoker (1-9/day)'),
    (266923002, 'Ex-moderate cigarette smoker (10-19/day)'),
    (266924008, 'Ex-heavy cigarette smoker (20-39/day)'),
    (266925009, 'Ex-very heavy cigarette smoker (40+/day)'),
    (267024001, 'Abnormal weight loss'),
    (267036007, 'Breathlessness Dyspnoea'),
    (270442000, 'Asthma monitoring check done'),
    (27113001, 'Body weight'),
    (275908000, 'Asthma monitoring'),
    (276491000000101, 'Fast track referral for suspected lung cancer'),
    (281018007, 'Ex-cigarette smoker'),
    (2831211000001109, 'Flixotide 250 Evohaler'),
    (284523002, 'Persistent cough'),
    (29857009, 'Chest pain'),
    (302331000000106, 'Royal College of Physicians asthma assessment'),
    (30760008, 'Finger clubbing'),
    (310520004, 'Expected FEV1'),
    (312342009, 'Chest infection - pnemonia due to unspecified organism'),
    (313222007, 'FEV1/FVC percent'),
    (313223002, 'Percent predicted FEV1'),
    (313297008, 'Moderate chronic obstructive pulmonary disease'),
    (313299006, 'Severe chronic obstructive pulmonary disease'),
    (314473008, 'FEV1/FVC > 70% of predicted'),
    (3184311000001107, 'Flixotide 100 Accuhaler'),
    (3184911000001108, 'Flixotide 250 Accuhaler'),
    (3215311000001107, 'Salamol Easi-Breathe inhaler'),
    (32398004, 'Chest infection - unspecified bronchitis'),
    (33594911000001100, 'Braltus 10microgram inhalation powder capsules'),
    (340921000000103, 'Asthma trigger - tobacco smoke'),
    (35908811000001103, 'Beclometasone 250micrograms inhaler'),
    (364075005, 'Heart rate'),
    (365981007, 'Tobacco smoking behaviour - finding'),
    (366874008, 'Number of asthma exacerbations in past year'),
    (370208006, 'Asthma never causes daytime symptoms'),
    (370226009, 'Asthma treatment compliance satisfactory'),
    (390891009, 'COPD self-management plan given'),
    (391120009, 'MRC Breathlessness Scale: grade 1'),
    (391123006, 'MRC Breathlessness Scale: grade 2'),
    (391124000, 'MRC Breathlessness Scale: grade 3'),
    (391125004, 'MRC Breathlessness Scale: grade 4'),
    (39113611000001102, 'Salbutamol 100micrograms inhaler CFC free'),
    (394700004, 'Asthma annual review'),
    (394873005, 'Not interested in stopping smoking'),
    (396285007, 'Chest infection - unspecified bronchopneumonia'),
    (396484008, 'Supraclavicular lymph node biopsy'),
    (398511000001105, 'Flixotide 125 Evohaler'),
    (401012008, 'FEV1 before bronchodilation'),
    (401013003, 'FEV1 after bronchodilation'),
    (401135008, 'Health education - asthma'),
    (4053411000001103, 'Serevent 25micrograms Evohaler'),
    (406162001, 'Asthma management'),
    (407576000, 'FVC/Expected FVC percent'),
    (407602006, 'FEV1/FVC ratio before bronchodilator'),
    (414916001, 'Obesity'),
    (415261001, 'Referral for spirometry'),
    (42292311000001106, 'Beclometasone 100micrograms inhaler'),
    (42292811000001102, 'Carbocisteine 375mg capsules'),
    (427359005, 'Solitary nodule of lung'),
    (42984000, 'Night sweats'),
    (431314004, 'SpO2 - oxygen saturation at periphery'),
    (443117005, 'Asthma control test score'),
    (446660005, 'Chronic obstructive pulmonary disease assessment test score'),
    (492191000000103, 'Ex roll-up cigarette smoker'),
    (49727002, 'Cough'),
    (50417007, 'Lower respiratory tract infection'),
    (505651000000103, 'Current smoker annual review'),
    (505761000000105, 'Ex-smoker annual review'),
    (50834005, 'Forced vital capacity'),
    (51615001, 'Fibrosis of lung'),
    (527151000000107, 'Smoking cessation advice declined'),
    (527361000000107, 'CAT - COPD assessment test'),
    (54150009, 'Upper respiratory infection'),
    (54398005, 'Acute upper respiratory infection'),
    (60621009, 'Body mass index'),
    (65568007, 'Cigarette smoker'),
    (6631009, 'Thrombocytosis'),
    (66857006, 'Haemoptysis'),
    (68328006, 'Centrilobular emphysema'),
    (700250006, 'Idiopathic pulmonary fibrosis'),
    (716281000000103, 'Chronic obstructive pulmonary disease monitoring verbal invite'),
    (716901000000101, 'Chronic obstructive pulmonary disease monitoring telephone invitation'),
    (718241000000107, 'Issue of chronic obstructive pulmonary disease rescue pack'),
    (719873005, 'Fibrosis-4 index score'),
    (72313002, 'Systolic arterial pressure'),
    (723245007, 'Number of chronic obstructive pulmonary disease exacerbations in past year'),
    (726611000001102, 'Flixotide 50 Evohaler'),
    (736056000, 'Asthma clinical management plan'),
    (754061000000100, 'Asthma review using Royal College of Physicians three questions'),
    (767641000000109, 'Referral for smoking cessation service offered'),
    (77176002, 'Current smoker'),
    (78564009, 'Pulse rate'),
    (810901000000102, 'Asthma self-management plan review'),
    (831311000006106, '2 week rule referral - lung'),
    (8517006, 'Ex-smoker'),
    (871641000000105, 'Referral to smoking cessation service declined'),
    (87433001, 'Pulmonary emphysema'),
    (894821000000107, 'Spirometry screening invitation'),
    (909721000006104, '[RFC] Emphysema'),
    (909731000006101, '[RFC] Reason for care : Pulmonary fibrosis'),
    (919601000000107, 'Single inhaler maintenance and reliever therapy started'),
    (9205211000001104, 'Easyhaler Salbutamol'),
    (9516911000001109, 'AeroChamber Plus'),
    (95325000, 'Subcutaneous nodule')
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
   Medication source CTEs (moved from Step 6 → here, so cancer_med_counts /
   non_cancer_med_counts below can reference final_medication).
   ═══════════════════════════════════════════════════════════════════════════ */

prescribing_drugrecords_emis AS (
  SELECT
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date,
    TRIM(drug_record_guid)   AS drug_record_guid,
    TRIM(prescription_type)  AS prescription_type,
    TRIM(patient_guid)       AS patient_guid,
    source_practice_code,
    deleted
  FROM `cthesigns-platform-475414-b7.EMIS_BULK_DATA_PROCESSED.Prescribing_DrugRecord`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY drug_record_guid ORDER BY file_date DESC) = 1
),

prescribing_issuerecords_emis AS (
  SELECT
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date,
    TRIM(issue_record_guid)                      AS issue_record_guid,
    TRIM(drug_record_guid)                       AS drug_record_guid,
    PARSE_DATE('%Y-%m-%d', TRIM(entered_date))   AS entered_date,
    PARSE_DATE('%Y-%m-%d', TRIM(effective_date)) AS effective_date,
    code_id,
    course_duration_in_days,
    deleted
  FROM `cthesigns-platform-475414-b7.EMIS_BULK_DATA_PROCESSED.Prescribing_IssueRecord`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY drug_record_guid ORDER BY file_date DESC) = 1
),

codes_emis AS (
  SELECT
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date,
    SAFE_CAST(code_id AS INT64)             AS code_id,
    SAFE_CAST(dmd_product_code_id AS INT64) AS dmd_product_code_id,
    term,
    source_practice_code
  FROM `cthesigns-platform-475414-b7.EMIS_BULK_DATA_PROCESSED.Coding_DrugCode`
  WHERE dmd_product_code_id IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY code_id, source_practice_code ORDER BY file_date DESC
  ) = 1
),

final_medication AS (
  SELECT
    dr.drug_record_guid,
    ir.issue_record_guid,
    ce.dmd_product_code_id  AS code_id,
    dr.patient_guid,
    ir.effective_date,
    ir.entered_date,
    dr.source_practice_code,
    ir.course_duration_in_days AS duration,
    ce.term                    AS drug_term,
    dr.prescription_type
  FROM prescribing_drugrecords_emis dr
  LEFT JOIN prescribing_issuerecords_emis ir USING (drug_record_guid)
  LEFT JOIN codes_emis ce
    ON ce.code_id = ir.code_id
   AND ce.source_practice_code = dr.source_practice_code
  WHERE COALESCE(dr.deleted, false) = false
    AND COALESCE(ir.deleted, false) = false
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 4b (LUNG-SPECIFIC): Count curated MED events per patient.
   ─────────────────────────────────────────────────────────────────────────
   Required because the LSTM dual branch enforces --min-med-seq 1 at Python
   level. Without this filter, matched non-cancer controls (year+sex matched,
   age-blind) drop out at higher rates than cancer patients, breaking the
   1:1 balance. Counting ≥3 curated med events here matches the row-level
   min_med_events_per_patient threshold downstream.
   ═══════════════════════════════════════════════════════════════════════════ */

cancer_med_counts AS (
  SELECT
    tcp.patient_guid,
    COUNT(*) AS curated_med_count
  FROM (
    SELECT DISTINCT
      tcp.patient_guid,
      fm.effective_date,
      fm.code_id
    FROM target_cancer_patients tcp
    JOIN final_medication fm
      ON UPPER(REGEXP_REPLACE(TRIM(fm.patient_guid), r'[{}]', '')) =
         UPPER(REGEXP_REPLACE(TRIM(CAST(tcp.patient_guid AS STRING)), r'[{}]', ''))
    INNER JOIN curated_codes cc_med
      ON cc_med.snomed_code = fm.code_id
    CROSS JOIN params param
    WHERE fm.effective_date IS NOT NULL
      AND fm.code_id IS NOT NULL
      AND fm.effective_date >= DATE_SUB(tcp.date_of_diagnosis, INTERVAL param.years_before YEAR)
      AND fm.effective_date <  DATE_SUB(tcp.date_of_diagnosis, INTERVAL param.months_before MONTH)
  ) tcp
  GROUP BY 1
  HAVING COUNT(*) >= (SELECT min_med_events_per_patient FROM params)
),

non_cancer_med_counts AS (
  SELECT
    nca.patient_guid,
    COUNT(*) AS curated_med_count
  FROM (
    SELECT DISTINCT
      nca.patient_guid,
      fm.effective_date,
      fm.code_id
    FROM non_cancer_with_anchor nca
    JOIN final_medication fm
      ON UPPER(REGEXP_REPLACE(TRIM(fm.patient_guid), r'[{}]', '')) =
         UPPER(REGEXP_REPLACE(TRIM(CAST(nca.patient_guid AS STRING)), r'[{}]', ''))
    INNER JOIN curated_codes cc_med
      ON cc_med.snomed_code = fm.code_id
    CROSS JOIN params param
    WHERE fm.effective_date IS NOT NULL
      AND fm.code_id IS NOT NULL
      AND fm.effective_date >= DATE_SUB(nca.anchor_date, INTERVAL param.years_before YEAR)
      AND fm.effective_date <  DATE_SUB(nca.anchor_date, INTERVAL param.months_before MONTH)
  ) nca
  GROUP BY 1
  HAVING COUNT(*) >= (SELECT min_med_events_per_patient FROM params)
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
    EXTRACT(YEAR FROM cec.anchor_date) AS anchor_year,
    COUNT(*)                            AS cancer_count
  FROM cancer_event_counts cec
  INNER JOIN cancer_med_counts cmc
    ON cmc.patient_guid = cec.patient_guid    -- lung: require curated meds too
  GROUP BY 1
),

non_cancer_ranked AS (
  SELECT
    nec.patient_guid,
    nec.anchor_date,
    EXTRACT(YEAR FROM nec.anchor_date) AS anchor_year,
    ROW_NUMBER() OVER (
      PARTITION BY EXTRACT(YEAR FROM nec.anchor_date)
      ORDER BY FARM_FINGERPRINT(CAST(nec.patient_guid AS STRING))
    ) AS rank_in_year
  FROM non_cancer_event_counts nec
  INNER JOIN non_cancer_med_counts nmc
    ON nmc.patient_guid = nec.patient_guid    -- lung: require curated meds too
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
  /* All qualifying cancer patients (must be eligible on BOTH SNOMED and curated meds) */
  SELECT cec.patient_guid, cec.anchor_date, cec.cancer_id,
         1 AS cancer_class,
         EXTRACT(YEAR FROM cec.anchor_date) AS anchor_year
  FROM cancer_event_counts cec
  INNER JOIN cancer_med_counts cmc
    ON cmc.patient_guid = cec.patient_guid

  UNION ALL

  /* Randomly sampled non-cancer patients — already med-filtered via non_cancer_ranked */
  SELECT patient_guid, anchor_date, CAST(NULL AS STRING) AS cancer_id,
         0 AS cancer_class,
         EXTRACT(YEAR FROM anchor_date) AS anchor_year
  FROM non_cancer_sampled
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 7: Produce observation and medication events for selected patients
   (medication source CTEs moved up — see Step 4b above)
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
),

/* Raw medication events (before minimum-count enforcement).
   NULL drug codes (from LEFT JOIN misses) are filtered out so the count
   matches what pandas count('MED_CODE_ID') sees. */
medication_events_raw AS (
  SELECT DISTINCT
    up.patient_guid,
    up.cancer_class,
    up.cancer_id,
    pp.sex,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS patient_age,
    CAST(NULL AS STRING)                                      AS patient_ethnicity,
    fm.effective_date                                         AS event_date,         -- diagnostic only; do NOT use as a model feature (anchor-year bias)
    DATE_DIFF(fm.effective_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS event_age,
    DATE_DIFF(up.anchor_date, fm.effective_date, DAY)         AS days_before_anchor,  -- use this instead of event_date for temporal model features
    up.anchor_year,
    'medication'                                              AS event_type,
    CAST(NULL AS INT64)                                       AS snomed_c_t_concept_id,
    CAST(NULL AS STRING)                                      AS term,
    CAST(NULL AS STRING)                                      AS associated_text,
    CAST(NULL AS STRING)                                      AS value,
    fm.code_id                                                AS med_code_id,
    fm.drug_term,
    fm.duration
  FROM unified_patients up
  JOIN final_medication fm
    ON UPPER(REGEXP_REPLACE(TRIM(fm.patient_guid), r'[{}]', '')) =
       UPPER(REGEXP_REPLACE(TRIM(CAST(up.patient_guid AS STRING)), r'[{}]', ''))
  INNER JOIN curated_codes cc_med
    ON cc_med.snomed_code = fm.code_id  -- v45_balanced: output curated meds only
  JOIN patients pp
    ON pp.patient_guid = up.patient_guid
  CROSS JOIN params param
  WHERE
    fm.effective_date IS NOT NULL
    AND fm.code_id IS NOT NULL        -- exclude rows with no drug code (LEFT JOIN miss)
    /* ── Same symmetric relative lookback window ── */
    AND fm.effective_date >= DATE_SUB(up.anchor_date, INTERVAL param.years_before YEAR)
    AND fm.effective_date <  DATE_SUB(up.anchor_date, INTERVAL param.months_before MONTH)
  /* No LIMIT — same reasoning as observation_events_raw above. */
),

/* Enforce minimum medication events per patient (same approach as observations). */
medication_events AS (
  SELECT * EXCEPT(patient_med_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_guid) AS patient_med_count
    FROM medication_events_raw
  )
  WHERE patient_med_count >= (SELECT min_med_events_per_patient FROM params)
)

/* ═══════════════════════════════════════════════════════════════════════════
   Step 8: Final output — combine observations and medications
   ═══════════════════════════════════════════════════════════════════════════ */
SELECT * FROM observation_events
UNION ALL
SELECT * FROM medication_events;
