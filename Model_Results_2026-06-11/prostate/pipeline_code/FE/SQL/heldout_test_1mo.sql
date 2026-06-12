/*
 * Unified cancer / non-cancer patient events for ML training — PROSTATE variant.
 *
 * This is a fixed-cohort variant of unified_cancer_noncancer42_v3.sql:
 *   - Cancer patients      = the 1,500 ids in  prj-cts-ai-dev-sp.EMIS_BULK_DATA_temp.prostate_cancer_1500
 *   - Non-cancer patients  = the 50,000 ids in prj-cts-ai-dev-sp.EMIS_BULK_DATA_temp.no_cancer_prostate_50000
 *
 * Instead of qualifying / per-year balancing / random sampling inside the query,
 * the patient cohort is taken verbatim from those tables. Each patient's anchor
 * date is recomputed here with the SAME deterministic logic used to build the
 * tables (target_cancer_patients for cancer; non_cancer_with_anchor for non-cancer),
 * so the anchors reproduce exactly. The event-building pipeline (lookback window,
 * exclusions, leakage filters, min-event enforcement, medications) is unchanged.
 */

WITH params AS (
  SELECT
    DATE '1900-01-01' AS longterm_mh_start,
    DATE '2025-01-01' AS longterm_mh_end,
    DATE '2010-01-01' AS anchor_window_start,        -- inclusive lower bound for anchor dates
    DATE '2025-01-01' AS anchor_window_end,          -- exclusive upper bound for anchor dates
    100                AS years_before,
    1                 AS months_before,
    0                 AS min_obs_events_per_patient,  -- minimum observation events required per patient
    0                 AS min_med_events_per_patient,  -- minimum medication events required per patient
    '%prostate%'          AS target_cancer_pattern        -- target cancer (must match prostate_cancer_1500)
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
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Coding_ClinicalCode`
  WHERE snomed_c_t_concept_id IS NOT NULL
    AND term IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY code_id, source_practice_code ORDER BY file_date DESC
  ) = 1
),

patients AS (
  /* Dedupe per patient_guid — Admin_Patient has multiple snapshots per patient
     across bulk-file dates, and they can carry conflicting date_of_birth values.
     ANY_VALUE would dedupe but pick a non-deterministic version, so the same
     patient can pick up different DOBs across runs and produce impossible ages
     (e.g. age_at_anchor = -7). Pick the LATEST snapshot deterministically via
     ROW_NUMBER() ordered by the bulk-file date parsed from file_name. */
  SELECT pa.patient_guid, pa.sex, pa.date_of_birth, pa.deleted,
         rc.LABEL_16 AS patient_ethnicity_16,
         rc.LABEL_6  AS patient_ethnicity_6
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Admin_Patient` pa
  LEFT JOIN `prj-cts-ai-dev-sp.EMIS_BULK_DATA_temp.Patient_Ethnicity` rc
      on rc.patient_guid = pa.patient_guid
  WHERE pa.deleted != true OR pa.deleted IS NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY pa.patient_guid
    ORDER BY pa.source_date DESC
  ) = 1
),

/* Problem-list metadata, one row per observation_guid (latest by SOURCE_DATE).
   CareRecord_Problem shares observation_guid with CareRecord_Observation, so it
   enriches the observation events that are also flagged as problems. */
care_record_problem AS (
  SELECT
    observation_guid,
    patient_guid,
    source_practice_code,
    problem_status_description,
    significance_description
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Problem`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_guid, source_practice_code, observation_guid ORDER BY SOURCE_DATE DESC
  ) = 1
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
    dc.source_practice_code                      AS dc_source_practice_code,
    prob.problem_status_description              AS problem_status_description,
    prob.significance_description                AS significance_description
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` AS co
  LEFT JOIN diagnostic_codes AS dc
    ON SAFE_CAST(co.code_id AS INT64) = SAFE_CAST(dc.code_id AS INT64)
   AND co.source_practice_code = dc.source_practice_code
  LEFT JOIN care_record_problem AS prob
    ON prob.observation_guid = co.observation_guid
   AND prob.patient_guid = co.patient_guid
   AND prob.source_practice_code = co.source_practice_code
  CROSS JOIN params param
  WHERE co.effective_date IS NOT NULL
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) >= param.longterm_mh_start
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) <  param.longterm_mh_end
    /* NOTE: do NOT filter cancer terms here. medical_history is the single source
       used to IDENTIFY cancer patients (any_cancer_patient / target_cancer_patients).
       Removing 'cancer'-termed rows here dropped real diagnosis codes and could
       mislabel cancer patients as non-cancer. Label-leakage in the event features
       is handled separately by the cancer_snomed_codes exclusion downstream. */
   QUALIFY ROW_NUMBER() OVER (
    PARTITION BY co.patient_guid, co.source_practice_code, co.observation_guid ORDER BY co.source_date DESC, SAFE_CAST(co.code_id AS INT64)
  ) = 1
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 1: Cancer-related codes + anchor recomputation
   ═══════════════════════════════════════════════════════════════════════════ */

cancer_snomed_codes AS (
  SELECT DISTINCT SAFE_CAST(SNOMED_CODE AS INT64) AS snomed_code
  FROM `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes`
  WHERE LOWER(cancer_id) NOT LIKE '%disease%'
),

any_cancer_patient AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
  JOIN cancer_snomed_codes csc
    ON mh.snomed_c_t_concept_id = csc.snomed_code
),

/* Earliest target-cancer diagnosis date per patient = anchor (same as table build). */
target_cancer_patients AS (
  SELECT patient_guid, date_of_diagnosis, cancer_id
  FROM (
    SELECT
      mh.patient_guid,
      mh.effective_date AS date_of_diagnosis,
      dcc.cancer_id,
      ROW_NUMBER() OVER (PARTITION BY mh.patient_guid ORDER BY mh.effective_date, dcc.cancer_id) AS rn
    FROM medical_history mh
    JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` dcc
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

palliative_patients AS (
  SELECT DISTINCT patient_guid
  FROM medical_history
  WHERE snomed_c_t_concept_id = 1403151000000103
),

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

/* Non-cancer anchor = one deterministically-sampled in-window observation event
   (identical to the logic that produced no_cancer_prostate_50000). */
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
    /* Moved from medical_history: exclude cancer-termed observations from the
       event features (leakage) without affecting cancer-patient identification. */
    AND LOWER(mh.term) NOT LIKE '%cancer%'
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
   Step 2: Fixed patient cohort — ids taken verbatim from the sample tables.
   No qualifying / per-year balancing / sampling here; the tables define the set.
   ═══════════════════════════════════════════════════════════════════════════ */

cancer_selected AS (
  SELECT tcp.patient_guid,
         tcp.date_of_diagnosis AS anchor_date,
         tcp.cancer_id
  FROM target_cancer_patients tcp
  WHERE tcp.patient_guid IN (
    SELECT patient_guid FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_temp.prostate_cancer_1500`
  )
),

non_cancer_selected AS (
  SELECT nca.patient_guid, nca.anchor_date
  FROM non_cancer_with_anchor nca
  WHERE nca.patient_guid IN (
    SELECT patient_guid FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_temp.no_cancer_prostate_50000`
  )
),

unified_patients AS (
  SELECT patient_guid, anchor_date, cancer_id,
         1 AS cancer_class,
         EXTRACT(YEAR FROM anchor_date) AS anchor_year
  FROM cancer_selected

  UNION ALL

  SELECT patient_guid, anchor_date, CAST(NULL AS STRING) AS cancer_id,
         0 AS cancer_class,
         EXTRACT(YEAR FROM anchor_date) AS anchor_year
  FROM non_cancer_selected
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 3: Build medication data from prescribing tables
   ═══════════════════════════════════════════════════════════════════════════ */

prescribing_drugrecords_emis AS (
  SELECT
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date,
    TRIM(drug_record_guid)   AS drug_record_guid,
    TRIM(prescription_type)  AS prescription_type,
    TRIM(patient_guid)       AS patient_guid,
    source_practice_code,
    deleted
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Prescribing_DrugRecord`
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
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Prescribing_IssueRecord`
  /* Dedup on the issue record's own PK to collapse the same issue appearing across
     multiple bulk-data file dates. Partitioning by drug_record_guid was wrong: a drug
     record has many issues (repeat prescriptions), so it kept only one issue per
     authorisation and discarded the prescribing history. */
  QUALIFY ROW_NUMBER() OVER (PARTITION BY issue_record_guid ORDER BY file_date DESC) = 1
),

codes_emis AS (
  SELECT
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date,
    SAFE_CAST(code_id AS INT64)             AS code_id,
    SAFE_CAST(dmd_product_code_id AS INT64) AS dmd_product_code_id,
    term,
    source_practice_code
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Coding_DrugCode`
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
   Step 4: Produce observation and medication events for selected patients
   ═══════════════════════════════════════════════════════════════════════════ */

observation_events_raw AS (
  SELECT DISTINCT
    up.patient_guid,
    up.cancer_class,
    up.cancer_id,
    pp.sex,
    pp.patient_ethnicity_16,
    pp.patient_ethnicity_6,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS patient_age,         -- age at longterm_mh_end (default 2026-01-01); kept for backward compat
    DATE_DIFF(up.anchor_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS age_at_anchor,       -- age at the patient's anchor_date (clinically correct)
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
    CAST(mh.problem_status_description AS STRING)             AS problem_status_description,
    CAST(mh.significance_description AS STRING)               AS significance_description,
    CAST(NULL AS INT64)                                       AS med_code_id,
    CAST(NULL AS STRING)                                      AS drug_term,
    CAST(NULL AS INT64)                                       AS duration
  FROM unified_patients up
  JOIN medical_history mh
    ON mh.patient_guid = up.patient_guid
  JOIN patients pp
    ON pp.patient_guid = up.patient_guid
  LEFT JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Ethnicity_Snowmed_Codes` rc
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
    /* Moved from medical_history: exclude cancer-termed observations from the
       event features (leakage) without affecting cancer-patient identification. */
    AND LOWER(mh.term) NOT LIKE '%cancer%'
),

observation_events AS (
  SELECT * EXCEPT(patient_obs_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_guid) AS patient_obs_count
    FROM observation_events_raw
  )
  WHERE patient_obs_count >= (SELECT min_obs_events_per_patient FROM params)
),

medication_events_raw AS (
  SELECT DISTINCT
    up.patient_guid,
    up.cancer_class,
    up.cancer_id,
    pp.sex,
    CAST(NULL AS STRING)                                      AS patient_ethnicity_16,
    CAST(NULL AS STRING)                                      AS patient_ethnicity_6,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS patient_age,         -- age at longterm_mh_end (default 2026-01-01); kept for backward compat
    DATE_DIFF(up.anchor_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS age_at_anchor,       -- age at the patient's anchor_date (clinically correct)
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
    CAST(NULL AS STRING)                                      AS problem_status_description,
    CAST(NULL AS STRING)                                      AS significance_description,
    fm.code_id                                                AS med_code_id,
    fm.drug_term,
    fm.duration
  FROM unified_patients up
  JOIN final_medication fm
    ON UPPER(REGEXP_REPLACE(TRIM(fm.patient_guid), r'[{}]', '')) =
       UPPER(REGEXP_REPLACE(TRIM(CAST(up.patient_guid AS STRING)), r'[{}]', ''))
  JOIN patients pp
    ON pp.patient_guid = up.patient_guid
  CROSS JOIN params param
  WHERE
    fm.effective_date IS NOT NULL
    AND fm.code_id IS NOT NULL        -- exclude rows with no drug code (LEFT JOIN miss)
    /* ── Same symmetric relative lookback window ── */
    AND fm.effective_date >= DATE_SUB(up.anchor_date, INTERVAL param.years_before YEAR)
    AND fm.effective_date <  DATE_SUB(up.anchor_date, INTERVAL param.months_before MONTH)
),

medication_events AS (
  SELECT * EXCEPT(patient_med_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_guid) AS patient_med_count
    FROM medication_events_raw
  )
  WHERE patient_med_count >= (SELECT min_med_events_per_patient FROM params)
)

/* ═══════════════════════════════════════════════════════════════════════════
   Step 5: Final output — combine observations and medications
   ═══════════════════════════════════════════════════════════════════════════ */
SELECT * FROM observation_events
UNION ALL
SELECT * FROM medication_events
ORDER BY patient_guid;
