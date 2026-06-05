/*
 * Prostate cancer prediction — Approach B + Age-Band Matching (v3)
 *
 * Pure data extraction. Codelist filtering and CATEGORY/TIME_WINDOW assignment
 * happen in the Python preprocessor before FE.
 *
 * COHORT DESIGN:
 *   - Cancer:     all prostate cancer patients diagnosed in [2017-01-01, 2026-04-25]
 *   - Non-cancer: ALL male patients without any cancer diagnosis,
 *                 age-band stratified random sampling at 1:10 ratio per band.
 *                 NO curated-codes filter — sparse-record patients included
 *                 so the model learns "no clinical signal = no cancer" boundary.
 *
 * MATCHING:
 *   - 5-year age bands stratified random sampling
 *   - Year of anchor flows naturally (NOT per-cell matched, avoids sparsity)
 *   - Male only (prostate-specific)
 *   - For each age_band: sample 10x cancer count from non-cancer pool
 *   - Result: similar (not identical) age distributions; age shortcut removed,
 *     year-of-anchor variation preserved within bands
 *
 * ANCHOR:
 *   - Cancer:     date_of_diagnosis
 *   - Non-cancer: random sampled event date in [2017-01-01, 2026-04-25]
 *                 (deterministic via FARM_FINGERPRINT)
 *
 * LOOKBACK:
 *   - Events in [anchor - 60 months, anchor - {months_before}]
 *   - Change months_before for different prediction windows:
 *       12mo window: months_before = 12
 *        9mo window: months_before = 9
 *        6mo window: months_before = 6
 *        3mo window: months_before = 3
 *        2mo window: months_before = 2
 *        1mo window: months_before = 1
 *
 * LEAKAGE PREVENTION:
 *   - Cancer SNOMED codes excluded from event rows for both groups
 *   - Term-level NOT LIKE '%cancer%' filter (defence-in-depth)
 *   - Palliative-care patients excluded from both groups
 *   - Same filters/exclusions applied identically to both classes
 *   - Curated leakage codes (PSA surveillance, cystoscopy, urology pathway,
 *     urology imaging) to be applied in Python preprocessor using leakage sheets
 *
 * OUTPUT SCHEMA:
 *   PATIENT_GUID, CANCER_CLASS, CANCER_ID, SEX, PATIENT_AGE, AGE_BAND,
 *   PATIENT_ETHNICITY, ANCHOR_DATE, ANCHOR_YEAR, EVENT_DATE, EVENT_AGE,
 *   DAYS_BEFORE_ANCHOR, MONTHS_BEFORE_ANCHOR, EVENT_TYPE,
 *   SNOMED_C_T_CONCEPT_ID, TERM, ASSOCIATED_TEXT, VALUE,
 *   MED_CODE_ID, DRUG_TERM, DURATION
 */

WITH params AS (
  SELECT
    DATE '1950-01-01' AS longterm_mh_start,
    DATE '2026-04-25' AS longterm_mh_end,
    DATE '2017-01-01' AS anchor_window_start,
    DATE '2026-04-25' AS anchor_window_end,
    5                 AS years_before,                -- 60mo lookback
    3                 AS months_before,               -- ← CHANGE THIS: 12, 9, 6, 3, 2, or 1
    10                AS min_obs_events_per_patient,
    1                 AS min_med_events_per_patient,
    10                AS non_cancer_ratio,            -- 1:10 cancer:non-cancer
    '%prostate%'      AS target_cancer_pattern
),

/* ═══════════════════════════════════════════════════════════════════════════
   Shared reference tables
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
  SELECT patient_guid, sex, date_of_birth, deleted
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Admin_Patient`
  WHERE deleted != true OR deleted IS NULL
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
    dc.term,
    SAFE_CAST(dc.snomed_c_t_concept_id AS INT64) AS snomed_c_t_concept_id,
    dc.snomed_c_t_description_id
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` AS co
  LEFT JOIN diagnostic_codes AS dc
    ON SAFE_CAST(co.code_id AS INT64) = SAFE_CAST(dc.code_id AS INT64)
   AND co.source_practice_code = dc.source_practice_code
  CROSS JOIN params param
  WHERE co.effective_date IS NOT NULL
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) >= param.longterm_mh_start
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) <  param.longterm_mh_end
    AND (dc.term IS NULL OR LOWER(dc.term) NOT LIKE '%cancer%')
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY co.patient_guid, co.code_id, co.effective_date
    ORDER BY co.entered_date DESC
  ) = 1
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 1: Identify cancer patients
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

target_cancer_patients AS (
  SELECT patient_guid, date_of_diagnosis, cancer_id
  FROM (
    SELECT
      mh.patient_guid,
      mh.effective_date AS date_of_diagnosis,
      dcc.cancer_id,
      ROW_NUMBER() OVER (
        PARTITION BY mh.patient_guid
        ORDER BY mh.effective_date, dcc.cancer_id
      ) AS rn
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

/* ═══════════════════════════════════════════════════════════════════════════
   Step 2: Find non-cancer patients (male only, no cancer, no palliative)
   ═══════════════════════════════════════════════════════════════════════════ */

non_cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
  JOIN patients pp
    ON pp.patient_guid = mh.patient_guid
  WHERE pp.sex = 'M'
    AND NOT EXISTS (
      SELECT 1 FROM any_cancer_patient acp
      WHERE acp.patient_guid = mh.patient_guid
    )
    AND NOT EXISTS (
      SELECT 1 FROM palliative_patients pal
      WHERE pal.patient_guid = mh.patient_guid
    )
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 3: Random anchor date per non-cancer patient
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
   Step 4: Count events in lookback window + apply minimums
   ═══════════════════════════════════════════════════════════════════════════ */

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
      CAST(mh.term AS STRING)             AS term,
      CAST(mh.associated_text AS STRING)  AS associated_text,
      CAST(mh.value AS STRING)            AS value
    FROM target_cancer_patients tcp
    JOIN medical_history mh
      ON mh.patient_guid = tcp.patient_guid
    CROSS JOIN params param
    WHERE NOT EXISTS (
        SELECT 1 FROM palliative_patients pal WHERE pal.patient_guid = tcp.patient_guid
      )
      AND mh.effective_date >= DATE_SUB(tcp.date_of_diagnosis, INTERVAL param.years_before YEAR)
      AND mh.effective_date <  DATE_SUB(tcp.date_of_diagnosis, INTERVAL param.months_before MONTH)
      AND mh.snomed_c_t_concept_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM cancer_snomed_codes csc
        WHERE csc.snomed_code = mh.snomed_c_t_concept_id
      )
      AND mh.observation_type NOT IN ('Immunisation')
  )
  GROUP BY 1, 2, 3
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

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
      CAST(mh.term AS STRING)             AS term,
      CAST(mh.associated_text AS STRING)  AS associated_text,
      CAST(mh.value AS STRING)            AS value
    FROM non_cancer_with_anchor nca
    JOIN medical_history mh
      ON mh.patient_guid = nca.patient_guid
    CROSS JOIN params param
    WHERE mh.effective_date >= DATE_SUB(nca.anchor_date, INTERVAL param.years_before YEAR)
      AND mh.effective_date <  DATE_SUB(nca.anchor_date, INTERVAL param.months_before MONTH)
      AND mh.snomed_c_t_concept_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM cancer_snomed_codes csc
        WHERE csc.snomed_code = mh.snomed_c_t_concept_id
      )
      AND mh.observation_type NOT IN ('Immunisation')
  )
  GROUP BY 1, 2
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 5: Age-band stratified random sampling at 1:10 ratio
   ─────────────────────────────────────────────────────────────────────────
   Per age band (5yr bins):
     - Count cancer patients
     - Randomly sample 10x non-cancer patients from same age_band
   Year of anchor flows naturally (no per-cell year constraint).

   This removes the "old = cancer" shortcut while:
     - avoiding sparse (age_band, year) cells that strict matching causes
     - preserving natural year-of-anchor distribution within each age band
   Both groups end up with similar (not identical) age distributions.
   ═══════════════════════════════════════════════════════════════════════════ */

/* Cancer patients with age band */
cancer_with_age AS (
  SELECT
    cec.patient_guid,
    cec.anchor_date,
    cec.cancer_id,
    EXTRACT(YEAR FROM cec.anchor_date) AS anchor_year,
    CAST(FLOOR(
      FLOOR(DATE_DIFF(cec.anchor_date, PARSE_DATE('%Y-%m-%d', pp.date_of_birth), DAY) / 365.25) / 5
    ) * 5 AS INT64) AS age_band
  FROM cancer_event_counts cec
  JOIN patients pp ON pp.patient_guid = cec.patient_guid
),

/* Count cancer patients per age_band */
cancer_age_counts AS (
  SELECT age_band, COUNT(*) AS cancer_count
  FROM cancer_with_age
  GROUP BY 1
),

/* Non-cancer patients with age band */
non_cancer_with_age AS (
  SELECT
    ncec.patient_guid,
    ncec.anchor_date,
    EXTRACT(YEAR FROM ncec.anchor_date) AS anchor_year,
    CAST(FLOOR(
      FLOOR(DATE_DIFF(ncec.anchor_date, PARSE_DATE('%Y-%m-%d', pp.date_of_birth), DAY) / 365.25) / 5
    ) * 5 AS INT64) AS age_band
  FROM non_cancer_event_counts ncec
  JOIN patients pp ON pp.patient_guid = ncec.patient_guid
),

/* Rank non-cancer within each age_band randomly (year flows naturally) */
non_cancer_ranked AS (
  SELECT
    patient_guid,
    anchor_date,
    anchor_year,
    age_band,
    ROW_NUMBER() OVER (
      PARTITION BY age_band
      ORDER BY FARM_FINGERPRINT(CAST(patient_guid AS STRING))
    ) AS rank_in_group
  FROM non_cancer_with_age
),

/* Sample 10x cancer count per age_band */
non_cancer_sampled AS (
  SELECT ncr.patient_guid, ncr.anchor_date, ncr.age_band, ncr.anchor_year
  FROM non_cancer_ranked ncr
  JOIN cancer_age_counts cac
    ON ncr.age_band = cac.age_band
  CROSS JOIN params param
  WHERE ncr.rank_in_group <= param.non_cancer_ratio * cac.cancer_count
),

/* ═══════════════════════════════════════════════════════════════════════════
   Unified patient set: all cancer + age-year-matched non-cancer
   ═══════════════════════════════════════════════════════════════════════════ */

unified_patients AS (
  SELECT patient_guid, anchor_date, cancer_id,
         1 AS cancer_class,
         anchor_year,
         age_band
  FROM cancer_with_age

  UNION ALL

  SELECT patient_guid, anchor_date, CAST(NULL AS STRING) AS cancer_id,
         0 AS cancer_class,
         anchor_year,
         age_band
  FROM non_cancer_sampled
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 6: Medication source CTEs
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
  QUALIFY ROW_NUMBER() OVER (PARTITION BY drug_record_guid ORDER BY file_date DESC) = 1
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
   Step 7: Produce observation and medication events
   ═══════════════════════════════════════════════════════════════════════════ */

observation_events_raw AS (
  SELECT DISTINCT
    up.patient_guid                                            AS PATIENT_GUID,
    up.cancer_class                                            AS CANCER_CLASS,
    up.cancer_id                                               AS CANCER_ID,
    pp.sex                                                     AS SEX,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR)  AS PATIENT_AGE,
    up.age_band                                                AS AGE_BAND,
    CAST(rc.ethnicity AS STRING)                               AS PATIENT_ETHNICITY,
    up.anchor_date                                             AS ANCHOR_DATE,
    up.anchor_year                                             AS ANCHOR_YEAR,
    mh.effective_date                                          AS EVENT_DATE,
    DATE_DIFF(mh.effective_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR)  AS EVENT_AGE,
    DATE_DIFF(up.anchor_date, mh.effective_date, DAY)          AS DAYS_BEFORE_ANCHOR,
    DATE_DIFF(up.anchor_date, mh.effective_date, MONTH)        AS MONTHS_BEFORE_ANCHOR,
    'observation'                                              AS EVENT_TYPE,
    mh.snomed_c_t_concept_id                                   AS SNOMED_C_T_CONCEPT_ID,
    CAST(mh.term AS STRING)                                    AS TERM,
    CAST(mh.associated_text AS STRING)                         AS ASSOCIATED_TEXT,
    CAST(mh.value AS STRING)                                   AS VALUE,
    CAST(NULL AS INT64)                                        AS MED_CODE_ID,
    CAST(NULL AS STRING)                                       AS DRUG_TERM,
    CAST(NULL AS INT64)                                        AS DURATION
  FROM unified_patients up
  JOIN medical_history mh
    ON mh.patient_guid = up.patient_guid
  JOIN patients pp
    ON pp.patient_guid = up.patient_guid
  LEFT JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Ethnicity_Snowmed_Codes` rc
    ON CAST(mh.snomed_c_t_concept_id AS STRING) = rc.code
  CROSS JOIN params param
  WHERE
    mh.effective_date >= DATE_SUB(up.anchor_date, INTERVAL param.years_before YEAR)
    AND mh.effective_date <  DATE_SUB(up.anchor_date, INTERVAL param.months_before MONTH)
    AND NOT EXISTS (
      SELECT 1 FROM cancer_snomed_codes csc
      WHERE csc.snomed_code = mh.snomed_c_t_concept_id
    )
    AND mh.snomed_c_t_concept_id IS NOT NULL
    AND mh.observation_type NOT IN ('Immunisation')
),

observation_events AS (
  SELECT * EXCEPT(patient_obs_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY PATIENT_GUID) AS patient_obs_count
    FROM observation_events_raw
  )
  WHERE patient_obs_count >= (SELECT min_obs_events_per_patient FROM params)
),

medication_events_raw AS (
  SELECT DISTINCT
    up.patient_guid                                            AS PATIENT_GUID,
    up.cancer_class                                            AS CANCER_CLASS,
    up.cancer_id                                               AS CANCER_ID,
    pp.sex                                                     AS SEX,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR)  AS PATIENT_AGE,
    up.age_band                                                AS AGE_BAND,
    CAST(NULL AS STRING)                                       AS PATIENT_ETHNICITY,
    up.anchor_date                                             AS ANCHOR_DATE,
    up.anchor_year                                             AS ANCHOR_YEAR,
    fm.effective_date                                          AS EVENT_DATE,
    DATE_DIFF(fm.effective_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR)  AS EVENT_AGE,
    DATE_DIFF(up.anchor_date, fm.effective_date, DAY)          AS DAYS_BEFORE_ANCHOR,
    DATE_DIFF(up.anchor_date, fm.effective_date, MONTH)        AS MONTHS_BEFORE_ANCHOR,
    'medication'                                               AS EVENT_TYPE,
    CAST(NULL AS INT64)                                        AS SNOMED_C_T_CONCEPT_ID,
    CAST(NULL AS STRING)                                       AS TERM,
    CAST(NULL AS STRING)                                       AS ASSOCIATED_TEXT,
    CAST(NULL AS STRING)                                       AS VALUE,
    fm.code_id                                                 AS MED_CODE_ID,
    fm.drug_term                                               AS DRUG_TERM,
    fm.duration                                                AS DURATION
  FROM unified_patients up
  JOIN final_medication fm
    ON UPPER(REGEXP_REPLACE(TRIM(fm.patient_guid), r'[{}]', '')) =
       UPPER(REGEXP_REPLACE(TRIM(CAST(up.patient_guid AS STRING)), r'[{}]', ''))
  JOIN patients pp
    ON pp.patient_guid = up.patient_guid
  CROSS JOIN params param
  WHERE
    fm.effective_date IS NOT NULL
    AND fm.code_id IS NOT NULL
    AND fm.effective_date >= DATE_SUB(up.anchor_date, INTERVAL param.years_before YEAR)
    AND fm.effective_date <  DATE_SUB(up.anchor_date, INTERVAL param.months_before MONTH)
),

medication_events AS (
  SELECT * EXCEPT(patient_med_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY PATIENT_GUID) AS patient_med_count
    FROM medication_events_raw
  )
  WHERE patient_med_count >= (SELECT min_med_events_per_patient FROM params)
)

/* ═══════════════════════════════════════════════════════════════════════════
   Step 8: Final output
   ═══════════════════════════════════════════════════════════════════════════ */

SELECT * FROM observation_events
UNION ALL
SELECT * FROM medication_events
ORDER BY PATIENT_GUID, DAYS_BEFORE_ANCHOR;
