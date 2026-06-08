/*
 * Bladder cancer prediction — 1mo window — Approach B (v4)
 *
 * Pure data extraction. Codelist filtering and CATEGORY/TIME_WINDOW assignment
 * happen in the Python preprocessor before FE.
 *
 * COHORT DESIGN (Approach B):
 *   - Cancer:     all bladder cancer patients diagnosed at any date (no anchor-window cutoff)
 *   - Non-cancer: ALL patients (both sexes) without any cancer diagnosis, year-stratified
 *                 to match cancer year distribution at 1:1 ratio (balanced cohort). NO curated-codes
 *                 filter — sparse-record patients are included so the model learns
 *                 the "no clinical signal = no cancer" boundary directly.
 *
 * ANCHOR:
 *   - Cancer:     date_of_diagnosis (their actual diagnosis date)
 *   - Non-cancer: random sampled event date (no anchor-window cutoff)
 *                 (deterministic via FARM_FINGERPRINT — reproducible)
 *
 * LOOKBACK WINDOW (1mo prediction = 1-month gap before anchor):
 *   - Events in [anchor - 12 months, anchor - 1 month]  (12-month lookback)
 *   - Only 1mo + 12mo windows are used (dual-horizon B+C)
 *
 * LEAKAGE PREVENTION:
 *   - Cancer SNOMED codes excluded from event rows for both groups
 *   - Term-level NOT LIKE '%cancer%' filter at medical_history layer (defence-in-depth)
 *   - Palliative-care patients excluded from both groups
 *   - Same filters/exclusions applied identically to both classes
 *   - No matching on age/gender/ethnicity (preserves real predictive signal)
 *
 * OUTPUT SCHEMA (no CATEGORY or TIME_WINDOW — added by Python preprocessor):
 *   PATIENT_GUID, CANCER_CLASS, CANCER_ID, SEX, PATIENT_ETHNICITY_16, PATIENT_ETHNICITY_6, PATIENT_AGE,
 *   ANCHOR_DATE, ANCHOR_YEAR, EVENT_DATE, EVENT_AGE,
 *   DAYS_BEFORE_ANCHOR, MONTHS_BEFORE_ANCHOR, EVENT_TYPE,
 *   SNOMED_C_T_CONCEPT_ID, TERM, ASSOCIATED_TEXT, VALUE,
 *   MED_CODE_ID, DRUG_TERM, DURATION
 */

WITH params AS (
  SELECT
    DATE '1950-01-01' AS longterm_mh_start,
    DATE '2026-04-25' AS longterm_mh_end,
    1                 AS years_before,                -- v3-aligned: 12-month lookback (1mo horizon)
    1                 AS months_before,               -- 1mo gap before anchor (matches 1mo-prediction)
    5                AS min_obs_events_per_patient,  -- v3-aligned: symmetric min obs events (cancer = non-cancer)
    DATE '2010-01-01' AS anchor_window_start,        -- v3: anchor dates restricted to recent window (both classes)
    DATE '2026-01-01' AS anchor_window_end,
    1                 AS min_med_events_per_patient,  -- row-level filter on med output (not cohort gate)
    1                 AS non_cancer_ratio,            -- v3-aligned: 1:1 year-stratified (balance-to-1:1 kept at split as safety)
    '%bladder%'      AS target_cancer_pattern
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
  SELECT
    pa.patient_guid,
    pa.sex,
    pa.date_of_birth,
    pa.deleted,
    rc.LABEL_16 AS patient_ethnicity_16,                 -- detailed 16-class (lung-style LEFT JOIN)
    rc.LABEL_6  AS patient_ethnicity_6                   -- aggregated 6-class (lung-style LEFT JOIN)
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Admin_Patient` pa
  LEFT JOIN `prj-cts-ai-dev-sp.EMIS_BULK_DATA_temp.Patient_Ethnicity` rc
    ON rc.patient_guid = pa.patient_guid
  WHERE pa.deleted != true OR pa.deleted IS NULL
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
    AND (dc.term IS NULL OR LOWER(dc.term) NOT LIKE '%cancer%')   -- defence: hide cancer-term events
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
    JOIN patients pp ON pp.patient_guid = mh.patient_guid
    CROSS JOIN params param
    WHERE mh.effective_date IS NOT NULL
      AND mh.effective_date >= param.anchor_window_start
      AND mh.effective_date <  param.anchor_window_end
      AND DATE_DIFF(mh.effective_date, PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) >= 18
      AND LOWER(dcc.cancer_id) NOT LIKE '%disease%'
      AND LOWER(dcc.cancer_id) NOT LIKE '%gallbladder%'  -- exclude gallbladder cancer (different organ, would contaminate cohort)
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
   Step 2: Find non-cancer patients (Approach B — NO curated-codes filter)
   All patients (both sexes) without any cancer diagnosis, not in palliative care.
   ═══════════════════════════════════════════════════════════════════════════ */

non_cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
  JOIN patients pp
    ON pp.patient_guid = mh.patient_guid
  WHERE pp.sex IN ('M', 'F')                          -- bladder: both sexes
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
   Step 3: Sample one random anchor date per non-cancer patient
   Deterministic via FARM_FINGERPRINT for reproducibility.
   ═══════════════════════════════════════════════════════════════════════════ */

-- ── v3-aligned anchor: ONE uniform-random own event per non-cancer patient ──
-- (deterministic FARM_FINGERPRINT), restricted to the anchor_window. Matches lung
-- v3: both classes' anchors fall in [anchor_window_start, anchor_window_end), so
-- anchor-year distributions align by construction (no gold-pool / fingerprint-forcing).
non_cancer_with_anchor AS (
  SELECT
    ncp.patient_guid,
    mh.effective_date AS anchor_date
  FROM non_cancer_patients ncp
  JOIN medical_history mh
    ON mh.patient_guid = ncp.patient_guid
  JOIN patients pp ON pp.patient_guid = ncp.patient_guid
  CROSS JOIN params param
  WHERE DATE_DIFF(mh.effective_date, PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) >= 18
    AND mh.effective_date >= param.anchor_window_start
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
   Step 4: Count events in lookback window per patient. Apply minimums.
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
      AND mh.effective_date >= DATE_SUB(tcp.date_of_diagnosis, INTERVAL (param.years_before * 12 + param.months_before) MONTH)
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
    WHERE mh.effective_date >= DATE_SUB(nca.anchor_date, INTERVAL (param.years_before * 12 + param.months_before) MONTH)
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
   Step 5: 1:1 year-stratified non-cancer sample
   Within each anchor year, randomly pick (1 × cancer_count[year]) non-cancer.
   No matching on age/gender/ethnicity — preserve them as predictive signals.
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

unified_patients AS (
  SELECT patient_guid, anchor_date, cancer_id,
         1 AS cancer_class,
         EXTRACT(YEAR FROM anchor_date) AS anchor_year
  FROM cancer_event_counts

  UNION ALL

  SELECT patient_guid, anchor_date, CAST(NULL AS STRING) AS cancer_id,
         0 AS cancer_class,
         EXTRACT(YEAR FROM anchor_date) AS anchor_year
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
   Step 7: Produce observation and medication events for selected patients
   ═══════════════════════════════════════════════════════════════════════════ */

observation_events_raw AS (
  SELECT DISTINCT
    up.patient_guid                                            AS PATIENT_GUID,
    up.cancer_class                                            AS CANCER_CLASS,
    up.cancer_id                                               AS CANCER_ID,
    pp.sex                                                     AS SEX,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR)  AS PATIENT_AGE,
    pp.patient_ethnicity_16                                    AS PATIENT_ETHNICITY_16,
    pp.patient_ethnicity_6                                     AS PATIENT_ETHNICITY_6,
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
  CROSS JOIN params param
  WHERE
    mh.effective_date >= DATE_SUB(up.anchor_date, INTERVAL (param.years_before * 12 + param.months_before) MONTH)
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
    pp.patient_ethnicity_16                                    AS PATIENT_ETHNICITY_16,
    pp.patient_ethnicity_6                                     AS PATIENT_ETHNICITY_6,
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
    AND fm.effective_date >= DATE_SUB(up.anchor_date, INTERVAL (param.years_before * 12 + param.months_before) MONTH)
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
   Step 8: Final output — observations and medications combined
   ═══════════════════════════════════════════════════════════════════════════ */

SELECT * FROM observation_events
UNION ALL
SELECT * FROM medication_events
ORDER BY PATIENT_GUID, DAYS_BEFORE_ANCHOR;
