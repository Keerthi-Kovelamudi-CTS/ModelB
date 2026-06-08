/*
 * Unified cancer / non-cancer patient events for ML training.
 *
 * PURPOSE:
 *   Produce a single dataset with both cancer and non-cancer patient events
 *   (observations + medications) for training an ML model to predict cancer.
 *   The following columns distinguishe the two groups:
 *        - `cancer_class` (1/0)
 *        - `cancer_id` (null for non-cancer patients)
 *
 * HOW IT WORKS:
 *   1. Find cancer patients and their diagnosis dates.
 *   2. Find non-cancer patients (no cancer diagnosis of any kind).
 *   3. For each patient, collect their medical events from a lookback window
 *      ending before diagnosis (cancer) or before their last record (non-cancer).
 *   4. Apply the same filters to both groups so the ML model can only learn
 *      from genuine clinical differences or legitimate features (e.g. age), not data collection or processing artifacts.
 *   5. Keep all cancer patients that pass the filters and data controls (e.g. minimum number of observation events), and randomly sample non-cancer patients to get
 *      equal (or balanced) numbers of each cohort. Natural differences in age, gender, ethnicity,
 *      and visit frequency are preserved as potential signals the ML model can learn from.
 *
 * HOW INFORMATION LEAKAGE IS PREVENTED OR MITIGATED SO FAR:
 *   - Same relative lookback window for both groups: [anchor - 20 years, anchor - 12 months)
 *   - Cancer-related SNOMED codes excluded from BOTH groups
 *   - Palliative-care patients excluded from BOTH groups
 *   - All filters (excluded codes, immunisation, etc.) applied identically
 *   - Minimum event thresholds applied equally (configurable in params)
 *   - No matching on clinical features (age, gender, event frequency) that
 *     could remove real predictive signals
 *
 * WHAT IS NOT MATCHED (on purpose, because these could be real predictive signals/features):
 *   - Age, gender, ethnicity distributions
 *   - Number of events per patient (visit frequency) [in the sql, but in the ML model, only the last N observation events is used to avoid information leakage related to differences in medical
 *     history due to data processing artifacts, as opposed to actual medical reasons]
 *   - Types of observations and medications
 */

WITH params AS (
  SELECT
    DATE '1950-01-01' AS longterm_mh_start,
    DATE '2026-01-01' AS longterm_mh_end,
    15               AS years_before,
    1               AS months_before,
    10              AS min_obs_events_per_patient,  -- minimum observation events required per patient
    0               AS min_med_events_per_patient,   -- minimum medication events required per patient
    1                AS non_cancer_ratio, -- default ratio of non-cancer to cancer patients is 1:1. You can change it to 2 if you like 2 times more non-cancer patients (2:1)
    '%lung%'         AS target_cancer_pattern
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
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Coding_ClinicalCode`
  WHERE snomed_c_t_concept_id IS NOT NULL
    AND term IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY code_id, source_practice_code ORDER BY file_date DESC
  ) = 1
),

patients AS (
  SELECT pa.patient_guid, pa.sex, pa.date_of_birth, pa.deleted, rc.LABEL_16 as patient_ethnicity_16, rc.LABEL_6 as patient_ethnicity_6
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Admin_Patient` pa
  LEFT JOIN `prj-cts-ai-dev-sp.EMIS_BULK_DATA_temp.Patient_Ethnicity` rc
      on rc.patient_guid = pa.patient_guid
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
    co.ENTERED_BY_USER_IN_ROLE_GUID,
    dc.term,
    SAFE_CAST(dc.snomed_c_t_concept_id AS INT64) AS snomed_c_t_concept_id,
    dc.snomed_c_t_description_id,
    dc.source_practice_code                      AS dc_source_practice_code
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` AS co
  LEFT JOIN diagnostic_codes AS dc
    ON SAFE_CAST(co.code_id AS INT64) = SAFE_CAST(dc.code_id AS INT64)
   AND co.source_practice_code = dc.source_practice_code
  CROSS JOIN params param
  WHERE co.effective_date IS NOT NULL
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) >= param.longterm_mh_start
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) <  param.longterm_mh_end
    AND lower(dc.term) NOT like '%cancer%'
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
  FROM `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes`
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

/* The specific cancer type we want to predict (e.g. breast cancer).
   For each patient, find their earliest diagnosis date and cancer type.
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
    JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` dcc
      ON CAST(mh.snomed_c_t_concept_id AS STRING) = dcc.SNOMED_CODE
    CROSS JOIN params param
    WHERE mh.effective_date IS NOT NULL
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
   Non-cancer patients use their last observation date as the anchor.
   The lookback window [anchor - 20y, anchor - 12m) is then applied
   identically to both groups.
   ═══════════════════════════════════════════════════════════════════════════ */

non_cancer_with_anchor AS (
  SELECT
    ncp.patient_guid,
    MAX(mh.effective_date) AS anchor_date
  FROM non_cancer_patients ncp
  JOIN medical_history mh
    ON mh.patient_guid = ncp.patient_guid
  WHERE mh.effective_date IS NOT NULL
  GROUP BY 1
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 4: Count observation events per patient and enforce minimums
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
   Step 5: Balance the dataset — equal cancer and non-cancer patients
   ─────────────────────────────────────────────────────────────────────────
   Keep ALL qualifying cancer patients (to maximize the positive class).
   Randomly sample non-cancer patients to match the cancer patient count.

   Important: we do NOT match on age, gender, ethnicity, or event frequency.
   These are real clinical signals that the ML model should learn from.
   For example, if breast cancer patients tend to be older or female,
   the model should see that — it's a genuine pattern, not a data artifact.

   Leakage is prevented by the symmetric SQL design: both groups go through
   the same time window, same filters, same exclusions, same minimum
   event thresholds. The only difference is the class label.
   ═══════════════════════════════════════════════════════════════════════════ */

non_cancer_sampled AS (
  SELECT patient_guid, anchor_date
  FROM (
    SELECT *,
      ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(CAST(patient_guid AS STRING))) AS sample_rn
    FROM non_cancer_event_counts
  )
  CROSS JOIN params param
  WHERE sample_rn <= (param.non_cancer_ratio * (SELECT COUNT(*) FROM cancer_event_counts))
),

/* ═══════════════════════════════════════════════════════════════════════════
   Unified patient set: all cancer + randomly sampled non-cancer
   ═══════════════════════════════════════════════════════════════════════════ */

unified_patients AS (
  /* All qualifying cancer patients */
  SELECT patient_guid, anchor_date, cancer_id,
         1 AS cancer_class
  FROM cancer_event_counts

  UNION ALL

  /* Randomly sampled non-cancer patients (same count as cancer) */
  SELECT patient_guid, anchor_date, CAST(NULL AS STRING) AS cancer_id,
         0 AS cancer_class
  FROM non_cancer_sampled
),

/* ═══════════════════════════════════════════════════════════════════════════
   Step 6: Build medication data from prescribing tables
   ─────────────────────────────────────────────────────────────────────────
   Join drug records, issue records, and drug codes to get medication events.
   Same pipeline for both cancer and non-cancer patients.
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
   ─────────────────────────────────────────────────────────────────────────
   Each event type is capped independently (LIMIT) so one type cannot
   crowd out the other. Minimum event counts per patient are enforced
   using a window function on the actual deduplicated output rows.
   ═══════════════════════════════════════════════════════════════════════════ */

/* Raw observation events (before minimum-count enforcement) */
observation_events_raw AS (
  SELECT DISTINCT
    up.patient_guid,
    up.cancer_class,
    up.cancer_id,
    pp.sex,
    pp.patient_ethnicity_16,
    pp.patient_ethnicity_6,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS patient_age,
    --CAST(rc.ethnicity AS STRING)                              AS patient_ethnicity,
    mh.effective_date                                         AS event_date,
    DATE_DIFF(mh.effective_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS event_age,
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
  JOIN patients pp
    ON pp.patient_guid = up.patient_guid
  -- LEFT JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Ethnicity_Snowmed_Codes` rc
  --   ON CAST(mh.snomed_c_t_concept_id AS STRING) = rc.code
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
    CAST(NULL AS STRING)                                      AS patient_ethnicity_16,
    CAST(NULL AS STRING)                                      AS patient_ethnicity_6,
    DATE_DIFF(param.longterm_mh_end,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS patient_age,
    --CAST(NULL AS STRING)                                      AS patient_ethnicity,
    fm.effective_date                                         AS event_date,
    DATE_DIFF(fm.effective_date,
              PARSE_DATE('%Y-%m-%d', pp.date_of_birth), YEAR) AS event_age,
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
--select cancer_class, count(distinct patient_guid)
--FROM
--(
SELECT * FROM observation_events
--UNION ALL
--SELECT * FROM medication_events
--WHERE upper(patient_guid) like '%EEAA410B420E%'  --missing '%003EC38C-6E37-42D9-AA14-EEAA410B420E%'
--WHERE patient_guid like '%003F0495-664E-4805-BBA6-40305A14D210%'
ORDER BY patient_guid;
--)
--group by cancer_class

