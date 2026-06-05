
WITH params AS (
  SELECT
    'Lung' AS cancer_cohort,                                                -- ← 'Lung' | 'Breast' | 'Pancreatic'
    5    AS years_before,
    1    AS months_before,
    50   AS min_obs_events_per_patient,
    10   AS min_med_events_per_patient,
    1    AS non_cancer_ratio,
    DATE '2026-01-01' AS reference_date,
    ARRAY['Florida','Connecticut','Wisconsin','Oregon','Mississippi','Vermont'] AS states_in
),

cancer_type_config AS (
  SELECT
    cancer_cohort,
    CASE cancer_cohort
      WHEN 'Lung'       THEN r'^C34'
      WHEN 'Breast'     THEN r'^C50'
      WHEN 'Pancreatic' THEN r'^C25'
    END AS icd10_pattern,
    CASE cancer_cohort
      WHEN 'Lung'       THEN 'lung'
      WHEN 'Breast'     THEN 'breast'
      WHEN 'Pancreatic' THEN 'pancrea'                                       -- matches 'pancreas' + 'pancreatic'
    END AS snomed_keyword,
    CASE cancer_cohort
      WHEN 'Lung'       THEN 'lung'
      WHEN 'Breast'     THEN 'breast'
      WHEN 'Pancreatic' THEN 'pancreas'
    END AS cohort_output_label
  FROM params
),

eligible_patients AS (
  SELECT p.person_id AS patient_id
  FROM `prj-cts-ai-dev-sp.truveta_gold.patient` p
  CROSS JOIN cancer_type_config cfg
  CROSS JOIN params pa
  WHERE p.cohort IN (cfg.cancer_cohort, 'Non-Cancer')
    AND p.is_deleted = FALSE
    AND p.state_or_province IN UNNEST(pa.states_in)
),

cancer_concepts AS (
  SELECT c.concept_id, c.display AS term
  FROM `prj-cts-ai-dev-sp.truveta_gold.concept` c
  CROSS JOIN cancer_type_config cfg
  WHERE c.cohort = cfg.cancer_cohort                                         -- gold.concept is 4x-duplicated across cohorts; pick one
    AND c.is_deleted = FALSE
    AND c.domain = 'Condition'
    AND (
      (c.vocabulary = 'icd10cm' AND REGEXP_CONTAINS(c.code, cfg.icd10_pattern))
      OR (c.vocabulary = 'snomed'
          AND LOWER(c.display) LIKE CONCAT('%', cfg.snomed_keyword, '%')
          AND (LOWER(c.display) LIKE '%cancer%'
            OR LOWER(c.display) LIKE '%carcinoma%'
            OR LOWER(c.display) LIKE '%malignant neoplasm%'))
    )
),

any_cancer_concepts AS (
  SELECT concept_id
  FROM `prj-cts-ai-dev-sp.truveta_gold.concept`
  WHERE cohort = (SELECT cancer_cohort FROM params)
    AND is_deleted = FALSE
    AND (
      (vocabulary = 'icd10cm' AND (
        REGEXP_CONTAINS(code, r'^(C[0-9]|D0[0-9]|D3[7-9]|D4[0-8])')
        OR REGEXP_CONTAINS(code, r'^Z85')
        OR REGEXP_CONTAINS(code, r'^Z51\.[01]')
        OR REGEXP_CONTAINS(code, r'^Z08')
        OR REGEXP_CONTAINS(code, r'^Z80')
        OR REGEXP_CONTAINS(code, r'^Z90')
      ))
      OR (vocabulary = 'snomed' AND (
        LOWER(display) LIKE '%cancer%' OR
        LOWER(display) LIKE '%neoplas%' OR
        LOWER(display) LIKE '%carcinoma%' OR
        LOWER(display) LIKE '%lymphoma%' OR
        LOWER(display) LIKE '%leukaemia%' OR LOWER(display) LIKE '%leukemia%' OR
        LOWER(display) LIKE '%melanoma%' OR
        LOWER(display) LIKE '%sarcoma%' OR
        LOWER(display) LIKE '%tumor%' OR LOWER(display) LIKE '%tumour%' OR
        LOWER(display) LIKE '%metasta%' OR
        LOWER(display) LIKE '%antineoplastic%' OR
        LOWER(display) LIKE '%chemotherap%' OR
        LOWER(display) LIKE '%radiotherap%' OR
        LOWER(display) LIKE '%radiation therap%' OR
        LOWER(display) LIKE '%tnm stag%' OR LOWER(display) LIKE '%cancer stag%' OR
        LOWER(display) LIKE '%lobectomy%' OR
        LOWER(display) LIKE '%aromatase inhibitor%'
      ))
      OR (vocabulary = 'icd10pcs' AND (
        LOWER(display) LIKE '%neoplas%' OR
        LOWER(display) LIKE '%lymph%diagnost%' OR
        LOWER(display) LIKE '%lobectomy%'
      ))
      OR (vocabulary IN ('cpt','hcpcs') AND (
        LOWER(display) LIKE '%neoplas%' OR
        LOWER(display) LIKE '%therapeutic radiology%' OR
        LOWER(display) LIKE '%radiation dosimetry%' OR
        LOWER(display) LIKE '%radiation treatment%' OR
        LOWER(display) LIKE '%port image%' OR
        LOWER(display) LIKE '%fluorodeoxyglucose%' OR LOWER(display) LIKE '%fdg%' OR
        LOWER(display) LIKE '%positron emission%' OR
        LOWER(display) LIKE '%pet/ct%' OR LOWER(display) LIKE '%pet ct%' OR
        LOWER(display) LIKE '%palonosetron%' OR LOWER(display) LIKE '%ondansetron%' OR
        LOWER(display) LIKE '%doxorubicin%' OR LOWER(display) LIKE '%cisplatin%' OR
        LOWER(display) LIKE '%carboplatin%' OR LOWER(display) LIKE '%pembrolizumab%' OR
        LOWER(display) LIKE '%nivolumab%' OR LOWER(display) LIKE '%paclitaxel%' OR
        LOWER(display) LIKE '%bronchoscopy%biopsy%' OR
        LOWER(display) LIKE '%bronchoscopy%ebus%' OR
        LOWER(display) LIKE '%bronchoscopy%navigation%' OR
        LOWER(display) LIKE '%core needle biopsy%lung%' OR
        LOWER(display) LIKE '%core needle biopsy%mediastinum%' OR
        LOWER(display) LIKE '%medical physics consultation%'
      ))
      OR (vocabulary = 'loinc' AND (
        LOWER(display) LIKE '%ecog%' OR
        LOWER(display) LIKE '%karnofsky%' OR
        LOWER(display) LIKE '%tumor%' OR
        LOWER(display) LIKE '%cancer%' OR
        LOWER(display) LIKE '%neoplas%'
      ))
    )
),

medical_history AS (
  SELECT
    CASE ce.cohort
      WHEN cfg.cancer_cohort THEN cfg.cohort_output_label
      WHEN 'Non-Cancer'      THEN 'non_cancer'
    END                          AS cohort,
    ce.source_record_id,
    ce.person_id                 AS patient_id,
    ce.event_date,
    ce.recorded_date,
    ce.source_concept_id         AS concept_id,
    ce.source_concept_code       AS source_concept_id_code,
    ce.source_vocabulary,
    ce.term,
    ce.value_numeric,
    ce.value_text,
    ce.event_domain              AS domain
  FROM `prj-cts-ai-dev-sp.truveta_gold.clinical_event` ce
  CROSS JOIN cancer_type_config cfg
  WHERE ce.cohort IN (cfg.cancer_cohort, 'Non-Cancer')
    AND ce.event_date IS NOT NULL
    AND ce.is_deleted = FALSE
    AND ce.person_id IN (SELECT patient_id FROM eligible_patients)           -- FIX 1
),

demographics AS (
  SELECT
    CASE p.cohort
      WHEN cfg.cancer_cohort THEN cfg.cohort_output_label
      WHEN 'Non-Cancer'      THEN 'non_cancer'
    END                  AS cohort,
    p.person_id          AS patient_id,
    p.date_of_birth,
    p.sex,
    p.ethnicity,
    p.state_or_province,
    p.postal_code,
    p.deceased
  FROM `prj-cts-ai-dev-sp.truveta_gold.patient` p
  CROSS JOIN cancer_type_config cfg
  CROSS JOIN params pa
  WHERE p.cohort IN (cfg.cancer_cohort, 'Non-Cancer')
    AND p.is_deleted = FALSE
    AND p.state_or_province IN UNNEST(pa.states_in)
),

medication_history AS (
  SELECT
    CASE m.cohort
      WHEN cfg.cancer_cohort THEN cfg.cohort_output_label
      WHEN 'Non-Cancer'      THEN 'non_cancer'
    END                    AS cohort,
    m.medication_id        AS source_record_id,
    m.person_id            AS patient_id,
    m.event_date,
    m.event_date           AS start_date,
    m.end_date             AS stop_date,
    m.source_concept_id    AS med_concept_id,
    m.drug_term,
    m.duration_days
  FROM `prj-cts-ai-dev-sp.truveta_gold.medication` m
  CROSS JOIN cancer_type_config cfg
  WHERE m.cohort IN (cfg.cancer_cohort, 'Non-Cancer')
    AND m.event_date IS NOT NULL
    AND m.is_deleted = FALSE
    AND m.person_id IN (SELECT patient_id FROM eligible_patients)           -- FIX 1
),

target_cancer_patients AS (
  SELECT patient_id, date_of_diagnosis, cancer_term
  FROM (
    SELECT
      mh.patient_id,
      mh.event_date AS date_of_diagnosis,
      mh.term       AS cancer_term,
      ROW_NUMBER() OVER (PARTITION BY mh.patient_id ORDER BY mh.event_date, mh.term) AS rn
    FROM medical_history mh
    JOIN cancer_concepts cc ON mh.concept_id = cc.concept_id
    CROSS JOIN cancer_type_config cfg
    WHERE mh.cohort = cfg.cohort_output_label
      AND mh.domain = 'condition'
  )
  WHERE rn = 1
),

non_cancer_with_anchor AS (
  SELECT mh.patient_id, MAX(mh.event_date) AS anchor_date
  FROM medical_history mh
  WHERE mh.cohort = 'non_cancer'
    AND mh.event_date IS NOT NULL
  GROUP BY 1
),

cancer_event_counts AS (
  SELECT patient_id, anchor_date, cancer_term, COUNT(*) AS obs_event_count
  FROM (
    SELECT DISTINCT
      tcp.patient_id,
      tcp.date_of_diagnosis AS anchor_date,
      tcp.cancer_term,
      mh.event_date, mh.concept_id, mh.term, mh.value_numeric
    FROM target_cancer_patients tcp
    JOIN medical_history mh
      ON mh.patient_id = tcp.patient_id
    CROSS JOIN cancer_type_config cfg
    CROSS JOIN params p
    WHERE mh.cohort = cfg.cohort_output_label
      AND mh.event_date >= DATE_SUB(tcp.date_of_diagnosis, INTERVAL p.years_before  YEAR)
      AND mh.event_date <  DATE_SUB(tcp.date_of_diagnosis, INTERVAL p.months_before MONTH)
      AND NOT EXISTS (SELECT 1 FROM any_cancer_concepts acc WHERE acc.concept_id = mh.concept_id)  -- FIX 2
  )
  GROUP BY 1, 2, 3
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

non_cancer_event_counts AS (
  SELECT patient_id, anchor_date, COUNT(*) AS obs_event_count
  FROM (
    SELECT DISTINCT
      nca.patient_id, nca.anchor_date,
      mh.event_date, mh.concept_id, mh.term, mh.value_numeric
    FROM non_cancer_with_anchor nca
    JOIN medical_history mh
      ON mh.patient_id = nca.patient_id AND mh.cohort = 'non_cancer'
    CROSS JOIN params p
    WHERE mh.event_date >= DATE_SUB(nca.anchor_date, INTERVAL p.years_before  YEAR)
      AND mh.event_date <  DATE_SUB(nca.anchor_date, INTERVAL p.months_before MONTH)
      AND NOT EXISTS (SELECT 1 FROM any_cancer_concepts acc WHERE acc.concept_id = mh.concept_id)  -- FIX 2
  )
  GROUP BY 1, 2
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

non_cancer_sampled AS (
  SELECT patient_id, anchor_date
  FROM (
    SELECT *,
      ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(CAST(patient_id AS STRING))) AS sample_rn
    FROM non_cancer_event_counts
  )
  CROSS JOIN params p
  WHERE sample_rn <= p.non_cancer_ratio * (SELECT COUNT(*) FROM cancer_event_counts)
),

unified_patients AS (
  SELECT
    patient_id,
    anchor_date,
    cancer_term AS cancer_id,
    1 AS cancer_class,
    (SELECT cohort_output_label FROM cancer_type_config) AS cohort
  FROM cancer_event_counts
  UNION ALL
  SELECT patient_id, anchor_date, CAST(NULL AS STRING), 0, 'non_cancer'
  FROM non_cancer_sampled
),

observation_events_raw AS (
  SELECT DISTINCT
    up.patient_id, up.cancer_class, up.cancer_id, up.cohort,
    d.sex, d.ethnicity,
    d.state_or_province, d.postal_code,
    DATE_DIFF((SELECT reference_date FROM params), d.date_of_birth, YEAR) AS patient_age,
    mh.event_date,
    DATE_DIFF(mh.event_date, d.date_of_birth, YEAR) AS event_age,
    'observation' AS event_type,
    mh.source_record_id,
    mh.concept_id,
    mh.source_concept_id_code AS source_concept_id,
    mh.source_vocabulary,
    mh.term,
    mh.value_numeric,
    COALESCE(CAST(mh.value_numeric AS STRING), mh.value_text) AS value,
    CAST(NULL AS INT64)  AS value_concept_id,
    CAST(NULL AS INT64)  AS med_concept_id,
    CAST(NULL AS STRING) AS drug_term,
    CAST(NULL AS INT64)  AS duration
  FROM unified_patients up
  JOIN medical_history  mh ON mh.patient_id = up.patient_id AND mh.cohort = up.cohort
  JOIN demographics     d  ON d.patient_id  = up.patient_id AND d.cohort  = up.cohort
  CROSS JOIN params p
  WHERE mh.event_date >= DATE_SUB(up.anchor_date, INTERVAL p.years_before  YEAR)
    AND mh.event_date <  DATE_SUB(up.anchor_date, INTERVAL p.months_before MONTH)
    AND NOT EXISTS (SELECT 1 FROM any_cancer_concepts acc WHERE acc.concept_id = mh.concept_id)

),

observation_events AS (
  SELECT * EXCEPT(patient_obs_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_id) AS patient_obs_count
    FROM observation_events_raw
  )
  WHERE patient_obs_count >= (SELECT min_obs_events_per_patient FROM params)
),

medication_events_raw AS (
  SELECT DISTINCT
    up.patient_id, up.cancer_class, up.cancer_id, up.cohort,
    d.sex, d.ethnicity,
    d.state_or_province, d.postal_code,
    DATE_DIFF((SELECT reference_date FROM params), d.date_of_birth, YEAR) AS patient_age,
    mh.event_date,
    DATE_DIFF(mh.event_date, d.date_of_birth, YEAR) AS event_age,
    'medication' AS event_type,
    mh.source_record_id,
    CAST(NULL AS INT64)   AS concept_id,
    CAST(NULL AS STRING)  AS source_concept_id,
    CAST(NULL AS STRING)  AS source_vocabulary,
    CAST(NULL AS STRING)  AS term,
    CAST(NULL AS NUMERIC) AS value_numeric,
    CAST(NULL AS STRING)  AS value,
    CAST(NULL AS INT64)   AS value_concept_id,
    mh.med_concept_id, mh.drug_term,
    mh.duration_days AS duration
  FROM unified_patients up
  JOIN medication_history mh ON mh.patient_id = up.patient_id AND mh.cohort = up.cohort
  JOIN demographics       d  ON d.patient_id  = up.patient_id AND d.cohort  = up.cohort
  CROSS JOIN params p
  WHERE mh.event_date >= DATE_SUB(up.anchor_date, INTERVAL p.years_before  YEAR)
    AND mh.event_date <  DATE_SUB(up.anchor_date, INTERVAL p.months_before MONTH)
),

medication_events AS (
  SELECT * EXCEPT(patient_med_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_id) AS patient_med_count
    FROM medication_events_raw
  )
  WHERE patient_med_count >= (SELECT min_med_events_per_patient FROM params)
)

SELECT * FROM observation_events
UNION ALL
SELECT * FROM medication_events
ORDER BY patient_id
