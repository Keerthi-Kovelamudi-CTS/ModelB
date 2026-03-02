WITH params AS (
  SELECT
    DATE '1950-01-01'  AS mh_start,
    DATE '2026-02-25'  AS mh_end,
    13                 AS months_lookback_start,   -- start of feature window (months before dx)
    1                  AS months_lookback_end,     -- end of feature window (months before dx)
    5                  AS min_patient_threshold    -- minimum patients per SNOMED for output
),

/* ── Deduplicated SNOMED lookup: latest file per code+practice ── */
diagnostic_codes AS (
  SELECT
    TRY_TO_NUMBER(RAW_RECORDS:code_id::string)          AS code_id,
    PRACTICE_ID                                         AS source_practice_code,
    RAW_RECORDS:term::string                            AS term,
    TRY_TO_NUMBER(RAW_RECORDS:snomed_c_t_concept_id::string) AS snomed_c_t_concept_id,
    RAW_RECORDS:snomed_c_t_description_id::string       AS snomed_c_t_description_id
  FROM CTSUK_BULK.RAW.CODING_CLINICALCODE
  WHERE RAW_RECORDS:snomed_c_t_concept_id IS NOT NULL
    AND RAW_RECORDS:term IS NOT NULL
    AND TRY_TO_NUMBER(RAW_RECORDS:code_id::string) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRY_TO_NUMBER(RAW_RECORDS:code_id::string), PRACTICE_ID
    ORDER BY TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    ) DESC
  ) = 1
),

patients AS (
  SELECT
    patient_guid,
    sex,
    date_of_birth
  FROM CTSUK_BULK.STAGING.PATIENTS_EMIS
  WHERE patient_guid IS NOT NULL
    AND date_of_birth IS NOT NULL
),

/* ── All clinical observations with valid SNOMED mapping ── */
medical_history AS (
  SELECT
    co.practice_id,
    co.RAW_RECORDS:patient_guid::string                          AS patient_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date::string)           AS effective_date,
    co.RAW_RECORDS:observation_type::string                      AS observation_type,
    co.RAW_RECORDS:value::string                                 AS value,
    co.RAW_RECORDS:associated_text::string                       AS associated_text,
    dc.term,
    dc.snomed_c_t_concept_id
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  INNER JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = dc.code_id
   AND co.practice_id = dc.source_practice_code
  CROSS JOIN params p
  WHERE TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) IS NOT NULL
    AND TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) >= p.mh_start
    AND TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) <  p.mh_end
    AND dc.snomed_c_t_concept_id IS NOT NULL
),

/* ── Bladder cancer patients: first diagnosis date ── */
cancer_patients AS (
  SELECT
    mh.patient_guid,
    dcc.cancer_id,
    MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history AS mh
  INNER JOIN analytics.cts.dim_cancer_codes AS dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    AND LOWER(dcc.cancer_id) LIKE '%bladder%'
  GROUP BY 1, 2
  HAVING MIN(mh.effective_date) IS NOT NULL
),

/* ── Excluded SNOMED codes: administrative, noise, ethnicity ── */
excluded_snomeds AS (
  SELECT code AS snomed_code FROM EXPERIMENTS.MARTA_S.ETHNICITY_SNOMED_CODES
  UNION ALL
  SELECT column1 FROM (VALUES
    (1572871000006101), (279991000000102), (428481002),
    (979851000000101),  (887641000000105), (1958701000006105),
    (185317003),        (788007007),       (283511000000105),
    (182836005),        (182888003),       (184103008),
    (498521000006103),  -- Attachment for medical notes
    (386472008)         -- Telephone consultations
  )
),

/* ── Feature events: observations in [-13m, -1m] before diagnosis ── */
feature_events AS (
  SELECT
    cp.patient_guid,
    pp.sex,
    DATEDIFF(year, pp.date_of_birth, cp.date_of_diagnosis)    AS diagnosis_age,
    cp.cancer_id,
    cp.date_of_diagnosis,
    mh.effective_date                                          AS event_date,
    DATEDIFF(year, pp.date_of_birth, mh.effective_date)        AS event_age,
    mh.snomed_c_t_concept_id,
    mh.term,
    mh.observation_type,
    mh.associated_text,
    mh.value
  FROM cancer_patients cp
  INNER JOIN medical_history mh
    ON mh.patient_guid = cp.patient_guid
  INNER JOIN patients pp
    ON pp.patient_guid = cp.patient_guid
  CROSS JOIN params p
  /* ── Time window: [-13m, -1m] before diagnosis ── */
  WHERE mh.effective_date >= DATEADD(month, -p.months_lookback_start, cp.date_of_diagnosis)
    AND mh.effective_date <  DATEADD(month, -p.months_lookback_end,   cp.date_of_diagnosis)
    /* ── Exclude cancer diagnosis codes (data leakage prevention) ── */
    AND NOT EXISTS (
      SELECT 1 FROM analytics.cts.dim_cancer_codes dcc
      WHERE dcc.snomed_code = mh.snomed_c_t_concept_id
    )
    /* ── Exclude administrative / noise / ethnicity codes ── */
    AND NOT EXISTS (
      SELECT 1 FROM excluded_snomeds ex
      WHERE ex.snomed_code = mh.snomed_c_t_concept_id
    )
    /* ── Exclude non-clinical observation types ── */
    AND COALESCE(mh.observation_type, '') NOT IN ('Immunisation')
),

/* ── Deduplicate: one row per patient × SNOMED × date ── */
deduped_events AS (
  SELECT *
  FROM feature_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_guid, snomed_c_t_concept_id, event_date
    ORDER BY event_date
  ) = 1
)

/* ════════════════════════════════════════════════════════════════
   FINAL OUTPUT: Aggregated SNOMED ranking
   ════════════════════════════════════════════════════════════════ */
SELECT
  term,
  MAX(snomed_c_t_concept_id)                                   AS snomed_id,
  MEDIAN(diagnosis_age)                                        AS median_diagnosis_age,
  MEDIAN(event_age)                                            AS median_snomed_age,
  COUNT(DISTINCT patient_guid)                                 AS n_patient_count,
  (SELECT COUNT(DISTINCT patient_guid) FROM deduped_events)    AS n_patient_count_total,
  ROUND(
    COUNT(DISTINCT patient_guid)::FLOAT
    / NULLIF((SELECT COUNT(DISTINCT patient_guid) FROM deduped_events), 0),
    6
  )                                                            AS prevalence,
  COUNT(*)                                                     AS total_event_count,
  ROUND(COUNT(*)::FLOAT / NULLIF(COUNT(DISTINCT patient_guid), 0), 2)
                                                               AS avg_events_per_patient,
  AVG(CAST(IFF(TRIM(value) = '' OR value IS NULL, NULL, value) AS REAL))
                                                               AS avg_value,
  MEDIAN(CAST(IFF(TRIM(value) = '' OR value IS NULL, NULL, value) AS REAL))
                                                               AS median_value,
  STDDEV(CAST(IFF(TRIM(value) = '' OR value IS NULL, NULL, value) AS REAL))
                                                               AS std_value,
  MODE(CAST(IFF(TRIM(value) = '' OR value IS NULL, NULL, value) AS REAL))
                                                               AS freq_value
FROM deduped_events
GROUP BY term
HAVING COUNT(DISTINCT patient_guid) >= (SELECT min_patient_threshold FROM params)
ORDER BY n_patient_count DESC;
