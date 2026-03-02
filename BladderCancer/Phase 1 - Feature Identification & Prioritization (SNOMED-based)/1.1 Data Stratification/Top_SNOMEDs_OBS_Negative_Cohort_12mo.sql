WITH params AS (
  SELECT
    DATE '1950-01-01'  AS mh_start,
    DATE '2026-02-25'  AS mh_end,
    13                 AS months_lookback_start,
    1                  AS months_lookback_end,
    70                 AS pseudo_index_age,         -- median bladder cancer dx age
    5                  AS min_patient_threshold
),

diagnostic_codes AS (
  SELECT
    TRY_TO_NUMBER(RAW_RECORDS:code_id::string)                       AS code_id,
    PRACTICE_ID                                                      AS source_practice_code,
    RAW_RECORDS:term::string                                         AS term,
    TRY_TO_NUMBER(RAW_RECORDS:snomed_c_t_concept_id::string)         AS snomed_c_t_concept_id,
    RAW_RECORDS:snomed_c_t_description_id::string                    AS snomed_c_t_description_id
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
  SELECT patient_guid, sex, date_of_birth
  FROM CTSUK_BULK.STAGING.PATIENTS_EMIS
  WHERE patient_guid IS NOT NULL
    AND date_of_birth IS NOT NULL
),

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

/* ── ALL cancer patients (any type — to exclude them) ── */
cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history AS mh
  INNER JOIN analytics.cts.dim_cancer_codes AS dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
),

/* ── Excluded SNOMED codes ── */
excluded_snomeds AS (
  SELECT code AS snomed_code FROM EXPERIMENTS.MARTA_S.ETHNICITY_SNOMED_CODES
  UNION ALL
  SELECT column1 FROM (VALUES
    (1572871000006101), (279991000000102), (428481002),
    (979851000000101),  (887641000000105), (1958701000006105),
    (185317003),        (788007007),       (283511000000105),
    (182836005),        (182888003),       (184103008),
    (498521000006103),  (386472008)
  )
),

/* ── Non-cancer adults: no cancer ever, no palliative care ── */
non_cancer_patients AS (
  SELECT DISTINCT
    p.patient_guid,
    p.date_of_birth
  FROM patients p
  INNER JOIN medical_history mh
    ON mh.patient_guid = p.patient_guid
  CROSS JOIN params param
  WHERE
    /* Adult check */
    DATEDIFF(year, p.date_of_birth, param.mh_end) > 18
    /* No cancer ever */
    AND NOT EXISTS (
      SELECT 1 FROM cancer_patients cp
      WHERE cp.patient_guid = p.patient_guid
    )
    /* No palliative care */
    AND NOT EXISTS (
      SELECT 1 FROM medical_history pmh
      WHERE pmh.patient_guid = p.patient_guid
        AND pmh.snomed_c_t_concept_id = 1403151000000103
    )
    /* Must have reached pseudo index age within data window */
    AND DATEADD(year, param.pseudo_index_age, p.date_of_birth) <= param.mh_end
    AND DATEADD(year, param.pseudo_index_age, p.date_of_birth) >= param.mh_start
),

/* ── Feature events: observations in [-13m, -1m] before pseudo index age ── */
feature_events AS (
  SELECT
    ncp.patient_guid,
    pp.sex,
    DATEDIFF(year, ncp.date_of_birth, mh.effective_date)       AS event_age,
    mh.effective_date                                          AS event_date,
    mh.snomed_c_t_concept_id,
    mh.term,
    mh.observation_type,
    mh.associated_text,
    mh.value
  FROM non_cancer_patients ncp
  INNER JOIN medical_history mh
    ON mh.patient_guid = ncp.patient_guid
  INNER JOIN patients pp
    ON pp.patient_guid = ncp.patient_guid
  CROSS JOIN params p
  WHERE
    mh.effective_date >= DATEADD(month, -p.months_lookback_start,
                           DATEADD(year, p.pseudo_index_age, ncp.date_of_birth))
    AND mh.effective_date < DATEADD(month, -p.months_lookback_end,
                              DATEADD(year, p.pseudo_index_age, ncp.date_of_birth))
    /* Exclude cancer diagnosis codes */
    AND NOT EXISTS (
      SELECT 1 FROM analytics.cts.dim_cancer_codes dcc
      WHERE dcc.snomed_code = mh.snomed_c_t_concept_id
    )
    /* Exclude administrative / noise / ethnicity codes */
    AND NOT EXISTS (
      SELECT 1 FROM excluded_snomeds ex
      WHERE ex.snomed_code = mh.snomed_c_t_concept_id
    )
    AND COALESCE(mh.observation_type, '') NOT IN ('Immunisation')
),

deduped_events AS (
  SELECT *
  FROM feature_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_guid, snomed_c_t_concept_id, event_date
    ORDER BY event_date
  ) = 1
)

/* ════════════════════════════════════════════════════════════════ */
SELECT
  term,
  MAX(snomed_c_t_concept_id)                                   AS snomed_id,
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
