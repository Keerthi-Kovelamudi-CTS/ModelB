/*╔══════════════════════════════════════════════════════════════════╗
  ║  BREAST (Truveta) — Phase 1.1                                  ║
  ║  Top Medications — NEGATIVE COHORT                             ║
  ║                                                                 ║
  ║  Source:  prj-cts-ai-dev-sp.truveta_gold               ║
  ║           .breast_cohort_events  (cohort = 'non_cancer')        ║
  ║                                                                 ║
  ║  Window:  [anchor - 15mo, anchor - 1mo)                         ║
  ║                                                                 ║
  ║  Mirror of the positive-cohort med query with cohort filter     ║
  ║  flipped. See positive med SQL header for output contract.      ║
  ╚══════════════════════════════════════════════════════════════════╝*/

WITH params AS (
  SELECT
    5  AS min_patient_threshold,
    15 AS months_lookback_start,
    1  AS months_lookback_end
),

cohort_events AS (
  SELECT
    patient_id,
    med_concept_id,
    drug_term,
    duration,
    event_date,
    event_age,
    event_age + DATE_DIFF(anchor_date, event_date, YEAR) AS diagnosis_age
  FROM `prj-cts-ai-dev-sp.truveta_gold.breast_cohort_events`
  CROSS JOIN params p
  WHERE cohort = 'non_cancer'
    AND event_type = 'medication'
    AND patient_age > 18
    AND med_concept_id IS NOT NULL
    AND event_date >= DATE_SUB(anchor_date, INTERVAL p.months_lookback_start MONTH)
    AND event_date <  DATE_SUB(anchor_date, INTERVAL p.months_lookback_end   MONTH)
),

total_negative_cohort AS (
  SELECT COUNT(DISTINCT patient_id) AS total_n
  FROM cohort_events
),

deduped_events AS (
  SELECT *
  FROM cohort_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_id, med_concept_id, event_date
    ORDER BY event_date
  ) = 1
)

SELECT
  med_concept_id                                                                AS med_code_id,
  MAX(drug_term)                                                                AS term,
  'rxnorm'                                                                      AS vocabulary,
  APPROX_QUANTILES(diagnosis_age, 2)[OFFSET(1)]                                 AS median_diagnosis_age,
  APPROX_QUANTILES(event_age, 2)[OFFSET(1)]                                     AS median_event_age,
  COUNT(DISTINCT patient_id)                                                    AS n_patient_count,
  (SELECT total_n FROM total_negative_cohort)                                   AS n_patient_count_total,
  ROUND(
    CAST(COUNT(DISTINCT patient_id) AS FLOAT64)
    / NULLIF((SELECT total_n FROM total_negative_cohort), 0),
    6
  )                                                                             AS prevalence,
  COUNT(*)                                                                      AS total_prescriptions,
  ROUND(CAST(COUNT(*) AS FLOAT64) / NULLIF(COUNT(DISTINCT patient_id), 0), 2)
                                                                                AS avg_prescriptions_per_patient,
  AVG(duration)                                                                 AS avg_duration,
  APPROX_QUANTILES(duration, 2)[OFFSET(1)]                                      AS median_duration,
  STDDEV(duration)                                                              AS std_duration
FROM deduped_events
GROUP BY med_concept_id
HAVING COUNT(DISTINCT patient_id) >= (SELECT min_patient_threshold FROM params)
ORDER BY n_patient_count DESC;
