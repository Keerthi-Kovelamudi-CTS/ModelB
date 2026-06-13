"""
Build the Phase-1 scoring inputs from the SAME cohort the lung model trains on.

Horizons: builds BOTH `12mo` and `1mo` in a single run. Each horizon reads
SQL/{horizon}_1to1.sql and writes to its own Data/{horizon}/.

INPUTS
------
  ./SQL/{horizon}_1to1.sql   - the model's training cohort (run on BigQuery). It reads:
        * EMIS_BULK_DATA_PROCESSED.CareRecord_Observation / CareRecord_Problem  (observations)
        * EMIS_BULK_DATA_PROCESSED.Prescribing_DrugRecord / Prescribing_IssueRecord
          / Coding_DrugCode                                                     (medications)
        * EMIS_BULK_DATA_temp.lung_cancer_500 / no_cancer_lung_50000            (held-out,
          EXCLUDED so scoring never overlaps the real-world evaluation set)
      No local input files are read - everything comes from the SQL above via BigQuery.

OUTPUTS (written to ./Data/{horizon}/, the dir the scorer reads)
---------------------------------------------------------------
  Count CSVs (statistical + prevalence scoring):
      Lung_positive_obs.csv,  Lung_negative_obs.csv
      Lung_positive_meds.csv, Lung_negative_meds.csv
        columns: snomed_id|med_code_id, term, n_patient_count, n_patient_count_total
  Patient matrices (GENUINE ML feature-importance - real per-patient co-occurrence):
      Lung_patient_obs_matrix.csv,  Lung_patient_meds_matrix.csv
        columns: patient_guid, feature_name, event_count, label   (long format)

This uses the model's own {horizon}_1to1.sql cohort, so the Phase-1 code rankings and the
trained model share ONE identical cohort.

Run (on a VM with BigQuery access, e.g. lungenv) - both horizons in one go:
    python build_score_counts.py             # builds 12mo + 1mo
    python "Combined Scoring (ML+Stat).py"   # scores 12mo + 1mo
"""
import os
import re
from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZONS = ["12mo", "1mo"]      # always build BOTH horizons in one run

# (cancer_class, event_type, output filename, code column, term column, output code-column name)
COUNT_SPEC = [
    (1, "observation", "Lung_positive_obs.csv",  "snomed_c_t_concept_id", "term",      "snomed_id"),
    (0, "observation", "Lung_negative_obs.csv",  "snomed_c_t_concept_id", "term",      "snomed_id"),
    (1, "medication",  "Lung_positive_meds.csv", "med_code_id",           "drug_term", "med_code_id"),
    (0, "medication",  "Lung_negative_meds.csv", "med_code_id",           "drug_term", "med_code_id"),
]
# (event_type, code-for-feature-name column, output matrix filename)
MATRIX_SPEC = [
    ("observation", "term",      "Lung_patient_obs_matrix.csv"),
    ("medication",  "drug_term", "Lung_patient_meds_matrix.csv"),
]


def load_events(horizon):
    """Run the {horizon}_1to1 cohort SQL and pull only the columns needed for counting + matrices."""
    cohort_sql = os.path.join(HERE, "SQL", f"{horizon}_1to1.sql")
    sql = open(cohort_sql, encoding="utf-8").read().rstrip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    sql = re.sub(r"ORDER\s+BY\s+patient_guid\s*$", "", sql, flags=re.I).rstrip()  # pointless inside a subquery
    query = f"""
        SELECT patient_guid, cancer_class, event_type,
               snomed_c_t_concept_id, term, med_code_id, drug_term
        FROM (
{sql}
        )
    """
    print(f"Running {horizon}_1to1 cohort on BigQuery ...")
    ev = bigquery.Client().query(query).to_dataframe()
    ev["cancer_class"] = ev["cancer_class"].astype(int)
    print(f"  pulled {len(ev):,} events | {ev.patient_guid.nunique():,} patients "
          f"({ev[ev.cancer_class == 1].patient_guid.nunique():,} cancer / "
          f"{ev[ev.cancer_class == 0].patient_guid.nunique():,} non-cancer)")
    return ev


def code_counts(ev, cancer_class, event_type, code_col, term_col, out_code):
    """Distinct-patient count per code for one (class, type) slice -> scorer count schema."""
    sub = ev[(ev.cancer_class == cancer_class) & (ev.event_type == event_type)]
    sub = sub[sub[code_col].notna()]
    total = sub["patient_guid"].nunique()                      # denominator (the cohort size)
    g = (sub.groupby(code_col)
            .agg(n_patient_count=("patient_guid", "nunique"),
                 term=(term_col, lambda s: s.value_counts().index[0] if len(s) else ""))
            .reset_index()
            .rename(columns={code_col: out_code}))
    g["n_patient_count_total"] = total
    g[out_code] = g[out_code].astype("int64")
    return g[[out_code, "term", "n_patient_count", "n_patient_count_total"]] \
            .sort_values("n_patient_count", ascending=False)


def patient_matrix(ev, event_type, name_col):
    """Long-format real patient x feature matrix for genuine ML:
       one row per (patient, feature) with the event count + class label."""
    sub = ev[(ev.event_type == event_type) & ev[name_col].notna()]
    mat = (sub.groupby(["patient_guid", name_col]).size()
              .reset_index(name="event_count")
              .rename(columns={name_col: "feature_name"}))
    labels = ev[["patient_guid", "cancer_class"]].drop_duplicates("patient_guid") \
               .rename(columns={"cancer_class": "label"})
    return mat.merge(labels, on="patient_guid", how="left")[
        ["patient_guid", "feature_name", "event_count", "label"]]


def main():
    for horizon in HORIZONS:
        data_dir = os.path.join(HERE, "Data", horizon)
        os.makedirs(data_dir, exist_ok=True)
        print(f"\n===================  {horizon}  ===================")
        ev = load_events(horizon)

        print("Count CSVs:")
        for cls, ftype, fname, code_col, term_col, out_code in COUNT_SPEC:
            df = code_counts(ev, cls, ftype, code_col, term_col, out_code)
            df.to_csv(os.path.join(data_dir, fname), index=False)
            print(f"  wrote {fname:26} {len(df):>5} codes | cohort size = {df['n_patient_count_total'].iat[0]:,}")

        print("Patient matrices (genuine-ML inputs):")
        for ftype, name_col, fname in MATRIX_SPEC:
            mat = patient_matrix(ev, ftype, name_col)
            mat.to_csv(os.path.join(data_dir, fname), index=False)
            print(f"  wrote {fname:26} {len(mat):>8,} rows | {mat.patient_guid.nunique():,} patients")

    print(f'\nAll horizons done ({", ".join(HORIZONS)}). Now run:  python "Combined Scoring (ML+Stat).py"')


if __name__ == "__main__":
    main()
