"""
Build the FE codelist (Phase-1 scoring -> 2_FE) for BOTH horizons — curated + data-driven.

Pipeline:  build_score_counts.py -> "Combined Scoring (ML+Stat).py" -> build_codelist.py -> 2_FE

Built PER (horizon, lookback window) — each lookback re-discovers its codes on its OWN scores.
The codelist = a focused top-N data-driven set, optionally UNION the curated clinical-concept codes:
  * data-driven: codes passing 3 gates — Bonferroni p<0.05  AND  OR >= OR_MIN  AND  prevalence >= PREV_MIN
                 of cancer patients — then ranked by combined_rank and cut to the top TOP_N.
  * curated   : curated clinical-concept codes (curated_codes.json), force-included even if sub-threshold
                — only when USE_CURATED is on (else the codelist is the data-driven top-N alone).

Per (horizon, lookback) it:
  1. reads the combined ranked scores  (output/{h}/{years}yr/Scores_lung/lung_combined_all.csv)
  2. builds the codelist (top-N eligible [U curated])
  3. one lean BigQuery aggregate marks each code value-bearing (>= MIN_VALUE_FRAC numeric)
  4. writes ../2_FE/codelist/lung_codelist_{h}_{years}yr.csv  (columns: Code, Name, Value)

Run (VM/env with BigQuery, after Combined Scoring):  python build_codelist.py [12mo|1mo] [years]
Tunable via env: TOP_N (default 500), OR_MIN (2.0), PREV_MIN (0.01), USE_CURATED (1).
"""
import os
import re
import sys
import json
import pandas as pd
from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # V3 root, for `import config`
import config as C

HORIZONS = list(C.HORIZONS)
if len(sys.argv) > 1:          # e.g. `python build_codelist.py 12mo 5`  (horizon [years lookback])
    HORIZONS = [a for a in sys.argv[1:] if a in ("12mo", "1mo")] or HORIZONS
# FE/scoring lookback in years: reads this lookback's scores and writes lung_codelist_{h}_{YEARS}yr.csv
YEARS = next((int(a) for a in sys.argv[1:] if a.isdigit()), C.SCORING_YEARS_BEFORE)
MIN_VALUE_FRAC = C.MIN_VALUE_FRAC   # value-bearing threshold (KEEP IN SYNC with 2_FE/build_features)
TOP_N = C.TOP_N                     # # of top data-driven codes by combined_rank (eligible pool)
OR_MIN = C.OR_MIN                   # eligibility: risk-increasing odds ratio floor
PREV_MIN = C.PREV_MIN               # eligibility: >= this fraction of cancer patients


def _curated_codes():
    """Curated clinical-concept codes (symptoms/comorbidities/smoking/meds/labs), from curated_codes.json.
    Returns the empty set when USE_CURATED is off (codelist becomes the data-driven top-N only)."""
    if not getattr(C, "USE_CURATED", True):
        return set()
    p = os.path.join(HERE, "curated_codes.json")
    return set(json.load(open(p))) if os.path.exists(p) else set()


SCORES = os.path.join(HERE, "output", "{h}", f"{YEARS}yr", "Scores_lung", "lung_combined_all.csv")
OUT_DIR = os.path.join(HERE, "..", "2_FE", "codelist")


def value_bearing_flags(horizon):
    """Per-code fraction of events that carry a numeric value, via one BigQuery aggregate over the
    {horizon}_1to1 cohort SQL. Returns {code:int -> is_value_bearing:0/1}. The unified `code` is the
    SNOMED id for observations and the med_code_id for medications (matches 2_FE/build_features)."""
    cohort_sql = os.path.join(HERE, "..", "0_SQL", f"{horizon}_1to1.sql")   # shared single-source SQL
    sql = open(cohort_sql, encoding="utf-8").read().rstrip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    sql = re.sub(r"ORDER\s+BY\s+patient_guid\s*$", "", sql, flags=re.I).rstrip()
    sql = re.sub(r"\d+(\s+AS years_before\b)", rf"{YEARS}\1", sql, count=1)   # value-bearing at the SAME lookback as scoring
    query = f"""
        SELECT code,
               COUNTIF(SAFE_CAST(value AS FLOAT64) IS NOT NULL) AS n_num,
               COUNT(*)                                         AS n_tot
        FROM (
          SELECT CAST(CASE WHEN event_type = 'observation'
                      THEN snomed_c_t_concept_id ELSE med_code_id END AS STRING) AS code,
                 value
          FROM (
{sql}
          )
        )
        WHERE code IS NOT NULL
        GROUP BY code
    """
    print(f"  [{horizon}] BigQuery: per-code numeric-value fraction ...")
    df = bigquery.Client().query(query).to_dataframe()
    df["frac"] = df["n_num"] / df["n_tot"].replace(0, pd.NA)
    df["Code"] = pd.to_numeric(df["code"], errors="coerce")
    df = df.dropna(subset=["Code"])
    df["Code"] = df["Code"].astype("int64")
    df["is_vb"] = (df["frac"] >= MIN_VALUE_FRAC).astype(int)
    print(f"  [{horizon}] value-bearing codes: {int(df['is_vb'].sum()):,} / {len(df):,}")
    return dict(zip(df["Code"], df["is_vb"]))


def build(horizon):
    d = pd.read_csv(SCORES.format(h=horizon))
    d["Code"] = pd.to_numeric(d["code_id"], errors="coerce").fillna(
                pd.to_numeric(d.get("med_code_id"), errors="coerce"))
    d = d[d["Code"].notna()].copy(); d["Code"] = d["Code"].astype("int64")

    cur = _curated_codes()
    prev_pos = pd.to_numeric(d.get("prevalence_pos"), errors="coerce")
    if prev_pos.isna().all():                       # fallback if no prevalence_pos column
        prev_pos = pd.to_numeric(d.get("n_patient_count_pos"), errors="coerce") / \
                   pd.to_numeric(d.get("n_patient_count_total_pos"), errors="coerce")
    elig = d[(pd.to_numeric(d["p_value_bonferroni"], errors="coerce") < 0.05)
             & (pd.to_numeric(d["odds_ratio"], errors="coerce") >= OR_MIN)
             & (prev_pos >= PREV_MIN)]
    top = set(elig.nsmallest(TOP_N, "combined_rank")["Code"])   # combined_rank 1 = strongest
    cur_present = set(d["Code"]) & cur
    sig = d[d["Code"].isin(top | cur_present)].copy()
    print(f"  [{horizon}] eligible(Bonf<0.05, OR>={OR_MIN}, prev>={PREV_MIN:.0%})={len(elig):,} "
          f"-> top-{TOP_N} ({len(top)}) U curated-present ({len(cur_present)}/{len(cur)}) "
          f"-> {sig['Code'].nunique():,} codes")

    vb = value_bearing_flags(horizon)            # code -> 0/1
    out = pd.DataFrame({
        "Code":  sig["Code"],
        "Name":  sig["term"].fillna(""),
        "Value": sig["Code"].map(vb).fillna(0).astype(int),       # 1 = value-bearing, 0 = code-only
        "_rank": sig["combined_rank"],
    }).drop_duplicates("Code").sort_values("_rank").drop(columns="_rank")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"lung_codelist_{horizon}_{YEARS}yr.csv")
    out.to_csv(path, index=False)
    print(f"  [{horizon}] wrote {len(out):,} codes ({int(out.Value.sum())} value-bearing) -> {path}")


def main():
    for h in HORIZONS:
        print(f"\n===================  {h}  ===================")
        build(h)
    print("\nDone. FE reads 2_FE/codelist/lung_codelist_{12mo,1mo}.csv (Code column).")


if __name__ == "__main__":
    main()
