"""
Preprocess raw SQL output → unified FE-ready CSV.

Reads:  data/raw/lymphoma_{window}_raw.csv   (output of SQL Queries/v4/{window}_v4.sql)
Writes: data/{window}/lymphoma_{window}.parquet  (single unified file — obs + med UNION'd)

(The FE pipeline's 1_sanity_check.py reads this unified file, splits internally,
 and produces *_obs_dropped.csv / *_med_dropped.csv from it.)

What it does:
  1. Renames SQL columns to FE-expected names (ANCHOR_DATE → INDEX_DATE, etc.)
  2. Derives CATEGORY by mapping SNOMED/MED codes via code_category_mapping.json
     (codes not in mapping → CATEGORY = NULL → patient ends up with sparse/zero features)
  3. Derives TIME_WINDOW (A/B) by binning MONTHS_BEFORE_INDEX
  4. Adds a __PLACEHOLDER__ row per patient so zero-categorized-event patients
     still appear in the FE feature matrix (Approach B requirement)
  5. Writes a single unified CSV (FE pipeline splits obs/med internally)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import read_table, write_table

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW = SCRIPT_DIR / "data" / "raw" / "lymphoma_1mo_raw.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "data" / "1mo"
DEFAULT_MAPPING = SCRIPT_DIR.parent / "codelist2.0" / "code_category_mapping_2.0.json"

# A/B time-window boundary. Events ≤ TIME_WINDOW_MID = window B (recent).
# Events > TIME_WINDOW_MID and ≤ years_before*12 = window A (earlier).
# v3 horizon-specific A/B midpoint (months before anchor) = half of each window's
# lookback, so window A (earlier) and B (recent) are balanced:
#   1mo horizon  → 12-month lookback → mid 6    (B = 1-6mo,   A = 6-12mo)
#   12mo horizon → 20-year lookback  → mid 126  (B = 12-126mo, A = 126-240mo)
TIME_WINDOW_MID_BY_WINDOW = {"1mo": 6, "12mo": 126}
TIME_WINDOW_MID_MONTHS = 30   # default; main() overrides per --prefix window (v3)

PLACEHOLDER_CATEGORY = "__PLACEHOLDER__"

# FE-expected schema (matches v1 1_preprocess_holdout.py OUT_COLS)
OUT_COLS = [
    "PATIENT_GUID", "SEX", "PATIENT_ETHNICITY_16", "PATIENT_ETHNICITY_6",
    "INDEX_DATE", "EVENT_DATE",
    "AGE_AT_INDEX", "EVENT_AGE", "MONTHS_BEFORE_INDEX", "TIME_WINDOW",
    "EVENT_TYPE", "CATEGORY", "CODE_ID", "TERM", "ASSOCIATED_TEXT",
    "VALUE", "LABEL",
]


def load_mapping(path):
    """Load SNOMED/MED → CATEGORY mapping."""
    with open(path) as f:
        m = json.load(f)
    obs_map = {str(k).strip(): v for k, v in m.get("obs", {}).items()}
    med_map = {str(k).strip(): v for k, v in m.get("med", {}).items()}
    return obs_map, med_map


def assign_time_window(months):
    """1-30 mo → B (recent); 30+ mo → A (earlier); else NULL."""
    out = np.full(len(months), None, dtype=object)
    out[(months >= 1) & (months <= TIME_WINDOW_MID_MONTHS)] = "B"
    out[months > TIME_WINDOW_MID_MONTHS] = "A"
    return out


def assign_category(code_series, mapping):
    """Lookup code in mapping; return CATEGORY or None."""
    str_codes = code_series.astype("Int64").astype(str).replace("<NA>", "")
    return str_codes.map(mapping).where(str_codes != "", None)


def build_placeholder_rows(unified_patients, sex_lookup, age_lookup,
                            ethnicity_16_lookup=None, ethnicity_6_lookup=None):
    """One row per patient with CATEGORY=__PLACEHOLDER__.

    ethnicity_16_lookup / ethnicity_6_lookup may be None (older callers) → ethnicity
    columns will be None for placeholder rows.
    Ensures every unified_patient appears in the FE feature matrix even if
    they have no curated codes (Approach B: keep sparse-record patients).

    EVENT_DATE is set to (INDEX_DATE - TIME_WINDOW_MID_MONTHS) so it sits
    safely inside the lookback window — otherwise the FE clean step's leak
    guard (`EVENT_DATE >= INDEX_DATE`) would drop the placeholder rows
    and the sparse patients would vanish from training."""
    rows = []
    placeholder_offset = pd.DateOffset(months=TIME_WINDOW_MID_MONTHS)
    for pid, anchor_date, label in unified_patients.itertuples(index=False):
        anchor_ts = pd.Timestamp(anchor_date)
        event_ts = anchor_ts - placeholder_offset
        rows.append({
            "PATIENT_GUID": pid,
            "SEX": sex_lookup.get(pid),
            "PATIENT_ETHNICITY_16": ethnicity_16_lookup.get(pid, None),
            "PATIENT_ETHNICITY_6":  ethnicity_6_lookup.get(pid, None),
            "INDEX_DATE": anchor_ts.strftime("%Y-%m-%d"),
            "EVENT_DATE": event_ts.strftime("%Y-%m-%d"),
            "AGE_AT_INDEX": age_lookup.get(pid, 0),
            "EVENT_AGE": age_lookup.get(pid, 0),
            "MONTHS_BEFORE_INDEX": TIME_WINDOW_MID_MONTHS,
            "TIME_WINDOW": "A",
            "EVENT_TYPE": "observation",
            "CATEGORY": PLACEHOLDER_CATEGORY,
            "CODE_ID": -1,
            "TERM": "__placeholder__",
            "ASSOCIATED_TEXT": "",
            "VALUE": "",
            "LABEL": label,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",     default=DEFAULT_RAW, type=Path,    help="raw SQL output CSV")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path, help="dir to write *_obs_dropped.csv / *_med_dropped.csv")
    ap.add_argument("--mapping", default=DEFAULT_MAPPING, type=Path, help="code_category_mapping.json")
    ap.add_argument("--prefix",  default="lymphoma_1mo",            help="output filename prefix")
    args = ap.parse_args()

    if not args.raw.exists():
        sys.exit(f"ERROR: raw file not found: {args.raw}")
    if not args.mapping.exists():
        sys.exit(f"ERROR: mapping file not found: {args.mapping}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # v3: resolve the A/B midpoint for THIS window from the --prefix suffix.
    global TIME_WINDOW_MID_MONTHS
    _window = args.prefix.rsplit("_", 1)[-1]
    TIME_WINDOW_MID_MONTHS = TIME_WINDOW_MID_BY_WINDOW.get(_window, TIME_WINDOW_MID_MONTHS)
    print(f"  A/B midpoint for window '{_window}': TIME_WINDOW_MID_MONTHS={TIME_WINDOW_MID_MONTHS}")

    print(f"Loading raw: {args.raw}  ({args.raw.stat().st_size/1e9:.2f} GB)")
    df = read_table(args.raw, low_memory=False)
    print(f"  rows: {len(df):,}  cols: {df.shape[1]}")

    # ─── Defensive: derive anchor metadata if SQL did not emit it ──────────
    # Future-proofs the pipeline if upstream switches to the shared lung SQL
    # (which currently lacks anchor_date / months_before_anchor / days_before_anchor).
    # For new-patient inference MAX(EVENT_DATE) is a valid anchor proxy; for
    # training data the SQL is expected to provide ANCHOR_DATE explicitly.
    if "ANCHOR_DATE" not in df.columns:
        print("  WARNING: ANCHOR_DATE missing — deriving from MAX(EVENT_DATE) per patient")
        df["EVENT_DATE"] = pd.to_datetime(df["EVENT_DATE"], errors="coerce")
        df["ANCHOR_DATE"] = df.groupby("PATIENT_GUID")["EVENT_DATE"].transform("max")
    if "MONTHS_BEFORE_ANCHOR" not in df.columns:
        print("  WARNING: MONTHS_BEFORE_ANCHOR missing — deriving from (ANCHOR_DATE - EVENT_DATE)")
        df["EVENT_DATE"]  = pd.to_datetime(df["EVENT_DATE"],  errors="coerce")
        df["ANCHOR_DATE"] = pd.to_datetime(df["ANCHOR_DATE"], errors="coerce")
        df["MONTHS_BEFORE_ANCHOR"] = ((df["ANCHOR_DATE"] - df["EVENT_DATE"]).dt.days // 30).astype("Int64")
    if "DAYS_BEFORE_ANCHOR" not in df.columns:
        df["EVENT_DATE"]  = pd.to_datetime(df["EVENT_DATE"],  errors="coerce")
        df["ANCHOR_DATE"] = pd.to_datetime(df["ANCHOR_DATE"], errors="coerce")
        df["DAYS_BEFORE_ANCHOR"] = (df["ANCHOR_DATE"] - df["EVENT_DATE"]).dt.days.astype("Int64")

    obs_map, med_map = load_mapping(args.mapping)
    print(f"  mapping: {len(obs_map)} obs codes, {len(med_map)} med codes")

    # Rename SQL columns to FE-expected names
    df = df.rename(columns={
        "ANCHOR_DATE":          "INDEX_DATE",
        "PATIENT_AGE":          "AGE_AT_INDEX",
        "MONTHS_BEFORE_ANCHOR": "MONTHS_BEFORE_INDEX",
        "CANCER_CLASS":         "LABEL",
    })

    # Single CODE_ID column: SNOMED for obs, MED for medications
    is_obs = df["EVENT_TYPE"] == "observation"
    df["CODE_ID"] = np.where(is_obs, df["SNOMED_C_T_CONCEPT_ID"], df["MED_CODE_ID"])
    df["CODE_ID"] = pd.to_numeric(df["CODE_ID"], errors="coerce").astype("Int64")

    # Single TERM column: TERM for obs, DRUG_TERM for medications
    df["TERM"] = np.where(is_obs, df["TERM"], df["DRUG_TERM"])

    # Assign CATEGORY by looking up CODE_ID in mapping (NULL for non-curated codes)
    df["CATEGORY"] = None
    df.loc[is_obs,  "CATEGORY"] = assign_category(df.loc[is_obs,  "CODE_ID"], obs_map).values
    df.loc[~is_obs, "CATEGORY"] = assign_category(df.loc[~is_obs, "CODE_ID"], med_map).values

    # Assign TIME_WINDOW from MONTHS_BEFORE_INDEX
    months = pd.to_numeric(df["MONTHS_BEFORE_INDEX"], errors="coerce").fillna(-1).astype(int).values
    df["TIME_WINDOW"] = assign_time_window(months)

    # Drop rows with NULL CATEGORY (codes not in mapping → no feature contribution).
    # Patients whose every event is NULL-category will be re-added below as
    # placeholder rows so they still appear in the FE feature matrix.
    n_before = len(df)
    df_categorized = df[df["CATEGORY"].notna() & df["TIME_WINDOW"].notna()].copy()
    print(f"  after category/window filter: {len(df_categorized):,} rows kept ({len(df_categorized)/n_before*100:.1f}%)")

    # Identify all unified_patients (who appeared in raw extract)
    all_patients = df[["PATIENT_GUID", "INDEX_DATE", "LABEL"]].drop_duplicates("PATIENT_GUID").reset_index(drop=True)
    sex_lookup = df.groupby("PATIENT_GUID")["SEX"].first().to_dict()
    age_lookup = df.groupby("PATIENT_GUID")["AGE_AT_INDEX"].first().to_dict()
    ethnicity_16_lookup = (
        df.groupby("PATIENT_GUID")["PATIENT_ETHNICITY_16"].first().to_dict()
        if "PATIENT_ETHNICITY_16" in df.columns else {}
    )
    ethnicity_6_lookup = (
        df.groupby("PATIENT_GUID")["PATIENT_ETHNICITY_6"].first().to_dict()
        if "PATIENT_ETHNICITY_6" in df.columns else {}
    )

    # Patients with at least one categorized event
    categorized_pids = set(df_categorized["PATIENT_GUID"].unique())
    n_total = len(all_patients)
    n_categorized = len(categorized_pids)
    n_placeholder = n_total - n_categorized
    print(f"  patients: {n_total:,} total | {n_categorized:,} have curated events | {n_placeholder:,} need placeholder rows")

    # Build placeholder rows for the all-zero-category patients
    if n_placeholder > 0:
        sparse = all_patients[~all_patients["PATIENT_GUID"].isin(categorized_pids)]
        placeholder_df = build_placeholder_rows(sparse, sex_lookup, age_lookup, ethnicity_16_lookup, ethnicity_6_lookup)
        df_categorized = pd.concat([df_categorized, placeholder_df], ignore_index=True)
        print(f"  added {len(placeholder_df):,} placeholder rows")

    # Restrict to FE-expected columns (drop SQL extras)
    df_categorized = df_categorized[OUT_COLS]

    # Coerce types for clean parquet write — VALUE was string from SQL with
    # empty-strings mixed with numerics; force numeric (NaN for non-numeric).
    # CODE_ID may also have mixed Int64/string from earlier ops — pin to string.
    df_categorized['VALUE']    = pd.to_numeric(df_categorized['VALUE'], errors='coerce')
    df_categorized['CODE_ID']  = df_categorized['CODE_ID'].astype('string')
    df_categorized['LABEL']    = pd.to_numeric(df_categorized['LABEL'], errors='coerce').astype('Int64')
    df_categorized['AGE_AT_INDEX'] = pd.to_numeric(df_categorized['AGE_AT_INDEX'], errors='coerce')
    df_categorized['EVENT_AGE']    = pd.to_numeric(df_categorized['EVENT_AGE'], errors='coerce')
    df_categorized['MONTHS_BEFORE_INDEX'] = pd.to_numeric(df_categorized['MONTHS_BEFORE_INDEX'], errors='coerce')
    for c in ('PATIENT_GUID', 'SEX', 'PATIENT_ETHNICITY_16', 'PATIENT_ETHNICITY_6', 'TIME_WINDOW', 'EVENT_TYPE', 'CATEGORY', 'TERM', 'ASSOCIATED_TEXT'):
        if c in df_categorized.columns:
            df_categorized[c] = df_categorized[c].astype('string')

    # Write unified file as Parquet (5-10× faster I/O, half the disk).
    # Sanity-check reads via read_table which auto-detects parquet/csv.
    out_path = args.out_dir / f"{args.prefix}.parquet"
    write_table(df_categorized, out_path, index=False)
    n_obs = (df_categorized['EVENT_TYPE'] == 'observation').sum()
    n_med = (df_categorized['EVENT_TYPE'] == 'medication').sum()
    print(f"\nWrote:")
    print(f"  {out_path}  ({len(df_categorized):,} rows total — obs:{n_obs:,}, med:{n_med:,}, {df_categorized['PATIENT_GUID'].nunique():,} patients)")


if __name__ == "__main__":
    main()
