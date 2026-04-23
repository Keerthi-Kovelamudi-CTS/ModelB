"""
Preprocess the 300K non-cancer raw CSV into FE-ready per-window holdout files.

Produces (per window):
  data/fe_input/{window}/prostate_{window}_obs_dropped.csv
  data/fe_input/{window}/prostate_{window}_med_dropped.csv

Keeps ALL male non-training patients — including those with zero
Prostate-relevant codes. Patients with zero matches get a single
placeholder row per window (category="__PLACEHOLDER__") so the
downstream FE still produces a patient entry (all features = 0 →
low risk score, as expected).

Chunked streaming (500K rows / chunk) to stay under 24 GB RAM on a Mac.

FIXES vs earlier stub:
  - UPPERCASE dtype keys so pandas actually applies them (prevents
    SNOMED codes being read as floats → '.0' suffix → no matches).
  - Uses two separate code columns (SNOMED_C_T_CONCEPT_ID for obs,
    MED_CODE_ID for med) picked per row by EVENT_TYPE.
  - SEX='M' filter applied early (saves ~60% of work).
  - Placeholder rows for zero-code patients so realistic-prevalence
    evaluation covers the full male non-training population.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_FILE = SCRIPT_DIR / "300K_NonCancer_Patients.csv"
MAPPING_FILE = SCRIPT_DIR / "code_category_mapping.json"
EXCLUDE_FILE = BASE / "2_Feature_Engineering" / "unique_patient_guids_all_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "data" / "fe_input"

# Must match training SQL windowing
WINDOW_DEFS = {
    "3mo":  {"B": (3, 14),  "A": (15, 27)},
    "6mo":  {"B": (6, 17),  "A": (18, 30)},
    "12mo": {"B": (12, 23), "A": (24, 36)},
}
INDEX_DATE = pd.Timestamp("2026-02-25")
CHUNKSIZE = 500_000

OUT_COLS = [
    "PATIENT_GUID", "SEX", "INDEX_DATE", "EVENT_DATE",
    "AGE_AT_INDEX", "EVENT_AGE", "MONTHS_BEFORE_INDEX", "TIME_WINDOW",
    "EVENT_TYPE", "CATEGORY", "CODE_ID", "TERM", "ASSOCIATED_TEXT",
    "VALUE", "LABEL",
]

# For zero-code patients we write one placeholder row per window so the
# FE pipeline still creates a patient entry. The category isn't in any
# real config list so it won't contribute to real features.
PLACEHOLDER_CATEGORY = "__PLACEHOLDER__"


def main():
    t0 = time.time()
    print(f"Raw file:     {RAW_FILE}  ({RAW_FILE.stat().st_size/1e9:.1f} GB)")

    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    obs_map = {str(k): v for k, v in mapping["obs"].items()}
    med_map = {str(k): v for k, v in mapping["med"].items()}
    print(f"Mapping:      {len(obs_map)} obs + {len(med_map)} med codes")

    exclude = set()
    if EXCLUDE_FILE.exists():
        ex = pd.read_csv(EXCLUDE_FILE)
        exclude = set(ex[ex.columns[0]].astype(str).unique())
    print(f"Exclude:      {len(exclude):,} training patients\n")

    handles, first_write, out_counts = {}, {}, {}
    for w in WINDOW_DEFS:
        for et in ("obs", "med"):
            p = OUTPUT_DIR / w / f"prostate_{w}_{et}_dropped.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            handles[(w, et)] = open(p, "w")
            first_write[(w, et)] = True
            out_counts[(w, et)] = 0

    total_rows = sex_kept = mapped_kept = excluded_kept = 0
    all_candidates = {}      # PATIENT_GUID -> max observed AGE_AT_INDEX (after male+non-training)
    matched_guids = set()    # PATIENT_GUID with ≥1 mapped event written

    dtype = {
        "PATIENT_GUID": str, "SEX": str, "CANCER_ID": str,
        "EVENT_TYPE": str, "SNOMED_C_T_CONCEPT_ID": str, "MED_CODE_ID": str,
        "TERM": str, "DRUG_TERM": str, "ASSOCIATED_TEXT": str, "VALUE": str,
    }

    reader = pd.read_csv(
        RAW_FILE, chunksize=CHUNKSIZE, low_memory=False,
        encoding="utf-8-sig", dtype=dtype,
    )

    for ci, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Male-only + drop training patients first
        chunk = chunk[chunk["SEX"] == "M"].copy()
        if exclude:
            chunk = chunk[~chunk["PATIENT_GUID"].astype(str).isin(exclude)]
        sex_kept += len(chunk)
        if chunk.empty:
            continue

        # Register every male-non-training patient (for placeholder coverage).
        ages = pd.to_numeric(chunk.get("EVENT_AGE"), errors="coerce")
        for guid, age in zip(chunk["PATIENT_GUID"].astype(str), ages):
            prev = all_candidates.get(guid)
            if prev is None or (pd.notna(age) and (prev is None or age > prev)):
                all_candidates[guid] = age

        # Row-wise code pick
        et = chunk["EVENT_TYPE"].astype(str).str.lower()
        is_obs = et.str.startswith("obs")

        snomed = chunk["SNOMED_C_T_CONCEPT_ID"] if "SNOMED_C_T_CONCEPT_ID" in chunk.columns else pd.Series([None]*len(chunk), index=chunk.index)
        medc   = chunk["MED_CODE_ID"]            if "MED_CODE_ID"            in chunk.columns else pd.Series([None]*len(chunk), index=chunk.index)

        chunk["CODE_ID"] = np.where(is_obs, snomed.astype(str), medc.astype(str))
        chunk["CODE_ID"] = chunk["CODE_ID"].str.strip().str.replace(r"\.0$", "", regex=True)
        chunk.loc[chunk["CODE_ID"].isin(["nan", "None", ""]), "CODE_ID"] = pd.NA
        chunk["EVENT_TYPE"] = np.where(is_obs, "observation", "medication")

        # Map to category
        cat = pd.Series(pd.NA, index=chunk.index, dtype="object")
        cat[is_obs]  = chunk.loc[is_obs,  "CODE_ID"].map(obs_map)
        cat[~is_obs] = chunk.loc[~is_obs, "CODE_ID"].map(med_map)
        chunk["CATEGORY"] = cat
        chunk = chunk[chunk["CATEGORY"].notna()].copy()
        mapped_kept += len(chunk)
        if chunk.empty:
            continue

        excluded_kept += len(chunk)
        matched_guids.update(chunk["PATIENT_GUID"].astype(str).unique().tolist())

        # Dates and derived fields
        chunk["EVENT_DATE"] = pd.to_datetime(chunk["EVENT_DATE"], errors="coerce")
        chunk["MONTHS_BEFORE_INDEX"] = ((INDEX_DATE - chunk["EVENT_DATE"]).dt.days / 30.44).round(1)
        chunk["INDEX_DATE"] = INDEX_DATE
        chunk["AGE_AT_INDEX"] = pd.to_numeric(chunk.get("EVENT_AGE"), errors="coerce")
        chunk["EVENT_AGE"] = chunk["AGE_AT_INDEX"]
        chunk["TERM"] = np.where(is_obs[chunk.index], chunk.get("TERM"), chunk.get("DRUG_TERM"))
        chunk["ASSOCIATED_TEXT"] = chunk.get("ASSOCIATED_TEXT", pd.NA)
        chunk["VALUE"] = pd.to_numeric(chunk.get("VALUE"), errors="coerce")
        chunk["LABEL"] = 0  # holdout = all non-cancer

        # Per-window write
        for w, ranges in WINDOW_DEFS.items():
            b_lo, b_hi = ranges["B"]
            a_lo, a_hi = ranges["A"]
            win = chunk.copy()
            win["TIME_WINDOW"] = pd.NA
            win.loc[(win["MONTHS_BEFORE_INDEX"] >= b_lo) & (win["MONTHS_BEFORE_INDEX"] < b_hi), "TIME_WINDOW"] = "B"
            win.loc[(win["MONTHS_BEFORE_INDEX"] >= a_lo) & (win["MONTHS_BEFORE_INDEX"] < a_hi), "TIME_WINDOW"] = "A"
            win = win[win["TIME_WINDOW"].notna()]
            if win.empty:
                continue
            for et_label, out_key in [("observation", "obs"), ("medication", "med")]:
                sub = win[win["EVENT_TYPE"] == et_label][OUT_COLS]
                if sub.empty:
                    continue
                fh = handles[(w, out_key)]
                sub.to_csv(fh, index=False, header=first_write[(w, out_key)])
                first_write[(w, out_key)] = False
                out_counts[(w, out_key)] += len(sub)

        if (ci + 1) % 20 == 0 or ci == 0:
            print(f"  chunk {ci+1}: total={total_rows:,} sex={sex_kept:,} "
                  f"mapped={mapped_kept:,} matched_guids={len(matched_guids):,} "
                  f"all_male_non_training={len(all_candidates):,} t={time.time()-t0:.0f}s")

    # Append placeholder rows for patients with zero mapped events
    placeholder_guids = [g for g in all_candidates if g not in matched_guids]
    n_placeholder = len(placeholder_guids)
    print(f"\nPlaceholder patients (zero mapped events): {n_placeholder:,}")

    if n_placeholder > 0:
        for w, ranges in WINDOW_DEFS.items():
            b_lo, b_hi = ranges["B"]
            mid_months = (b_lo + b_hi) / 2
            event_date = INDEX_DATE - pd.Timedelta(days=mid_months * 30.44)
            rows = pd.DataFrame({
                "PATIENT_GUID": placeholder_guids,
                "SEX": "M",
                "INDEX_DATE": INDEX_DATE,
                "EVENT_DATE": event_date,
                "AGE_AT_INDEX": [all_candidates[g] for g in placeholder_guids],
                "EVENT_AGE":    [all_candidates[g] for g in placeholder_guids],
                "MONTHS_BEFORE_INDEX": mid_months,
                "TIME_WINDOW": "B",
                "EVENT_TYPE": "observation",
                "CATEGORY": PLACEHOLDER_CATEGORY,
                "CODE_ID": "",
                "TERM": PLACEHOLDER_CATEGORY,
                "ASSOCIATED_TEXT": "",
                "VALUE": np.nan,
                "LABEL": 0,
            })[OUT_COLS]
            fh = handles[(w, "obs")]
            rows.to_csv(fh, index=False, header=first_write[(w, "obs")])
            first_write[(w, "obs")] = False
            out_counts[(w, "obs")] += len(rows)
            print(f"  {w}: +{len(rows):,} placeholder obs rows")

    for fh in handles.values():
        fh.close()

    total_holdout_patients = len(all_candidates)
    print(f"\n{'='*70}\nDONE in {time.time()-t0:.1f}s")
    print(f"Total raw rows:                 {total_rows:,}")
    print(f"After SEX=M filter:             {sex_kept:,}")
    print(f"After mapping (kept rows):      {mapped_kept:,}")
    print(f"Matched patients:               {len(matched_guids):,}")
    print(f"Placeholder patients added:     {n_placeholder:,}")
    print(f"Total holdout patients:         {total_holdout_patients:,}")
    print(f"\nPer-window output row counts:")
    for key, n in sorted(out_counts.items()):
        print(f"  {key}: {n:,}")


if __name__ == "__main__":
    main()
