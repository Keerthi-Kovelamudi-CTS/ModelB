"""
Preprocess the 300K non-cancer raw CSV into FE-ready per-window holdout files.

ALIGNED to Prostate_Cancer_2.0 pipeline:
  - Lookback: 60 months total (matches SQL `years_before=5`)
  - TIME_WINDOW: 1-30mo = 'B' (recent), 30-60mo = 'A' (earlier)
    (single boundary, same as 2_Feature_Engineering/0_preprocess_to_fe.py)
  - Per-window event filter: keep events in [months_before, 60]
    (matches SQL `months_before` for each window)
  - Codelist: ../codelists/code_category_mapping_v2.json
  - Windows: imported from 2_Feature_Engineering/config.py
  - Output schema: matches OUT_COLS of 0_preprocess_to_fe.py

Produces (per window):
  data/fe_input/{window}/prostate_{window}_obs_dropped.csv
  data/fe_input/{window}/prostate_{window}_med_dropped.csv

Keeps ALL male non-training patients — including those with zero
prostate-relevant codes. Patients with zero matches get a single
placeholder row per window (CATEGORY="__PLACEHOLDER__") so the
downstream FE still produces a patient entry (all features = 0 →
low risk score, as expected).

Chunked streaming (500K rows / chunk) to stay under 24 GB RAM.

Filters:
  - SEX='M' (prostate cancer is male-only)
  - PATIENT_GUID NOT IN training set (../2_Feature_Engineering/unique_patient_guids_all_data.csv if present)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Import WINDOWS from 2.0 config so this stays in sync if windows change
sys.path.insert(0, str(BASE / '2_Feature_Engineering'))
import config as fe_config

RAW_FILE     = SCRIPT_DIR / "300K_NonCancer_Patients.csv"
MAPPING_FILE = BASE / "codelists" / "code_category_mapping_v2.json"
EXCLUDE_FILE = BASE / "2_Feature_Engineering" / "unique_patient_guids_all_data.csv"
OUTPUT_DIR   = SCRIPT_DIR / "data" / "fe_input"

# 2.0 pipeline constants — must match SQL params + 0_preprocess_to_fe.py
LOOKBACK_MONTHS         = 60        # SQL: years_before=5 → 60 months total
TIME_WINDOW_MID_MONTHS  = 30        # 0_preprocess_to_fe.py: 1-30=B, 30-60=A
WINDOWS                 = list(fe_config.WINDOWS)  # ['1mo','3mo','6mo','12mo']
WINDOW_MONTHS_BEFORE    = {w: int(w.replace('mo', '')) for w in WINDOWS}

# INDEX_DATE for the holdout — matches SQL anchor_window_end so events have
# the same recency cap as training data.
INDEX_DATE = pd.Timestamp("2026-04-25")

CHUNKSIZE = 500_000

OUT_COLS = [
    "PATIENT_GUID", "SEX", "INDEX_DATE", "EVENT_DATE",
    "AGE_AT_INDEX", "EVENT_AGE", "MONTHS_BEFORE_INDEX", "TIME_WINDOW",
    "EVENT_TYPE", "CATEGORY", "CODE_ID", "TERM", "ASSOCIATED_TEXT",
    "VALUE", "LABEL",
]

PLACEHOLDER_CATEGORY = "__PLACEHOLDER__"


def main():
    t0 = time.time()
    print(f"Raw file:     {RAW_FILE}  ({RAW_FILE.stat().st_size/1e9:.1f} GB)")
    print(f"Codelist:     {MAPPING_FILE}")
    print(f"Windows:      {WINDOWS}  (months_before: {WINDOW_MONTHS_BEFORE})")
    print(f"Lookback:     {LOOKBACK_MONTHS} months  |  Time-window mid: {TIME_WINDOW_MID_MONTHS} months")
    print(f"INDEX_DATE:   {INDEX_DATE.strftime('%Y-%m-%d')}\n")

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
    for w in WINDOWS:
        for et in ("obs", "med"):
            p = OUTPUT_DIR / w / f"prostate_{w}_{et}_dropped.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            handles[(w, et)] = open(p, "w")
            first_write[(w, et)] = True
            out_counts[(w, et)] = 0

    total_rows = sex_kept = mapped_kept = 0
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

        # Row-wise CODE_ID pick: SNOMED for obs, MED_CODE_ID for medications
        et_lc = chunk["EVENT_TYPE"].astype(str).str.lower()
        is_obs = et_lc.str.startswith("obs")

        snomed = chunk["SNOMED_C_T_CONCEPT_ID"] if "SNOMED_C_T_CONCEPT_ID" in chunk.columns else pd.Series([None]*len(chunk), index=chunk.index)
        medc   = chunk["MED_CODE_ID"]            if "MED_CODE_ID"            in chunk.columns else pd.Series([None]*len(chunk), index=chunk.index)

        chunk["CODE_ID"] = np.where(is_obs, snomed.astype(str), medc.astype(str))
        chunk["CODE_ID"] = chunk["CODE_ID"].str.strip().str.replace(r"\.0$", "", regex=True)
        chunk.loc[chunk["CODE_ID"].isin(["nan", "None", ""]), "CODE_ID"] = pd.NA
        chunk["EVENT_TYPE"] = np.where(is_obs, "observation", "medication")

        # Map CODE_ID to CATEGORY via codelist (per-row obs vs med map)
        cat = pd.Series(pd.NA, index=chunk.index, dtype="object")
        cat[is_obs]  = chunk.loc[is_obs,  "CODE_ID"].map(obs_map)
        cat[~is_obs] = chunk.loc[~is_obs, "CODE_ID"].map(med_map)
        chunk["CATEGORY"] = cat
        chunk = chunk[chunk["CATEGORY"].notna()].copy()
        mapped_kept += len(chunk)
        if chunk.empty:
            continue

        matched_guids.update(chunk["PATIENT_GUID"].astype(str).unique().tolist())

        # Derived fields
        chunk["EVENT_DATE"] = pd.to_datetime(chunk["EVENT_DATE"], errors="coerce")
        chunk["MONTHS_BEFORE_INDEX"] = ((INDEX_DATE - chunk["EVENT_DATE"]).dt.days / 30.44).round(1)
        chunk["INDEX_DATE"] = INDEX_DATE
        chunk["AGE_AT_INDEX"] = pd.to_numeric(chunk.get("EVENT_AGE"), errors="coerce")
        chunk["EVENT_AGE"] = chunk["AGE_AT_INDEX"]
        chunk["TERM"] = np.where(is_obs[chunk.index], chunk.get("TERM"), chunk.get("DRUG_TERM"))
        chunk["ASSOCIATED_TEXT"] = chunk.get("ASSOCIATED_TEXT", pd.NA)
        chunk["VALUE"] = pd.to_numeric(chunk.get("VALUE"), errors="coerce")
        chunk["LABEL"] = 0  # holdout = all non-cancer

        # Single TIME_WINDOW assignment (matches 2.0 0_preprocess_to_fe.py)
        chunk["TIME_WINDOW"] = pd.NA
        m = chunk["MONTHS_BEFORE_INDEX"]
        chunk.loc[(m >= 1) & (m <= TIME_WINDOW_MID_MONTHS), "TIME_WINDOW"] = "B"
        chunk.loc[(m > TIME_WINDOW_MID_MONTHS) & (m <= LOOKBACK_MONTHS), "TIME_WINDOW"] = "A"

        # Per-window write: keep events in [months_before, LOOKBACK_MONTHS]
        for w in WINDOWS:
            mb = WINDOW_MONTHS_BEFORE[w]
            win = chunk[
                (chunk["MONTHS_BEFORE_INDEX"] >= mb)
                & (chunk["MONTHS_BEFORE_INDEX"] <= LOOKBACK_MONTHS)
                & (chunk["TIME_WINDOW"].notna())
            ]
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
        # Place EVENT_DATE = INDEX_DATE - TIME_WINDOW_MID_MONTHS so it sits inside
        # the lookback range and survives the FE leak guard (matches 0_preprocess_to_fe.py).
        placeholder_offset = pd.DateOffset(months=TIME_WINDOW_MID_MONTHS)
        event_date = INDEX_DATE - placeholder_offset
        for w in WINDOWS:
            rows = pd.DataFrame({
                "PATIENT_GUID": placeholder_guids,
                "SEX": "M",
                "INDEX_DATE": INDEX_DATE,
                "EVENT_DATE": event_date,
                "AGE_AT_INDEX": [all_candidates[g] for g in placeholder_guids],
                "EVENT_AGE":    [all_candidates[g] for g in placeholder_guids],
                "MONTHS_BEFORE_INDEX": TIME_WINDOW_MID_MONTHS,
                "TIME_WINDOW": "A",
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
