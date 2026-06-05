"""
Generate `unique_patient_guids_all_data.csv` — the training-cohort exclusion list
that 1_preprocess_holdout.py reads to filter the 300K holdout against any
patient seen in any training window.

WHY THIS EXISTS:
The holdout-preprocess script silently skips exclusion if this file is missing.
Without it, training patients leak into the holdout set → inflated metrics.
This script must be run AFTER the FE pipeline finishes (cleanup outputs ready)
and BEFORE 1_preprocess_holdout.py.

Output: ../2_Feature_Engineering/unique_patient_guids_all_data.csv
        Single column, header `patient_guid`, one GUID per line.
        Format matches v1 (Breast_Cancer/) — for compatibility with
        1_preprocess_holdout.py.

Usage:
    python 0_generate_training_guids.py
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
CLEANUP_DIR = BASE / "2_Feature_Engineering" / "results" / "5_cleanup"
OUTPUT_FILE = BASE / "2_Feature_Engineering" / "unique_patient_guids_all_data.csv"

WINDOWS = ["1mo", "12mo"]


def main():
    if not CLEANUP_DIR.exists():
        sys.exit(f"ERROR: cleanup dir missing: {CLEANUP_DIR}\nRun the FE pipeline first.")

    all_guids = set()
    print(f"Reading cleanup outputs from {CLEANUP_DIR}/")
    for w in WINDOWS:
        p = CLEANUP_DIR / w / f"feature_matrix_clean_{w}.parquet"
        if not p.exists():
            p = CLEANUP_DIR / w / f"feature_matrix_clean_{w}.csv"
        if not p.exists():
            print(f"  [{w}] MISSING — skipping")
            continue
        if p.suffix == ".parquet":
            df = pd.read_parquet(p, columns=None)
            # The PATIENT_GUID is the index after cleanup
            guids = df.index.astype(str).tolist()
        else:
            df = pd.read_csv(p, index_col=0)
            guids = df.index.astype(str).tolist()
        n_before = len(all_guids)
        all_guids.update(guids)
        print(f"  [{w}] {len(guids):,} patients → +{len(all_guids) - n_before:,} new (cumulative {len(all_guids):,})")

    if not all_guids:
        sys.exit("ERROR: zero GUIDs found across all windows — cleanup outputs empty?")

    # Write in v1 format: single column, header `patient_guid`
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_guid": sorted(all_guids)}).to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(all_guids):,} unique training GUIDs → {OUTPUT_FILE}")
    print("\nNext: 1_preprocess_holdout.py will now exclude these from the 300K holdout.")


if __name__ == "__main__":
    main()
