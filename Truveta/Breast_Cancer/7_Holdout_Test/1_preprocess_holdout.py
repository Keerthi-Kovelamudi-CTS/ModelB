"""
Truveta breast — build the holdout BQ cohort table.

Truveta uses a BQ-cohort paradigm (not raw-CSV like EMIS), so the "preprocess"
step here is: build a `truveta_gold.breast_holdout_cohort_events_{window}` table
containing events for ~270K non-cancer non-training female patients per window.

WORKFLOW
========
1. Read training patient_ids from 2_Feature_Engineering/unique_patient_guids_all_data.csv
   (produced by 0_generate_training_guids.py).
2. Upload training guids to a scratch BQ table — works at any cohort size,
   no query-string-length limits (Fix #2).
3. For each window, clone breast_truveta_{window}.sql but
     - bump non_cancer_ratio=1→15 (gives ~270K non-cancer for breast)
     - merge a NOT EXISTS filter against the scratch training table
       at the top-level CTE chain (Fix #3 — no nested WITH).
4. Execute via `bq query --destination_table=...breast_holdout_cohort_events_{w}`.

Downstream: 2_run_holdout_fe.py invokes 0_extract.py + 1_build_features.py on
each holdout table to produce holdout feature matrices.

Usage:
    python 1_preprocess_holdout.py
    python 1_preprocess_holdout.py --windows 1mo 12mo
    python 1_preprocess_holdout.py --keep-scratch-table   # don't auto-drop
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent

sys.path.insert(0, str(BASE / "2_Feature_Engineering"))
import config as fe_config

TRAINING_GUIDS_FILE = BASE / "2_Feature_Engineering" / "unique_patient_guids_all_data.csv"
SCRATCH_DATASET     = "truveta_gold"
SCRATCH_TABLE       = "_scratch_breast_training_guids"
SCRATCH_FQN         = f"prj-cts-ai-dev-sp.{SCRATCH_DATASET}.{SCRATCH_TABLE}"


def upload_training_guids_to_bq(guids_csv: Path) -> str:
    """Upload the training guids CSV to a scratch BQ table.

    Returns the fully-qualified scratch table name.
    """
    # Write a column-named CSV that BQ schema-detect can ingest.
    df = pd.read_csv(guids_csv).rename(columns={"patient_guid": "patient_id"})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        tmp_csv = f.name

    try:
        print(f"  uploading {len(df):,} training guids → {SCRATCH_FQN}")
        result = subprocess.run(
            ["bq", "load",
             f"--project_id={fe_config.BQ_PROJECT}",
             "--location=us-central1",
             "--source_format=CSV",
             "--skip_leading_rows=1",
             "--replace",
             "--autodetect",
             f"{SCRATCH_DATASET}.{SCRATCH_TABLE}",
             tmp_csv],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            sys.exit(f"bq load failed:\nSTDERR: {result.stderr}")
    finally:
        os.unlink(tmp_csv)
    return SCRATCH_FQN


def build_holdout_sql(scratch_table_fqn: str, window: str) -> str:
    """Build a holdout cohort SQL for one window.

    Strategy:
      - Read breast_truveta_{window}.sql (the training-cohort SQL).
      - Strip its `WITH params AS (...)` prefix so we can merge CTEs.
      - Prepend our own `WITH _training_ids AS (...), params AS (...), ...`
        to keep a single flat CTE chain (Fix #3).
      - Bump non_cancer_ratio: 1 → 15.
      - Wrap final SELECT with NOT EXISTS against _training_ids.
    """
    src = (BASE / "2_Feature_Engineering" / "SQL" / f"breast_truveta_{window}.sql").read_text()

    # Bump non_cancer_ratio
    src = src.replace(
        "1    AS non_cancer_ratio,",
        "15   AS non_cancer_ratio,    -- bumped for holdout (~270K non-cancer pool)",
        1,
    )

    # Strip leading `WITH params AS` → leave just `params AS` so we can merge.
    src_lines = src.splitlines()
    idx = next(i for i, ln in enumerate(src_lines) if ln.startswith("WITH params AS"))
    src_lines[idx] = src_lines[idx].replace("WITH params AS", "params AS", 1)
    body = "\n".join(src_lines).rstrip().rstrip(";")

    # Insert NOT EXISTS into the FINAL SELECT block (which is at the bottom):
    #   SELECT * FROM observation_events UNION ALL SELECT * FROM medication_events ORDER BY patient_id
    # Replace with:
    #   SELECT * FROM (...) holdout_events WHERE NOT EXISTS (...)
    # We do this surgically on the trailing SELECT...UNION ALL...ORDER BY block.
    if "SELECT * FROM observation_events\nUNION ALL\nSELECT * FROM medication_events\nORDER BY patient_id" not in body:
        # Be tolerant of whitespace differences
        marker = "SELECT * FROM observation_events"
        if marker not in body:
            raise ValueError("Could not find final-SELECT marker in source SQL")

    flat = (
        "WITH _training_ids AS (\n"
        f"  SELECT patient_id FROM `{scratch_table_fqn}`\n"
        "),\n"
        f"{body[body.index('params AS'):]}\n"
    )
    # The body already ends with the final SELECT. Wrap it.
    # Find the final SELECT in flat and wrap it with NOT EXISTS.
    # Simpler: append a final outer SELECT.
    final_select_start = flat.rfind("SELECT * FROM observation_events")
    inner = flat[final_select_start:]
    prefix = flat[:final_select_start]
    wrapped = (
        f"{prefix}"
        f"SELECT * FROM (\n  {inner}\n) holdout_events\n"
        f"WHERE NOT EXISTS (\n"
        f"  SELECT 1 FROM _training_ids t WHERE t.patient_id = holdout_events.patient_id\n"
        f");"
    )
    return wrapped


def main(windows, keep_scratch_table):
    if not TRAINING_GUIDS_FILE.exists():
        sys.exit(
            f"ERROR: {TRAINING_GUIDS_FILE} not found. "
            f"Run 0_generate_training_guids.py first."
        )

    n_guids = len(pd.read_csv(TRAINING_GUIDS_FILE))
    print(f"Loaded {n_guids:,} training patient_ids to exclude.")

    scratch_fqn = upload_training_guids_to_bq(TRAINING_GUIDS_FILE)
    print(f"  ✓ scratch table ready: {scratch_fqn}")

    try:
        for w in windows:
            sql = build_holdout_sql(scratch_fqn, w)
            out_sql_path = SCRIPT_DIR / f"breast_truveta_holdout_{w}.sql"
            out_sql_path.write_text(sql)
            print(f"\n[{w}] wrote SQL → {out_sql_path.name}")

            dst = f"prj-cts-ai-dev-sp:truveta_gold.breast_holdout_cohort_events_{w}"
            print(f"[{w}] building BQ table {dst}")
            result = subprocess.run(
                ["bq", "query", "--use_legacy_sql=false",
                 f"--project_id={fe_config.BQ_PROJECT}",
                 "--location=us-central1",
                 "--destination_table", dst,
                 "--replace",
                 "--max_rows=1"],
                input=sql, capture_output=True, text=True,
            )
            if result.returncode != 0:
                sys.exit(f"BQ holdout build failed for {w}:\nSTDERR: {result.stderr}")
            print(f"[{w}] ✓ holdout cohort table built")
    finally:
        if not keep_scratch_table:
            print(f"\n  dropping scratch table {scratch_fqn}")
            subprocess.run(
                ["bq", "rm", "-f", "-t",
                 f"--project_id={fe_config.BQ_PROJECT}",
                 f"{SCRATCH_DATASET}.{SCRATCH_TABLE}"],
                capture_output=True, text=True,
            )

    print("\nDone. Next: 2_run_holdout_fe.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=fe_config.WINDOWS,
                    choices=fe_config.WINDOWS,
                    help="Which windows to build holdout cohorts for")
    ap.add_argument("--keep-scratch-table", action="store_true",
                    help="Don't drop the scratch training-guids BQ table at the end")
    args = ap.parse_args()
    main(args.windows, args.keep_scratch_table)
