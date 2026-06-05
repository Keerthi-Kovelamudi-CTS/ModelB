#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Leukaemia_2.0_1to1 — Phase 1.1: Run 4 P1 SQLs on BigQuery and
# land results in Data_Input/ with the filenames Ranking.py expects.
#
# Prereqs:
#   - Google Cloud SDK installed and authenticated (`gcloud auth login`)
#   - Default project set (`gcloud config set project <PROJECT_ID>`)
#   - The `bq` CLI on PATH (ships with Google Cloud SDK)
#
# Usage:
#   bash run_bq_extract.sh              # run all 4 queries
#   bash run_bq_extract.sh pos_obs      # run only positive observations
#   bash run_bq_extract.sh neg_obs
#   bash run_bq_extract.sh pos_med
#   bash run_bq_extract.sh neg_med
#
# Output:
#   Data_Input/Leukaemia_positive_obs.csv
#   Data_Input/Leukaemia_negative_obs.csv
#   Data_Input/Leukaemia_positive_med.csv
#   Data_Input/Leukaemia_negative_med.csv
#
# Next step after this:
#   python3 Ranking.py
#   python3 compare_to_curated.py
# ════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SQL_DIR="$ROOT/SQL Queries/BigQuery"
OUT_DIR="$ROOT/Data_Input"
MAX_ROWS="${MAX_ROWS:-1000000}"   # safe ceiling for aggregated outputs (1 row per code)
STEP="${1:-all}"

mkdir -p "$OUT_DIR"

run_one () {
  local sql_path="$1"
  local out_csv="$2"
  local label="$3"

  if [ ! -f "$sql_path" ]; then
    echo "❌ SQL not found: $sql_path"
    exit 1
  fi

  echo "==== $label ===="
  echo "  SQL: $sql_path"
  echo "  OUT: $out_csv"

  local t0
  t0=$(date +%s)
  bq query \
    --use_legacy_sql=false \
    --max_rows="$MAX_ROWS" \
    --format=csv \
    < "$sql_path" \
    > "$out_csv"

  local rows
  rows=$(($(wc -l < "$out_csv") - 1))   # subtract header
  local size
  size=$(ls -lh "$out_csv" | awk '{print $5}')
  local dt=$(( $(date +%s) - t0 ))
  echo "  ✅ wrote $rows rows ($size) in ${dt}s"
  echo
}

run_pos_obs () {
  run_one \
    "$SQL_DIR/P1- Top SNOMEDs — Positive Cohort (Observations).sql" \
    "$OUT_DIR/Leukaemia_positive_obs.csv" \
    "STEP 1/4: Top SNOMEDs — POSITIVE cohort (observations)"
}

run_neg_obs () {
  run_one \
    "$SQL_DIR/P1- Top SNOMEDs — Negative Cohort (Observations).sql" \
    "$OUT_DIR/Leukaemia_negative_obs.csv" \
    "STEP 2/4: Top SNOMEDs — NEGATIVE cohort (observations)"
}

run_pos_med () {
  run_one \
    "$SQL_DIR/P1 - Top Medications — Positive Cohort.sql" \
    "$OUT_DIR/Leukaemia_positive_med.csv" \
    "STEP 3/4: Top Medications — POSITIVE cohort"
}

run_neg_med () {
  run_one \
    "$SQL_DIR/P1 - Top Medications — Negative Cohort.sql" \
    "$OUT_DIR/Leukaemia_negative_med.csv" \
    "STEP 4/4: Top Medications — NEGATIVE cohort"
}

case "$STEP" in
  pos_obs) run_pos_obs ;;
  neg_obs) run_neg_obs ;;
  pos_med) run_pos_med ;;
  neg_med) run_neg_med ;;
  all)
    run_pos_obs
    run_neg_obs
    run_pos_med
    run_neg_med
    ;;
  *)
    echo "Unknown step: $STEP. Use one of: pos_obs | neg_obs | pos_med | neg_med | all"
    exit 1
    ;;
esac

echo "==== ✅ DONE — CSVs in $OUT_DIR ===="
ls -lh "$OUT_DIR"/*.csv 2>/dev/null || echo "(no CSVs found — did all steps run?)"
echo
echo "Next:"
echo "  cd \"$ROOT\""
echo "  python3 Ranking.py"
echo "  python3 compare_to_curated.py"
