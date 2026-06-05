#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Prostate B+C — full pipeline driver
#
# Runs the dual-horizon experiment end-to-end:
#   1. BQ extract × 3 windows (fast pattern: dest_table → GCS → gsutil)
#   2. FE pipeline × 3 windows (preprocess → cleanup → JSON-pack + sidecar)
#   3. Build unified cohort + shared train/val/test split
#   4. Train 3 models (12mo, 1mo_5y, 1mo_12mo) on the shared split
#   5. Compare dual-horizon Setup A (12mo+1mo_5y) vs Setup B (12mo+1mo_12mo)
#
# Skip-flags:
#   SKIP_BQ=1     SKIP_FE=1     SKIP_SPLIT=1     SKIP_TRAIN=1     SKIP_COMPARE=1
#
# Estimated wall-clock on cpu-01:
#   BQ extract:   ~5min  × 3 = 15min
#   FE pipeline:  ~20min × 3 = 60min
#   Shared split: ~1min
#   Training:     ~60min × 3 = 180min
#   Compare:      ~5min
#   TOTAL:        ~4-5 hours
# ════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WINDOWS=("12mo" "1mo_5y" "1mo_12mo")

SKIP_BQ="${SKIP_BQ:-0}"
SKIP_FE="${SKIP_FE:-0}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

BQ_SCRATCH_DATASET="${BQ_SCRATCH_DATASET:-prj-cts-ai-dev-sp:temp_output_us}"
GCS_STAGE_PATH="${GCS_STAGE_PATH:-gs://gcs-ai-dev-model-artifacts/keerthi}"
BQ_LOCATION="${BQ_LOCATION:-US}"

banner() {
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════"
}

# ─── Step 1: BQ extracts (3 windows in sequence) ───
if [ "$SKIP_BQ" = "1" ]; then
    banner "STEP 1/6: BQ extracts — SKIPPED"
else
    for window in "${WINDOWS[@]}"; do
        banner "STEP 1/6: BQ extract — $window"
        sql_file="$ROOT/1_SQL/${window}.sql"
        raw_csv="$ROOT/2_Feature_Engineering/data/raw/prostate_${window}_raw.csv"
        scratch_table="${BQ_SCRATCH_DATASET}.prostate_${window}_BC"
        gcs_path="${GCS_STAGE_PATH}/prostate_${window}_BC"

        if [ -f "$raw_csv" ] && [ "$(stat -c %s "$raw_csv" 2>/dev/null || stat -f %z "$raw_csv")" -gt 100000000 ]; then
            echo "  ↳ skip — $raw_csv already exists ($(ls -lh "$raw_csv" | awk '{print $5}'))"
            continue
        fi

        mkdir -p "$(dirname "$raw_csv")"

        echo "  ↳ 1a. BQ query → $scratch_table"
        bq query --use_legacy_sql=false --location="$BQ_LOCATION" \
            --destination_table="$scratch_table" --replace=true --quiet < "$sql_file"

        echo "  ↳ 1b. Export → GCS"
        bq extract --destination_format=CSV --print_header=true --location="$BQ_LOCATION" \
            "$scratch_table" "${gcs_path}_*.csv"

        echo "  ↳ 1c. Download → local"
        gsutil -m cp "${gcs_path}_*.csv" "$(dirname "$raw_csv")/"

        echo "  ↳ 1d. Concat shards → $raw_csv"
        cd "$(dirname "$raw_csv")"
        first=1
        for s in $(ls -1 "$(basename "$gcs_path")"_*.csv | sort); do
            if [ "$first" = "1" ]; then
                cat "$s" > "$(basename "$raw_csv")"; first=0
            else
                tail -n +2 "$s" >> "$(basename "$raw_csv")"
            fi
        done
        rm -f "$(basename "$gcs_path")"_*.csv
        cd "$ROOT"

        echo "  ↳ 1e. Cleanup BQ scratch + GCS shards"
        bq rm -f -t "$scratch_table" || true
        gsutil -m rm "${gcs_path}_*.csv" 2>/dev/null || true

        rows=$(( $(wc -l < "$raw_csv") - 1 ))
        echo "  ✅ $window — $rows rows extracted"
    done
fi

# ─── Step 2: FE pipeline per window ───
if [ "$SKIP_FE" = "1" ]; then
    banner "STEP 2/6: FE pipeline — SKIPPED"
else
    for window in "${WINDOWS[@]}"; do
        banner "STEP 2/6: FE — $window"
        cd "$ROOT/2_Feature_Engineering"
        WINDOW=$window python3 0_preprocess.py
        WINDOW=$window python3 3_cleanup.py
        WINDOW=$window python3 4_transform_features_json.py --no-bq
    done
fi

# ─── Step 3: Shared split ───
if [ "$SKIP_SPLIT" = "1" ]; then
    banner "STEP 3/6: Shared split — SKIPPED"
else
    banner "STEP 3/6: Build unified cohort + shared split"
    cd "$ROOT/3_Modeling"
    python3 _shared_split.py
fi

# ─── Step 4: Train 3 models ───
if [ "$SKIP_TRAIN" = "1" ]; then
    banner "STEP 4/6: Training — SKIPPED"
else
    banner "STEP 4/6: Train 3 models (12mo, 1mo_5y, 1mo_12mo) on shared split"
    cd "$ROOT/3_Modeling"
    python3 1_training.py
fi

# ─── Step 5: Comparison ───
if [ "$SKIP_COMPARE" = "1" ]; then
    banner "STEP 5/6: Comparison — SKIPPED"
else
    banner "STEP 5/6: Compare dual-horizon Setup A vs Setup B"
    cd "$ROOT/7_Holdout_Test"
    python3 test_dual_horizon_compare.py
fi

# ─── Step 6: Explainability (batch SHAP per window) ───
SKIP_EXPLAIN="${SKIP_EXPLAIN:-0}"
if [ "$SKIP_EXPLAIN" = "1" ]; then
    banner "STEP 6/6: Explainability — SKIPPED"
else
    banner "STEP 6/6: Batch SHAP explainability per window"
    cd "$ROOT/6_Explainability"
    for window in "${WINDOWS[@]}"; do
        echo "  ─── SHAP: $window ─── "
        WINDOW=$window python3 1_explain_predictions.py --window $window || echo "    ⚠ explainability for $window failed (non-fatal — training/inference outputs are intact)"
    done
fi

banner "✅ ALL DONE"
echo "  Output: 7_Holdout_Test/test_compare_per_patient.csv"
echo "  Decision: pick winning 1mo variant from console summary"
