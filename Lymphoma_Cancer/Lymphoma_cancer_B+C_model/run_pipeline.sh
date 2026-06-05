#!/bin/bash
# Lymphoma_2.0_1to1 end-to-end pipeline (Approach B)
#
# Steps:
#   1. SQL extract  → data/raw/lymphoma_1mo_raw.csv  (run separately in BQ Studio or via bq client)
#   2. Preprocess   → data/1mo/{lymphoma_1mo_obs_dropped, lymphoma_1mo_med_dropped}.csv
#   3. FE pipeline  → 2_Feature_Engineering/results/1mo/...features.csv
#   4. Modeling     → 3_Modeling/results/1_training/1mo/{predictions, saved_models, ...}
#   5. Holdout eval → 4_ExpandedData_Test/results/{combined_eval_1mo.txt, predictions_unseen_1mo.csv}
#
# Usage:
#   bash run_pipeline.sh preprocess     # run only step 2
#   bash run_pipeline.sh fe             # run only step 3
#   bash run_pipeline.sh model          # run only step 4
#   bash run_pipeline.sh eval           # run only step 5
#   bash run_pipeline.sh all            # run steps 2-5

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Override with WINDOW=3mo (etc.) to run a single window. Default loops all.
WINDOW="${WINDOW:-}"
WINDOWS_ALL=(1mo 3mo 6mo 12mo)
STEP="${1:-all}"

run_preprocess () {
  cd "$ROOT/2_Feature_Engineering"
  local windows=("${WINDOWS_ALL[@]}")
  [ -n "$WINDOW" ] && windows=("$WINDOW")
  for w in "${windows[@]}"; do
    echo "==== STEP 2: preprocess raw → FE-ready CSVs (window=$w) ===="
    python3 0_preprocess_to_fe.py \
      --raw     data/raw/lymphoma_${w}_raw.csv \
      --out-dir data/${w} \
      --prefix  lymphoma_${w}
  done
}

run_fe () {
  echo "==== STEP 3: feature engineering (window=$WINDOW) ===="
  cd "$ROOT/2_Feature_Engineering"
  python3 0_run_pipeline.py
}

run_model () {
  echo "==== STEP 4: train models (window=$WINDOW) ===="
  cd "$ROOT/3_Modeling"
  python3 1_run_modeling.py
}

run_eval () {
  echo "==== STEP 5: holdout combined evaluation (window=$WINDOW) ===="
  cd "$ROOT/4_ExpandedData_Test"
  if [ ! -s "data/fe_input/${WINDOW}/lymphoma_${WINDOW}_obs_dropped.csv" ]; then
    echo "  (preprocessing 300K holdout)"
    python3 1_preprocess_holdout.py
  fi
  python3 2_run_holdout_fe.py

  cd "$ROOT/3_Modeling"
  python3 2_predict_unseen.py --window $WINDOW \
    --data "$ROOT/4_ExpandedData_Test/results/${WINDOW}/holdout_features_${WINDOW}.csv"
  cp "results/2_predictions/${WINDOW}/predictions_unseen_${WINDOW}.csv" \
     "$ROOT/4_ExpandedData_Test/results/predictions_unseen_${WINDOW}.csv"

  cd "$ROOT/4_ExpandedData_Test"
  python3 3_combined_evaluation.py --window $WINDOW
}

case "$STEP" in
  preprocess) run_preprocess ;;
  fe)         run_fe ;;
  model)      run_model ;;
  eval)       run_eval ;;
  all)        run_preprocess && run_fe && run_model && run_eval ;;
  *)          echo "Unknown step: $STEP. Use preprocess|fe|model|eval|all"; exit 1 ;;
esac

echo "==== DONE ===="
