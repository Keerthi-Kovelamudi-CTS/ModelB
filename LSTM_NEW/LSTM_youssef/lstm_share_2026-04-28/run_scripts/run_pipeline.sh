#!/bin/bash
# Run the deployment-grade LSTM pipeline (lung + prostate).
#
# Config (current best, locked in):
#   Data:    v45_balanced_min3 SQL  (1:1 cohort, HAVING ≥3 curated events)
#   Filter:  curated codelist (155 lung / 237 prostate codes; no PSA)
#   Tokens:  --max-snomed 50, --min-snomed-seq 3, --min-code-frequency 5
#   Branches: lung uses dual (snomed + med); prostate snomed-only
#
# Optional: --variance runs 5 seeds (42 + 13/99/7/2024) per cancer for stability.
#
# Prereqs:
#   1. Run sql/{cancer}_v45_balanced_min3.sql in BigQuery → export to local CSV.
#   2. Drop CSVs into the data dir referenced inside source_notebooks/{cancer}_lstm_pipeline_v1.ipynb
#      (default: ~/lstm_work/data/v45_balanced_min3/).
#   3. Conda env from environment.yml activated (default name: k-dev).

set -e
VARIANCE=0
[[ "$1" == "--variance" ]] && VARIANCE=1

cd ~/lstm_work
source ~/miniconda3/etc/profile.d/conda.sh
conda activate k-dev

LOG=~/lstm_work/pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "==== pipeline start: $(date -u)  variance=$VARIANCE ===="

run_one () {
  local CANCER=$1 SEED=${2:-42}
  local SUFFIX=""
  [[ "$SEED" != "42" ]] && SUFFIX="_seed${SEED}"

  local args=(
    --cancer $CANCER --variant pipeline_v1$SUFFIX
    --source-notebook source_notebooks/${CANCER}_lstm_pipeline_v1.ipynb
    --data-version v45_balanced_min3
    --max-snomed 50
    --min-snomed-seq 3
    --min-code-frequency 5
    --add-relevance-codelist data/codelists/${CANCER}_cancer_relevance/${CANCER}_curated_workup.csv
    --seed $SEED --split-seed $SEED
  )
  [[ "$CANCER" == "lung" ]] && args+=(--use-med-branch --min-med-seq 1)

  echo
  echo "==== ${CANCER} pipeline_v1 seed=${SEED} ===="
  python run_notebook.py "${args[@]}" \
    || echo "  ${CANCER} seed=${SEED} FAILED — continuing"
}

if [[ $VARIANCE -eq 1 ]]; then
  SEEDS="42 13 99 7 2024"
else
  SEEDS="42"
fi

for CANCER in prostate lung; do
  for SEED in $SEEDS; do
    run_one $CANCER $SEED
  done
done

echo
echo "==== pipeline DONE: $(date -u) ===="
echo PIPELINE_DONE > ~/lstm_work/pipeline_done.flag
