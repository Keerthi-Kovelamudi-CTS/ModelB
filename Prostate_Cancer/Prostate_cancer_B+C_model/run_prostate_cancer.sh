#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Prostate B+C — full pipeline, uniform EXPORT DATA extract (like EMIS).
# Runs A0 (top-500), A1 (all-clinical), A2 (all+utilization).
# Matrix-check guard: skips modeling if FE produced no training matrix.
# Within-run set -e (a hard failure aborts THIS cancer); the calling
# chain uses set +e so the rest of the rollout continues.
# ════════════════════════════════════════════════════════════════
set -e
ROOT=$HOME/Prostate_cancer_B+C_model
FE=$ROOT/2_Feature_Engineering
MOD=$ROOT/3_Modeling
GCS=gs://cancer_transformed_ai/cancer_runs_v5_bc/prostate
MAP=$FE/codelist2.0/code_category_mapping_2.0.json
L=$HOME/cancer_runs/prostate_bc; mkdir -p $L; rm -f $L/PROSTATE_DONE
WINDOWS=("12mo" "1mo_12mo")

echo "### Prostate: BQ extract (EXPORT DATA) $(date) ###"
for W in "${WINDOWS[@]}"; do
  SQL=$(cat "$FE/SQL/${W}.sql")
  bq query --use_legacy_sql=false --format=none \
    "EXPORT DATA OPTIONS(uri='${GCS}/${W}/raw_*.csv', format='CSV', overwrite=true, header=true) AS ${SQL}" 2>&1 | tail -1
  echo "  ${W}: $(gsutil ls ${GCS}/${W}/ 2>/dev/null | wc -l) shards"
done

echo "### Prostate: pull + preprocess $(date) ###"
cd "$FE"
for W in "${WINDOWS[@]}"; do
  raw=data/raw; mkdir -p "$raw" "data/$W"; rm -f $raw/raw_*.csv
  gsutil -q -m cp "${GCS}/${W}/raw_*.csv" "$raw/"
  final="$raw/prostate_${W}_raw.csv"; first=$(ls $raw/raw_*.csv | head -1)
  head -1 "$first" > "$final"; for s in $raw/raw_*.csv; do tail -n +2 "$s" >> "$final"; done
  rm -f $raw/raw_*.csv
  echo "  [$W] $(wc -l < "$final") rows -> preprocess"
  WINDOW=$W python3 0_preprocess.py --raw "$final" --out-dir "data/$W" --prefix "prostate_$W" --mapping "$MAP" > "$L/pre_$W.log" 2>&1
  rm -f "$final"
done

echo "### Prostate: FE (sanity-split -> clean -> build -> cleanup) $(date) ###"
python3 run_pipeline.py --step all > "$L/fe.log" 2>&1
gsutil -q -m rm -r "${GCS}/" 2>/dev/null && echo "  Prostate GCS shards removed"

echo "### Prostate: matrix-check guard ###"
MISSING=0
for W in "${WINDOWS[@]}"; do
  M="results/5_cleanup/$W/feature_matrix_clean_${W}.parquet"
  if [ -f "$M" ]; then echo "  [$W] matrix OK"; else echo "  [$W] !! MATRIX MISSING — FE driver failed"; MISSING=1; fi
done
if [ "$MISSING" = "1" ]; then echo "ABORT: FE produced no training matrix (see $L/fe.log)"; exit 2; fi

echo "### Prostate: shared split $(date) ###"
cd "$MOD"
python3 _shared_split.py > "$L/split.log" 2>&1

echo "### Prostate: modeling A0 / A1 / A2 $(date) ###"
python3 1_training.py > "$L/A0.log" 2>&1
rm -rf results_A0_top500; cp -r results results_A0_top500; echo "  A0 done"
ALL_CLINICAL=1 python3 1_training.py > "$L/A1.log" 2>&1
rm -rf results_A1_allclinical; cp -r results results_A1_allclinical; echo "  A1 done"
ALL_CLINICAL=1 INCLUDE_UTILIZATION=1 python3 1_training.py > "$L/A2.log" 2>&1
rm -rf results_A2_allutil; cp -r results results_A2_allutil; echo "  A2 done"

touch "$L/PROSTATE_DONE"
echo "### PROSTATE A0/A1/A2 DONE $(date) ###"
