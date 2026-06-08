#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Breast B+C — A0 only (top-500). Uniform EXPORT DATA extract.
# Matrix-check guard before modeling. Within-run set -e.
# ════════════════════════════════════════════════════════════════
set -e
ROOT=$HOME/Breast_cancer_B+C_model
FE=$ROOT/2_Feature_Engineering
MOD=$ROOT/3_Modeling
GCS=gs://cancer_transformed_ai/cancer_runs_v5_bc/breast
MAP=$ROOT/codelist2.0/code_category_mapping_2.0.json
L=$HOME/cancer_runs/breast_bc; mkdir -p $L; rm -f $L/BREAST_DONE
WINDOWS=("12mo" "1mo")

echo "### Breast: BQ extract (EXPORT DATA) $(date) ###"
for W in "${WINDOWS[@]}"; do
  SQL=$(cat "$FE/SQL/${W}.sql")
  bq query --use_legacy_sql=false --format=none \
    "EXPORT DATA OPTIONS(uri='${GCS}/${W}/raw_*.csv', format='CSV', overwrite=true, header=true) AS ${SQL}" 2>&1 | tail -1
  echo "  ${W}: $(gsutil ls ${GCS}/${W}/ 2>/dev/null | wc -l) shards"
done

echo "### Breast: pull + preprocess $(date) ###"
cd "$FE"
for W in "${WINDOWS[@]}"; do
  raw=data/raw; mkdir -p "$raw" "data/$W"; rm -f $raw/raw_*.csv
  gsutil -q -m cp "${GCS}/${W}/raw_*.csv" "$raw/"
  final="$raw/breast_${W}_raw.csv"; first=$(ls $raw/raw_*.csv | head -1)
  head -1 "$first" > "$final"; for s in $raw/raw_*.csv; do tail -n +2 "$s" >> "$final"; done
  rm -f $raw/raw_*.csv
  echo "  [$W] $(wc -l < "$final") rows -> preprocess"
  WINDOW=$W python3 0_preprocess.py --raw "$final" --out-dir "data/$W" --prefix "breast_$W" --mapping "$MAP" > "$L/pre_$W.log" 2>&1
  rm -f "$final"
done

echo "### Breast: FE (sanity-split -> clean -> build -> cleanup) $(date) ###"
python3 run_pipeline.py --step all > "$L/fe.log" 2>&1
gsutil -q -m rm -r "${GCS}/" 2>/dev/null && echo "  Breast GCS shards removed"

echo "### Breast: matrix-check guard ###"
MISSING=0
for W in "${WINDOWS[@]}"; do
  M="results/5_cleanup/$W/feature_matrix_clean_${W}.parquet"
  if [ -f "$M" ]; then echo "  [$W] matrix OK"; else echo "  [$W] !! MATRIX MISSING"; MISSING=1; fi
done
if [ "$MISSING" = "1" ]; then echo "ABORT: Breast FE produced no matrix (see $L/fe.log)"; exit 2; fi

echo "### Breast: shared split + A0 $(date) ###"
cd "$MOD"
python3 _shared_split.py > "$L/split.log" 2>&1
python3 1_training.py > "$L/A0.log" 2>&1
rm -rf results_A0_top500; cp -r results results_A0_top500

touch "$L/BREAST_DONE"
echo "### BREAST A0 DONE $(date) ###"
