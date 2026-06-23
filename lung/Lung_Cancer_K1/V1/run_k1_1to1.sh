#!/bin/bash
# 1:1 ONLY (12mo + 1mo), 5yr, NO Optuna tuning (TUNE=0) — fast results with the new problem_comment +
# comment/global FE. train -> rebuild held-out FE -> held-out; then the dual eval.
set -e
cd ~/lung_k1/V1
source ~/lungenv/bin/activate
HO=4_Holdout; [ -d 4_Heldout ] && HO=4_Heldout
export TUNE=0                                  # skip the slow 100-trial Optuna step

for g in 12 1; do
  echo "[1to1] === train ${g}mo 1:1 5yr (no tuning) ==="
  GAP=$g NC_RATIO=1 python run_lookback_experiment.py 5yr
  rm -f ${g}mo_1to1/lookback/5yr/heldout_features_5yr.csv   # rebuild held-out FE with new SQL/FE
  echo "[1to1] === held-out ${g}mo 1:1 5yr ==="
  GAP=$g NC_RATIO=1 WINDOW=5yr python $HO/evaluate_heldout.py
done

echo "[1to1] === dual 1:1 (12mo + 1mo) ==="
NC_RATIO=1 WINDOW=5yr python $HO/evaluate_dual_horizon.py
echo "[1to1] ALL DONE"
