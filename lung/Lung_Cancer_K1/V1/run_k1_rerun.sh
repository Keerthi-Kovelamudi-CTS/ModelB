#!/bin/bash
# Full re-run after the xpoll leakage fix (xpoll now fit on TRAIN only). SQL unchanged -> reuses
# cached cohort events. For each ratio x horizon: retrain (5yr) -> rebuild held-out FE with the new
# train xpoll ref -> single-horizon held-out; then the dual eval per ratio. Sequential.
set -e
cd ~/lung_k1/V1
source ~/lungenv/bin/activate
HO=4_Holdout; [ -d 4_Heldout ] && HO=4_Heldout

for r in 1 5 10; do
  for g in 12 1; do
    echo "[rerun] === train ${g}mo 1:${r} 5yr (xpoll-fixed) ==="
    GAP=$g NC_RATIO=$r python run_lookback_experiment.py 5yr
    # force held-out FE rebuild so it uses the NEW train-fit xpoll reference
    rm -f ${g}mo_1to${r}/lookback/5yr/heldout_features_5yr.csv
    echo "[rerun] === held-out ${g}mo 1:${r} 5yr ==="
    GAP=$g NC_RATIO=$r WINDOW=5yr python $HO/evaluate_heldout.py
  done
  echo "[rerun] === dual 1:${r} ==="
  NC_RATIO=$r WINDOW=5yr python $HO/evaluate_dual_horizon.py
done
echo "[rerun] ALL DONE"
