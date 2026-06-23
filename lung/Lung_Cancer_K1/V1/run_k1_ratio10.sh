#!/bin/bash
# 1:10 arm — queued AFTER the 1:5 sequence (sequential, no parallel). Waits on the 1:5 held-out
# waiter pid, then trains 1:10 (12mo + 1mo, 5yr) and runs their held-out evals, one at a time.
set -e
cd ~/lung_k1/V1
source ~/lungenv/bin/activate

WAIT_PID="${1:-}"   # pid of the 1:5 held-out waiter (run_k1_heldout5/holdout5.sh) — last in the current sequence
echo "[r10] waiting for the 1:5 sequence (pid $WAIT_PID) to finish..."
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
echo "[r10] 1:5 sequence done. Starting 1:10."

# held-out folder: VM may still have 4_Holdout (git rename not yet applied here)
HO=4_Holdout; [ -d 4_Heldout ] && HO=4_Heldout

echo "[r10] === 1:10 train 12mo 5yr ==="
GAP=12 NC_RATIO=10 python run_lookback_experiment.py 5yr

echo "[r10] === 1:10 train 1mo 5yr ==="
GAP=1 NC_RATIO=10 python run_lookback_experiment.py 5yr

echo "[r10] === 1:10 held-out 12mo 5yr ==="
GAP=12 NC_RATIO=10 WINDOW=5yr python $HO/evaluate_heldout.py

echo "[r10] === 1:10 held-out 1mo 5yr ==="
GAP=1 NC_RATIO=10 WINDOW=5yr python $HO/evaluate_heldout.py

echo "[r10] ALL DONE"
