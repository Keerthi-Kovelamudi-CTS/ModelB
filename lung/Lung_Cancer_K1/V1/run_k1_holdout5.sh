#!/bin/bash
# Runs the 1:5 held-out evals AFTER the main training chain finishes — sequential, no parallel.
# Waits on the main chain pid (so all 1:5 training is done first), verifies both 1:5 models exist,
# then scores each on the fixed held-out and Platt-recalibrates, one at a time.
set -e
cd ~/lung_k1/V1
source ~/lungenv/bin/activate

CHAIN_PID="${1:-}"
echo "[ho5] waiting for main training chain (pid $CHAIN_PID) to finish ALL 1:5 training..."
if [ -n "$CHAIN_PID" ]; then while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done; fi
echo "[ho5] main chain finished."

M12=12mo_1to5/lookback/5yr/model_5yr_1to5.joblib
M1=1mo_1to5/lookback/5yr/model_5yr_1to5.joblib
[ -f "$M12" ] || { echo "[ho5] ABORT: missing $M12 (1:5 12mo training did not produce a model)"; exit 1; }
[ -f "$M1" ]  || { echo "[ho5] ABORT: missing $M1 (1:5 1mo training did not produce a model)"; exit 1; }

echo "[ho5] === 1:5 held-out — 12mo 5yr ==="
GAP=12 NC_RATIO=5 WINDOW=5yr python 4_Holdout/evaluate_heldout.py

echo "[ho5] === 1:5 held-out — 1mo 5yr ==="
GAP=1 NC_RATIO=5 WINDOW=5yr python 4_Holdout/evaluate_heldout.py

echo "[ho5] ALL DONE"
