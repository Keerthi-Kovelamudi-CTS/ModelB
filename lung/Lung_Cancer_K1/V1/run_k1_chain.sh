#!/bin/bash
# Sequential chain (NO parallelism, per request): finish the in-flight 12mo held-out,
# then 1mo held-out, then 1:5 training on 12mo + 1mo (5yr window). One step at a time.
set -e
cd ~/lung_k1/V1
source ~/lungenv/bin/activate

HELDOUT_PID="${1:-}"                       # pid of the already-running 12mo 5yr held-out (optional)
REC=12mo_1to1/lookback/5yr/heldout_recalib_5yr.txt

if [ -n "$HELDOUT_PID" ]; then
  echo "[chain] waiting for in-flight 12mo 5yr held-out (pid $HELDOUT_PID)..."
  while kill -0 "$HELDOUT_PID" 2>/dev/null; do sleep 60; done
  [ -f "$REC" ] || { echo "[chain] ABORT: 12mo held-out produced no $REC"; exit 1; }
  echo "[chain] 12mo 5yr held-out finished:"; cat "$REC"
fi

echo "[chain] === 1mo 5yr held-out (1:1 model) ==="
GAP=1 NC_RATIO=1 WINDOW=5yr python 4_Holdout/evaluate_heldout.py

echo "[chain] === 1:5 training — 12mo 5yr ==="
GAP=12 NC_RATIO=5 python run_lookback_experiment.py 5yr

echo "[chain] === 1:5 training — 1mo 5yr ==="
GAP=1 NC_RATIO=5 python run_lookback_experiment.py 5yr

echo "[chain] ALL DONE"
