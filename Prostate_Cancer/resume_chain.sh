#!/bin/bash
# Resume chain: FE (skip cached) → cleanup → text → modeling → SHAP (all 6 windows)
set -e

BASE_DIR="$HOME/pipeline/Prostate_Cancer"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/resume_chain_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"; }

log "════════════════════════════════════════════════════"
log "  PROSTATE — RESUME CHAIN (FE → modeling → SHAP)"
log "  Started: $(date)"
log "════════════════════════════════════════════════════"

FE_DIR="$BASE_DIR/2_Feature_Engineering"
cd "$FE_DIR"

log "Step FE (skip cached 1mo/2mo/3mo/6mo, run 9mo/12mo)..."
python3 0_run_pipeline.py --step fe 2>&1 | tee -a "$MAIN_LOG"
log "FE DONE"

log "Step cleanup..."
python3 0_run_pipeline.py --step cleanup 2>&1 | tee -a "$MAIN_LOG"
log "Cleanup DONE"

log "Step text..."
python3 0_run_pipeline.py --step text 2>&1 | tee -a "$MAIN_LOG"
log "Text DONE"

MODEL_DIR="$BASE_DIR/3_Modeling"
cd "$MODEL_DIR"
log "Modeling (Optuna + ensemble, all windows)..."
python3 1_run_modeling.py 2>&1 | tee -a "$MAIN_LOG"
log "Modeling DONE"

EXPLAIN_DIR="$BASE_DIR/6_Explainability"
cd "$EXPLAIN_DIR"
for window in 1mo 2mo 3mo 6mo 9mo 12mo; do
    log "[$window] SHAP analysis..."
    python3 1_explain_predictions.py --window $window 2>&1 | tee -a "$MAIN_LOG" || log "[$window] SHAP FAILED — continuing"
    log "[$window] Audit features..."
    python3 2_audit_features.py --window $window 2>&1 | tee -a "$MAIN_LOG" || log "[$window] AUDIT FAILED — continuing"
    log "[$window] Enhancement loop..."
    python3 3_enhancement_loop.py --window $window 2>&1 | tee -a "$MAIN_LOG" || log "[$window] ENHANCE FAILED — continuing"
done

log "════════════════════════════════════════════════════"
log "  ALL DONE — finished $(date)"
log "  Log: $MAIN_LOG"
log "════════════════════════════════════════════════════"
