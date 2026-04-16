#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# PROSTATE CANCER — FULL PIPELINE (GCloud VM)
# Runs Phase 2 → 3 → 6 in sequence with logging
# ═══════════════════════════════════════════════════════════════

set -e  # stop on first error

BASE_DIR="$HOME/pipeline/Prostate_Cancer"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/full_pipeline_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "═══════════════════════════════════════════════════════════"
log "  PROSTATE CANCER — FULL PIPELINE"
log "  Started: $(date)"
log "  VM: $(hostname)"
log "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
log "═══════════════════════════════════════════════════════════"

# ─── Check Python dependencies ────────────────────────────────
log ""
log "Checking dependencies..."
pip install --quiet pandas numpy scikit-learn xgboost lightgbm catboost optuna joblib sentence-transformers shap 2>&1 | tee -a "$MAIN_LOG"
log "Dependencies OK"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
FE_DIR="$BASE_DIR/2_Feature_Engineering"
cd "$FE_DIR"

log ""
log "═══════════════════════════════════════════════════════════"
log "  PHASE 2: FEATURE ENGINEERING"
log "═══════════════════════════════════════════════════════════"

# Step 1: Sanity check
log ""
log "  Step 1: Sanity check + patient overlap fix..."
python3 0_run_pipeline.py --step sanity 2>&1 | tee -a "$MAIN_LOG"
log "  Step 1 DONE"

# Step 2: Clean data
log ""
log "  Step 2: Lab range clamping + data cleaning..."
python3 0_run_pipeline.py --step clean 2>&1 | tee -a "$MAIN_LOG"
log "  Step 2 DONE"

# Step 3: Feature engineering
log ""
log "  Step 3: Feature engineering (this takes the longest)..."
python3 0_run_pipeline.py --step fe 2>&1 | tee -a "$MAIN_LOG"
log "  Step 3 DONE"

# Step 4: Cleanup
log ""
log "  Step 4: Feature cleanup..."
python3 0_run_pipeline.py --step cleanup 2>&1 | tee -a "$MAIN_LOG"
log "  Step 4 DONE"

# Step 5: Text features
log ""
log "  Step 5: Text features (keywords + TF-IDF + BERT)..."
python3 0_run_pipeline.py --step text 2>&1 | tee -a "$MAIN_LOG"
log "  Step 5 DONE"

log ""
log "  PHASE 2 COMPLETE"

# ═══════════════════════════════════════════════════════════════
# PHASE 3: MODELING
# ═══════════════════════════════════════════════════════════════
MODEL_DIR="$BASE_DIR/3_Modeling"
cd "$MODEL_DIR"

log ""
log "═══════════════════════════════════════════════════════════"
log "  PHASE 3: MODELING (Optuna tuning + ensemble)"
log "═══════════════════════════════════════════════════════════"

python3 1_run_modeling.py 2>&1 | tee -a "$MAIN_LOG"

log ""
log "  PHASE 3 COMPLETE"

# ═══════════════════════════════════════════════════════════════
# PHASE 6: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
EXPLAIN_DIR="$BASE_DIR/6_Explainability"
cd "$EXPLAIN_DIR"

log ""
log "═══════════════════════════════════════════════════════════"
log "  PHASE 6: EXPLAINABILITY"
log "═══════════════════════════════════════════════════════════"

for window in 12mo 6mo 3mo; do
    log ""
    log "  ${window}: SHAP analysis..."
    python3 1_explain_predictions.py --window $window 2>&1 | tee -a "$MAIN_LOG"

    log "  ${window}: Audit features..."
    python3 2_audit_features.py --window $window 2>&1 | tee -a "$MAIN_LOG"

    log "  ${window}: Enhancement loop..."
    python3 3_enhancement_loop.py --window $window 2>&1 | tee -a "$MAIN_LOG"
done

log ""
log "  PHASE 6 COMPLETE"

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════"
log "  ALL PHASES COMPLETE"
log "  Finished: $(date)"
log "  Log: $MAIN_LOG"
log "═══════════════════════════════════════════════════════════"

# Show key results
log ""
log "  KEY RESULTS:"
if [ -f "$MODEL_DIR/results/final_results.csv" ]; then
    log "  Modeling results:"
    cat "$MODEL_DIR/results/final_results.csv" | tee -a "$MAIN_LOG"
fi

for window in 12mo 6mo 3mo; do
    summary="$EXPLAIN_DIR/results/$window/enhancement_summary_${window}.json"
    if [ -f "$summary" ]; then
        log "  Explainability ($window):"
        cat "$summary" | tee -a "$MAIN_LOG"
    fi
done

log ""
log "  DONE. Download results with:"
log "  gcloud compute scp --recurse cts-ai-dev-gpu-05-a100:~/pipeline/Prostate_Cancer/3_Modeling/results/ ./results_modeling/ --zone us-central1-c --tunnel-through-iap"
log "  gcloud compute scp --recurse cts-ai-dev-gpu-05-a100:~/pipeline/Prostate_Cancer/6_Explainability/results/ ./results_explainability/ --zone us-central1-c --tunnel-through-iap"
