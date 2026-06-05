#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Pull the small modeling artifacts from the VM after a run finishes.
#
# Pulled (lightweight — for local inspection / compare):
#   - 3_Modeling/results/final_results.csv
#   - 3_Modeling/results/modeling.log
#   - 3_Modeling/results/1_training/<window>/feature_importances_<window>.csv
#   - 3_Modeling/results/1_training/<window>/selected_features_<window>.csv
#
# Does NOT pull saved_models/ or *.parquet (huge — stay on VM).
# ════════════════════════════════════════════════════════════════

set -euo pipefail

VM="${VM:-cts-ai-dev-cpu-01}"
VM_ZONE="${VM_ZONE:-us-central1-f}"
VM_USER="${VM_USER:-keerthikovelamudi_cthesigns_com}"
VM_PATH="${VM_PATH:-~/Prostate_2.0_1to1}"
WINDOW="${WINDOW:-12mo}"

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${VM_USER}@${VM}"

echo "════════════════════════════════════════════════════════════════"
echo "  Pulling results from VM"
echo "════════════════════════════════════════════════════════════════"

scp_from_vm () {
  local src="$1"
  local dst="$2"
  echo
  echo "  ↓ ${src##*/}  →  ${dst#$LOCAL_ROOT/}"
  gcloud compute scp \
    --zone="$VM_ZONE" \
    "${TARGET}:${VM_PATH}/${src}" \
    "${dst}" 2>&1 | tail -1 || echo "    (file missing or not yet produced)"
}

local_dir="$LOCAL_ROOT/3_Modeling/results"
mkdir -p "$local_dir/1_training/${WINDOW}"

scp_from_vm "3_Modeling/results/final_results.csv" \
            "$local_dir/final_results.csv"
scp_from_vm "3_Modeling/results/modeling.log" \
            "$local_dir/modeling.log"
scp_from_vm "3_Modeling/results/1_training/${WINDOW}/feature_importances_${WINDOW}.csv" \
            "$local_dir/1_training/${WINDOW}/feature_importances_${WINDOW}.csv"
scp_from_vm "3_Modeling/results/1_training/${WINDOW}/selected_features_${WINDOW}.csv" \
            "$local_dir/1_training/${WINDOW}/selected_features_${WINDOW}.csv"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Pull complete → $local_dir"
echo "════════════════════════════════════════════════════════════════"
