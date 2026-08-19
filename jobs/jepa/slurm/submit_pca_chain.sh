#!/bin/bash
# Submit PCA hypotension prediction pipeline with dependency chaining.
#
# Chain structure:
#   1. fit_pca (one-shot, no resume needed)
#   2a. train_pca_feature × N_CHAIN (each resumes from latest checkpoint)
#   2b. train_pca_temporal × N_CHAIN (each resumes from latest checkpoint)
#
# Usage:
#   bash submit_pca_chain.sh            # default: 3 links per training chain
#   bash submit_pca_chain.sh 5          # 5 links (5×12h = 60h capacity)
#   bash submit_pca_chain.sh 3 feature  # only feature variant
#   bash submit_pca_chain.sh 3 temporal # only temporal variant
#
# Each training job resumes from the latest checkpoint in its run directory.
# If a job finishes all 20 epochs, subsequent chain links exit immediately
# (the trainer detects max_epochs reached).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_CHAIN="${1:-3}"
VARIANT="${2:-both}"  # "both", "feature", or "temporal"

echo "=== PCA Hypotension Chain Submission ==="
echo "  Chain length: ${N_CHAIN} (${N_CHAIN}×12h = $((N_CHAIN*12))h capacity per variant)"
echo "  Variant: ${VARIANT}"
echo "  Script dir: ${SCRIPT_DIR}"
echo ""

# --- Step 1: Submit fit_pca ---
FIT_JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/fit_pca.sbatch")
echo "Submitted fit_pca: job ${FIT_JOB_ID}"

# --- Step 2a: Feature PCA training chain ---
if [[ "${VARIANT}" == "both" || "${VARIANT}" == "feature" ]]; then
    PREV_ID="${FIT_JOB_ID}"
    echo ""
    echo "Feature PCA chain (${N_CHAIN} links):"
    for i in $(seq 1 "${N_CHAIN}"); do
        JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_ID} \
            "${SCRIPT_DIR}/train_hypotension_pca_feature.sbatch")
        echo "  Link ${i}/${N_CHAIN}: job ${JOB_ID} (after ${PREV_ID})"
        PREV_ID="${JOB_ID}"
    done
fi

# --- Step 2b: Temporal PCA training chain ---
if [[ "${VARIANT}" == "both" || "${VARIANT}" == "temporal" ]]; then
    PREV_ID="${FIT_JOB_ID}"
    echo ""
    echo "Temporal PCA chain (${N_CHAIN} links):"
    for i in $(seq 1 "${N_CHAIN}"); do
        JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_ID} \
            "${SCRIPT_DIR}/train_hypotension_pca_temporal.sbatch")
        echo "  Link ${i}/${N_CHAIN}: job ${JOB_ID} (after ${PREV_ID})"
        PREV_ID="${JOB_ID}"
    done
fi

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER --format='%j %i %P %T %M %l %r'"
