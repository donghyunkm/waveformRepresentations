#!/bin/bash
#
# Submit exactly one processing stage. This script never chains later stages.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA}"
SOURCE_ROOT="${SOURCE_ROOT:-/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/manifests/waveform_manifest.csv}"
RECORDS_PER_TASK="${RECORDS_PER_TASK:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
MAX_ARRAY_SIZE=5001
STAGE="${1:-}"

usage() {
    echo "Usage: $0 {manifest|convert|summarize|labels|merge}"
    echo
    echo "Optional environment variables:"
    echo "  SOURCE_ROOT, OUTPUT_ROOT, MANIFEST"
    echo "  RECORDS_PER_TASK (default: 10)"
    echo "  MAX_CONCURRENT (default: 32)"
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got: $value" >&2
        exit 2
    fi
}

submit_array() {
    local script="$1"
    if [[ ! -f "$MANIFEST" ]]; then
        echo "Manifest does not exist: $MANIFEST" >&2
        exit 2
    fi
    require_positive_integer RECORDS_PER_TASK "$RECORDS_PER_TASK"
    require_positive_integer MAX_CONCURRENT "$MAX_CONCURRENT"

    local records
    local tasks
    local last_task
    records="$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$MANIFEST")"
    if [[ "$records" -lt 1 ]]; then
        echo "Manifest contains no records: $MANIFEST" >&2
        exit 2
    fi
    tasks="$(( (records + RECORDS_PER_TASK - 1) / RECORDS_PER_TASK ))"
    if [[ "$tasks" -gt "$MAX_ARRAY_SIZE" ]]; then
        echo "Array requires $tasks tasks, above MaxArraySize=$MAX_ARRAY_SIZE." >&2
        echo "Increase RECORDS_PER_TASK and retry." >&2
        exit 2
    fi
    last_task="$((tasks - 1))"
    echo "Submitting $records records as $tasks tasks; concurrency cap $MAX_CONCURRENT"
    sbatch \
        --array="0-${last_task}%${MAX_CONCURRENT}" \
        --export="ALL,SOURCE_ROOT=${SOURCE_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},MANIFEST=${MANIFEST},RECORDS_PER_TASK=${RECORDS_PER_TASK}" \
        "$script"
}

case "$STAGE" in
    manifest|convert|summarize|labels|merge)
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

mkdir -p \
    "${OUTPUT_ROOT}/logs/slurm" \
    "${OUTPUT_ROOT}/manifests" \
    "${OUTPUT_ROOT}/labels/minute_shards"

case "$STAGE" in
    manifest)
        sbatch \
            --export="ALL,SOURCE_ROOT=${SOURCE_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},MANIFEST=${MANIFEST}" \
            "${SCRIPT_DIR}/build_manifest.sbatch"
        ;;
    convert)
        submit_array "${SCRIPT_DIR}/convert_array.sbatch"
        ;;
    summarize)
        sbatch \
            --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},MANIFEST=${MANIFEST},RECORDS_PER_TASK=${RECORDS_PER_TASK}" \
            "${SCRIPT_DIR}/summarize_conversion.sbatch"
        ;;
    labels)
        submit_array "${SCRIPT_DIR}/extract_labels_array.sbatch"
        ;;
    merge)
        sbatch \
            --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},MANIFEST=${MANIFEST},RECORDS_PER_TASK=${RECORDS_PER_TASK}" \
            "${SCRIPT_DIR}/merge_labels.sbatch"
        ;;
esac
