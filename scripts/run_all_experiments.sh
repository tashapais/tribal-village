#!/usr/bin/env bash
#
# Multi-GPU Tribal Village Contrastive Experiments
#
# Runs baseline, auxiliary contrastive, and GC-CRL experiments
# in parallel across 4 A100 GPUs with multiple seeds.
#
# Usage:
#   ./scripts/run_all_experiments.sh [--seeds N] [--steps T]
#

set -euo pipefail

NUM_SEEDS=${1:-5}
STEPS=${2:-100000000}
NUM_GPUS=4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/train_dir/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Tribal Village Multi-GPU Experiments"
echo "=============================================="
echo "Seeds: $NUM_SEEDS"
echo "Steps: $STEPS"
echo "GPUs: $NUM_GPUS"
echo "Log dir: $LOG_DIR"
echo ""

# Experiment modes
declare -a MODES=("baseline" "aux" "gc-crl")

# Build job queue
declare -a JOBS=()
for mode in "${MODES[@]}"; do
    for seed in $(seq 1 "$NUM_SEEDS"); do
        JOBS+=("$mode:$seed")
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# Track GPU assignments
declare -a GPU_PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_PIDS+=("")
done

job_idx=0
total_jobs=${#JOBS[@]}

while [[ $job_idx -lt $total_jobs ]]; do
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        # Check if GPU is free
        if [[ -z "${GPU_PIDS[$gpu]}" ]] || ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
            if [[ $job_idx -lt $total_jobs ]]; then
                job="${JOBS[$job_idx]}"
                mode="${job%:*}"
                seed="${job#*:}"
                log_file="${LOG_DIR}/${mode}_seed${seed}.log"

                echo "[$((job_idx + 1))/$total_jobs] GPU $gpu: $mode seed $seed"

                "$ROOT_DIR/scripts/train_contrastive.sh" \
                    --"$mode" \
                    --seed "$seed" \
                    --gpu "$gpu" \
                    --steps "$STEPS" \
                    > "$log_file" 2>&1 &

                GPU_PIDS[$gpu]=$!
                ((job_idx++))
            fi
        fi
    done
    sleep 60
done

# Wait for remaining jobs
wait

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Logs: $LOG_DIR"
echo "=============================================="
