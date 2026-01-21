#!/usr/bin/env bash
#
# V2 Tribal Village Contrastive Experiments
#
# Runs baseline vs GC-CRL style contrastive learning on Tribal Village.
# Uses 3 seeds across 4 A100 GPUs.
#

set -euo pipefail

NUM_SEEDS=3
STEPS=100000000
NUM_GPUS=4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/train_dir/v2_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "V2 Tribal Village Contrastive Experiments"
echo "=============================================="
echo "Seeds: $NUM_SEEDS"
echo "Steps: $STEPS"
echo "GPUs: $NUM_GPUS"
echo "Log dir: $LOG_DIR"
echo ""

# Modes to test
declare -a MODES=("baseline" "gc-crl")

# Build job queue
declare -a JOB_QUEUE=()
for mode in "${MODES[@]}"; do
    for seed in $(seq 1 "$NUM_SEEDS"); do
        JOB_QUEUE+=("$mode:$seed")
    done
done

total_jobs=${#JOB_QUEUE[@]}
echo "Total jobs: $total_jobs"
echo ""

# Track GPU PIDs
declare -a GPU_PIDS=("" "" "" "")

job_idx=0

while [[ $job_idx -lt $total_jobs ]]; do
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [[ -z "${GPU_PIDS[$gpu]}" ]] || ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
            if [[ $job_idx -lt $total_jobs ]]; then
                job="${JOB_QUEUE[$job_idx]}"
                mode="${job%:*}"
                seed="${job#*:}"
                log_file="${LOG_DIR}/tribal_${mode}_seed${seed}.log"

                echo "[$((job_idx + 1))/$total_jobs] GPU $gpu: $mode (seed $seed)"

                nohup "$ROOT_DIR/scripts/train_contrastive.sh" \
                    --"$mode" \
                    --seed "$seed" \
                    --gpu "$gpu" \
                    --steps "$STEPS" \
                    > "$log_file" 2>&1 &

                GPU_PIDS[$gpu]=$!
                ((job_idx++))
                sleep 5
            fi
        fi
    done
    sleep 30
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo "Monitor with: tail -f $LOG_DIR/*.log"
echo ""

wait

echo "=============================================="
echo "All Tribal Village experiments completed!"
echo "Logs: $LOG_DIR"
echo "=============================================="
