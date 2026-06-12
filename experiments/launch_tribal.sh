#!/usr/bin/env bash
# Launch all Tribal Village reward-geometry runs concurrently, each streaming
# unbuffered output to its own logfile (tail -f logs/<run>.log to watch live).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nimby/nim/bin:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # avoid thread thrash across concurrent runs
mkdir -p logs results checkpoints

TOTAL_STEPS=${TOTAL_STEPS:-4000000}
SEEDS=${SEEDS:-"0 1 2"}
PROJECT=${PROJECT:-rl_workshop_2026}
ENTITY=${ENTITY:-tashapais}

declare -A FRACS=( [individual]=0.0 [mixed]=0.8 [shared]=1.0 )

pids=()
for cond in individual mixed shared; do
  frac=${FRACS[$cond]}
  for seed in $SEEDS; do
    run="tv_${cond}_seed${seed}"
    log="logs/${run}.log"
    echo "launching $run -> $log"
    python3 -u experiments/tribal_reward_geometry.py \
      --shared_frac "$frac" --seed "$seed" \
      --total_steps "$TOTAL_STEPS" \
      --wandb_project "$PROJECT" --wandb_entity "$ENTITY" \
      --run_name "$run" > "$log" 2>&1 &
    pids+=($!)
    sleep 2   # stagger startup so wandb init / lib load don't collide
  done
done

echo "launched ${#pids[@]} runs: ${pids[*]}"
echo "watch with: tail -f logs/tv_*.log"
wait "${pids[@]}"
echo "ALL TRIBAL RUNS COMPLETE"
