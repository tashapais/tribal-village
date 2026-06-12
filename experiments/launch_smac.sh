#!/usr/bin/env bash
# Launch SMACv2 reward-geometry runs (Table 2), streaming to per-run logs.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export SC2PATH=${SC2PATH:-/home/tasha/StarCraftII}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs results checkpoints

TOTAL_STEPS=${TOTAL_STEPS:-2000000}
SEEDS=${SEEDS:-"0 1 2"}
PROJECT=${PROJECT:-rl_workshop_2026}
ENTITY=${ENTITY:-tashapais}

pids=()
for rt in individual shared; do
  for seed in $SEEDS; do
    run="smac_${rt}_seed${seed}"
    echo "launching $run -> logs/${run}.log"
    python3 -u experiments/smac_reward_geometry.py \
      --reward_type "$rt" --seed "$seed" --total_steps "$TOTAL_STEPS" \
      --wandb_project "$PROJECT" --wandb_entity "$ENTITY" \
      --run_name "$run" > "logs/${run}.log" 2>&1 &
    pids+=($!)
    sleep 8   # stagger SC2 process startup
  done
done
echo "launched ${#pids[@]} SMAC runs: ${pids[*]}"
wait "${pids[@]}"
echo "ALL SMAC RUNS COMPLETE"
