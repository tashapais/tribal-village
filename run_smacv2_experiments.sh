#!/usr/bin/env bash
# Run 6 SMACv2 experiments in parallel: 2 conditions × 3 seeds
# Uses tribal-village conda env which has torch + smacv2 + sklearn

PYTHON=/Users/tasha/miniconda3/envs/tribal-village/bin/python
SCRIPT=/Users/tasha/Documents/Github/tribal-village/train_smacv2.py
LOGDIR=/Users/tasha/Documents/Github/tribal-village/logs_smacv2
mkdir -p "$LOGDIR"

echo "Launching 6 runs (individual+shared × seeds 0,1,2)..."

for SEED in 0 1 2; do
  for REWARD in individual shared; do
    LOG="$LOGDIR/${REWARD}_seed${SEED}.log"
    echo "  $REWARD seed=$SEED -> $LOG"
    $PYTHON "$SCRIPT" --reward-type "$REWARD" --seed "$SEED" > "$LOG" 2>&1 &
  done
done

echo "All launched. PIDs: $(jobs -p)"
echo "Waiting for all to finish..."
wait
echo "Done. Results in /Users/tasha/Documents/Github/tribal-village/results_smacv2.tsv"
