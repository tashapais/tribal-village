#!/usr/bin/env bash
#
# Tribal Village Contrastive Learning Training Script
#
# Trains agents with optional contrastive learning loss.
#
# Usage:
#   ./scripts/train_contrastive.sh [OPTIONS]
#
# Options:
#   --steps N           Total training timesteps (default: 100000000)
#   --seed S            Random seed (default: 1)
#   --gpu G             GPU device ID (default: 0)
#   --contrastive-coef  Contrastive loss coefficient (default: 0.1)
#   --baseline          Run without contrastive loss (baseline)
#   --aux               Use auxiliary contrastive loss (lower coef)
#   --gc-crl            Use goal-conditioned CRL (higher coef)
#

set -euo pipefail

# Defaults
STEPS=100000000
SEED=1
GPU=0
CONTRASTIVE_COEF=0.1
CONTRASTIVE_ENABLED=True
MODE="gc-crl"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --contrastive-coef)
            CONTRASTIVE_COEF="$2"
            shift 2
            ;;
        --baseline)
            CONTRASTIVE_ENABLED=False
            MODE="baseline"
            shift
            ;;
        --aux)
            CONTRASTIVE_COEF=0.001
            MODE="aux"
            shift
            ;;
        --gc-crl)
            CONTRASTIVE_COEF=0.1
            MODE="gc-crl"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METTA_DIR="${METTA_DIR:-$ROOT_DIR/../metta}"

if [ ! -d "$METTA_DIR" ]; then
    echo "Metta repo not found at $METTA_DIR. Set METTA_DIR to the metta checkout." >&2
    exit 1
fi

RUN_NAME="tribal_village_${MODE}_seed${SEED}"
LOG_DIR="${ROOT_DIR}/train_dir/${RUN_NAME}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Tribal Village Contrastive Training"
echo "=============================================="
echo "Mode: $MODE"
echo "Steps: $STEPS"
echo "Seed: $SEED"
echo "GPU: $GPU"
echo "Contrastive coefficient: $CONTRASTIVE_COEF"
echo "Contrastive enabled: $CONTRASTIVE_ENABLED"
echo "Run name: $RUN_NAME"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU
export TRIBAL_VECTOR_BACKEND=multiprocessing

# Resolve symlinks
METTA_DIR="$(cd "$METTA_DIR" && pwd -P)"

cd "$METTA_DIR"

# Run training with contrastive settings
exec uv run python -c "
import sys
import os
sys.path.insert(0, '${ROOT_DIR}')

import torch
from tribal_village_env.cogames.train import train_with_contrastive
from tribal_village_env.cogames.contrastive_trainer import ContrastiveConfig

# Training settings
settings = {
    'steps': ${STEPS},
    'seed': ${SEED},
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'checkpoints_path': '${LOG_DIR}',
    'batch_size': 65536,
    'minibatch_size': 4096,
    'policy_class_path': 'mettagrid.policy.lstm.LSTMPolicy',
    'initial_weights_path': None,
    'vector_num_envs': 4,
    'vector_num_workers': 4,
    'env_config': {
        'max_steps': 1000,
        'heart_reward': 1.0,
        'ore_reward': 0.1,
        'bar_reward': 0.8,
    },
}

# Contrastive config
contrastive_config = ContrastiveConfig(
    enabled=${CONTRASTIVE_ENABLED},
    contrastive_coef=${CONTRASTIVE_COEF},
    hidden_dim=512,
    embed_dim=64,
    logsumexp_coef=0.1,
)

print('Starting training with settings:')
for k, v in settings.items():
    print(f'  {k}: {v}')
print(f'  contrastive_config: {contrastive_config}')
print()

train_with_contrastive(settings, contrastive_config)
"
