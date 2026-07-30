#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/tone/complete_package0722

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

export PYTHONPATH=.

BASE_SCRIPT="./train_rc_memnet_14_targets.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV="${CSV:-output/0722/data/wu_vowel_segments.paper_regions.csv}"
K_SHOT="${K_SHOT:-4}"
NUM_TASKS="${NUM_TASKS:-100}"
MAX_STEPS="${MAX_STEPS:-3000}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
PROMPT_VARIANT="${PROMPT_VARIANT:-no_prompt_routing}"
MEMORY_VARIANT="${MEMORY_VARIANT:-count_based_adaptation}"
PROMPT_ROOT="${PROMPT_ROOT:-output/0722/ablation_prompt_${TIMESTAMP}}"
MEMORY_ROOT="${MEMORY_ROOT:-output/0722/ablation_memory_${TIMESTAMP}}"

echo "prompt_variant=${PROMPT_VARIANT}"
echo "memory_variant=${MEMORY_VARIANT}"
echo "prompt_root=${PROMPT_ROOT}"
echo "memory_root=${MEMORY_ROOT}"

CSV="${CSV}" K_SHOT="${K_SHOT}" NUM_TASKS="${NUM_TASKS}" MAX_STEPS="${MAX_STEPS}" SEED="${SEED}" DEVICE="${DEVICE}" ABLATION_VARIANT="${PROMPT_VARIANT}" BATCH_ROOT="${PROMPT_ROOT}" "${BASE_SCRIPT}"
CSV="${CSV}" K_SHOT="${K_SHOT}" NUM_TASKS="${NUM_TASKS}" MAX_STEPS="${MAX_STEPS}" SEED="${SEED}" DEVICE="${DEVICE}" ABLATION_VARIANT="${MEMORY_VARIANT}" BATCH_ROOT="${MEMORY_ROOT}" "${BASE_SCRIPT}"
