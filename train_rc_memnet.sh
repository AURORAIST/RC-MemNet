#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/tone/complete_package0722

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

export PYTHONPATH=.

PYTHON="${PYTHON:-/home/ustc1958/miniconda3/envs/graph/bin/python}"
CSV="${CSV:-output/0722/data/wu_vowel_segments.paper_regions.csv}"
TARGET_REGION="${TARGET_REGION:-04青阳}"
TARGET_COLUMN="${TARGET_COLUMN:-paper_region}"
K_SHOT="${K_SHOT:-4}"
NUM_TASKS="${NUM_TASKS:-20}"
MAX_STEPS="${MAX_STEPS:-1500}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
TIME_TAG="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-output/0722/rc_memnet_${TARGET_REGION}_${TIME_TAG}}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

echo "method=RC-MemNet"
echo "target_region=${TARGET_REGION}"
echo "target_column=${TARGET_COLUMN}"
echo "k_shot=${K_SHOT}"
echo "num_tasks=${NUM_TASKS}"
echo "max_steps=${MAX_STEPS}"
echo "out_dir=${OUT_DIR}"

"${PYTHON}" -u tools/experiments/run_rc_memnet.py \
  --csv "${CSV}" \
  --output "${OUT_DIR}/result.json" \
  --holdout-region "${TARGET_REGION}" \
  --region-column "${TARGET_COLUMN}" \
  --ablation-variant rc_memnet \
  --max-steps "${MAX_STEPS}" \
  --eval-every 300 \
  --eval-split-runs "${NUM_TASKS}" \
  --target-adapt-steps 30 \
  --target-support-shots "${K_SHOT}" \
  --batch-size 64 \
  --eval-batch-size 256 \
  --hidden-dim 256 \
  --score-dim 128 \
  --prompt-dim 128 \
  --num-slots 4 \
  --num-prompts 8 \
  --write-top-k 1 \
  --device "${DEVICE}" \
  --save-checkpoint \
  --cache-dir /home/ustc1958/lxy/graph/tone/complete_package0712/output/salmonn_style_whisper_cache_base \
  --aux-cache-dir output/0722/feature_memory_aux_cache \
  --matrix-cache-dir /home/ustc1958/lxy/graph/tone/complete_package0712/output/0712/matrix_cache \
  --curve-output "${OUT_DIR}/curve.csv" \
  --audit-output "${OUT_DIR}/audit.json" \
  --predictions-output "${OUT_DIR}/predictions.csv" \
  2>&1 | tee "${LOG_FILE}"
