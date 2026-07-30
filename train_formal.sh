#!/usr/bin/env bash
set -e

cd /home/ustc1958/lxy/graph/tone/complete_package0614

unset LD_LIBRARY_PATH
export PYTHONPATH=.
export PC_DLCMNET_REQUIRE_CUDA=1

PYTHON="/home/ustc1958/miniconda3/envs/graph/bin/python"
TARGET_REGION="${1:-04青阳}"
TARGET_COLUMN="${2:-region}"
K_SHOT="${K_SHOT:-4}"
NUM_TASKS="${NUM_TASKS:-100}"
MAX_STEPS="${MAX_STEPS:-10000}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1e-4}"
EARLY_STOP_MIN_STEPS="${EARLY_STOP_MIN_STEPS:-0}"
LR="${LR:-3e-4}"
SEED="${SEED:-0}"

CSV="data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv"
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="pc_dlcmnet_4shot_${TARGET_REGION}_${TIME_TAG}"
OUT_DIR="output/0614/${RUN_NAME}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

echo "target_region: ${TARGET_REGION}"
echo "target_column: ${TARGET_COLUMN}"
echo "source_regions: all ${TARGET_COLUMN} values except ${TARGET_REGION}"
echo "k_shot: ${K_SHOT}"
echo "num_tasks: ${NUM_TASKS}"
echo "max_steps: ${MAX_STEPS}"
echo "early_stop_patience: ${EARLY_STOP_PATIENCE}"
echo "output_dir: ${OUT_DIR}"
echo "log_file: ${LOG_FILE}"

wait_for_cuda() {
    local attempt
    for attempt in $(seq 1 30); do
        if "${PYTHON}" - <<'PY'
import torch
torch.cuda.init()
print(torch.cuda.get_device_name(0), flush=True)
PY
        then
            return 0
        fi
        echo "[cuda-preflight] attempt ${attempt}/30 failed; retrying in 10s"
        sleep 10
    done
    echo "[cuda-preflight] CUDA is still unavailable after retries"
    return 1
}

wait_for_cuda

CMD=(
"${PYTHON}" -u tools/experiments/paper_dual_memory.py
    --csv "${CSV}" \
    --region-column "${TARGET_COLUMN}" \
    --holdout-region "${TARGET_REGION}" \
    --device cuda \
    --eval-support-shots "${K_SHOT}" \
    --eval-split-runs "${NUM_TASKS}" \
    --eval-split-mode global_support \
    --support-shots "${K_SHOT}" \
    --max-steps "${MAX_STEPS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
    --early-stop-min-steps "${EARLY_STOP_MIN_STEPS}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --lambda-global 0.5 \
    --hidden-dim 256 \
    --score-dim 128 \
    --prompt-dim 128 \
    --num-prompts 8 \
    --layers 3 \
    --heads 4 \
    --ffn-dim 768 \
    --episode-length 32 \
    --episode-batch-size 2 \
    --eval-ensemble-runs 1 \
    --log-every 50 \
    --eval-every 50 \
    --save-checkpoint \
    --output "${OUT_DIR}/result.json" \
    --curve-output "${OUT_DIR}/curve.csv" \
    --debug-output "${OUT_DIR}/debug.json" \
    --audit-output "${OUT_DIR}/audit.json" \
    --predictions-output "${OUT_DIR}/predictions.csv"
)

if [[ "${RUN_FOREGROUND:-0}" == "1" ]]; then
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
    nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    PID=$!

    echo "Started formal 4-shot run"
    echo "PID: ${PID}"
    echo "Tail: tail -f ${LOG_FILE}"
fi
