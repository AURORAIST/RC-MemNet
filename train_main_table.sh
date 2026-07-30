#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/tone/complete_package0614

export PYTHONPATH=.
export PC_DLCMNET_REQUIRE_CUDA=1

PYTHON="${PYTHON:-/home/ustc1958/miniconda3/envs/graph/bin/python}"
CSV="${CSV:-data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv}"
K_SHOT="${K_SHOT:-4}"
TRAIN_SUPPORT_SHOTS="${TRAIN_SUPPORT_SHOTS:-2}"
NUM_TASKS="${NUM_TASKS:-100}"
MAX_STEPS="${MAX_STEPS:-3000}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-200}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1e-4}"
EARLY_STOP_MIN_STEPS="${EARLY_STOP_MIN_STEPS:-300}"
LR="${LR:-3e-4}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
EPISODE_LENGTH="${EPISODE_LENGTH:-32}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-2}"
CUDA_WAIT_ATTEMPTS="${CUDA_WAIT_ATTEMPTS:-0}"
CUDA_STABLE_SUCCESSES="${CUDA_STABLE_SUCCESSES:-3}"
TARGET_RETRY_ATTEMPTS="${TARGET_RETRY_ATTEMPTS:-0}"
RUN_ROOT="output/0614/main_table_retrain_pcdlcmnet_4shot_$(date +"%Y%m%d_%H%M%S")"

mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/main_table.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "CSV=${CSV}"
echo "K_SHOT=${K_SHOT}"
echo "TRAIN_SUPPORT_SHOTS=${TRAIN_SUPPORT_SHOTS}"
echo "NUM_TASKS=${NUM_TASKS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE}"
echo "EARLY_STOP_MIN_STEPS=${EARLY_STOP_MIN_STEPS}"
echo "LR=${LR}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "EPISODE_LENGTH=${EPISODE_LENGTH}"
echo "EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE}"
echo "CUDA_WAIT_ATTEMPTS=${CUDA_WAIT_ATTEMPTS} (0 means wait forever)"
echo "CUDA_STABLE_SUCCESSES=${CUDA_STABLE_SUCCESSES}"
echo "TARGET_RETRY_ATTEMPTS=${TARGET_RETRY_ATTEMPTS} (0 means retry forever)"
echo "MODE=fresh retrain for a new main-table result set"

wait_for_cuda() {
    if [[ "${DEVICE}" != cuda* ]]; then
        return 0
    fi
    local attempt success_count
    attempt=1
    success_count=0
    while [[ "${CUDA_WAIT_ATTEMPTS}" == "0" || "${attempt}" -le "${CUDA_WAIT_ATTEMPTS}" ]]; do
        if "${PYTHON}" - <<'PY'
import torch
torch.empty(1, device="cuda")
print(torch.cuda.get_device_name(0), flush=True)
PY
        then
            success_count=$((success_count + 1))
            if [[ "${success_count}" -ge "${CUDA_STABLE_SUCCESSES}" ]]; then
                return 0
            fi
            echo "[cuda-preflight] success ${success_count}/${CUDA_STABLE_SUCCESSES}; checking again in 2s"
            sleep 2
            continue
        fi
        success_count=0
        if [[ "${CUDA_WAIT_ATTEMPTS}" == "0" ]]; then
            echo "[cuda-preflight] attempt ${attempt} failed; retrying in 10s"
        else
            echo "[cuda-preflight] attempt ${attempt}/${CUDA_WAIT_ATTEMPTS} failed; retrying in 10s"
        fi
        attempt=$((attempt + 1))
        sleep 10
    done
    echo "[cuda-preflight] CUDA is still unavailable after retries"
    return 1
}

run_target() {
    local area="$1"
    local column="$2"
    local target="$3"
    local out_dir="${RUN_ROOT}/${area}"

    mkdir -p "${out_dir}"
    echo "==== ${area}: target=${target}, column=${column} ===="
    echo "source_regions: all ${column} values except ${target}"
    echo "OUT_DIR=${out_dir}"

    local target_attempt
    target_attempt=1
    while [[ "${TARGET_RETRY_ATTEMPTS}" == "0" || "${target_attempt}" -le "${TARGET_RETRY_ATTEMPTS}" ]]; do
        wait_for_cuda
        if "${PYTHON}" -u tools/experiments/paper_dual_memory.py \
        --csv "${CSV}" \
        --region-column "${column}" \
        --holdout-region "${target}" \
        --device "${DEVICE}" \
        --eval-support-shots "${K_SHOT}" \
        --eval-split-runs "${NUM_TASKS}" \
        --eval-split-mode global_support \
        --support-shots "${TRAIN_SUPPORT_SHOTS}" \
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
        --episode-length "${EPISODE_LENGTH}" \
        --episode-batch-size "${EPISODE_BATCH_SIZE}" \
        --eval-ensemble-runs 1 \
        --log-every 50 \
        --eval-every 50 \
        --save-checkpoint \
        --output "${out_dir}/result.json" \
        --curve-output "${out_dir}/curve.csv" \
        --debug-output "${out_dir}/debug.json" \
        --audit-output "${out_dir}/audit.json" \
        --predictions-output "${out_dir}/predictions.csv"
        then
            return 0
        fi

        if [[ "${TARGET_RETRY_ATTEMPTS}" == "0" ]]; then
            echo "[target-retry] ${area} attempt ${target_attempt} failed; retrying in 20s"
        else
            echo "[target-retry] ${area} attempt ${target_attempt}/${TARGET_RETRY_ATTEMPTS} failed; retrying in 20s"
        fi
        target_attempt=$((target_attempt + 1))
        sleep 20
    done

    echo "[target-retry] ${area} failed after retries"
    return 1
}

wait_for_cuda

run_target "Qingyang" "region" "04青阳"
run_target "Tongling" "region" "06铜陵"
run_target "Jingxian" "region" "08泾县"
run_target "Nanling" "region" "10南陵"
run_target "Ningguo" "site" "12宁国"
run_target "Lishui" "site" "14溧水"

echo "All retrained main-table PC-DLCMNet runs finished."
echo "RUN_ROOT=${RUN_ROOT}"
