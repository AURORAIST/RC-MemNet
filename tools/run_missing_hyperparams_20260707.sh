#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_CMD="${PYTHON_CMD:-/home/ustc1958/miniconda3/envs/graph/bin/python}"
OUT_DIR="${OUT_DIR:-output/0614/hyperparam_sensitivity_8targets_k4_full_20260703_4sweeps}"
LOG="${LOG:-output/0614/hyperparam_sensitivity_8targets_k4_full_20260703_4sweeps_add_missing_20260707_queued.log}"
GPU_WAIT_USED_MB="${GPU_WAIT_USED_MB:-12000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"

mkdir -p "$(dirname "$LOG")"

echo "[INFO] root=$ROOT"
echo "[INFO] output=$OUT_DIR"
echo "[INFO] log=$LOG"
echo "[INFO] waiting for GPU used memory <= ${GPU_WAIT_USED_MB} MiB"

while true; do
  used="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | head -n 1 \
      | tr -d ' '
  )"
  if [[ -z "${used:-}" ]]; then
    echo "[WARN] nvidia-smi unavailable; retry in ${GPU_POLL_SECONDS}s"
    sleep "$GPU_POLL_SECONDS"
    continue
  fi
  echo "[INFO] $(date '+%F %T') gpu_used=${used}MiB"
  if (( used <= GPU_WAIT_USED_MB )); then
    break
  fi
  sleep "$GPU_POLL_SECONDS"
done

echo "[INFO] starting missing hyperparameter runs at $(date '+%F %T')"

"$PYTHON_CMD" -u tools/experiments/hyperparam_sensitivity_8targets_k4.py \
  --output-dir "$OUT_DIR" \
  --python "$PYTHON_CMD" \
  --device cuda \
  --workers 1 \
  --k-shot 4 \
  --num-tasks 100 \
  --max-steps 3000 \
  --sweeps num_prompts lambda_global temperature episode_length \
  --base-num-prompts 8 \
  --base-lambda-global 0.5 \
  --base-temperature 0.5 \
  --base-episode-length 32 \
  --num-prompts-values 2 4 6 8 10 \
  --lambda-global-values 0.1 0.3 0.5 0.7 0.9 \
  --temperature-values 0.1 0.3 0.5 0.7 0.9 \
  --episode-length-values 16 32 48 64 128 \
  --no-export-prompt-route-curves

echo "[DONE] finished at $(date '+%F %T')"
echo "[DONE] summary: $ROOT/$OUT_DIR/summary.csv"
echo "[DONE] across-area summary: $ROOT/$OUT_DIR/summary_across_8areas.csv"
