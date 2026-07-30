#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_CMD="${PYTHON_CMD:-/home/ustc1958/miniconda3/envs/graph/bin/python}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-1}"
K_SHOT="${K_SHOT:-4}"
NUM_TASKS="${NUM_TASKS:-100}"
MAX_STEPS="${MAX_STEPS:-3000}"
RUN_NUM_PROMPTS="${RUN_NUM_PROMPTS:-1}"
OUT_DIR="${OUT_DIR:-output/0614/hyperparam_sensitivity_8targets_k4_complete_$(date +%Y%m%d_%H%M%S)}"

SWEEPS=(lambda_global temperature episode_length)
if [[ "$RUN_NUM_PROMPTS" == "1" ]]; then
  SWEEPS=(num_prompts "${SWEEPS[@]}")
fi

echo "[INFO] root: $ROOT"
echo "[INFO] output: $OUT_DIR"
echo "[INFO] sweeps: ${SWEEPS[*]}"
echo "[INFO] device=$DEVICE workers=$WORKERS k=$K_SHOT tasks=$NUM_TASKS max_steps=$MAX_STEPS"

"$PYTHON_CMD" -u tools/experiments/hyperparam_sensitivity_8targets_k4.py \
  --output-dir "$OUT_DIR" \
  --python "$PYTHON_CMD" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --k-shot "$K_SHOT" \
  --num-tasks "$NUM_TASKS" \
  --max-steps "$MAX_STEPS" \
  --sweeps "${SWEEPS[@]}" \
  --base-num-prompts 8 \
  --base-lambda-global 0.5 \
  --base-temperature 0.5 \
  --base-episode-length 32 \
  --num-prompts-values 2 4 6 8 10 \
  --lambda-global-values 0.1 0.3 0.5 0.7 \
  --temperature-values 0.1 0.3 0.5 0.7 \
  --episode-length-values 16 32 48

echo "[DONE] summary: $ROOT/$OUT_DIR/summary.csv"
echo "[DONE] across-area summary: $ROOT/$OUT_DIR/summary_across_8areas.csv"
echo "[DONE] prompt route curves: $ROOT/$OUT_DIR/prompt_route_weights_by_iteration.csv"
