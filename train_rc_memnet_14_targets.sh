#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/tone/complete_package0722

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

export PYTHONPATH=.

PYTHON="${PYTHON:-/home/ustc1958/miniconda3/envs/graph/bin/python}"
CSV="${CSV:-output/0722/data/wu_vowel_segments.paper_regions.csv}"
K_SHOT="${K_SHOT:-4}"
NUM_TASKS="${NUM_TASKS:-100}"
MAX_STEPS="${MAX_STEPS:-3000}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
ABLATION_VARIANT="${ABLATION_VARIANT:-rc_memnet}"
BATCH_ROOT="${BATCH_ROOT:-output/0722/main_table_rc_memnet_14targets_$(date +%Y%m%d_%H%M%S)}"
export BATCH_ROOT

mkdir -p "${BATCH_ROOT}"

echo "method=RC-MemNet"
echo "batch_root=${BATCH_ROOT}"
echo "k_shot=${K_SHOT}"
echo "num_tasks=${NUM_TASKS}"
echo "max_steps=${MAX_STEPS}"

while IFS='|' read -r STEM LABEL COLUMN HOLDOUT; do
  [[ -z "${STEM}" ]] && continue
  OUT_DIR="${BATCH_ROOT}/${STEM}"
  mkdir -p "${OUT_DIR}"
  echo "[run] ${LABEL} -> ${OUT_DIR}"
  TARGET_REGION="${HOLDOUT}"
  TARGET_COLUMN="${COLUMN}"
  OUT_DIR="${OUT_DIR}"
  MAX_STEPS="${MAX_STEPS}"
  NUM_TASKS="${NUM_TASKS}"
  K_SHOT="${K_SHOT}"
  SEED="${SEED}"
  DEVICE="${DEVICE}"
  "${PYTHON}" -u tools/experiments/run_rc_memnet.py \
    --csv "${CSV}" \
    --output "${OUT_DIR}/result.json" \
    --holdout-region "${TARGET_REGION}" \
    --region-column "${TARGET_COLUMN}" \
    --ablation-variant "${ABLATION_VARIANT}" \
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
    2>&1 | tee "${OUT_DIR}/run.log"
done <<'EOF'
01_Dangtu|01当涂|region|01当涂
02_Wuhu|02芜湖|region|02芜湖
03_Chizhou|03池州|region|03池州
04_Qingyang|04青阳|region|04青阳
05_Suncun|05孙村|region|05孙村
06_Tongling|06铜陵|region|06铜陵
07_Xuancheng|07宣城|region|07宣城
08_Jingxian|08泾县|region|08泾县
09_Fanchang|09繁昌|region|09繁昌
10_Nanling|10南陵|region|10南陵
11_Huangshan|11黄山|region|11黄山
12_Ningguo|12宁国|site|12宁国
13_Gaochun|13高淳|region|13高淳
14_Lishui|14溧水|site|14溧水
EOF

"${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["BATCH_ROOT"])
rows = []
for stem, label in [
    ("01_Dangtu", "01当涂"),
    ("02_Wuhu", "02芜湖"),
    ("03_Chizhou", "03池州"),
    ("04_Qingyang", "04青阳"),
    ("05_Suncun", "05孙村"),
    ("06_Tongling", "06铜陵"),
    ("07_Xuancheng", "07宣城"),
    ("08_Jingxian", "08泾县"),
    ("09_Fanchang", "09繁昌"),
    ("10_Nanling", "10南陵"),
    ("11_Huangshan", "11黄山"),
    ("12_Ningguo", "12宁国"),
    ("13_Gaochun", "13高淳"),
    ("14_Lishui", "14溧水"),
]:
    result = root / stem / "result.json"
    row = {"stem": stem, "Target": label, "result_file": str(result)}
    if result.exists():
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            val_metrics = data.get("val_metrics", {})
            row.update({
                "status": "done",
                "accuracy": metrics.get("accuracy", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "weighted_f1": metrics.get("weighted_f1", ""),
                "val_accuracy": val_metrics.get("accuracy", ""),
                "val_macro_f1": val_metrics.get("macro_f1", ""),
            })
        except Exception as exc:
            row["status"] = f"error: {exc}"
    else:
        row["status"] = "missing"
    rows.append(row)

summary_csv = root / "main_table_14_summary.csv"
summary_json = root / "main_table_14_summary.json"
with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader()
    writer.writerows(rows)
summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json)}, ensure_ascii=False))
PY
