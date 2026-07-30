#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/tone/complete_package0722
source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

models=(
  wav2vec2-base
  hubert-base-ls960
  wavlm-base
  whisper-base
  wav2vec2-large-robust-hfcache
  wav2vec2-xls-r-300m-hfcache
  mHuBERT-147
  MR-HuBERT
  MS-HuBERT
  allophant-hierarchical-hfcache
)

targets=(
  Qingyang
  Tongling
  Jingxian
  Nanling
  Ningguo
  Lishui
  Chizhou
  Huangshan
)

output_dir="${1:-output/0722/ssl_finetune_paper_models_8targets}"
mkdir -p "${output_dir}/logs"

python -u tools/experiments/ssl_finetune_source_then_4shot.py \
  --csv output/0722/data/wu_vowel_segments.paper_regions.csv \
  --output-dir "${output_dir}" \
  --models "${models[@]}" \
  --targets "${targets[@]}" \
  --device cuda \
  --train-scheme last_layer \
  --max-steps 1500 \
  --eval-every 150 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --k 4 \
  --splits 100 \
  2>&1 | tee "${output_dir}/logs/run.log"
