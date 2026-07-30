#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ustc1958/lxy/graph/tone/complete_package0712"
cd "$ROOT"

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u tools/experiments/local_hf_ssl_4shot_eval.py \
  --csv output/0712/data/wu_vowel_segments.fixed_paths.csv \
  --output-dir output/0712/hf_cache_3models_4shot_s100 \
  --cache-dir output/0712/hf_cache_3models_feature_cache \
  --device cuda \
  --batch-size 8 \
  --k 4 \
  --splits 100 \
  --models \
    wav2vec2-xls-r-300m-hfcache \
    wav2vec2-large-robust-hfcache \
    allophant-hierarchical-hfcache
