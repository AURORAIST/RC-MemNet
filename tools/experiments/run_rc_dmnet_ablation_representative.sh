#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/tone/complete_package0722
source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

variants=(support_prototype single_global multi_global dual_level uniform_prompt_fusion rc_memnet)
targets=("04青阳:qingyang" "08泾县:jingxian" "03池州:chizhou")

for variant in "${variants[@]}"; do
  for item in "${targets[@]}"; do
    target="${item%%:*}"
    slug="${item##*:}"
    outdir="output/0722/rc_memnet_ablation/${variant}"
    mkdir -p "${outdir}" output/0722/rc_memnet_ablation/logs
    echo "[ablation] start ${variant} ${target} $(date '+%F %T')"
    PYTHONPATH=. python -u tools/experiments/run_rc_memnet.py \
      --csv output/0722/data/wu_vowel_segments.paper_regions.csv \
      --output "${outdir}/${slug}.json" \
      --holdout-region "${target}" \
      --region-column paper_region \
      --ablation-variant "${variant}" \
      --max-steps 1500 \
      --eval-every 300 \
      --eval-split-runs 20 \
      --target-adapt-steps 30 \
      --target-support-shots 4 \
      --batch-size 64 \
      --eval-batch-size 256 \
      --hidden-dim 256 \
      --score-dim 128 \
      --prompt-dim 128 \
      --num-slots 4 \
      --num-prompts 8 \
      --device cuda \
      --cache-dir /home/ustc1958/lxy/graph/tone/complete_package0712/output/salmonn_style_whisper_cache_base \
      --aux-cache-dir output/0722/feature_memory_aux_cache \
      --matrix-cache-dir /home/ustc1958/lxy/graph/tone/complete_package0712/output/0712/matrix_cache
    echo "[ablation] done ${variant} ${target} $(date '+%F %T')"
  done
done
