数据在/home/ustc1958/lxy/graph/tone/tone/01语音/01语音信号

# 0722 RC-MemNet

This package inherits the 0712/0614 PC-DLCMNet training utilities and adds the
paper-aligned RC-MemNet runner:

- `pc_dlcmnet/models/rc_dmnet.py` now implements RC-MemNet while preserving the
  old import name for compatibility.
- `tools/experiments/run_rc_memnet.py` is the preferred entrypoint.
- `train_rc_memnet.sh` launches a single 4-shot leave-one-region run in the
  `graph` conda environment.

Quick smoke test:

```bash
cd /home/ustc1958/lxy/graph/tone/complete_package0722
conda activate graph
PYTHONPATH=. python -u tools/experiments/run_rc_memnet.py \
  --csv output/0722/data/wu_vowel_segments.paper_regions.csv \
  --output output/0722/smoke_rc_memnet/result.json \
  --holdout-region "04青阳" \
  --region-column paper_region \
  --ablation-variant rc_memnet \
  --max-steps 1 \
  --eval-every 1 \
  --eval-split-runs 1 \
  --target-adapt-steps 1 \
  --target-support-shots 1 \
  --batch-size 8 \
  --eval-batch-size 32 \
  --hidden-dim 64 \
  --score-dim 32 \
  --prompt-dim 32 \
  --num-slots 2 \
  --num-prompts 2 \
  --device cpu \
  --cache-dir /home/ustc1958/lxy/graph/tone/complete_package0712/output/salmonn_style_whisper_cache_base \
  --aux-cache-dir output/0722/feature_memory_aux_cache \
  --matrix-cache-dir /home/ustc1958/lxy/graph/tone/complete_package0712/output/0712/matrix_cache
```

Default training:

```bash
cd /home/ustc1958/lxy/graph/tone/complete_package0722
./train_rc_memnet.sh
```

You can override the main knobs without editing the script:

```bash
TARGET_REGION="08泾县" TARGET_COLUMN=paper_region MAX_STEPS=3000 NUM_TASKS=100 ./train_rc_memnet.sh
```

Run the paper 01-14 target-area table in one sequential batch:

```bash
MAX_STEPS=3000 NUM_TASKS=100 K_SHOT=4 ./train_rc_memnet_14_targets.sh
```

This writes per-target `result.json`, `result.pt`, `predictions.csv`,
`curve.csv`, `audit.json`, and `run.log` files under
`output/0722/main_table_rc_memnet_14targets_*`, plus summary CSV/JSON files at
the batch root.

Run the two module ablations used for the paper table:

```bash
MAX_STEPS=3000 NUM_TASKS=100 K_SHOT=4 ./run_rc_memnet_ablations.sh
```

The default ablations are `no_prompt_routing` for the RPL/prompt side and
`count_based_adaptation` for the RGMA memory-write adapter side.

The RC-MemNet implementation follows the pasted paper draft:

1. RPL regional prompts are concatenated with acoustic tokens and encoded by a
   shared Transformer encoder.
2. Encoded prompt context modulates the memory query with FiLM-style
   `gamma/beta` projections.
3. RGMA uses one shared category-aligned key-value memory across all source
   regions.
4. Source regions follow a region-level read-before-write procedure.
5. Unseen-region adaptation optimizes only source-prompt fusion logits; memory
   values remain read-only.

# PC-DLCMNet for Cross-Regional Wu Vowel Recognition

This project packages the 0614 experiments as a cleaner PC-DLCMNet codebase.
It follows the method outline in `skill.md`: reusable model/data/training code
lives in `pc_dlcmnet/`, runnable experiment and analysis commands live in
`tools/`, and generated artifacts are written under `output/`.

```text
complete_package0614/
├── pc_dlcmnet/
│   ├── data/          # acoustic features and support/query episodes
│   ├── models/        # encoder, GEMA/PCMR memory modules, PC-DLCMNet wrapper
│   ├── training/      # supervised runner and training loop
│   ├── evaluation/    # downstream k-shot/prototype evaluation
│   └── utils/         # path and tensor helpers
├── tools/
│   ├── experiments/   # runnable experiment entrypoints
│   └── analysis/      # tables, plots, diagnostics, result collection
├── configs/           # config placeholders
├── data/              # manifests, raw audio, extracted features
└── output/            # checkpoints, logs, results, tables, figures
```

This folder contains the 0614 shared class-token method plus a recurrent
class-memory variant that adds explicit read, write, and within-episode
persistence.

## Paper-aligned Reproduction

`tools/experiments/paper_dual_memory.py` and `pc_dlcmnet.models.core` provide
a closer implementation of the paper section supplied in this workspace:

1. Multi-source acoustic representation with adaptive mel / MFCC / delta fusion
2. Prompt-router memory reading with prompt-generated attention bias
3. Global class memory and support-driven episode class memory
4. Cosine matching between pooled speech query and class memory
5. Joint training objective `L = L_episode + lambda_g * L_global`

Practical assumptions used where the paper leaves details implicit:

- `Fuse(H_s, H_f)` is a learned gate in hidden space.
- Prompt bias is generated for the cached fixed token length and sliced to the
  active sequence length.
- Episodes default to `speaker_id` when available, otherwise `region`.

Single-run example:

```powershell
python -u tools\experiments\paper_dual_memory.py `
  --holdout-region "01褰撴秱" `
  --device cuda `
  --output output\0614\paper_dual_memory\01dt_paper.json
```

Fast smoke test on real cached data:

```powershell
python -u tools\experiments\paper_dual_memory.py `
  --holdout-region "01褰撴秱" `
  --device cpu `
  --max-steps 1 `
  --episode-batch-size 1 `
  --eval-episode-limit 5 `
  --output output\0614\paper_dual_memory_smoke\01dt_paper_smoke.json
```

## Method

The model has one Transformer computation body.

1. Whisper tokens provide the speech stream `H_s`.
2. Auxiliary acoustic features provide the feature stream `H_f`.
   With wav files available, the runner extracts token-aligned Log-Mel, MFCC, and MFCC Delta/Delta2 sequences.
   If raw wav files are unavailable, the runner uses Whisper-token statistics as an explicit fallback and records this in JSON logs.
3. The feature stream conditions speech tokens:
   `H_0 = LayerNorm(Proj(H_s) + Fuse(H_f))`.
4. Each vowel class has one shared class prototype token:
   `M^0 = [m_1, ..., m_C]`.
5. `M^0` is shared across all source regions and all samples; there is no external retrieval memory and no sample-specific memory residual before the Transformer.
6. The `region_memory_prompt` variant learns source-region acoustic memory tokens `R`.
   The auxiliary acoustic condition softly matches these source-region memories and creates an inferred region token for each sample.
   This token is prepended to the same Transformer and conditions prompt gates, but it does not overwrite `M^0` before the Transformer.
7. The joint sequence enters a single Transformer:
   `Z_i^0 = [M^0; H_i^0]`.
8. Prompt is a learned per-layer Q/K/V attention gate; in `region_memory_prompt`, the gate is conditioned by inferred region memory.
9. Final logits are scored only from final class memory tokens.

The shared-token variants are better described as `Class Prototype Memory`:
each sample starts from the same learnable `M^0`. For strict persistent memory,
use `recurrent_memory_prompt` or `recurrent_region_memory_prompt`.

## Recurrent Class Memory

The recurrent variants implement:

```text
[M_{t-1}; H_t^0] -> [M~_t; H_t^L] -> Predict -> Write -> M_t
```

- Each episode starts from learned global memory `M_global`.
- `M_{t-1}` is prepended to the current speech segment and read by the Transformer.
- The final memory tokens become candidate memory `M~_t`.
- Classification is done before writing.
- A learned gate writes only the selected class memory:

```text
M_t = M_{t-1} + w_t * (1 - G_t) * (M~_t - M_{t-1})
```

During supervised source training, `w_t` is formed from the true vowel label
after prediction. During target inference, `--recurrent-eval-mode frozen`
keeps `M_global` fixed; `--recurrent-eval-mode online` performs unsupervised
online writes only when prediction confidence exceeds
`--recurrent-online-threshold`.

Episodes default to `speaker_id` when available, otherwise `region`.
Override with `--recurrent-episode-column`; chunks are capped by
`--recurrent-episode-length`.

## Training Objective

The default experiment is leave-one-region-out domain generalization.

- Train on all source regions except the held-out target region.
- Use source vowel labels for classification.
- Do not use any target-region labels for training.
- Optionally use source region IDs as an auxiliary training signal on the acoustic condition vector.
- Infer target samples without target labels or target region IDs.

Default main loss:

```text
L = L_cls
```

Optional diagnostic regularizers are exposed by command-line flags, but they are disabled by default.
For `region_memory_prompt`, use `--lambda-region-cls` to supervise acoustic-to-source-region memory matching on source domains.

## Single Run

Before running on a new machine, start from the copied `complete_package0614`
directory and point these paths at your server data if they are not under this
folder:

```powershell
$env:TONE0614_DATASET = "output\datasets\wu_low_resource_vowel_dataset.csv"
$env:TONE0614_AUDIO_ROOT = "audio"
```

Convenience wrappers aligned with `skill.md`:

```bash
python tools/run_all_targets.py ...
python tools/run_ablation.py ...
python tools/run_support_weighting.py ...
python tools/run_hyperparam_sensitivity.py ...
```

```powershell
python -u pc_dlcmnet\training\supervised.py `
  --holdout-region "01褰撴秱" `
  --variant feature_memory_prompt `
  --anchor-kind onehot `
  --aux-source acoustic `
  --aux-representation sequence `
  --audio-root $env:TONE0614_AUDIO_ROOT `
  --train-fraction 1.0 `
  --max-steps 800 `
  --device cuda `
  --output output\0614\feature_memory_single\dt_prompt.json
```

Region-memory candidate:

```powershell
python -u pc_dlcmnet\training\supervised.py `
  --holdout-region "01褰撴秱" `
  --variant region_memory_prompt `
  --anchor-kind onehot `
  --aux-source acoustic `
  --aux-representation sequence `
  --audio-root $env:TONE0614_AUDIO_ROOT `
  --lambda-region-cls 0.1 `
  --lambda-region-compact 0.01 `
  --train-fraction 1.0 `
  --max-steps 800 `
  --device cuda `
  --output output\0614\feature_memory_single\dt_region_memory_prompt.json
```

Recurrent class-memory candidate:

```powershell
python -u pc_dlcmnet\training\supervised.py `
  --holdout-region "01褰撴秱" `
  --variant recurrent_memory_prompt `
  --anchor-kind onehot `
  --aux-source acoustic `
  --aux-representation sequence `
  --audio-root $env:TONE0614_AUDIO_ROOT `
  --recurrent-episode-column speaker_id `
  --recurrent-episode-length 16 `
  --recurrent-eval-mode frozen `
  --train-fraction 1.0 `
  --max-steps 800 `
  --device cuda `
  --output output\0614\feature_memory_single\dt_recurrent_memory_prompt.json
```

## Core Ablation

```powershell
python -u tools\experiments\feature_memory_grid.py `
  --regions "01褰撴秱" "08娉惧幙" "11榛勫北" "13楂樻烦" `
  --variants speech_baseline memory_only feature_memory_no_prompt feature_memory_prompt region_memory_prompt recurrent_memory_prompt `
  --train-fractions 1.0 `
  --seed 0 `
  --max-steps 800 `
  --device cuda `
  --aux-source acoustic `
  --aux-representation sequence `
  --output-dir output\0614\feature_memory_grid\hard4_full `
  --extra-args --audio-root $env:TONE0614_AUDIO_ROOT --lambda-region-cls 0.1 --lambda-region-compact 0.01
```

## Full LORO Main Experiment

```powershell
python -u tools\experiments\feature_memory_grid.py `
  --regions all `
  --variants speech_baseline memory_only feature_memory_no_prompt feature_memory_prompt region_memory_prompt recurrent_memory_prompt `
  --train-fractions 1.0 `
  --seed 0 `
  --max-steps 800 `
  --device cuda `
  --aux-source acoustic `
  --aux-representation sequence `
  --output-dir output\0614\feature_memory_grid\all_regions_full `
  --extra-args --audio-root $env:TONE0614_AUDIO_ROOT --lambda-region-cls 0.1 --lambda-region-compact 0.01
```

The grid runner writes one log per job, plus `summary.csv`, `summary.json`,
`summary.md`, and metric plots.

Each single run also writes:

- `*.audit.json`: LORO split audit, including all regions, source regions, train/val/test region counts, and checks that target rows are absent from train/val.
- `*.curve.csv`: training trace with losses, validation metrics, module gradient norms (`grad_class_memory`, `grad_memory_write`, `grad_feature_fusion`, `grad_prompt`, etc.), feature weights, prompt gates, write gates, and memory norms.
- `*.debug.json`: compact run/debug metadata for quick inspection.
