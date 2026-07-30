# Chizhou unified sanity check

| Model | ACC | Macro-F1 | Zero-shot ACC | Zero-shot Macro |
|---|---:|---:|---:|---:|
| hubert-base-ls960 | 52.60 ± 3.39 | 47.19 ± 2.87 | 59.92 | 55.08 |
| wav2vec2-base | 39.33 ± 3.96 | 35.47 ± 2.72 | 48.00 | 42.46 |

- `wavlm-base` 在训练中触发了 CUDA `cublasSgemmStridedBatched` 错误，当前批次中断。
