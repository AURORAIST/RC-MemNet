# Statistical Significance

Updated: 2026-07-27T14:33:04

Tests use saved artifacts only. Region-level rows are paired by holdout region. Per-run rows are reported only when a saved `per_run_metrics.csv` exists.

## Summary

| Baseline | Metric | n | Baseline | RC-MemNet | Gain | 95% bootstrap CI | Sign p | t approx p | Sig. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mas_lora | accuracy | 8 | 65.94 | 65.06 | -0.87 | [-4.79, 2.64] | 0.7266 | 0.6698 | n.s. |
| mas_lora per-run (04青阳) | accuracy | 100 | 67.84 | 71.40 | 3.56 | [3.11, 4.05] | <1e-4 | <1e-4 | *** |
| mas_lora per-run (04青阳) | macro_f1 | 100 | 61.13 | 64.46 | 3.32 | [2.91, 3.75] | <1e-4 | <1e-4 | *** |
| mas_lora per-run (04青阳) | weighted_f1 | 100 | 68.56 | 71.82 | 3.26 | [2.81, 3.72] | <1e-4 | <1e-4 | *** |
| whisaid | accuracy | 8 | 48.00 | 65.06 | 17.06 | [12.79, 21.00] | 0.0078 | <1e-4 | ** |

## Region Pairs

| Baseline | Region | Metric | Baseline | RC-MemNet | Gain |
|---|---|---|---:|---:|---:|
| mas_lora | 03池州 | accuracy | 66.29 | 71.77 | 5.48 |
| mas_lora | 11黄山 | accuracy | 66.34 | 60.47 | -5.87 |
| mas_lora | 08泾县 | accuracy | 64.99 | 53.97 | -11.01 |
| mas_lora | 14溧水 | accuracy | 65.86 | 60.79 | -5.07 |
| mas_lora | 10南陵 | accuracy | 65.07 | 67.56 | 2.49 |
| mas_lora | 12宁国 | accuracy | 65.76 | 65.90 | 0.14 |
| mas_lora | 04青阳 | accuracy | 67.84 | 71.40 | 3.56 |
| mas_lora | 06铜陵 | accuracy | 65.35 | 68.65 | 3.30 |
| whisaid | 03池州 | accuracy | 46.25 | 71.77 | 25.52 |
| whisaid | 11黄山 | accuracy | 48.44 | 60.47 | 12.03 |
| whisaid | 08泾县 | accuracy | 47.39 | 53.97 | 6.59 |
| whisaid | 14溧水 | accuracy | 48.75 | 60.79 | 12.03 |
| whisaid | 10南陵 | accuracy | 46.09 | 67.56 | 21.46 |
| whisaid | 12宁国 | accuracy | 47.82 | 65.90 | 18.07 |
| whisaid | 04青阳 | accuracy | 49.59 | 71.40 | 21.81 |
| whisaid | 06铜陵 | accuracy | 49.68 | 68.65 | 18.97 |

## Outputs

- `significance_summary.csv`
- `region_level_pairs.csv`
- `significance_summary.json`
- `significance_table.tex`
