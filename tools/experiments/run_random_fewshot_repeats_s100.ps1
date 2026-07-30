$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $Root

& python `
  "tools\experiments\random_fewshot_repeats.py" `
  --stage-dir "output\0614\multiregion_paper_dual_memory" `
  --output-dir "output\0614\random_fewshot_repeats_rolling_s100_e1" `
  --eval-support-shots 4 `
  --eval-split-runs 100 `
  --eval-ensemble-runs 1 `
  --eval-split-mode rolling `
  --support-write-mode label `
  --support-label-blend 1.0 `
  --eval-query-shots-per-class 1 `
  --device cuda `
  --skip-completed `
  --quiet-child `
  --progress-child
