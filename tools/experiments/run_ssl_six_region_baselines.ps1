$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

& python `
  "tools\experiments\ssl_pretrained_baselines.py" `
  --csv "output\datasets\wu_low_resource_vowel_dataset.csv" `
  --output-dir "output\0614\ssl_pretrained_six_regions" `
  --cache-dir "output\0614\ssl_feature_cache" `
  --models wav2vec2-base hubert-base wavlm-base-plus `
  --methods ridge svm centroid `
  --holdout-regions "04青阳" "06铜陵" "08泾县" "10南陵" "11黄山" "13高淳" `
  --device cuda `
  --batch-size 16 `
  --quiet
