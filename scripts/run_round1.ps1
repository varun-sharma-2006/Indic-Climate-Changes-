# Round-1 experiment pipeline. Run from the repo root:   .\scripts\run_round1.ps1
# Each step prints progress bars / metrics; outputs land in data/processed, reports/ and runs/.
$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$CS = "data/challenge/challenge_hi_bn.csv"

Write-Host "`n=== 1. Download external datasets ===" -ForegroundColor Cyan
python -m stance.download_external

Write-Host "`n=== 2. Build English-only train/valid (valid = GWSD author test + SemEval test) ===" -ForegroundColor Cyan
python -m stance.build_dataset --train "data/external/processed/*.csv" --valid-from "gwsd_test|semeval2016_test" --holdout-whole --name en_only

Write-Host "`n=== 3. EDA ===" -ForegroundColor Cyan
python -m stance.eda --input "data/external/processed/*.csv" --name external_en --compare $CS
python -m stance.eda --input $CS --name challenge

Write-Host "`n=== 4. TF-IDF baseline ===" -ForegroundColor Cyan
python -m stance.baseline --data data/processed/en_only --eval $CS
python -m stance.error_analysis --pred runs/tfidf_lr_en_only/pred_challenge_hi_bn.csv

Write-Host "`n=== 5. Zero-shot NLI baseline (downloads ~1.1 GB once) ===" -ForegroundColor Cyan
python -m stance.zero_shot_nli --input $CS data/processed/en_only/valid.csv

Write-Host "`n=== 6. XLM-R zero-shot cross-lingual (GPU recommended) ===" -ForegroundColor Cyan
python -m stance.train --data data/processed/en_only --model xlm-roberta-base --pair-with-claim --claim-lang match `
    --epochs 5 --batch-size 8 --grad-accum 2 --eval $CS --name xlmr_en_only
python -m stance.error_analysis --pred runs/xlmr_en_only/pred_challenge_hi_bn.csv
python -m stance.robustness --model runs/xlmr_en_only/best --input $CS data/processed/en_only/valid.csv
python -m stance.explain --model runs/xlmr_en_only/best --input $CS --n 64

Write-Host "`nDone. See reports/ and runs/." -ForegroundColor Green
