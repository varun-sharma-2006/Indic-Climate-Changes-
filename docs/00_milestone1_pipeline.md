# 🌍 Indic Climate Stance Challenge

Stance detection for **Hindi and Bengali social-media (YouTube) comments** toward the claim

> **"Climate change and global warming is a serious concern."** → `Favour` / `Against` / `None`

This is the AISoME 2026 (FIRE 2026) shared task. The labelled public data is **English**; the hidden test set is **Hindi + Bengali**, with romanised, code-mixed and noisy text. We treat it as a **cross-lingual, cross-domain, noise-robust** stance problem, not a plain 3-class classifier.

| Doc | Content |
|---|---|
| [docs/01_problem_understanding.md](docs/01_problem_understanding.md) | Task, labels, why it is hard, Hindi/Bengali phenomena, research questions |
| [docs/02_literature_review.md](docs/02_literature_review.md) | Stance, climate stance, Indic, code-mixed, LLM work, research gap |
| [docs/03_datasets.md](docs/03_datasets.md) | Official + external datasets, label mappings, challenge set |
| [docs/04_sota.md](docs/04_sota.md) | State of the art: models, preprocessing, training, metrics, limitations |
| [docs/05_methodology.md](docs/05_methodology.md) | Proposed pipeline and hypotheses |
| [docs/06_experiment_log.md](docs/06_experiment_log.md) | All runs and findings |
| `reports/eda/*/report.md` | Generated EDA reports |

---

## Setup (Windows / Linux)

```powershell
# 1) PyTorch WITH CUDA first (a CPU-only build cannot use your GPU)
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True

# 2) Everything else
pip install -r requirements.txt
```

Tested with Python 3.13, transformers 5.17, torch 2.10, on an RTX 2050 (4 GB).

## Data

| What | How |
|---|---|
| Official train/test (email) | put the files in `data/official/`, then `python -m stance.prepare_official --file data/official/<file> --split train` (and `--split test`) |
| GWSD + SemEval-2016 climate | automatic: `python -m stance.download_external` |
| **Kaggle Twitter Climate Change Sentiment** (43k tweets) | download from https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset, unzip, put `twitter_sentiment_data.csv` in `data/external/raw/`, re-run `download_external` |

## Pipeline: commands in order

```powershell
# 0. one shot for Round-1 experiments
.\scripts\run_round1.ps1

# --- or step by step ---
python -m stance.download_external
python -m stance.build_dataset --train "data/external/processed/*.csv" --valid-from "gwsd_test|semeval2016_test" --holdout-whole --name en_only
python -m stance.eda --input "data/external/processed/*.csv" --name external_en --compare data/challenge/challenge_hi_bn.csv
python -m stance.baseline --data data/processed/en_only --eval data/challenge/challenge_hi_bn.csv
python -m stance.zero_shot_nli --input data/challenge/challenge_hi_bn.csv
python -m stance.train --data data/processed/en_only --model xlm-roberta-base --pair-with-claim --claim-lang match --eval data/challenge/challenge_hi_bn.csv --name xlmr_en_only

# translate-train + augmentation
python -m stance.translate --input data/external/processed/gwsd.csv --langs hi bn
python -m stance.translate --input data/external/processed/semeval2016_climate.csv --langs hi bn
python -m stance.translate --input data/external/processed/kaggle_twitter_climate.csv --langs hi bn --max-rows 6000 --balanced
python -m stance.augment --input "data/translated/*.csv" --output data/augmented/translated_aug.csv
# validation stays = original English held-out rows; translations of those rows are dropped from train to avoid leakage
python -m stance.build_dataset --train "data/external/processed/*.csv" "data/translated/*.csv" "data/augmented/*.csv" --valid-from "^gwsd_test$|^semeval2016_test$" --holdout-whole --exclude-sources gwsd_test_ semeval2016_test_ --name full
python -m stance.train --data data/processed/full --model xlm-roberta-base --pair-with-claim --claim-lang match --sampler lang_label --eval data/challenge/challenge_hi_bn.csv --name xlmr_full

# analysis
python -m stance.error_analysis --pred runs/xlmr_full/pred_challenge_hi_bn.csv
python -m stance.robustness --model runs/xlmr_full/best --input data/challenge/challenge_hi_bn.csv
python -m stance.explain --model runs/xlmr_full/best --input data/challenge/challenge_hi_bn.csv
python -m stance.ensemble --preds runs/xlmr_full/pred_challenge_hi_bn.csv runs/zero_shot_nli/pred_challenge_hi_bn.csv --output runs/ensemble/pred_cs.csv

# final predictions on the hidden test file
python -m stance.predict --model runs/xlmr_full/best --input data/official/test.csv --output submissions/xlmr_full.csv
```

## Repository layout

```
stance/                     python package (run modules with  python -m stance.<module>)
  config.py                 labels, claim in en/hi/bn, paths
  text.py                   Unicode/Indic normalisation, script & language ID, Code-Mixing Index, lexicons
  data_io.py                robust loader -> unified schema [id,text,label,lang,source]
  prepare_official.py       convert the emailed files
  download_external.py      GWSD, SemEval-2016 climate, Kaggle climate tweets
  translate.py              NLLB-200 translate-train (en -> hi, bn)
  augment.py                romanisation, code-mix swap, typos, emoji/URL noise
  build_dataset.py          merge, dedup, leak-free train/valid split
  eda.py                    statistics + log-odds distinctive words + figures
  baseline.py / tfidf.py    TF-IDF + LR/SVM
  train.py                  multilingual transformer fine-tuning (progress bars, early stopping, LoRA option)
  zero_shot_nli.py          stance-as-NLI, no training
  predict.py                inference + submission file
  error_analysis.py         breakdown by language/phenomenon/negation/length + confident errors + reliability
  robustness.py             CheckList-style perturbation test (macro-F1 drop, flip rate)
  explain.py                occlusion word attributions -> HTML
  ensemble.py               probability averaging
data/challenge/             hand-written Hindi/Bengali diagnostic set (phenomenon-tagged)
docs/                       research documentation
scripts/run_round1.ps1      end-to-end Round-1 script
reports/, runs/             generated outputs
```

## Model information

| Model | HF id | Params | Notes |
|---|---|---|---|
| XLM-R base | `xlm-roberta-base` | 278M | default backbone |
| MuRIL base | `google/muril-base-cased` | 237M | trained with transliterated Indic text |
| IndicBERT v2 | `ai4bharat/IndicBERTv2-MLM-only` | 278M | Indic-focused |
| mDeBERTa-v3 NLI | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 279M | zero-shot NLI and NLI-initialised fine-tuning |
| NLLB-200 distilled | `facebook/nllb-200-distilled-600M` | 600M | translation |

Every run writes `runs/<name>/{train_args.json, history.csv, metrics.json, pred_*.csv, best/}`.

## Results so far

See [docs/06_experiment_log.md](docs/06_experiment_log.md). Headline: TF-IDF reaches **0.60** macro-F1 in English but only **0.33** on Hindi/Bengali, which quantifies the cross-lingual gap our methods target.

## Team

| Name | Enrollment no. |
|---|---|
| Varun Sharma | S24CSEU0170 |
| Yashika Garg | S24CSEU0235 |
| Priyanshu Raj | S24CSEU1142 |
| Advay Sinha | S24CSEU2059 |

Milestone 1 report: [docs/Milestone1_Report.docx](docs/Milestone1_Report.docx)
