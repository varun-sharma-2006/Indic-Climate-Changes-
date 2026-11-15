# 3. Datasets

All counts below come from our own scripts (`python -m stance.download_external`, `python -m stance.eda`), not from the papers.

## 3.1 Official data (AISoME 2026 / course)

| Split | Source | Language | Notes |
|---|---|---|---|
| train | sent by email | (to be filled after receipt) | convert with `python -m stance.prepare_official --file … --split train` |
| test (hidden) | organisers | Hindi + Bengali YouTube comments | final evaluation; macro-F1 |

> Gotcha we hit: `pandas.read_csv` treats the string **"None"** as missing by default, which silently deletes the entire *None* class. All our readers use `keep_default_na=False`.

## 3.2 External datasets used for training

| # | Dataset | Get it | Size (after our cleaning) | Domain | Original labels → ours |
|---|---|---|---|---|---|
| 1 | **GWSD**: Global Warming Stance Dataset (Luo et al., 2020) | auto: `github.com/yiweiluo/GWStance` → `GWSD.tsv` | 2,300 sentences (2,100 train + 200 author held-out test). Favour 864 / Against 450 / None 986 | news | agree→Favour, disagree→Against, neutral→None. Gold label = argmax of the released MACE-style probabilities; 250 screening items have none, so we use the majority vote of 8 annotators |
| 2 | **SemEval-2016 Task 6, climate target** (Mohammad et al., 2016) | auto: HF `cardiffnlp/tweet_eval`, config `stance_climate` | 564 tweets. Favour 335 / Against 26 / None 203 | Twitter | favor/against/none (tweet_eval ids 2/1/0). `#SemST` tag removed |
| 3 | **Twitter Climate Change Sentiment** (Kaggle, E. Qian) | **manual**: https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset → put `twitter_sentiment_data.csv` in `data/external/raw/` | 43,943 tweets (fewer after removing retweets and duplicates) | Twitter | 1 (pro)→Favour, −1 (anti)→Against, 0 (neutral)→None, 2 (news)→None (configurable `--kaggle-news-as`) |
| 4 | **Translate-train copies** of 1–3 | generated: `python -m stance.translate` (NLLB-200 600M) | same as source × {hi, bn} | — | labels copied |
| 5 | **Noise augmentation** | generated: `python -m stance.augment` | configurable | — | labels copied |

**Why Kaggle #3 is the main dataset to download:** it is 15× larger than GWSD and SemEval combined, it is user-generated social-media text (the closest genre to YouTube comments), and it contains enough *Against* examples (about 4k) to fix the minority-class problem visible in GWSD (17%) and SemEval (5%).

## 3.3 Evaluation-only data

| Dataset | Size | Purpose |
|---|---|---|
| `data/challenge/challenge_hi_bn.csv` (**ours**) | 64 hand-written comments: hi 25, bn 22, hi-rom 9, bn-rom 6, en 2. Favour 29 / Against 20 / None 15 | Diagnostic only, **never trained on**. Each row is tagged with a phenomenon: explicit, implicit, negation, sarcasm, rhetorical, romanized, codemix, codemix-script, typo, topical-neutral, hard-none, unrelated. Labels assigned by the team; to be double-checked by a second native speaker of each language (inter-annotator agreement to be reported) |
| GWSD author test split (200) + SemEval test (169) | 369 | in-distribution English validation (`data/processed/en_only/valid.csv`) |

## 3.4 Other datasets considered (not used yet)

| Dataset | Why it could help | Why not (yet) |
|---|---|---|
| ClimaText (Varini et al., 2020) | climate relevance labels → helps separate None from stance | relevance only, not stance |
| CLIMATE-FEVER | NLI-like claim/evidence pairs | encyclopaedic text, not opinions |
| climatebert/climate_sentiment, climate_detection (HF) | climate-domain text | corporate reports |
| Dhoroni (Bengali climate news, 2024) | only Bengali climate-stance resource | news domain; check licence and availability |
| SentNoB (noisy Bengali), SemEval-2020 SentiMix (Hinglish) | in-language noisy social media for **domain-adaptive MLM pre-training** | sentiment, not stance; planned for the DAPT step |
| IndicSentiment / IndicXNLI (AI4Bharat) | in-language NLI for an NLI-initialised model | planned |
| Unlabelled Hindi/Bengali YouTube climate comments | self-training / pseudo-labelling | need scraping (YouTube Data API) and ethics review |

## 3.5 Licences and ethics

* GWSD, tweet_eval and Kaggle each carry their own terms. Check each source page before redistributing; our repo stores only *download scripts*, not the raw files.
* Tweets and comments are user-generated. We do not publish user identifiers. `@mentions` are masked to `@USER` in preprocessing.
