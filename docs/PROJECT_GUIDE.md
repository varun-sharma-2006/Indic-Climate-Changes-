# Project Guide: everything about our project in one place

**Project:** Detecting Stance in Media on Global Warming for Indic Languages
**Team:** Varun Sharma (S24CSEU0170) · Yashika Garg (S24CSEU0235) · Priyanshu Raj (S24CSEU1142) · Advay Sinha (S24CSEU2059)
**Guide:** Dr. Supriya Chanda, SCSET, Bennett University
**Code:** https://github.com/varun-sharma-2006/Indic-Climate-Changes-

Every team member should read this whole file. The evaluator can ask anyone about any part.

---

## 1. The project in one paragraph

We built a system that reads a Hindi or Bengali comment and decides whether the writer **agrees (Favour)**, **disagrees (Against)** or has **no clear stance (None)** towards the claim *"Climate change and global warming is a serious concern."* Labelled data exists only in English, so we take two public English stance datasets, **translate them into Hindi and Bengali** with a neural translation model, and **fine-tune multilingual transformer models** on all three languages. Our base paper (Luo et al., 2020) solved the same task for English news only. Its model scores Macro F1 **0.28 on Hindi and 0.12 on Bengali**; our best model (MuRIL) scores **0.66 and 0.65**.

---

## 2. The task

| Item | Detail |
|---|---|
| Input | One comment / sentence in Hindi or Bengali (also works for English) |
| Output | Favour, Against or None |
| Claim (target) | "Climate change and global warming is a serious concern." |
| Main metric | **Macro F1**: F1 averaged over the 3 classes equally, so the rare "Against" class matters as much as the common "Favour" |
| Other metrics | Accuracy, macro precision, macro recall, confusion matrix |
| Related benchmark | Same task as the AISoMe 2026 shared task at FIRE 2026 (training data English, test data Hindi + Bengali) |

**Stance is not sentiment.** "The government is doing nothing about these heatwaves!" is angry (negative sentiment) but **Favour** (it treats climate change as serious).

---

## 3. Base paper

> **Yiwei Luo, Dallas Card, Dan Jurafsky. "Detecting Stance in Media on Global Warming." Findings of EMNLP 2020.**
> - ACL Anthology (official): https://aclanthology.org/2020.findings-emnlp.296
> - PDF (arXiv): https://arxiv.org/abs/2010.15149
> - DOI: https://doi.org/10.18653/v1/2020.findings-emnlp.296
> - Code and data: https://github.com/yiweiluo/GWStance

**What they did**
1. Collected sentences about global warming from news articles.
2. Had **8 crowd workers** (Amazon Mechanical Turk) label each sentence as *agrees / disagrees / neutral* towards "Climate change/global warming is a serious concern". This is **exactly our claim**, and the reason we chose this paper. Our project title is also the paper's title.
3. Combined the 8 votes with a Bayesian model into a probability for each label → the **GWSD** dataset (Global Warming Stance Dataset).
4. Fine-tuned **English BERT** and reached about **75% accuracy**, close to human agreement.
5. Used the classifier on about 56,000 news articles to study how climate-accepting and climate-sceptical outlets frame their arguments.

**Limitations we address**

| Base paper | Our project |
|---|---|
| English only | Hindi + Bengali (+ English) |
| Formal news sentences | News + tweets; preprocessing for noisy social-media text; romanised Hindi/Bengali |
| 2,050 training sentences | 5,630 English examples × 3 languages (+ 2 romanised versions) |
| One model (English BERT) | 14+ experiments: TF-IDF, BERT, mBERT, XLM-R, MuRIL, LLM, ensembles |
| No special handling of the rare "disagree" class | Class-weighted loss, Macro F1 model selection |
| No explanations | LIME explanations + detailed error analysis |

**We reproduced the base paper:** our English BERT trained on English gets **75.0% accuracy on English**, the same as the paper. Then we tested it on Hindi/Bengali (F1 0.28 / 0.12), which shows why our extension is needed.

---

## 4. Datasets: what we used and where it comes from

### 4.1 GWSD: Global Warming Stance Dataset (from the base paper)

| | |
|---|---|
| Source | Luo, Card & Jurafsky (2020) |
| Download link | https://github.com/yiweiluo/GWStance → file **`GWSD.tsv`** (raw URL: https://raw.githubusercontent.com/yiweiluo/GWStance/master/GWSD.tsv) |
| Language / genre | English, sentences from news articles about global warming |
| Size | 2,300 rows in the file; 250 are annotator-screening sentences without label probabilities, so **2,050 labelled sentences** remain. |
| Annotation | 8 workers per sentence (columns `worker_0` … `worker_7`) + probability columns `agree`, `disagree`, `neutral` |
| Our label | The class with the highest probability: agree → **Favour**, disagree → **Against**, neutral → **None** |
| Split | The authors marked **200 sentences as a held-out test set** (`in_held_out_test = True`); we use them as test. From the remaining 1,850 we take a stratified 10% (185) as validation and 1,665 as training. |
| Labels | Favour 777 · Against 400 · None 873 |
| Average length | 18.8 words |

### 4.2 ClimateStance (Vaid et al., 2022)

| | |
|---|---|
| Source | Vaid, Pant & Shrivastava, "Towards Fine-grained Classification of Climate Change related Social Media Text", ACL 2022 Student Research Workshop. Paper: https://aclanthology.org/2022.acl-srw.35 |
| Download link | Official repo: https://github.com/roopalv54/finegrained-climate-change-social-media. Files we download: https://huggingface.co/datasets/eci-io/climate-evaluation, folder `ClimateStance/` (`train.csv`, `val.csv`, `test.csv`) |
| Language / genre | English **tweets** posted 1–14 December 2019, around the UN climate conference COP25 |
| Annotation | 2 annotators with a linguistics background; Cohen's kappa 0.817 (reported in the paper) |
| Original labels | favour / against / ambiguous **towards climate-change prevention**; sarcastic tweets were marked ambiguous |
| Size | The paper says 3,777 tweets; the **public files contain 3,580** (2,871 train / 354 val / 355 test). We use the official split as it is. |
| Our label | Label `1` → **Favour**, `2` → **Against**, `0` (ambiguous) → **None**. The files only contain numbers; we worked out the mapping by reading example tweets of each number (e.g. "Climate change hoax COLLAPSES…" is `2`). |
| Labels | Favour 2,863 · Against 330 · None 387 |
| Average length | 28.3 words |

### 4.3 Combined English data (`data/processed/english.csv`)

| Split | GWSD | ClimateStance | Total | Favour | Against | None |
|---|---|---|---|---|---|---|
| train | 1,665 | 2,871 | **4,536** | 2,925 | 594 | 1,017 |
| val | 185 | 354 | **539** | 354 | 62 | 123 |
| test | 200 | 355 | **555** | 361 | 74 | 120 |
| **Total** | 2,050 | 3,580 | **5,630** | 3,640 (65%) | 730 (13%) | 1,260 (22%) |

Columns: `id, source, split, text, label`. The data is **imbalanced**: only 13% is Against.

### 4.4 Hindi and Bengali versions (made by us, `data/processed/hindi.csv`, `bengali.csv`)

- Every English row is translated with **NLLB-200 distilled 600M** (`facebook/nllb-200-distilled-600M`, Meta AI, open-source; paper: https://arxiv.org/abs/2207.04672). Language codes `hin_Deva` and `ben_Beng`, beam search with 2 beams, on our GPU (≈ 20 min for both languages).
- Text is cleaned before translation (URLs and @mentions removed) so the translator doesn't try to translate them.
- **Same `id`, `split` and `label` as the English row.** A sentence and its translations are always in the same split, so no test sentence leaks into training through its translation.
- **Quality fix:** about 0.4% of outputs got stuck repeating a word ("ओह ओह ओह …"). `translate.py --fix` detects these (output > 2.5× longer than the input, or a word repeated 4+ times in a row) and re-translates them with repetition blocking (about 40 rows in total were repaired).
- Average length: Hindi 28.3 words, Bengali 21.0 words.
- **Why we translate (translate-train):** the models then see Hindi/Bengali text during training. This is a form of **data augmentation**: 5,630 → 16,890 training-pool examples.

### 4.5 Romanised versions (`data/processed/hindi_roman.csv`, `bengali_roman.csv`)

Many Indians type Hindi/Bengali in English letters ("garmi bahut badh gayi hai"). `src/romanize_data.py` converts the Hindi/Bengali translations to Latin script with the `romanize()` function from our Milestone 1 code (`stance/augment.py`): ITRANS transliteration, lower-cased, final "a" dropped. Example: "ग्लोबल वार्मिंग एक गंभीर समस्या है" → "global varmimg eka gambhir samasya hai".

### 4.6 Hand-written challenge set (`data/challenge/challenge_hi_bn.csv`, from Milestone 1)

64 comments written in YouTube style, created during Milestone 1 and stored in our repo (not translations): Hindi 25, Bengali 22, romanised Hindi 9, romanised Bengali 6, English 2. Labels: Favour 29, Against 20, None 15. Each is tagged with a phenomenon (explicit, implicit, sarcasm, negation, romanised, unrelated, code-mixed, typo, …). It is small, so we use it **only for testing**, to see how models handle natural comment-style text.

### 4.7 Data still to come

| Data | Status |
|---|---|
| Course dataset (Hindi/Bengali YouTube comments, from the instructor) | Not released yet. `predict.py` already writes the required `ID, Predicted Label` file. |
| Our own YouTube comments | Script ready (`src/collect_youtube.py`, YouTube Data API v3). Needs an API key; then two members label ~500 comments per language (guide in `data/youtube/README.md`). |

---

## 5. The pipeline, step by step

| Step | Script | What it does | Owner |
|---|---|---|---|
| 1 | `src/download_data.py` | Downloads GWSD + ClimateStance, maps labels, makes splits → `english.csv` | Varun |
| 2 | `src/translate.py` | English → Hindi, Bengali with NLLB-200 (+ `--fix` for broken outputs) | Varun |
| 2b | `src/romanize_data.py` | Romanised Hindi/Bengali copies | Varun |
| 3 | `src/data_stats.py` | Statistics, class distribution and length plots | Varun |
| – | `src/preprocess.py` | Cleaning: HTML entities, Unicode NFC, URLs removed, @user, hashtag splitting (#ClimateHoax → climate hoax), emoji → words (🔥 → fire), repeated letters shortened | Yashika |
| 4a | `src/baselines.py` | TF-IDF (word 1–2 grams + character 2–5 grams) + Logistic Regression / Linear SVM | Yashika |
| – | `src/data_utils.py` | Data loading, metrics, confusion matrices | Yashika |
| 4b | `src/train_transformer.py` | Fine-tunes BERT / mBERT / XLM-R / MuRIL | Priyanshu |
| – | `src/models.py` | [CLS] classifier or mean-pooling classifier (MuRIL) | Priyanshu |
| 4c | `src/llm_fewshot.py` | Qwen2.5-1.5B-Instruct LLM with a few-shot prompt | Advay |
| 5 | `src/ensemble.py` | Averages the class probabilities of several models | Advay |
| 6 | `src/error_analysis.py`, `src/eval_challenge.py`, `src/evaluate_saved.py` | Errors by language/dataset/class; challenge set; romanised test sets | Advay |
| 7 | `src/explain_lime.py` | LIME: which words caused each prediction | Advay |
| 8 | `src/summarize_results.py` | Results table + comparison chart | Advay |
| – | `src/predict.py` | Demo on one sentence, or the submission file | Advay |
| – | `docs/make_progress_report.py` | Builds the 2-page progress report from the results | Advay |

Everything runs with `.\run_all.ps1` (about 2 hours on our RTX 2050 laptop GPU).

---

## 6. The models, explained simply

| Model | What it is | Why we use it |
|---|---|---|
| **TF-IDF + LR / SVM** | Counts words and character pieces, weights rare-but-informative ones, then a linear classifier | Fast baseline; shows how much the deep models add |
| **English BERT** (`bert-base-uncased`) | Transformer pre-trained on English text | The **base paper's model**, our reference point |
| **mBERT** (`bert-base-multilingual-cased`) | BERT pre-trained on Wikipedia in 104 languages | First multilingual baseline |
| **XLM-R** (`xlm-roberta-base`) | Pre-trained on 2.5 TB of web text in 100 languages | Strong cross-lingual transfer |
| **MuRIL** (`google/muril-base-cased`) | BERT pre-trained by Google on 17 languages (16 Indian + English), **including transliterated (romanised) text** | Built for Indian languages; our best model |
| **Qwen2.5-1.5B-Instruct** | Small multilingual large language model | LLM baseline: no training, only a prompt with definitions + 1 example per class |
| **Ensemble** | Average of the probabilities of several fine-tuned models | Combines models that make different mistakes |

**Two training settings:**
- **English only / zero-shot:** train on English, test on Hindi/Bengali. This is what the base paper would do.
- **Translate-train (ours):** train on English + Hindi + Bengali translations.

**Training settings:** AdamW optimizer, learning rate 3e-5, batch size 16, max 128 tokens, 4 epochs (6 for MuRIL), 10% warm-up, mixed precision (fp16), **class-weighted cross-entropy** (Against errors count ~5× more than Favour errors), best epoch picked by validation Macro F1. The **word-embedding layer is frozen**, because XLM-R and MuRIL do not fit in our 4 GB GPU otherwise. Freezing also keeps the multilingual word vectors aligned across languages.

---

## 7. Results

Test sets: 555 examples per language (Hindi/Bengali test sets are translations of the English test set).

| Model | Training data | F1 En | F1 Hi | F1 Bn |
|---|---|---|---|---|
| TF-IDF + LR | English only | 0.620 | 0.171 | 0.155 |
| **English BERT (base paper)** | English only | 0.667 | **0.275** | **0.120** |
| mBERT | English only | 0.654 | 0.420 | 0.346 |
| XLM-R | English only | 0.689 | 0.585 | 0.497 |
| MuRIL | English only | 0.637 | 0.567 | 0.542 |
| Qwen2.5-1.5B LLM | Few-shot prompt | 0.510 | 0.424 | 0.384 |
| TF-IDF + LR | Translate-train | 0.614 | 0.592 | 0.583 |
| mBERT | Translate-train | 0.651 | 0.555 | 0.564 |
| XLM-R | Translate-train | 0.673 | 0.641 | 0.619 |
| **MuRIL (ours)** | **Translate-train** | 0.655 | **0.662** | **0.650** |
| Ensemble (mBERT+XLM-R+MuRIL) | Translate-train | 0.671 | 0.675 | 0.620 |

**Romanised Hindi/Bengali** (the same test sets written in Latin letters, e.g. "global varmimg eka gambhir samasya hai"):

| Model | F1 Hi | F1 Bn | F1 Hi-romanised | F1 Bn-romanised |
|---|---|---|---|---|
| XLM-R translate-train | 0.641 | 0.619 | 0.236 | 0.143 |
| MuRIL translate-train | 0.662 | 0.650 | 0.521 | 0.487 |
| MuRIL trained from scratch with romanised copies (4 epochs) | 0.552 | 0.524 | 0.564 | 0.567 |
| **MuRIL translate-train + 2 more epochs with romanised copies** | 0.655 | 0.634 | **0.623** | **0.563** |

XLM-R almost fails on romanised text, while MuRIL (pre-trained with transliterated text) keeps working. Continued training with romanised copies adds **+0.10 / +0.08** F1 on romanised text for a small loss on native script. Training from scratch on all five versions underfitted in 4 epochs (loss still 1.02), so continuing from the trained model was the better choice. On the 64 hand-written comments the romanised model gets 10/15 romanised comments right (vs 9/15) but 2 fewer native-script ones, so **`muril_tt` stays our main model** and `muril_tt_rom` is the option for mostly-romanised input.

**What the results show**
1. The base paper's approach works for English (75% accuracy) but **fails on Hindi/Bengali**.
2. **Translate-train beats English-only for every model** (+0.06 to +0.22 F1).
3. **MuRIL > XLM-R > mBERT** after translate-train: pre-training on Indian languages helps.
4. The ensemble gives the best **Hindi** score but is worse on Bengali, so MuRIL alone is our final model.
5. A small LLM with only a prompt is much weaker than fine-tuned models.
6. On the **hand-written challenge set**: XLM-R 0.68, MuRIL 0.66 macro F1. Explicit statements are always right; **sarcasm (38%)** and **romanised text (20%)** were the weak points (see 4.5 and the romanised results above).
7. **Error analysis:** 55% of errors are Favour ↔ None. ClimateStance's "ambiguous" class (it includes sarcasm) fits our None poorly (F1 < 0.15), while GWSD's neutral class reaches 0.70–0.74.
8. **LIME** shows sensible cues: ঝুঁকি/खतरा ("risk/danger") → Favour; না ("not") → Against; प्रतिशत ("percent", survey statistics) → None.

All numbers: `results/results_table.md`, `results/metrics/*.json`, chart `results/figures/model_comparison.png`, confusion matrices `results/figures/cm_*.png`.

---

## 8. Problems we hit and how we solved them

| Problem | How we found it | Fix |
|---|---|---|
| MuRIL predicted only "Favour" (F1 0.26) | Training loss stuck at 1.098 = ln 3 (random guessing). Its [CLS] vectors had cosine similarity ≈ 1.0 for completely different sentences. | Classify from the **average of all token vectors** (mean pooling) instead of [CLS] |
| NLLB repeated words endlessly | Some translations were 255 words long while no English text is longer than 78 | Automatic detection + re-translation with `no_repeat_ngram_size=3` |
| GPU has only 4 GB | XLM-R/MuRIL full fine-tuning needs > 4 GB | Freeze word embeddings, fp16 |
| Label "None" read as a missing value by pandas | Labels showed as NaN | `keep_default_na=False` when reading CSVs |
| LIME split Hindi/Bengali words into single letters | Explanations showed "त", "प" | Split on spaces only |
| Only 13% Against | Class counts | Class-weighted loss + Macro F1 |

---

## 9. Likely evaluator questions (with answers)

1. **Why this base paper?** It defines exactly our claim and labels, its dataset is public, and our title is its title. We reproduced its result (75% accuracy) and extended it to Hindi/Bengali.
2. **Where is your Hindi/Bengali training data from?** No labelled Hindi/Bengali climate-stance data exists publicly. We machine-translated GWSD and ClimateStance with NLLB-200, and we keep the same labels and splits.
3. **Isn't testing on translated data unrealistic?** Yes, it is cleaner than real comments. That is why we also test on 64 hand-written comments, and why we are collecting real YouTube comments. The course test set will be the final check.
4. **Why Macro F1 and not accuracy?** Favour is 65% of the data, so always predicting Favour gives 65% accuracy but only 0.26 Macro F1. Macro F1 treats all three classes equally.
5. **What is translate-train?** Training on translated copies of the training data, so the model sees the target languages during training.
6. **What is zero-shot cross-lingual transfer?** Training a multilingual model on English only and directly testing it on Hindi/Bengali. It works partly because the model has learned shared representations across languages.
7. **Why is MuRIL better than XLM-R here?** It was pre-trained specifically on Indian languages and on transliterated text, so it understands Hindi/Bengali (and romanised forms) better.
8. **What did you change in MuRIL and why?** Mean pooling instead of [CLS], because its [CLS] vector was almost the same for every sentence, so the model could not learn (see section 8).
9. **Why freeze the embeddings?** To fit the model in 4 GB of GPU memory. It also keeps the multilingual word vectors aligned.
10. **How do you avoid data leakage?** A sentence and all its translations share one `id` and one split; GWSD's own held-out set and ClimateStance's official test split are our test sets.
11. **What does the ensemble do?** It averages the predicted probabilities of several models. It helped Hindi but not Bengali, because mBERT is weaker.
12. **Why did the LLM do badly?** It is a small 1.5B model, used without any training, only a prompt. Fine-tuned models learn the exact label definitions from thousands of examples.
13. **What is LIME?** It removes words at random, watches how the prediction changes, and fits a simple model to estimate each word's importance.
14. **What are your main weaknesses?** Sarcasm, romanised text, and the Favour/None boundary.
15. **What will you do before the final evaluation?** See section 11.

---

## 10. Live demo (what to run, in this order)

```powershell
cd "D:\Stance in Indic Climate"
.\.venv\Scripts\Activate.ps1

# 1. show the data
python src/data_stats.py

# 2. classify sentences live with our best model
python src/predict.py --model_dir models/muril_tt --text "ग्लोबल वार्मिंग एक गंभीर समस्या है, हमें अभी कदम उठाने होंगे"
python src/predict.py --model_dir models/muril_tt --text "ये सब क्लाइमेट चेंज का ड्रामा है, गर्मी तो हमेशा पड़ती थी"
python src/predict.py --model_dir models/muril_tt --text "ভিডিওটা দারুণ হয়েছে দাদা, পরের পর্ব কবে?"
#    expected: Favour / Against / None

# 3. show results: results/results_table.md, results/figures/model_comparison.png,
#    results/figures/cm_muril_tt_hi.png, results/lime/summary.md

# 4. (optional, ~10 min) train a model live: python src/baselines.py
```

The trained models are in `models/` on the laptop only (too large for GitHub). Bring the laptop to the evaluation.

---

## 11. What is left before the final evaluation

1. **Every member commits their own files to GitHub** (see `CONTRIBUTIONS.md`). The evaluation checks contributions per member.
2. Collect and label real YouTube comments; report inter-annotator agreement.
3. Run on the course dataset when it arrives (`predict.py`).
4. LoRA fine-tuning of an LLM (needs a bigger GPU: Kaggle/Colab).
5. Final report (max 8 pages) and presentation.

---

## 12. References

1. Luo, Card, Jurafsky (2020). Detecting Stance in Media on Global Warming. *Findings of EMNLP.* https://aclanthology.org/2020.findings-emnlp.296
2. Vaid, Pant, Shrivastava (2022). Towards Fine-grained Classification of Climate Change related Social Media Text. *ACL SRW.* https://aclanthology.org/2022.acl-srw.35
3. Khanuja et al. (2021). MuRIL: Multilingual Representations for Indian Languages. https://arxiv.org/abs/2103.10730
4. Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). https://arxiv.org/abs/1911.02116
5. Devlin et al. (2019). BERT. https://arxiv.org/abs/1810.04805
6. NLLB Team (2022). No Language Left Behind. https://arxiv.org/abs/2207.04672
7. Qwen Team (2024). Qwen2.5 Technical Report. https://arxiv.org/abs/2412.15115
8. Ribeiro, Singh, Guestrin (2016). LIME. https://arxiv.org/abs/1602.04938
9. Thapa et al. (2024). Stance and Hate Event Detection in Tweets Related to Climate Activism, CASE 2024. https://aclanthology.org/2024.case-1.33
10. Swami et al. (2018). An English-Hindi Code-Mixed Corpus: Stance Annotation and Baseline System. https://arxiv.org/abs/1805.11868
