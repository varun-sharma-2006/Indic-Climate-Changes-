# 2. Literature review

> Reading list organised by theme. For each work: **what they did → what we take from it → limitation we address.**
> ⚠ Before the final report, verify every bibliographic detail (venue, year, pages) against the ACL Anthology, arXiv or the publisher's page.

## 2.1 Stance detection: foundations and surveys

| Work | Contribution | Relevance to us |
|---|---|---|
| **Mohammad et al., 2016.** *SemEval-2016 Task 6: Detecting Stance in Tweets.* SemEval. | First shared benchmark: 5 targets incl. **"Climate Change is a Real Concern"**, labels FAVOR/AGAINST/NONE, metric = mean F1 of FAVOR and AGAINST | Our label scheme and claim are almost identical; their climate target data (via `tweet_eval`) is one of our training sources. The climate target was the hardest because *Against* had only ~5% of examples |
| **Küçük & Can, 2020.** *Stance Detection: A Survey.* ACM Computing Surveys. | Taxonomy: target-specific, multi-target, cross-target, cross-lingual | Our task is *cross-lingual + cross-domain, single target* |
| **ALDayel & Magdy, 2021.** *Stance detection on social media: State of the art and trends.* Information Processing & Management. | Reviews features (text, network, user), shows stance ≠ sentiment | Motivates not using sentiment as a proxy |
| **Allaway & McKeown, 2020.** *Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations* (VAST). EMNLP. | Zero-shot stance on unseen targets | Supports framing stance as an NLI-style text-pair task |
| **Hardalov et al., 2021.** *Cross-Domain Label-Adaptive Stance Detection.* EMNLP. | 16 stance datasets; label-space mismatch across domains | Our label mapping across GWSD, SemEval and Kaggle follows the same reasoning |
| **Hardalov et al., 2022.** *Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-Training.* AAAI. | Cross-lingual stance over 15 datasets in 12 languages | Closest methodological prior for English → Indic transfer |
| **Vamvas & Sennrich, 2020.** *X-Stance: A Multilingual Multi-Target Dataset for Stance Detection.* SwissText. | Multilingual BERT transfers stance across de/fr/it | Evidence that a multilingual encoder plus the target as a second segment works cross-lingually |

## 2.2 Climate-change stance and opinion

| Work | Contribution | Relevance |
|---|---|---|
| **Luo, Card & Jurafsky, 2020.** *Detecting Stance in Media on Global Warming* (GWSD / DeSMOG). Findings of EMNLP. | 2,300 news sentences labelled agree / neutral / disagree with **"Climate change is a serious concern"**; BERT reaches 0.75 accuracy / 0.73 macro-F1, comparable to humans (0.71) | **Same claim**, so its labels map 1:1 to ours. The project title is taken from this paper |
| **Kaggle: Twitter Climate Change Sentiment Dataset** (E. Qian, 2018/19). | 43,943 tweets labelled pro / anti / neutral / news on man-made climate change | Largest noisy social-media source; "pro/anti" is stance, despite the "sentiment" name |
| **Varini et al., 2020.** *ClimaText: A Dataset for Climate Change Topic Detection.* | Sentence-level climate *relevance* | Relevance is the first half of our None-vs-stance decision |
| **Diggelmann et al., 2020.** *CLIMATE-FEVER.* | Claim verification on climate claims | Optional NLI-style auxiliary data |
| **Webersinke et al., 2021.** *ClimateBERT.* | Domain-adaptive pre-training on climate text (English) | Domain adaptation idea; English-only |
| **Vaid et al., 2022.** *Towards Fine-grained Classification of Climate Change related Social Media Text.* ACL SRW. | Climate tweets labelled for stance, category, sentiment | More social-media climate stance data |
| **Wang et al., 2024.** *MultiClimate: Multimodal Stance Detection on Climate Change Videos.* NLP4PI. | YouTube-video stance (English) | Same platform; shows video context matters, though we only have comment text |
| **Dhoroni (2024).** arXiv:2410.17225. | Bengali climate *news* dataset with a stance task; BanglaBERT baselines | Only Bengali climate-stance resource we found; news domain; low reported F1 shows the task is hard |

**Gap 1:** All sizeable climate-stance corpora are English. Hindi climate-stance data appears not to exist publicly, and Bengali has only news (Dhoroni). Nothing covers Indic *YouTube comments*.

## 2.3 Hindi, Bengali and Indic NLP

| Work | Contribution | Relevance |
|---|---|---|
| **Kakwani et al., 2020.** *IndicNLPSuite* (IndicBERT v1, IndicCorp). Findings of EMNLP. | ALBERT for 12 Indic languages | Baseline Indic encoder |
| **Doddapaneni et al., 2023.** *Towards Leaving No Indic Language Behind* (IndicBERT v2, IndicXTREME). ACL. | Larger IndicCorp v2; IndicBERT v2 beats XLM-R on many Indic tasks | Candidate encoder `ai4bharat/IndicBERTv2-MLM-only` |
| **Khanuja et al., 2021.** *MuRIL: Multilingual Representations for Indian Languages.* arXiv. | BERT trained on 17 Indian languages **plus their transliterated (romanised) forms** | Promising for romanised Hinglish/Banglish |
| **Bhattacharjee et al., 2022.** *BanglaBERT.* Findings of NAACL. | Strong Bengali-only encoder | Language-specific specialist for an ensemble |
| **Gala et al., 2023.** *IndicTrans2.* TMLR. | State-of-the-art en↔22 Indic languages translation | Alternative translator for translate-train |
| **NLLB Team, 2022.** *No Language Left Behind.* | 200-language MT; `nllb-200-distilled-600M` fits on a 4 GB GPU | Our default translate-train engine |
| **Islam et al., 2021.** *SentNoB: Sentiment on Noisy Bangla Texts.* Findings of EMNLP. | Noisy Bengali social-media comments | Documents Bengali social-media noise; possible auxiliary task |
| **Swami et al., 2018.** *An English-Hindi Code-Mixed Corpus: Stance Annotation and Baseline System.* arXiv:1805.11868. | Stance on code-mixed Hindi-English tweets | Only Hinglish stance prior; different target |

## 2.4 Code-mixed and low-resource NLP

| Work | Contribution | Relevance |
|---|---|---|
| **Das & Gambäck, 2014; Gambäck & Das, 2016.** *Code-Mixing Index (CMI).* | Measure of how mixed an utterance is | Implemented in `stance/text.py` for our EDA |
| **Khanuja et al., 2020.** *GLUECoS.* ACL. | Code-switching benchmark (Hi-En) | Multilingual models degrade on code-mixed text |
| **Patwa et al., 2020.** *SemEval-2020 Task 9: SentiMix.* | Hinglish/Spanglish sentiment | Romanised Hinglish data, usable for augmentation or domain-adaptive pre-training |
| **Wu & Dredze, 2019.** *Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT.* EMNLP. | Freezing lower layers and embeddings helps zero-shot cross-lingual transfer | We freeze the word embeddings by default |
| **Hu et al., 2020.** *XTREME.* ICML. | Translate-train usually beats zero-shot transfer | Motivates RQ2 |
| **Conneau et al., 2020.** *XLM-R: Unsupervised Cross-lingual Representation Learning at Scale.* ACL. | Strongest general multilingual encoder at base size | Main backbone |

## 2.5 Transformers and LLMs for stance

| Work | Contribution | Relevance |
|---|---|---|
| **Devlin et al., 2019 (BERT); He et al., 2023 (DeBERTa-v3).** | Encoder fine-tuning as the standard approach | Backbones |
| **Laurer et al., 2024.** *Less Annotating, More Classifying: … BERT-NLI.* Political Analysis. | Recasting classification as NLI improves low-data performance; multilingual NLI checkpoints | Our `zero_shot_nli.py` and NLI-initialised fine-tuning |
| **Zhang, Ding & Jing, 2022.** *How would Stance Detection Techniques Evolve after the Launch of ChatGPT?* arXiv. | Zero-shot ChatGPT matches or beats supervised SOTA on SemEval-2016 | LLMs as zero-shot annotators or teachers |
| **Cruickshank & Ng, 2024.** *Prompting and Fine-Tuning Open-Sourced LLMs for Stance Classification.* | LLM performance varies widely across datasets and prompts | LLMs are not a guaranteed win and need careful prompting |
| **Hu et al., 2022.** *LoRA.* ICLR. | Parameter-efficient fine-tuning | `--lora` option for limited GPUs |
| **Ribeiro et al., 2020.** *CheckList.* ACL. | Behavioural testing with perturbation and invariance tests | Our robustness suite (`stance/robustness.py`) |

## 2.6 Synthesis: the research gap

1. **No in-language training data.** Climate-stance work is English. Indic stance work covers other targets or other domains.
2. **Domain mismatch.** The best-matched dataset (GWSD) is *news*, while the test set is *YouTube comments*.
3. **Romanised and code-mixed input** is rarely evaluated in cross-lingual stance. Encoders trained mostly on native script (XLM-R) lose accuracy on it.
4. **Evaluation is usually a single in-distribution score.** Few papers report robustness to negation, sarcasm or romanisation, which is exactly what the hidden test will probe.

**Our contribution:** a cross-lingual pipeline that combines (a) target-aware multilingual encoders, (b) translate-train into Hindi and Bengali, (c) noise-aware augmentation that simulates romanisation and code-mixing, (d) stance-as-NLI, and (e) a phenomenon-tagged Hindi/Bengali challenge set with a CheckList-style robustness evaluation.
