# 4. Current state of the art (related problems)

| Aspect | Current practice | Strengths | Limitations for *our* setting |
|---|---|---|---|
| **Models** | Fine-tuned encoders (BERT, RoBERTa, DeBERTa-v3; multilingual XLM-R, mBERT, MuRIL, IndicBERT v2, BanglaBERT); zero/few-shot LLMs (GPT-4-class, Llama/Qwen/Mistral); NLI-based zero-shot classifiers | Encoders are cheap and strong in-domain; LLMs handle implicit stance and sarcasm better | Encoders overfit to source-language lexical cues; LLMs are costly, prompt-sensitive and weaker in Bengali; most results are English-only |
| **Datasets** | SemEval-2016 T6, P-Stance, VAST, COVID-stance, WT-WT, X-Stance; climate: GWSD, SemEval climate target, Kaggle climate tweets, ClimaConvo, MultiClimate | Standard benchmarks | Almost no Indic or YouTube-comment stance data |
| **Preprocessing** | Unicode normalisation, URL and @user masking, hashtag segmentation, emoji kept or converted to text, lowercasing (Latin only) | Reduces sparsity | Aggressive cleaning removes stance cues (emoji, "!!!", negation). Indic-specific: nukta/ZWJ normalisation, script-aware handling |
| **Input format** | Target-aware text pair `[CLS] text [SEP] target [SEP]` (Vamvas & Sennrich 2020; Allaway & McKeown 2020) | Lets one model reason about the text-target relation | Small gain for a single fixed target; useful when the claim is translated into the comment's language |
| **Training strategies** | Class-weighted or focal loss; multi-task with sentiment or sarcasm; domain-adaptive pre-training (ClimateBERT-style); translate-train / translate-test; data augmentation (back-translation, EDA); adversarial training; LLM-generated pseudo-labels or knowledge distillation | — | Translation errors flip stance (especially negation); augmentation may create label noise |
| **Evaluation** | SemEval: F_avg = mean F1(Favour, Against); otherwise macro-F1 over 3 classes, accuracy; increasingly robustness and calibration | — | Single-split evaluation hides failure modes; we add phenomenon-wise and perturbation-wise evaluation |

## Reference numbers (for context, not directly comparable)

| Setting | Reported score | Source |
|---|---|---|
| GWSD, BERT-base (weighted) | 0.75 acc / 0.73 macro-F1; human 0.71 acc | Luo et al., 2020 |
| GWSD, LR uni+bigrams | 0.62 acc / 0.60 macro-F1 | Luo et al., 2020 |
| SemEval-2016 climate target, best supervised systems | per-target F_avg below 0.5; the hardest target because Against has very few examples | Mohammad et al., 2016 and participant papers |
| **Ours, TF-IDF LR, English valid** | 0.60 macro-F1 | `runs/tfidf_lr_en_only` |
| **Ours, TF-IDF LR, Hindi/Bengali challenge set** | 0.33 macro-F1 (bn 0.13, hi 0.19) | same run |

Our English TF-IDF baseline is in the same range as the published GWSD LR number (0.60). The evaluation sets differ (ours = GWSD test + SemEval test), so this is only a sanity check.
