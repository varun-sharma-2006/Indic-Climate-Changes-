# 5. Proposed methodology (Round 1, work in progress)

## 5.1 Overview

```
                    ┌────────────── English labelled data ───────────────┐
                    │  GWSD (news)  +  SemEval-16 climate  +  Kaggle tweets │
                    └──────────────┬──────────────────────────────────────┘
                                   │ label harmonisation (Favour/Against/None), dedup, leak-free split
          ┌────────────────────────┼─────────────────────────────┐
          ▼                        ▼                             ▼
   (A) zero-shot X-lingual   (B) translate-train            (C) noise augmentation
   train on EN only          NLLB-200 en→hi, en→bn          romanise · code-mix swap
                             (+ official data if any)       · typos · emoji/URL/elongation
          └────────────────────────┼─────────────────────────────┘
                                   ▼
          Target-aware multilingual encoder  [CLS] comment [SEP] claim(lang) [SEP]
          XLM-R-base / MuRIL / IndicBERT-v2 / mDeBERTa-NLI-init
          frozen word embeddings · class-weighted CE + label smoothing · lang-balanced sampler
                                   │
          (D) zero-shot stance-as-NLI (mDeBERTa-XNLI) ─────┐
                                   ▼                         ▼
                     probability-averaging ensemble  (E)
                                   ▼
     Evaluation: macro-F1 (primary) · per-class F1 · MCC · per-language
                 · phenomenon-wise (challenge set) · robustness (flip rate) · calibration
     Explainability: occlusion word attributions · confident-error analysis
```

## 5.2 Components and why we believe they will work

| Id | Component | Hypothesis | Tested by |
|---|---|---|---|
| A | **Zero-shot cross-lingual transfer** with multilingual encoders | Shared multilingual representations carry stance from English to Hindi/Bengali (X-Stance, XTREME). This is the lower bound for neural models | `train.py` on `en_only`, evaluated on the challenge set |
| B | **Translate-train** (NLLB-200) | Seeing Hindi/Bengali surface forms during training gives large gains (XTREME). Risk: translation drops negation, so we check 100 translations by hand | `translate.py` → `build_dataset` → `train.py` |
| C | **Noise-aware augmentation** | The test set is noisy (romanisation, code-mixing). Training on simulated noise reduces flip rate | `augment.py`; `robustness.py` flip-rate before and after |
| — | **Target-aware input**, claim in the comment's language (`--claim-lang match`) | Gives the model an explicit anchor for *relation-to-claim*, which is what separates None from topical comments | ablation: `--pair-with-claim` on/off, `en` vs `match` |
| — | **Frozen word embeddings** | Keeps the cross-lingual embedding alignment intact (Wu & Dredze 2019) **and** fits XLM-R-base on a 4 GB GPU (≈ 86M instead of 278M trainable params) | ablation `--no-freeze-embeddings` on Colab |
| — | **Class-weighted CE + label smoothing** | *Against* is a 5–17% minority; smoothing absorbs noise from translation and annotation | ablation `--class-weights none` |
| D | **Stance-as-NLI** | Stance toward a claim is close to entailment/contradiction. NLI models are trained on XNLI (which includes Hindi) and should handle negation better | `zero_shot_nli.py`; then fine-tune *starting from* the NLI checkpoint (`--model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) |
| E | **Ensemble** (XLM-R + MuRIL + NLI) | Different pre-training data (native vs romanised script, NLI) gives complementary errors | `ensemble.py` |
| — | **Explainability & error analysis** | Shows whether the model relies on real stance cues or shortcuts (e.g. "hoax" → Against, any negation → Against) | `explain.py`, `error_analysis.py` |

## 5.3 Planned next steps (after Round 1)

1. **Domain-adaptive MLM pre-training** (DAPT) of XLM-R/MuRIL on unlabelled Hindi/Bengali social-media text (SentNoB, SentiMix, scraped climate-video comments).
2. **LLM teacher:** label a few thousand unlabelled Hindi/Bengali climate comments with a strong LLM (few-shot, chain-of-thought), keep high-agreement items, and train the encoder on them (knowledge distillation / self-training).
3. **Translate-test** as a further ensemble member: translate Hindi/Bengali test comments to English and apply an English stance model trained on all English data.
4. **Multi-task learning** with sentiment and sarcasm heads (SentNoB, SentiMix).
5. Grow the challenge set to about 300 items with two annotators per language, and report Cohen's κ.

## 5.4 Experimental protocol

* Seeds: 42, 43 and 44 for every neural configuration; report mean ± std.
* Model selection: macro-F1 on the official validation split (once available), otherwise on the **English held-out set**. Never on the challenge set, because selecting on it would leak information.
* All runs log arguments, history and metrics to `runs/<name>/` (`train_args.json`, `history.csv`, `metrics.json`, `pred_*.csv`).
* Results are collected in [06_experiment_log.md](06_experiment_log.md).

## 5.5 Hardware budget

The local RTX 2050 has 4 GB of VRAM. Base-size encoders fit with frozen embeddings, fp16, batch 8 and gradient accumulation 2, at max_len 128. Large models (XLM-R-large, MuRIL-large) and LLMs run on Google Colab or Kaggle T4/P100 GPUs.
