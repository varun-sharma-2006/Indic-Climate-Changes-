# 6. Experiment log

Fill one row per run. Numbers come from `runs/<name>/metrics.json`. **CS** = our Hindi/Bengali challenge set (64 items, diagnostic only).

| # | Date | Run name | Train data | Model | Key settings | EN valid macro-F1 | CS macro-F1 | CS hi | CS bn | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-13 | `tfidf_lr_en_only` | GWSD + SemEval (en, 2,241) | TF-IDF char2-5 + word1-2, LR | class_weight=balanced, C=2 | **0.602** | **0.330** | 0.19 | 0.13 | Predicts *None* for most native-script Indic text (no lexical overlap). Romanised rows score better because they contain English climate words |
| 2 | | `zero_shot_nli` | none | mDeBERTa-v3-base XNLI | τ=0.5 | | | | | |
| 3 | | `xlmr_en_only` | en_only | XLM-R base | pair+claim(match), frozen emb, sqrt weights | | | | | |
| 4 | | `muril_en_only` | en_only | MuRIL base | same | | | | | |
| 5 | | `xlmr_en_kaggle` | + Kaggle | XLM-R base | same | | | | | |
| 6 | | `xlmr_translate_train` | + NLLB hi/bn | XLM-R base | sampler=lang_label | | | | | |
| 7 | | `xlmr_tt_aug` | + augmentation | XLM-R base | | | | | | |
| 8 | | `ensemble` | — | 3/5/7 + NLI | mean probs | | | | | |

## Observations so far

1. **The lexical gap is almost total.** 99.5% of word types (79% of tokens) in Hindi/Bengali comments never occur in the English training data (`reports/eda/external_en/report.md`). Word-level models cannot transfer.
2. **The TF-IDF baseline fails in a revealing way.** It labels 37 of the 49 Favour/Against challenge items as *None*, because with no known features it falls back to the class prior. Its perturbation results are also telling: swapping Hindi/Bengali climate words for English ones (*codemix* test) *raises* its macro-F1 from 0.33 to 0.49. Any useful model has to map Indic surface forms onto the English-learned stance features.
3. **The training data has label skew and shortcut risks.** *Against* is 17% of the data. In English, *Against* has 2× the negation rate of the other classes (0.22 vs 0.09–0.10), and the strongest Against cue by far is "hoax" (log-odds z = 6.7). Models will likely learn "negation or hoax → Against". The challenge set's negation items (Favour with negation) test exactly this.
4. **Genre gap.** GWSD is news (no emoji, no URLs). Kaggle tweets bring social-media noise, which is why we recommend downloading them.
