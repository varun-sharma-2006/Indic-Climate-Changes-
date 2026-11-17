# Team Contributions

We discussed and worked on every part of the project together. Each member took **primary responsibility** for one area: they implemented it, maintain it and explain it in depth, while helping on the others.

| Member | Enrolment No. | Responsibility | Files owned |
|---|---|---|---|
| Varun Sharma | S24CSEU0170 | Data collection, label mapping, translation, YouTube comment collection and labelling | `src/download_data.py`, `src/translate.py`, `src/collect_youtube.py`, `src/data_stats.py`, `src/romanize_data.py`, `src/extra_data.py`, `data/synthetic/` |
| Yashika Garg | S24CSEU0235 | Preprocessing, TF-IDF baselines, evaluation metrics | `src/preprocess.py`, `src/baselines.py`, `src/data_utils.py`, `src/error_analysis.py`, `src/sanity_check.py`, `src/robustness_report.py` |
| Priyanshu Raj | S24CSEU1142 | Transformer fine-tuning (BERT, mBERT, XLM-R, MuRIL), zero-shot vs translate-train experiments | `src/train_transformer.py`, `src/models.py`, `src/config.py` |
| Advay Sinha | S24CSEU2059 | Ensemble, explainable AI (LIME), prediction script, results summary, report | `src/ensemble.py`, `src/explain_lime.py`, `src/predict.py`, `src/summarize_results.py`, `src/eval_challenge.py`, `src/evaluate_saved.py`, `docs/make_progress_report.py`, `app/` (web demo) |

Every member should be able to explain the whole pipeline (see the README), not only their own part.
