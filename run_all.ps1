# Runs the full pipeline in order (Windows PowerShell). Takes about 1.5-2 hours on an RTX 2050.
# Activate the environment first:  .\.venv\Scripts\Activate.ps1
$ErrorActionPreference = "Stop"

python src/download_data.py                      # 1. English data (GWSD + ClimateStance)
python src/translate.py --langs hi bn            # 2. Hindi + Bengali translations
python src/data_stats.py                         # 3. statistics and plots
python src/baselines.py                          # 4a. TF-IDF baselines

# 4b. base paper setting: English BERT trained on English only
python src/train_transformer.py --model bert --train_langs en
# 4c. zero-shot cross-lingual: multilingual models trained on English only
python src/train_transformer.py --model xlmr --train_langs en
python src/train_transformer.py --model muril --train_langs en --epochs 6
python src/train_transformer.py --model mbert --train_langs en
# 4d. translate-train (our method)
python src/train_transformer.py --model mbert --train_langs en hi bn
python src/train_transformer.py --model xlmr --train_langs en hi bn --save
python src/train_transformer.py --model muril --train_langs en hi bn --epochs 6 --save

# 5. ensemble of the translate-train models
python src/ensemble.py --tags mbert_tt xlmr_tt muril_tt --name ensemble_tt
python src/ensemble.py --tags xlmr_tt muril_tt --name ensemble2_tt

# 6. explainability and summary
python src/llm_fewshot.py                        # 4e. LLM few-shot baseline
python src/error_analysis.py --tag muril_tt
python src/eval_challenge.py --model_dirs models/xlmr_tt models/muril_tt
python src/explain_lime.py --model_dir models/xlmr_tt --n 3 --num_samples 300
python src/summarize_results.py
