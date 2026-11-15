"""Shared paths, label names and settings used by every script."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDS_DIR = RESULTS_DIR / "predictions"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# keep downloaded Hugging Face models inside the project folder (C: drive is nearly full)
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

LABELS = ["Favour", "Against", "None"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

LANGS = {"en": "English", "hi": "Hindi", "bn": "Bengali"}
ROMAN_LANGS = {"hr": "Hindi (romanised)", "br": "Bengali (romanised)"}  # made by src/romanize_data.py

CLAIM = "Climate change and global warming is a serious concern."

# short names for the transformer models we compare
MODEL_NAMES = {
    "bert": "bert-base-uncased",                    # base paper's model (English only)
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
}

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


for d in (RAW_DIR, PROCESSED_DIR, METRICS_DIR, PREDS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
