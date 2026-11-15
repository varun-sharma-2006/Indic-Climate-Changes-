"""Uniform predictor interface (texts -> probability matrix over LABELS) for every model type,
used by predict / robustness / explain / ensemble."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS


def write_predictions(df: pd.DataFrame, probs: np.ndarray, path) -> pd.DataFrame:
    out = df.copy()
    out["pred"] = [LABELS[i] for i in probs.argmax(1)]
    for i, l in enumerate(LABELS):
        out[f"prob_{l}"] = probs[:, i].round(5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"[pred] wrote {path}")
    return out


class TransformerPredictor:
    def __init__(self, model_dir, batch_size=32, max_len=None):
        import json

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.dir = Path(model_dir)
        cfg = json.loads((self.dir / "train_args.json").read_text(encoding="utf-8"))
        self.pair = cfg.get("pair_with_claim", False)
        self.claim_lang = cfg.get("claim_lang", "en")
        self.max_len = max_len or cfg.get("max_len", 128)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir).to(self.dev).eval()
        self.bs = batch_size

    def encode(self, texts, langs=None):
        from .train import build_inputs
        return build_inputs(self.tok, texts, langs, self.pair, self.claim_lang, self.max_len)

    def predict_proba(self, texts, langs=None):
        from tqdm.auto import tqdm
        probs = []
        for i in tqdm(range(0, len(texts), self.bs), desc="predict", leave=False, disable=len(texts) < 200):
            enc = self.encode(texts[i:i + self.bs], None if langs is None else langs[i:i + self.bs])
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            with self.torch.no_grad(), self.torch.autocast(self.dev, enabled=self.dev == "cuda"):
                logits = self.model(**enc).logits.float()
            probs.append(self.torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros((0, len(LABELS)))


class PickledPredictor:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def predict_proba(self, texts, langs=None):
        return self.model.predict_proba(list(texts))


def load_predictor(path):
    p = Path(path)
    if p.suffix == ".pkl":
        return PickledPredictor(p)
    if p.is_dir() and (p / "model.pkl").exists():
        return PickledPredictor(p / "model.pkl")
    if p.is_dir() and (p / "best").exists():
        return TransformerPredictor(p / "best")
    return TransformerPredictor(p)
