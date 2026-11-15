"""Make the submission file: ID, Predicted Label.

Input CSV must have an ID column and a text column (names can be set).
Usage:
  python src/predict.py --model_dir models/muril_tt --input data/test.csv --output results/predictions.csv
  python src/predict.py --model_dir models/muril_tt --text "ग्लोबल वार्मिंग एक गंभीर समस्या है"
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer

from config import LABELS
from models import load_trained
from preprocess import clean_text


class StancePredictor:
    def __init__(self, model_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = load_trained(model_dir).to(self.device).eval()

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64, max_len=128):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [clean_text(t) for t in texts[i:i + batch_size]]
            enc = self.tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu())
        return torch.cat(out).numpy()

    def predict(self, texts):
        return [LABELS[i] for i in self.predict_proba(list(texts)).argmax(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input")
    ap.add_argument("--output", default="results/submission.csv")
    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--text", help="classify a single comment")
    args = ap.parse_args()

    predictor = StancePredictor(args.model_dir)
    if args.text:
        probs = predictor.predict_proba([args.text])[0]
        print(f"{LABELS[probs.argmax()]}  " + "  ".join(f"{l}={p:.2f}" for l, p in zip(LABELS, probs)))
        return

    df = pd.read_csv(args.input)
    df["Predicted Label"] = predictor.predict(df[args.text_col].astype(str).tolist())
    df[[args.id_col, "Predicted Label"]].rename(columns={args.id_col: "ID"}).to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions -> {args.output}")
    print(df["Predicted Label"].value_counts())


if __name__ == "__main__":
    main()
