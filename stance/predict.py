"""Generate predictions for any file (e.g. the hidden test set) with a trained model.

Usage:
    python -m stance.predict --model runs/xlmr_full/best --input data/official/test.csv --output submissions/test_pred.csv
    python -m stance.predict --model runs/tfidf_lr_full --input data/processed/official_test.csv

Writes a full file (with probabilities) and a minimal `<output>_submission.csv` with columns id,label.
"""
import argparse
from pathlib import Path

from .data_io import read_table, to_unified
from .predictors import load_predictor, write_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="transformer dir (runs/x/best) or TF-IDF run dir / model.pkl")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="submissions/predictions.csv")
    ap.add_argument("--text-col")
    ap.add_argument("--id-col")
    a = ap.parse_args()

    raw = read_table(a.input)
    df = to_unified(raw, source=Path(a.input).stem, text_col=a.text_col, id_col=a.id_col)
    pred = load_predictor(a.model)
    probs = pred.predict_proba(df["text"].tolist(), df["lang"].tolist())
    out = write_predictions(df, probs, a.output)
    sub = Path(a.output).with_name(Path(a.output).stem + "_submission.csv")
    out[["id", "pred"]].rename(columns={"pred": "label"}).to_csv(sub, index=False, encoding="utf-8")
    print(f"[predict] submission file -> {sub}")
    print(out["pred"].value_counts().to_string())


if __name__ == "__main__":
    main()
