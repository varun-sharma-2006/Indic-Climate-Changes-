"""Classical baselines: TF-IDF (char 2-5 + word 1-2 n-grams) + Logistic Regression / Linear SVM.

Char n-grams are robust to spelling variation and work across scripts without a tokenizer, which
makes them a strong low-resource baseline. They CANNOT transfer English -> Hindi/Bengali (no shared
surface forms) - measuring exactly how badly they fail is itself an informative result.

Usage:
    python -m stance.baseline --data data/processed/en_only --eval data/challenge/challenge_hi_bn.csv
"""
import argparse
import pickle
import time
from pathlib import Path

from .config import LABELS, RUNS
from .data_io import load_unified
from .metrics import by_group, compute, confusion, dump, pretty
from .predictors import write_predictions
from .tfidf import TfidfModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train.csv and valid.csv (from stance.build_dataset)")
    ap.add_argument("--clf", default="lr", choices=["lr", "svm"])
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--eval", nargs="*", default=[], help="extra labelled sets to evaluate (e.g. challenge set)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    data = Path(a.data)
    name = a.name or f"tfidf_{a.clf}_{data.name}"
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    tr, va = load_unified(data / "train.csv"), load_unified(data / "valid.csv")
    tr = tr[tr["label"].notna()]
    print(f"[baseline] train {len(tr)}  valid {len(va)}  -> {out}")

    t0 = time.time()
    model = TfidfModel(a.clf, a.C).fit(tr["text"].tolist(), tr["label"].tolist())
    print(f"[baseline] fitted in {time.time() - t0:.1f}s")
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    results = {}
    for tag, path in [("valid", data / "valid.csv")] + [(Path(p).stem, p) for p in a.eval]:
        df = load_unified(path)
        probs = model.predict_proba(df["text"].tolist())
        pdf = write_predictions(df, probs, out / f"pred_{tag}.csv")
        g = pdf[pdf["label"].isin(LABELS)]
        if len(g):
            m = compute(g["label"], g["pred"])
            results[tag] = m
            print(pretty(m, f"{name} on {tag}"))
            print(confusion(g["label"], g["pred"]).to_string())
            print(by_group(g, "lang").to_string(index=False), "\n")
    dump(results, out / "metrics.json")


if __name__ == "__main__":
    main()
