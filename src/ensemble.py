"""Step 5: ensemble = average of the class probabilities of several fine-tuned models.

Usage:  python src/ensemble.py --tags mbert_tt xlmr_tt muril_tt --name ensemble_tt
(each tag must have prediction files in results/predictions/)
"""
import argparse

import pandas as pd

from config import LABELS, PREDS_DIR
from data_utils import compute_metrics, save_results

TEST_LANGS = ["en", "hi", "bn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--name", default="ensemble")
    args = ap.parse_args()

    cols = [f"p_{l}" for l in LABELS]
    results = {}
    for lang in TEST_LANGS:
        frames = [pd.read_csv(PREDS_DIR / f"{t}_{lang}.csv") for t in args.tags]
        probs = sum(f[cols].values for f in frames) / len(frames)
        results[lang] = compute_metrics(frames[0]["y"], probs.argmax(1))
        out = frames[0][["id", "y"]].copy()
        out[cols] = probs
        out.to_csv(PREDS_DIR / f"{args.name}_{lang}.csv", index=False)
    save_results(args.name, results, {"model": "average of " + ", ".join(args.tags), "members": args.tags})


if __name__ == "__main__":
    main()
