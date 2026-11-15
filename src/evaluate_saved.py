"""Evaluate a saved model on the test split of any languages (e.g. the romanised sets).

Usage:  python src/evaluate_saved.py --model_dir models/muril_tt --langs hr br
Output: printed metrics + results/metrics_extra/<model>_<langs>.json
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse
import json
from pathlib import Path

from config import LABELS, RESULTS_DIR
from data_utils import compute_metrics, load_split
from predict import StancePredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--langs", nargs="+", default=["hr", "br"])
    args = ap.parse_args()

    predictor = StancePredictor(args.model_dir)
    results = {}
    for lang in args.langs:
        test = load_split([lang], "test")
        results[lang] = compute_metrics(test["y"], predictor.predict_proba(test["text"].tolist()).argmax(1))
        r = results[lang]
        print(f"{lang}: macro F1 {r['macro_f1']:.3f}  acc {r['accuracy']:.3f}  per class {r['per_class_f1']}")

    out_dir = RESULTS_DIR / "metrics_extra"
    out_dir.mkdir(exist_ok=True)
    name = Path(args.model_dir).name
    with open(out_dir / f"{name}_{'_'.join(args.langs)}.json", "w", encoding="utf8") as f:
        json.dump({"model_dir": str(args.model_dir), "labels": LABELS, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
