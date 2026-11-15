"""CheckList-style robustness test: apply label-preserving perturbations and measure
(a) macro-F1 drop and (b) prediction flip rate (fraction of predictions that change).

Usage:
    python -m stance.robustness --model runs/xlmr_full/best --input data/processed/full/valid.csv data/challenge/challenge_hi_bn.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from .augment import PERTURBATIONS, sanscript
from .config import LABELS, SEED
from .data_io import load_unified
from .metrics import compute
from .predictors import load_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--max-rows", type=int, default=1000)
    a = ap.parse_args()
    random.seed(SEED)
    pred = load_predictor(a.model)
    ops = [o for o in PERTURBATIONS if not (o == "romanize" and sanscript is None)]
    rows = []
    for path in a.input:
        df = load_unified(path)
        df = df[df["label"].notna()]
        if len(df) > a.max_rows:
            df = df.sample(a.max_rows, random_state=SEED)
        texts, langs = df["text"].tolist(), df["lang"].tolist()
        base = pred.predict_proba(texts, langs).argmax(1)
        base_f1 = compute(df["label"], [LABELS[i] for i in base])["macro_f1"]
        rows.append({"set": Path(path).stem, "perturbation": "original", "macro_f1": round(base_f1, 4), "delta": 0.0, "flip_rate": 0.0})
        for op in ops:
            pt = [PERTURBATIONS[op](t) for t in texts]
            changed = [x != y for x, y in zip(texts, pt)]
            p = pred.predict_proba(pt, langs).argmax(1)
            f1 = compute(df["label"], [LABELS[i] for i in p])["macro_f1"]
            flips = sum(int(b != q) for b, q, c in zip(base, p, changed) if c) / max(sum(changed), 1)
            rows.append({"set": Path(path).stem, "perturbation": op, "macro_f1": round(f1, 4),
                         "delta": round(f1 - base_f1, 4), "flip_rate": round(flips, 4), "n_changed": sum(changed)})
            print(rows[-1])
    res = pd.DataFrame(rows)
    out = Path(a.model)
    out = (out.parent if out.name == "best" else out) / "robustness.csv"
    res.to_csv(out, index=False)
    print("\n" + res.to_string(index=False))
    print(f"\n[robustness] -> {out}")


if __name__ == "__main__":
    main()
