"""Probability-averaging ensemble over prediction files that share the same rows (same `id` order).

Usage:
    python -m stance.ensemble --preds runs/xlmr_full/pred_valid.csv runs/muril_full/pred_valid.csv --weights 0.6 0.4 \
        --output runs/ensemble/pred_valid.csv
"""
import argparse

import numpy as np
import pandas as pd

from .config import LABELS
from .metrics import compute, pretty
from .predictors import write_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.preds))
    w = w / w.sum()
    frames = [pd.read_csv(p, keep_default_na=False, na_values=[""], dtype={"id": str}) for p in a.preds]
    base = frames[0]
    for f, p in zip(frames[1:], a.preds[1:]):
        if not (f["id"].values == base["id"].values).all():
            raise SystemExit(f"row ids differ between {a.preds[0]} and {p}")
    cols = [f"prob_{l}" for l in LABELS]
    probs = sum(wi * f[cols].values for wi, f in zip(w, frames))
    keep = [c for c in base.columns if c not in cols + ["pred"]]
    out = write_predictions(base[keep], probs, a.output)
    g = out[out["label"].isin(LABELS)]
    for f, p in zip(frames, a.preds):
        fg = f[f["label"].isin(LABELS)]
        if len(fg):
            print(f"{compute(fg['label'], fg['pred'])['macro_f1']:.4f}  {p}")
    if len(g):
        print(pretty(compute(g["label"], g["pred"]), "ENSEMBLE"))


if __name__ == "__main__":
    main()
