"""Shared evaluation: macro-F1 (primary), per-class P/R/F1, accuracy, MCC, per-language breakdown."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef)

from .config import LABELS


def compute(y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        # SemEval-2016 official metric: mean F1 of Favour and Against only
        "f1_favour_against": float(np.mean([rep[l]["f1-score"] for l in ["Favour", "Against"]])),
        "per_class": {l: {k: round(rep[l][k], 4) for k in ["precision", "recall", "f1-score", "support"]} for l in LABELS},
        "n": len(y_true),
    }


def by_group(df: pd.DataFrame, group_col: str, true_col="label", pred_col="pred") -> pd.DataFrame:
    rows = []
    for g, s in df.groupby(group_col):
        s = s[s[true_col].isin(LABELS)]
        if len(s) == 0:
            continue
        m = compute(s[true_col], s[pred_col])
        rows.append({group_col: g, "n": len(s), "macro_f1": round(m["macro_f1"], 4), "accuracy": round(m["accuracy"], 4)})
    return pd.DataFrame(rows)


def pretty(m: dict, title="") -> str:
    lines = [f"== {title} ==" if title else "",
             f"macro-F1 {m['macro_f1']:.4f} | acc {m['accuracy']:.4f} | MCC {m['mcc']:.4f} | F1(fav,ag) {m['f1_favour_against']:.4f} | n={m['n']}"]
    for l in LABELS:
        p = m["per_class"][l]
        lines.append(f"  {l:8s} P {p['precision']:.3f}  R {p['recall']:.3f}  F1 {p['f1-score']:.3f}  (n={int(p['support'])})")
    return "\n".join(lines)


def confusion(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(list(y_true), list(y_pred), labels=LABELS),
                        index=[f"gold:{l}" for l in LABELS], columns=[f"pred:{l}" for l in LABELS])


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
