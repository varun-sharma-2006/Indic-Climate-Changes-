"""Loading the processed datasets and computing evaluation metrics."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import FIG_DIR, LABEL2ID, LABELS, METRICS_DIR, PROCESSED_DIR
from preprocess import clean_text

FILES = {"en": "english.csv", "hi": "hindi.csv", "bn": "bengali.csv",
         "hr": "hindi_roman.csv", "br": "bengali_roman.csv"}


def available_test_langs():
    """en, hi, bn plus the romanised sets if they have been generated."""
    return [l for l in FILES if (PROCESSED_DIR / FILES[l]).exists()]


def load_lang(lang: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / FILES[lang], keep_default_na=False)  # keep the label "None" as text
    df["lang"] = lang
    df["clean"] = df["text"].astype(str).map(clean_text)
    df["y"] = df["label"].map(LABEL2ID)
    return df


def load_split(langs, split) -> pd.DataFrame:
    return pd.concat([load_lang(l)[lambda d: d["split"] == split] for l in langs], ignore_index=True)


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "macro_f1": round(float(mf), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(mp), 4),
        "macro_recall": round(float(mr), 4),
        "per_class_f1": {LABELS[i]: round(float(f[i]), 4) for i in range(len(LABELS))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
    }


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    plt.figure(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_results(tag: str, results: dict, info: dict):
    """results = {lang: metrics dict}; writes results/metrics/<tag>.json and confusion matrices."""
    with open(METRICS_DIR / f"{tag}.json", "w", encoding="utf8") as f:
        json.dump({"experiment": tag, **info, "results": results}, f, indent=2)
    for lang, m in results.items():
        plot_confusion(m["confusion_matrix"], f"{tag} ({lang})", FIG_DIR / f"cm_{tag}_{lang}.png")
    summary = "  ".join(f"{l}: F1={m['macro_f1']:.3f} Acc={m['accuracy']:.3f}" for l, m in results.items())
    print(f"[{tag}] {summary}")
