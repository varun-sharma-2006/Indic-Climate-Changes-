"""Collect every results/metrics/*.json into one table (results/results_table.md + .csv)
and draw a comparison chart of Macro F1 per language.

Usage:  python src/summarize_results.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DIR, METRICS_DIR, RESULTS_DIR

SETTING_NAMES = {"en": "English only", "tt": "Translate-train (En+Hi+Bn)"}


def main():
    rows = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        m = json.loads(path.read_text(encoding="utf8"))
        langs = m.get("train_langs")
        if "members" in m:
            setting = "Ensemble"
        elif langs == ["prompt only"]:
            setting = "Few-shot prompt (no training)"
        elif m.get("extra_train"):
            setting = "Translate-train + romanised + SemEval/Kaggle/comments"
        elif "hr" in langs:
            setting = "Translate-train + romanised (En+Hi+Bn+HiRom+BnRom)"
        else:
            setting = SETTING_NAMES["en" if langs == ["en"] else "tt"]
        row = {"experiment": m["experiment"], "model": m["model"], "training data": setting}
        for lang, r in m["results"].items():
            row[f"{lang} F1"] = r["macro_f1"]
            row[f"{lang} Acc"] = r["accuracy"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["avg Indic F1"] = df[["hi F1", "bn F1"]].mean(axis=1).round(4)
    df = df.sort_values("avg Indic F1", ascending=False)
    df.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    (RESULTS_DIR / "results_table.md").write_text(
        "# Results (Macro F1 and accuracy on each test set)\n\n" + df.to_markdown(index=False), encoding="utf8")
    print(df.to_string(index=False))

    plot = df.set_index("experiment")[["en F1", "hi F1", "bn F1"]].iloc[::-1]
    ax = plot.plot(kind="barh", figsize=(7, 0.45 * len(plot) + 1.2), color=["#8a8f98", "#c0532b", "#2a7f62"])
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 1)
    ax.set_title("Macro F1 on English, Hindi and Bengali test sets")
    ax.legend(["English", "Hindi", "Bengali"], loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
