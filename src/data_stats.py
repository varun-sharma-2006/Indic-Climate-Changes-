"""Step 3: dataset statistics and plots.

Writes results/dataset_stats.md and figures in results/figures/.
Usage:  python src/data_stats.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DIR, LABELS, LANGS, RESULTS_DIR
from data_utils import load_lang


def main():
    df = pd.concat([load_lang(l) for l in LANGS], ignore_index=True)
    df["n_words"] = df["clean"].str.split().str.len()
    en = df[df["lang"] == "en"]

    lines = ["# Dataset statistics\n"]

    tab = pd.crosstab([en["source"], en["split"]], en["label"], margins=True, margins_name="Total")[LABELS + ["Total"]]
    lines += ["## English data by source, split and label\n", tab.to_markdown(), ""]

    split_tab = pd.crosstab(en["split"], en["label"], margins=True, margins_name="Total")[LABELS + ["Total"]]
    split_tab = split_tab.reindex(["train", "val", "test", "Total"])
    lines += ["## Size of each split (per language; Hindi and Bengali are translations of the same rows)\n",
              split_tab.to_markdown(), ""]

    dist = (en["label"].value_counts(normalize=True)[LABELS] * 100).round(1)
    lines += ["## Class distribution (%)\n", dist.to_frame("percent").to_markdown(), ""]

    length = df.groupby("lang")["n_words"].describe()[["mean", "50%", "max"]].round(1)
    length.index = [LANGS[l] for l in length.index]
    lines += ["## Text length in words (after cleaning)\n", length.to_markdown(), ""]

    by_src = en.groupby("source")["n_words"].mean().round(1)
    lines += ["## Average English length by source (words)\n", by_src.to_frame("mean words").to_markdown(), ""]

    (RESULTS_DIR / "dataset_stats.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))

    # figure 1: class distribution per source
    ax = pd.crosstab(en["source"], en["label"])[LABELS].plot(kind="bar", figsize=(5.5, 3.5),
                                                              color=["#2a7f62", "#c0532b", "#8a8f98"])
    ax.set_xlabel("")
    ax.set_ylabel("Number of examples")
    ax.set_title("Class distribution by dataset")
    plt.xticks(rotation=0)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # figure 2: length distribution per language
    plt.figure(figsize=(5.5, 3.5))
    for lang, name in LANGS.items():
        plt.hist(df[df["lang"] == lang]["n_words"].clip(upper=80), bins=40, alpha=0.5, label=name)
    plt.xlabel("Words per text (clipped at 80)")
    plt.ylabel("Count")
    plt.title("Text length by language")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "length_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
