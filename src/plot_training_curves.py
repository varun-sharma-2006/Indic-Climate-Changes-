"""Plot training loss and validation Macro F1 per epoch from training logs.

Reads lines like "epoch 3: loss 0.6917  val macro-F1 0.6394" written by train_transformer.py.

Usage:  python src/plot_training_curves.py results/logs/seed13.txt results/logs/seed7.txt
Output: results/figures/training_curves.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIG_DIR

LINE_RE = re.compile(r"epoch (\d+): loss ([\d.]+)\s+val macro-F1 ([\d.]+)")


def read_log(path):
    text = Path(path).read_text(encoding="utf8", errors="ignore").replace("\r", "\n")
    rows = [tuple(map(float, m.groups())) for m in LINE_RE.finditer(text)]
    return [int(r[0]) for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=str(FIG_DIR / "training_curves.png"))
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for log in args.logs:
        epochs, loss, f1 = read_log(log)
        if not epochs:
            print(f"no epoch lines in {log}")
            continue
        name = Path(log).stem
        ax1.plot(epochs, loss, marker="o", label=name)
        ax2.plot(epochs, f1, marker="o", label=name)
        best = max(range(len(f1)), key=f1.__getitem__)
        ax2.scatter([epochs[best]], [f1[best]], s=120, facecolors="none", edgecolors="black")
        print(f"{name}: best epoch {epochs[best]} (val macro-F1 {f1[best]:.4f})")
    ax1.set(title="Training loss", xlabel="epoch", ylabel="loss")
    ax2.set(title="Validation Macro F1 (circle = epoch kept)", xlabel="epoch", ylabel="macro F1")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
