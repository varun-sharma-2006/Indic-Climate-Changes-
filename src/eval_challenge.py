"""Evaluate saved models on the hand-written Hindi/Bengali challenge set
(data/challenge/challenge_hi_bn.csv, 64 comments written in YouTube style,
tagged by phenomenon: explicit, implicit, sarcasm, romanised, ...).

Unlike our test sets, these are not machine translations, so they show how the
models cope with more natural comment-style text.

Usage:  python src/eval_challenge.py --model_dirs models/xlmr_tt models/muril_tt
        (with several models, their average = ensemble is also reported)
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse
import json
from pathlib import Path

import pandas as pd

from config import FIG_DIR, LABEL2ID, LABELS, RESULTS_DIR, ROOT
from data_utils import compute_metrics, plot_confusion
from predict import StancePredictor

CHALLENGE = ROOT / "data" / "challenge" / "challenge_hi_bn.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    args = ap.parse_args()

    df = pd.read_csv(CHALLENGE, keep_default_na=False)
    df["y"] = df["label"].map(LABEL2ID)
    probs = {}
    for d in args.model_dirs:
        probs[Path(d).name] = StancePredictor(d).predict_proba(df["text"].tolist())
    if len(probs) > 1:
        probs["ensemble"] = sum(probs.values()) / len(probs)

    lines = [f"# Challenge set results ({len(df)} hand-written comments)\n",
             f"Labels: {df['label'].value_counts().to_dict()}  |  languages: {df['lang'].value_counts().to_dict()}\n"]
    summary, all_metrics = [], {}
    for name, p in probs.items():
        df[f"pred_{name}"] = p.argmax(1)
        m = compute_metrics(df["y"], df[f"pred_{name}"])
        all_metrics[name] = m
        row = {"model": name, "macro F1": m["macro_f1"], "accuracy": m["accuracy"]}
        for lang in ("hi", "bn"):
            g = df[df["lang"].str.startswith(lang)]  # includes romanised "hi-rom" / "bn-rom"
            row[f"F1 {lang}"] = compute_metrics(g["y"], g[f"pred_{name}"])["macro_f1"]
        summary.append(row)
        plot_confusion(m["confusion_matrix"], f"{name} (challenge set)", FIG_DIR / f"cm_challenge_{name}.png")
    lines += ["## Overall\n", pd.DataFrame(summary).to_markdown(index=False), ""]

    best = max(all_metrics, key=lambda k: all_metrics[k]["macro_f1"])
    by_ph = df.groupby("phenomenon").apply(
        lambda g: pd.Series({"n": len(g), "accuracy": round((g[f"pred_{best}"] == g["y"]).mean(), 2)}),
        include_groups=False)
    lines += [f"## Accuracy of `{best}` by phenomenon\n", by_ph.to_markdown(), ""]

    wrong = df[df[f"pred_{best}"] != df["y"]]
    lines += [f"## Errors of `{best}`\n"] + [
        f"- [{r.lang}, {r.phenomenon}] true **{r.label}**, predicted **{LABELS[getattr(r, f'pred_{best}')]}**: {r.text}"
        for r in wrong.itertuples()]

    (RESULTS_DIR / "challenge_set_results.md").write_text("\n".join(lines), encoding="utf8")
    with open(RESULTS_DIR / "challenge_set_metrics.json", "w", encoding="utf8") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
