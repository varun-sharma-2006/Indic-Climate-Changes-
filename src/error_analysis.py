"""Step 7: error analysis of one experiment's test predictions.

Breaks the results down by language and by source dataset (news vs tweets),
lists the most common confusions and saves example errors for manual review.

Usage:  python src/error_analysis.py --tag ensemble_tt
Output: results/error_analysis_<tag>.md
"""
import argparse
from collections import Counter

import pandas as pd

from config import LABELS, LANGS, PREDS_DIR, RESULTS_DIR
from data_utils import compute_metrics, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    cols = [f"p_{l}" for l in LABELS]
    lines = [f"# Error analysis: {args.tag}\n"]
    rows, confusions = [], Counter()
    examples = []
    for lang in LANGS:
        test = load_split([lang], "test")
        preds = pd.read_csv(PREDS_DIR / f"{args.tag}_{lang}.csv")
        df = test.merge(preds[["id"] + cols], on="id")
        df["pred"] = df[cols].values.argmax(1)
        df["conf"] = df[cols].values.max(1)
        for src, g in df.groupby("source"):
            m = compute_metrics(g["y"], g["pred"])
            rows.append({"language": LANGS[lang], "source": src, "n": len(g),
                         "macro F1": m["macro_f1"], "accuracy": m["accuracy"],
                         **{f"F1 {k}": v for k, v in m["per_class_f1"].items()}})
        wrong = df[df["pred"] != df["y"]]
        confusions.update(f"{LABELS[t]} -> {LABELS[p]}" for t, p in zip(wrong["y"], wrong["pred"]))
        if lang in ("hi", "bn"):
            # confident mistakes are the most informative
            for r in wrong.sort_values("conf", ascending=False).head(args.examples).itertuples():
                examples.append(f"- **{LANGS[lang]}**, {r.source}: true **{LABELS[r.y]}**, predicted "
                                f"**{LABELS[r.pred]}** ({r.conf:.2f})  \n  {r.text[:220]}")

    lines += ["## Macro F1 by language and source dataset\n", pd.DataFrame(rows).to_markdown(index=False), ""]
    total = sum(confusions.values())
    conf_df = pd.DataFrame([(k, v, f"{100 * v / total:.0f}%") for k, v in confusions.most_common()],
                           columns=["true -> predicted", "errors", "share"])
    lines += ["## Most common confusions (all languages)\n", conf_df.to_markdown(index=False), ""]
    lines += ["## Confident mistakes in Hindi and Bengali\n"] + examples

    out = RESULTS_DIR / f"error_analysis_{args.tag}.md"
    out.write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines[:6]))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
