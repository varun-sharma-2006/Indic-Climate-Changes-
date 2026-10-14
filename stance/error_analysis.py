"""Error analysis on any prediction file produced in this repo (needs gold `label` + `pred`).

Breaks performance down by language, detected script, length bin, negation, climate-term presence,
denial vocabulary, code-mixing, emoji, and (if present) the `phenomenon` column of the challenge set.
Writes the misclassified rows sorted by model confidence (most confident errors first: these are the
most informative - often label noise or systematic shortcut learning).

Usage:
    python -m stance.error_analysis --pred runs/xlmr_en_only/pred_challenge_hi_bn.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import LABELS
from .metrics import by_group, compute, confusion, pretty
from .text import surface_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    p = Path(a.pred)
    df = pd.read_csv(p, keep_default_na=False, na_values=[""])
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    f = pd.DataFrame([surface_features(t) for t in df["text"]])
    df = pd.concat([df, f], axis=1)
    df["len_bin"] = pd.cut(df["n_tokens"], [0, 5, 10, 20, 40, 80, 10_000], labels=["1-5", "6-10", "11-20", "21-40", "41-80", "80+"])
    df["code_mixed"] = df["cmi"] > 0
    df["correct"] = df["label"] == df["pred"]
    df["confidence"] = df[[f"prob_{l}" for l in LABELS]].max(1)

    out = p.parent / f"error_analysis_{p.stem}"
    out.mkdir(exist_ok=True)
    R = [f"# Error analysis: {p}\n", "```", pretty(compute(df["label"], df["pred"])), "```\n",
         "## Confusion matrix\n```", confusion(df["label"], df["pred"]).to_string(), "```\n"]
    groups = ["lang", "detected_lang", "len_bin", "has_negation", "has_climate_term", "has_denial_term", "code_mixed"]
    if "phenomenon" in df.columns:
        groups.insert(0, "phenomenon")
    for g in groups:
        t = by_group(df.assign(**{g: df[g].astype(str)}), g)
        if len(t):
            R += [f"## By `{g}`\n```", t.to_string(index=False), "```\n"]

    # calibration-ish: accuracy per confidence bin
    df["conf_bin"] = pd.cut(df["confidence"], [0, .4, .5, .6, .7, .8, .9, 1.0])
    cal = df.groupby("conf_bin", observed=True)["correct"].agg(["mean", "count"]).round(3)
    R += ["## Accuracy by confidence bin (reliability)\n```", cal.to_string(), "```\n"]
    fig, ax = plt.subplots(figsize=(5, 4))
    mids = [iv.mid for iv in cal.index]
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.plot(mids, cal["mean"], "o-")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability")
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=120); plt.close(fig)

    err = df[~df["correct"]].sort_values("confidence", ascending=False)
    cols = [c for c in ["id", "lang", "phenomenon", "label", "pred", "confidence", "has_negation", "text"] if c in err.columns]
    err[cols].to_csv(out / "errors.csv", index=False, encoding="utf-8")
    R += ["## Most confident errors (top 25)\n", "| gold | pred | conf | text |", "|---|---|---|---|"]
    for _, r in err.head(25).iterrows():
        R.append(f"| {r['label']} | {r['pred']} | {r['confidence']:.2f} | {str(r['text'])[:140].replace('|', '/')} |")
    R += ["\n## Error type counts\n```", err.groupby(["label", "pred"]).size().to_string(), "```"]
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R[:60]))
    print(f"\n[error_analysis] -> {out}")


if __name__ == "__main__":
    main()
