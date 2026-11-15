"""Model-agnostic explanations by word occlusion (leave-one-word-out).

importance(w) = p(pred | text) - p(pred | text without w). Positive = the word supports the prediction.
Works for any predictor (transformer or TF-IDF). Output is an HTML file with words highlighted
(red = supports the predicted label, blue = opposes it).

Usage:
    python -m stance.explain --model runs/xlmr_full/best --input data/challenge/challenge_hi_bn.csv --n 30
"""
import argparse
import html
from pathlib import Path

import numpy as np

from .config import LABELS
from .data_io import load_unified
from .predictors import load_predictor


def explain_one(pred, text, lang):
    words = text.split()
    base = pred.predict_proba([text], [lang])[0]
    k = int(base.argmax())
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    if not variants:
        return LABELS[k], base[k], []
    p = pred.predict_proba(variants, [lang] * len(variants))[:, k]
    return LABELS[k], base[k], list(zip(words, base[k] - p))


def to_html(rows):
    parts = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:900px;margin:auto}"
             "span{padding:1px 2px;border-radius:3px;line-height:2}.c{border-bottom:1px solid #ccc;padding:8px 0}</style></head><body>"
             "<h2>Occlusion explanations</h2><p>red = word pushes toward the predicted label; blue = against it.</p>"]
    for r in rows:
        mx = max([abs(s) for _, s in r["imp"]] + [1e-6])
        spans = "".join(
            f"<span style='background:rgba({'220,50,47' if s > 0 else '38,139,210'},{abs(s) / mx:.2f})' title='{s:+.3f}'>{html.escape(w)}</span> "
            for w, s in r["imp"])
        parts.append(f"<div class='c'><b>gold:</b> {r['gold']} &nbsp; <b>pred:</b> {r['pred']} ({r['conf']:.2f})"
                     f"{' ✅' if r['gold'] == r['pred'] else ' ❌'}<br>{spans}</div>")
    return "\n".join(parts + ["</body></html>"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    pred = load_predictor(a.model)
    df = load_unified(a.input).head(a.n)
    rows = []
    for _, r in df.iterrows():
        lab, conf, imp = explain_one(pred, r["text"], r["lang"])
        rows.append({"gold": r["label"], "pred": lab, "conf": conf, "imp": imp})
        top = sorted(imp, key=lambda x: -x[1])[:3]
        print(f"[{r['label']} -> {lab} {conf:.2f}] top words: {[(w, round(float(s), 3)) for w, s in top]}")
    m = Path(a.model)
    out = (m.parent if m.name == "best" else m) / f"explain_{Path(a.input).stem}.html"
    out.write_text(to_html(rows), encoding="utf-8")
    print(f"[explain] -> {out}")


if __name__ == "__main__":
    main()
