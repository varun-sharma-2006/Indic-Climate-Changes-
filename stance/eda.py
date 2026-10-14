"""Scientific exploratory data analysis for any unified CSV(s).

Produces reports/eda/<name>/report.md + PNG figures covering: size, class balance, language and
script distribution, length statistics, vocabulary statistics (types, TTR, hapax, OOV vs other
sets), duplicates/missing, code-mixing index, surface-noise features, negation/denial-cue rates
per class, and the most class-distinctive words via weighted log-odds with an informative
Dirichlet prior (Monroe, Colaresi & Quinn, 2008).

Usage:
    python -m stance.eda --input data/processed/official_train.csv --name official_train
    python -m stance.eda --input data/external/processed/*.csv --name external
"""
import argparse
import glob
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import LABELS, REPORTS
from .data_io import load_unified
from .text import STOPWORDS, normalize, surface_features, tokenize


def log_odds(counts_a: Counter, counts_b: Counter, prior: Counter, top=15):
    """Weighted log-odds ratio with informative Dirichlet prior; returns top z-scored words for a vs b."""
    na, nb, a0 = sum(counts_a.values()), sum(counts_b.values()), sum(prior.values())
    scores = {}
    for w in set(counts_a) | set(counts_b):
        aw = prior[w]
        if aw == 0:
            continue
        ya, yb = counts_a[w], counts_b[w]
        la = math.log((ya + aw) / (na + a0 - ya - aw))
        lb = math.log((yb + aw) / (nb + a0 - yb - aw))
        var = 1 / (ya + aw) + 1 / (yb + aw)
        scores[w] = (la - lb) / math.sqrt(var)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string() + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--compare", nargs="*", default=None, help="other unified CSV(s) to compute vocabulary overlap / OOV against")
    a = ap.parse_args()

    paths = [p for x in a.input for p in (sorted(glob.glob(x)) or [x])]
    df = load_unified(paths)
    out = REPORTS / "eda" / a.name
    (out / "fig").mkdir(parents=True, exist_ok=True)
    R = [f"# EDA report: `{a.name}`\n", f"Files: {', '.join(paths)}\n"]

    # ---------------- basics
    df["text"] = df["text"].fillna("")
    n = len(df)
    empty = (df["text"].str.strip() == "").sum()
    df["_norm"] = df["text"].map(lambda t: normalize(t, lowercase_latin=True))
    dup_exact = df.duplicated("text").sum()
    dup_norm = df.duplicated("_norm").sum()
    conflict = df[df["label"].notna()].groupby("_norm")["label"].nunique()
    conflict = int((conflict > 1).sum())
    R += ["## 1. Size and quality\n",
          f"| metric | value |\n|---|---|\n| instances | {n} |\n| empty texts | {empty} |\n"
          f"| missing labels | {df['label'].isna().sum()} |\n| exact duplicates | {dup_exact} |\n"
          f"| duplicates after normalisation | {dup_norm} |\n| duplicate texts with CONFLICTING labels | {conflict} |\n"]

    # ---------------- features
    feats = pd.DataFrame([surface_features(t) for t in df["text"]])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    lab = df[df["label"].notna()]

    # ---------------- class & language distribution
    cd = lab["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    R += ["## 2. Class distribution\n", md_table(pd.DataFrame({"count": cd, "share": (cd / cd.sum()).round(3)})),
          f"\nImbalance ratio (max/min): {cd.max() / max(cd.min(), 1):.2f}\n"]
    R += ["## 3. Language distribution\n",
          "Given `lang` column vs script-based detection (`detected_lang`: hi, bn, en, hi-rom, bn-rom, mixed).\n",
          md_table(pd.crosstab(df["lang"], df["detected_lang"], margins=True)), "\n",
          "### Label x language\n", md_table(pd.crosstab(lab["label"], lab["lang"], margins=True, normalize=False)), "\n"]

    # ---------------- lengths
    def lstats(key):
        g = df.groupby(key, dropna=False)
        return pd.DataFrame({"n": g.size(), "chars_mean": g["n_chars"].mean(), "chars_median": g["n_chars"].median(),
                             "tokens_mean": g["n_tokens"].mean(), "tokens_median": g["n_tokens"].median(),
                             "tokens_p95": g["n_tokens"].quantile(0.95), "tokens_max": g["n_tokens"].max()}).round(1)

    R += ["## 4. Length statistics (by label)\n", md_table(lstats("label")), "\n",
          "### By language\n", md_table(lstats("lang")), "\n",
          f"Share of texts > 128 tokens (whitespace): {(df['n_tokens'] > 128).mean():.3f} "
          f"-> informs `--max-len` for transformers.\n"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for l in LABELS:
        s = lab.loc[lab["label"] == l, "n_tokens"]
        if len(s):
            ax.hist(s.clip(upper=s.quantile(0.99)), bins=40, alpha=0.5, label=f"{l} (n={len(s)})")
    ax.set_xlabel("tokens"); ax.set_ylabel("count"); ax.set_title("Length by class"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig" / "length_by_class.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    cd.plot.bar(ax=ax, color=["#2a9d8f", "#e76f51", "#8d99ae"]); ax.set_title("Class distribution")
    fig.tight_layout(); fig.savefig(out / "fig" / "class_distribution.png", dpi=130); plt.close(fig)

    # ---------------- vocabulary
    toks = df["_norm"].map(tokenize)
    vocab = Counter(t for ts in toks for t in ts)
    n_tok = sum(vocab.values())
    hapax = sum(1 for c in vocab.values() if c == 1)
    R += ["## 5. Vocabulary\n",
          f"| metric | value |\n|---|---|\n| tokens | {n_tok} |\n| types | {len(vocab)} |\n"
          f"| type/token ratio | {len(vocab) / max(n_tok, 1):.3f} |\n| hapax legomena | {hapax} ({hapax / max(len(vocab), 1):.1%} of types) |\n"]
    by_lang_vocab = {lg: Counter(t for ts in toks[df["lang"] == lg] for t in ts) for lg in df["lang"].unique()}
    R += ["\nTypes per language: " + ", ".join(f"{k}={len(v)}" for k, v in by_lang_vocab.items()) + "\n"]
    if a.compare:
        cmp = load_unified([p for x in a.compare for p in (sorted(glob.glob(x)) or [x])])
        cv = Counter(t for s in cmp["text"].map(lambda t: normalize(t, lowercase_latin=True)) for t in tokenize(s))
        oov_types = sum(1 for w in vocab if w not in cv)
        oov_tok = sum(c for w, c in vocab.items() if w not in cv)
        R += [f"\n**OOV vs comparison set** ({', '.join(a.compare)}): {oov_types / len(vocab):.1%} of types, "
              f"{oov_tok / n_tok:.1%} of tokens are unseen -> lexical overlap is low; motivates subword / multilingual models.\n"]

    # ---------------- noise / code-mixing
    noise_cols = ["n_emoji", "has_url", "has_mention", "has_hashtag", "has_elongation", "n_exclaim", "n_question"]
    R += ["## 6. Social-media noise and code-mixing\n",
          md_table(df.groupby("lang")[noise_cols + ["cmi"]].mean().round(3)), "\n",
          f"Texts with CMI > 0 (any mixing): {(df['cmi'] > 0).mean():.1%}; CMI > 20: {(df['cmi'] > 20).mean():.1%}\n"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["cmi"], bins=30, color="#457b9d"); ax.set_title("Code-Mixing Index"); ax.set_xlabel("CMI")
    fig.tight_layout(); fig.savefig(out / "fig" / "cmi_hist.png", dpi=130); plt.close(fig)

    # ---------------- cue rates per class
    cue = lab.groupby("label")[["has_negation", "has_climate_term", "has_denial_term", "n_emoji", "n_question", "n_exclaim"]].mean().round(3)
    R += ["## 7. Linguistic cues per class\n",
          "Rates of negation markers, explicit climate vocabulary, denial vocabulary (hoax/झूठ/মিথ্যা...), emoji, questions.\n",
          md_table(cue), "\n",
          "Interpretation hints: if `None` has a low climate-term rate, relevance detection is easy; if `Against` has a high "
          "negation rate, the model may learn negation as a shortcut (check with the challenge set).\n"]

    # ---------------- frequent and distinctive words
    R += ["## 8. Frequent words (stopwords removed)\n"]
    content = lambda ts: [t for t in ts if t not in STOPWORDS and not t.isdigit() and len(t) > 1]
    for lg in sorted(df["lang"].unique()):
        c = Counter(t for ts in toks[df["lang"] == lg] for t in content(ts))
        R += [f"- **{lg}**: " + ", ".join(f"{w} ({k})" for w, k in c.most_common(25)) + "\n"]
    R += ["\n## 9. Most class-distinctive words (log-odds, informative Dirichlet prior)\n"]
    prior = Counter(t for ts in toks for t in content(ts))
    lab_toks = toks[df["label"].notna()]
    cls_counts = {l: Counter(t for ts in lab_toks[lab["label"] == l] for t in content(ts)) for l in LABELS}
    for l in LABELS:
        rest = Counter()
        for o in LABELS:
            if o != l:
                rest.update(cls_counts[o])
        if sum(cls_counts[l].values()) == 0:
            continue
        top = log_odds(cls_counts[l], rest, prior)
        R += [f"- **{l}**: " + ", ".join(f"{w} ({z:.1f})" for w, z in top) + "\n"]

    # ---------------- examples
    R += ["\n## 10. Random examples per class\n"]
    for l in LABELS:
        s = lab[lab["label"] == l]
        for t in s["text"].sample(min(3, len(s)), random_state=0):
            R += [f"- [{l}] {t[:200]}\n"]

    R += ["\n## Figures\n", "![](fig/class_distribution.png)\n", "![](fig/length_by_class.png)\n", "![](fig/cmi_hist.png)\n"]
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")
    df.drop(columns="_norm").to_csv(out / "features.csv", index=False, encoding="utf-8")
    print("\n".join(R[:12]))
    print(f"\n[eda] full report -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
