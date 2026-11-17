"""Compares models on data never used for training, to check robustness on comment-style text:
  * hand-written challenge set (64 comments; sarcasm, romanised, negation, ...)
  * sanity-check sentences (18, src/sanity_check.py)
  * extra test sets from src/extra_data.py (SemEval-2016 test tweets; Kaggle held-out tweets)

Usage:  python src/robustness_report.py --model_dirs models/muril_tt models/muril_robust1
Output: results/robustness_report.md
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse
from pathlib import Path

import pandas as pd

from config import DATA_DIR, LABEL2ID, LABELS, RESULTS_DIR, ROOT
from data_utils import compute_metrics
from predict import StancePredictor
from sanity_check import CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    args = ap.parse_args()

    ch = pd.read_csv(ROOT / "data" / "challenge" / "challenge_hi_bn.csv", keep_default_na=False)
    ch["y"] = ch["label"].map(LABEL2ID)
    extra_path = DATA_DIR / "extra" / "extra_test.csv"
    extra = pd.read_csv(extra_path, keep_default_na=False) if extra_path.exists() else pd.DataFrame()
    if len(extra):
        extra["y"] = extra["label"].map(LABEL2ID)

    rows, sarc_rows = [], []
    for d in args.model_dirs:
        name = Path(d).name
        pred = StancePredictor(d)
        row = {"model": name}

        p = pred.predict_proba(ch["text"].tolist()).argmax(1)
        m = compute_metrics(ch["y"], p)
        row["challenge F1 (64)"] = m["macro_f1"]
        row["challenge acc"] = m["accuracy"]
        ok = p == ch["y"].values
        for ph in ("sarcasm", "romanized", "implicit", "negation"):
            mask = (ch["phenomenon"] == ph).values
            row[f"{ph} acc"] = f"{ok[mask].sum()}/{mask.sum()}"
        rom = ch["lang"].str.endswith("rom").values
        row["romanised-lang acc"] = f"{ok[rom].sum()}/{rom.sum()}"

        sp = pred.predict_proba([t for t, _, _ in CASES]).argmax(1)
        row["sanity (18)"] = f"{sum(LABELS[i] == e for i, (_, e, _) in zip(sp, CASES))}/{len(CASES)}"

        for (src, lang), g in (extra.groupby(["source", "lang"]) if len(extra) else []):
            if lang in ("en", "hi", "bn"):
                gp = pred.predict_proba(g["text"].tolist()).argmax(1)
                row[f"{src} {lang} F1"] = compute_metrics(g["y"], gp)["macro_f1"]
        rows.append(row)

        for r, pi in zip(ch.itertuples(), p):
            if r.phenomenon == "sarcasm":
                sarc_rows.append({"model": name, "text": r.text, "true": r.label, "pred": LABELS[pi]})

    df = pd.DataFrame(rows)
    lines = ["# Robustness report (data never used for training)\n", df.set_index("model").T.to_markdown(), "",
             "## Sarcasm examples in the challenge set\n", pd.DataFrame(sarc_rows).to_markdown(index=False)]
    (RESULTS_DIR / "robustness_report.md").write_text("\n".join(lines), encoding="utf8")
    print(df.set_index("model").T.to_string())


if __name__ == "__main__":
    main()
