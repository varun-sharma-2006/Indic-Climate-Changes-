"""Mean and standard deviation of Macro F1 over runs with different random seeds.

Usage:  python src/seed_summary.py muril_tt muril_tt_seed13 muril_tt_seed7
Output: printed table and results/seed_summary.md
"""
import argparse
import json
import re
import statistics

import pandas as pd

from config import METRICS_DIR, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="experiment tags in results/metrics/")
    args = ap.parse_args()

    rows = []
    for tag in args.tags:
        m = json.loads((METRICS_DIR / f"{tag}.json").read_text(encoding="utf8"))
        seed = re.search(r"seed(\d+)", tag)  # tags like muril_tt_seed13; train_transformer.py's default is 42
        rows.append({"run": tag, "seed": m.get("seed") or (seed.group(1) if seed else "42"),
                     **{lang: m["results"][lang]["macro_f1"] for lang in ("en", "hi", "bn")}})
    df = pd.DataFrame(rows)
    summary = {"run": "mean ± std", "seed": ""}
    for lang in ("en", "hi", "bn"):
        vals = df[lang].tolist()
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[lang] = f"{statistics.mean(vals):.3f} ± {sd:.3f}"
    df = pd.concat([df.astype({l: str for l in ("en", "hi", "bn")}), pd.DataFrame([summary])], ignore_index=True)
    table = df.rename(columns={"en": "F1 En", "hi": "F1 Hi", "bn": "F1 Bn"}).to_markdown(index=False)
    (RESULTS_DIR / "seed_summary.md").write_text("# Macro F1 over random seeds\n\n" + table + "\n", encoding="utf8")
    print(table)


if __name__ == "__main__":
    main()
