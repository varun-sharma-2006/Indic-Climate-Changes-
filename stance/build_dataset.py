"""Merge any set of unified CSVs into a leak-free train / valid split.

Examples
    # English-only (cross-lingual zero-shot setting)
    python -m stance.build_dataset --train "data/external/processed/*.csv" --valid-from "gwsd_test|semeval2016_test" --holdout-whole --name en_only

    # Everything: official + external + translations + augmentation
    python -m stance.build_dataset --train data/processed/official_train.csv data/external/processed/*.csv data/translated/*.csv --name full

Validation: if --valid is given it is used as is. Otherwise a stratified (label x lang) split is
taken from the rows whose source matches --valid-from (default: official data if present).
Exact duplicates (after normalisation) are removed, and any train row identical to a valid row is dropped.
"""
import argparse
import glob

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PROCESSED, SEED
from .data_io import load_unified, save_unified
from .text import normalize


def expand(paths):
    out = []
    for p in paths:
        out.extend(sorted(glob.glob(p)) or [p])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", nargs="*", default=None)
    ap.add_argument("--valid-from", default="official", help="substring of `source` to carve validation from")
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--holdout-whole", action="store_true", help="use ALL rows matching --valid-from as validation (e.g. official held-out test splits)")
    ap.add_argument("--exclude-sources", nargs="*", default=[], help="drop rows whose source contains any of these")
    ap.add_argument("--name", default="default")
    a = ap.parse_args()

    df = load_unified(expand(a.train))
    df = df[df["label"].notna()]
    for s in a.exclude_sources:
        df = df[~df["source"].str.contains(s)]
    df["_norm"] = df["text"].map(lambda t: normalize(t, lowercase_latin=True))
    before = len(df)
    df = df[df["_norm"].str.len() > 0].drop_duplicates("_norm")
    print(f"[build] {before} rows -> {len(df)} after removing empty/duplicate texts")

    if a.valid:
        va = load_unified(expand(a.valid))
        va = va[va["label"].notna()]
        tr = df
    else:
        pool = df[df["source"].str.contains(a.valid_from)]
        if len(pool) < 50:
            print(f"[build] <50 rows with source~'{a.valid_from}', carving validation from all data instead")
            pool = df
        if a.holdout_whole:
            va = pool
        else:
            strat = pool["label"] + "|" + pool["lang"]
            strat = strat.where(strat.map(strat.value_counts()) >= 2, pool["label"])
            _, va = train_test_split(pool, test_size=a.valid_frac, random_state=SEED, stratify=strat)
        tr = df.drop(index=va.index)
    va_norm = set(va["text"].map(lambda t: normalize(t, lowercase_latin=True)))
    leak = tr["_norm"].isin(va_norm)
    if leak.any():
        print(f"[build] dropping {leak.sum()} train rows that also appear in valid")
    tr = tr[~leak].drop(columns="_norm")
    va = va.drop(columns="_norm", errors="ignore")

    save_unified(tr.sample(frac=1, random_state=SEED), PROCESSED / a.name / "train.csv")
    save_unified(va, PROCESSED / a.name / "valid.csv")
    print("\ntrain label x lang:\n", pd.crosstab(tr["label"], tr["lang"], margins=True).to_string())
    print("\nvalid label x lang:\n", pd.crosstab(va["label"], va["lang"], margins=True).to_string())


if __name__ == "__main__":
    main()
