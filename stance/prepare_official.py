"""Convert the officially provided file(s) (sent by email) into the unified schema.

Usage:
    python -m stance.prepare_official --file data/official/train.xlsx --split train
    python -m stance.prepare_official --file data/official/test.csv --split test
    # if columns are not auto-detected:
    python -m stance.prepare_official --file x.csv --split train --text-col Comment --label-col Stance
    # if labels are numeric:
    python -m stance.prepare_official --file x.csv --split train --label-map 0=None 1=Favour 2=Against
"""
import argparse

from .config import PROCESSED
from .data_io import read_table, save_unified, to_unified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    ap.add_argument("--lang-col")
    ap.add_argument("--id-col")
    ap.add_argument("--label-map", nargs="*", default=None, help="raw=Label pairs, e.g. 0=None 1=Favour")
    a = ap.parse_args()

    df = read_table(a.file)
    print(f"[info] read {a.file}: {df.shape}, columns = {list(df.columns)}")
    nmap = dict(kv.split("=", 1) for kv in a.label_map) if a.label_map else None
    u = to_unified(df, source=f"official_{a.split}", text_col=a.text_col, label_col=a.label_col,
                   lang_col=a.lang_col, id_col=a.id_col, numeric_map=nmap)
    print(u.head().to_string())
    print("\nlabel counts:\n", u["label"].value_counts(dropna=False).to_string())
    print("\nlang counts:\n", u["lang"].value_counts().to_string())
    save_unified(u, PROCESSED / f"official_{a.split}.csv")


if __name__ == "__main__":
    main()
