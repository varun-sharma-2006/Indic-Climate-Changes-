"""Step 1: download GWSD and ClimateStance and merge them into one English file.

Output: data/processed/english.csv with columns
    id, source, split, text, label

Label mapping to our three classes:
    GWSD          agree -> Favour, disagree -> Against, neutral -> None
                  (label = class with the highest annotator probability)
    ClimateStance 1 -> Favour, 2 -> Against, 0 -> None (ambiguous)

Splits:
    GWSD          authors' 200 held-out sentences = test,
                  10% of the rest (stratified) = val, remainder = train
    ClimateStance the official train / val / test files
"""
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_DIR, PROCESSED_DIR, SEED

GWSD_URL = "https://raw.githubusercontent.com/yiweiluo/GWStance/master/GWSD.tsv"
CS_URL = "https://huggingface.co/datasets/eci-io/climate-evaluation/resolve/main/ClimateStance/{split}.csv"


def download(url, path):
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_gwsd():
    df = pd.read_csv(download(GWSD_URL, RAW_DIR / "GWSD.tsv"), sep="\t")
    df = df.dropna(subset=["agree", "disagree", "neutral"])  # first rows are annotator screening sentences
    probs = df[["agree", "disagree", "neutral"]].astype(float)
    df["label"] = probs.idxmax(axis=1).map({"agree": "Favour", "disagree": "Against", "neutral": "None"})
    df["split"] = df["in_held_out_test"].map({True: "test", False: "train", "True": "test", "False": "train"})

    rest = df[df["split"] == "train"]
    _, val_idx = train_test_split(rest.index, test_size=0.1, stratify=rest["label"], random_state=SEED)
    df.loc[val_idx, "split"] = "val"

    return pd.DataFrame({
        "id": "gwsd_" + df["guid"].astype(str),
        "source": "GWSD",
        "split": df["split"],
        "text": df["sentence"],
        "label": df["label"],
    })


def load_climatestance():
    frames = []
    for split, fname in [("train", "train"), ("val", "val"), ("test", "test")]:
        part = pd.read_csv(download(CS_URL.format(split=fname), RAW_DIR / f"ClimateStance_{fname}.csv"))
        part["split"] = split
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "id": [f"cs_{i}" for i in range(len(df))],
        "source": "ClimateStance",
        "split": df["split"],
        "text": df["text"],
        "label": df["label"].astype(int).map({1: "Favour", 2: "Against", 0: "None"}),
    })


def main():
    df = pd.concat([load_gwsd(), load_climatestance()], ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    out = PROCESSED_DIR / "english.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print(pd.crosstab([df["source"], df["split"]], df["label"], margins=True))


if __name__ == "__main__":
    main()
