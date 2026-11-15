"""Download and convert the external, publicly available English climate-stance datasets.

Sources (see docs/03_datasets.md for licences and label mappings):
  1. GWSD  - Global Warming Stance Dataset (Luo, Card & Jurafsky, Findings EMNLP 2020).
             Same claim as ours: "Climate change/global warming is a serious concern".
             agree -> Favour, disagree -> Against, neutral -> None. ~2.3k news sentences.
  2. SemEval-2016 Task 6, target "Climate Change is a Real Concern" (via cardiffnlp/tweet_eval,
             config stance_climate). favor/against/none. ~560 tweets.
  3. Kaggle "Twitter Climate Change Sentiment Dataset" (edqian). 43,943 tweets labelled
             1=pro (believes in man-made CC) / -1=anti / 0=neutral / 2=news.
             Needs a Kaggle account -> see instructions printed below.

Usage:
    python -m stance.download_external                 # all available
    python -m stance.download_external --kaggle-news-as None
"""
import argparse
import re
import shutil
import subprocess
import urllib.request
import zipfile

import pandas as pd

from .config import EXT_PROC, EXT_RAW
from .data_io import save_unified

GWSD_URL = "https://raw.githubusercontent.com/yiweiluo/GWStance/master/GWSD.tsv"
TWEETEVAL_URL = "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/main/stance_climate/{split}-00000-of-00001.parquet"
KAGGLE_SLUG = "edqian/twitter-climate-change-sentiment-dataset"
KAGGLE_FILE = "twitter_sentiment_data.csv"

RT_RE = re.compile(r"^RT @\w+:\s*")


def fetch(url, dest):
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def gwsd():
    p = fetch(GWSD_URL, EXT_RAW / "GWSD.tsv")
    d = pd.read_csv(p, sep="\t", keep_default_na=False, na_values=[""])
    probs = d[["disagree", "agree", "neutral"]].apply(pd.to_numeric, errors="coerce")
    workers = d[[c for c in d.columns if c.startswith("worker_")]]
    vote = workers.apply(lambda r: r[r.astype(bool)].value_counts().idxmax(), axis=1)  # screening items: majority vote
    raw = probs.idxmax(axis=1).where(probs.notna().all(axis=1), vote.str.rstrip("s"))
    m = {"agree": "Favour", "disagree": "Against", "neutral": "None"}
    out = pd.DataFrame({
        "id": "gwsd_" + d["guid"].astype(str),
        "text": d["sentence"],
        "label": raw.map(m),
        "lang": "en",
        "source": "gwsd" + d["in_held_out_test"].astype(str).str.lower().map({"true": "_test", "false": ""}),
    })
    save_unified(out.dropna(subset=["label"]), EXT_PROC / "gwsd.csv")


def semeval():
    frames = []
    for split in ["train", "validation", "test"]:
        p = fetch(TWEETEVAL_URL.format(split=split), EXT_RAW / f"semeval2016_climate_{split}.parquet")
        d = pd.read_parquet(p)
        d["split"] = split
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    m = {0: "None", 1: "Against", 2: "Favour"}  # tweet_eval stance mapping
    out = pd.DataFrame({
        "id": [f"semeval_{s}_{i}" for i, s in enumerate(d["split"])],
        "text": d["text"].str.replace("#SemST", "", regex=False).str.strip(),
        "label": d["label"].map(m),
        "lang": "en",
        "source": "semeval2016_" + d["split"],
    })
    save_unified(out, EXT_PROC / "semeval2016_climate.csv")


def kaggle(news_as: str):
    raw = EXT_RAW / KAGGLE_FILE
    if not raw.exists() and shutil.which("kaggle"):
        print("[kaggle] trying kaggle CLI ...")
        r = subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(EXT_RAW)])
        z = EXT_RAW / "twitter-climate-change-sentiment-dataset.zip"
        if r.returncode == 0 and z.exists():
            with zipfile.ZipFile(z) as f:
                f.extract(KAGGLE_FILE, EXT_RAW)
    if not raw.exists():
        print(f"""
[kaggle] NOT FOUND: {raw}
  Download manually (free Kaggle account needed):
    https://www.kaggle.com/datasets/{KAGGLE_SLUG}
  -> click Download, unzip, put '{KAGGLE_FILE}' into  {EXT_RAW}
  then re-run:  python -m stance.download_external
""")
        return
    d = pd.read_csv(raw, keep_default_na=False, na_values=[""])
    m = {1: "Favour", -1: "Against", 0: "None", 2: news_as}
    text = d["message"].astype(str).map(lambda t: RT_RE.sub("", t))
    out = pd.DataFrame({
        "id": "kaggle_" + d["tweetid"].astype(str),
        "text": text,
        "label": d["sentiment"].map(m),
        "lang": "en",
        "source": "kaggle_twitter_climate",
    })
    n = len(out)
    out = out.drop_duplicates(subset=["text"]).dropna(subset=["label"])
    print(f"[kaggle] removed {n - len(out)} duplicate/retweet rows")
    save_unified(out, EXT_PROC / "kaggle_twitter_climate.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["gwsd", "semeval", "kaggle"], default=["gwsd", "semeval", "kaggle"])
    ap.add_argument("--kaggle-news-as", default="None", choices=["None", "Favour"],
                    help="How to map Kaggle 'news' tweets (2). Default None (factual reporting, no explicit stance).")
    a = ap.parse_args()
    if "gwsd" in a.only:
        gwsd()
    if "semeval" in a.only:
        semeval()
    if "kaggle" in a.only:
        kaggle(a.kaggle_news_as)
    for f in sorted(EXT_PROC.glob("*.csv")):
        d = pd.read_csv(f, keep_default_na=False, na_values=[""])
        print(f"\n{f.name}: {len(d)} rows\n{d['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
