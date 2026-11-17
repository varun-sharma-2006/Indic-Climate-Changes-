"""Step 2c: extra training data for robustness on short, informal comments.

Sources (training only; our test sets are unchanged):
  1. SemEval-2016 Task 6, target "Climate Change is a Real Concern" (human-labelled tweets),
     via TweetEval: https://huggingface.co/datasets/cardiffnlp/tweet_eval (stance_climate).
     train + validation -> training; its test split (169 tweets) -> extra English test set.
  2. Kaggle "Twitter Climate Change Sentiment Dataset" (Edward Qian, 43,943 human-labelled tweets)
     https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset
     Needs a manual download: put twitter_sentiment_data.csv in data/external/raw/.
     pro (1) -> Favour, anti (-1) -> Against, neutral (0) and news (2) -> None.
     A class-balanced sample is used; 10% of it is held out as an extra test set.
  3. data/synthetic/youtube_style_comments.csv: 184 short YouTube-style comments in Hindi,
     Bengali, romanised Hindi/Bengali and English, written with an AI assistant (LLM-based
     data augmentation) and checked by us. Sentences similar to any test / challenge /
     sanity-check sentence were removed.

English sentences are translated to Hindi/Bengali with NLLB-200 (translate.py) and romanised
(romanize_data.py), like the main data. Translations are cached per source.

Output: data/extra/extra_train.csv, data/extra/extra_test.csv
        (columns: id, source, lang, split, text, label)
Usage:  python src/extra_data.py [--kaggle_per_class 2000]
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse
import re
import sys
import urllib.request

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DATA_DIR, ROOT, SEED
from preprocess import clean_text
from translate import MT_MODEL, NLLB_CODES, is_degenerate, translate_all

sys.path.insert(0, str(ROOT))
from stance.augment import romanize  # noqa: E402

EXTRA = DATA_DIR / "extra"
RAW = DATA_DIR / "external" / "raw"
SEMEVAL_URL = "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/main/stance_climate/{split}-00000-of-00001.parquet"
KAGGLE_FILE = RAW / "twitter_sentiment_data.csv"
SYNTHETIC = DATA_DIR / "synthetic" / "youtube_style_comments.csv"
LANG_CODES = {"hi": "hi", "bn": "bn", "hi-rom": "hr", "bn-rom": "br", "en": "en"}


def semeval():
    rows = []
    for split, ours in [("train", "train"), ("validation", "train"), ("test", "test")]:
        path = RAW / f"semeval2016_climate_{split}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(SEMEVAL_URL.format(split=split), path)
        d = pd.read_parquet(path)
        d["text"] = d["text"].str.replace("#SemST", "", regex=False).str.strip()
        d["label"] = d["label"].map({0: "None", 1: "Against", 2: "Favour"})
        d["split"] = ours
        rows.append(d[["text", "label", "split"]])
    d = pd.concat(rows, ignore_index=True)
    d["id"] = [f"semeval_{i}" for i in range(len(d))]
    d["source"] = "SemEval2016"
    return d


def kaggle(per_class):
    if not KAGGLE_FILE.exists():
        print(f"(Kaggle file not found at {KAGGLE_FILE}; skipping it)")
        return None
    d = pd.read_csv(KAGGLE_FILE)
    d["text"] = d["message"].astype(str).map(lambda t: re.sub(r"^RT @\w+:\s*", "", t))
    d["label"] = d["sentiment"].map({1: "Favour", -1: "Against", 0: "None", 2: "None"})
    d = d.dropna(subset=["label"]).drop_duplicates("text")
    d = d[d["text"].map(lambda t: len(clean_text(t).split()) >= 4)]
    d = d.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), per_class), random_state=SEED))
    d = d.sample(frac=1, random_state=SEED).reset_index(drop=True)
    d["split"] = ["test" if i % 10 == 0 else "train" for i in range(len(d))]
    d["id"] = "kaggle_" + d["tweetid"].astype(str)
    d["source"] = "KaggleClimateTweets"
    return d[["id", "source", "split", "text", "label"]]


def multilingual(df, name, tok, model, device):
    """English rows + Hindi/Bengali translations + romanised copies (cached in data/extra/)."""
    parts = [df.assign(lang="en")]
    src = [clean_text(t).replace("@user", "").strip() or "." for t in df["text"]]
    for lang in ("hi", "bn"):
        cache = EXTRA / f"{name}_{lang}.csv"
        if cache.exists() and len(pd.read_csv(cache, keep_default_na=False)) == len(df):
            tr = pd.read_csv(cache, keep_default_na=False)
        else:
            out = translate_all(src, tok, model, NLLB_CODES[lang], 24, device)
            bad = [i for i in range(len(out)) if is_degenerate(src[i], out[i])]
            if bad:
                fixed = translate_all([src[i] for i in bad], tok, model, NLLB_CODES[lang], 24, device,
                                      num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.2)
                for i, t in zip(bad, fixed):
                    out[i] = t
            tr = df.assign(lang=lang, text=out)
            tr.to_csv(cache, index=False)
        parts.append(tr)
        parts.append(tr.assign(lang={"hi": "hr", "bn": "br"}[lang], text=tr["text"].map(romanize)))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_per_class", type=int, default=2000)
    args = ap.parse_args()
    EXTRA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MT_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

    frames = [multilingual(semeval(), "semeval", tok, model, device)]
    k = kaggle(args.kaggle_per_class)
    if k is not None:
        frames.append(multilingual(k, f"kaggle{args.kaggle_per_class}", tok, model, device))

    syn = pd.read_csv(SYNTHETIC, keep_default_na=False)
    syn = syn.assign(id=[f"synthetic_{i}" for i in range(len(syn))], source="SyntheticYouTubeStyle",
                     split="train", lang=syn["lang"].map(LANG_CODES))
    frames.append(syn[["id", "source", "split", "text", "label", "lang"]])

    allx = pd.concat(frames, ignore_index=True)[["id", "source", "lang", "split", "text", "label"]]
    allx[allx["split"] == "train"].to_csv(EXTRA / "extra_train.csv", index=False)
    allx[allx["split"] == "test"].to_csv(EXTRA / "extra_test.csv", index=False)
    print(pd.crosstab([allx["source"], allx["split"]], allx["label"], margins=True))


if __name__ == "__main__":
    main()
