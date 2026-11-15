"""Translate-train: machine-translate English stance data into Hindi and Bengali with NLLB-200.

Rationale: the test set is Hindi/Bengali, labelled data is English. XTREME (Hu et al., 2020)
shows translate-train usually beats pure zero-shot cross-lingual transfer. Labels are kept
(stance is assumed to survive translation; spot-check this, see docs/05_methodology.md).

Usage (GPU strongly recommended; ~1.3 GB VRAM in fp16):
    python -m stance.translate --input data/external/processed/gwsd.csv --langs hi bn
    python -m stance.translate --input data/external/processed/kaggle_twitter_climate.csv --max-rows 6000 --balanced
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import SEED, TRANSLATED
from .data_io import load_unified, save_unified
from .text import normalize

NLLB_CODES = {"hi": "hin_Deva", "bn": "ben_Beng", "en": "eng_Latn"}


def balanced_sample(df, n, seed=SEED):
    per = max(1, n // df["label"].nunique())
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per), random_state=seed))
              .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--langs", nargs="+", default=["hi", "bn"])
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--balanced", action="store_true", help="class-balanced subsample when --max-rows is set")
    ap.add_argument("--num-beams", type=int, default=2)
    a = ap.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    df = load_unified(a.input)
    df = df[df["label"].notna()].reset_index(drop=True)
    if a.max_rows and len(df) > a.max_rows:
        df = balanced_sample(df, a.max_rows) if a.balanced else df.sample(a.max_rows, random_state=SEED)
    df["text"] = df["text"].map(normalize)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[translate] {len(df)} rows -> {a.langs} with {a.model} on {dev}")
    tok = AutoTokenizer.from_pretrained(a.model, src_lang=NLLB_CODES["en"])
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)
    if dev == "cuda":
        model = model.half()
    model.to(dev).eval()

    stem = Path(a.input).stem
    for lang in a.langs:
        out_path = TRANSLATED / f"{stem}_{lang}.csv"
        texts, outs = df["text"].tolist(), []
        bos = tok.convert_tokens_to_ids(NLLB_CODES[lang])
        for i in tqdm(range(0, len(texts), a.batch_size), desc=f"en->{lang}", unit="batch"):
            batch = texts[i:i + a.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=160).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=192, num_beams=a.num_beams)
            outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        t = df.copy()
        t["text"] = outs
        t["lang"] = lang
        t["id"] = t["id"].astype(str) + f"_mt{lang}"
        t["source"] = t["source"].astype(str) + f"_mt_{lang}"
        save_unified(t, out_path)
        print(pd.DataFrame({"en": texts[:3], lang: outs[:3]}).to_string())


if __name__ == "__main__":
    main()
