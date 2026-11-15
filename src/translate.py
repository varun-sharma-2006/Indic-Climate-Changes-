"""Step 2: translate the English data into Hindi and Bengali (translate-train).

We use NLLB-200 (distilled 600M), an open translation model that supports
Hindi (hin_Deva) and Bengali (ben_Beng) and runs with plain `transformers`.
Text is cleaned first, so URLs and @mentions are not translated.

Output: data/processed/hindi.csv and data/processed/bengali.csv
        (same id / split / label as english.csv, translated text)

Some outputs get stuck repeating one word ("ओह ओह ओह ..."). `--fix` finds those
rows (translation more than 2.5x longer than the source, or a word repeated 4+
times in a row) and re-translates only them with repetition blocking.

Usage:  python src/translate.py --langs hi bn --batch_size 16
        python src/translate.py --langs hi bn --fix
"""
import re
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import PROCESSED_DIR
from preprocess import clean_text

MT_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {"hi": "hin_Deva", "bn": "ben_Beng"}
OUT_NAMES = {"hi": "hindi.csv", "bn": "bengali.csv"}


def is_degenerate(src: str, tr: str) -> bool:
    too_long = len(tr.split()) > 2.5 * max(len(src.split()), 3)
    return too_long or bool(re.search(r"(?:^|\s)(\S+)(?:\s+\1){3,}(?=\s|$)", tr))


def translate_all(texts, tok, model, tgt_code, batch_size, device, **gen_kwargs):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch = faster
    out = [None] * len(texts)
    bos = tok.convert_tokens_to_ids(tgt_code)
    for start in tqdm(range(0, len(order), batch_size), desc=tgt_code):
        idx = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=256,
                                 **(gen_kwargs or {"num_beams": 2}))
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hi", "bn"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fix", action="store_true", help="only re-translate degenerate rows of existing files")
    ap.add_argument("--limit", type=int, default=None, help="translate only the first N rows (quick test)")
    args = ap.parse_args()

    df = pd.read_csv(PROCESSED_DIR / "english.csv", keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    src = [clean_text(t).replace("@user", "").strip() or "." for t in df["text"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MT_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

    for lang in args.langs:
        out = PROCESSED_DIR / OUT_NAMES[lang]
        if args.fix:
            out_df = pd.read_csv(out, keep_default_na=False)
            bad = [i for i in range(len(out_df)) if is_degenerate(src[i], out_df["text"][i])]
            fixed = translate_all([src[i] for i in bad], tok, model, NLLB_CODES[lang], args.batch_size, device,
                                  num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.2)
            out_df.loc[bad, "text"] = fixed
            still_bad = sum(is_degenerate(src[i], out_df["text"][i]) for i in bad)
            out_df.to_csv(out, index=False)
            print(f"{lang}: re-translated {len(bad)} rows, {still_bad} still look degenerate -> {out}")
            continue
        out_df = df[["id", "source", "split", "label"]].copy()
        out_df["text"] = translate_all(src, tok, model, NLLB_CODES[lang], args.batch_size, device)
        out_df.to_csv(out, index=False)
        print(f"Saved {len(out_df)} rows -> {out}")
        print(out_df[["label", "text"]].head(3).to_string())


if __name__ == "__main__":
    main()
