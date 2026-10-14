"""Label-preserving perturbations that simulate real YouTube-comment noise.

Used twice:
  * as TRAINING augmentation (python -m stance.augment ...), and
  * as a ROBUSTNESS test suite (stance.robustness), CheckList-style (Ribeiro et al., 2020).

Perturbations
  romanize   Devanagari/Bengali -> Latin (simulates Hinglish/Banglish typing)
  codemix    swap Hindi/Bengali climate words for their English equivalents
  typo       random character drop/duplicate/swap (spelling variation)
  noise      add emoji / elongation / extra punctuation / URL
"""
import argparse
import random

import pandas as pd

from .config import SEED
from .data_io import load_unified, save_unified
from .text import token_script

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:  # optional dependency
    sanscript = None

CODEMIX_SWAP = {
    "जलवायु परिवर्तन": "climate change", "ग्लोबल वार्मिंग": "global warming", "प्रदूषण": "pollution",
    "पर्यावरण": "environment", "तापमान": "temperature", "गर्मी": "heat", "सरकार": "government",
    "गंभीर": "serious", "समस्या": "problem", "झूठ": "fake", "पेड़": "trees",
    "জলবায়ু পরিবর্তন": "climate change", "বিশ্ব উষ্ণায়ন": "global warming", "দূষণ": "pollution",
    "পরিবেশ": "environment", "তাপমাত্রা": "temperature", "গরম": "heat", "সরকার": "government",
    "গুরুতর": "serious", "সমস্যা": "problem", "মিথ্যা": "fake", "গাছ": "trees",
}
STRIP = dict.fromkeys(map(ord, "़়‌‍"), None)  # leftover nukta / joiners
EMOJIS = ["😂", "🙏", "😡", "🔥", "😢", "👍", "🌍", "😱", "💯", "🤣"]


def romanize(text: str) -> str:
    """Lossy ITRANS romanisation, lower-cased to look like casual Hinglish/Banglish."""
    if sanscript is None:
        raise ImportError("pip install indic-transliteration")
    out = []
    for tok in text.split(" "):
        s = token_script(tok)
        if s == "deva":
            tok = transliterate(tok, sanscript.DEVANAGARI, sanscript.ITRANS)
        elif s == "beng":
            tok = transliterate(tok, sanscript.BENGALI, sanscript.ITRANS)
        else:
            out.append(tok)
            continue
        tok = tok.translate(STRIP)
        if len(tok) > 3 and tok.endswith("a") and not tok.endswith(("aa", "A")):
            tok = tok[:-1]  # word-final schwa deletion: parivartana -> parivartan
        out.append(tok.lower().replace("~", "").replace(".", "").replace("^", ""))
    return " ".join(out)


def codemix(text: str) -> str:
    for k, v in CODEMIX_SWAP.items():
        text = text.replace(k, v)
    return text


def typo(text: str, rate=0.06, rng=random) -> str:
    chars = list(text)
    out, i = [], 0
    while i < len(chars):
        c = chars[i]
        if c.isspace() or rng.random() > rate:
            out.append(c)
        else:
            op = rng.choice(["drop", "dup", "swap"])
            if op == "dup":
                out += [c, c]
            elif op == "swap" and i + 1 < len(chars):
                out += [chars[i + 1], c]
                i += 1
            # drop: append nothing
        i += 1
    return "".join(out)


def noise(text: str, rng=random) -> str:
    ops = rng.sample(["emoji", "elong", "punct", "url"], k=2)
    if "emoji" in ops:
        text = text + " " + "".join(rng.choices(EMOJIS, k=rng.randint(1, 3)))
    if "elong" in ops:
        words = text.split()
        if words:
            j = rng.randrange(len(words))
            words[j] = words[j] + words[j][-1] * 3
            text = " ".join(words)
    if "punct" in ops:
        text = text + rng.choice(["!!!", "???", "...", " !!"])
    if "url" in ops:
        text = text + " https://youtu.be/xyz"
    return text


PERTURBATIONS = {"romanize": romanize, "codemix": codemix, "typo": typo, "noise": noise}


def main():
    ap = argparse.ArgumentParser(description="Create an augmented copy of a unified CSV")
    ap.add_argument("--input", required=True, nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ops", nargs="+", default=["romanize", "codemix", "typo", "noise"], choices=list(PERTURBATIONS))
    ap.add_argument("--frac", type=float, default=0.3, help="fraction of rows perturbed per op")
    a = ap.parse_args()
    random.seed(SEED)
    df = load_unified(a.input)
    df = df[df["label"].notna()]
    frames = []
    for op in a.ops:
        s = df.sample(frac=a.frac, random_state=SEED + len(frames)).copy()
        s["text"] = s["text"].map(PERTURBATIONS[op])
        s["id"] = s["id"].astype(str) + f"_aug{op}"
        s["source"] = s["source"].astype(str) + f"_aug_{op}"
        if op == "romanize":
            s["lang"] = s["lang"].map(lambda l: f"{l}-rom" if l in ("hi", "bn") else l)
        frames.append(s)
        print(f"[augment] {op}: {len(s)} rows, e.g. {s['text'].iloc[0][:100]!r}")
    save_unified(pd.concat(frames, ignore_index=True), a.output)


if __name__ == "__main__":
    main()
