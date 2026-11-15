"""Loading any tabular file into the unified schema [id, text, label, lang, source].

IMPORTANT: pandas treats the string "None" as missing by default, which silently deletes
the whole None class. Every reader here uses keep_default_na=False.
"""
from pathlib import Path

import pandas as pd

from .config import COLUMNS, LABELS
from .text import detect_lang

TEXT_CANDIDATES = ["text", "comment", "comments", "sentence", "tweet", "message", "content", "post", "body"]
LABEL_CANDIDATES = ["label", "stance", "class", "target", "category", "gold", "annotation"]
LANG_CANDIDATES = ["lang", "language", "langid"]
ID_CANDIDATES = ["id", "comment_id", "tweetid", "tweet_id", "guid", "uid", "index"]

LABEL_ALIASES = {
    "Favour": ["favour", "favor", "favoured", "support", "supports", "pro", "agree", "agrees", "for", "fav", "f"],
    "Against": ["against", "oppose", "opposes", "anti", "disagree", "disagrees", "deny", "denier", "aga", "a"],
    "None": ["none", "neutral", "unrelated", "no stance", "nostance", "other", "irrelevant", "news", "n", "neither"],
}
_ALIAS = {a: lab for lab, al in LABEL_ALIASES.items() for a in al}


def normalize_label(x, numeric_map: dict | None = None):
    """Map a raw label to Favour / Against / None, or None (python) if unknown/missing."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "":
        return None
    if numeric_map is not None and s in numeric_map:
        return numeric_map[s]
    return _ALIAS.get(s.lower())


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    kw = dict(keep_default_na=False, na_values=[""])
    if suf in {".csv"}:
        return pd.read_csv(path, **kw)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", **kw)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kw)
    if suf == ".json":
        return pd.read_json(path)
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _find(cols, candidates):
    low = {c.lower().strip(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def to_unified(df: pd.DataFrame, source: str, text_col=None, label_col=None, lang_col=None, id_col=None,
               numeric_map: dict | None = None, drop_unlabeled=False) -> pd.DataFrame:
    text_col = text_col or _find(df.columns, TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(f"Could not find a text column in {list(df.columns)}; pass --text-col")
    label_col = label_col or _find(df.columns, LABEL_CANDIDATES)
    lang_col = lang_col or _find(df.columns, LANG_CANDIDATES)
    id_col = id_col or _find(df.columns, ID_CANDIDATES)

    out = pd.DataFrame()
    out["id"] = df[id_col].astype(str) if id_col else [f"{source}_{i}" for i in range(len(df))]
    out["text"] = df[text_col].astype(str).fillna("")
    if label_col:
        raw = df[label_col]
        out["label"] = [normalize_label(x, numeric_map) for x in raw]
        unknown = sorted({str(r) for r, l in zip(raw, out["label"]) if l is None and str(r).strip() != ""})
        if unknown:
            print(f"[warn] {source}: {len(unknown)} unrecognised label values (kept as empty): {unknown[:10]}"
                  f"  -> use --label-map to map them")
    else:
        out["label"] = None
    if lang_col:
        out["lang"] = df[lang_col].astype(str).str.lower().str.strip().replace(
            {"hindi": "hi", "bengali": "bn", "bangla": "bn", "english": "en"})
    else:
        out["lang"] = [detect_lang(t) for t in out["text"]]
    out["source"] = source
    if drop_unlabeled:
        out = out[out["label"].notna()]
    return out[COLUMNS].reset_index(drop=True)


def load_unified(paths) -> pd.DataFrame:
    """Read one or more CSVs that are already in the unified schema."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in paths:
        d = pd.read_csv(p, keep_default_na=False, na_values=[""], dtype={"id": str})
        missing = set(COLUMNS) - set(d.columns)
        if missing:
            raise ValueError(f"{p} is not in unified schema, missing {missing}. Run `python -m stance.prepare_official` first.")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].where(df["label"].isin(LABELS), None)
    return df


def save_unified(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}  ({len(df)} rows)")
