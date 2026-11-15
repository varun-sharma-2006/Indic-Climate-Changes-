"""Text normalisation and linguistic feature extraction for noisy Hindi/Bengali/English social-media text.

Everything here is rule-based and dependency-free so it can be used for EDA, baselines,
error analysis and as a preprocessing step before transformer tokenisation.
"""
import re
import unicodedata
from collections import Counter

# --------------------------------------------------------------------------- scripts
DEVANAGARI = (0x0900, 0x097F)
BENGALI = (0x0980, 0x09FF)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
ELONG_RE = re.compile(r"(.)\1{2,}")  # "bahuuuut" -> "bahuut", "!!!!!" -> "!!"
WS_RE = re.compile(r"\s+")
ZERO_WIDTH = dict.fromkeys(map(ord, "​﻿⁠­"), None)  # keep ZWJ/ZWNJ: meaningful in Indic conjuncts
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF️]"
)
# Devanagari and Bengali digits -> ASCII
DIGIT_MAP = {**{0x0966 + i: str(i) for i in range(10)}, **{0x09E6 + i: str(i) for i in range(10)}}

TOKEN_RE = re.compile(r"[\wऀ-ॿঀ-৿‌‍]+", re.UNICODE)


def char_script(ch: str) -> str:
    cp = ord(ch)
    if DEVANAGARI[0] <= cp <= DEVANAGARI[1]:
        return "deva"
    if BENGALI[0] <= cp <= BENGALI[1]:
        return "beng"
    if ch.isascii() and ch.isalpha():
        return "latn"
    return "other"


def token_script(tok: str) -> str:
    c = Counter(char_script(ch) for ch in tok)
    c.pop("other", None)
    return c.most_common(1)[0][0] if c else "other"


def normalize(text: str, *, mask_urls=True, mask_users=True, split_hashtags=True,
              reduce_elongation=True, lowercase_latin=False) -> str:
    """Conservative normalisation that keeps stance cues (negation, emoji, punctuation)."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text).translate(ZERO_WIDTH).translate(DIGIT_MAP)
    if mask_urls:
        t = URL_RE.sub(" URL ", t)
    if mask_users:
        t = MENTION_RE.sub(" @USER ", t)
    if split_hashtags:
        t = HASHTAG_RE.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", t)
    if reduce_elongation:
        t = ELONG_RE.sub(r"\1\1", t)
    if lowercase_latin:
        t = "".join(ch.lower() if ch.isascii() else ch for ch in t)
    return WS_RE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


# --------------------------------------------------------------------------- language id
# Small function-word lists: enough to separate English from romanised Hindi/Bengali.
EN_FUNC = set("the a an is are was were be to of and in that it this for on with as not no you i we they he she "
              "but or so do does did have has had will would can could should about what why how all very me my our your "
              "us by from at more than if up out his her its their them been there here who".split())
ROM_HI = set("hai hain ho tha thi the nahi nahin nhi kya kyu kyun ki ka ke ko se mein me main hum tum aap ye yeh "
             "wo woh bhi aur par lekin sab bahut bohot jo koi kuch kar karo karna raha rahi rahe hota hoti garmi "
             "paani pani mausam log logon desh sarkar bhai ji".split())
ROM_BN = set("ami tumi apni amra tomra ache achhe nei na kore korche korbe hobe hoy hoyeche eta ota ki keno "
             "kemon r ar o er theke jonno kintu shob sob khub onek gorom brishti bristi manush desh sorkar "
             "bhalo kharap dada".split())


def detect_lang(text: str) -> str:
    """Return one of: hi, bn, en, hi-rom, bn-rom, mixed, unk.

    Script decides first; Latin-only text is split into English vs romanised Indic with
    function-word evidence. 'mixed' = substantial share of two scripts (code-mixed).
    """
    toks = TOKEN_RE.findall(text or "")
    if not toks:
        return "unk"
    scripts = Counter(token_script(t) for t in toks)
    scripts.pop("other", None)
    total = sum(scripts.values())
    if total == 0:
        return "unk"
    top, n_top = scripts.most_common(1)[0]
    if n_top / total < 0.8 and len(scripts) > 1:
        return "mixed"
    if top == "deva":
        return "hi"
    if top == "beng":
        return "bn"
    low = [t.lower() for t in toks]
    en = sum(t in EN_FUNC for t in low)
    hi = sum(t in ROM_HI for t in low)
    bn = sum(t in ROM_BN for t in low)
    best = max(en, hi, bn)
    if best == 0:
        return "en"
    return "en" if best == en else ("hi-rom" if best == hi else "bn-rom")


def code_mixing_index(text: str) -> float:
    """Script-level Code-Mixing Index (Das & Gambäck, 2014): 100*(1 - max_lang/(n - u)).

    Tokens are tagged by script; romanised Latin tokens that are Hindi/Bengali function words
    are counted as Indic so that 'global warming bahut serious hai' is recognised as mixed.
    """
    toks = TOKEN_RE.findall(text or "")
    tags = []
    for t in toks:
        s = token_script(t)
        if s == "latn":
            lt = t.lower()
            # English wins ambiguous forms (to, me, the ...)
            s = "en" if lt in EN_FUNC else "hi" if lt in ROM_HI else "bn" if lt in ROM_BN else "en"
        elif s == "deva":
            s = "hi"
        elif s == "beng":
            s = "bn"
        else:
            continue  # numbers etc. = language-independent
        tags.append(s)
    if not tags:
        return 0.0
    c = Counter(tags)
    return 100.0 * (1 - max(c.values()) / len(tags))


# --------------------------------------------------------------------------- lexicons
NEGATION = {
    "hi": ["नहीं", "नही", "ना", "न", "मत", "बिल्कुल नहीं", "कभी नहीं"],
    "bn": ["না", "নয়", "নয়", "নেই", "নাই", "নি", "কখনো না"],
    "rom": ["nahi", "nahin", "nhi", "na", "mat", "nai", "nei", "noy", "not", "no", "never", "dont", "don't", "isn't", "isnt"],
}
CLIMATE_TERMS = {
    "hi": ["जलवायु", "ग्लोबल वार्मिंग", "वार्मिंग", "गर्मी", "तापमान", "प्रदूषण", "पर्यावरण", "बाढ़", "सूखा",
           "पेड़", "कार्बन", "मौसम", "ग्लेशियर", "बारिश", "हीटवेव", "लू"],
    "bn": ["জলবায়ু", "বিশ্ব উষ্ণায়ন", "উষ্ণায়ন", "গ্লোবাল ওয়ার্মিং", "গরম", "তাপমাত্রা", "দূষণ", "পরিবেশ",
           "বন্যা", "খরা", "গাছ", "কার্বন", "আবহাওয়া", "হিমবাহ", "বৃষ্টি", "তাপপ্রবাহ"],
    "en": ["climate", "global warming", "warming", "heat", "temperature", "pollution", "environment", "flood",
           "drought", "tree", "carbon", "co2", "weather", "glacier", "rain", "heatwave", "emission", "fossil"],
}
DENIAL_TERMS = {
    "hi": ["झूठ", "धोखा", "नाटक", "बकवास", "अफवाह", "प्रोपेगेंडा", "साजिश", "जुमला", "ड्रामा"],
    "bn": ["মিথ্যা", "ভুয়া", "নাটক", "বাজে কথা", "গুজব", "ষড়যন্ত্র", "প্রোপাগান্ডা", "ধাপ্পা"],
    "en": ["hoax", "scam", "fake", "lie", "lies", "propaganda", "agenda", "fraud", "myth", "drama", "jhooth", "bakwas"],
}


def _contains_any(text: str, words) -> bool:
    # NFC on both sides: e.g. Bengali য় (U+09DF) is a composition exclusion and decomposes under NFC.
    low = unicodedata.normalize("NFC", text.lower())
    toks = set(tokenize(low))
    for w in words:
        w = unicodedata.normalize("NFC", w)
        if " " in w:
            if w in low:
                return True
        elif w in toks:
            return True
    return False


def has_negation(text: str) -> bool:
    return _contains_any(text, [w for v in NEGATION.values() for w in v])


def has_climate_term(text: str) -> bool:
    return _contains_any(text, [w for v in CLIMATE_TERMS.values() for w in v])


def has_denial_term(text: str) -> bool:
    return _contains_any(text, [w for v in DENIAL_TERMS.values() for w in v])


def surface_features(text: str) -> dict:
    """Per-text descriptors used by EDA and error analysis."""
    t = text or ""
    toks = TOKEN_RE.findall(t)
    return {
        "n_chars": len(t),
        "n_tokens": len(toks),
        "n_emoji": len(EMOJI_RE.findall(t)),
        "has_url": bool(URL_RE.search(t)),
        "has_mention": bool(MENTION_RE.search(t)),
        "has_hashtag": bool(HASHTAG_RE.search(t)),
        "has_elongation": bool(re.search(r"(\w)\1{2,}", t)),
        "n_exclaim": t.count("!"),
        "n_question": t.count("?"),
        "cmi": code_mixing_index(t),
        "detected_lang": detect_lang(t),
        "has_negation": has_negation(t),
        "has_climate_term": has_climate_term(t),
        "has_denial_term": has_denial_term(t),
    }


STOPWORDS = EN_FUNC | ROM_HI | ROM_BN | set(
    # Hindi
    "का के की है हैं को में से और यह वह ये वो भी तो ही पर था थी थे हो कर एक इस उस जो कि लिए नहीं ना क्या "
    "रहा रही रहे गया गई होता होती हम आप तुम मैं सब कुछ अब बहुत साथ जब तक "
    # Bengali
    "এই সেই আর ও কি যে না এবং তা তার এর একটি করে হয় হবে ছিল আমি আমরা তুমি আপনি সব কিন্তু থেকে জন্য সাথে "
    "দিয়ে নিয়ে এখন খুব কোন কেন".split()
)
