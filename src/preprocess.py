"""Text cleaning for noisy social media text (English, Hindi, Bengali).

Steps:
  1. HTML entities (&amp;) -> characters, Unicode NFC normalisation
  2. URLs removed, @mentions -> "@user"
  3. hashtags split into words  (#ClimateChange -> climate change)
  4. emojis -> words (🔥 -> fire), because emojis often carry the stance
  5. letters repeated 3+ times shortened to 2 (bahuttt -> bahutt)
  6. extra whitespace removed
"""
import html
import re
import unicodedata

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")
SPACE_RE = re.compile(r"\s+")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def split_hashtag(match: re.Match) -> str:
    word = match.group(1)
    return CAMEL_RE.sub(" ", word).replace("_", " ").lower()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub("@user", text)
    text = HASHTAG_RE.sub(split_hashtag, text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)  # emoji names use underscores
    text = REPEAT_RE.sub(r"\1\1", text)
    text = re.sub(r"(@user\s*){2,}", "@user ", text)  # long reply chains on Twitter
    text = SPACE_RE.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    samples = [
        "@a @b @c Climate change is a HOAX!!! 😂😂 https://t.co/xyz #ClimateHoax",
        "ग्लोबल वार्मिंग बहुतttt बढ़ गई है 🔥🔥",
        "জলবায়ু পরিবর্তন সত্যিই ভয়ংকর &amp; চিন্তার বিষয়",
    ]
    for s in samples:
        print(f"{s}\n -> {clean_text(s)}\n")
