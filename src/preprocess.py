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


# (step name, function) in the order they are applied; the web demo shows each step
STEPS = [
    ("HTML entities → characters", html.unescape),
    ("Unicode normalisation (NFC)", lambda t: unicodedata.normalize("NFC", t)),
    ("Remove URLs", lambda t: URL_RE.sub(" ", t)),
    ("@mentions → @user", lambda t: MENTION_RE.sub("@user", t)),
    ("Split hashtags", lambda t: HASHTAG_RE.sub(split_hashtag, t)),
    ("Emojis → words",  # emoji names use underscores: face_with_tears_of_joy
     lambda t: re.sub(r"(?<=\w)_(?=\w)", " ", emoji.demojize(t, delimiters=(" ", " ")))),
    ("Shorten repeated letters", lambda t: REPEAT_RE.sub(r"\1\1", t)),
    ("Merge @user chains, extra spaces",  # long reply chains on Twitter
     lambda t: SPACE_RE.sub(" ", re.sub(r"(@user\s*){2,}", "@user ", t)).strip()),
]


def clean_text_steps(text: str):
    """Returns [(step name, text after the step), ...]."""
    out = []
    if not isinstance(text, str):
        text = ""
    for name, fn in STEPS:
        text = fn(text)
        out.append((name, text))
    return out


def clean_text(text: str) -> str:
    return clean_text_steps(text)[-1][1]


def show_steps(text: str):
    """Print what each cleaning step changes (used in the live demo)."""
    print(f"input : {text}")
    prev = text
    for name, out in clean_text_steps(text):
        mark = "changed" if out != prev else "-"
        print(f"  {name:34s} {mark:8s} {out}")
        prev = out
    print(f"output: {prev}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Show the text cleaning step by step.")
    ap.add_argument("--text", help="clean this text (otherwise three built-in examples)")
    args = ap.parse_args()
    if args.text:
        show_steps(args.text)
        raise SystemExit
    samples = [
        "@a @b @c Climate change is a HOAX!!! 😂😂 https://t.co/xyz #ClimateHoax",
        "ग्लोबल वार्मिंग बहुतttt बढ़ गई है 🔥🔥",
        "জলবায়ু পরিবর্তন সত্যিই ভয়ংকর &amp; চিন্তার বিষয়",
    ]
    for s in samples:
        print(f"{s}\n -> {clean_text(s)}\n")
