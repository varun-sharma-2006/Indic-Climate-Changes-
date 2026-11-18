"""Small checks of the text cleaning in src/preprocess.py.

Run:  python -m pytest tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import clean_text  # noqa: E402


def test_url_removed():
    assert clean_text("Check this https://t.co/abc now") == "Check this now"


def test_mentions_become_one_user_token():
    assert clean_text("@greta @un this is real") == "@user this is real"


def test_hashtag_split_into_words():
    assert clean_text("#ClimateHoax is fake") == "climate hoax is fake"


def test_emoji_converted_to_words():
    # emojis often carry the stance (😂 on a denial comment), so we keep them as words
    assert clean_text("so hot 😂") == "so hot face with tears of joy"


def test_repeated_letters_shortened():
    assert clean_text("sooooo hot!!!!") == "soo hot!!"


def test_html_entities_unescaped():
    assert clean_text("I &amp; you") == "I & you"


def test_hindi_and_bengali_unchanged():
    for text in ["ग्लोबल वार्मिंग एक गंभीर समस्या है", "গ্লোবাল ওয়ার্মিং হচ্ছে না"]:
        assert clean_text(text) == text
