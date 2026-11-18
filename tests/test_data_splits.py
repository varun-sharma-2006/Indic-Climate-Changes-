"""Checks that the processed datasets have no train/test leakage.

A sentence and its Hindi/Bengali translations share one id and one split, so a test
sentence can never appear (in another language) in the training data.

Run:  python -m pytest tests
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import LABELS, PROCESSED_DIR  # noqa: E402

FILES = {"en": "english.csv", "hi": "hindi.csv", "bn": "bengali.csv"}


@pytest.fixture(scope="module")
def data():
    return {l: pd.read_csv(PROCESSED_DIR / f, keep_default_na=False) for l, f in FILES.items()}


def test_same_rows_in_every_language(data):
    assert len(data["en"]) == len(data["hi"]) == len(data["bn"])


def test_ids_are_unique(data):
    for df in data.values():
        assert df["id"].is_unique


def test_translations_keep_split_and_label(data):
    en = data["en"].set_index("id")[["split", "label"]]
    for lang in ("hi", "bn"):
        other = data[lang].set_index("id")[["split", "label"]].loc[en.index]
        assert (other["split"] == en["split"]).all(), f"{lang}: a sentence changed split"
        assert (other["label"] == en["label"]).all(), f"{lang}: a sentence changed label"


def test_no_id_in_two_splits(data):
    en = data["en"]
    train, test = set(en[en.split == "train"].id), set(en[en.split == "test"].id)
    assert not train & test


def test_labels_are_valid(data):
    for df in data.values():
        assert set(df["label"]) <= set(LABELS)
        assert (df["text"].str.strip() != "").all()
