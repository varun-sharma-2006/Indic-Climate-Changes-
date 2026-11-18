"""Checks of compute_metrics() in src/data_utils.py and of why we report Macro F1.

Run:  python -m pytest tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import compute_metrics  # noqa: E402

FAVOUR, AGAINST, NONE = 0, 1, 2


def test_perfect_predictions():
    y = [FAVOUR, AGAINST, NONE, FAVOUR]
    m = compute_metrics(y, y)
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_majority_class_has_high_accuracy_but_low_macro_f1():
    # like our data: Favour is the large class; a model that always says Favour looks good on accuracy
    y_true = [FAVOUR] * 65 + [AGAINST] * 13 + [NONE] * 22
    y_pred = [FAVOUR] * 100
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.65
    assert m["macro_f1"] < 0.30  # Favour F1 0.79, Against 0, None 0 -> about 0.26
    assert m["per_class_f1"]["Against"] == 0.0


def test_confusion_matrix_rows_are_true_labels():
    m = compute_metrics([AGAINST, AGAINST], [FAVOUR, AGAINST])
    assert m["confusion_matrix"][AGAINST] == [1, 1, 0]  # one Against predicted as Favour, one correct
