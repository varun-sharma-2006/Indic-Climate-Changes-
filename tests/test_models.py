"""Checks of the mean-pooling classifier in src/models.py (the fix for MuRIL's [CLS] collapse).

Uses a tiny fake encoder, so no model download or GPU is needed.
Run:  python -m pytest tests
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import MeanPoolClassifier  # noqa: E402


class FakeEncoder(nn.Module):
    """Returns the token ids themselves as 1-dimensional 'hidden states'."""
    config = SimpleNamespace(hidden_size=1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float())


def pooled(input_ids, attention_mask):
    model = MeanPoolClassifier(FakeEncoder(), num_labels=1, dropout=0.0)
    model.classifier.weight.data.fill_(1.0)  # identity head: logit = pooled value
    model.classifier.bias.data.zero_()
    return model(torch.tensor(input_ids), torch.tensor(attention_mask)).logits.squeeze(-1)


def test_mean_of_real_tokens():
    assert pooled([[2, 4, 6]], [[1, 1, 1]]).item() == 4.0


def test_padding_is_ignored():
    # same sentence with two padding tokens (value 100) must give the same vector
    assert pooled([[2, 4, 6, 100, 100]], [[1, 1, 1, 0, 0]]).item() == 4.0


def test_different_sentences_give_different_vectors():
    # [CLS] pooling in MuRIL gave almost identical vectors for every sentence; mean pooling must not
    out = pooled([[1, 1, 0], [9, 9, 9]], [[1, 1, 0], [1, 1, 1]])
    assert out[0].item() == 1.0 and out[1].item() == 9.0


def test_output_has_one_logit_per_label():
    model = MeanPoolClassifier(FakeEncoder(), num_labels=3)
    out = model(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))
    assert out.logits.shape == (1, 3)
