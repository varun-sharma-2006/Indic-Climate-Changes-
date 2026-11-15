"""Model builders shared by training and prediction.

pooling="cls"  - standard Hugging Face classifier on the [CLS] token
pooling="mean" - average of all token vectors + linear layer.
                 Needed for MuRIL: its [CLS] vector is almost identical for every
                 input sentence, and with the [CLS] head the model did not learn.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification

from config import ID2LABEL, LABEL2ID, LABELS

HEAD_FILE = "mean_pool_head.pt"


class MeanPoolClassifier(nn.Module):
    def __init__(self, encoder, num_labels=len(LABELS), dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)

    @property
    def base_model(self):
        return self.encoder

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return SimpleNamespace(logits=self.classifier(self.dropout(pooled)))

    def save_pretrained(self, out_dir):
        self.encoder.save_pretrained(out_dir)
        torch.save(self.classifier.state_dict(), Path(out_dir) / HEAD_FILE)

    @classmethod
    def from_saved(cls, model_dir):
        model = cls(AutoModel.from_pretrained(model_dir))
        model.classifier.load_state_dict(torch.load(Path(model_dir) / HEAD_FILE, map_location="cpu"))
        return model


def build_model(hf_name, pooling="cls"):
    if pooling == "mean":
        return MeanPoolClassifier(AutoModel.from_pretrained(hf_name))
    return AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)


def load_trained(model_dir):
    if (Path(model_dir) / HEAD_FILE).exists():
        return MeanPoolClassifier.from_saved(model_dir)
    return AutoModelForSequenceClassification.from_pretrained(model_dir)
