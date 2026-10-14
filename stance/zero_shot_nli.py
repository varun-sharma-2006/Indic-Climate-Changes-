"""Training-free baseline: stance detection as multilingual Natural Language Inference.

Stance toward a claim is close to entailment/contradiction (Allaway & McKeown 2020; Laurer et al. 2024).
A multilingual NLI model (mDeBERTa-v3 trained on XNLI + multilingual NLI) is asked, for each comment
(premise), whether it entails each hypothesis. Hypotheses are written in English: the cross-lingual NLI
model aligns Hindi/Bengali premises with English hypotheses.

Decision rule:
    p_rel = P(entail | "This text is about climate, weather, heat, pollution or the environment.")
    s_fav = P(entail | favour hypotheses),  s_ag = P(entail | against hypotheses)
    if p_rel < tau and max(s_fav, s_ag) < tau -> None, else argmax over {Favour: s_fav, Against: s_ag, None: 1 - p_rel}

Usage:
    python -m stance.zero_shot_nli --input data/challenge/challenge_hi_bn.csv
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import LABELS, RUNS
from .data_io import load_unified
from .metrics import by_group, compute, confusion, dump, pretty
from .predictors import write_predictions
from .text import normalize

HYP = {
    "Favour": ["Climate change is a serious problem.",
               "The author is worried about global warming.",
               "Global warming is real and dangerous."],
    "Against": ["Climate change is not a serious problem.",
                "Global warming is a hoax or exaggerated.",
                "The author does not believe in climate change."],
    "Relevant": ["This text is about climate, weather, heat, pollution or the environment."],
}


class NLIStance:
    def __init__(self, model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", batch_size=16, tau=0.5):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.dev).eval()
        l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ent = l2i.get("entailment", 0)
        self.bs, self.tau = batch_size, tau

    @torch.no_grad()
    def _entail(self, premises, hyp):
        out = []
        for i in range(0, len(premises), self.bs):
            p = premises[i:i + self.bs]
            enc = self.tok(p, [hyp] * len(p), truncation="only_first", max_length=256, padding=True, return_tensors="pt").to(self.dev)
            with torch.autocast(self.dev, enabled=self.dev == "cuda"):
                logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, -1)[:, self.ent].cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, texts, langs=None):
        prem = [normalize(t) for t in texts]
        s = {k: np.mean([self._entail(prem, h) for h in tqdm(hs, desc=k, leave=False)], axis=0) for k, hs in HYP.items()}
        fav, ag, rel = s["Favour"], s["Against"], s["Relevant"]
        none = 1 - rel
        none = np.where((rel < self.tau) & (np.maximum(fav, ag) < self.tau), np.maximum(none, 1.0), none)
        m = np.stack([fav, ag, none], 1)
        return m / m.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    out = RUNS / "zero_shot_nli"
    nli = NLIStance(a.model, tau=a.tau)
    res = {}
    for p in a.input:
        df = load_unified(p)
        probs = nli.predict_proba(df["text"].tolist())
        pdf = write_predictions(df, probs, out / f"pred_{Path(p).stem}.csv")
        g = pdf[pdf["label"].isin(LABELS)]
        if len(g):
            res[Path(p).stem] = compute(g["label"], g["pred"])
            print(pretty(res[Path(p).stem], f"zero-shot NLI on {Path(p).stem}"))
            print(confusion(g["label"], g["pred"]).to_string())
            print(by_group(g, "lang").to_string(index=False))
    dump(res, out / "metrics.json")


if __name__ == "__main__":
    main()
