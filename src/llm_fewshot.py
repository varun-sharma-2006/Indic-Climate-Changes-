"""Step 4c: LLM baseline - few-shot prompting of a small instruction-tuned LLM.

The LLM is given the claim, short label definitions and one example per label
(taken from the training set of the same language), then asked for one word.
Instead of free generation we compare the model's scores for the three label
words, so the answer is always a valid label.

Model: Qwen2.5-1.5B-Instruct (multilingual, fits in 4 GB GPU memory in fp16).
Usage:  python src/llm_fewshot.py --limit 300
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CLAIM, LABELS, set_seed
from data_utils import compute_metrics, load_split, save_results

LLM = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_LANGS = ["en", "hi", "bn"]

INSTRUCTIONS = (
    f'Claim: "{CLAIM}"\n'
    "Decide the stance of the text towards the claim.\n"
    "Favour = agrees that climate change / global warming is real, harmful or needs action.\n"
    "Against = denies, mocks or downplays climate change or the need to act on it.\n"
    "None = neutral, unrelated, or no clear stance.\n"
    "Answer with exactly one word: Favour, Against or None."
)


def build_prompt(tok, examples, text):
    msgs = [{"role": "system", "content": INSTRUCTIONS}]
    for ex_text, ex_label in examples:
        msgs += [{"role": "user", "content": f"Text: {ex_text}"}, {"role": "assistant", "content": ex_label}]
    msgs.append({"role": "user", "content": f"Text: {text}"})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def classify(model, tok, prompt, label_ids, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1]
    return int(torch.stack([logits[i] for i in label_ids]).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="test examples per language (default: all)")
    args = ap.parse_args()
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(LLM)
    model = AutoModelForCausalLM.from_pretrained(
        LLM, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    # first token of each label word decides the answer
    label_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LABELS]
    assert len(set(label_ids)) == 3, "label words must start with different tokens"

    results = {}
    for lang in TEST_LANGS:
        train = load_split([lang], "train")
        examples = [(train[train["label"] == l]["clean"].iloc[0], l) for l in LABELS]
        test = load_split([lang], "test")
        if args.limit:
            test = test.groupby("y", group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, int(args.limit * len(g) / len(test)))), random_state=42))
        preds = [classify(model, tok, build_prompt(tok, examples, t), label_ids, device)
                 for t in tqdm(test["clean"], desc=lang)]
        results[lang] = compute_metrics(test["y"], preds)

    save_results("llm_qwen1.5b_fewshot", results, {
        "model": LLM, "train_langs": ["prompt only"], "n_test_per_lang": args.limit or "all",
        "prompt": INSTRUCTIONS, "shots": "1 example per label from the same language"})


if __name__ == "__main__":
    main()
