"""Fine-tune a multilingual encoder (XLM-R, MuRIL, IndicBERT v2, mBERT, mDeBERTa ...) for 3-way stance.

Design choices (see docs/05_methodology.md):
  * target-aware input  : [CLS] comment [SEP] claim [SEP]  (--pair-with-claim), claim in en or the comment's language
  * frozen word embeddings by default: keeps the multilingual embedding space aligned for cross-lingual
    transfer (Wu & Dredze 2019) AND removes ~70% of trainable params -> fits a 4 GB GPU
  * class-weighted cross-entropy + label smoothing for imbalance and noisy labels
  * group-balanced sampling (--sampler lang) so translated Hindi/Bengali rows are not drowned by English
  * model selection on macro-F1 of a chosen eval set, early stopping, fp16 autocast, gradient accumulation
  * optional LoRA (--lora) for parameter-efficient training

Usage:
    python -m stance.train --data data/processed/en_only --model xlm-roberta-base --pair-with-claim \
        --eval data/challenge/challenge_hi_bn.csv --name xlmr_en_only
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import CLAIM, ID2LABEL, LABEL2ID, LABELS, RUNS
from .data_io import load_unified
from .metrics import by_group, compute, confusion, dump, pretty
from .predictors import write_predictions
from .text import normalize


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def claim_for(lang, claim_lang):
    if claim_lang == "match":
        base = str(lang).split("-")[0]
        return CLAIM.get(base, CLAIM["en"])
    return CLAIM[claim_lang]


def build_inputs(tok, texts, langs, pair, claim_lang, max_len):
    texts = [normalize(t) for t in texts]
    if pair:
        langs = langs if langs is not None else ["en"] * len(texts)
        claims = [claim_for(l, claim_lang) for l in langs]
        return tok(texts, claims, truncation="only_first", max_length=max_len, padding=True, return_tensors="pt")
    return tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def make_sampler(df, mode):
    if mode == "random":
        return None
    key = {"lang": df["lang"], "label": df["label"], "lang_label": df["lang"] + "|" + df["label"]}[mode]
    w = 1.0 / key.map(key.value_counts())
    return WeightedRandomSampler(torch.tensor(w.values, dtype=torch.double), num_samples=len(df), replacement=True)


@torch.no_grad()
def predict(model, tok, df, a, dev):
    model.eval()
    probs = []
    for i in range(0, len(df), a.eval_batch_size):
        s = df.iloc[i:i + a.eval_batch_size]
        enc = build_inputs(tok, s["text"].tolist(), s["lang"].tolist(), a.pair_with_claim, a.claim_lang, a.max_len)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast(dev, enabled=a.fp16):
            logits = model(**enc).logits.float()
        probs.append(torch.softmax(logits, -1).cpu().numpy())
    model.train()
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", nargs="*", default=[], help="extra labelled sets evaluated every epoch")
    ap.add_argument("--select-on", default="valid", help="'valid' or the file stem of an --eval set used for model selection")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--class-weights", default="sqrt", choices=["none", "balanced", "sqrt"])
    ap.add_argument("--sampler", default="random", choices=["random", "lang", "label", "lang_label"])
    ap.add_argument("--pair-with-claim", action="store_true")
    ap.add_argument("--claim-lang", default="en", choices=["en", "hi", "bn", "match"])
    ap.add_argument("--no-freeze-embeddings", dest="freeze_embeddings", action="store_false")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate every N optimizer steps (0 = per epoch only)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.fp16 = dev == "cuda"
    data = Path(a.data)
    a.name = a.name or f"{a.model.split('/')[-1]}_{data.name}"
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"run: {a.name}   model: {a.model}   device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)" if dev == "cuda" else
             "   [WARNING: CPU only - install CUDA PyTorch, see README]"))

    tr = load_unified(data / "train.csv")
    tr = tr[tr["label"].notna()].reset_index(drop=True)
    if a.max_train_rows and len(tr) > a.max_train_rows:
        tr = tr.sample(a.max_train_rows, random_state=a.seed).reset_index(drop=True)
    evals = {"valid": load_unified(data / "valid.csv")}
    for p in a.eval:
        evals[Path(p).stem] = load_unified(p)
    evals = {k: v[v["label"].notna()].reset_index(drop=True) for k, v in evals.items()}
    if a.select_on not in evals:
        raise SystemExit(f"--select-on {a.select_on} not in {list(evals)}")
    print(f"train {len(tr)} | " + " | ".join(f"{k} {len(v)}" for k, v in evals.items()))
    print("train label dist:", tr["label"].value_counts().to_dict())
    print("train lang dist :", tr["lang"].value_counts().to_dict())

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)
    if a.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable()
    if a.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.1,
                                                 target_modules=["query", "key", "value", "dense"]))
    model.to(dev)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_all / 1e6:.1f}M total, {n_tr / 1e6:.1f}M trainable ({n_tr / n_all:.1%})")

    # token-length check on a sample
    lens = [len(tok(t)["input_ids"]) for t in tr["text"].sample(min(500, len(tr)), random_state=0).map(normalize)]
    print(f"subword length: median {np.median(lens):.0f}, p95 {np.percentile(lens, 95):.0f}, "
          f"truncated at {a.max_len}: {np.mean(np.array(lens) > a.max_len):.1%}")

    counts = tr["label"].value_counts().reindex(LABELS).fillna(0).values.astype(float)
    if a.class_weights == "none":
        w = np.ones(len(LABELS))
    else:
        w = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
        w = np.sqrt(w) if a.class_weights == "sqrt" else w
    print("class weights:", dict(zip(LABELS, np.round(w, 3))))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=dev), label_smoothing=a.label_smoothing)

    head_names = ("classifier", "score", "pooler")
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(h in n for h in head_names)], "lr": a.lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(h in n for h in head_names)], "lr": a.head_lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    sampler = make_sampler(tr, a.sampler)
    loader = DataLoader(list(range(len(tr))), batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler)
    total_steps = math.ceil(len(loader) / a.grad_accum) * a.epochs
    warm = int(a.warmup * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    scaler = torch.amp.GradScaler(enabled=a.fp16)
    json.dump(vars(a), open(out / "train_args.json", "w"), indent=2)

    history, best, bad, step = [], -1.0, 0, 0
    t_start = time.time()

    def evaluate(tag):
        nonlocal best, bad
        row = {"epoch": tag, "step": step, "elapsed_min": round((time.time() - t_start) / 60, 2)}
        for k, df in evals.items():
            p = predict(model, tok, df, a, dev)
            m = compute(df["label"], [LABELS[i] for i in p.argmax(1)])
            row[f"{k}_macro_f1"] = round(m["macro_f1"], 4)
            row[f"{k}_acc"] = round(m["accuracy"], 4)
            print(pretty(m, f"{k} @ {tag}"))
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        score = row[f"{a.select_on}_macro_f1"]
        if score > best:
            best, bad = score, 0
            if a.lora:  # adapter only; merged into the base model after training
                model.save_pretrained(out / "best_adapter")
            else:
                model.save_pretrained(out / "best")
                tok.save_pretrained(out / "best")
                json.dump(vars(a), open(out / "best" / "train_args.json", "w"), indent=2)
            print(f"  ** new best {a.select_on} macro-F1 = {best:.4f}  -> saved")
            return True
        bad += 1
        print(f"  (no improvement, best {best:.4f}, patience {bad}/{a.patience})")
        return False

    model.train()
    for ep in range(1, a.epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {ep}/{a.epochs}", unit="batch", dynamic_ncols=True)
        run_loss, seen = 0.0, 0
        for bi, idx in enumerate(pbar):
            b = tr.iloc[idx.tolist()]
            enc = build_inputs(tok, b["text"].tolist(), b["lang"].tolist(), a.pair_with_claim, a.claim_lang, a.max_len)
            enc = {k: v.to(dev) for k, v in enc.items()}
            y = torch.tensor([LABEL2ID[l] for l in b["label"]], device=dev)
            with torch.autocast(dev, enabled=a.fp16):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), y) / a.grad_accum
            scaler.scale(loss).backward()
            if (bi + 1) % a.grad_accum == 0 or bi + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if a.eval_every and step % a.eval_every == 0:
                    evaluate(f"ep{ep}-step{step}")
            run_loss += loss.item() * a.grad_accum
            seen += 1
            mem = f"{torch.cuda.max_memory_allocated() / 2**30:.2f}GB" if dev == "cuda" else "-"
            pbar.set_postfix(loss=f"{run_loss / seen:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}", mem=mem)
        print(f"epoch {ep} train loss {run_loss / max(seen, 1):.4f}")
        evaluate(f"epoch{ep}")
        if bad >= a.patience:
            print(f"[early stop] no improvement for {a.patience} evaluations")
            break

    # ---------------- final evaluation with the best checkpoint
    print("=" * 80)
    print(f"training done in {(time.time() - t_start) / 60:.1f} min; reloading best checkpoint")
    if a.lora:
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(
            a.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)
        merged = PeftModel.from_pretrained(base, out / "best_adapter").merge_and_unload()
        merged.save_pretrained(out / "best")
        tok.save_pretrained(out / "best")
        json.dump(vars(a), open(out / "best" / "train_args.json", "w"), indent=2)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(out / "best").to(dev)
    final = {}
    for k, df in evals.items():
        p = predict(model, tok, df, a, dev)
        pdf = write_predictions(df, p, out / f"pred_{k}.csv")
        final[k] = compute(pdf["label"], pdf["pred"])
        print(pretty(final[k], f"BEST on {k}"))
        print(confusion(pdf["label"], pdf["pred"]).to_string())
        print(by_group(pdf, "lang").to_string(index=False))
    dump({"best_select_score": best, "final": final, "args": vars(a)}, out / "metrics.json")
    print(f"\nall outputs in {out}")


if __name__ == "__main__":
    main()
