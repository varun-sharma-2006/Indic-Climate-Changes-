"""Step 4b: fine-tune a transformer for 3-class stance classification.

Examples:
  # base paper setting: English BERT trained on English data only
  python src/train_transformer.py --model bert --train_langs en

  # zero-shot cross-lingual: multilingual model trained on English only
  python src/train_transformer.py --model muril --train_langs en

  # translate-train (our main setting): English + Hindi + Bengali
  python src/train_transformer.py --model muril --train_langs en hi bn --save

Details:
  * weighted cross-entropy loss (Against is only ~13% of the data)
  * the word-embedding layer is frozen: it saves GPU memory (XLM-R/MuRIL do not
    fit in 4 GB otherwise) and keeps the multilingual word vectors aligned
  * mixed precision (fp16) on GPU
  * MuRIL uses mean pooling instead of the [CLS] token (see src/models.py)
  * best epoch chosen by Macro F1 on the validation set (same languages as training)
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import LABEL2ID, LABELS, MODEL_NAMES, MODELS_DIR, PREDS_DIR, set_seed
from data_utils import available_test_langs, compute_metrics, load_split, save_results
from models import build_model, load_trained
from preprocess import clean_text

TEST_LANGS = available_test_langs()


def batches(df, tok, max_len, bs, shuffle):
    texts, labels = df["clean"].tolist(), df["y"].tolist()
    idx = list(range(len(texts)))

    def collate(ix):
        enc = tok([texts[i] for i in ix], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in ix])
        return enc

    return DataLoader(idx, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def freeze_embeddings(model):
    for p in model.base_model.embeddings.word_embeddings.parameters():
        p.requires_grad = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--train_langs", nargs="+", default=["en", "hi", "bn"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pooling", choices=["cls", "mean"], default=None,
                    help="default: mean for MuRIL, cls for the others")
    ap.add_argument("--extra_train", default=None,
                    help="extra training CSV from src/extra_data.py (id, source, lang, split, text, label)")
    ap.add_argument("--repeat_synthetic", type=int, default=3,
                    help="how many times the small synthetic comment set is repeated in --extra_train")
    ap.add_argument("--init_dir", default=None,
                    help="continue training from a saved model (e.g. models/muril_tt)")
    ap.add_argument("--save", action="store_true", help="save the best model (needed for predict.py / LIME)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    pooling = args.pooling or ("mean" if args.model == "muril" else "cls")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setting = "en" if args.train_langs == ["en"] else "tt" if set(args.train_langs) == {"en", "hi", "bn"} else "_".join(args.train_langs)
    tag = args.tag or f"{args.model}_{setting}"
    hf_name = MODEL_NAMES[args.model]
    print(f"== {tag}: {hf_name} on {device}, train langs {args.train_langs}")

    train = load_split(args.train_langs, "train")
    if args.extra_train:
        extra = pd.read_csv(args.extra_train, keep_default_na=False)
        extra = extra[extra["lang"].isin(args.train_langs)]
        syn = extra[extra["source"] == "SyntheticYouTubeStyle"]
        extra = pd.concat([extra] + [syn] * (args.repeat_synthetic - 1), ignore_index=True)
        extra["clean"] = extra["text"].astype(str).map(clean_text)
        extra["y"] = extra["label"].map(LABEL2ID)
        print(f"extra training rows: {len(extra)} ({extra['source'].value_counts().to_dict()})")
        train = pd.concat([train, extra[train.columns.intersection(extra.columns)]], ignore_index=True)
    val = load_split(args.train_langs, "val")
    tests = {l: load_split([l], "test") for l in TEST_LANGS}

    if args.init_dir:  # continued fine-tuning of one of our saved models
        tok = AutoTokenizer.from_pretrained(args.init_dir)
        model = load_trained(args.init_dir).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(hf_name)
        model = build_model(hf_name, pooling).to(device)
    freeze_embeddings(model)

    counts = np.bincount(train["y"], minlength=len(LABELS))
    weights = torch.tensor(len(train) / (len(LABELS) * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    train_dl = batches(train, tok, args.max_len, args.batch_size, shuffle=True)
    val_dl = batches(val, tok, args.max_len, 64, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, best_state, history = -1, None, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        val_f1 = compute_metrics(val["y"], predict_probs(model, val_dl, device).argmax(1))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val_macro_f1": val_f1})
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    results = {}
    for lang, test in tests.items():
        probs = predict_probs(model, batches(test, tok, args.max_len, 64, shuffle=False), device)
        results[lang] = compute_metrics(test["y"], probs.argmax(1))
        pd.DataFrame({"id": test["id"], "y": test["y"], **{f"p_{l}": probs[:, i] for i, l in enumerate(LABELS)}}) \
            .to_csv(PREDS_DIR / f"{tag}_{lang}.csv", index=False)

    save_results(tag, results, {
        "model": hf_name, "pooling": pooling, "init_dir": args.init_dir, "extra_train": args.extra_train, "repeat_synthetic": args.repeat_synthetic, "train_langs": args.train_langs, "n_train": len(train),
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_len": args.max_len,
        "best_val_macro_f1": best_f1, "history": history, "train_minutes": round((time.time() - start) / 60, 1),
    })

    if args.save:
        out = MODELS_DIR / tag
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"Model saved -> {out}")


if __name__ == "__main__":
    main()
