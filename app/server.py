"""Web demo: shows step by step how our model classifies a comment.

Run:   python app/server.py
Open:  http://127.0.0.1:5000

API
  POST /api/predict  {"text": ..., "model": "muril_tt"}  -> every step of the pipeline + prediction + word importance
  POST /api/batch    CSV upload (ID,text)                -> CSV download (ID, Predicted Label)
  GET  /api/info                                       -> available models, examples, results, dataset statistics
"""
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402,F401  (sets the model cache folder before transformers is imported)
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402

from config import FIG_DIR, LABELS, METRICS_DIR, MODELS_DIR, PROCESSED_DIR  # noqa: E402
from predict import StancePredictor  # noqa: E402
from preprocess import clean_text_steps  # noqa: E402

app = Flask(__name__, static_folder=str(ROOT / "app" / "static"), static_url_path="/static")
app.json.sort_keys = False  # keep Favour / Against / None order

MODEL_INFO = {  # first available model is the default
    "muril_robust2": "MuRIL robust: + SemEval, Kaggle tweets, YouTube-style comments (best for typed comments)",
    "muril_tt": "MuRIL, translate-train (best on the formal test sets)",
    "muril_tt_rom": "MuRIL + romanised training (for Hindi/Bengali in English letters)",
    "xlmr_tt": "XLM-R, translate-train",
}

EXAMPLES = [
    {"lang": "Hindi", "text": "ग्लोबल वार्मिंग एक गंभीर समस्या है, हमें अभी कदम उठाने होंगे"},
    {"lang": "Hindi", "text": "ये सब क्लाइमेट चेंज का ड्रामा है, गर्मी तो हमेशा पड़ती थी"},
    {"lang": "Hindi", "text": "गर्मी हर साल बढ़ रही है, ये जलवायु परिवर्तन का असर है 🔥🔥"},
    {"lang": "Bengali", "text": "জলবায়ু পরিবর্তন সত্যিই ভয়ংকর"},
    {"lang": "Bengali", "text": "ভিডিওটা দারুণ হয়েছে দাদা, পরের পর্ব কবে?"},
    {"lang": "Bengali", "text": "গ্লোবাল ওয়ার্মিং হচ্ছে না"},
    {"lang": "Romanised", "text": "global warming bahut serious problem hai yaar, sabko milke kuch karna chahiye"},
    {"lang": "English", "text": "@user Climate change is a HOAX!!! 😂😂 https://t.co/xyz #ClimateHoax"},
]

ROMAN_HINTS = {"hai", "nahi", "bhai", "yaar", "kya", "bahut", "hain", "raha", "rahi", "karna", "ki", "ka", "ko",
               "se", "ye", "sab", "ami", "tumi", "khub", "kichu", "bhalo", "ache", "hobe", "garmi"}

_loaded = {"name": None, "predictor": None}


def get_predictor(name):
    """Keeps one model in memory at a time (the laptop GPU has 4 GB)."""
    if _loaded["name"] != name:
        _loaded["predictor"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _loaded["predictor"] = StancePredictor(MODELS_DIR / name)
        _loaded["name"] = name
    return _loaded["predictor"]


def default_model():
    return next(k for k in MODEL_INFO if (MODELS_DIR / k).exists())


def detect_script(text):
    counts = {
        "Devanagari (Hindi)": len(re.findall(r"[ऀ-ॿ]", text)),
        "Bengali script": len(re.findall(r"[ঀ-৿]", text)),
        "Latin letters": len(re.findall(r"[A-Za-z]", text)),
    }
    total = sum(counts.values()) or 1
    main = max(counts, key=counts.get)
    note = ""
    if main == "Latin letters":
        words = set(re.findall(r"[a-z]+", text.lower()))
        if len(words & ROMAN_HINTS) >= 2:
            note = "Looks like Hindi/Bengali typed in English letters (romanised): try the romanised model."
        else:
            note = "Looks like English."
    return {"main": main, "shares": {k: round(100 * v / total) for k, v in counts.items() if v}, "note": note}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/figures/<path:name>")
def figures(name):
    return send_from_directory(FIG_DIR, name)


@app.get("/api/info")
def info():
    models = {k: v for k, v in MODEL_INFO.items() if (MODELS_DIR / k).exists()}
    results = []
    order = ["tfidf_lr_en", "bert_en", "mbert_en", "xlmr_en", "muril_en", "llm_qwen1.5b_fewshot", "tfidf_lr_tt",
             "mbert_tt", "xlmr_tt", "muril_tt", "ensemble_tt", "muril_tt_rom", "muril_robust1", "muril_robust2"]
    names = {"tfidf_lr_en": "TF-IDF + LR", "bert_en": "English BERT (base paper)", "mbert_en": "mBERT",
             "xlmr_en": "XLM-R", "muril_en": "MuRIL", "llm_qwen1.5b_fewshot": "Qwen2.5-1.5B LLM",
             "tfidf_lr_tt": "TF-IDF + LR", "mbert_tt": "mBERT", "xlmr_tt": "XLM-R", "muril_tt": "MuRIL (main)",
             "muril_robust1": "MuRIL robust 1 (+SemEval, comments)",
             "muril_robust2": "MuRIL robust 2 (+SemEval, Kaggle, comments)",
             "ensemble_tt": "Ensemble mBERT+XLM-R+MuRIL", "muril_tt_rom": "MuRIL + romanised"}
    settings = {"en": "English only", "tt": "Translate-train", "llm": "Prompt only", "ens": "Translate-train",
                "rom": "Translate-train + romanised", "rob": "Translate-train + romanised + extra data"}
    for tag in order:
        p = METRICS_DIR / f"{tag}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf8"))["results"]
        kind = ("llm" if tag.startswith("llm") else "ens" if tag.startswith("ensemble") else "rob" if "robust" in tag
                else "rom" if tag.endswith("rom")
                else "en" if tag.endswith("_en") else "tt")
        results.append({"tag": tag, "name": names[tag], "setting": settings[kind],
                        **{lang: r[lang]["macro_f1"] for lang in ("en", "hi", "bn")}})
    en = pd.read_csv(PROCESSED_DIR / "english.csv", keep_default_na=False)
    stats = {
        "splits": {s: int((en["split"] == s).sum()) for s in ("train", "val", "test")},
        "labels": {l: int((en["label"] == l).sum()) for l in LABELS},
        "sources": {s: int((en["source"] == s).sum()) for s in en["source"].unique()},
        "total": len(en),
    }
    figs = [f for f in ("model_comparison.png", "class_distribution.png", "length_distribution.png",
                        "cm_muril_tt_hi.png", "cm_muril_tt_bn.png") if (FIG_DIR / f).exists()]
    return jsonify(models=models, examples=EXAMPLES, results=results, stats=stats, figures=figs,
                   device="GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


@app.post("/api/predict")
def api_predict():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()[:1000]
    name = body.get("model") or default_model()
    if not text:
        return jsonify(error="Please type a comment."), 400
    if name not in MODEL_INFO or not (MODELS_DIR / name).exists():
        return jsonify(error=f"Model {name} not found in models/."), 400

    t0 = time.time()
    steps = clean_text_steps(text)
    shown, prev = [], text
    for step, out in steps:
        shown.append({"step": step, "text": out, "changed": out != prev})
        prev = out
    clean = steps[-1][1]

    predictor = get_predictor(name)
    tokens = predictor.tok.tokenize(clean)
    probs = predictor.predict_proba([text])[0]
    label = int(probs.argmax())

    # word importance: remove one word at a time and see how much the predicted class loses
    words = clean.split()[:60]
    importance = []
    if len(words) > 1:
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        drops = probs[label] - predictor.predict_proba(variants)[:, label]
        importance = [{"word": w, "score": round(float(d), 3)} for w, d in zip(words, drops)]

    return jsonify(
        input=text, steps=shown, clean=clean, script=detect_script(text),
        tokens=tokens[:80], n_tokens=len(tokens),
        probs={l: round(float(p), 4) for l, p in zip(LABELS, probs)}, label=LABELS[label],
        importance=importance, model=name, model_desc=MODEL_INFO[name],
        ms=round((time.time() - t0) * 1000),
    )


@app.post("/api/batch")
def api_batch():
    f = request.files.get("file")
    name = request.form.get("model") or default_model()
    if not f:
        return jsonify(error="Upload a CSV file."), 400
    df = pd.read_csv(f, keep_default_na=False)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id", df.columns[0])
    text_col = cols.get("text", cols.get("comment", df.columns[-1]))
    preds = get_predictor(name).predict(df[text_col].astype(str).tolist())
    out = pd.DataFrame({"ID": df[id_col], "Predicted Label": preds})
    buf = io.BytesIO(out.to_csv(index=False).encode("utf-8-sig"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")


if __name__ == "__main__":
    print("Loading the main model ...")
    get_predictor(default_model())
    print("Open http://127.0.0.1:5000 in your browser")
    if "--open" in sys.argv:  # used by start_demo.bat: open the browser once the model is loaded
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
