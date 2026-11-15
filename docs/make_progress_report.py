"""Builds docs/Progress_Report_Round2.docx (two pages) from the files in results/.

Usage:  python docs/make_progress_report.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import LABELS, METRICS_DIR, PROCESSED_DIR  # noqa: E402

OUT = ROOT / "docs" / "Progress_Report_Round2.docx"

# experiments shown in the results table, in reading order
ROWS = [
    ("tfidf_lr_en", "TF-IDF + Logistic Regression", "English only"),
    ("bert_en", "English BERT (base paper setting)", "English only"),
    ("mbert_en", "mBERT", "English only (zero-shot)"),
    ("xlmr_en", "XLM-R", "English only (zero-shot)"),
    ("muril_en", "MuRIL", "English only (zero-shot)"),
    ("llm_qwen1.5b_fewshot", "Qwen2.5-1.5B LLM, few-shot prompt", "No training"),
    ("tfidf_lr_tt", "TF-IDF + Logistic Regression", "Translate-train"),
    ("mbert_tt", "mBERT", "Translate-train"),
    ("xlmr_tt", "XLM-R", "Translate-train"),
    ("muril_tt", "MuRIL (mean pooling)", "Translate-train"),
    ("ensemble_tt", "Ensemble (mBERT + XLM-R + MuRIL)", "Translate-train"),
    ("ensemble2_tt", "Ensemble (XLM-R + MuRIL)", "Translate-train"),
]


def load(tag):
    p = METRICS_DIR / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc, header, rows, widths_cm, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        shade(c, "D9E2F3")
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.bold = k in bold_rows
    for row in t.rows:
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
            for p in row.cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(8.5)
    return t


def heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def para(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    return p


def bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)


def main():
    en = pd.read_csv(PROCESSED_DIR / "english.csv", keep_default_na=False)
    m = {tag: load(tag) for tag, _, _ in ROWS}
    f1 = lambda tag, lang: m[tag]["results"][lang]["macro_f1"] if m.get(tag) else None  # noqa: E731

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.6)
    sec.top_margin = sec.bottom_margin = Cm(1.3)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(9.5)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(0)
    r = t.add_run("Round 2 Progress Report: Detecting Stance in Media on Global Warming for Indic Languages")
    r.bold = True
    r.font.size = Pt(13)
    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.paragraph_format.space_after = Pt(4)
    s.add_run("Varun Sharma (S24CSEU0170) · Yashika Garg (S24CSEU0235) · Priyanshu Raj (S24CSEU1142) · "
              "Advay Sinha (S24CSEU2059)  |  Guide: Dr. Supriya Chanda  |  GitHub: github.com/varun-sharma-2006/Indic-Climate-Changes-").font.size = Pt(8.5)

    # 1. dataset
    heading(doc, "1. Dataset Description and Statistics")
    para(doc, "The course dataset has not been released, so we use two public English datasets labelled for stance "
              "towards our exact claim, and machine-translate them into Hindi and Bengali. GWSD comes from our base paper, "
              "Luo et al. (2020): 2,050 news sentences, each labelled by 8 annotators "
              "(github.com/yiweiluo/GWStance). ClimateStance (Vaid et al., 2022) has 3,580 tweets "
              "(huggingface.co/datasets/eci-io/climate-evaluation). Labels were mapped to Favour / Against / None. "
              "Every English example was translated with NLLB-200 (600M), giving three parallel versions (En, Hi, Bn) "
              "with identical splits, so a sentence and its translations never fall into different splits.")
    split_rows = []
    for split in ["train", "val", "test"]:
        g = en[en["split"] == split]
        c = g["label"].value_counts()
        split_rows.append([split, len(g), *[c.get(l, 0) for l in LABELS], f"{len(g) * 3:,}"])
    tot = en["label"].value_counts()
    split_rows.append(["Total", len(en), *[tot[l] for l in LABELS], f"{len(en) * 3:,}"])
    add_table(doc, ["Split", "English", "Favour", "Against", "None", "En+Hi+Bn"], split_rows,
              [2.2, 2.2, 2.2, 2.2, 2.2, 2.4], bold_rows=(3,))
    para(doc, f"The classes are imbalanced: Favour {100 * tot['Favour'] / len(en):.0f}%, None "
              f"{100 * tot['None'] / len(en):.0f}%, Against only {100 * tot['Against'] / len(en):.0f}%. "
              "We therefore use Macro F1 as the main metric and a class-weighted loss. Average length is 25 words "
              "(English), 28 (Hindi) and 21 (Bengali). The GWSD test set is the authors' own 200-sentence held-out set; "
              "the ClimateStance test set is its official split.").paragraph_format.space_before = Pt(3)

    # 2. method
    heading(doc, "2. Current Methodology and Implementation")
    bullet(doc, "HTML/Unicode normalisation, URL removal, @mentions → @user, hashtag splitting "
                "(#ClimateHoax → climate hoax), emojis → words (😂 → face with tears of joy), repeated letters shortened.",
           "Preprocessing: ")
    bullet(doc, "English → Hindi/Bengali with NLLB-200. Outputs stuck in repetition loops were detected "
                "automatically and re-translated with repetition blocking.", "Translation (data augmentation): ")
    bullet(doc, "TF-IDF (word 1–2 and character 2–5 grams) with Logistic Regression and Linear SVM; English BERT, "
                "the base paper's model; multilingual mBERT, XLM-R and MuRIL; Qwen2.5-1.5B-Instruct "
                "LLM with a few-shot prompt; probability-averaging ensemble.", "Models: ")
    bullet(doc, "(a) English only, as in the base paper / zero-shot transfer; (b) translate-train on En+Hi+Bn (ours).",
           "Two training settings: ")
    bullet(doc, "AdamW, lr 3e-5, batch 16, max length 128, fp16, class-weighted cross-entropy, best epoch chosen "
                "by validation Macro F1, frozen word embeddings (4 GB GPU). Hardware: NVIDIA RTX 2050 laptop GPU.",
           "Training: ")
    bullet(doc, "LIME word-importance explanations and a per-dataset / per-class error analysis.", "Explainability: ")

    # 3. results
    heading(doc, "3. Experimental Results and Performance Analysis")
    res_rows, bold = [], []
    for tag, name, setting in ROWS:
        if not m.get(tag):
            continue
        r = m[tag]["results"]
        res_rows.append([name, setting, *[f"{r[l]['macro_f1']:.3f}" for l in ("en", "hi", "bn")],
                         f"{r['hi']['accuracy']:.3f}", f"{r['bn']['accuracy']:.3f}"])
        if tag == "muril_tt":
            bold.append(len(res_rows) - 1)
    add_table(doc, ["Model", "Training data", "F1 En", "F1 Hi", "F1 Bn", "Acc Hi", "Acc Bn"], res_rows,
              [5.2, 3.6, 1.5, 1.5, 1.5, 1.5, 1.5], bold_rows=tuple(bold))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(3)
    cr = cap.add_run("Macro F1 and accuracy on the 555-example test set of each language (Hi/Bn = translated test sets).")
    cr.italic = True
    cr.font.size = Pt(8)

    for lead, text in ANALYSIS(m, f1):
        bullet(doc, text, lead)
    rom = ROOT / "results" / "metrics_extra" / "muril_tt_hr_br.json"
    if rom.exists() and (METRICS_DIR / "muril_tt_rom.json").exists():
        before = json.loads(rom.read_text(encoding="utf8"))["results"]
        after = load("muril_tt_rom")["results"]
        bullet(doc, f"MuRIL scores {before['hr']['macro_f1']:.2f} / {before['br']['macro_f1']:.2f} on romanised Hindi/Bengali "
                    "(XLM-R collapses to ~0.2). Two more epochs with romanised copies of the training data raise this to "
                    f"{after['hr']['macro_f1']:.2f} / {after['br']['macro_f1']:.2f}, at a small cost on native script "
                    f"({after['hi']['macro_f1']:.3f} / {after['bn']['macro_f1']:.3f}).", "Romanised text (augmentation): ")
    ch_path = ROOT / "results" / "challenge_set_metrics.json"
    if ch_path.exists():
        ch = json.loads(ch_path.read_text(encoding="utf8"))
        bullet(doc, "On 64 hand-written Hindi/Bengali YouTube-style comments (not translations), XLM-R reaches "
                    f"macro F1 {ch['xlmr_tt']['macro_f1']:.2f} and MuRIL {ch['muril_tt']['macro_f1']:.2f}. Explicit "
                    "statements are always correct, but sarcasm (38% accuracy) and romanized Hindi/Bengali (20%) are "
                    "the main weaknesses.", "Hand-written challenge set: ")

    fig = ROOT / "results" / "figures" / "model_comparison.png"
    if fig.exists():
        doc.add_picture(str(fig), width=Cm(9.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 4. challenges
    heading(doc, "4. Challenges Encountered")
    for lead, text in [
        ("No Hindi/Bengali labelled data: ", "solved for now with machine translation; the translated test sets are "
         "cleaner than real YouTube comments, so scores on real comments are expected to be lower."),
        ("MuRIL did not learn: ", "its [CLS] vector was almost identical for all sentences (cosine similarity ≈ 1.0) "
         "and the model predicted only Favour (F1 0.26). Replacing [CLS] with mean pooling fixed it."),
        ("Translation errors: ", "about 0.4% of NLLB outputs got stuck repeating one word; we detect and re-translate them."),
        ("Limited GPU (4 GB): ", "XLM-R and MuRIL did not fit for full fine-tuning; we freeze the word-embedding "
         "layer and use mixed precision."),
        ("Class imbalance: ", "only 13% Against; class-weighted loss and Macro F1 model selection."),
        ("Label mismatch: ", "ClimateStance's 'ambiguous' class (which also covers sarcastic tweets) maps poorly to our "
         "None: its F1 is below 0.15 in every language, while GWSD's neutral class reaches about 0.70-0.74 F1 "
         "(results/error_analysis_muril_tt.md). The most common errors are Favour <-> None (55% of all errors)."),
    ]:
        bullet(doc, text, lead)

    # 5. future work
    heading(doc, "5. Future Work Before the Final Evaluation")
    for lead, text in [
        ("Real test data: ", "collect Hindi/Bengali YouTube comments with our script (YouTube Data API) and label "
         "~500 per language with two annotators (Cohen's kappa); evaluate on the course test set when released."),
        ("Romanized text: ", "transliterate Roman-script Hindi/Bengali (e.g. 'garmi bahut hai') to native script with IndicXlit."),
        ("Better LLM: ", "LoRA fine-tuning of a multilingual LLM on the translate-train data (Kaggle/Colab GPU)."),
        ("Ensemble and analysis: ", "weighted/stacked ensemble tuned on validation data; full error analysis with "
         "sarcasm, off-topic and code-mixed categories; LIME examples in the final report."),
        ("Submission: ", "predict.py already writes the required 'ID, Predicted Label' file."),
    ]:
        bullet(doc, text, lead)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    rr = p.add_run("References: [1] Luo, Card, Jurafsky. Detecting Stance in Media on Global Warming. Findings of EMNLP 2020, "
                   "doi.org/10.18653/v1/2020.findings-emnlp.296. [2] Vaid, Pant, Shrivastava. Towards Fine-grained "
                   "Classification of Climate Change related Social Media Text. ACL SRW 2022, doi.org/10.18653/v1/2022.acl-srw.35. "
                   "[3] Khanuja et al. MuRIL, arXiv:2103.10730. [4] Conneau et al. XLM-R, arXiv:1911.02116. "
                   "[5] NLLB Team. No Language Left Behind, arXiv:2207.04672.")
    rr.font.size = Pt(7.5)

    doc.save(OUT)
    print(f"Saved -> {OUT}")


def ANALYSIS(m, f1):
    """Findings written from the numbers in results/metrics."""
    out = []
    if m.get("bert_en"):
        out.append(("Base paper reproduced: ", f"English BERT trained on English reaches "
                    f"{m['bert_en']['results']['en']['accuracy'] * 100:.1f}% accuracy on English, matching the ~75% in "
                    f"Luo et al., but only F1 {f1('bert_en', 'hi'):.2f} (Hindi) and {f1('bert_en', 'bn'):.2f} (Bengali): "
                    "the base paper's approach does not work for Indian languages."))
    if m.get("tfidf_lr_en") and m.get("tfidf_lr_tt"):
        out.append(("Translation is essential: ", f"TF-IDF goes from F1 {f1('tfidf_lr_en', 'hi'):.2f} to "
                    f"{f1('tfidf_lr_tt', 'hi'):.2f} on Hindi and from {f1('tfidf_lr_en', 'bn'):.2f} to "
                    f"{f1('tfidf_lr_tt', 'bn'):.2f} on Bengali when trained on translated data."))
    zs = [t for t in ("mbert", "xlmr", "muril") if m.get(f"{t}_en") and m.get(f"{t}_tt")]
    if zs:
        gains = [(f1(f"{t}_tt", l) - f1(f"{t}_en", l)) for t in zs for l in ("hi", "bn")]
        out.append(("Zero-shot vs translate-train: ", "multilingual models transfer some knowledge from English alone "
                    f"(XLM-R best zero-shot), and translate-train changes Indic F1 by {min(gains):+.2f} to "
                    f"{max(gains):+.2f} depending on the model."))
    best = max((t for t, _, _ in ROWS if m.get(t)), key=lambda t: (f1(t, "hi") + f1(t, "bn")) / 2)
    names = {t: n for t, n, _ in ROWS}
    out.append(("Best system: ", f"{names[best]} (translate-train) with Macro F1 {f1(best, 'hi'):.3f} (Hindi) and {f1(best, 'bn'):.3f} "
                f"(Bengali), against {f1('bert_en', 'hi'):.3f} / {f1('bert_en', 'bn'):.3f} for the base paper's model."))
    return out


if __name__ == "__main__":
    main()
