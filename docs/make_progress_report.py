"""Builds docs/Progress_Report_Round2.docx (two pages) from the files in results/.

Usage:  python docs/make_progress_report.py
"""
import json
import re
import statistics
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
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x5B, 0x66)
REPORT_DATE = "19 November 2026"

# experiments shown in the results table, in reading order
ROWS = [
    ("tfidf_lr_en", "TF-IDF + Logistic Regression", "English only"),
    ("bert_en", "English BERT (base paper setting)", "English only"),
    ("mbert_en", "mBERT", "English only (zero-shot)"),
    ("xlmr_en", "XLM-R", "English only (zero-shot)"),
    ("muril_en", "MuRIL", "English only (zero-shot)"),
    ("llm_qwen1.5b_fewshot", "Qwen2.5-1.5B LLM (few-shot)", "No training"),
    ("tfidf_lr_tt", "TF-IDF + Logistic Regression", "Translate-train"),
    ("mbert_tt", "mBERT", "Translate-train"),
    ("xlmr_tt", "XLM-R", "Translate-train"),
    ("muril_tt", "MuRIL (mean pooling)", "Translate-train"),
    ("ensemble_tt", "Ensemble (mBERT + XLM-R + MuRIL)", "Translate-train"),
    ("ensemble2_tt", "Ensemble (XLM-R + MuRIL)", "Translate-train"),
    ("muril_robust2", "MuRIL Robust (final model)", "+ romanised, SemEval, Kaggle, comments"),
]

# robustness on data never used for training (results/robustness_report.md, results/challenge_set_metrics.json,
# results/metrics/*.json for the romanised test sets)
ROBUST_ROWS = [
    ("Hand-written challenge set, 64 comments (Macro F1)", "0.658", "0.773"),
    ("  Sarcasm / negation / implicit (correct)", "0/8 · 5/8 · 8/13", "5/8 · 7/8 · 10/13"),
    ("Evaluator-style sanity sentences (correct)", "16/18", "17/18"),
    ("Kaggle held-out tweets, Hi / Bn (Macro F1)", "0.479 / 0.460", "0.663 / 0.649"),
    ("SemEval-2016 test tweets, Hi / Bn (Macro F1)", "0.458 / 0.425", "0.551 / 0.546"),
    ("Romanised test sets, Hi / Bn (Macro F1)", "0.521 / 0.487", "0.617 / 0.614"),
    ("Formal translated test sets, Hi / Bn (Macro F1)", "0.662 / 0.650", "0.631 / 0.647"),
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


def set_cell_margins(table, cm=0.08):
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side in ("top", "bottom"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(int(cm * 567)))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def add_table(doc, header, rows, widths_cm, bold_rows=(), size=8, center_from=1, highlight_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_margins(t, 0.02)
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "1F3A5F")
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.bold = k in bold_rows
            if k in highlight_rows:
                shade(cells[i], "E3ECF7")
            elif k % 2 == 1:
                shade(cells[i], "F5F7FA")
    for n, row in enumerate(t.rows):
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
            for p in row.cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.keep_with_next = n < len(t.rows) - 1  # keep the table on one page
                if i >= center_from:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(size)
    return t


def caption(doc, text):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(1)
    cap.paragraph_format.keep_with_next = True
    cr = cap.add_run(text)
    cr.italic = True
    cr.font.size = Pt(7.5)
    cr.font.color.rgb = GREY


def heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = NAVY
    # thin rule under the heading
    p_pr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "1"), ("w:color", "9DB1CC")):
        bottom.set(qn(k), v)
    bdr.append(bottom)
    p_pr.append(bdr)


def para(doc, text, bold_lead=None, size=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text)
    if size:
        r.font.size = Pt(size)
    return p


def bullet(doc, text, bold_lead=None, gap=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    if gap:
        p.paragraph_format.space_before = Pt(gap)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)


def summary_box(doc, lines):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    c.width = Cm(17.8)
    shade(c, "EEF3F9")
    c.text = ""
    first = True
    for lead, text in lines:
        p = c.paragraphs[0] if first else c.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(1)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(lead)
        r.bold = True
        r.font.color.rgb = NAVY
        r.font.size = Pt(9)
        r2 = p.add_run(text)
        r2.font.size = Pt(9)


def seed_stats():
    tags = [t for t in ("muril_tt", "muril_tt_seed13", "muril_tt_seed7") if load(t)]
    out = {}
    for lang in ("hi", "bn"):
        vals = [load(t)["results"][lang]["macro_f1"] for t in tags]
        out[lang] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return len(tags), out


def main():
    en = pd.read_csv(PROCESSED_DIR / "english.csv", keep_default_na=False)
    m = {tag: load(tag) for tag, _, _ in ROWS}
    f1 = lambda tag, lang: m[tag]["results"][lang]["macro_f1"] if m.get(tag) else None  # noqa: E731
    n_seeds, seeds = seed_stats()
    n_tests = sum(len(re.findall(r"^def test_", p.read_text(encoding="utf8"), re.M))
                  for p in (ROOT / "tests").glob("test_*.py"))

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.6)
    sec.top_margin = sec.bottom_margin = Cm(1.2)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(9.5)

    # ---- title block
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(0)
    r = t.add_run("Detecting Stance in Media on Global Warming for Indic Languages")
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = NAVY
    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.paragraph_format.space_after = Pt(0)
    r = s.add_run("Round 2 Progress Report")
    r.bold = True
    r.font.size = Pt(10.5)
    for line in ["Varun Sharma (S24CSEU0170) · Yashika Garg (S24CSEU0235) · Priyanshu Raj (S24CSEU1142) · Advay Sinha (S24CSEU2059)",
                 f"Guide: Dr. Supriya Chanda  ·  Bennett University  ·  {REPORT_DATE}  ·  "
                 "Code: github.com/varun-sharma-2006/Indic-Climate-Changes-"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        rr = p.add_run(line)
        rr.font.size = Pt(8.5)
        rr.font.color.rgb = GREY
    doc.paragraphs[-1].paragraph_format.space_after = Pt(5)

    summary_box(doc, [
        ("Task. ", "Classify Hindi and Bengali YouTube comments as Favour / Against / None towards the claim "
                   "“Climate change and global warming is a serious concern”; metric: Macro F1."),
        ("Approach. ", "Extend the English-only base paper (Luo et al., 2020) to Indian languages by translating "
                       "two English stance datasets with NLLB-200 and fine-tuning multilingual transformers (translate-train)."),
        ("Results so far. ", f"MuRIL raises Hindi / Bengali Macro F1 from {f1('bert_en', 'hi'):.2f} / {f1('bert_en', 'bn'):.2f} "
                             f"(base-paper BERT) to {f1('muril_tt', 'hi'):.2f} / {f1('muril_tt', 'bn'):.2f} "
                             f"(stable over {n_seeds} random seeds). Our final robust model reaches 0.77 on hand-written "
                             "YouTube-style comments and runs in a live web demo."),
    ])

    # ---- 1. dataset
    heading(doc, "1. Dataset Description and Statistics")
    para(doc, "No labelled Hindi/Bengali climate-stance data is publicly available and the course dataset has not been "
              "released yet. We therefore use two public English datasets labelled for stance towards our exact claim, "
              "map their labels to Favour / Against / None, and translate every example into Hindi and Bengali with "
              "NLLB-200 (600M), plus romanised copies (e.g. “global warming bahut gambhir hai”). A sentence and its "
              "translations share one id and one split, so there is no train/test leakage (checked by unit tests).")
    caption(doc, "Table 1. Datasets. The first two form the main benchmark; the others make the final model robust to comment-style text.")
    add_table(doc, ["Dataset", "Source", "Text type", "Size", "Used for"], [
        ["GWSD (base paper)", "Luo et al. 2020, github.com/yiweiluo/GWStance", "News", "2,050", "Train / val / test"],
        ["ClimateStance", "Vaid et al. 2022, HF eci-io/climate-evaluation", "Tweets", "3,580", "Train / val / test"],
        ["SemEval-2016 Task 6 (climate)", "HF cardiffnlp/tweet_eval", "Tweets", "564", "Train 395 / test 169"],
        ["Kaggle Climate Change Tweets", "Kaggle edqian, balanced sample", "Tweets", "6,000", "Train 5,400 / test 600"],
        ["YouTube-style comments", "Ours, written with an AI assistant", "Comments", "184", "Training only"],
        ["Challenge set", "Ours, hand-written", "Comments", "64", "Evaluation only"],
    ], [4.4, 6.4, 1.5, 1.4, 3.6], center_from=2, size=8)

    split_rows = []
    for split in ["train", "val", "test"]:
        g = en[en["split"] == split]
        c = g["label"].value_counts()
        split_rows.append([split.capitalize() if split != "val" else "Validation", f"{len(g):,}",
                           *[f"{c.get(l, 0):,}" for l in LABELS], f"{len(g) * 3:,}"])
    tot = en["label"].value_counts()
    split_rows.append(["Total", f"{len(en):,}", *[f"{tot[l]:,}" for l in LABELS], f"{len(en) * 3:,}"])
    caption(doc, f"Table 2. Main benchmark (GWSD + ClimateStance) per split; identical in every language. "
                 f"Classes are imbalanced: Favour {100 * tot['Favour'] / len(en):.0f}%, None {100 * tot['None'] / len(en):.0f}%, "
                 f"Against {100 * tot['Against'] / len(en):.0f}%, so we use Macro F1 and a class-weighted loss.")
    add_table(doc, ["Split", "English", "Favour", "Against", "None", "En + Hi + Bn"], split_rows,
              [2.6, 2.3, 2.3, 2.3, 2.3, 2.6], bold_rows=(3,), size=8)

    # ---- 2. method
    heading(doc, "2. Methodology and Implementation")
    para(doc, "Pipeline: preprocessing → translation and romanisation → fine-tuning → evaluation → "
              "explanation → prediction file / web demo.", "")
    bullet(doc, "Unicode/HTML normalisation, URL removal, @mentions → @user, hashtag splitting "
                "(#ClimateHoax → climate hoax), emojis → words (they often carry the stance), repeated letters shortened.",
           "Preprocessing: ")
    bullet(doc, "TF-IDF (word 1–2 and character 2–5 grams) with Logistic Regression and Linear SVM; English BERT "
                "(base paper); multilingual mBERT, XLM-R and MuRIL; Qwen2.5-1.5B-Instruct with a few-shot prompt; "
                "probability-averaging ensembles.", "Models: ")
    bullet(doc, "(a) English only, as in the base paper (zero-shot transfer to Hindi/Bengali); (b) translate-train on "
                "English + Hindi + Bengali (ours); (c) continued training on romanised and comment-style data (final model).",
           "Training settings: ")
    bullet(doc, "AdamW, learning rate 3e-5, batch 16, max length 128, fp16, class-weighted cross-entropy, best epoch by "
                "validation Macro F1, frozen word embeddings to fit a 4 GB NVIDIA RTX 2050 laptop GPU.", "Training details: ")
    bullet(doc, "LIME word importance, occlusion-based word importance in the demo, and error analysis by language, "
                "source dataset and class.", "Explainability: ")

    # ---- 3. results
    heading(doc, "3. Experimental Results and Performance Analysis")
    res_rows, bold, hl = [], [], []
    for tag, name, setting in ROWS:
        if not m.get(tag):
            continue
        r = m[tag]["results"]
        res_rows.append([name, setting, *[f"{r[l]['macro_f1']:.3f}" for l in ("en", "hi", "bn")],
                         f"{r['hi']['accuracy']:.3f}", f"{r['bn']['accuracy']:.3f}"])
        if tag == "muril_robust2":
            bold.append(len(res_rows) - 1)
        if tag in ("bert_en", "muril_tt", "muril_robust2"):
            hl.append(len(res_rows) - 1)
    caption(doc, "Table 3. Macro F1 and accuracy on the 555-example test set of each language. "
                 "Shaded: base-paper setting, best translate-train model, final model.")
    add_table(doc, ["Model", "Training data", "F1 En", "F1 Hi", "F1 Bn", "Acc Hi", "Acc Bn"], res_rows,
              [4.6, 5.4, 1.55, 1.55, 1.55, 1.55, 1.55], bold_rows=tuple(bold), highlight_rows=tuple(hl), center_from=2)

    for lead, text in ANALYSIS(m, f1):
        bullet(doc, text, lead)
    bullet(doc, f"re-training MuRIL with {n_seeds} random seeds gives Macro F1 {seeds['hi'][0]:.3f} ± {seeds['hi'][1]:.3f} "
                f"(Hindi) and {seeds['bn'][0]:.3f} ± {seeds['bn'][1]:.3f} (Bengali); the ranking of models does not change.",
           "Stability: ")

    caption(doc, "Table 4. Robustness. Continued training on SemEval-2016, Kaggle climate tweets, romanised text and "
                 "184 comments improves every comment-style test, at a 0.03 cost on the formal Hindi test set.")
    add_table(doc, ["Evaluation on data never used for training", "MuRIL (translate-train)", "MuRIL Robust (final)"],
              [list(r) for r in ROBUST_ROWS], [8.6, 4.4, 4.4], size=8, bold_rows=(0,))
    bullet(doc, "much better on informal, comment-style text, which is our actual task; sarcasm in romanised "
                "Hindi/Bengali remains the main weakness, and 55–57% of all errors are Favour ↔ None.", "Final model: ", gap=4)

    # ---- 4. deliverables
    heading(doc, "4. Implementation Status and Deliverables")
    bullet(doc, "complete, documented code in src/ (data download, translation, training, evaluation, explanation), "
                "a README with datasets, results and run instructions, and a one-command pipeline (run_all.ps1).",
           "GitHub repository: ")
    bullet(doc, "a web demo shows each step for any typed comment (cleaning → tokens → probabilities → "
                "important words) and classifies a whole CSV; predict.py writes the required ‘ID, Predicted Label’ file.",
           "Executable demo and submission: ")
    bullet(doc, f"{n_tests} automated unit tests (preprocessing, metrics, data leakage, mean pooling), all passing; "
                "all reported numbers are regenerated by the scripts and stored in results/.", "Testing and reproducibility: ")

    # ---- 5. challenges
    heading(doc, "5. Challenges Encountered and Solutions")
    for lead, text in [
        ("No Hindi/Bengali labelled data: ", "machine translation of English datasets; evaluation also on hand-written comments."),
        ("MuRIL did not learn: ", "its [CLS] vectors were nearly identical for all sentences (loss stuck at ln 3, "
         "Macro F1 0.26); mean pooling over all tokens fixed it."),
        ("Translation loops and limited GPU: ", "about 0.4% of NLLB outputs repeated one word and were re-translated "
         "automatically; frozen embeddings and fp16 made XLM-R and MuRIL fit into 4 GB."),
        ("Class imbalance and label mismatch: ", "only 13% Against (class-weighted loss); ClimateStance’s "
         "‘ambiguous’ class maps poorly to None (F1 below 0.4)."),
    ]:
        bullet(doc, text, lead)

    # ---- 6. future work
    heading(doc, "6. Future Work Before the Final Evaluation")
    for lead, text in [
        ("Real data: ", "collect and label Hindi/Bengali YouTube comments with two annotators (Cohen’s kappa); "
         "evaluate on the course test set as soon as it is released."),
        ("Sarcasm and romanised text: ", "targeted examples and transliteration to native script (IndicXlit)."),
        ("Stronger models: ", "LoRA fine-tuning of a multilingual LLM; ensemble weights tuned on validation data."),
    ]:
        bullet(doc, text, lead)

    p = para(doc, "We used an AI coding assistant (Claude, Anthropic) to help write and debug code, prepare documents "
                  "and draft the 184 synthetic training comments. The team designed the approach, ran every experiment "
                  "and verified all results.", "Use of AI tools: ", size=8.5)
    p.paragraph_format.space_before = Pt(4)
    p.runs[0].font.size = Pt(8.5)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    rr = p.add_run("References: [1] Luo, Card, Jurafsky. Detecting Stance in Media on Global Warming. Findings of EMNLP 2020. "
                   "[2] Vaid, Pant, Shrivastava. Towards Fine-grained Classification of Climate Change related Social Media "
                   "Text. ACL SRW 2022. [3] Mohammad et al. SemEval-2016 Task 6: Detecting Stance in Tweets. SemEval 2016. "
                   "[4] Khanuja et al. MuRIL. arXiv:2103.10730. [5] Conneau et al. XLM-R. ACL 2020. "
                   "[6] NLLB Team. No Language Left Behind. arXiv:2207.04672. [7] Ribeiro et al. LIME. KDD 2016.")
    rr.font.size = Pt(7.5)
    rr.font.color.rgb = GREY

    doc.save(OUT)
    print(f"Saved -> {OUT}")


def ANALYSIS(m, f1):
    """Findings written from the numbers in results/metrics."""
    out = []
    if m.get("bert_en"):
        out.append(("Base paper reproduced: ", f"English BERT reaches {m['bert_en']['results']['en']['accuracy'] * 100:.1f}% "
                    f"accuracy on English, matching the ~75% of Luo et al., but only Macro F1 {f1('bert_en', 'hi'):.2f} (Hindi) "
                    f"and {f1('bert_en', 'bn'):.2f} (Bengali): the base paper’s approach does not transfer to Indian languages."))
    if m.get("tfidf_lr_en") and m.get("tfidf_lr_tt"):
        out.append(("Translation is essential: ", f"TF-IDF improves from {f1('tfidf_lr_en', 'hi'):.2f} to "
                    f"{f1('tfidf_lr_tt', 'hi'):.2f} (Hindi) and from {f1('tfidf_lr_en', 'bn'):.2f} to "
                    f"{f1('tfidf_lr_tt', 'bn'):.2f} (Bengali) when trained on translated data."))
    zs = [t for t in ("mbert", "xlmr", "muril") if m.get(f"{t}_en") and m.get(f"{t}_tt")]
    if zs:
        gains = [(f1(f"{t}_tt", l) - f1(f"{t}_en", l)) for t in zs for l in ("hi", "bn")]
        out.append(("Translate-train beats zero-shot: ", "for every multilingual model, by "
                    f"{min(gains):+.2f} to {max(gains):+.2f} Macro F1. MuRIL, pre-trained on Indian languages, is the best "
                    f"single model ({f1('muril_tt', 'hi'):.3f} / {f1('muril_tt', 'bn'):.3f}); the ensemble helps Hindi only, "
                    f"and the untrained LLM is far behind ({f1('llm_qwen1.5b_fewshot', 'hi'):.2f})."))
    return out


if __name__ == "__main__":
    main()
