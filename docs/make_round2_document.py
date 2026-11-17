"""Builds docs/Round2_Complete_Documentation.docx: one Word file with everything for Round 2.

Part A  Round 2 checklist (each evaluation criterion -> status -> evidence)
Part B  Full project explanation (converted from docs/PROJECT_GUIDE.md, with figures)
Part C  Individual contributions: what each member built, explains and still has to do
Part D  Evaluation-day plan: speaking order, demo commands, backups

Usage:  python docs/make_round2_document.py
"""
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import METRICS_DIR  # noqa: E402

OUT = ROOT / "docs" / "Round2_Complete_Documentation.docx"
FIG = ROOT / "results" / "figures"
REPO = "https://github.com/varun-sharma-2006/Indic-Climate-Changes-"
BLUE = RGBColor(0x1F, 0x3A, 0x5F)

# figures placed at the end of these guide sections
FIGURES = {
    "### 4.3": [("class_distribution.png", 12, "Class distribution in the two English datasets")],
    "### 4.4": [("length_distribution.png", 11, "Text length by language after cleaning")],
    "## 7.": [("model_comparison.png", 12, "Macro F1 of every experiment on the English, Hindi and Bengali test sets"),
              (("cm_muril_tt_hi.png", "cm_muril_tt_bn.png"), 7.5,
               "Confusion matrices of our main model (MuRIL translate-train) on Hindi (left) and Bengali (right)")],
}


# ---------------------------------------------------------------- helpers
def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_hyperlink(paragraph, text, url, size=None):
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1F5FBF")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(color)
    rpr.append(underline)
    if size:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int(size * 2)))
        rpr.append(sz)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|https?://[^\s|)]+|\*[^*\s][^*]*\*)")


def add_inline(paragraph, text, size=None, bold=False):
    """Writes markdown-style inline text: **bold**, *italic*, `code`, [links](url), bare URLs."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            add_inline(paragraph, part[2:-2], size, bold=True)
            continue
        if part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt((size or 10.5) - 1)
            r.bold = bold
            continue
        m = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part)
        if m:
            url = m.group(2)
            if not url.startswith("http"):
                url = f"{REPO}/blob/main/" + url.replace("../", "").lstrip("./")
            add_hyperlink(paragraph, m.group(1), url, size)
            continue
        if part.startswith("http"):
            add_hyperlink(paragraph, part, part, size)
            continue
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            r.italic = True
        else:
            r = paragraph.add_run(part)
        r.bold = bold
        if size:
            r.font.size = Pt(size)


def table(doc, rows, widths=None, header=True, size=9):
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j, val in enumerate(row):
            p = cells[j].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            add_inline(p, str(val).strip(), size=size, bold=header and i == 0)
            if header and i == 0:
                shade(cells[j], "D9E2F3")
            if widths:
                cells[j].width = Cm(widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def para(doc, text="", size=None, style=None, bold=False, align=None, after=4):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if align:
        p.alignment = align
    add_inline(p, text, size=size, bold=bold)
    return p


def bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(1)
    add_inline(p, text)
    return p


def numbered(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    p.paragraph_format.space_after = Pt(2)
    p.add_run(f"{num}. ")
    add_inline(p, text)


def code_block(doc, lines):
    for ln in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(ln if ln else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x22, 0x55, 0x22) if ln.strip().startswith("#") else RGBColor(0x20, 0x20, 0x20)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(doc, spec, width_cm, caption):
    if isinstance(spec, tuple):  # two images side by side
        files = [FIG / s for s in spec]
        if not all(f.exists() for f in files):
            return
        t = doc.add_table(rows=1, cols=len(files))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, f in zip(t.rows[0].cells, files):
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].add_run().add_picture(str(f), width=Cm(width_cm))
    else:
        f = FIG / spec
        if not f.exists():
            return
        doc.add_picture(str(f), width=Cm(width_cm))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = para(doc, caption, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
    for r in c.runs:
        r.italic = True


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- markdown guide -> docx
def render_markdown(doc, md_text):
    lines = md_text.splitlines()
    i, current, pending = 0, None, []
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            code_block(doc, block)
            i += 1
            continue
        m = re.match(r"^(#{1,3}) (.*)", line)
        if m:
            for spec in pending:
                figure(doc, *spec)
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:  # the guide's title is replaced by our own part heading
                i += 1
                continue
            h = doc.add_heading(level=level)
            add_inline(h, re.sub(r"`", "", text))
            pending = next((v for k, v in FIGURES.items() if line.startswith(k)), [])
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in cells):
                    rows.append(cells)
                i += 1
            table(doc, rows)
            continue
        if s in ("---", ""):
            i += 1
            continue
        if s.startswith(">"):
            body = s.lstrip("> ").strip()
            if body.startswith("- "):
                bullet(doc, body[2:])
            elif body:
                p = para(doc, body)
                p.paragraph_format.left_indent = Cm(0.6)
            i += 1
            continue
        if re.match(r"^\s*- ", line):
            bullet(doc, s[2:], level=1 if line.startswith("  ") else 0)
            i += 1
            continue
        m = re.match(r"^(\d+)\. (.*)", s)
        if m:
            numbered(doc, m.group(1), m.group(2))
            i += 1
            continue
        para(doc, s)
        i += 1
    for spec in pending:
        figure(doc, *spec)


# ---------------------------------------------------------------- parts written here
def f1(tag, lang):
    p = METRICS_DIR / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf8"))["results"][lang]["macro_f1"] if p.exists() else float("nan")


def part_a(doc):
    doc.add_heading("Part A. Round 2 Checklist", level=1)
    para(doc, "Each requirement from the Round 2 evaluation criteria, what we have, and where the evidence is. "
              "Code was re-run on 17 November 2026: baselines, live predictions and the submission file all work.")
    table(doc, [
        ["Requirement", "Status", "Evidence / where to show it", "Presented by"],
        ["**Dataset**: which, why, statistics, class distribution", "Done",
         "Part B §4; `results/dataset_stats.md`; `results/figures/class_distribution.png`", "Varun"],
        ["**Methodology**: models, algorithms, workflow", "Done", "Part B §5–6; README §3", "Priyanshu"],
        ["**Implementation**: running code + performance metrics", "Done",
         "`src/` (18 scripts), `run_all.ps1`; live demo with `src/predict.py`; `results/results_table.md`", "Yashika, Advay"],
        ["**GitHub**: complete code, proper organisation", "Done", REPO, "Varun"],
        ["**GitHub**: contributions from **all** members", "**TO DO**",
         "Currently all commits are by Varun. Each member must push their own commit (Part C)", "Everyone"],
        ["**Individual contributions** clearly explained", "Done (document)", "`CONTRIBUTIONS.md`; Part C of this file", "Everyone"],
        ["**Every member understands the project**", "**TO DO**",
         "Each member reads Part B and rehearses the questions in §9 and Part C", "Everyone"],
        ["**Two-page progress report**", "Done",
         "`docs/Progress_Report_Round2.docx` / `.pdf` (2 pages, checked in Word). **Print and bring a copy.**", "Advay"],
        ["Code is **executable** and demonstrated", "Done", "Part D demo script (tested sentences)", "Advay"],
        ["**All members participate**", "Planned", "Part D speaking plan (about 2–3 min each)", "Everyone"],
        ["**Slot booking** and reporting on time", "**TO DO**", "Book the slot; bring the laptop (trained models are only on it)", "Varun"],
    ], widths=[4.6, 2.0, 7.6, 2.6])

    doc.add_heading("Headline numbers", level=2)
    table(doc, [
        ["Model", "Macro F1 Hindi", "Macro F1 Bengali"],
        ["Base paper's approach (English BERT, English data)", f"{f1('bert_en', 'hi'):.3f}", f"{f1('bert_en', 'bn'):.3f}"],
        ["**Our main model: MuRIL, translate-train**", f"**{f1('muril_tt', 'hi'):.3f}**", f"**{f1('muril_tt', 'bn'):.3f}**"],
    ], widths=[9, 3.5, 3.5])


MEMBERS = [
    {
        "name": "Varun Sharma (S24CSEU0170)",
        "role": "Data: datasets, label mapping, splits, translation, romanisation, statistics",
        "files": "`src/download_data.py`, `src/translate.py`, `src/romanize_data.py`, `src/data_stats.py`, `src/collect_youtube.py`",
        "built": [
            "Downloaded GWSD (base paper) and ClimateStance from their official sources and mapped both to Favour / Against / None.",
            "Made the splits: GWSD's own 200 held-out sentences + ClimateStance's official test split as test; same split for every language (no leakage).",
            "Translated all 5,630 sentences to Hindi and Bengali with NLLB-200 and fixed ~40 outputs that got stuck repeating one word.",
            "Created romanised Hindi/Bengali copies (e.g. \"global varmimg eka gambhir samasya hai\").",
        ],
        "explain": [
            "Why GWSD: same claim as ours, from the base paper. Why ClimateStance: social-media text (tweets), closer to YouTube comments.",
            "Class distribution: Favour 65%, None 22%, Against 13%. Show `results/figures/class_distribution.png`.",
            "Why translation is needed: no labelled Hindi/Bengali climate-stance data exists.",
        ],
        "questions": [
            "How did you decide ClimateStance label 1 = Favour? (We read example tweets for each number.)",
            "Why does the paper say 3,777 tweets but you have 3,580? (The public files contain 3,580; we use what is released.)",
            "Isn't translated test data unrealistic? (Yes, cleaner than real comments; that is why we use the challenge set and will label real comments.)",
        ],
        "task": "Get a YouTube Data API key, run `python src/collect_youtube.py`, label 100 comments, and commit a short "
                "`data/youtube/annotation_log.md` (how many comments, label counts, examples of hard cases). Do not commit the comments themselves.",
    },
    {
        "name": "Yashika Garg (S24CSEU0235)",
        "role": "Preprocessing, TF-IDF baselines, evaluation metrics, error analysis",
        "files": "`src/preprocess.py`, `src/baselines.py`, `src/data_utils.py`, `src/error_analysis.py`",
        "built": [
            "Cleaning for noisy text: HTML entities, Unicode NFC, URLs removed, @mentions → @user, hashtag splitting, emojis → words, repeated letters shortened.",
            "TF-IDF (word 1–2 grams + character 2–5 grams) with Logistic Regression and Linear SVM.",
            "Metric code: Macro F1, accuracy, macro precision/recall, per-class F1, confusion matrices.",
            "Error analysis by language, source dataset and class.",
        ],
        "explain": [
            f"TF-IDF trained on English only scores {f1('tfidf_lr_en', 'hi'):.2f} on Hindi; with translations "
            f"{f1('tfidf_lr_tt', 'hi'):.2f}. This is the simplest proof that translate-train matters.",
            "Why Macro F1: always predicting Favour gives 65% accuracy but only 0.26 Macro F1.",
            "Error analysis: 55% of errors are Favour ↔ None; ClimateStance's \"ambiguous\" class is the hardest.",
        ],
        "questions": [
            "Why character n-grams? (They cope with spelling variants and work for any script.)",
            "Why turn emojis into words instead of deleting them? (Emojis often carry the stance, e.g. 😂 on a denial comment.)",
            "What is the difference between precision and recall?",
        ],
        "task": "Add `tests/test_preprocess.py` with 5–6 small checks of `clean_text()` (URL removed, hashtag split, emoji converted, "
                "repeated letters, Hindi/Bengali text unchanged) and run it with `pip install pytest` then `python -m pytest tests`. Commit and push.",
    },
    {
        "name": "Priyanshu Raj (S24CSEU1142)",
        "role": "Transformer models: BERT, mBERT, XLM-R, MuRIL; zero-shot vs translate-train",
        "files": "`src/train_transformer.py`, `src/models.py`, `src/config.py`",
        "built": [
            "Training loop: AdamW, lr 3e-5, batch 16, fp16, class-weighted loss, best epoch by validation Macro F1.",
            "Frozen word embeddings so XLM-R and MuRIL fit in the 4 GB GPU.",
            "Found and fixed the MuRIL problem: [CLS] vectors were identical for all sentences; switched to mean pooling.",
            "Continued fine-tuning (`--init_dir`) for the romanised model.",
        ],
        "explain": [
            f"Base paper's BERT: 75% English accuracy, but {f1('bert_en', 'hi'):.2f} / {f1('bert_en', 'bn'):.2f} F1 on Hindi/Bengali.",
            "Zero-shot (English only) vs translate-train: translate-train wins for every model (+0.06 to +0.22).",
            f"MuRIL > XLM-R > mBERT after translate-train; MuRIL is best ({f1('muril_tt', 'hi'):.3f} / {f1('muril_tt', 'bn'):.3f}).",
        ],
        "questions": [
            "What is fine-tuning? What does the classification head do?",
            "How did you find the MuRIL bug? (Loss stuck at 1.098 = ln 3; cosine similarity of [CLS] vectors ≈ 1.0.)",
            "Why freeze embeddings? What is mixed precision (fp16)?",
        ],
        "task": "Re-train the main model with a different random seed to check that our result is stable: "
                "`python src/train_transformer.py --model muril --train_langs en hi bn --epochs 6 --seed 13 --tag muril_tt_seed13` "
                "(~25 min). Commit `results/metrics/muril_tt_seed13.json` and add one line about the variation to `README.md`.",
    },
    {
        "name": "Advay Sinha (S24CSEU2059)",
        "role": "LLM baseline, ensemble, explainability (LIME), prediction / demo, results and report",
        "files": "`src/llm_fewshot.py`, `src/ensemble.py`, `src/explain_lime.py`, `src/predict.py`, `src/eval_challenge.py`, "
                 "`src/evaluate_saved.py`, `src/summarize_results.py`, `docs/make_progress_report.py`, `app/server.py` (web demo)",
        "built": [
            "Qwen2.5-1.5B-Instruct few-shot prompt baseline (label definitions + 1 example per class).",
            "Probability-averaging ensembles (3 models and 2 models).",
            "LIME explanations with whitespace word splitting for Hindi/Bengali.",
            "Prediction script and the `ID, Predicted Label` submission file; challenge-set evaluation.",
            "Web demo (`app/`): shows each pipeline step, the probabilities and word importance for any comment.",
        ],
        "explain": [
            f"LLM without training is weak ({f1('llm_qwen1.5b_fewshot', 'hi'):.2f} / {f1('llm_qwen1.5b_fewshot', 'bn'):.2f}); fine-tuning matters.",
            "Ensemble helps Hindi but not Bengali, so the single MuRIL model is our main model.",
            "LIME cues: ঝুঁকি / खतरा (risk) → Favour; না (not) → Against; प्रतिशत (percent) → None.",
        ],
        "questions": [
            "How does LIME work? (Removes words at random, watches the prediction change, fits a simple model.)",
            "Why did the LLM do badly? (Small 1.5B model, no training, only one example per class.)",
            "How will you make the final submission file?",
        ],
        "task": "Run LIME on our main model, `python src/explain_lime.py --model_dir models/muril_tt --n 3 --num_samples 300`, "
                "compare its top words with the XLM-R explanations, and commit a short `results/lime/comparison.md` (5–8 lines).",
    },
]


def part_c(doc):
    doc.add_heading("Part C. Individual Contributions", level=1)
    para(doc, "**How we worked:** the four of us discussed and worked on every part of the project together. "
              "For clarity, each member took **primary responsibility** for one area, listed below. That is the part "
              "they implemented, maintain and explain in depth, while helping on the others.")
    para(doc, "Every member must be able to explain the whole project (Part B) and, in detail, their own part. "
              "The evaluation also checks **GitHub contributions from every member**. At the moment all commits are by Varun, "
              "so each member must make **their own commit** before the evaluation (the task under each name).")
    for m in MEMBERS:
        doc.add_heading(m["name"], level=2)
        para(doc, f"**Role:** {m['role']}")
        para(doc, f"**Files:** {m['files']}")
        para(doc, "**What was built:**", after=1)
        for b in m["built"]:
            bullet(doc, b)
        para(doc, "**What to explain in the evaluation:**", after=1)
        for b in m["explain"]:
            bullet(doc, b)
        para(doc, "**Questions to prepare:**", after=1)
        for b in m["questions"]:
            bullet(doc, b)
        para(doc, f"**GitHub task before the evaluation (own commit):** {m['task']}")

    doc.add_heading("How each member pushes their own commit", level=2)
    numbered(doc, 1, f"Varun adds the others as collaborators: {REPO}/settings/access → *Add people*.")
    numbered(doc, 2, "Each member, on their own laptop and GitHub account:")
    code_block(doc, [
        f"git clone {REPO}.git",
        "cd Indic-Climate-Changes-",
        'git config user.name  "Your Name"',
        'git config user.email "your-github-email@example.com"',
        "# ... do your task ...",
        "git add <your files>",
        'git commit -m "Short description of what you did"',
        "git pull --rebase",
        "git push",
    ])
    numbered(doc, 3, "Check the result on GitHub: *Insights → Contributors* should show all four names.")


def part_d(doc):
    doc.add_heading("Part D. Evaluation-Day Plan", level=1)
    para(doc, "About 10–12 minutes in total. Everyone speaks. Open these before the slot: the GitHub page, "
              "`results/results_table.md`, `results/figures/model_comparison.png`, a PowerShell window in the project folder "
              "with `.\\.venv\\Scripts\\Activate.ps1` already run.")
    table(doc, [
        ["Order", "Who", "Time", "What to say / show"],
        ["1", "Varun", "2–3 min", "Problem and claim; base paper and how we extend it; datasets, links, sizes, class distribution "
         "(`results/dataset_stats.md`, class_distribution.png); GitHub repository structure"],
        ["2", "Yashika", "2 min", "Preprocessing live: `python src/preprocess.py`; TF-IDF results English-only vs translate-train; why Macro F1"],
        ["3", "Priyanshu", "3 min", "Transformer models, zero-shot vs translate-train, MuRIL mean-pooling fix; results table and chart"],
        ["4", "Advay", "3 min", "Web demo: live predictions step by step (below), submission file, LLM and ensemble results, LIME, challenges and future work; hand over the printed 2-page report"],
    ], widths=[1.3, 2.2, 1.8, 11.5])

    doc.add_heading("Web demo (show this to the teacher)", level=2)
    para(doc, "Double-click **start_demo.bat** in the project folder. The browser opens at http://127.0.0.1:5000 once the model "
              "is loaded (~10 seconds). Keep the black window open; closing it stops the demo.")
    for s in ["**Live demo tab:** click an example (or type a comment) and press *Analyse*. The page shows the six steps: raw input → "
              "each cleaning step and what it changed → script detection → the sub-word tokens the model reads → probabilities "
              "for Favour / Against / None → word importance (each word removed in turn to see how much the confidence drops).",
              "**Model menu:** switch to *MuRIL + romanised* for comments typed in English letters (the page suggests this automatically).",
              "**How it works tab:** training pipeline, dataset statistics, base paper vs our project.",
              "**Results tab:** all experiments, charts and confusion matrices.",
              "**Batch file tab:** upload a CSV with ID,text and download the submission file."]:
        bullet(doc, s)
    doc.add_heading("Live demo commands (tested on 17 Nov 2026)", level=2)
    code_block(doc, [
        'cd "D:\\Stance in Indic Climate"',
        ".\\.venv\\Scripts\\Activate.ps1",
        "",
        "# 1. preprocessing example (Yashika)",
        "python src/preprocess.py",
        "",
        "# 2. live predictions with our main model (Advay)        expected",
        'python src/predict.py --model_dir models/muril_tt --text "ग्लोबल वार्मिंग एक गंभीर समस्या है, हमें अभी कदम उठाने होंगे"   # Favour',
        'python src/predict.py --model_dir models/muril_tt --text "ये सब क्लाइमेट चेंज का ड्रामा है, गर्मी तो हमेशा पड़ती थी"      # Against',
        'python src/predict.py --model_dir models/muril_tt --text "ভিডিওটা দারুণ হয়েছে দাদা, পরের পর্ব কবে?"                      # None',
        'python src/predict.py --model_dir models/muril_tt --text "জলবায়ু পরিবর্তন সত্যিই ভয়ংকর"                                  # Favour',
        "",
        "# 3. submission file from a CSV with columns ID,text",
        "python src/predict.py --model_dir models/muril_tt --input my_test.csv --output submission.csv",
        "",
        "# 4. (optional, about 1 minute) re-run the TF-IDF baselines live",
        "python src/baselines.py",
    ])
    para(doc, "**Known weakness to mention honestly if asked:** romanised sarcasm such as \"ye sab jhooth hai bhai, garmi to har "
              "saal padti hai\" (\"it's all lies, it's hot every year\") is predicted as None instead of Against. This is in our future work.")
    doc.add_heading("If something goes wrong", level=2)
    bullet(doc, "GPU problem: predictions also run on the CPU. In PowerShell: `$env:CUDA_VISIBLE_DEVICES=\"\"` then the same command (a few seconds slower).")
    bullet(doc, "Laptop problem: show the saved results on GitHub: `results/results_table.md`, `results/figures/`, the progress report PDF.")
    bullet(doc, "Asked to train live: `python src/baselines.py` (about 1 minute) is safe; a transformer takes 5–25 minutes.")


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        doc.styles[name].font.color.rgb = BLUE
        doc.styles[name].font.size = Pt(size)

    # title page
    for _ in range(5):
        doc.add_paragraph()
    para(doc, "Detecting Stance in Media on Global Warming for Indic Languages", size=22, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    para(doc, "Round 2: Complete Project Documentation", size=15, align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
    for line in ["Varun Sharma (S24CSEU0170)", "Yashika Garg (S24CSEU0235)", "Priyanshu Raj (S24CSEU1142)",
                 "Advay Sinha (S24CSEU2059)"]:
        para(doc, line, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "", after=20)
    para(doc, "Guide: Dr. Supriya Chanda, SCSET, Bennett University", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    p = para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_hyperlink(p, REPO, REPO, 11)
    para(doc, "November 2026", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    page_break(doc)

    doc.add_heading("Contents", level=1)
    for line in ["Part A. Round 2 Checklist", "Part B. Full Project Explanation (task, base paper, datasets, pipeline, "
                 "models, results, problems, evaluator questions, demo, references)", "Part C. Individual Contributions",
                 "Part D. Evaluation-Day Plan"]:
        bullet(doc, line)
    para(doc, "Separate file to submit: **docs/Progress_Report_Round2.docx / .pdf** (the mandatory two-page report).")
    page_break(doc)

    part_a(doc)
    page_break(doc)
    doc.add_heading("Part B. Full Project Explanation", level=1)
    render_markdown(doc, (ROOT / "docs" / "PROJECT_GUIDE.md").read_text(encoding="utf8"))
    page_break(doc)
    part_c(doc)
    page_break(doc)
    part_d(doc)

    doc.save(OUT)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
