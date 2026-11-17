"""Make PDF copies of everything needed for the Round 2 evaluation, in one folder: Round2_PDFs/

  1_Progress_Report_Round2.pdf          the 2-page progress report (Word -> PDF)
  2_Round2_Complete_Documentation.pdf   checklist, guide, contributions, demo plan (Word -> PDF)
  3_Project_Guide.pdf                   docs/PROJECT_GUIDE.md (Markdown -> HTML -> PDF with Edge)
  4_README_GitHub_Overview.pdf          README.md, the GitHub front page
  5_Base_Paper_Luo_Card_Jurafsky_2020.pdf

Run on Windows with Word and Edge installed:
  python docs/make_progress_report.py ; python docs/make_round2_document.py ; python docs/make_pdfs.py
"""
import shutil
import subprocess
from pathlib import Path

import markdown
import win32com.client  # pywin32

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "Round2_PDFs"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
body { font-family: 'Segoe UI', 'Nirmala UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1d1d1f; max-width: 100%; margin: 0; }
h1 { font-size: 19pt; color: #123c3a; border-bottom: 2px solid #1f6f6a; padding-bottom: 4px; }
h2 { font-size: 14pt; color: #123c3a; margin-top: 20px; border-bottom: 1px solid #d0d7d6; padding-bottom: 2px; }
h3 { font-size: 12pt; color: #1f6f6a; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #b9c3c2; padding: 3px 6px; vertical-align: top; }
th { background: #e6efee; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f1f3f3; padding: 0 2px; }
pre { background: #f1f3f3; padding: 8px; font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
img { max-width: 100%; max-height: 110mm; display: block; margin: 0 auto; }
blockquote { border-left: 3px solid #1f6f6a; margin-left: 0; padding-left: 10px; color: #444; }
a { color: #1f6f6a; text-decoration: none; }
@page { size: A4; margin: 16mm 15mm; }
"""


def word_to_pdf(word, src: Path, dst: Path):
    doc = word.Documents.Open(str(src), False, True)
    doc.ExportAsFixedFormat(str(dst), 17)  # 17 = wdExportFormatPDF
    doc.Close(False)


def md_to_pdf(src: Path, dst: Path, title: str):
    body = markdown.markdown(src.read_text(encoding="utf8"), extensions=["tables", "fenced_code", "sane_lists"])
    # relative links/images (results/figures/...) resolve from the folder of the Markdown file
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<base href='{src.parent.as_uri()}/'><style>{CSS}</style></head><body>{body}</body></html>")
    tmp = OUT / f"_{dst.stem}.html"
    tmp.write_text(html, encoding="utf8")
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={dst}", tmp.as_uri()], check=True, timeout=120)
    tmp.unlink()


def main():
    OUT.mkdir(exist_ok=True)
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        word_to_pdf(word, DOCS / "Progress_Report_Round2.docx", OUT / "1_Progress_Report_Round2.pdf")
        word_to_pdf(word, DOCS / "Round2_Complete_Documentation.docx", OUT / "2_Round2_Complete_Documentation.pdf")
    finally:
        word.Quit()
    shutil.copy(OUT / "1_Progress_Report_Round2.pdf", DOCS / "Progress_Report_Round2.pdf")
    md_to_pdf(DOCS / "PROJECT_GUIDE.md", OUT / "3_Project_Guide.pdf", "Project Guide")
    md_to_pdf(ROOT / "README.md", OUT / "4_README_GitHub_Overview.pdf", "README")
    shutil.copy(DOCS / "Base_Paper_Luo_Card_Jurafsky_2020.pdf", OUT / "5_Base_Paper_Luo_Card_Jurafsky_2020.pdf")
    for p in sorted(OUT.glob("*.pdf")):
        print(f"{p.name:45s} {p.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
