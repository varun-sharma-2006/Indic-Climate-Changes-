"""Sanity check on hand-written sentences of the kind an evaluator might type.

Covers negation, "I don't think ...", sentiment without stance, implicit stance,
off-topic questions, Bengali, romanised Hindi and sarcasm. These sentences were
written to test the models; they are NOT used for training.

Usage:  python src/sanity_check.py --model_dirs models/muril_tt models/xlmr_tt
"""
import config  # noqa: F401  (must come first: keeps the model cache on the project drive)
import argparse
import sys
from pathlib import Path

from config import LABELS
from predict import StancePredictor

CASES = [  # (text, expected, what it tests)
    ("मैं खुश नहीं हूँ।", "None", "negative sentiment, no climate content"),
    ("मैं जलवायु परिवर्तन से खुश नहीं हूँ, ये बहुत खतरनाक है", "Favour", "worried about climate"),
    ("जलवायु परिवर्तन कोई समस्या नहीं है", "Against", "negation"),
    ("ग्लोबल वार्मिंग झूठ है", "Against", "short denial"),
    ("ग्लोबल वार्मिंग सच है", "Favour", "short acceptance"),
    ("मुझे नहीं लगता कि ग्लोबल वार्मिंग असली है", "Against", "\"I don't think it is real\""),
    ("जलवायु परिवर्तन गंभीर समस्या नहीं है, ये सब मीडिया का प्रचार है", "Against", "not serious + propaganda"),
    ("पेड़ लगाओ, धरती बचाओ", "Favour", "implicit (call to action)"),
    ("वाह, दिसंबर में भी AC चल रहा है, फिर भी लोग कहते हैं कुछ नहीं बदला", "Favour", "sarcasm (favour)"),
    ("हाँ हाँ, धरती तो कल ही खत्म हो जाएगी 🙄", "Against", "sarcasm (against)"),
    ("आज क्रिकेट मैच कौन जीतेगा?", "None", "off-topic"),
    ("ये वीडियो किसने बनाया?", "None", "off-topic question"),
    ("জলবায়ু পরিবর্তন একটা ভুয়া জিনিস", "Against", "Bengali denial"),
    ("জলবায়ু পরিবর্তন নিয়ে আমি খুব চিন্তিত", "Favour", "Bengali, worried"),
    ("আমি খুশি নই", "None", "Bengali, sentiment only"),
    ("বিশ্ব উষ্ণায়ন বলে কিছু নেই", "Against", "Bengali, \"there is no global warming\""),
    ("global warming jhooth hai", "Against", "romanised denial"),
    ("climate change bahut khatarnak hai", "Favour", "romanised acceptance"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", default=["models/muril_tt"])
    args = ap.parse_args()
    for d in args.model_dirs:
        probs = StancePredictor(d).predict_proba([t for t, _, _ in CASES])
        correct = 0
        print(f"== {Path(d).name}")
        for (text, exp, what), p in zip(CASES, probs):
            got = LABELS[p.argmax()]
            correct += got == exp
            print(f"  {'OK   ' if got == exp else 'WRONG'} expected {exp:7} got {got:7} ({p.max():.2f})  [{what}]  {text}")
        print(f"  {correct}/{len(CASES)} correct\n")


if __name__ == "__main__":
    main()
