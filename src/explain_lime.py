"""Step 6: explain predictions with LIME (which words pushed the model to its label).

Picks test examples in Hindi and Bengali (some correct, some wrong) and saves
an HTML explanation for each plus a summary of the most influential words.

Usage:  python src/explain_lime.py --model_dir models/muril_tt --n 4
"""
import argparse

from lime.lime_text import LimeTextExplainer

from config import LABELS, RESULTS_DIR, SEED
from data_utils import load_split
from predict import StancePredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--n", type=int, default=4, help="examples per language")
    ap.add_argument("--num_samples", type=int, default=500)
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "lime"
    out_dir.mkdir(exist_ok=True)
    predictor = StancePredictor(args.model_dir)
    explainer = LimeTextExplainer(class_names=LABELS, random_state=SEED)
    summary = ["# LIME explanations\n"]

    for lang in ["hi", "bn"]:
        test = load_split([lang], "test")
        test["pred"] = predictor.predict_proba(test["text"].tolist()).argmax(1)
        right = test[test["pred"] == test["y"]].groupby("y").head(1)
        wrong = test[test["pred"] != test["y"]].head(max(1, args.n - len(right)))
        for k, row in enumerate(list(right.itertuples()) + list(wrong.itertuples())):
            exp = explainer.explain_instance(row.clean, lambda t: predictor.predict_proba(list(t)),
                                             num_features=8, labels=[row.pred], num_samples=args.num_samples)
            exp.save_to_file(str(out_dir / f"{lang}_{k}.html"))
            words = ", ".join(f"{w} ({v:+.2f})" for w, v in exp.as_list(label=row.pred))
            status = "correct" if row.pred == row.y else "WRONG"
            summary += [f"## {lang}_{k}: true={LABELS[row.y]}, predicted={LABELS[row.pred]} ({status})",
                        f"Text: {row.clean}", f"Top words for '{LABELS[row.pred]}': {words}", ""]
            print(summary[-3], "\n", summary[-2])

    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf8")


if __name__ == "__main__":
    main()
