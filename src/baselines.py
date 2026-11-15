"""Step 4a: TF-IDF baselines (Logistic Regression and Linear SVM).

Features: word 1-2 grams + character 2-5 grams (character n-grams cope
with spelling variation in social media text).

Two training settings:
    en  - train on English only (what the base paper had)
    tt  - translate-train: English + Hindi + Bengali
Each model is tested on the English, Hindi and Bengali test sets.

Usage:  python src/baselines.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from config import set_seed
from data_utils import compute_metrics, load_split, save_results

SETTINGS = {"en": ["en"], "tt": ["en", "hi", "bn"]}
TEST_LANGS = ["en", "hi", "bn"]


def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)),
    ])


CLASSIFIERS = {
    "lr": lambda: LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    "svm": lambda: LinearSVC(C=0.5, class_weight="balanced"),
}


def main():
    set_seed()
    tests = {l: load_split([l], "test") for l in TEST_LANGS}
    for setting, train_langs in SETTINGS.items():
        train = load_split(train_langs, "train")
        for name, make_clf in CLASSIFIERS.items():
            model = make_pipeline(make_features(), make_clf())
            model.fit(train["clean"], train["y"])
            results = {l: compute_metrics(t["y"], model.predict(t["clean"])) for l, t in tests.items()}
            save_results(f"tfidf_{name}_{setting}", results,
                         {"model": f"TF-IDF + {name.upper()}", "train_langs": train_langs, "n_train": len(train)})


if __name__ == "__main__":
    main()
