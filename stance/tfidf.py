"""TF-IDF + linear classifier model (kept in its own module so pickles load from any entry point)."""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .config import LABELS
from .text import normalize


class TfidfModel:
    def __init__(self, clf="lr", C=1.0):
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300_000)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                    token_pattern=r"(?u)[\wऀ-ॿঀ-৿]+", max_features=200_000)
        self.clf = (LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
                    if clf == "lr" else LinearSVC(C=C, class_weight="balanced"))

    @staticmethod
    def _prep(texts):
        return [normalize(t, lowercase_latin=True) for t in texts]

    def fit(self, texts, labels):
        t = self._prep(texts)
        X = hstack([self.char.fit_transform(t), self.word.fit_transform(t)]).tocsr()
        self.clf.fit(X, [LABELS.index(l) for l in labels])
        return self

    def predict_proba(self, texts):
        t = self._prep(texts)
        X = hstack([self.char.transform(t), self.word.transform(t)]).tocsr()
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(X)
        else:  # softmax over SVM margins as pseudo-probabilities
            d = self.clf.decision_function(X)
            e = np.exp(d - d.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
        full = np.zeros((len(texts), len(LABELS)))
        full[:, self.clf.classes_] = p
        return full
