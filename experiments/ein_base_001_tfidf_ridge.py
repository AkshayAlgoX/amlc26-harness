"""Baseline + reference implementation of the contract. Copy this file to start a new experiment:
change EXP_ID (new NNN every time) and the model block — nothing else.

  QUICK=1 python -m experiments.ein_base_001_tfidf_ridge   # 10% smoke run
  python -m experiments.ein_base_001_tfidf_ridge           # full run -> artifacts/

regression:      TF-IDF (word 1-2 + char 3-5) -> Ridge on log1p(target), preds back in target units
classification:  same features -> LogisticRegression
extraction:      not a text-regression problem; use a VLM/OCR pipeline + log_external()
"""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from src.config import col, task
from src.data import text_series
from src.registry import Experiment

EXP_ID = "ein-base-001-tfidfridge"


def featurize(tr_text, others):
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000, sublinear_tf=True)
    xtr = hstack([vw.fit_transform(tr_text), vc.fit_transform(tr_text)]).tocsr()
    return xtr, [hstack([vw.transform(t), vc.transform(t)]).tocsr() for t in others]


def main():
    ttype = task()["type"]
    if ttype == "extraction":
        raise SystemExit("extraction task: build a VLM/OCR pipeline and register it with log_external()")
    exp = Experiment(EXP_ID, notes="TF-IDF word+char -> Ridge(alpha=2) on log1p(target)", test_agg="geo")
    id_col, tgt = col("id"), col("target")
    test_text = text_series(exp.test, 2000)
    for fold, trn, val in exp.split():
        xtr, (xva, xte) = featurize(text_series(trn, 2000), [text_series(val, 2000), test_text])
        if ttype == "regression":
            y = np.log1p(np.clip(trn[tgt].to_numpy(float), 0, None))
            m = Ridge(alpha=2.0).fit(xtr, y)
            to_units = lambda z: np.clip(np.expm1(z), 0.01, None)  # contract: original units
            exp.add_oof(fold, val[id_col], to_units(m.predict(xva)))
            exp.add_test_fold(to_units(m.predict(xte)))
        else:
            m = LogisticRegression(max_iter=2000, C=4.0).fit(xtr, trn[tgt])
            exp.add_oof(fold, val[id_col], m.predict(xva))
            exp.add_test_fold(m.predict(xte))
        print(f"fold {fold} done")
    exp.finish()


if __name__ == "__main__":
    main()
