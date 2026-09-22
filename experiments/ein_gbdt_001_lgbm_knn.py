"""GBDT track template (regression): regex numeric features + text-SVD + kNN retrieval features -> LightGBM.

kNN features = the neighbours' log-target stats (the 'retrieval regressor' trick). They are computed
strictly inside each fold: training rows look up neighbours in the training part with themselves
excluded; validation and test rows look up neighbours in the training part. No labels leak.
The TF-IDF/SVD text embedding is fit on train+test text WITHOUT labels (unsupervised, allowed).

  QUICK=1 python -m experiments.ein_gbdt_001_lgbm_knn
  python -m experiments.ein_gbdt_001_lgbm_knn
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import col, task
from src.data import text_series
from src.registry import Experiment

EXP_ID = "ein-gbdt-001-lgbm-knn"
K_LIST = (1, 5, 20)
SVD_DIM = 128
SVD_AS_FEATS = 32
SEED = 42
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")  # lightgbm>=4.7 rename; keep old API for Kaggle images


def numeric_features(text: pd.Series) -> pd.DataFrame:
    """Generic catalog-text parsing. Extend on Day 1 after EDA (units, sizes, counts...)."""
    t = text.fillna("").astype(str)
    out = pd.DataFrame(index=text.index)
    out["value"] = pd.to_numeric(t.str.extract(r"(?i)value:\s*(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    out["unit"] = t.str.extract(r"(?i)unit:\s*([a-z ]{1,20})", expand=False).str.strip().str.lower().fillna("none")
    pk = t.str.extract(r"(?i)pack of\s*(\d+)|(\d+)\s*(?:count|ct|pk|pack)\b")
    out["pack"] = pd.to_numeric(pk[0].fillna(pk[1]), errors="coerce")
    nums = t.str.findall(r"\d+(?:\.\d+)?")
    out["n_numbers"] = nums.str.len()
    out["log_max_number"] = np.log1p(nums.map(lambda xs: max((float(x) for x in xs), default=np.nan)))
    out["len_chars"] = t.str.len()
    out["n_words"] = t.str.split().str.len()
    out["value_x_pack"] = out["value"] * out["pack"].fillna(1)
    return out


def embed(train_text: pd.Series, test_text: pd.Series):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000,
                          sublinear_tf=True, dtype=np.float32)
    x = vec.fit_transform(pd.concat([train_text, test_text], ignore_index=True))
    dim = min(SVD_DIM, x.shape[1] - 1, x.shape[0] - 1)
    z = normalize(TruncatedSVD(n_components=dim, random_state=SEED).fit_transform(x)).astype(np.float32)
    return z[: len(train_text)], z[len(train_text):]


def knn_feats(q: np.ndarray, base: np.ndarray, base_y: np.ndarray, exclude_self: bool = False,
              chunk: int = 1024) -> pd.DataFrame:
    kmax = min(max(K_LIST), len(base) - 1)
    cols = {f"knn_mean_{k}": [] for k in K_LIST} | {"knn_wmean": [], "knn_std": [], "knn_sim1": [], "knn_sim5": []}
    for i in range(0, len(q), chunk):
        s = q[i:i + chunk] @ base.T
        if exclude_self:
            r = np.arange(s.shape[0])
            s[r, i + r] = -np.inf
        idx = np.argpartition(-s, kmax - 1, axis=1)[:, :kmax]
        top_s = np.take_along_axis(s, idx, axis=1)
        order = np.argsort(-top_s, axis=1)
        idx, top_s = np.take_along_axis(idx, order, axis=1), np.take_along_axis(top_s, order, axis=1)
        top_y = base_y[idx]
        for k in K_LIST:
            cols[f"knn_mean_{k}"].append(top_y[:, :min(k, kmax)].mean(axis=1))
        w = np.clip(top_s, 0, None) + 1e-6
        cols["knn_wmean"].append((top_y * w).sum(axis=1) / w.sum(axis=1))
        cols["knn_std"].append(top_y.std(axis=1))
        cols["knn_sim1"].append(top_s[:, 0])
        cols["knn_sim5"].append(top_s[:, :5].mean(axis=1))
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})


def make_model():
    try:
        import lightgbm as lgb
        return "lgb", lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=20,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                        random_state=SEED, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "hgb", HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.05, random_state=SEED)


def main():
    if task()["type"] != "regression":
        raise SystemExit("this template is for regression tasks")
    exp = Experiment(EXP_ID, notes="regex numerics + char-TFIDF SVD32 + in-fold kNN(1/5/20) -> LGBM on log1p", test_agg="geo")
    id_col, tgt = col("id"), col("target")
    tr_text, te_text = text_series(exp.train, 1500), text_series(exp.test, 1500)
    ztr, zte = embed(tr_text, te_text)
    num_tr, num_te = numeric_features(tr_text), numeric_features(te_text)
    units = pd.Categorical(pd.concat([num_tr["unit"], num_te["unit"]])).categories
    for d in (num_tr, num_te):
        d["unit"] = pd.Categorical(d["unit"], categories=units).codes
    svd_tr = pd.DataFrame(ztr[:, :SVD_AS_FEATS], columns=[f"svd{i}" for i in range(SVD_AS_FEATS)])
    svd_te = pd.DataFrame(zte[:, :SVD_AS_FEATS], columns=svd_tr.columns)
    ylog = np.log1p(np.clip(exp.train[tgt].to_numpy(float), 0, None))
    rng = np.random.default_rng(SEED)

    def frame(rows_num, rows_svd, knn):
        return pd.concat([rows_num.reset_index(drop=True), rows_svd.reset_index(drop=True), knn], axis=1)

    for fold, tr, va in exp.split_idx():
        x_tr = frame(num_tr.iloc[tr], svd_tr.iloc[tr], knn_feats(ztr[tr], ztr[tr], ylog[tr], exclude_self=True))
        x_va = frame(num_tr.iloc[va], svd_tr.iloc[va], knn_feats(ztr[va], ztr[tr], ylog[tr]))
        x_te = frame(num_te, svd_te, knn_feats(zte, ztr[tr], ylog[tr]))
        kind, model = make_model()
        if kind == "lgb":
            import lightgbm as lgb
            inner = rng.random(len(tr)) < 0.1  # early-stop on an inner split, never on the scored fold
            model.fit(x_tr[~inner], ylog[tr][~inner], eval_set=[(x_tr[inner], ylog[tr][inner])],
                      categorical_feature=["unit"], callbacks=[lgb.early_stopping(150, verbose=False)])
        else:
            model.fit(x_tr, ylog[tr])
        to_units = lambda z: np.clip(np.expm1(z), 0.01, None)
        exp.add_oof(fold, exp.train[id_col].iloc[va], to_units(model.predict(x_va)))
        exp.add_test_fold(to_units(model.predict(x_te)))
        print(f"fold {fold} done")
    exp.finish()


if __name__ == "__main__":
    main()
