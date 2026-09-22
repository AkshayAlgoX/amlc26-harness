"""Build data/folds.csv ONCE (captain), and diagnose which fold scheme to use.

  python -m src.folds diagnose        # which scheme makes CV look like the test set?
  python -m src.folds build           # writes data/folds.csv (refuses to overwrite)
  python -m src.folds build --force   # only if you are sure; tell the team, all OOF is invalidated

Why diagnose: catalogs are full of near-duplicates (same product, different pack size).
If test items have near-duplicates in train at the same rate as a random split, random
folds are honest. If they don't, random folds leak and CV will look far better than the LB.
The diagnostic measures nearest-neighbour text similarity test->train and val->rest for each
scheme and recommends the scheme whose distribution matches test->train best.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from .config import col, data_path, load_config, task
from .data import load_test, load_train, text_series

_UNITS = (
    r"mg|g|gm|gms|gram|grams|kg|kgs|kilogram|kilograms|oz|ounce|ounces|lb|lbs|pound|pounds|"
    r"ml|l|ltr|liter|litre|liters|litres|fl|floz|count|ct|pack|packs|pk|pcs|pc|piece|pieces|"
    r"cm|mm|m|inch|inches|in|ft|foot|feet|w|watt|watts|v|volt|volts|kw|mah|x|of|value|unit"
)
_UNIT_RE = re.compile(rf"\b(?:{_UNITS})\b")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_NONALPHA_RE = re.compile(r"[^a-z\s]")


def family_key(text: str, n_tokens: int = 10) -> str:
    """Heuristic product-family key: drop numbers + unit words, keep the first n tokens.
    'Brand X Chips 100g (Pack of 6)' and 'Brand X Chips 200g (Pack of 2)' -> same key."""
    t = str(text).lower()
    t = _NUM_RE.sub(" ", t)
    t = _NONALPHA_RE.sub(" ", t)
    t = _UNIT_RE.sub(" ", t)
    return " ".join(t.split()[:n_tokens])


def _strat_labels(train: pd.DataFrame, cfg: dict) -> np.ndarray:
    fcfg, ttype = cfg["folds"], task()["type"]
    if fcfg.get("strat_col"):
        return train[fcfg["strat_col"]].fillna("NA").astype(str).to_numpy()
    y = train[col("target")]
    if ttype == "regression":
        v = np.log1p(np.clip(pd.to_numeric(y, errors="coerce").fillna(0).to_numpy(float), 0, None))
        return pd.qcut(v, q=int(fcfg.get("n_bins", 20)), labels=False, duplicates="drop").astype(int)
    if ttype == "extraction":
        return (y.astype(str).str.strip() == "").astype(int).to_numpy()
    return y.astype(str).to_numpy()


def _groups(train: pd.DataFrame) -> np.ndarray:
    gcol = col("group")
    if gcol:
        return train[gcol].fillna("NA").astype(str).to_numpy()
    keys = text_series(train).map(family_key)
    return pd.factorize(keys)[0]


def assign_folds(train: pd.DataFrame, scheme: str, n_splits: int, seed: int, cfg: dict) -> np.ndarray:
    fold = np.full(len(train), -1, dtype=int)
    X = np.zeros(len(train))
    if scheme == "kfold":
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    elif scheme == "stratified":
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, _strat_labels(train, cfg))
    elif scheme == "group":
        splits = GroupKFold(n_splits).split(X, groups=_groups(train))
    elif scheme == "stratified_group":
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            X, _strat_labels(train, cfg), groups=_groups(train)
        )
    else:
        raise ValueError(f"unknown fold scheme '{scheme}'")
    for k, (_, va) in enumerate(splits):
        fold[va] = k
    assert (fold >= 0).all(), "some rows got no fold"
    return fold


def build(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    out_path = data_path("folds")
    if out_path.exists() and not force:
        sys.exit(f"{out_path} already exists. Refusing to overwrite (use --force and tell the team).")
    fcfg = cfg["folds"]
    train = load_train()
    fold = assign_folds(train, fcfg["scheme"], int(fcfg["n_splits"]), int(fcfg["seed"]), cfg)
    df = pd.DataFrame({"id": train[col("id")].astype(str).to_numpy(), "fold": fold})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  scheme={fcfg['scheme']}  n_splits={fcfg['n_splits']}  rows={len(df)}")
    sizes = df["fold"].value_counts().sort_index()
    print("fold sizes:", {int(k): int(v) for k, v in sizes.items()})
    if task()["type"] == "regression":
        y = pd.to_numeric(train[col("target")], errors="coerce")
        print("median target per fold:", {int(k): round(float(y[fold == k].median()), 3) for k in sizes.index})
    return df


# ---------------------------------------------------------------- diagnostic
def _embed(train_text: pd.Series, test_text: pd.Series, dim: int, seed: int):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300_000,
                          sublinear_tf=True, dtype=np.float32)
    xtr = vec.fit_transform(train_text)
    xte = vec.transform(test_text)
    dim = max(2, min(dim, xtr.shape[1] - 1, xtr.shape[0] - 1))
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    ztr = normalize(svd.fit_transform(xtr)).astype(np.float32)
    zte = normalize(svd.transform(xte)).astype(np.float32)
    return ztr, zte


def _max_sim(q: np.ndarray, base: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), chunk):
        out[i:i + chunk] = (q[i:i + chunk] @ base.T).max(axis=1)
    return out


def _summ(sims: np.ndarray) -> dict:
    qs = np.quantile(sims, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"p10": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p90": qs[4], "dup>=0.95": float((sims >= 0.95).mean())}


def diagnose(sample: int = 3000, dim: int = 256) -> str:
    cfg = load_config()
    fcfg = cfg["folds"]
    seed, k = int(fcfg["seed"]), int(fcfg["n_splits"])
    train, test = load_train(), load_test()
    rng = np.random.default_rng(seed)
    print(f"embedding {len(train)} train + {len(test)} test texts (char TF-IDF -> SVD{dim}) ...")
    ztr, zte = _embed(text_series(train, 1000), text_series(test, 1000), dim, seed)

    rows = {}
    q = rng.choice(len(zte), size=min(sample, len(zte)), replace=False)
    rows["test -> train"] = _summ(_max_sim(zte[q], ztr))

    candidates = ["kfold", "stratified", "group", "stratified_group"]
    for scheme in candidates:
        fold = assign_folds(train, scheme, k, seed, cfg)
        va = np.where(fold == 0)[0]
        rest = np.where(fold != 0)[0]
        qv = rng.choice(va, size=min(sample, len(va)), replace=False)
        rows[f"{scheme}: val -> rest"] = _summ(_max_sim(ztr[qv], ztr[rest]))

    table = pd.DataFrame(rows).T.round(3)
    print(table.to_string())
    target = table.loc["test -> train"]

    def dist(r):
        return float(np.abs(r[["p10", "p25", "p50", "p75", "p90"]] - target[["p10", "p25", "p50", "p75", "p90"]]).sum()
                     + 2 * abs(r["dup>=0.95"] - target["dup>=0.95"]))

    scores = {s: dist(table.loc[f"{s}: val -> rest"]) for s in candidates}
    best = min(scores, key=scores.get)
    print("\ndistance to test->train (lower = more honest CV):", {s: round(v, 3) for s, v in scores.items()})
    print(f"RECOMMENDED folds.scheme: {best}   (set it in config.yaml, then `python -m src.folds build`)")
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    d = sub.add_parser("diagnose")
    d.add_argument("--sample", type=int, default=3000)
    d.add_argument("--dim", type=int, default=256)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(force=args.force)
    else:
        diagnose(sample=args.sample, dim=args.dim)


if __name__ == "__main__":
    main()
