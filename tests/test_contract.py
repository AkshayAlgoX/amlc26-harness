import json

import numpy as np
import pandas as pd
import pytest

from src import blend, folds
from src.board import collect
from src.config import artifacts_dir, col
from src.data import load_test, load_train, train_with_folds
from src.make_submission import build as build_sub
from src.metric import score
from src.registry import Experiment, log_external
from src.validate import validate


def test_family_key_groups_pack_variants():
    a = folds.family_key("Acme Chips Classic 100g (Pack of 6)")
    b = folds.family_key("Acme Chips Classic 200 g (Pack of 2)")
    assert a == b and a != folds.family_key("Zenith Honey 500 g")


def test_folds_cover_each_id_once(regression_env):
    df = folds.build()
    tr = load_train()
    assert set(df["id"]) == set(tr[col("id")]) and df["id"].is_unique
    assert set(df["fold"]) == {0, 1, 2}
    with pytest.raises(SystemExit):
        folds.build()  # refuses to overwrite


@pytest.mark.parametrize("scheme", ["kfold", "stratified", "group", "stratified_group"])
def test_all_schemes_assign_every_row(regression_env, scheme):
    from src.config import load_config
    tr = load_train()
    f = folds.assign_folds(tr, scheme, 3, 1, load_config())
    assert (f >= 0).all() and set(f) == {0, 1, 2}


def _median_model_run(exp_id):
    exp = Experiment(exp_id, notes="median baseline")
    for k, trn, val in exp.split():
        m = float(trn[col("target")].median())
        exp.add_oof(k, val[col("id")], np.full(len(val), m))
        exp.add_test_fold(np.full(len(exp.test), m))
    return exp.finish()


def test_experiment_roundtrip(regression_env):
    folds.build()
    rec = _median_model_run("ein-base-001-median")
    a = artifacts_dir()
    oof = pd.read_csv(a / "oof" / "ein-base-001-median.csv", dtype={"id": str})
    test = pd.read_csv(a / "test" / "ein-base-001-median.csv", dtype={"id": str})
    assert list(oof.columns) == ["id", "fold", "pred"] and len(oof) == len(load_train())
    assert list(test.columns) == ["id", "pred"] and len(test) == len(load_test())
    tr = train_with_folds()
    m = oof.merge(tr[[col("id"), col("target")]], left_on="id", right_on=col("id"))
    assert rec["cv"] == pytest.approx(score(m[col("target")], m["pred"]), rel=1e-6)
    logged = json.loads((a / "logs" / "ein-base-001-median.json").read_text())
    assert logged["exp_id"] == "ein-base-001-median" and len(logged["fold_scores"]) == 3
    with pytest.raises(FileExistsError):
        Experiment("ein-base-001-median")  # new idea = new NNN


def test_bad_exp_id_rejected(regression_env):
    folds.build()
    with pytest.raises(ValueError):
        Experiment("MyCoolModel")


def test_missing_fold_violates_contract(regression_env):
    folds.build()
    exp = Experiment("ein-base-002-broken")
    for k, trn, val in exp.split():
        if k == 0:
            continue  # forgot a fold
        exp.add_oof(k, val[col("id")], np.ones(len(val)))
        exp.add_test_fold(np.ones(len(exp.test)))
    with pytest.raises(ValueError, match="contract violated"):
        exp.finish()


def test_quick_runs_are_isolated(regression_env, monkeypatch):
    folds.build()
    monkeypatch.setenv("QUICK", "1")
    _median_model_run("ein-base-003-quick")
    monkeypatch.delenv("QUICK")
    assert collect().empty and not collect(include_quick=True).empty


def _external(exp_id, noise, seed):
    rng = np.random.default_rng(seed)
    tr, te = load_train(), load_test()
    y = tr[col("target")].to_numpy(float)
    oof = pd.DataFrame({"id": tr[col("id")], "pred": y * rng.lognormal(0, noise, len(y))})
    test = pd.DataFrame({"id": te[col("id")], "pred": rng.lognormal(3, 0.5, len(te))})
    return log_external(exp_id, oof=oof, test=test, notes=f"noise {noise}")


def test_external_blend_submit_validate(regression_env, tmp_path):
    folds.build()
    r1 = _external("tm1-deep-001-a", 0.30, 1)
    r2 = _external("tm2-deep-001-b", 0.35, 2)
    blend.main(["--top", "5"])
    rec = json.loads((artifacts_dir() / "logs" / "cap-blend-001.json").read_text())
    assert sum(rec["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert rec["cv"] <= min(r1["cv"], r2["cv"]) + 1e-9  # blend never worse than best single on OOF
    out = tmp_path / "sub.csv"
    build_sub("cap-blend-001", str(out))
    assert validate(out) == []
    broken = pd.read_csv(out, dtype={"sample_id": str}).iloc[1:]
    broken.loc[broken.index[0], "price"] = np.nan
    broken.to_csv(tmp_path / "broken.csv", index=False)
    errs = validate(tmp_path / "broken.csv")
    assert any("rows" in e for e in errs) and any("missing" in e for e in errs) and any("non-numeric" in e for e in errs)


def test_extraction_vote_blend(extraction_env, tmp_path):
    folds.build()
    tr, te = load_train(), load_test()
    y = tr[col("target")].to_numpy(object)
    rng = np.random.default_rng(5)
    choices = np.array(["10 gram", "5 ounce", "250 gram", "2 ounce", ""], dtype=object)
    recs = []
    for i, acc in enumerate([0.8, 0.7, 0.6]):
        p = np.where(rng.random(len(y)) < acc, y, choices[rng.integers(len(choices), size=len(y))])
        tp = choices[rng.integers(len(choices), size=len(te))]
        recs.append(log_external(f"tm{i+1}-vlm-001-v", oof=pd.DataFrame({"id": tr[col("id")], "pred": p}),
                                 test=pd.DataFrame({"id": te[col("id")], "pred": tp})))
    blend.main(["--top", "3"])
    rec = json.loads((artifacts_dir() / "logs" / "cap-blend-001.json").read_text())
    assert rec["cv"] >= max(r["cv"] for r in recs) - 1e-9  # F1: higher is better
    out = tmp_path / "sub.csv"
    build_sub("cap-blend-001", str(out))
    assert validate(out) == []
