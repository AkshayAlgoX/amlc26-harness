"""End-to-end invariants of the production pipeline on synthetic data (seconds; no real data).

1. H4 mode of agy2_pipeline.py == the validated reference agy2_exp9_h4_eval.py (same selection, same H1/H4 reports).
2. B2 feature parity: pipeline H1 matcher rows == exp9 cached feature build (agy2_exp9_retrain c4) row by row.
3. Export round-trip; H4 labels never persisted in artifacts.
4. Test mode: country x hash shards partition test S1 exactly; sharding is exact (1 shard == 2 shards);
   France (unseen) is scored with country=NaN; merge writes both TSVs that pass the validators; one-to-one holds.
5. Fail-closed: pinned-module mismatch, tampered export, missing shard.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import pipeline_synth  # noqa: E402
from experiments import agy2_exp8_store as ST  # noqa: E402
from experiments.agy2_exp7_freeze import build as build_exp7  # noqa: E402
from experiments.agy2_exp7_prep_h3 import sha256_file  # noqa: E402

RETR = {"pool_channels": ["A", "B", "F200", "R5", "B2C", "B3C", "B4C"], "max_df": 1.0, "rev_df_cap": 100000,
        "k_fwd": 50, "chunk": 1500, "cap": 100000, "fwd_min_sim": 0.05, "canonical_view": False}
BUDGET = {"top_n": 10, "floor": 0.0}
ORGANISER_VALIDATOR = Path.home() / "Downloads/student_resource/student_resource/utils/validate_submission.py"


def _run(script, env, cwd=ROOT, expect_ok=True):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **{k: str(v) for k, v in env.items()})
    r = subprocess.run([sys.executable, str(ROOT / script)], cwd=cwd, env=e, capture_output=True, text=True, timeout=900)
    if expect_ok and r.returncode != 0:
        raise AssertionError(f"{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def _read(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet" and p.is_file():
        return ST.read_parquet(p)
    return pd.read_pickle(p.with_suffix(".pkl"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    syn = pipeline_synth.make(root)
    F = syn["files"]
    # exp7-lineage config for the H1 dump (the source of the exp9 cache)
    c7 = build_exp7(Path("/"), {k: F[k] for k in ("H0", "H1", "H2", "H3")}, retrieval=RETR, budget=BUDGET, h3_n=syn["h3_n"])
    for m in ("pruner", "matcher"):
        c7[m]["max_rounds"] = 150
    (root / "exp7_config.json").write_text(json.dumps(c7))
    _run("experiments/agy2_exp7_h1dump.py", {"EXP7_CONFIG": root / "exp7_config.json", "DATA_ROOT": syn["data"], "OUT_DIR": root / "dump"})
    for n in ("h1_matcher_oof", "h1_queries", "h1_truth"):
        if not (root / "dump" / f"{n}.parquet").is_file():
            ST.write_parquet(pd.read_pickle(root / "dump" / f"{n}.pkl"), root / "dump" / f"{n}.parquet")
    dump = ST.read_parquet(root / "dump" / "h1_matcher_oof.parquet")
    pre = json.loads((ROOT / "experiments/configs/exp9_prereg.json").read_text())
    pre["base"].update(expected_rows=len(dump), expected_selected=int(dump.selected.sum()))
    (root / "prereg.json").write_text(json.dumps(pre))
    _run("experiments/agy2_exp9_retrain.py", {"EXP9_PREREG": root / "prereg.json", "EXP7_CONFIG": root / "exp7_config.json",
                                               "EXP8_IDS_ROOT": "/", "EXP8_STORE": root / "store", "EXP8_SEALED_LISTS": F["H4"],
                                               "DATA_ROOT": syn["data"], "OUT_DIR": root / "exp9",
                                               "DUMP_GLOB": root / "dump" / "h1_matcher_oof.parquet", "VARIANTS": ""})
    # pipeline config: frozen builder + test-only overrides (exp7 stats exclusion for parity; full rule grids)
    from experiments.agy2_pipeline_freeze import build_pipeline_config
    fake9 = {"variants": {"B2": {"macro_f05": 0.0, "rule": "expF", "params": {"m": 0.0, "s": 0.5}}}}
    cfg = build_pipeline_config(Path("/"), F, fake9, "smoke", smoke=True, retrieval=RETR, budget=BUDGET, h3_n=syn["h3_n"])
    cfg["stats_exclude"] = ["H2", "H3"]
    cfg["set_rules"] = {**c7["set_rules"]}
    for m in ("pruner", "matcher"):
        cfg[m]["max_rounds"] = 150
    (root / "pipe_config.json").write_text(json.dumps(cfg))
    penv = {"PIPELINE_SMOKE": "1", "PIPELINE_CONFIG": root / "pipe_config.json", "DATA_ROOT": syn["data"]}
    _run("experiments/agy2_pipeline.py", {**penv, "PIPELINE_MODE": "h4", "OUT_DIR": root / "h4"})
    # reference: the validated exp9 H4 script on the identical configuration
    ref = dict(cfg, code_sha256={"experiments/agy2_exp9_h4_eval.py": sha256_file(ROOT / "experiments/agy2_exp9_h4_eval.py")})
    (root / "ref_config.json").write_text(json.dumps(ref))
    _run("experiments/agy2_exp9_h4_eval.py", {"EXP9_H4_CONFIG": root / "ref_config.json", "DATA_ROOT": syn["data"], "OUT_DIR": root / "ref"})
    return {"root": root, "syn": syn, "penv": penv, "cfg": cfg}


def _shards(w, out, plan):
    for country, count in plan.items():
        for i in range(count):
            _run("experiments/agy2_pipeline.py", {**w["penv"], "PIPELINE_MODE": "test", "MODELS_DIR": w["root"] / "h4" / "export",
                                                  "SHARD_COUNTRY": country, "SHARD_INDEX": i, "SHARD_COUNT": count, "OUT_DIR": out})


def test_h4_mode_equals_validated_reference(world):
    a = json.loads((world["root"] / "h4" / "pipeline_h4_results.json").read_text())
    b = json.loads((world["root"] / "ref" / "exp9_h4_results.json").read_text())
    assert a["selected"] == b["selected"]
    assert a["reports"] == b["reports"]
    assert a["pruner"]["best_iters"] == b["pruner"]["best_iters"] and a["matcher"]["best_iters"] == b["matcher"]["best_iters"]
    assert a["matcher"]["export_roundtrip_max_abs_diff"] <= 1e-9 and a["pruner"]["export_roundtrip_max_abs_diff"] <= 1e-9


def test_b2_feature_parity_with_exp9_cache(world):
    from experiments import agy2_exp9_features as F9
    store = ST.Store(root=world["root"] / "store", forbidden=set())
    c4 = [store.open("c4_mfeat", k) for k in store.find("c4_mfeat")][0]
    v0 = [store.open("v0_seed", k) for k in store.find("v0_seed")][0]
    nf = c4.table("newfeat").merge(v0.table("queries", columns=["q", "s1_id"]), on="q")
    pm = _read(world["root"] / "h4" / "artifacts" / "c4c5_matcher_rows.parquet")
    pm = pm[pm.g == 0]
    a = nf.set_index(["s1_id", "code"]).sort_index()
    b = pm.set_index(["s1_id", "code"]).sort_index()
    assert a.index.equals(b.index), "H1 budget rows differ between exp9 cache and pipeline"
    for f in F9.NEW_FEATURES:
        x, y = a[f].to_numpy(np.float64), b[f].to_numpy(np.float64)
        assert np.array_equal(np.isnan(x), np.isnan(y)), f
        assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=0, atol=1e-6), f


def test_export_manifest_and_no_h4_labels(world):
    exp = world["root"] / "h4" / "export"
    man = json.loads((exp / "manifest.json").read_text())
    assert {f"{t}_fold{k}.txt" for t in ("pruner", "matcher") for k in range(5)} | {"decision.json"} <= set(man["files"])
    for f in ("c2c3_pool_prebudget.parquet", "c4c5_matcher_rows.parquet"):
        d = _read(world["root"] / "h4" / "artifacts" / f)
        assert (d.label[d.g == 1] == -1).all(), f"H4 labels persisted in {f}"


def test_test_mode_sharding_exact_and_merge_valid(world):
    root = world["root"]
    _shards(world, root / "shA", {"US": 2, "India": 2, "France": 1})
    _shards(world, root / "shB", {"India": 1})
    ia = pd.concat([_read(p) for p in sorted((root / "shA").glob("shard_India_*_scores.parquet"))])
    ib = _read(sorted((root / "shB").glob("shard_India_*_scores.parquet"))[0])
    key = ["s1_pos", "code"]
    ia, ib = ia.sort_values(key).reset_index(drop=True), ib.sort_values(key).reset_index(drop=True)
    assert ia[key].equals(ib[key]) and np.array_equal(ia.p.values, ib.p.values), "sharding changed India scores"
    fr = json.loads(next((root / "shA").glob("shard_France_*_meta.json")).read_text())
    assert fr["s1"] > 0 and fr["pairs"] > 0
    from experiments import agy2_pipeline_merge as MG
    args = ["--shards", str(root / "shA"), "--models", str(root / "h4" / "export"), "--data-root", str(world["syn"]["data"]),
            "--out", str(root / "out"), "--repo-validator"]
    if ORGANISER_VALIDATOR.is_file():
        args += ["--organiser-validator", str(ORGANISER_VALIDATOR)]
    env_backup = dict(os.environ)
    os.environ.update({k: str(v) for k, v in world["penv"].items()})
    try:
        import importlib
        from experiments import agy2_pipeline as PL
        importlib.reload(PL)
        importlib.reload(MG)
        rep = MG.main(args)
    finally:
        os.environ.clear(); os.environ.update(env_backup)
    assert rep["stream_validator"]["valid"], rep["stream_validator"]
    assert rep["repo_validator"]["valid"], rep["repo_validator"]
    if ORGANISER_VALIDATOR.is_file():
        assert rep["organiser_validator"]["exit_code"] == 0, rep["organiser_validator"]
    mr = pd.read_csv(root / "out" / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    cp = pd.read_csv(root / "out" / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(mr) == len(cp) == world["syn"]["n_test_s1"]
    targets = [t for s in mr.matched_entity_ids for t in s.split(",") if t]
    assert len(targets) == len(set(targets)), "one-to-one violated"
    cand = dict(zip(cp.source1_entity_id, cp.candidate_entity_ids))
    for s1, m in zip(mr.source1_entity_id, mr.matched_entity_ids):
        assert set(filter(None, m.split(","))) <= set(filter(None, cand[s1].split(","))), s1
    one = (mr.matched_entity_ids != "") & (mr.matched_entity_ids.str.count(",") == 0)
    assert (mr.matched_entity_ids == "").any() and one.any() and (mr.matched_entity_ids.str.count(",") >= 1).any()
    t1 = pd.read_csv(world["syn"]["data"] / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    fr = set(t1.entity_id[t1.country == "France"])
    assert fr and fr <= set(mr.source1_entity_id)                                  # unseen country present, no crash


def test_fail_closed_on_pinned_module_mismatch(world, tmp_path):
    bad = dict(world["cfg"])
    bad["code_sha256"] = {**bad["code_sha256"], "src/er/loader.py": "0" * 64}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    r = _run("experiments/agy2_pipeline.py", {**world["penv"], "PIPELINE_CONFIG": tmp_path / "bad.json", "PIPELINE_MODE": "h4",
                                              "OUT_DIR": tmp_path / "o"}, expect_ok=False)
    assert r.returncode != 0 and "FAIL-CLOSED" in (r.stdout + r.stderr)


def test_fail_closed_on_tampered_export(world, tmp_path):
    exp = tmp_path / "export"
    shutil.copytree(world["root"] / "h4" / "export", exp)
    (exp / "decision.json").write_text((exp / "decision.json").read_text() + " ")          # any byte change
    r = _run("experiments/agy2_pipeline.py", {**world["penv"], "PIPELINE_MODE": "test", "MODELS_DIR": exp, "SHARD_COUNTRY": "US",
                                              "SHARD_INDEX": 0, "SHARD_COUNT": 1, "OUT_DIR": tmp_path / "o"}, expect_ok=False)
    assert r.returncode != 0 and "FAIL-CLOSED" in (r.stdout + r.stderr)


def test_merge_fails_closed_on_missing_shard(world, tmp_path):
    part = tmp_path / "part"
    part.mkdir()
    for f in (world["root"] / "shA").glob("shard_*"):
        if "_India_1of2_" not in f.name:
            shutil.copy(f, part / f.name)
    code = ("import os,sys; sys.path.insert(0, %r); from experiments import agy2_pipeline_merge as MG; "
            "MG.main(['--shards', %r, '--models', %r, '--data-root', %r, '--out', %r])"
            % (str(ROOT), str(part), str(world["root"] / "h4" / "export"), str(world["syn"]["data"]), str(tmp_path / "o")))
    e = dict(os.environ, **{k: str(v) for k, v in world["penv"].items()})
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "FAIL-CLOSED" in (r.stdout + r.stderr)


def test_country_encoding_is_deterministic_and_unseen_is_nan():
    from experiments.agy2_pipeline import COUNTRY_CODE, country_code
    assert COUNTRY_CODE == {"India": 0.0, "US": 1.0}
    assert country_code("India") == 0.0 and country_code("US") == 1.0 and np.isnan(country_code("France"))
    # identical to the exp7/exp9 training encoding enumerate(sorted({"India", "US"}))
    assert {c: float(i) for i, c in enumerate(sorted({"India", "US"}))} == COUNTRY_CODE


def test_lightgbm_routes_unseen_country_nan_deterministically():
    """Documents (does not tune) how an unseen country reaches the trees: no NaN in training -> NaN treated as 0."""
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 2000).astype(np.float32)
    X = np.c_[x, rng.random(2000)]
    y = (x == 1).astype(int)
    m = lgb.train({"objective": "binary", "verbose": -1, "seed": 1, "num_threads": 1}, lgb.Dataset(X, y), 20)
    p_nan, p0, p1 = m.predict(np.array([[np.nan, .5], [0, .5], [1, .5]], dtype=np.float32))
    assert p_nan == p0 != p1
