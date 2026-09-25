"""Hardening invariants (synthetic data only): H4<->test feature parity on identical records, B2nc drops `country`
from BOTH stages, unseen-country safety, fail-closed guards, freeze evidence gate, streamed validation equivalence,
dev mode + romanization lane, pruner-from-pool reproduction, kernel generation."""
from __future__ import annotations

import json
import os
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
from experiments.agy2_exp7_prep_h3 import sha256_file  # noqa: E402
from experiments.agy2_pipeline_freeze import build_pipeline_config, dev_config  # noqa: E402

RETR = {"pool_channels": ["A", "B", "F200", "R5", "B2C", "B3C", "B4C"], "max_df": 1.0, "rev_df_cap": 100000,
        "k_fwd": 50, "chunk": 1500, "cap": 100000, "fwd_min_sim": 0.05, "canonical_view": False}
FAKE9 = {"variants": {"B2": {"macro_f05": 0.9394136402248061, "rule": "expF", "params": {"m": 0.0, "s": 0.5}}}}
ORGANISER = ROOT / "third_party" / "organiser" / "validate_submission.py"


def run(script, env, expect_ok=True):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **{k: str(v) for k, v in env.items()})
    r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=e, capture_output=True, text=True, timeout=900)
    if expect_ok and r.returncode != 0:
        raise AssertionError(f"{script}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def rd(p: Path) -> pd.DataFrame:
    return ST.read_parquet(p) if p.is_file() else pd.read_pickle(p.with_suffix(".pkl"))


def cfg_for(syn, variant="B2", stats_exclude=None):
    c = build_pipeline_config(Path("/"), syn["files"], FAKE9, "smoke", variant=variant, smoke=True, retrieval=RETR,
                              budget={"top_n": 10, "floor": 0.0}, h3_n=syn["h3_n"])
    if stats_exclude is not None:
        c["stats_exclude"] = stats_exclude
    for m in ("pruner", "matcher"):
        c[m]["max_rounds"] = 120
    return c


@pytest.fixture(scope="module")
def pw(tmp_path_factory):
    """Parity world: test files are byte copies of the train files; statistics exclude nothing, so h4 mode and test mode
    see identical statistics for identical records."""
    root = tmp_path_factory.mktemp("parity")
    syn = pipeline_synth.make(root, n_train=1500, translit=40, train_as_test=True)
    cfg = cfg_for(syn, stats_exclude=[])
    (root / "cfg.json").write_text(json.dumps(cfg))
    env = {"PIPELINE_SMOKE": "1", "PIPELINE_CONFIG": root / "cfg.json", "DATA_ROOT": syn["data"]}
    run("experiments/agy2_pipeline.py", {**env, "PIPELINE_MODE": "h4", "OUT_DIR": root / "h4"})
    run("experiments/agy2_pipeline.py", {**env, "PIPELINE_MODE": "test", "MODELS_DIR": root / "h4" / "export", "SAVE_FEATURES": "1",
                                         "SHARD_COUNTRY": "India", "SHARD_INDEX": 0, "SHARD_COUNT": 1, "OUT_DIR": root / "t"})
    return {"root": root, "syn": syn, "cfg": cfg, "env": env}


@pytest.fixture(scope="module")
def nw(tmp_path_factory):
    """B2nc world with a real France test split."""
    root = tmp_path_factory.mktemp("b2nc")
    syn = pipeline_synth.make(root, n_train=1500, n_test=600)
    cfg = cfg_for(syn, variant="B2nc")
    (root / "cfg.json").write_text(json.dumps(cfg))
    env = {"PIPELINE_SMOKE": "1", "PIPELINE_CONFIG": root / "cfg.json", "DATA_ROOT": syn["data"]}
    run("experiments/agy2_pipeline.py", {**env, "PIPELINE_MODE": "h4", "OUT_DIR": root / "h4"})
    run("experiments/agy2_pipeline.py", {**env, "PIPELINE_MODE": "test", "MODELS_DIR": root / "h4" / "export",
                                         "SHARD_COUNTRY": "France", "SHARD_INDEX": 0, "SHARD_COUNT": 1, "OUT_DIR": root / "t"})
    return {"root": root, "syn": syn, "cfg": cfg, "env": env}


def test_h4_and_test_mode_produce_identical_features_and_scores(pw):
    root = pw["root"]
    h4 = rd(root / "h4" / "artifacts" / "c4c5_matcher_rows.parquet")
    feats = pw["cfg"]["matcher"]["features"]
    t1 = pd.read_csv(pw["syn"]["data"] / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    tf = rd(next((root / "t").glob("shard_*_features.parquet")))
    tf["s1_id"] = t1.entity_id.values[tf.s1_pos.values]
    india = set(t1.entity_id[t1.country == "India"])
    a = h4[h4.s1_id.isin(india)].set_index(["s1_id", "code"]).sort_index()
    b = tf[tf.s1_id.isin(set(a.index.get_level_values(0)))].set_index(["s1_id", "code"]).sort_index()
    assert len(a) > 0 and a.index.equals(b.index), "different candidate rows for the same S1"
    assert len(feats) == 61
    for f in feats + ["p"]:
        x, y = a[f].to_numpy(np.float64), b[f].to_numpy(np.float64)
        assert np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]), f
    assert (a.country == 0.0).all()                                   # India keeps its training code


def test_b2nc_drops_country_from_both_stages_and_france_runs(nw):
    cfg = nw["cfg"]
    assert cfg["drop_features"] == ["country"]
    assert "country" not in cfg["pruner"]["features"] and "country" not in cfg["matcher"]["features"]
    dec = json.loads((nw["root"] / "h4" / "export" / "decision.json").read_text())
    assert "country" not in dec["pruner_features"] and "country" not in dec["matcher_features"]
    assert dec["candidate"] == "B2nc"
    import lightgbm as lgb
    for tag, n in (("pruner", len(dec["pruner_features"])), ("matcher", len(dec["matcher_features"]))):
        assert lgb.Booster(model_file=str(nw["root"] / "h4" / "export" / f"{tag}_fold0.txt")).num_feature() == n
    meta = json.loads(next((nw["root"] / "t").glob("shard_France_*_meta.json")).read_text())
    assert meta["s1"] > 0 and meta["pairs"] > 0


def test_standard_b2_keeps_country_and_france_never_shifts_codes():
    from experiments.agy2_pipeline import COUNTRY_CODE, country_code
    codes = {c: country_code(c) for c in ("US", "India", "France", "Germany")}
    assert codes["India"] == 0.0 and codes["US"] == 1.0 and np.isnan(codes["France"]) and np.isnan(codes["Germany"])
    assert COUNTRY_CODE == {"India": 0.0, "US": 1.0}


def test_fail_closed_on_wrong_fold_feature_list_and_export_mode(pw, tmp_path):
    for mutate in ({"folds": {**pw["cfg"]["folds"], "salt": "other"}},
                   {"pruner": {**pw["cfg"]["pruner"], "features": list(reversed(pw["cfg"]["pruner"]["features"]))}}):
        bad = {**pw["cfg"], **mutate}
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        r = run("experiments/agy2_pipeline.py", {**pw["env"], "PIPELINE_CONFIG": tmp_path / "bad.json", "PIPELINE_MODE": "h4",
                                                 "OUT_DIR": tmp_path / "o"}, expect_ok=False)
        assert r.returncode != 0 and "FAIL-CLOSED" in r.stdout + r.stderr


def test_freeze_gate_requires_exp10_evidence():
    from experiments.agy2_pipeline_freeze import FILES
    with pytest.raises(SystemExit):
        build_pipeline_config(ROOT, FILES, FAKE9, "x", variant="B2nc")
    with pytest.raises(SystemExit):
        build_pipeline_config(ROOT, FILES, FAKE9, "x", top_n=40, exp10={"anchor": {"b2_reproduced": False}, "decisions": {}})
    ok = {"anchor": {"b2_reproduced": True}, "decisions": {"country_agnostic_for_h4": True, "top40_budget_for_h4": True}}
    c = build_pipeline_config(ROOT, FILES, FAKE9, "x", variant="B2nc", top_n=40, exp10=ok, exp10_sha="s")
    assert c["candidate"]["name"] == "B2nc" and c["budget"]["top_n"] == 40 and c["candidate"]["exp10_evidence"]
    assert c["set_rules"]["thr_t"] == [] and c["set_rules"]["expF_s"]


def _tiny(tmp):
    d = tmp / "test"
    d.mkdir()
    (d / "test_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\ta\tb\tUS\nS1-2\tc\td\tFrance\n")
    (d / "test_source2.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\ta\tb\tUS\nS2-2\tc\td\tFrance\n")
    (d / "test_source3.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-1\ta\tb\tUS\n")
    return d


@pytest.mark.parametrize("case", ["valid", "missing_s1", "dup_s1", "unknown_id", "s1_prefix", "dup_in_list", "match_not_candidate"])
def test_stream_validator_agrees_with_repo_and_organiser(tmp_path, case):
    from experiments.agy2_pipeline_merge import stream_validate
    from src.er.validator import validate_submission
    d = _tiny(tmp_path)
    m = {"S1-1": "S2-1,S3-1", "S1-2": ""}
    c = {"S1-1": "S2-1,S3-1,S2-2", "S1-2": "S2-2"}
    rows = ["S1-1", "S1-2"]
    if case == "missing_s1":
        rows = ["S1-1"]
    if case == "dup_s1":
        rows = ["S1-1", "S1-1", "S1-2"]
    if case == "unknown_id":
        m["S1-1"] = c["S1-1"] = "S2-99"
    if case == "s1_prefix":
        m["S1-1"] = "S1-2"; c["S1-1"] = "S1-2"
    if case == "dup_in_list":
        m["S1-1"] = "S2-1,S2-1"
    if case == "match_not_candidate":
        m["S1-2"] = "S2-1"
    mp, cp = tmp_path / "m.tsv", tmp_path / "c.tsv"
    mp.write_text("source1_entity_id\tmatched_entity_ids\n" + "".join(f"{s}\t{m[s]}\n" for s in rows))
    cp.write_text("source1_entity_id\tcandidate_entity_ids\n" + "".join(f"{s}\t{c[s]}\n" for s in rows))
    ours = stream_validate(mp, cp, d)["valid"]
    repo = validate_submission(mp, cp, test_dir=d, check_ids=True).is_valid
    org = subprocess.run([sys.executable, str(ORGANISER), "--matching", str(mp), "--candidate", str(cp), "--test-dir", str(d),
                          "--check-ids"],
                         capture_output=True, text=True).returncode == 0
    if case == "match_not_candidate":                 # organiser/repo only WARN on this; we fail closed (stricter)
        assert repo and org and not ours
    else:
        assert ours == repo == org == (case == "valid"), (case, ours, repo, org)


def test_dev_mode_with_romanization_lane(pw, tmp_path):
    root = pw["root"]
    assert "ground_truth" not in (ROOT / "experiments/agy2_lane_romanization.py").read_text().split('"""', 2)[2]
    run("experiments/agy2_lane_romanization.py", {"LANE_CONFIG": root / "cfg.json", "LANE_IDS_ROOT": "/",
                                                    "DATA_ROOT": pw["syn"]["data"], "OUT_DIR": tmp_path / "lane"})
    man = json.loads((tmp_path / "lane" / "lane_translit_manifest.json").read_text())
    assert man["pairs"] > 0 and man["countries"]["India"]["non_ascii_targets"] >= 40
    lane = {"name": "translit", "path": str(tmp_path / "lane" / "lane_translit.parquet"), "sha256": man["sha256"]}
    dc = dev_config(pw["cfg"], lanes=[lane])
    (tmp_path / "dev.json").write_text(json.dumps(dc))
    run("experiments/agy2_pipeline.py", {**pw["env"], "PIPELINE_CONFIG": tmp_path / "dev.json", "PIPELINE_MODE": "dev",
                                         "OUT_DIR": tmp_path / "dev"})
    res = json.loads((tmp_path / "dev" / "pipeline_dev_results.json").read_text())
    assert set(res["reports"]) == {"dev"}                             # H4 never scored in dev mode
    dec = json.loads((tmp_path / "dev" / "export" / "decision.json").read_text())
    assert dec["lanes"] == ["translit"] and "lane_translit_score" in dec["matcher_features"]
    assert json.loads((tmp_path / "dev" / "export" / "manifest.json").read_text())["mode"] == "dev"
    r = run("experiments/agy2_pipeline.py", {**pw["env"], "PIPELINE_CONFIG": tmp_path / "dev.json", "PIPELINE_MODE": "h4",
                                             "OUT_DIR": tmp_path / "x"}, expect_ok=False)
    assert r.returncode != 0 and "FAIL-CLOSED" in r.stdout + r.stderr   # lanes are dev-only
    r = run("experiments/agy2_pipeline.py", {**pw["env"], "PIPELINE_MODE": "test", "MODELS_DIR": tmp_path / "dev" / "export",
                                             "SHARD_COUNTRY": "US", "SHARD_INDEX": 0, "SHARD_COUNT": 1, "OUT_DIR": tmp_path / "y"},
            expect_ok=False)
    assert r.returncode != 0 and "FAIL-CLOSED" in r.stdout + r.stderr   # test inference needs the H4 export


def test_artifacts_c0_c1_saved_without_label_leak(pw):
    art = pw["root"] / "h4" / "artifacts"
    for f in ("c0_s1_texts", "c0_target_texts", "c1_name_df"):
        assert (art / f"{f}.parquet").is_file() or (art / f"{f}.pkl").is_file(), f
    assert list(art.glob("c1_stats_*.pkl"))
    pool = rd(art / "c2c3_pool_prebudget.parquet")
    tt = rd(art / "c0_target_texts.parquet")
    assert set(tt.code) <= set(pool.code)                             # no truth-only targets persisted


def test_pruner_from_pool_reproduces_stored_scores(pw, tmp_path):
    run("experiments/agy2_pruner_from_pool.py", {"POOL": pw["root"] / "h4" / "artifacts" / "c2c3_pool_prebudget.parquet",
                                                  "CONFIG": pw["root"] / "cfg.json", "OUT_DIR": tmp_path})
    r = json.loads((tmp_path / "pruner_from_pool.json").read_text())
    assert r["base"]["max_abs_diff_vs_stored_prune_p"] <= 1e-9


def test_make_kernels_builds_fail_closed_chain(tmp_path):
    """In a git checkout: build the real bundle + kernels from HEAD. In an exported bundle (git archive, no .git, e.g.
    inside a Kaggle kernel): validate the bundled commit manifest and the kernel generator without git."""
    sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
    import make_kernels as MK
    if not (ROOT / ".git").exists():
        plan = json.loads((ROOT / "experiments/configs/test_shards.json").read_text())
        total_shards = sum(plan["shards"].values())
        commit = (ROOT / "BUNDLE_COMMIT").read_text().strip()
        assert len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit), "malformed BUNDLE_COMMIT"
        assert not [p for p in ROOT.rglob("*") if p.is_symlink()], "symlink inside the exported bundle"
        cfg = json.loads((ROOT / "experiments/configs/h4_frozen.json").read_text())
        assert all(sha256_file(ROOT / m) == h for m, h in cfg["code_sha256"].items()), "pinned module mismatch in bundle"
        job = {"bundle_commit": commit, "config_sha256": sha256_file(ROOT / "experiments/configs/h4_frozen.json"),
               "kind": "test_shards", "shards": {"country": "India", "count": plan["shards"]["India"], "indices": [0]}}
        kid = MK.kernel(tmp_path, "test_India_0", "amlc26-p-test-india-0", job, "akshayalgox/amlc26-agy2-test", ["amlc26-p-h4"])
        meta = json.loads((tmp_path / "test_India_0" / "kernel-metadata.json").read_text())
        assert kid.endswith("amlc26-p-test-india-0") and not meta["enable_internet"] and not meta["enable_gpu"]
        stub = (tmp_path / "test_India_0" / "run.py").read_text()
        assert '"kind": "test_shards"' in stub and f'"count": {plan["shards"]["India"]}' in stub and commit in stub
        return
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "cat-file", "-e", f"{head}:scripts/kaggle/kaggle_job.py"], cwd=ROOT).returncode != 0:
        pytest.skip("kernel generator not committed yet (archives the committed tree)")
    subprocess.run([sys.executable, str(ROOT / "scripts/kaggle/make_kernels.py"), "--commit", head,
                    "--bundle-slug", "akshayalgox/amlc26-agy2-test", "--out", str(tmp_path)], check=True, cwd=ROOT)
    stage = tmp_path / "bundle_stage" / "amlc26-agy2"
    assert (stage / "BUNDLE_COMMIT").read_text().strip() == head
    assert not [p for p in stage.rglob("*") if p.is_symlink()], "symlink exported into the bundle"
    assert (stage / "experiments/configs/h4_frozen.json").is_file() and not (stage / "artifacts").exists()
    # read the plan from the archived bundle — that is what make_kernels.py actually used
    plan = json.loads((stage / "experiments/configs/test_shards.json").read_text())
    total_shards = sum(plan["shards"].values())
    metas = {d.name: json.loads((d / "kernel-metadata.json").read_text()) for d in tmp_path.iterdir() if (d / "run.py").is_file()}
    # expect h4 + one per shard + merge
    expected_dirs = {"h4", "merge"}
    for c, n in plan["shards"].items():
        for i in range(n):
            expected_dirs.add(f"test_{c}_{i}")
    assert set(metas) == expected_dirs, f"expected {sorted(expected_dirs)}, got {sorted(metas)}"
    assert all(not m["enable_internet"] and not m["enable_gpu"] for m in metas.values())
    # merge sources: h4 + all test kernels
    assert metas["merge"]["kernel_sources"][0].endswith("amlc26-p-h4")
    assert len(metas["merge"]["kernel_sources"]) == 1 + total_shards
    # Kaggle hard limit: merge has 2 dataset_sources + (1 + total_shards) kernel_sources <= 20
    assert 2 + 1 + total_shards <= 20, f"merge would need {2 + 1 + total_shards} sources (max 20)"
    # validate a sample shard stub
    stub = (tmp_path / "test_India_0" / "run.py").read_text()
    assert '"kind": "test_shards"' in stub and f'"count": {plan["shards"]["India"]}' in stub and head in stub
    assert '"indices": [0]' in stub


def test_extra_train_gate_and_disjoint_draw(tmp_path):
    from experiments.agy2_pipeline_freeze import FILES
    with pytest.raises(SystemExit):                                   # no exp10 evidence
        build_pipeline_config(ROOT, FILES, FAKE9, "x", extra_train=50000)
    no = {"anchor": {"b2_reproduced": True}, "decisions": {"learning_curve": {"more_training_s1_run_justified": False}}}
    with pytest.raises(SystemExit):
        build_pipeline_config(ROOT, FILES, FAKE9, "x", extra_train=50000, exp10=no)
    ok = {"anchor": {"b2_reproduced": True}, "decisions": {"learning_curve": {"more_training_s1_run_justified": True}}}
    with pytest.raises(SystemExit):                                   # above the memory envelope
        build_pipeline_config(ROOT, FILES, FAKE9, "x", extra_train=50001, exp10=ok)
    c = build_pipeline_config(ROOT, FILES, FAKE9, "x", extra_train=50000, exp10=ok, exp10_sha="s")
    assert c["extra_dev"] == {"n": 50000, "salt": "amlc26-extra-train-v1"} and c["candidate"]["name"] == "B2"
    assert c["candidate"]["exp10_evidence"] == {"more_training_s1_run_justified": True}
    # synthetic h4 run: extra S1 join the training group only, disjoint from H0-H4, deterministic
    syn = pipeline_synth.make(tmp_path, n_train=3000, n_test=300)
    cfg = cfg_for(syn)
    cfg["extra_dev"] = {"n": 300, "salt": "amlc26-extra-train-v1"}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    run("experiments/agy2_pipeline.py", {"PIPELINE_SMOKE": "1", "PIPELINE_CONFIG": tmp_path / "cfg.json",
                                         "DATA_ROOT": syn["data"], "PIPELINE_MODE": "h4", "OUT_DIR": tmp_path / "h4"})
    res = json.loads((tmp_path / "h4" / "pipeline_h4_results.json").read_text())
    h1 = len(Path(syn["files"]["H1"]).read_text().split())
    assert res["meta"]["extra_train_s1"] == 300 and res["meta"]["dev_s1"] == h1 + 300
    assert res["meta"]["h4_s1"] == len(Path(syn["files"]["H4"]).read_text().split())
    from experiments.agy2_exp7_prep_h3 import salted_holdout
    s1 = pd.read_csv(syn["data"] / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).entity_id.tolist()
    held = set().union(*(set(Path(syn["files"][k]).read_text().split()) for k in ("H0", "H1", "H2", "H3", "H4")))
    extra = salted_holdout(s1, held, 300, "amlc26-extra-train-v1")
    assert not (set(extra) & held)
    import hashlib as _h
    assert res["meta"]["extra_train_sha256"] == _h.sha256("\n".join(extra).encode()).hexdigest()
    mf = rd(tmp_path / "h4" / "artifacts" / "c4c5_matcher_rows.parquet")
    assert set(mf[mf.g == 1].s1_id) <= set(Path(syn["files"]["H4"]).read_text().split())   # extra never in H4 group
    assert set(extra) & set(mf[mf.g == 0].s1_id)                                         # extra used for training
