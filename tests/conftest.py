import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import reload_config  # noqa: E402


def _cfg(tmp: Path, task_type: str, metric: str, target: str, pred_regex=None) -> dict:
    return {
        "competition": "test",
        "paths": {"train": str(tmp / "train.csv"), "test": str(tmp / "test.csv"),
                  "sample_submission": str(tmp / "sample.csv"), "folds": str(tmp / "folds.csv"),
                  "artifacts": str(tmp / "artifacts")},
        "columns": {"id": "sample_id", "target": target, "text": ["catalog_content"], "image_url": None, "group": None},
        "task": {"type": task_type, "metric": metric, "positive_only": task_type == "regression",
                 "pred_regex": pred_regex},
        "folds": {"n_splits": 3, "scheme": "kfold", "seed": 1, "n_bins": 5, "strat_col": None},
        "submission": {"id_col": "sample_id", "pred_col": target},
        "quick": {"frac": 0.5},
    }


def _activate(tmp: Path, cfg: dict, monkeypatch) -> None:
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("AMLC_CONFIG", str(p))
    monkeypatch.delenv("AMLC_ARTIFACTS", raising=False)
    monkeypatch.delenv("QUICK", raising=False)
    monkeypatch.delenv("OVERWRITE", raising=False)
    reload_config()


@pytest.fixture
def regression_env(tmp_path, monkeypatch):
    import make_synthetic

    make_synthetic.make(tmp_path, n_train=300, n_test=150, seed=3)
    (tmp_path / "sample.csv").write_text((tmp_path / "sample_test_out.csv").read_text())
    _activate(tmp_path, _cfg(tmp_path, "regression", "smape", "price"), monkeypatch)
    yield tmp_path
    monkeypatch.undo()
    reload_config()


@pytest.fixture
def extraction_env(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_tr, n_te = 240, 90
    vals = ["10 gram", "5 ounce", "250 gram", "2 ounce", ""]
    def rows(n, start, with_target):
        out = []
        for i in range(n):
            v = vals[rng.integers(len(vals))]
            r = {"sample_id": str(start + i), "catalog_content": f"product {i} weight {v or 'unknown'}"}
            if with_target:
                r["entity_value"] = v
            out.append(r)
        return pd.DataFrame(out)
    rows(n_tr, 1000, True).to_csv(tmp_path / "train.csv", index=False)
    te = rows(n_te, 5000, False)
    te.to_csv(tmp_path / "test.csv", index=False)
    te[["sample_id"]].head(10).assign(entity_value="").to_csv(tmp_path / "sample.csv", index=False)
    cfg = _cfg(tmp_path, "extraction", "f1_extraction", "entity_value",
               pred_regex=r"^\d+(\.\d+)? (gram|ounce)$")
    _activate(tmp_path, cfg, monkeypatch)
    yield tmp_path
    monkeypatch.undo()
    reload_config()
