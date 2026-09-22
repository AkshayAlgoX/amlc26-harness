"""Single place that reads config.yaml. Everything else imports from here.

Environment overrides:
  AMLC_CONFIG     path to an alternative config (e.g. config.synthetic.yaml)
  AMLC_ARTIFACTS  where oof/test/logs live (e.g. a mounted shared Drive folder)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg_path = _resolve(os.environ.get("AMLC_CONFIG", "config.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def reload_config() -> dict:
    load_config.cache_clear()
    return load_config()


def data_path(key: str) -> Path:
    return _resolve(load_config()["paths"][key])


def artifacts_dir() -> Path:
    return _resolve(os.environ.get("AMLC_ARTIFACTS") or load_config()["paths"]["artifacts"])


def col(key: str):
    return load_config()["columns"].get(key)


def task() -> dict:
    return load_config()["task"]


def text_cols() -> list[str]:
    cols = col("text") or []
    return [cols] if isinstance(cols, str) else list(cols)
