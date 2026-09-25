"""AMLC26 exp8 — content-addressed artifact store for the turbo experimentation path.

Every artifact is a node  STORE/<kind>/<key>/  holding data files plus manifest.json:
  key      = sha256(kind, parent keys, params, code sha) — a pure function of the inputs, so a changed parameter,
             parent or code file yields a new key (cache miss) instead of silently reusing a stale artifact.
  manifest = kind, key, parents, params, code_sha, per-file sha256/bytes/rows, groups (S1 counts), tags, extra.
Guards (enforced on write AND on open, recursively through the parent chain):
  * no sealed S1 id (H3 and any EXP8_SEALED_LISTS) may appear in a node's queries table; group names limited
    to the store's allowed set (dev store: {"dev", "h2"}),
  * a node opens only if every parent exists with the key recorded in its manifest, and every file still has
    the recorded sha256.
Parquet I/O uses pyarrow when present (Kaggle) and duckdb otherwise (local venv).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENV = os.environ.get
FROZEN_CONFIG = Path(ENV("EXP7_CONFIG", str(ROOT / "experiments" / "configs" / "exp7_h3_frozen.json")))
IDS_ROOT = Path(ENV("EXP8_IDS_ROOT", str(ROOT)))          # where exclude_ids/ and expected/ live (Kaggle bundle / stage dir)
STORE_ROOT = Path(ENV("EXP8_STORE", str(ROOT / "artifacts" / "exp8_store")))
DEV_GROUPS = ("dev", "h2")
S1_TABLES = ("s1_open",)                                   # other tables carrying S1 ids (checked like queries)
KINDS = ("v0_seed", "c0_norm", "c1_stats", "c2_retrieval", "c3_pruner", "c4_mfeat", "c5_matcher", "c6_decide")


class StoreError(RuntimeError):
    pass


# ------------------------------------------------------------------ hashing
def sha256_file(p, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def code_sha(*names) -> dict:
    """sha256 of repo files (paths relative to ROOT) — part of every node key."""
    return {n: sha256_file(ROOT / n) for n in sorted(set(names))}


def canon_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def node_key(kind: str, parents: dict, params: dict, code: dict) -> str:
    assert kind in KINDS, kind
    return hashlib.sha256(canon_json({"kind": kind, "parents": parents, "params": params, "code": code}).encode()).hexdigest()[:20]


# ------------------------------------------------------------------ parquet I/O
def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_parquet(df: pd.DataFrame, path) -> None:
    path = Path(path)
    if _has_pyarrow():
        df.to_parquet(path, index=False)
        return
    import duckdb
    con = duckdb.connect()
    con.register("_df", df.reset_index(drop=True))
    con.execute(f"COPY _df TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()


def read_parquet(path, columns=None) -> pd.DataFrame:
    path = Path(path)
    if _has_pyarrow():
        return pd.read_parquet(path, columns=columns)
    import duckdb
    con = duckdb.connect()
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    arrs = con.execute(f"SELECT {cols} FROM read_parquet('{path}')").fetchnumpy()
    con.close()
    out = {}
    for k, v in arrs.items():
        v = np.asarray(v)
        if v.dtype == object:                       # duckdb returns VARCHAR / categorical as object
            out[k] = v
        else:
            out[k] = v
    return pd.DataFrame(out)


def parquet_columns(path) -> list:
    if _has_pyarrow():
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema_arrow.names
    import duckdb
    con = duckdb.connect()
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    con.close()
    return cols


# ------------------------------------------------------------------ frozen groups / sealed ids
@lru_cache(maxsize=None)
def frozen_config() -> dict:
    return json.loads(FROZEN_CONFIG.read_text())


def read_ids(p) -> list:
    ids = [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]
    if len(ids) != len(set(ids)):
        raise StoreError(f"duplicate ids in {p}")
    return ids


def group_ids(g: str) -> list:
    """IDs of a frozen group (dev=H1, h2, h3), sha-verified against the frozen exp7 config (never modified)."""
    spec = frozen_config()["groups"][g]
    p = IDS_ROOT / spec["file"]
    if sha256_file(p) != spec["sha256"]:
        raise StoreError(f"{g} id list {p} differs from the frozen config")
    ids = read_ids(p)
    assert len(ids) == spec["n"]
    return ids


@lru_cache(maxsize=None)
def sealed_ids() -> frozenset:
    """S1 ids that must never enter a development artifact: H3 plus any extra lists in EXP8_SEALED_LISTS."""
    out = set(group_ids("h3"))
    for p in [x for x in ENV("EXP8_SEALED_LISTS", "").split(",") if x]:
        out |= set(read_ids(p))
    return frozenset(out)


def fold_cfg() -> tuple:
    f = frozen_config()["folds"]
    return f["salt"], int(f["nfold"])


def folds_for(ids) -> tuple:
    """(fold, early-stopping flag) per S1 — the exp7 pure-function assignment shared by pruner and matcher."""
    from experiments.agy2_exp7_h3 import fold_of, inner_es_flag
    salt, nfold = fold_cfg()
    return fold_of(ids, salt, nfold), inner_es_flag(ids, salt)


# ------------------------------------------------------------------ store
class Store:
    def __init__(self, root=None, forbidden=None, allowed_groups=DEV_GROUPS):
        self.root = Path(root or STORE_ROOT)
        self.forbidden = sealed_ids() if forbidden is None else frozenset(forbidden)
        self.allowed = tuple(allowed_groups)

    def path(self, kind, key) -> Path:
        return self.root / kind / key

    def exists(self, kind, key) -> bool:
        return (self.path(kind, key) / "manifest.json").is_file()

    def create(self, kind, parents: dict, params: dict, code: dict, tags=()) -> "NodeWriter":
        for name, pk in parents.items():
            pkind, pkey = pk.split("/")
            self.open(pkind, pkey)                         # parent must exist and verify
        key = node_key(kind, parents, params, code)
        return NodeWriter(self, kind, key, parents, params, code, tags)

    def open(self, kind, key, verify_files=True) -> "Node":
        p = self.path(kind, key)
        mf = p / "manifest.json"
        if not mf.is_file():
            raise StoreError(f"missing node {kind}/{key}")
        man = json.loads(mf.read_text())
        if man["kind"] != kind or man["key"] != key:
            raise StoreError(f"manifest identity mismatch at {p}")
        if node_key(kind, man["parents"], man["params"], man["code"]) != key:
            raise StoreError(f"{kind}/{key}: manifest does not hash to its key (tampered)")
        for name, pk in man["parents"].items():
            pkind, pkey = pk.split("/")
            if not self.exists(pkind, pkey):
                raise StoreError(f"{kind}/{key}: parent {name}={pk} missing (parent hash mismatch)")
            self.open(pkind, pkey, verify_files=False)
        if verify_files:
            for fn, meta in man["files"].items():
                fp = p / fn
                if not fp.is_file() or fp.stat().st_size != meta["bytes"] or sha256_file(fp) != meta["sha256"]:
                    raise StoreError(f"{kind}/{key}: file {fn} changed since it was written")
        bad = set(man.get("groups", {})) - set(self.allowed)
        if bad:
            raise StoreError(f"{kind}/{key}: groups {sorted(bad)} are not allowed in this store")
        node = Node(self, kind, key, man)
        if "queries.parquet" in man["files"]:
            self._check_ids(node.table("queries", columns=["s1_id", "group"]), f"{kind}/{key}")
        for t in S1_TABLES:
            if f"{t}.parquet" in man["files"]:
                self._check_ids(node.table(t, columns=["s1_id"]), f"{kind}/{key}", check_groups=False)
        return node

    def find(self, kind) -> list:
        d = self.root / kind
        return sorted(x.name for x in d.iterdir() if (x / "manifest.json").is_file()) if d.is_dir() else []

    def _check_ids(self, q: pd.DataFrame, where: str, check_groups=True):
        bad = set(q.s1_id.tolist()) & self.forbidden
        if bad:
            raise StoreError(f"{where}: {len(bad)} sealed S1 ids present — refusing")
        if not check_groups:
            return
        badg = set(q.group.tolist()) - set(self.allowed)
        if badg:
            raise StoreError(f"{where}: groups {sorted(badg)} not allowed in this store")


class Node:
    def __init__(self, store, kind, key, man):
        self.store, self.kind, self.key, self.man = store, kind, key, man
        self.dir = store.path(kind, key)

    @property
    def ref(self) -> str:
        return f"{self.kind}/{self.key}"

    def table(self, name, columns=None) -> pd.DataFrame:
        return read_parquet(self.dir / f"{name}.parquet", columns=columns)

    def file(self, name) -> Path:
        return self.dir / name

    def parent(self, name) -> "Node":
        pkind, pkey = self.man["parents"][name].split("/")
        return self.store.open(pkind, pkey, verify_files=False)

    def lineage(self, kind) -> "Node":
        """Nearest ancestor (or self) of the given kind."""
        if self.kind == kind:
            return self
        for name in self.man["parents"]:
            try:
                return self.parent(name).lineage(kind)
            except StoreError:
                continue
        raise StoreError(f"{self.ref} has no {kind} ancestor")

    def source(self) -> "Node":
        """The retrieval-level ancestor (c2_retrieval or v0_seed) that owns queries/pool."""
        for k in ("c2_retrieval", "v0_seed"):
            try:
                return self.lineage(k)
            except StoreError:
                pass
        raise StoreError(f"{self.ref} has no retrieval ancestor")


class NodeWriter:
    """Writes into a temp dir; finalize() verifies guards, hashes files and renames atomically into place."""

    def __init__(self, store, kind, key, parents, params, code, tags):
        self.store, self.kind, self.key = store, kind, key
        self.parents, self.params, self.code, self.tags = parents, params, code, list(tags)
        self.tmp = store.root / ".tmp" / f"{kind}-{key}-{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True, exist_ok=False)
        self.rows, self.groups, self.extra = {}, {}, {}
        self.t0 = time.time()

    @property
    def ref(self) -> str:
        return f"{self.kind}/{self.key}"

    def put_table(self, name, df: pd.DataFrame):
        if name == "queries":
            self.store._check_ids(df, self.ref)
            self.groups = {g: int((df.group == g).sum()) for g in sorted(set(df.group))}
        elif "s1_id" in df.columns:
            self.store._check_ids(df, self.ref, check_groups=False)
        write_parquet(df, self.tmp / f"{name}.parquet")
        self.rows[f"{name}.parquet"] = int(len(df))

    def put_file(self, name, src=None, writer=None):
        dst = self.tmp / name
        if src is not None:
            shutil.copyfile(src, dst)
        else:
            writer(dst)

    def put_json(self, name, obj):
        (self.tmp / name).write_text(json.dumps(obj, indent=2, default=_json_default))

    def finalize(self) -> Node:
        files = {}
        for fp in sorted(self.tmp.iterdir()):
            files[fp.name] = {"sha256": sha256_file(fp), "bytes": fp.stat().st_size, "rows": self.rows.get(fp.name)}
        man = {"kind": self.kind, "key": self.key, "parents": self.parents, "params": self.params, "code": self.code,
               "files": files, "groups": self.groups, "tags": self.tags, "extra": self.extra,
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_sec": round(time.time() - self.t0, 2)}
        bad = set(self.groups) - set(self.store.allowed)
        if bad:
            raise StoreError(f"{self.ref}: groups {sorted(bad)} not allowed")
        (self.tmp / "manifest.json").write_text(json.dumps(man, indent=2, default=_json_default))
        dst = self.store.path(self.kind, self.key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(self.tmp)
        else:
            os.replace(self.tmp, dst)
        return self.store.open(self.kind, self.key)

    def abort(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
