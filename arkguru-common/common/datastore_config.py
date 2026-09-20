"""
Loader + factory for the datastore -- keeps datastore config OUT of code.

Everything about *where* chunks and vectors live is declared in one file
(config/datastore.yaml). Code never hardcodes a DSN, table name, or dimension;
it calls `open_chunk_store()` and gets a ready ChunkStore. Switching database or
tables is a one-file edit; switching the connection is a single env var.

Precedence (highest first):
    explicit kwargs  >  environment variables  >  datastore.yaml  >  defaults

Env overrides (handy for CI / prod without editing files):
    DATASTORE_CONFIG   path to the yaml (default: config/datastore.yaml)
    DATASTORE_BACKEND  postgres | file | local
    PG_DSN (or whatever `dsn_env` names)  the connection string

``resolve_dsn`` also loads a stdlib ``.env`` from the cwd or a parent
directory (stops at a git root) so ``cp .env.example .env`` works without
python-dotenv. Existing environment variables are not overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_PATH = "config/datastore.yaml"
_MAX_DOTENV_PARENTS = 8

_DEFAULTS: dict[str, Any] = {
    "backend": "file",
    "postgres": {"dsn_env": "PG_DSN", "dsn": None,
                 "chunks_table": "chunks", "vectors_table": "chunk_embeddings",
                 "dim": 1024, "hnsw_ef_search": None, "schema": None},
    "file": {"out_dir": "data/processed", "out_format": "jsonl"},
    "local": {"path": "data/store/index"},
}


def load_datastore_config(path: Optional[str] = None) -> dict:
    """Read config/datastore.yaml (if present), merged over defaults, with env
    overrides for backend."""
    path = path or os.environ.get("DATASTORE_CONFIG", DEFAULT_PATH)
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}
    p = Path(path)
    if p.exists():
        import yaml
        loaded = (yaml.safe_load(p.read_text()) or {}).get("datastore", {}) or {}
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update({kk: vv for kk, vv in v.items() if vv is not None})
            elif v is not None:
                cfg[k] = v
    if os.environ.get("DATASTORE_BACKEND"):
        cfg["backend"] = os.environ["DATASTORE_BACKEND"]
    return cfg


def find_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """Return the first ``.env`` in ``start`` (default: cwd) or its parents.

    Stops at a git root (``.git`` present) so a repo without ``.env`` does
    not inherit ``~/.env``.
    """
    cur = (start or Path.cwd()).resolve()
    for _ in range(_MAX_DOTENV_PARENTS):
        candidate = cur / ".env"
        if candidate.is_file():
            return candidate
        if (cur / ".git").exists() or cur.parent == cur:
            break
        cur = cur.parent
    return None


def apply_dotenv(path: Path, *, override: bool = False) -> None:
    """Load ``KEY=VALUE`` lines into ``os.environ`` (stdlib; no python-dotenv).

    Existing variables are left unchanged unless ``override`` is true.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def load_dotenv(start: Optional[Path] = None, *, override: bool = False) -> Optional[Path]:
    """Find and apply a ``.env`` file. Returns the path loaded, if any."""
    found = find_dotenv(start)
    if found is not None:
        apply_dotenv(found, override=override)
    return found


def resolve_dsn(pg: dict, *, dotenv_start: Optional[Path] = None) -> Optional[str]:
    """DSN from explicit yaml value, else from the named env var (after ``.env``)."""
    load_dotenv(dotenv_start)
    if pg.get("dsn"):
        return pg["dsn"]
    value = os.environ.get(pg.get("dsn_env", "PG_DSN"))
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def open_chunk_store(path: Optional[str] = None, **overrides):
    """Construct a ChunkStore from datastore.yaml (+ env + kwargs). Does not
    connect until you call a method on it."""
    from .datastore import ChunkStore
    cfg = load_datastore_config(path)
    pg = dict(cfg.get("postgres", {}))
    pg.update({k: v for k, v in overrides.items() if v is not None})
    dsn = pg.get("dsn") or resolve_dsn(pg)
    return ChunkStore(
        dsn=dsn,
        table=pg.get("chunks_table", "chunks"),
        vectors_table=pg.get("vectors_table", "chunk_embeddings"),
        dim=int(pg.get("dim", 1024)),
        hnsw_ef_search=pg.get("hnsw_ef_search"),
        schema=pg.get("schema"),
    )
