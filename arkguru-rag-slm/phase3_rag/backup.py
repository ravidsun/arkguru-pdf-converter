"""
Daily custom-format dump of chunks + chunk_embeddings when the watermark moved.

  python -m phase3_rag.backup --once
  python -m phase3_rag.backup --interval 86400

Restore:
  pg_restore --clean --if-exists -d "$PG_DSN" data/backups/<file>.dump
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common.datastore_config import load_datastore_config, open_chunk_store, resolve_dsn
from common.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.backup")

DEFAULT_OUT_DIR = Path("data/backups")
KEEP_LAST = 7
WATERMARK_NAME = "last_ok.json"


def should_skip(current: str | None, previous: str | None) -> bool:
    """True when there is nothing new to dump (empty DB or unchanged watermark)."""
    if not current or current.startswith("-infinity"):
        return True
    return previous is not None and current == previous


def _require_pg_dump() -> str:
    exe = shutil.which("pg_dump")
    if not exe:
        raise RuntimeError(
            "pg_dump not found on PATH. Install PostgreSQL client tools "
            "and retry; backups are fail-closed (no silent skip)."
        )
    return exe


def _read_last_ok(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _prune(out_dir: Path, keep_last: int) -> None:
    dumps = sorted(out_dir.glob("arkguru-*.dump"), key=lambda p: p.name, reverse=True)
    for stale in dumps[keep_last:]:
        try:
            stale.unlink()
            log.info("removed old dump %s", stale.name)
        except OSError as e:
            log.warning("could not remove %s: %s", stale, e)


def run_backup(
    *,
    datastore_config: str | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    keep_last: int = KEEP_LAST,
) -> str:
    dsn = resolve_dsn(load_datastore_config(datastore_config).get("postgres", {}))
    if not dsn:
        raise RuntimeError("No Postgres DSN (set PG_DSN or datastore.yaml). Backup is fail-closed.")
    pg_dump = _require_pg_dump()
    store = open_chunk_store(datastore_config)
    watermark = store.created_at_watermark()
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / WATERMARK_NAME
    last = _read_last_ok(last_path)
    previous = last.get("watermark") if last else None
    if should_skip(watermark, previous):
        log.info("skipped: no changes")
        return "skipped: no changes"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dump_path = out_dir / f"arkguru-{stamp}.dump"
    cmd = [
        pg_dump,
        "-Fc",
        f"--dbname={dsn}",
        f"--table={store.chunks}",
        f"--table={store.vectors}",
        f"--file={dump_path}",
    ]
    log.info("dumping %s and %s -> %s", store.chunks, store.vectors, dump_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if dump_path.exists():
            dump_path.unlink(missing_ok=True)
        err = (result.stderr or result.stdout or "").strip() or f"exit {result.returncode}"
        raise RuntimeError(f"pg_dump failed: {err}")

    payload = {
        "watermark": watermark,
        "path": str(dump_path),
        "taken_at": datetime.now(timezone.utc).isoformat(),
    }
    last_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _prune(out_dir, keep_last)
    return f"wrote {dump_path.name}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Dump chunks + embeddings when the DB watermark moves")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--keep-last", type=int, default=KEEP_LAST)
    ap.add_argument("--interval", type=float, default=86400.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)

    def cycle():
        return run_backup(
            datastore_config=a.datastore_config,
            out_dir=Path(a.out_dir),
            keep_last=a.keep_last,
        )

    if a.once:
        print(cycle())
        return
    Worker("backup", cycle, interval=a.interval).run_forever()


if __name__ == "__main__":
    main()
