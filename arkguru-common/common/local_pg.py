"""Detect a local Postgres + pgvector DSN (native vs Docker) without printing secrets.

Precedence in ``detect_local_dsn``:
    1. Existing ``PG_DSN`` whose host is not loopback (injected hosted URI)
    2. Existing ``PG_DSN`` if it already connects
    3. Docker compose candidate on host port 5433 (``ARKGURU_PG_PORT``)
    4. Native candidates on 5432 (Cloud start_services, DATABASE_SETUP, peer $USER)

CLI (``python -m common.local_pg``) prints ``origin=`` and ``host:port/db`` only.
Pass ``--export-env FILE`` to write ``PG_DSN`` / ``DSN_ORIGIN`` for the shell helper.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from .datastore import _dsn_endpoint, _is_loopback_host, _parse_url_dsn

DOCKER_USER = "rag"
DOCKER_PASSWORD = "change-me"
DOCKER_DB = "rag"
DOCKER_HOST_PORT = 5433
NATIVE_PORT = 5432

ORIGIN_INJECTED = "injected"
ORIGIN_EXISTING = "existing"
ORIGIN_DOCKER = "docker"
ORIGIN_NATIVE = "native"


def docker_dsn(port: Optional[int] = None) -> str:
    host_port = int(os.environ.get("ARKGURU_PG_PORT") or port or DOCKER_HOST_PORT)
    return (
        f"postgresql://{DOCKER_USER}:{DOCKER_PASSWORD}"
        f"@127.0.0.1:{host_port}/{DOCKER_DB}"
    )


def native_candidates() -> list[str]:
    """Well-known native DSNs. Order matches Cloud start_services then docs."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "postgres"
    return [
        "postgresql://arkguru:arkguru@127.0.0.1:5432/arkguru",
        "postgresql://rag:change-me@127.0.0.1:5432/rag",
        f"postgresql://{user}@127.0.0.1:5432/rag",
        f"postgresql://{user}@127.0.0.1:5432/postgres",
    ]


def probe_dsn(dsn: str, timeout: float = 2.0) -> bool:
    """Return True if ``SELECT 1`` succeeds. Missing psycopg is a miss, not a crash."""
    try:
        import psycopg
    except ImportError:
        return False
    try:
        with psycopg.connect(dsn, connect_timeout=max(1, int(timeout))) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() is not None
    except Exception:
        return False


def _existing_dsn() -> Optional[str]:
    value = (os.environ.get("PG_DSN") or "").strip()
    return value or None


def detect_local_dsn(
    *,
    probe=probe_dsn,
) -> tuple[str, str]:
    """Return ``(dsn, origin)``. Raises ``RuntimeError`` if nothing is reachable."""
    current = _existing_dsn()
    if current:
        parsed = _parse_url_dsn(current)
        host = parsed.hostname if parsed is not None else None
        if parsed is not None and not _is_loopback_host(host):
            return current, ORIGIN_INJECTED
        if probe(current):
            return current, ORIGIN_EXISTING

    docker = docker_dsn()
    if probe(docker):
        return docker, ORIGIN_DOCKER

    for candidate in native_candidates():
        if probe(candidate):
            return candidate, ORIGIN_NATIVE

    raise RuntimeError(
        "No local Postgres answered. Start native on 5432 "
        "(Homebrew/apt or arkguru-common/scripts/start_services.sh) "
        "or Docker on 5433: docker compose up -d. "
        "Then re-run bash scripts/detect_local_pg.sh"
    )


def write_export_env(path: Path, dsn: str, origin: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"DSN_ORIGIN={origin}\n")
        fh.write(f"PG_DSN={dsn}\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-env",
        metavar="FILE",
        help="Write PG_DSN and DSN_ORIGIN to FILE (mode 0600). Not printed.",
    )
    args = parser.parse_args(argv)
    try:
        dsn, origin = detect_local_dsn()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"origin={origin} endpoint={_dsn_endpoint(dsn)}")
    if args.export_env:
        write_export_env(Path(args.export_env), dsn, origin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
