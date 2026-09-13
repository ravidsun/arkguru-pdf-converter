"""Unit tests for local native vs Docker DSN detection (no live Postgres)."""
from __future__ import annotations

from pathlib import Path

import pytest

from common.datastore import _dsn_endpoint
from common.local_pg import (
    ORIGIN_DOCKER,
    ORIGIN_EXISTING,
    ORIGIN_INJECTED,
    ORIGIN_NATIVE,
    detect_local_dsn,
    docker_dsn,
    main,
    native_candidates,
    write_export_env,
)


def test_docker_dsn_default_port(monkeypatch):
    monkeypatch.delenv("ARKGURU_PG_PORT", raising=False)
    assert docker_dsn() == "postgresql://rag:change-me@127.0.0.1:5433/rag"


def test_docker_dsn_honors_arkguru_pg_port(monkeypatch):
    monkeypatch.setenv("ARKGURU_PG_PORT", "55432")
    assert docker_dsn().endswith(":55432/rag")


def test_detect_keeps_injected_remote_without_probe(monkeypatch):
    remote = (
        "postgresql://postgres.ref:s3cret@aws-0-us-east-1.pooler.supabase.com"
        ":5432/postgres?sslmode=require"
    )
    monkeypatch.setenv("PG_DSN", remote)

    def boom(_dsn: str) -> bool:
        raise AssertionError("must not probe a remote injected DSN")

    dsn, origin = detect_local_dsn(probe=boom)
    assert dsn == remote
    assert origin == ORIGIN_INJECTED


def test_detect_keeps_existing_loopback_if_it_connects(monkeypatch):
    current = "postgresql://me:pw@127.0.0.1:5432/mine"
    monkeypatch.setenv("PG_DSN", current)
    dsn, origin = detect_local_dsn(probe=lambda candidate: candidate == current)
    assert dsn == current
    assert origin == ORIGIN_EXISTING


def test_detect_prefers_docker_when_5433_answers(monkeypatch):
    monkeypatch.delenv("PG_DSN", raising=False)
    monkeypatch.delenv("ARKGURU_PG_PORT", raising=False)
    docker = docker_dsn()

    def probe(candidate: str) -> bool:
        return candidate == docker

    dsn, origin = detect_local_dsn(probe=probe)
    assert dsn == docker
    assert origin == ORIGIN_DOCKER


def test_detect_falls_back_to_native_when_only_5432_answers(monkeypatch):
    monkeypatch.delenv("PG_DSN", raising=False)
    monkeypatch.delenv("ARKGURU_PG_PORT", raising=False)
    native = native_candidates()[0]

    def probe(candidate: str) -> bool:
        return candidate == native

    dsn, origin = detect_local_dsn(probe=probe)
    assert dsn == native
    assert origin == ORIGIN_NATIVE


def test_detect_raises_when_nothing_answers(monkeypatch):
    monkeypatch.delenv("PG_DSN", raising=False)
    with pytest.raises(RuntimeError, match="No local Postgres"):
        detect_local_dsn(probe=lambda _dsn: False)


def test_cli_stdout_has_no_password(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.delenv("PG_DSN", raising=False)
    monkeypatch.delenv("ARKGURU_PG_PORT", raising=False)
    docker = docker_dsn()
    monkeypatch.setattr(
        "common.local_pg.detect_local_dsn",
        lambda: (docker, ORIGIN_DOCKER),
    )
    export = tmp_path / "pg.env"
    assert main(["--export-env", str(export)]) == 0
    out = capsys.readouterr().out
    assert "origin=docker" in out
    assert "127.0.0.1:5433/rag" in out
    assert "change-me" not in out
    assert docker not in out
    text = export.read_text(encoding="utf-8")
    assert f"PG_DSN={docker}" in text
    assert "DSN_ORIGIN=docker" in text
    assert stat_is_private(export)


def test_cli_failure_mentions_both_start_paths(monkeypatch, capsys):
    monkeypatch.setattr(
        "common.local_pg.detect_local_dsn",
        lambda: (_ for _ in ()).throw(RuntimeError("No local Postgres answered. x")),
    )
    assert main([]) == 1
    err = capsys.readouterr().err
    assert "No local Postgres" in err


def test_write_export_env_mode(tmp_path: Path):
    path = tmp_path / "nested" / ".env"
    write_export_env(path, "postgresql://u:p@127.0.0.1:5432/db", "native")
    assert "PG_DSN=postgresql://u:p@127.0.0.1:5432/db" in path.read_text()
    assert stat_is_private(path)


def test_dsn_endpoint_used_by_cli_omits_password():
    assert "pw" not in _dsn_endpoint("postgresql://u:pw@127.0.0.1:5433/rag")


def stat_is_private(path: Path) -> bool:
    return (path.stat().st_mode & 0o077) == 0


def test_existing_loopback_that_fails_falls_through(monkeypatch):
    monkeypatch.setenv("PG_DSN", "postgresql://dead:x@127.0.0.1:9/none")
    docker = docker_dsn()
    dsn, origin = detect_local_dsn(probe=lambda candidate: candidate == docker)
    assert origin == ORIGIN_DOCKER
    assert dsn == docker
