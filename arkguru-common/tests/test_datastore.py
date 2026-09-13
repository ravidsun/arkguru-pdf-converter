"""Unit tests for ChunkStore write/iterate contracts (no live Postgres)."""
from __future__ import annotations

from common.datastore import (
    ChunkStore,
    _dsn_endpoint,
    _dsn_with_defaults,
    _ensure_search_path_has,
    _ensure_vector_extension,
    _migrate_chunk_columns,
    _redact_secret,
)
from common.schema import Chunk


class _FakeCursor:
    def __init__(self, *, name=None, rows=()):
        self.name = name
        self.itersize = None
        self.sql = None
        self.sqls: list[str] = []
        self._rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql = sql
        self.sqls.append(sql)

    def executemany(self, sql, rows):
        self.sql = sql
        self.sqls.append(sql)

    def fetchall(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor
        self.cursor_name = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, name=None):
        self.cursor_name = name
        self._cursor.name = name
        return self._cursor

    def commit(self):
        return None


def _store() -> ChunkStore:
    return ChunkStore(dsn="postgresql://unused")


def test_upsert_does_not_reset_created_at_on_conflict(monkeypatch):
    cur = _FakeCursor()
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    n = _store().upsert([
        Chunk(text="hello", source_type="pdf", source_id="a.pdf", chunk_index=0),
    ])
    assert n == 1
    assert "created_at" not in cur.sql
    assert "chunk_index" in cur.sql


def test_iter_missing_embeddings_uses_named_cursor(monkeypatch):
    rows = [(f"id{i}", f"text{i}") for i in range(5)]
    cur = _FakeCursor(rows=rows)
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    batches = list(_store().iter_missing_embeddings(batch=2))
    assert conn.cursor_name == "missing"
    assert cur.itersize == 2
    assert batches == [
        [("id0", "text0"), ("id1", "text1")],
        [("id2", "text2"), ("id3", "text3")],
        [("id4", "text4")],
    ]


def test_migrate_chunk_columns_adds_chunk_index_not_null():
    cur = _FakeCursor()
    _migrate_chunk_columns(cur, "chunks")
    joined = " ".join(cur.sqls)
    assert "ADD COLUMN IF NOT EXISTS chunk_index" in joined
    assert "ALTER COLUMN chunk_index SET NOT NULL" in joined
    assert "SET chunk_index = 0" in joined


def test_dsn_with_defaults_adds_sslmode_for_remote_hosts(monkeypatch):
    monkeypatch.delenv("PG_SSLMODE", raising=False)
    hosts = (
        "aws-0-us-east-1.pooler.supabase.com",
        "ep-cool-name.us-east-2.aws.neon.tech",
        "mydb.abc123.us-east-1.rds.amazonaws.com",
        "p.demo.crunchybridge.com",
    )
    for host in hosts:
        dsn = f"postgresql://user:x@{host}:5432/postgres"
        out = _dsn_with_defaults(dsn)
        assert "sslmode=require" in out, host
        assert _dsn_with_defaults(out) == out


def test_dsn_with_defaults_leaves_loopback_dsn(monkeypatch):
    monkeypatch.delenv("PG_SSLMODE", raising=False)
    for host in ("127.0.0.1", "localhost", "[::1]"):
        dsn = f"postgresql://rag:change-me@{host}:5432/rag"
        assert _dsn_with_defaults(dsn) == dsn


def test_dsn_with_defaults_explicit_sslmode_wins(monkeypatch):
    monkeypatch.setenv("PG_SSLMODE", "disable")
    dsn = "postgresql://u:p@db.example.com:5432/rag?sslmode=prefer"
    assert _dsn_with_defaults(dsn) == dsn


def test_dsn_with_defaults_pg_sslmode_override(monkeypatch):
    monkeypatch.setenv("PG_SSLMODE", "disable")
    dsn = "postgresql://u:p@db.example.com:5432/rag"
    assert "sslmode=disable" in _dsn_with_defaults(dsn)


def test_dsn_with_defaults_leaves_keyword_dsn():
    dsn = "host=localhost port=5432 dbname=rag user=rag password=x"
    assert _dsn_with_defaults(dsn) == dsn


def test_dsn_endpoint_omits_password():
    dsn = "postgresql://rag:s3cret@db.example.com:5432/rag"
    endpoint = _dsn_endpoint(dsn)
    assert "s3cret" not in endpoint
    assert endpoint == "db.example.com:5432/rag"
    assert "s3cret" not in _redact_secret("boom s3cret leaked", dsn)


def test_connect_error_names_endpoint_not_password(monkeypatch):
    import sys
    import types
    fake = types.ModuleType("psycopg")

    def boom(*_a, **_k):
        raise OSError("could not connect s3cret")

    fake.connect = boom
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    store = ChunkStore(dsn="postgresql://rag:s3cret@db.example.com:5432/rag")
    try:
        store._connect()
    except RuntimeError as exc:
        msg = str(exc)
        assert "db.example.com:5432/rag" in msg
        assert "s3cret" not in msg
        assert "PG_DSN" in msg
    else:
        raise AssertionError("expected RuntimeError")


class _PathCursor:
    def __init__(self, search_path: str):
        self.search_path = search_path
        self.sqls: list[str] = []
        self.params = None

    def execute(self, sql, params=None):
        self.sqls.append(sql)
        self.params = params

    def fetchone(self):
        return (self.search_path,)


def test_ensure_search_path_appends_extensions():
    cur = _PathCursor("public")
    _ensure_search_path_has(cur, "extensions")
    assert any("set_config" in s.lower() for s in cur.sqls)
    assert cur.params == ("public, extensions",)


def test_ensure_search_path_skips_when_present():
    cur = _PathCursor("public, extensions")
    _ensure_search_path_has(cur, "extensions")
    assert not any("set_config" in s.lower() for s in cur.sqls)


def test_ensure_vector_extension_sets_search_path_on_extensions_schema():
    class _Cur(_PathCursor):
        def __init__(self):
            super().__init__("public")
            self._first = True

        def fetchone(self):
            if self._first:
                self._first = False
                return (1,)
            return (self.search_path,)

    cur = _Cur()
    _ensure_vector_extension(cur)
    joined = " ".join(cur.sqls)
    assert "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions" in joined
    assert "SHOW search_path" in joined
    assert any("set_config" in s.lower() for s in cur.sqls)


def test_fetch_by_ids_selects_chunk_index(monkeypatch):
    cur = _FakeCursor(rows=[("id0", "hello", "S", "a.pdf", 1, "", 0, "T", "en")])
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    got = _store().fetch_by_ids(["id0"])
    assert "chunk_index" in cur.sql
    assert got["id0"]["chunk_index"] == 0
    assert got["id0"]["title"] == "T"
