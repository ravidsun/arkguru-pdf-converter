"""Unit tests for ChunkStore write/iterate contracts (no live Postgres)."""
from __future__ import annotations

import pytest

from common.datastore import (
    ChunkStore,
    _coerce_ef_search,
    _dsn_endpoint,
    _dsn_with_defaults,
    _ensure_search_path_has,
    _ensure_vector_extension,
    _migrate_chunk_columns,
    _redact_secret,
    _search_chunks_ddl,
    _sql_ident,
    _undefined_function,
    qualify_ident,
)
from common.schema import Chunk


class _FakeCursor:
    def __init__(self, *, name=None, rows=()):
        self.name = name
        self.itersize = None
        self.sql = None
        self.sqls: list[str] = []
        self._rows = list(rows)
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql = sql
        self.sqls.append(sql)
        self.rowcount = 0

    def executemany(self, sql, rows):
        self.sql = sql
        self.sqls.append(sql)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

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


def test_search_chunks_ddl_uses_hnsw_safe_dense_subquery():
    sql = _search_chunks_ddl("chunks", "chunk_embeddings")
    assert "CREATE OR REPLACE FUNCTION search_chunks" in sql
    assert "FROM chunk_embeddings v" in sql
    assert "ORDER BY v.embedding <=> query_embedding" in sql
    assert "LIMIT k_dense" in sql
    assert "is_parent = false" in sql
    assert "1.0 / (rrf_k + d.rnk)" in sql
    assert "FULL OUTER JOIN lexical" in sql
    # JOIN chunks must not wrap the ANN LIMIT (HNSW would be skipped).
    dense_block = sql.split("dense_hits AS")[0]
    assert "JOIN chunks" not in dense_block


def test_search_chunks_ddl_falls_back_from_and_to_or_lexical_query():
    """plainto ANDs every term, which a natural question rarely satisfies."""
    sql = _search_chunks_ddl("chunks", "chunk_embeddings")
    # AND is still tried first
    assert "lex_and AS (\n  SELECT plainto_tsquery('english', query_text)" in sql
    # the OR form is built from the query's own lexemes
    assert "tsvector_to_array(" in sql
    assert "' | '" in sql
    # and the switch is driven by how many rows the AND pass found
    assert "(SELECT n FROM lex_and_n) >= k_lexical" in sql
    assert "THEN (SELECT q FROM lex_and)" in sql
    assert "ELSE (SELECT q FROM lex_or)" in sql
    # ranking and truncation are unchanged, so the wider set stays cheap
    assert "ORDER BY ts_rank(c.ts, lex_q.q) DESC" in sql
    assert "LIMIT k_lexical" in sql


def test_search_chunks_ddl_probe_is_bounded():
    """The fallback test must not become a full count over the corpus."""
    sql = _search_chunks_ddl("chunks", "chunk_embeddings")
    probe = sql.split("lex_and_n AS (")[1].split("),")[0]
    assert "count(*)" in probe
    assert "LIMIT k_lexical" in probe


def test_search_chunks_ddl_quotes_lexemes():
    """Apostrophes and tsquery operators in user text must not break to_tsquery."""
    sql = _search_chunks_ddl("chunks", "chunk_embeddings")
    assert "quote_literal(lx)" in sql


def test_search_chunks_ddl_keeps_thirteen_column_contract():
    """Phase 3 indexes these positionally; new columns may only be appended."""
    sql = _search_chunks_ddl("chunks", "chunk_embeddings")
    returns = sql.split("RETURNS TABLE (")[1].split(")\nLANGUAGE")[0]
    cols = [c.strip().split()[0] for c in returns.replace("\n", " ").split(",")]
    assert cols == [
        "chunk_id", "text", "section", "source_id", "page", "url",
        "parent_id", "chunk_index", "title", "lang",
        "rrf_score", "dense_rank", "lexical_rank",
    ]


def test_coerce_ef_search_accepts_ints_and_numeric_strings():
    assert _coerce_ef_search(200) == 200
    assert _coerce_ef_search("200") == 200
    assert _coerce_ef_search(None) is None
    assert _coerce_ef_search("") is None


@pytest.mark.parametrize("bad", ["abc", "1; DROP TABLE chunks", "40 OR 1=1", 0, -5])
def test_coerce_ef_search_rejects_non_integers(bad):
    """SET cannot bind parameters, so the value is interpolated and must be safe."""
    with pytest.raises(ValueError):
        _coerce_ef_search(bad)


def test_search_chunks_sets_ef_search_when_configured(monkeypatch):
    cur = _FakeCursor(rows=[])
    statements = []

    class _Recording(_FakeCursor):
        def execute(self, sql, params=None):
            statements.append(sql)
            return super().execute(sql, params)

    rec = _Recording(rows=[])
    conn = _FakeConn(rec)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    store = ChunkStore(dsn="postgresql://u@localhost/db", hnsw_ef_search=200)
    store.search_chunks("q", [0.1], k_dense=60, k_lexical=60, k_final=120)
    assert "SET hnsw.ef_search = 200" in statements
    # the GUC must be raised before the search, or it cannot affect it
    assert statements.index("SET hnsw.ef_search = 200") < len(statements) - 1


def test_search_chunks_omits_ef_search_when_unset(monkeypatch):
    statements = []

    class _Recording(_FakeCursor):
        def execute(self, sql, params=None):
            statements.append(sql)
            return super().execute(sql, params)

    conn = _FakeConn(_Recording(rows=[]))
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    ChunkStore(dsn="postgresql://u@localhost/db").search_chunks("q", [0.1])
    assert not any("ef_search" in s for s in statements)


def test_undefined_function_detects_sqlstate():
    err = Exception("missing")
    err.sqlstate = "42883"
    assert _undefined_function(err)
    assert not _undefined_function(Exception("other"))


def test_search_chunks_selects_sql_function(monkeypatch):
    cur = _FakeCursor(rows=[
        ("id0", "hello", "S", "a.pdf", 1, "", None, 0, "T", "en", 0.5, 1, None),
    ])
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    rows = _store().search_chunks("q", [0.1, 0.2], k_dense=20, k_lexical=20, k_final=6)
    assert "FROM search_chunks(" in cur.sql
    assert "%s::vector" in cur.sql
    assert rows[0][0] == "id0"
    assert rows[0][1] == "hello"


def test_search_chunks_retries_once_after_ensure_schema(monkeypatch):
    class _Cur(_FakeCursor):
        def execute(self, sql, params=None):
            super().execute(sql, params)
            if not getattr(self, "_ok", False):
                self._ok = True
                err = Exception("function search_chunks(text, vector) does not exist")
                err.sqlstate = "42883"
                raise err

    cur = _Cur(rows=[("id0", "hello", "", "s", 1, "", None, 0, "", "", 0.1, 1, 2)])
    conn = _FakeConn(cur)
    store = _store()
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    called = {"n": 0}

    def fake_schema(self):
        called["n"] += 1

    monkeypatch.setattr(ChunkStore, "ensure_schema", fake_schema)
    rows = store.search_chunks("q", [0.0], k_final=6)
    assert called["n"] == 1
    assert rows[0][0] == "id0"


def test_qualify_ident_public_is_unqualified():
    assert qualify_ident("chunks") == "chunks"
    assert qualify_ident("chunks", None) == "chunks"
    assert qualify_ident("chunks", "public") == "chunks"
    assert qualify_ident("chunks", "") == "chunks"


def test_qualify_ident_web_schema():
    assert qualify_ident("chunks", "web") == "web.chunks"
    assert qualify_ident("search_chunks", "web") == "web.search_chunks"
    assert qualify_ident("chunk_embeddings", "web") == "web.chunk_embeddings"


@pytest.mark.parametrize("bad", [
    "web; DROP TABLE chunks", "chunks-x", "web.chunks", "1web",
    "web--", "web/*",
])
def test_qualify_ident_rejects_injection(bad):
    with pytest.raises(ValueError):
        _sql_ident(bad)
    with pytest.raises(ValueError):
        qualify_ident(bad, "web")
    with pytest.raises(ValueError):
        qualify_ident("chunks", bad)


def test_sql_ident_rejects_empty():
    with pytest.raises(ValueError):
        _sql_ident("")
    # empty schema is treated as public, not injection
    assert qualify_ident("chunks", "") == "chunks"


def test_web_store_qualifies_tables_and_search_fn():
    store = ChunkStore(dsn="postgresql://unused", schema="web")
    assert store.schema == "web"
    assert store.chunks == "web.chunks"
    assert store.vectors == "web.chunk_embeddings"
    assert store.search_fn == "web.search_chunks"
    assert store.is_public_schema is False
    assert ChunkStore(dsn="postgresql://unused").is_public_schema is True


def test_search_chunks_ddl_web_does_not_replace_public_function():
    sql = _search_chunks_ddl(
        "web.chunks", "web.chunk_embeddings", function="web.search_chunks")
    assert "CREATE OR REPLACE FUNCTION web.search_chunks(" in sql
    assert "FROM web.chunk_embeddings v" in sql
    assert "JOIN web.chunks c" in sql
    # Unqualified public.search_chunks must not be created.
    assert "FUNCTION search_chunks(" not in sql


def test_web_search_chunks_calls_schema_qualified_function(monkeypatch):
    cur = _FakeCursor(rows=[])
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    ChunkStore(dsn="postgresql://unused", schema="web").search_chunks(
        "q", [0.1], k_final=6)
    assert "FROM web.search_chunks(" in cur.sql
    assert "FROM search_chunks(" not in cur.sql.replace(
        "FROM web.search_chunks(", "")


def test_delete_by_source_id(monkeypatch):
    cur = _FakeCursor()
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    n = ChunkStore(dsn="postgresql://unused", schema="web").delete_by_source_id(
        "https://example.com/a")
    assert n == 0
    assert "DELETE FROM web.chunks WHERE source_id" in cur.sql


def test_ensure_schema_creates_web_schema(monkeypatch):
    class _Cur(_FakeCursor):
        def fetchone(self):
            # to_regclass -> missing tables; SHOW / extension probes
            return (None,)

    cur = _Cur()
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    ChunkStore(dsn="postgresql://unused", schema="web").ensure_schema()
    joined = " ".join(cur.sqls)
    assert "CREATE SCHEMA IF NOT EXISTS web" in joined
    assert "CREATE OR REPLACE FUNCTION web.search_chunks(" in joined
    assert "CREATE TABLE IF NOT EXISTS web.chunks" in joined
    assert "CREATE TABLE IF NOT EXISTS web.chunk_embeddings" in joined
    public_fn = [s for s in cur.sqls if "FUNCTION search_chunks(" in s
                 and "FUNCTION web.search_chunks(" not in s]
    assert public_fn == []


def test_ensure_schema_public_does_not_create_web_schema(monkeypatch):
    class _Cur(_FakeCursor):
        def fetchone(self):
            return (None,)

    cur = _Cur()
    conn = _FakeConn(cur)
    monkeypatch.setattr(ChunkStore, "_connect", lambda self: conn)
    ChunkStore(dsn="postgresql://unused").ensure_schema()
    joined = " ".join(cur.sqls)
    assert "CREATE SCHEMA IF NOT EXISTS web" not in joined
    assert "CREATE OR REPLACE FUNCTION search_chunks(" in joined
    assert "FUNCTION web.search_chunks(" not in joined


def test_open_chunk_store_reads_schema(tmp_path):
    from common.datastore_config import open_chunk_store
    p = tmp_path / "datastore.yaml"
    p.write_text(
        "datastore:\n"
        "  postgres:\n"
        "    dsn: postgresql://unused\n"
        "    schema: web\n"
        "    chunks_table: chunks\n"
        "    vectors_table: chunk_embeddings\n"
    )
    store = open_chunk_store(str(p))
    assert store.schema == "web"
    assert store.chunks == "web.chunks"
    assert store.search_fn == "web.search_chunks"
