"""Unit tests for ChunkStore write/iterate contracts (no live Postgres)."""
from __future__ import annotations

from common.datastore import ChunkStore
from common.schema import Chunk


class _FakeCursor:
    def __init__(self, *, name=None, rows=()):
        self.name = name
        self.itersize = None
        self.sql = None
        self._rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql = sql

    def executemany(self, sql, rows):
        self.sql = sql

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
