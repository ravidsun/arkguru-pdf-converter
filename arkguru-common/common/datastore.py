"""
Postgres + pgvector datastore -- TWO tables, cleanly separated:

  1. <chunks_table>      (default "chunks")           -- the chunk text + metadata.
     This is the JSONL-equivalent SOURCE OF TRUTH. No vectors here.
  2. <vectors_table>     (default "chunk_embeddings")  -- the embeddings only,
     keyed by chunk_id (FK -> chunks, ON DELETE CASCADE), with the HNSW index.

Why separate:
  - Embeddings are disposable & model-specific; chunk text/metadata is durable.
    Re-embed with a different model by truncating one table -- chunks untouched.
  - Phases 1 & 2 write ONLY the chunks table (no embedding model needed).
    Phase 3 fills the vectors table in place.
  - Lexical/full-text search lives on the chunks table; dense search JOINs the
    vectors table. Retrieval fuses both.

Phase flow:
    Phase 1/2  -> upsert()             -> chunks table (vectors table stays empty)
    Phase 3    -> update_embeddings()  -> chunk_embeddings table
    retrieve   -> search_dense()/search_lexical() (JOIN when dense)

Requires:  pip install "psycopg[binary]" pgvector
    export PG_DSN=postgresql://user:pass@localhost:5432/rag
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import logging
from .schema import Chunk

log = logging.getLogger("common.datastore")

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_SSLMODE_ENV = "PG_SSLMODE"

# columns of the chunks (source-of-truth) table
_COLS = ["chunk_id", "text", "source_type", "source_id", "chunk_index",
         "title", "section", "page", "url", "domain", "lang",
         "parent_id", "is_parent", "token_count", "overlap_tokens"]

_COL_DDL = {
    "chunk_id": "text",
    "text": "text",
    "source_type": "text",
    "source_id": "text",
    "chunk_index": "int",
    "title": "text",
    "section": "text",
    "page": "int",
    "url": "text",
    "domain": "text",
    "lang": "text",
    "parent_id": "text",
    "is_parent": "boolean",
    "token_count": "int",
    "overlap_tokens": "int",
}

# columns returned to retrieval callers. New fields are appended so existing
# positional indexes (chunk_id..parent_id) stay stable for Phase 3.
_HIT_COLS = ["chunk_id", "text", "section", "source_id", "page", "url",
             "parent_id", "chunk_index", "title", "lang"]


def _ensure_search_path_has(cur, schema: str) -> None:
    """Append ``schema`` to this session's ``search_path`` if it is missing.

    Needed when ``vector`` lives in ``extensions`` (hosted Supabase) so
    ``vector(dim)`` DDL and pgvector operators resolve without qualifying.
    """
    cur.execute("SHOW search_path")
    row = cur.fetchone()
    current = (row[0] if row else "") or ""
    tokens = [t.strip().strip('"') for t in current.split(",") if t.strip()]
    if any(t.lower() == schema.lower() for t in tokens):
        return
    updated = f"{current}, {schema}" if current.strip() else schema
    cur.execute("SELECT set_config('search_path', %s, false)", (updated,))


def _ensure_vector_extension(cur) -> None:
    """Install pgvector in schema ``extensions`` when that schema exists.

    Hosted Supabase puts the extension there. Local Docker / native installs
    often have no ``extensions`` schema; fall back to the default search_path.
    """
    cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", ("extensions",))
    if cur.fetchone():
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;")
        _ensure_search_path_has(cur, "extensions")
    else:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def _parse_url_dsn(dsn: str):
    """Return ``urlparse`` result for a URL DSN, or ``None`` for keyword form."""
    if "://" not in dsn:
        return None
    try:
        return urlparse(dsn)
    except Exception:
        return None


def _is_loopback_host(host: Optional[str]) -> bool:
    if not host:
        return False
    return host.lower().strip("[]") in _LOOPBACK_HOSTS


def _dsn_endpoint(dsn: str) -> str:
    """Host:port/db for logs and errors. Never includes user or password."""
    parsed = _parse_url_dsn(dsn)
    if parsed is None:
        return "(non-URL DSN)"
    host = parsed.hostname or "?"
    port = parsed.port or 5432
    db = (parsed.path or "/").lstrip("/") or "?"
    return f"{host}:{port}/{db}"


def _redact_secret(text: str, dsn: str) -> str:
    """Strip the DSN password from an exception string if it leaked."""
    parsed = _parse_url_dsn(dsn)
    if parsed is None or not parsed.password:
        return text
    return text.replace(parsed.password, "***")


def _dsn_with_defaults(dsn: str) -> str:
    """Fill in ``sslmode`` when the URI omitted it.

    Precedence: explicit ``sslmode`` in the URI, then ``$PG_SSLMODE``, then
    ``require`` for remote hosts. Loopback (localhost / 127.0.0.1 / ::1) is
    left unchanged so local Docker / native Postgres works without TLS.
    Keyword/value libpq strings are not rewritten.
    """
    parsed = _parse_url_dsn(dsn)
    if parsed is None:
        return dsn
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if any(k.lower() == "sslmode" for k, _ in pairs):
        return dsn
    override = (os.environ.get(_SSLMODE_ENV) or "").strip()
    if override:
        pairs.append(("sslmode", override))
        return urlunparse(parsed._replace(query=urlencode(pairs)))
    if _is_loopback_host(parsed.hostname):
        return dsn
    pairs.append(("sslmode", "require"))
    return urlunparse(parsed._replace(query=urlencode(pairs)))


def _migrate_chunk_columns(cur, table: str) -> None:
    """Add any missing ``chunks`` columns and make ``chunk_index`` NOT NULL.

    ``CREATE TABLE IF NOT EXISTS`` will not add columns to an older table, so
    an ingest can otherwise skip ``chunk_index``. Idempotent.
    """
    for name, typ in _COL_DDL.items():
        cur.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {typ}")
    cur.execute(
        f"UPDATE {table} SET chunk_index = 0 WHERE chunk_index IS NULL")
    cur.execute(
        f"ALTER TABLE {table} ALTER COLUMN chunk_index SET DEFAULT 0")
    cur.execute(
        f"ALTER TABLE {table} ALTER COLUMN chunk_index SET NOT NULL")


class ChunkStore:
    def __init__(self, dsn: Optional[str] = None, table: str = "chunks",
                 vectors_table: str = "chunk_embeddings", dim: int = 1024,
                 dsn_env: str = "PG_DSN"):
        self.dsn = dsn or os.environ.get(dsn_env)
        if not self.dsn:
            raise ValueError(
                f"No Postgres DSN. Pass dsn=... or set ${dsn_env}, e.g. "
                "postgresql://user:pass@localhost:5432/rag")
        self.chunks = table
        self.vectors = vectors_table
        self.dim = dim

    # -- connection --------------------------------------------------------
    def _connect(self):
        import psycopg
        try:
            conn = psycopg.connect(_dsn_with_defaults(self.dsn), connect_timeout=15)
        except Exception as e:
            endpoint = _dsn_endpoint(self.dsn)
            detail = _redact_secret(str(e), self.dsn)
            raise RuntimeError(
                f"Could not connect to Postgres at {endpoint}. "
                "Check PG_DSN (hosted: session/direct URI + sslmode=require; "
                "local: docker compose up, postgresql://rag:...@localhost:5432/rag). "
                "Transaction-mode poolers (e.g. Supabase :6543) break named cursors. "
                f"Underlying error: {detail}"
            ) from e
        try:
            from pgvector.psycopg import register_vector
            register_vector(conn)
        except Exception:
            pass
        return conn

    # -- schema (two tables) ----------------------------------------------
    def ensure_schema(self) -> None:
        """Create the two tables + indexes if missing. Idempotent; safe to call on
        every run. Logs whether each table was created or already present, and
        raises a clear error if the database is unreachable."""
        try:
            conn = self._connect()
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                "Could not connect to Postgres for the datastore. Check PG_DSN / "
                "config/datastore.yaml (see docs/DATABASE_SETUP.md). "
                f"Underlying error: {_redact_secret(str(e), self.dsn)}") from e
        with conn, conn.cursor() as cur:
            _ensure_vector_extension(cur)
            pre = {}
            for t in (self.chunks, self.vectors):
                cur.execute("SELECT to_regclass(%s)", (t,))
                pre[t] = cur.fetchone()[0] is not None
            # 1) chunks = source of truth (no embedding column)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.chunks} (
                    chunk_id     text PRIMARY KEY,
                    text         text NOT NULL,
                    source_type  text,
                    source_id    text,
                    chunk_index  int NOT NULL DEFAULT 0,
                    title        text,
                    section      text,
                    page         int,
                    url          text,
                    domain       text,
                    lang         text,
                    parent_id    text,
                    is_parent    boolean DEFAULT false,
                    token_count  int,
                    overlap_tokens int DEFAULT 0,
                    meta         jsonb DEFAULT '{{}}'::jsonb,
                    ts           tsvector GENERATED ALWAYS AS
                                 (to_tsvector('english', coalesce(text,''))) STORED,
                    created_at   timestamptz DEFAULT now()
                );""")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.chunks}_ts_idx "
                        f"ON {self.chunks} USING gin(ts);")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.chunks}_source_idx "
                f"ON {self.chunks} (source_type, source_id);")
            _migrate_chunk_columns(cur, self.chunks)
            # 2) vectors = embeddings only, keyed to chunks
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.vectors} (
                    chunk_id   text PRIMARY KEY
                               REFERENCES {self.chunks}(chunk_id) ON DELETE CASCADE,
                    embedding  vector({self.dim}),
                    model      text,
                    created_at timestamptz DEFAULT now()
                );""")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.vectors}_hnsw_idx "
                        f"ON {self.vectors} USING hnsw (embedding vector_cosine_ops);")
            conn.commit()
            for t in (self.chunks, self.vectors):
                log.info("datastore table '%s': %s", t,
                         "already present" if pre.get(t) else "created")

    # -- write chunks (Phases 1/2) ----------------------------------------
    def upsert(self, chunks: Iterable[Chunk]) -> int:
        """Insert/update chunk rows (source of truth). Idempotent by chunk_id.
        Does NOT touch the vectors table."""
        rows = []
        indexes: list[int] = []
        nulls = 0
        for c in chunks:
            d = c.to_dict()
            idx = d.get("chunk_index")
            if idx is None:
                nulls += 1
                d["chunk_index"] = 0
                idx = 0
            indexes.append(int(idx))
            rows.append(tuple(d.get(k) for k in _COLS) + (json.dumps(d.get("extra") or {}),))
        if not rows:
            return 0
        placeholders = ",".join(["%s"] * (len(_COLS) + 1))
        collist = ",".join(_COLS + ["meta"])
        updates = ",".join(f"{k}=EXCLUDED.{k}" for k in _COLS if k != "chunk_id")
        # Leave created_at on conflict so re-ingests do not advance the backup watermark.
        sql = (f"INSERT INTO {self.chunks} ({collist}) VALUES ({placeholders}) "
               f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}, meta=EXCLUDED.meta")
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
            conn.commit()
        log.info(
            "upsert %d row(s) into '%s'; chunk_index min=%s max=%s nulls=%d",
            len(rows), self.chunks,
            min(indexes) if indexes else None,
            max(indexes) if indexes else None,
            nulls,
        )
        return len(rows)

    # -- embeddings (Phase 3) ---------------------------------------------
    def iter_missing_embeddings(self, batch: int = 256
                                ) -> Iterator[list[tuple[str, str]]]:
        """Yield (chunk_id, text) for chunks that have no row in the vectors table.

        Uses a server-side cursor so the client does not buffer the full result set.
        """
        with self._connect() as conn, conn.cursor(name="missing") as cur:
            cur.itersize = batch
            cur.execute(
                f"SELECT c.chunk_id, c.text FROM {self.chunks} c "
                f"LEFT JOIN {self.vectors} v ON c.chunk_id = v.chunk_id "
                f"WHERE v.chunk_id IS NULL AND c.is_parent = false")
            buf: list[tuple[str, str]] = []
            for row in cur:
                buf.append((row[0], row[1]))
                if len(buf) >= batch:
                    yield buf
                    buf = []
            if buf:
                yield buf

    def update_embeddings(self, ids: list[str], vectors, model: Optional[str] = None) -> int:
        """Upsert embeddings into the vectors table (keyed by chunk_id)."""
        with self._connect() as conn, conn.cursor() as cur:
            for cid, vec in zip(ids, vectors):
                v = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                cur.execute(
                    f"INSERT INTO {self.vectors} (chunk_id, embedding, model) "
                    f"VALUES (%s, %s, %s) ON CONFLICT (chunk_id) DO UPDATE "
                    f"SET embedding = EXCLUDED.embedding, model = EXCLUDED.model, "
                    f"created_at = now()", (cid, v, model))
            conn.commit()
            return len(ids)

    # -- counts / reads ----------------------------------------------------
    def count(self, only_missing_embedding: bool = False) -> int:
        if only_missing_embedding:
            q = (f"SELECT count(*) FROM {self.chunks} c "
                 f"LEFT JOIN {self.vectors} v ON c.chunk_id = v.chunk_id "
                 f"WHERE v.chunk_id IS NULL AND c.is_parent = false")
        else:
            q = f"SELECT count(*) FROM {self.chunks}"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(q)
            return cur.fetchone()[0]

    def count_vectors(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.vectors}")
            return cur.fetchone()[0]

    def fetch_by_ids(self, ids: list[str]) -> dict[str, dict]:
        """Return ``chunk_id -> {text, section, source_id, page, url, chunk_index, title, lang}``."""
        if not ids:
            return {}
        cols = ["chunk_id", "text", "section", "source_id", "page", "url",
                "chunk_index", "title", "lang"]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {','.join(cols)} FROM {self.chunks} "
                f"WHERE chunk_id = ANY(%s)", (list(ids),))
            out: dict[str, dict] = {}
            for row in cur.fetchall():
                out[row[0]] = {
                    "text": row[1] or "",
                    "section": row[2] or "",
                    "source_id": row[3] or "",
                    "page": row[4],
                    "url": row[5] or "",
                    "chunk_index": row[6],
                    "title": row[7] or "",
                    "lang": row[8] or "",
                }
            return out

    def created_at_watermark(self) -> str:
        """Greatest ``created_at`` across chunks and embeddings, as text."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT greatest("
                f"  coalesce((SELECT max(created_at) FROM {self.chunks}), "
                f"           '-infinity'::timestamptz),"
                f"  coalesce((SELECT max(created_at) FROM {self.vectors}), "
                f"           '-infinity'::timestamptz)"
                f")::text")
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else "-infinity"

    def list_pdf_sources(self) -> list[tuple[str, int, int]]:
        """Per PDF ``source_id``: (source_id, child chunk count, embedded count)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT c.source_id, "
                f"       count(*) FILTER (WHERE c.is_parent = false) AS chunks, "
                f"       count(v.chunk_id) AS embedded "
                f"FROM {self.chunks} c "
                f"LEFT JOIN {self.vectors} v ON v.chunk_id = c.chunk_id "
                f"WHERE c.source_type = 'pdf' "
                f"GROUP BY c.source_id "
                f"ORDER BY c.source_id")
            return [(r[0] or "", int(r[1]), int(r[2])) for r in cur.fetchall()]

    def chunk_index_counts(self, source_id: str) -> list[tuple]:
        """``(chunk_index, count)`` for one ``source_id``, ordered by index."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT chunk_index, count(*) FROM {self.chunks} "
                f"WHERE source_id = %s GROUP BY 1 ORDER BY 1",
                (source_id,))
            return [(r[0], int(r[1])) for r in cur.fetchall()]

    def read_all(self, include_parents: bool = True) -> list[Chunk]:
        q = f"SELECT {','.join(_COLS)}, meta FROM {self.chunks}"
        if not include_parents:
            q += " WHERE is_parent = false"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(q)
            out = []
            for row in cur.fetchall():
                d = dict(zip(_COLS, row))
                d["extra"] = row[-1] or {}
                out.append(Chunk.from_dict(d))
            return out

    # -- search (Phase 3 retrieval) ---------------------------------------
    def search_dense(self, qvec, k: int = 20) -> list[tuple]:
        """Cosine NN over the vectors table, JOINed back to chunk text/metadata."""
        cols = ",".join(f"c.{x}" for x in _HIT_COLS)
        vec = "[" + ",".join(str(float(x)) for x in qvec) + "]"   # pgvector literal
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols}, 1 - (v.embedding <=> %s::vector) AS score "
                f"FROM {self.vectors} v JOIN {self.chunks} c ON c.chunk_id = v.chunk_id "
                f"ORDER BY v.embedding <=> %s::vector LIMIT %s", (vec, vec, k))
            return cur.fetchall()

    def search_lexical(self, query: str, k: int = 20) -> list[tuple]:
        """Full-text (BM25-like) search over the chunks table."""
        cols = ",".join(f"c.{x}" for x in _HIT_COLS)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols}, ts_rank(c.ts, plainto_tsquery('english', %s)) AS score "
                f"FROM {self.chunks} c "
                f"WHERE c.ts @@ plainto_tsquery('english', %s) "
                f"ORDER BY score DESC LIMIT %s", (query, query, k))
            return cur.fetchall()
