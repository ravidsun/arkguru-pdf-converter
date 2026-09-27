"""End-to-end search_chunks v2 on a throwaway schema (Postgres 16 + pgvector).

Skipped unless PG_DSN (or PG_DSN_TEST) points at loopback or the CI service
hostname ``postgres``. Never talks to a remote / production database.
Uses the hashing embedder with an explicit allow — no model downloads.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest

from pathlib import Path

from common.datastore import ChunkStore
from common.migrate import apply_migrations
from common.phrase import detect_phrase_terms, load_multiword_terms
from common.schema import Chunk
from common.validate_schema import run_checks
from phase3_rag.embedder import (
    Embedder,
    assert_datastore_backend_allowed,
    write_embeddings,
)

pytestmark = pytest.mark.pgvector

_SAFE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "postgres"})
_SCHEMA = "ci_search_v2"


def _safe_dsn() -> str | None:
    dsn = (os.environ.get("PG_DSN_TEST") or os.environ.get("PG_DSN") or "").strip()
    if not dsn or "://" not in dsn:
        return None
    host = (urlparse(dsn).hostname or "").lower().strip("[]")
    if host not in _SAFE_HOSTS:
        return None
    return dsn


@pytest.fixture(scope="module")
def pg_dsn():
    dsn = _safe_dsn()
    if not dsn:
        pytest.skip("no loopback/CI PG_DSN (set PG_DSN_TEST or PG_DSN to localhost)")
    return dsn


@pytest.fixture
def store(pg_dsn):
    st = ChunkStore(dsn=pg_dsn, schema=_SCHEMA, dim=1024)
    apply_migrations(st, allow_public=False, dim=1024)
    with st._connect() as conn, conn.cursor() as cur:
        cur.execute(f"TRUNCATE {st.chunks} CASCADE")
        conn.commit()
    return st


def _child(**kw) -> Chunk:
    kw.setdefault("source_type", "pdf")
    kw.setdefault("source_id", "kp.pdf")
    kw.setdefault("chunk_index", kw.get("chunk_index", 0))
    kw.setdefault("lang", "en")
    return Chunk(**kw)


def test_search_chunks_v2_phrase_dense_and_exclusions(store):
    lexicon = load_multiword_terms(
        Path(__file__).resolve().parents[1]
        / "phase3_rag" / "golden" / "domain_lexicon.json"
    )
    parent = _child(
        text="Parent section wrapping the KP horary chapter",
        chunk_index=0, is_parent=True, section="Horary",
    )
    kp = _child(
        text=(
            "In Krishnamurti Paddhati the sub sub lord of the cuspal "
            "sub lord decides the matter in a horary chart. "
            "The sub sub lord must signify the relevant houses."
        ),
        chunk_index=1, parent_id=parent.chunk_id, section="Horary",
    )
    noise = _child(
        text="Saturn in the seventh house indicates delay in marriage.",
        chunk_index=2, source_id="bphs.pdf", section="Saturn",
    )
    flagged = _child(
        text="@@@ garbled mojibake ##### |Col1| |Col2| sub sub lord horary",
        chunk_index=3, section="Junk",
    )
    flagged.extra["quality"] = {
        "passed": False, "embed": False, "reasons": ["low_score"],
    }
    unembedded = _child(
        text="The sub sub lord is also discussed in this leftover child.",
        chunk_index=4, section="Leftover",
    )
    store.upsert([parent, kp, noise, flagged, unembedded])

    assert_datastore_backend_allowed("hashing", allow_hashing=True)
    emb = Embedder(backend="hashing", dim=1024)
    write_embeddings(
        store,
        [kp.chunk_id, noise.chunk_id, flagged.chunk_id],
        [kp.text, noise.text, flagged.text],
        emb,
    )

    query = "What is the role of the sub sub lord in KP horary?"
    qvec = emb.encode([query])[0]
    phrases = detect_phrase_terms(query, lexicon)
    assert phrases, "lexicon must detect 'sub sub lord'"

    rows = store.search_chunks(
        query, qvec,
        k_dense=8, k_lexical=8, k_phrase=8, k_final=8,
        phrase_terms=phrases,
    )
    ids = [r[0] for r in rows]
    assert parent.chunk_id not in ids
    assert flagged.chunk_id not in ids
    assert kp.chunk_id in ids

    by_id = {r[0]: r for r in rows}
    hit = by_id[kp.chunk_id]
    # columns: ... rrf, dense_rank, lexical_rank, phrase_rank
    assert hit[11] is not None, "expected a live dense_rank"
    assert int(hit[11]) >= 1
    assert hit[13] is not None, "phrase leg should rank the KP passage"

    lang_rows = store.search_chunks(
        query, qvec, k_final=8, phrase_terms=phrases, lang="hr",
    )
    assert lang_rows == []

    pdf_rows = store.search_chunks(
        query, qvec, k_final=8, phrase_terms=phrases, source_type="pdf",
    )
    assert kp.chunk_id in [r[0] for r in pdf_rows]

    # unembedded child can appear via lexical/phrase but not as a dense hit
    if unembedded.chunk_id in by_id:
        assert by_id[unembedded.chunk_id][11] is None


def test_phase4_validation_cli_runs_readonly(store):
    results = run_checks(store)
    names = {r.name for r in results}
    assert "search_chunks_function" in names
    fn = next(r for r in results if r.name == "search_chunks_function")
    assert fn.ok, fn.error or fn.value
    assert all(r.sql.lstrip().upper().startswith("SELECT") for r in results)
