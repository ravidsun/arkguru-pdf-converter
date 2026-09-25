"""Retriever factory, parent expansion, and JSONL layout (no Ollama / no DB)."""
from __future__ import annotations

from pathlib import Path

import pytest

from common.schema import Chunk, write_jsonl
from phase3_rag.backup import run_backup, should_skip
from phase3_rag.retrieve import (
    Hit,
    JsonlRetriever,
    create_retriever,
    expand_hits_with_parents,
    list_jsonl_files,
)
from phase3_rag.run_pdfs import collect_processed_jsonl


class _FakeJsonl:
    def __init__(self, cfg, processed_dir="data/processed"):
        self.kind = "jsonl"
        self.cfg = cfg
        self.processed_dir = processed_dir


class _FakeHybrid:
    def __init__(self, cfg):
        self.kind = "hybrid"
        self.cfg = cfg


def test_no_dsn_falls_back_to_jsonl(monkeypatch):
    monkeypatch.setattr("common.datastore_config.resolve_dsn", lambda *a, **k: None)
    monkeypatch.setattr("phase3_rag.retrieve.JsonlRetriever", _FakeJsonl)
    retr = create_retriever({"datastore_config": "missing.yaml"})
    assert retr.kind == "jsonl"


def test_dsn_set_raises_on_connect_failure(monkeypatch):
    monkeypatch.setattr(
        "common.datastore_config.resolve_dsn", lambda *a, **k: "postgresql://example"
    )

    def boom(_dsn):
        raise OSError("connection refused")

    monkeypatch.setattr("phase3_rag.retrieve._probe_postgres", boom)
    with pytest.raises(RuntimeError, match="DSN is set"):
        create_retriever({"datastore_config": "missing.yaml"})


def test_dsn_set_uses_hybrid_when_connect_ok(monkeypatch):
    monkeypatch.setattr(
        "common.datastore_config.resolve_dsn", lambda *a, **k: "postgresql://example"
    )
    monkeypatch.setattr("phase3_rag.retrieve._probe_postgres", lambda _dsn: None)
    monkeypatch.setattr("phase3_rag.retrieve.HybridRetriever", _FakeHybrid)
    retr = create_retriever({"embedding_model": "x", "reranker_model": "y"})
    assert retr.kind == "hybrid"


def test_hybrid_search_calls_store_search_chunks():
    class _Store:
        def __init__(self):
            self.kw = None

        def search_chunks(self, query, qvec, **kw):
            self.kw = kw
            assert query == "what is yoga"
            assert qvec == [0.1, 0.2]
            return [(
                "id0", "hello", "S", "a.pdf", 1, "", None, 0, "T", "en",
                0.5, 1, None,
            )]

        def search_dense(self, *a, **k):
            raise AssertionError("search_dense must not be used when DSN is set")

        def search_lexical(self, *a, **k):
            raise AssertionError("search_lexical must not be used when DSN is set")

        def fetch_by_ids(self, ids):
            return {}

    class _Emb:
        def encode(self, query, normalize_embeddings=True):
            class _Vec:
                def tolist(self):
                    return [0.1, 0.2]
            return _Vec()

    class _RR:
        def predict(self, pairs):
            assert pairs == [("what is yoga", "hello")]
            return [0.91]

    from phase3_rag.retrieve import HybridRetriever

    retr = HybridRetriever.__new__(HybridRetriever)
    retr.cfg = {
        "retrieval": {
            "top_k_vector": 20, "top_k_bm25": 20, "top_k_final": 6,
            "top_k_candidates": 40,
            "use_parent_expansion": False,
        },
    }
    retr.embedder = _Emb()
    retr.store = _Store()
    retr.reranker = _RR()
    hits = retr.search("what is yoga")
    assert retr.store.kw["k_dense"] == 20
    assert retr.store.kw["k_lexical"] == 20
    assert retr.store.kw["k_final"] == 40
    assert len(hits) == 1
    assert hits[0].chunk_id == "id0"
    assert hits[0].text == "hello"
    assert hits[0].score == pytest.approx(0.91)


def test_hybrid_search_candidate_pool_defaults_to_both_legs():
    """Without top_k_candidates the pool is the full union, not max of the legs.

    Capping at max(k_dense, k_lexical) discarded candidates the fused set had
    already produced, before the cross-encoder ever saw them.
    """
    class _Store:
        def __init__(self):
            self.kw = None

        def search_chunks(self, query, qvec, **kw):
            self.kw = kw
            return []

        def fetch_by_ids(self, ids):
            return {}

    class _Emb:
        def encode(self, query, normalize_embeddings=True):
            class _Vec:
                def tolist(self):
                    return [0.1]
            return _Vec()

    from phase3_rag.retrieve import HybridRetriever

    retr = HybridRetriever.__new__(HybridRetriever)
    retr.cfg = {"retrieval": {"top_k_vector": 60, "top_k_bm25": 60,
                              "top_k_final": 6, "use_parent_expansion": False}}
    retr.embedder = _Emb()
    retr.store = _Store()
    retr.reranker = None
    retr.search("q")
    assert retr.store.kw["k_final"] == 120


def test_reranker_is_the_v2_multilingual_model():
    """A config-only assertion: the model download cannot run on a Cloud Agent.

    bge-reranker-base is the weakest of that family and is English-first, which
    is a poor fit for a corpus mixing English with transliterated Sanskrit.
    """
    import yaml

    cfg = yaml.safe_load(open("config/config.yaml"))["phase3"]
    assert cfg["reranker_model"] == "BAAI/bge-reranker-v2-m3"


def test_config_retrieval_knobs_are_consistent():
    """top_k_vector above hnsw_ef_search would be silently truncated by pgvector."""
    import yaml

    cfg = yaml.safe_load(open("config/config.yaml"))["phase3"]
    ds = yaml.safe_load(open("config/datastore.yaml"))["datastore"]["postgres"]
    rc = cfg["retrieval"]
    assert ds["hnsw_ef_search"] >= rc["top_k_vector"]
    assert rc["top_k_candidates"] >= max(rc["top_k_vector"], rc["top_k_bm25"])
    assert rc["top_k_final"] <= rc["top_k_candidates"]


def test_expand_hits_swaps_parent_text_and_dedupes():
    hits = [
        Hit("c1", "child one", 0.9, parent_id="p1"),
        Hit("c2", "child two", 0.8, parent_id="p1"),
        Hit("c3", "orphan", 0.7),
    ]
    parents = {"p1": {"text": "full parent section", "section": "PPE"}}
    out = expand_hits_with_parents(hits, parents, max_chars=10_000)
    assert [h.chunk_id for h in out] == ["c1", "c3"]
    assert out[0].text == "full parent section"
    assert out[0].section == "PPE"
    assert out[1].text == "orphan"


def test_expand_hits_caps_chars():
    hits = [
        Hit("c1", "aa", 1.0, parent_id="p1"),
        Hit("c2", "bb", 0.5, parent_id="p2"),
    ]
    parents = {
        "p1": {"text": "xxxx"},
        "p2": {"text": "yyyyyyyy"},
    }
    out = expand_hits_with_parents(hits, parents, max_chars=5)
    assert [h.chunk_id for h in out] == ["c1"]


def test_watermark_skip_empty_and_unchanged():
    assert should_skip("-infinity", None)
    assert should_skip("2026-01-01 00:00:00+00", "2026-01-01 00:00:00+00")
    assert not should_skip("2026-01-02 00:00:00+00", "2026-01-01 00:00:00+00")
    assert not should_skip("2026-01-02 00:00:00+00", None)


def test_backup_fails_closed_without_pg_dump(monkeypatch):
    monkeypatch.setattr(
        "phase3_rag.backup.resolve_dsn", lambda *a, **k: "postgresql://example"
    )
    monkeypatch.setattr("phase3_rag.backup.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="pg_dump"):
        run_backup()


def test_list_jsonl_files_rglob(tmp_path: Path):
    nested = tmp_path / "hvac" / "manual"
    nested.mkdir(parents=True)
    (nested / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "legacy.jsonl").write_text("{}\n", encoding="utf-8")
    names = {p.name for p in list_jsonl_files(tmp_path)}
    assert names == {"chunks.jsonl", "legacy.jsonl"}


def test_jsonl_retriever_loads_nested_and_keeps_parents(tmp_path: Path):
    nested = tmp_path / "manual"
    nested.mkdir()
    parent = Chunk(
        text="parent body", source_type="pdf", source_id="manual.pdf",
        chunk_index=0, is_parent=True, section="Safety",
    )
    child = Chunk(
        text="child body", source_type="pdf", source_id="manual.pdf",
        chunk_index=1, parent_id=parent.chunk_id, section="Safety",
    )
    write_jsonl([parent, child], nested / "chunks.jsonl")
    cfg = {
        "embedding_model": "unused",
        "reranker_model": "unused",
        "retrieval": {
            "top_k_vector": 5, "top_k_bm25": 5, "top_k_final": 3,
            "use_parent_expansion": True,
        },
        "serve": {"ctx": 128},
    }
    retr = JsonlRetriever.__new__(JsonlRetriever)
    retr.cfg = cfg
    retr.processed_dir = tmp_path
    retr._chunks = []
    retr._parents = {}
    retr._vecs = None
    retr._embedder = None
    retr._reranker = None
    retr._load_chunks()
    assert len(retr._chunks) == 1
    assert parent.chunk_id in retr._parents
    hits = [
        Hit(child.chunk_id, child.text, 1.0, parent_id=parent.chunk_id),
    ]
    expanded = retr._expand_parents(hits)
    assert expanded[0].text == "parent body"


def test_collect_processed_jsonl_nested_and_flat(tmp_path: Path):
    pdfs = tmp_path / "raw"
    (pdfs / "hvac").mkdir(parents=True)
    (pdfs / "hvac" / "manual.pdf").write_bytes(b"%PDF")
    (pdfs / "other.pdf").write_bytes(b"%PDF")
    out = tmp_path / "processed"
    (out / "hvac" / "manual").mkdir(parents=True)
    (out / "hvac" / "manual" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (out / "hvac" / "manual" / "parents.jsonl").write_text("{}\n", encoding="utf-8")
    (out / "other.jsonl").write_text("{}\n", encoding="utf-8")
    got = {p.name for p in collect_processed_jsonl(out, pdfs)}
    assert got == {"chunks.jsonl", "parents.jsonl", "other.jsonl"}
