"""CLI defaults for the postgres embed handoff."""
from __future__ import annotations

import pytest

from phase3_rag.embedder import DEFAULT_BACKEND
from phase3_rag.worker import main as worker_main
from phase3_rag.run_pdfs import main as run_pdfs_main
from phase3_rag.index import main as index_main
from phase3_rag.embed_datastore import main as embed_main


def test_worker_once_defaults_to_sentence_transformer(monkeypatch):
    seen = {}

    def fake_make(_cfg, backend, _model, _dim, _batch, **_kw):
        seen["backend"] = backend
        return lambda: "ok"

    class FakeWorker:
        def __init__(self, *a, **k):
            pass
        def run_once(self):
            seen["mode"] = "once"
        def run_forever(self):
            seen["mode"] = "forever"

    monkeypatch.setattr("phase3_rag.worker.make_run_once", fake_make)
    monkeypatch.setattr("phase3_rag.worker.Worker", FakeWorker)
    worker_main(["--once"])
    assert seen["backend"] == DEFAULT_BACKEND
    assert seen["mode"] == "once"


def test_worker_daemon_defaults_to_sentence_transformer(monkeypatch):
    seen = {}

    def fake_make(_cfg, backend, _model, _dim, _batch, **_kw):
        seen["backend"] = backend
        return lambda: "ok"

    class FakeWorker:
        def __init__(self, *a, **k):
            pass
        def run_once(self):
            seen["mode"] = "once"
        def run_forever(self):
            seen["mode"] = "forever"

    monkeypatch.setattr("phase3_rag.worker.make_run_once", fake_make)
    monkeypatch.setattr("phase3_rag.worker.Worker", FakeWorker)
    worker_main([])
    assert seen["backend"] == "sentence_transformer"
    assert seen["mode"] == "forever"


def test_worker_hashing_without_allow_exits():
    with pytest.raises(SystemExit, match="allow-hashing"):
        worker_main(["--once", "--embedder", "hashing"])


def test_run_pdfs_postgres_requires_pg_dsn(monkeypatch):
    monkeypatch.delenv("PG_DSN", raising=False)
    with pytest.raises(SystemExit, match="PG_DSN"):
        run_pdfs_main(["--pdfs", ".", "--sink", "postgres"])


def test_index_embed_only_skips_jsonl(monkeypatch, tmp_path):
    seen = {}

    def fake_build(_cfg, corpus, *, embed_only, embedder, **kw):
        seen["embed_only"] = embed_only
        seen["embedder"] = embedder
        seen["corpus"] = corpus
        seen["allow_hashing"] = kw.get("allow_hashing")

    monkeypatch.setattr("phase3_rag.index.build", fake_build)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("phase3: {}\n")
    index_main(["--config", str(cfg), "--embed-only",
                "--embedder", "hashing", "--allow-hashing"])
    assert seen == {"embed_only": True, "embedder": "hashing",
                    "corpus": "data/processed/corpus.jsonl",
                    "allow_hashing": True}


def test_index_default_embedder_is_sentence_transformer(monkeypatch, tmp_path):
    seen = {}

    def fake_build(_cfg, corpus, *, embed_only, embedder, **kw):
        seen["embedder"] = embedder
        seen["allow_hashing"] = kw.get("allow_hashing")

    monkeypatch.setattr("phase3_rag.index.build", fake_build)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("phase3: {}\n")
    index_main(["--config", str(cfg), "--embed-only"])
    assert seen["embedder"] == "sentence_transformer"
    assert seen["allow_hashing"] is False


def test_embed_datastore_refuses_hashing(monkeypatch):
    monkeypatch.setattr(
        "phase3_rag.embed_datastore.open_chunk_store",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not connect")),
    )
    with pytest.raises(SystemExit, match="allow-hashing"):
        embed_main(["--embedder", "hashing"])
    assert HASHING_DATASTORE_ERROR


def test_embed_datastore_writes_label_and_skips_invalid(monkeypatch):
    seen = {}

    class FakeEmb:
        backend = "hashing"
        label = "hashing"
        dim = 8
        device = "cpu"

        def __init__(self, **kw):
            seen["embedder_kw"] = kw

        def encode(self, texts, batch_size=None):
            import numpy as np
            out = np.zeros((len(texts), 8), dtype="float32")
            if texts and texts[0] == "hello":
                out[0, 0] = 1.0
            return out

    class FakeStore:
        schema = "public"

        def count_embeddings_to_fill(self, **kw):
            seen["count_kw"] = kw
            return 2

        def iter_chunks_for_embedding(self, **kw):
            seen["iter_kw"] = kw
            yield [("ok", "hello"), ("bad", "你好")]

        def update_embeddings(self, ids, vectors, model):
            seen["update"] = (list(ids), model)
            return len(ids)

    monkeypatch.setattr("phase3_rag.embed_datastore.Embedder", FakeEmb)
    monkeypatch.setattr(
        "phase3_rag.embed_datastore.open_chunk_store",
        lambda *a, **k: FakeStore(),
    )
    embed_main(["--embedder", "hashing", "--allow-hashing",
                "--reembed", "--schema", "public", "--device", "cpu"])
    assert seen["embedder_kw"]["backend"] == "hashing"
    assert seen["count_kw"]["reembed"] is True
    assert seen["count_kw"]["model"] == "hashing"
    assert seen["iter_kw"]["reembed"] is True
    assert seen["update"] == (["ok"], "hashing")
