"""CLI defaults for the postgres embed handoff."""
from __future__ import annotations

import pytest

from phase3_rag.worker import main as worker_main
from phase3_rag.run_pdfs import main as run_pdfs_main
from phase3_rag.index import main as index_main


def test_worker_once_defaults_to_hashing(monkeypatch):
    seen = {}

    def fake_make(_cfg, backend, _model, _dim, _batch):
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
    assert seen["backend"] == "hashing"
    assert seen["mode"] == "once"


def test_worker_daemon_defaults_to_sentence_transformer(monkeypatch):
    seen = {}

    def fake_make(_cfg, backend, _model, _dim, _batch):
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


def test_run_pdfs_postgres_requires_pg_dsn(monkeypatch):
    monkeypatch.delenv("PG_DSN", raising=False)
    with pytest.raises(SystemExit, match="PG_DSN"):
        run_pdfs_main(["--pdfs", ".", "--sink", "postgres"])


def test_index_embed_only_skips_jsonl(monkeypatch, tmp_path):
    seen = {}

    def fake_build(_cfg, corpus, *, embed_only, embedder):
        seen["embed_only"] = embed_only
        seen["embedder"] = embedder
        seen["corpus"] = corpus

    monkeypatch.setattr("phase3_rag.index.build", fake_build)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("phase3: {}\n")
    index_main(["--config", str(cfg), "--embed-only", "--embedder", "hashing"])
    assert seen == {"embed_only": True, "embedder": "hashing",
                    "corpus": "data/processed/corpus.jsonl"}
