"""Dedup + idempotent re-ingest (no live Postgres)."""
from __future__ import annotations

from pathlib import Path

from common.quality import annotate_chunks
from common.schema import Chunk, content_hash
from phase1_pdf.extract import _file_sha256
from phase1_pdf.pipeline import Phase1Config, relative_source_id, run


class _FakeStore:
    def __init__(self):
        self.by_sha: dict[str, str] = {}
        self.aliases: list[tuple[str, str]] = []
        self.replaced: list[tuple[str, int]] = []
        self.schema_ok = False
        self.rows: dict[str, list[Chunk]] = {}

    def ensure_schema(self):
        self.schema_ok = True

    def find_canonical_source_by_sha256(self, sha256: str):
        return self.by_sha.get(sha256)

    def record_source_alias(self, canon: str, alias: str):
        self.aliases.append((canon, alias))
        return 1

    def replace_source(self, source_id: str, chunks):
        chunks = list(chunks)
        self.rows[source_id] = chunks
        self.replaced.append((source_id, len(chunks)))
        if chunks:
            sha = (chunks[0].extra or {}).get("sha256")
            if sha and sha not in self.by_sha:
                self.by_sha[sha] = source_id
        return len(chunks)

    def chunk_index_counts(self, source_id: str):
        return [(i, 1) for i, _ in enumerate(self.rows.get(source_id, []))]


def test_file_sha256_stable(tmp_path: Path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-1.4 same-bytes")
    assert _file_sha256(p) == _file_sha256(p)
    q = tmp_path / "copy" / "a.pdf"
    q.parent.mkdir()
    q.write_bytes(p.read_bytes())
    assert _file_sha256(p) == _file_sha256(q)
    assert relative_source_id(p, tmp_path) != relative_source_id(q, tmp_path)


def test_duplicate_sha256_skips_second_copy(tmp_path, monkeypatch):
    ingest = tmp_path / "raw"
    (ingest / "one").mkdir(parents=True)
    (ingest / "two").mkdir(parents=True)
    payload = b"%PDF-1.4 duplicate-file"
    (ingest / "one" / "book.pdf").write_bytes(payload)
    (ingest / "two" / "book.pdf").write_bytes(payload)

    store = _FakeStore()
    store.by_sha[_file_sha256(ingest / "one" / "book.pdf")] = "one/book.pdf"

    import phase1_pdf.pipeline as pipe

    def fake_open(_cfg):
        return store

    monkeypatch.setattr(
        "common.datastore_config.open_chunk_store", fake_open)
    # Avoid actually extracting PDFs.
    monkeypatch.setattr(pipe, "_process_one", lambda args: (
        "book.pdf", "one/book", "one/book.pdf", "x", []
    ))

    cfg = Phase1Config(input_dir=str(ingest), sink="postgres",
                       datastore_config="unused")
    run(cfg, pdfs=None)
    assert store.aliases
    assert store.aliases[0][1].endswith("two/book.pdf") or store.aliases[0][0] == "one/book.pdf"


def test_replace_source_same_payload_is_idempotent():
    store = _FakeStore()
    chunks = annotate_chunks([
        Chunk(text="The seventh house and Saturn. " * 6,
              source_type="pdf", source_id="a.pdf", chunk_index=1),
    ])
    n1 = store.replace_source("a.pdf", chunks)
    n2 = store.replace_source("a.pdf", chunks)
    assert n1 == n2 == 1
    assert len(store.rows["a.pdf"]) == 1


def test_content_hash_same_for_duplicate_text():
    a = content_hash("  same body  ")
    b = content_hash("same body")
    assert a == b
    assert a != content_hash("other")
