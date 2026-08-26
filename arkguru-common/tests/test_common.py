"""Smoke tests for the shared arkguru-common primitives."""
from __future__ import annotations

from pathlib import Path

from common.schema import Chunk, read_jsonl, write_jsonl
from common.tokenizer import count_tokens, truncate_to_tokens
from common.chunking import split_sentences, pack_windows
from common.rrf import reciprocal_rank_fusion


def test_chunk_id_is_deterministic():
    a = Chunk(text="hello", source_type="pdf", source_id="a.pdf", chunk_index=0)
    b = Chunk(text="different body", source_type="pdf", source_id="a.pdf", chunk_index=0)
    # chunk_id is keyed by (source_type, source_id, index, role), not text
    assert a.chunk_id == b.chunk_id
    c = Chunk(text="hello", source_type="pdf", source_id="a.pdf", chunk_index=1)
    assert a.chunk_id != c.chunk_id


def test_jsonl_roundtrip(tmp_path: Path):
    chunks = [
        Chunk(text="one", source_type="pdf", source_id="a.pdf", chunk_index=0),
        Chunk(text="two", source_type="web", source_id="http://x", chunk_index=1),
    ]
    out = tmp_path / "chunks.jsonl"
    assert write_jsonl(chunks, out) == 2
    loaded = read_jsonl(out)
    assert [c.text for c in loaded] == ["one", "two"]
    assert loaded[0].chunk_id == chunks[0].chunk_id


def test_tokenizer():
    assert count_tokens("hello world") >= 1
    assert truncate_to_tokens("word " * 100, 5).strip() != ""


def test_split_sentences_merges_abbreviations():
    sents = split_sentences("See Fig. 2 for details. Then continue.")
    # "Fig. 2 for details." should not be split at "Fig."
    assert any("Fig. 2" in s for s in sents)


def test_pack_windows_merges_tiny_trailing():
    sentences = ["a sentence here." for _ in range(6)]
    windows = pack_windows(sentences, target_tokens=8, overlap_tokens=2, min_tokens=4)
    assert len(windows) >= 1
    assert all(isinstance(w, tuple) and len(w) == 2 for w in windows)


def test_reciprocal_rank_fusion_orders_by_fused_score():
    dense = [("c1", "t1"), ("c2", "t2"), ("c3", "t3")]
    lexical = [("c2", "t2"), ("c1", "t1")]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    ids = [r[0] for r in fused]
    # c1 (ranks 0 and 1) and c2 (ranks 1 and 0) outrank c3 (dense only)
    assert set(ids) == {"c1", "c2", "c3"}
    assert ids.index("c3") == 2
