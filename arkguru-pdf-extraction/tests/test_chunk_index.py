"""Phase 1 chunk_index helpers (no live Postgres)."""
from __future__ import annotations

from common.schema import Chunk
from phase1_pdf.chunk import _page_for_window, reindex_chunks
from phase1_pdf.extract import Block


def test_reindex_contiguous_keeps_chunk_id():
    chunks = [
        Chunk(text="a", source_type="pdf", source_id="a.pdf", chunk_index=0),
        Chunk(text="b", source_type="pdf", source_id="a.pdf", chunk_index=3),
    ]
    old_ids = [c.chunk_id for c in chunks]
    reindex_chunks(chunks)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.chunk_id for c in chunks] == old_ids


def test_page_for_window_uses_matching_block():
    blocks = [
        Block(text="Hello world on page one.", page=1),
        Block(text="Second page body here.", page=2),
    ]
    assert _page_for_window("Second page body here. more", blocks, 1) == 2
    assert _page_for_window("unrelated window text", blocks, 1) == 1
