"""
Phase 1, step 2: Blocks -> Chunk records.

Implements three complementary strategies (selectable via config):

  1. "structure"  -- structure-aware. Group blocks under their nearest heading,
                     then pack into ~target-token windows with % overlap.
                     Never crosses a heading boundary. Good default.

  2. "parent_child" -- emit BOTH a large "parent" chunk (whole section, capped)
                     and the small "child" windows inside it. Retrieval matches
                     on small children (precise) but you can fetch the parent for
                     full context at answer time (parent_id links them).

  3. "semantic"   -- split section text at sentence boundaries where adjacent
                     sentence groups drift in meaning. Uses an embedding model if
                     available; otherwise falls back to fixed sentence windows.

All strategies honor `target_tokens` (300-800) and `overlap_pct` (10-15%).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from common.schema import Chunk
from common.tokenizer import count_tokens
from common.chunking import split_sentences as _sentences, pack_windows as _pack_windows
from .extract import Block, Document


def _pages(blocks: list[Block]) -> list[int]:
    return [b.page for b in blocks if b.page is not None]


def _page_for_window(text: str, blocks: list[Block], fallback: Optional[int]) -> Optional[int]:
    """Page of the first body block whose text appears in this window."""
    for b in blocks:
        snippet = (b.text or "").strip()[:80]
        if snippet and snippet in text and b.page is not None:
            return b.page
    return fallback


def _section_extra(doc: Document, body: list[Block], extra: Optional[dict] = None) -> dict:
    extra = dict(extra or {})
    meta = doc.meta or {}
    extra.setdefault("backend", meta.get("backend"))
    extra.setdefault("sha256", meta.get("sha256"))
    extra.setdefault("page_count", meta.get("page_count"))
    pages = _pages(body)
    if pages:
        extra["page_start"] = min(pages)
        extra["page_end"] = max(pages)
    return extra


def reindex_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Assign contiguous 0-based ``chunk_index``. Does not rewrite ``chunk_id``."""
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


def _group_by_section(blocks: list[Block]) -> list[tuple[Optional[str], list[Block]]]:
    """Return [(heading, body_blocks), ...] preserving order."""
    sections: list[tuple[Optional[str], list[Block]]] = []
    cur_head: Optional[str] = None
    cur: list[Block] = []
    for b in blocks:
        if b.is_heading:
            if cur:
                sections.append((cur_head, cur))
                cur = []
            cur_head = b.heading
        else:
            cur.append(b)
    if cur:
        sections.append((cur_head, cur))
    return sections



# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------
def chunk_document(
    doc: Document,
    strategy: str = "structure",
    target_tokens: int = 400,
    overlap_pct: float = 0.15,
    parent_max_tokens: int = 2000,
    min_tokens: int = 80,
    semantic_embedder: Optional[Callable[[list[str]], list]] = None,
) -> list[Chunk]:
    overlap_tokens = max(1, int(target_tokens * overlap_pct))
    sections = _group_by_section(doc.blocks)
    chunks: list[Chunk] = []
    idx = 0

    for heading, body in sections:
        # Tables/figures are structurally different from prose: packing them
        # into sentence-based windows would garble or split them mid-row/mid-
        # caption, so they become their own standalone chunks and are kept
        # out of the prose windowing below.
        prose_blocks = [b for b in body if b.block_type == "text"]
        special_blocks = [b for b in body if b.block_type in ("table", "figure")]

        section_text = "\n".join(b.text for b in prose_blocks).strip()
        pages = _pages(body)
        page = pages[0] if pages else None
        extra = _section_extra(doc, body)

        parent_id = None
        if strategy == "parent_child" and section_text:
            parent = Chunk(
                text=_cap(section_text, parent_max_tokens),
                source_type="pdf", source_id=doc.source_id, chunk_index=idx,
                title=doc.title, section=heading, page=page, lang=doc.lang,
                is_parent=True,
                token_count=count_tokens(section_text),
                extra=dict(extra),
            )
            chunks.append(parent)
            parent_id = parent.chunk_id
            idx += 1

        if section_text:
            sents = _sentences(section_text)
            if sents:
                if strategy == "semantic" and semantic_embedder is not None:
                    windows = _semantic_windows(sents, target_tokens, overlap_tokens,
                                                semantic_embedder)
                else:
                    windows = _pack_windows(sents, target_tokens, overlap_tokens,
                                            min_tokens=min_tokens)

                for w_text, ov in windows:
                    body = _prefixed(heading, w_text)
                    chunks.append(Chunk(
                        text=body,
                        source_type="pdf", source_id=doc.source_id, chunk_index=idx,
                        title=doc.title, section=heading,
                        page=_page_for_window(w_text, prose_blocks, page),
                        lang=doc.lang,
                        parent_id=parent_id, is_parent=False,
                        token_count=count_tokens(body), overlap_tokens=ov,
                        extra=dict(extra),
                    ))
                    idx += 1

        for b in special_blocks:
            body = _prefixed(heading, b.text)
            spec = dict(extra)
            spec["block_type"] = b.block_type
            chunks.append(Chunk(
                text=body,
                source_type="pdf", source_id=doc.source_id, chunk_index=idx,
                title=doc.title, section=heading, page=b.page, lang=doc.lang,
                parent_id=parent_id, is_parent=False,
                token_count=count_tokens(body),
                extra=spec,
            ))
            idx += 1

    return chunks


def _prefixed(heading: Optional[str], body: str) -> str:
    """Put the section heading on the chunk body so embeddings are not orphan fragments."""
    h = (heading or "").strip()
    if not h:
        return body
    stripped = body.lstrip()
    if stripped.startswith(h):
        return body
    return f"{h}\n\n{body}"


def _cap(text: str, max_tokens: int) -> str:
    from common.tokenizer import truncate_to_tokens
    return truncate_to_tokens(text, max_tokens)


def _semantic_windows(sents, target_tokens, overlap_tokens, embedder):
    """Break where cosine similarity between adjacent sentences drops.

    `embedder(list[str]) -> list[vector]`. Kept dependency-free here: any
    callable returning vectors works (sentence-transformers, e5, bge, ...).
    """
    import numpy as np
    vecs = np.asarray(embedder(sents), dtype="float32")
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    # breakpoint where similarity is in the lowest quartile
    if len(sims):
        thresh = float(np.quantile(sims, 0.25))
    else:
        thresh = 0.0

    windows: list[tuple[str, int]] = []
    cur: list[str] = []
    tok = 0
    for i, s in enumerate(sents):
        t = count_tokens(s)
        boundary = i > 0 and sims[i - 1] < thresh
        if cur and (tok + t > target_tokens or boundary):
            windows.append((" ".join(cur), 0 if not windows else overlap_tokens))
            cur, tok = [], 0
        cur.append(s)
        tok += t
    if cur:
        windows.append((" ".join(cur), 0 if not windows else overlap_tokens))
    return windows
