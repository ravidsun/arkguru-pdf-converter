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

All strategies honor `target_tokens` (default 400), a hard `max_tokens` cap
(default 512, bge-m3), and `overlap_pct` (10-15%). Tables are linearised
``header: value`` rows and split by row; they are never left as pipe dumps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from common.chunking import pack_windows as _pack_windows
from common.chunking import split_for_packing
from common.schema import Chunk
from common.tables import (
    is_pipe_heavy,
    linearize_table,
    pack_table_windows,
    rows_from_markdown,
    strip_markdown_tables,
)
from common.text import body_without_heading, clean_text
from common.tokenizer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TARGET_TOKENS,
    count_tokens,
    split_to_max_tokens,
    truncate_to_tokens,
)
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


def _group_by_section(
    blocks: list[Block],
) -> list[tuple[Optional[str], int, Optional[str], list[Block]]]:
    """Return [(heading, level, parent_heading, body_blocks), ...] in order."""
    sections: list[tuple[Optional[str], int, Optional[str], list[Block]]] = []
    stack: list[tuple[int, str]] = []
    cur_head: Optional[str] = None
    cur_level = 0
    cur_parent: Optional[str] = None
    cur: list[Block] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            sections.append((cur_head, cur_level, cur_parent, cur))
            cur = []

    for b in blocks:
        if b.is_heading:
            flush()
            level = b.heading_level or 1
            heading = b.heading or b.text
            while stack and stack[-1][0] >= level:
                stack.pop()
            cur_parent = stack[-1][1] if stack else None
            if heading:
                stack.append((level, heading))
            cur_head = heading
            cur_level = level
        else:
            cur.append(b)
    flush()
    return sections


def _section_prose_chars(body: list[Block]) -> int:
    return sum(len((b.text or "").strip()) for b in body if b.block_type == "text")


def _same_heading_chain(
    a: tuple[Optional[str], int, Optional[str], list[Block]],
    b: tuple[Optional[str], int, Optional[str], list[Block]],
) -> bool:
    a_head, _a_lvl, a_parent, _a_body = a
    b_head, _b_lvl, b_parent, _b_body = b
    if a_parent and a_parent == b_parent:
        return True
    if a_head and a_head == b_head:
        return True
    if a_head and a_head == b_parent:
        return True
    if b_head and b_head == a_parent:
        return True
    if not a_head and not b_head:
        return True
    return False


def _merge_small_sections(
    sections: list[tuple[Optional[str], int, Optional[str], list[Block]]],
    min_chars: int,
) -> list[tuple[Optional[str], list[Block]]]:
    """Merge tiny sections that share a heading chain (or adjacent leftovers)."""
    if not sections:
        return []
    secs: list[list] = [
        [h, lvl, parent, list(body)] for h, lvl, parent, body in sections
    ]

    def chars(item: list) -> int:
        return _section_prose_chars(item[3])

    def as_tuple(item: list) -> tuple:
        return (item[0], item[1], item[2], item[3])

    if min_chars > 0 and len(secs) > 1:
        i = 0
        while i < len(secs):
            if chars(secs[i]) >= min_chars:
                i += 1
                continue
            if i > 0 and _same_heading_chain(as_tuple(secs[i - 1]), as_tuple(secs[i])):
                secs[i - 1][3].extend(secs[i][3])
                secs.pop(i)
                continue
            if i + 1 < len(secs) and _same_heading_chain(
                as_tuple(secs[i]), as_tuple(secs[i + 1])
            ):
                secs[i][3].extend(secs[i + 1][3])
                secs.pop(i + 1)
                continue
            i += 1

        i = 0
        while i < len(secs) - 1:
            if chars(secs[i]) < min_chars and chars(secs[i + 1]) < min_chars:
                secs[i][3].extend(secs[i + 1][3])
                secs.pop(i + 1)
                continue
            i += 1

    return [(item[0], item[3]) for item in secs]


def _normalize_table_text(text: str) -> str:
    """Linearise leftover markdown / pipe dumps; otherwise clean."""
    raw = text or ""
    if is_pipe_heavy(raw) or "|" in raw:
        _prose, tables = strip_markdown_tables(raw)
        if tables:
            return linearize_table(tables[0])
        rows = rows_from_markdown(raw)
        if rows:
            return linearize_table(rows)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if lines and all("|" in ln for ln in lines):
            fenced = "\n".join(
                ln if ln.strip().startswith("|") else f"| {ln.strip()} |"
                for ln in lines
            )
            rows = rows_from_markdown(fenced)
            if rows:
                return linearize_table(rows)
    return clean_text(raw)


def _keep_chunk(
    text: str,
    heading: Optional[str],
    block_type: str,
    min_chunk_chars: int,
    min_table_chars: int,
    min_figure_chars: int,
) -> bool:
    body = body_without_heading(text, heading)
    if not body.strip():
        return False
    n = len(body)
    if block_type == "table":
        return min_table_chars <= 0 or n >= min_table_chars
    if block_type == "figure":
        return min_figure_chars <= 0 or n >= min_figure_chars
    return min_chunk_chars <= 0 or n >= min_chunk_chars


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------
def chunk_document(
    doc: Document,
    strategy: str = "structure",
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_pct: float = 0.15,
    parent_max_tokens: int = 2000,
    min_tokens: int = 80,
    semantic_embedder: Optional[Callable[[list[str]], list]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_chunk_chars: int = 80,
    min_table_chars: int = 40,
    min_figure_chars: int = 40,
    merge_small_sections: bool = True,
    keep_tables: bool = True,
) -> list[Chunk]:
    overlap_tokens = max(1, int(target_tokens * overlap_pct))
    cap = max(1, max_tokens)
    grouped = _group_by_section(doc.blocks)
    if merge_small_sections:
        sections = _merge_small_sections(grouped, min_chunk_chars)
    else:
        sections = [(h, body) for h, _lvl, _parent, body in grouped]
    chunks: list[Chunk] = []
    idx = 0

    for heading, body in sections:
        # Tables/figures are structurally different from prose: packing them
        # into sentence-based windows would garble or split them mid-row/mid-
        # caption, so they become their own standalone chunks and are kept
        # out of the prose windowing below.
        prose_blocks: list[Block] = []
        special_blocks: list[Block] = []
        for b in body:
            kind = b.block_type or "text"
            if kind == "text" and is_pipe_heavy(b.text or ""):
                special_blocks.append(replace(
                    b, block_type="table", text=_normalize_table_text(b.text),
                ))
            elif kind == "table":
                special_blocks.append(replace(b, text=_normalize_table_text(b.text)))
            elif kind == "figure":
                special_blocks.append(replace(b, text=clean_text(b.text)))
            else:
                cleaned = clean_text(b.text)
                if cleaned:
                    prose_blocks.append(replace(b, text=cleaned))

        if not keep_tables:
            special_blocks = [b for b in special_blocks if b.block_type != "table"]

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
            units = split_for_packing(section_text)
            if units:
                if strategy == "semantic" and semantic_embedder is not None:
                    windows = _semantic_windows(
                        units, target_tokens, overlap_tokens,
                        semantic_embedder, max_tokens=cap,
                    )
                else:
                    windows = _pack_windows(
                        units, target_tokens, overlap_tokens,
                        min_tokens=min_tokens, max_tokens=cap,
                    )

                for w_text, ov in windows:
                    text = _prefixed(heading, w_text)
                    if not _keep_chunk(
                        text, heading, "text",
                        min_chunk_chars, min_table_chars, min_figure_chars,
                    ):
                        continue
                    chunks.append(Chunk(
                        text=text,
                        source_type="pdf", source_id=doc.source_id, chunk_index=idx,
                        title=doc.title, section=heading,
                        page=_page_for_window(w_text, prose_blocks, page),
                        lang=doc.lang,
                        parent_id=parent_id, is_parent=False,
                        token_count=count_tokens(text), overlap_tokens=ov,
                        extra=dict(extra),
                    ))
                    idx += 1

        for b in special_blocks:
            kind = b.block_type
            raw = b.text or ""
            if kind == "table":
                pieces = pack_table_windows(
                    raw, target_tokens=target_tokens, max_tokens=cap,
                )
                if not pieces:
                    pieces = [raw] if raw.strip() else []
            else:
                if count_tokens(raw) > cap:
                    pieces = split_to_max_tokens(raw, cap, overlap_tokens=overlap_tokens)
                else:
                    pieces = [raw] if raw.strip() else []
            for piece in pieces:
                if count_tokens(piece) > cap:
                    for hard in split_to_max_tokens(piece, cap, overlap_tokens=0):
                        text = _prefixed(heading, hard)
                        if not _keep_chunk(
                            text, heading, kind,
                            min_chunk_chars, min_table_chars, min_figure_chars,
                        ):
                            continue
                        spec = dict(extra)
                        spec["block_type"] = kind
                        chunks.append(Chunk(
                            text=text,
                            source_type="pdf", source_id=doc.source_id, chunk_index=idx,
                            title=doc.title, section=heading, page=b.page, lang=doc.lang,
                            parent_id=parent_id, is_parent=False,
                            token_count=count_tokens(text),
                            extra=spec,
                        ))
                        idx += 1
                    continue
                text = _prefixed(heading, piece)
                if not _keep_chunk(
                    text, heading, kind,
                    min_chunk_chars, min_table_chars, min_figure_chars,
                ):
                    continue
                spec = dict(extra)
                spec["block_type"] = kind
                chunks.append(Chunk(
                    text=text,
                    source_type="pdf", source_id=doc.source_id, chunk_index=idx,
                    title=doc.title, section=heading, page=b.page, lang=doc.lang,
                    parent_id=parent_id, is_parent=False,
                    token_count=count_tokens(text),
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
    return truncate_to_tokens(text, max_tokens)


def _semantic_windows(sents, target_tokens, overlap_tokens, embedder, max_tokens=DEFAULT_MAX_TOKENS):
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

    cap = max(1, max_tokens)
    target = min(target_tokens, cap)
    windows: list[tuple[str, int]] = []
    cur: list[str] = []
    tok = 0
    for i, s in enumerate(sents):
        t = count_tokens(s)
        boundary = i > 0 and sims[i - 1] < thresh
        if cur and (tok + t > target or boundary):
            windows.append((" ".join(cur), 0 if not windows else overlap_tokens))
            cur, tok = [], 0
        if t > cap:
            if cur:
                windows.append((" ".join(cur), 0 if not windows else overlap_tokens))
                cur, tok = [], 0
            for piece in split_to_max_tokens(s, cap, overlap_tokens=overlap_tokens):
                windows.append((piece, 0 if not windows else overlap_tokens))
            continue
        cur.append(s)
        tok += t
    if cur:
        windows.append((" ".join(cur), 0 if not windows else overlap_tokens))
    return windows
