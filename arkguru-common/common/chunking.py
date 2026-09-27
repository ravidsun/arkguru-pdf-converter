"""
Shared, framework-free chunking primitives used by BOTH Phase 1 (PDF) and
Phase 2 (web). Living in `common/` keeps each phase repo standalone.

Split order: paragraphs, then sentences (Latin ``.!?``, danda ``।`` / ``॥``,
and newline boundaries), then a hard token-cap fallback. ``pack_windows``
never emits a window above ``max_tokens`` (default 510 body tokens:
the bge-m3 512 window minus XLM-R specials).
"""
from __future__ import annotations

import re

from .tokenizer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TARGET_TOKENS,
    count_tokens,
    effective_max_tokens,
    split_to_max_tokens,
)

# Latin terminator + whitespace (any next char — not only A-Z, so OCR and
# transliteration still split). Danda / double-danda. Newline as a boundary
# so heading-less, punctuation-poor sections still break.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[।॥])\s*"
    r"|(?<=\n)"
)
_ABBREV_RE = re.compile(
    r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|approx|fig|figs|eq|eqs"
    r"|e\.g|i\.e|cf|al|no|vol|pp|ch|sec)\.$",
    re.IGNORECASE,
)
_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; drop empty paragraphs."""
    if not text:
        return []
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Split into sentences, merging false splits caused by common
    abbreviations (e.g. "Fig. 2", "Dr. Smith", "e.g. this") back together so
    a heading like "Fig. 2 shows..." isn't chopped mid-thought.

    Also splits on danda terminators and newline boundaries.
    """
    if not text:
        return []
    raw = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    merged: list[str] = []
    for s in raw:
        if merged and _ABBREV_RE.search(merged[-1]):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def split_for_packing(text: str) -> list[str]:
    """Paragraphs first, then sentences. Used by PDF and web chunkers."""
    units: list[str] = []
    paragraphs = split_paragraphs(text) or ([text.strip()] if text and text.strip() else [])
    for para in paragraphs:
        sents = split_sentences(para)
        units.extend(sents if sents else [para])
    return units


def pack_windows(
    sentences: list[str],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = 0,
    min_tokens: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[tuple[str, int]]:
    """Greedy-pack sentences into ~target_tokens windows with sentence overlap.

    ``max_tokens`` is a hard cap: a unit over the cap is token-split first,
    and a trailing sliver is not merged if that would exceed the cap.

    If `min_tokens` is set, a trailing window smaller than that is merged into
    the previous one rather than shipped as an accuracy-hurting sliver (this
    avoids tiny, low-context chunks that rank poorly at retrieval time).

    Returns list of (window_text, overlap_tokens_used).
    """
    cap = effective_max_tokens(max_tokens)
    target = min(max(1, target_tokens), cap)
    overlap_tokens = max(0, min(overlap_tokens, cap - 1))

    units: list[str] = []
    for s in sentences:
        piece = (s or "").strip()
        if not piece:
            continue
        if count_tokens(piece) > cap:
            units.extend(split_to_max_tokens(piece, cap, overlap_tokens=overlap_tokens))
        else:
            units.append(piece)

    windows: list[tuple[str, int]] = []
    i, n = 0, len(units)
    while i < n:
        cur: list[str] = []
        tok = 0
        j = i
        while j < n:
            t = count_tokens(units[j])
            if cur and tok + t > target:
                break
            if not cur and t > cap:
                # Should be unreachable after the expand pass.
                units[j:j + 1] = split_to_max_tokens(units[j], cap, overlap_tokens=0)
                n = len(units)
                t = count_tokens(units[j])
            cur.append(units[j])
            tok += t
            j += 1
        if not cur:
            break
        windows.append((" ".join(cur), 0 if not windows else overlap_tokens))
        if j >= n:
            break
        back_tok, k = 0, j
        while k > i and back_tok < overlap_tokens:
            k -= 1
            back_tok += count_tokens(units[k])
        i = max(k, i + 1)

    if min_tokens > 0 and len(windows) > 1:
        last_text, last_ov = windows[-1]
        if count_tokens(last_text) < min_tokens:
            prev_text, prev_ov = windows[-2]
            merged = prev_text + " " + last_text
            if count_tokens(merged) <= cap:
                windows[-2] = (merged, prev_ov)
                windows.pop()

    return windows
