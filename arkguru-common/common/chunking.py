"""
Shared, framework-free chunking primitives used by BOTH Phase 1 (PDF) and
Phase 2 (web). Living in `common/` keeps each phase repo standalone.
"""
from __future__ import annotations

import re
from .tokenizer import count_tokens

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ABBREV_RE = re.compile(
    r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|approx|fig|figs|eq|eqs"
    r"|e\.g|i\.e|cf|al|no|vol|pp|ch|sec)\.$",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, merging false splits caused by common
    abbreviations (e.g. "Fig. 2", "Dr. Smith", "e.g. this") back together so
    a heading like "Fig. 2 shows..." isn't chopped mid-thought."""
    raw = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    merged: list[str] = []
    for s in raw:
        if merged and _ABBREV_RE.search(merged[-1]):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def pack_windows(
    sentences: list[str],
    target_tokens: int,
    overlap_tokens: int,
    min_tokens: int = 0,
) -> list[tuple[str, int]]:
    """Greedy-pack sentences into ~target_tokens windows with sentence overlap.

    If `min_tokens` is set, a trailing window smaller than that is merged into
    the previous one rather than shipped as an accuracy-hurting sliver (this
    avoids tiny, low-context chunks that rank poorly at retrieval time).

    Returns list of (window_text, overlap_tokens_used).
    """
    windows: list[tuple[str, int]] = []
    i, n = 0, len(sentences)
    while i < n:
        cur: list[str] = []
        tok = 0
        j = i
        while j < n:
            t = count_tokens(sentences[j])
            if tok + t > target_tokens and cur:
                break
            cur.append(sentences[j]); tok += t; j += 1
        windows.append((" ".join(cur), 0 if not windows else overlap_tokens))
        if j >= n:
            break
        back_tok, k = 0, j
        while k > i and back_tok < overlap_tokens:
            k -= 1; back_tok += count_tokens(sentences[k])
        i = max(k, i + 1)

    if min_tokens > 0 and len(windows) > 1:
        last_text, last_ov = windows[-1]
        if count_tokens(last_text) < min_tokens:
            prev_text, prev_ov = windows[-2]
            windows[-2] = (prev_text + " " + last_text, prev_ov)
            windows.pop()

    return windows
