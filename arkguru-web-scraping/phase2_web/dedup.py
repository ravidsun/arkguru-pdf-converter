"""Near-duplicate collapse (MinHash LSH when datasketch is installed)."""
from __future__ import annotations

import hashlib
import logging
from typing import Iterable

from common.schema import Chunk

log = logging.getLogger("phase2.dedup")


def _md5(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def exact_dedup(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Drop exact-duplicate child text. Parents (if any) are kept."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.is_parent:
            out.append(c)
            continue
        key = _md5(c.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def near_dedup(chunks: list[Chunk], threshold: float = 0.9) -> list[Chunk]:
    """Collapse near-duplicate children (Jaccard ≥ ``threshold``).

    Falls back to exact-text dedup when ``datasketch`` is missing.
    """
    children = [c for c in chunks if not c.is_parent]
    parents = [c for c in chunks if c.is_parent]
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.info("datasketch not installed; exact-text dedup only")
        return parents + exact_dedup(children)

    if not children:
        return list(chunks)

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[Chunk] = []
    for i, c in enumerate(children):
        mh = MinHash(num_perm=128)
        tokens = set(c.text.lower().split())
        if not tokens:
            continue
        for t in tokens:
            mh.update(t.encode("utf-8"))
        key = f"{i}:{c.chunk_id}"
        if lsh.query(mh):
            continue
        lsh.insert(key, mh)
        kept.append(c)
    log.info("near-dedup %d -> %d children", len(children), len(kept))
    return parents + kept
