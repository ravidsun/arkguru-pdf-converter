"""
Reciprocal Rank Fusion (RRF) -- combine multiple ranked result lists into one.

Shared by the Phase 3 retrievers to fuse dense (vector) and lexical (BM25/FTS)
runs. Each "run" is an ordered sequence of rows and a row's first element
(index 0) is its unique id (``chunk_id``). A row's fused score is the sum of
``1 / (k + rank + 1)`` across the runs it appears in; rows are returned sorted
by fused score descending. ``k`` damps the influence of lower-ranked hits
(k=60 is the value from the original RRF paper).
"""
from __future__ import annotations

from typing import Any, Sequence


def reciprocal_rank_fusion(runs: Sequence[Sequence[Any]], k: int = 60) -> list:
    """Fuse ranked ``runs`` into a single ranked list of rows.

    Args:
        runs: an iterable of runs; each run is an ordered sequence of rows,
            where ``row[0]`` is the row's unique id.
        k: RRF damping constant (defaults to 60).

    Returns:
        The unique rows, ordered by descending fused RRF score.
    """
    scores: dict = {}
    rows: dict = {}
    for run in runs:
        for rank, r in enumerate(run):
            cid = r[0]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            rows[cid] = r
    return [rows[c] for c in sorted(scores, key=scores.get, reverse=True)]
