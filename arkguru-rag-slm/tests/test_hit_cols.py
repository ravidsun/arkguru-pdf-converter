"""Keep HybridRetriever column indexes aligned with common.datastore._HIT_COLS."""
from __future__ import annotations

from common.datastore import _HIT_COLS
from phase3_rag.retrieve import _C
from phase3_rag.run_pdfs import _HIT


def test_hit_col_indexes_match_datastore():
    assert _C["score"] == len(_HIT_COLS)
    for i, name in enumerate(_HIT_COLS):
        assert _C[name] == i, f"{name} expected index {i}, got {_C[name]}"


def test_run_pdfs_hit_map_matches_datastore():
    assert _HIT["score"] == len(_HIT_COLS)
    for i, name in enumerate(_HIT_COLS):
        assert _HIT[name] == i
