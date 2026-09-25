"""Query rewriting: variants, ordering, and fusion (no DB, no Ollama)."""
from __future__ import annotations

import pytest

from phase3_rag.rewrite import (
    LexiconRewriter,
    NoopRewriter,
    OllamaRewriter,
    build_rewriter,
)

_GROUPS = [
    ["\u015aani", "Shani", "Sani", "Saturn"],
    ["Bh\u0101va", "Bhava", "house"],
]


def test_noop_returns_query_unchanged():
    assert NoopRewriter().rewrite("what is a dasha") == ["what is a dasha"]


def test_lexicon_puts_the_declared_first_variant_first():
    """Variant 1 must be the accented spelling the corpus actually uses.

    Cycling by offset from the matched token left it unreachable: from "Sani",
    stepping through the group never lands on the accented form, which matches
    13,872 chunks against 300 for "Shani".
    """
    r = LexiconRewriter(_GROUPS, max_variants=3)
    out = r.rewrite("significance of Sani")
    assert out[0] == "significance of Sani"
    assert out[1] == "significance of \u015aani"


def test_lexicon_rewrites_every_recognised_term_together():
    r = LexiconRewriter(_GROUPS, max_variants=1)
    out = r.rewrite("Saturn in the seventh house")
    assert out[1] == "\u015aani in the seventh Bh\u0101va"


def test_lexicon_includes_ascii_folds_of_accented_variants():
    """A user typing plain ASCII must still match a term group."""
    r = LexiconRewriter([["M\u1e5btyubh\u0101ga"]], max_variants=2)
    # the fold is registered as a lookup key even though it was not declared
    assert r.rewrite("Mrtyubhaga danger")[0] == "Mrtyubhaga danger"
    r2 = LexiconRewriter([["M\u1e5btyubh\u0101ga", "fatal degree"]], max_variants=2)
    assert any("fatal degree" in v for v in r2.rewrite("Mrtyubhaga danger"))


def test_lexicon_is_a_noop_without_known_terms():
    r = LexiconRewriter(_GROUPS)
    assert r.rewrite("no domain vocabulary here") == ["no domain vocabulary here"]


def test_lexicon_respects_max_variants():
    r = LexiconRewriter(_GROUPS, max_variants=2)
    assert len(r.rewrite("Sani")) <= 3  # original + 2


def test_lexicon_never_duplicates_the_original():
    r = LexiconRewriter(_GROUPS, max_variants=3)
    out = r.rewrite("Sani")
    assert len(out) == len(set(out))


def test_ollama_rewriter_falls_back_when_unreachable(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    r = OllamaRewriter(model="domain-slm", host="http://127.0.0.1:1")
    assert r.rewrite("what is a dasha") == ["what is a dasha"]


@pytest.mark.parametrize("name,expected", [
    (None, "noop"), ("noop", "noop"), ("lexicon", "lexicon"), ("bogus", "noop"),
])
def test_build_rewriter_names(name, expected):
    assert build_rewriter(name, {}).name == expected


def test_hybrid_search_fuses_variant_runs():
    """Each variant is searched, and the ranked runs are fused through RRF."""
    from phase3_rag.retrieve import HybridRetriever

    class _Rewriter:
        name = "fake"

        def rewrite(self, q):
            return [q, q + " variant"]

    seen: list[str] = []

    class _Store:
        def search_chunks(self, query, qvec, **kw):
            seen.append(query)
            # each variant surfaces a different chunk, so fusion must keep both
            cid = "a" if "variant" not in query else "b"
            return [(cid, "text-" + cid, "S", "s.pdf", 1, "", None, 0, "T", "en",
                     0.5, 1, None)]

        def fetch_by_ids(self, ids):
            return {}

    class _Emb:
        def encode(self, q, normalize_embeddings=True):
            class _V:
                def tolist(self):
                    return [0.1]
            return _V()

    class _RR:
        def predict(self, pairs):
            return [0.9] * len(pairs)

    retr = HybridRetriever.__new__(HybridRetriever)
    retr.cfg = {"retrieval": {"top_k_vector": 5, "top_k_bm25": 5,
                              "top_k_candidates": 10, "top_k_final": 5,
                              "use_parent_expansion": False}}
    retr.embedder = _Emb()
    retr.store = _Store()
    retr.reranker = _RR()
    retr.rewriter = _Rewriter()

    hits = retr.search("q")
    assert seen == ["q", "q variant"]
    assert {h.chunk_id for h in hits} == {"a", "b"}


def test_single_variant_path_skips_fusion():
    """The default must behave exactly as it did before rewriting existed."""
    from phase3_rag.retrieve import HybridRetriever

    class _Store:
        def search_chunks(self, query, qvec, **kw):
            return [("a", "text", "S", "s.pdf", 1, "", None, 0, "T", "en",
                     0.5, 1, None)]

        def fetch_by_ids(self, ids):
            return {}

    class _Emb:
        def encode(self, q, normalize_embeddings=True):
            class _V:
                def tolist(self):
                    return [0.1]
            return _V()

    class _RR:
        def predict(self, pairs):
            return [0.9] * len(pairs)

    retr = HybridRetriever.__new__(HybridRetriever)
    retr.cfg = {"retrieval": {"top_k_vector": 5, "top_k_bm25": 5,
                              "top_k_final": 5, "use_parent_expansion": False}}
    retr.embedder = _Emb()
    retr.store = _Store()
    retr.reranker = _RR()
    retr.rewriter = NoopRewriter()
    assert [h.chunk_id for h in retr.search("q")] == ["a"]
