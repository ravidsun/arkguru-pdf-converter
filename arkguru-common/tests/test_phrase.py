"""Multi-word domain term detection for the search_chunks phrase leg."""
from __future__ import annotations

from pathlib import Path

from common.phrase import (
    detect_phrase_terms,
    is_multiword,
    load_multiword_terms,
    phrases_for_query,
)


LEXICON = Path(__file__).resolve().parents[2] / (
    "arkguru-rag-slm/phase3_rag/golden/domain_lexicon.json"
)


def test_is_multiword():
    assert is_multiword("sub sub lord")
    assert is_multiword("soul significator")
    assert not is_multiword("horary")
    assert not is_multiword("Saturn")


def test_detects_sub_sub_lord_over_sub_lord():
    terms = load_multiword_terms(LEXICON)
    assert any("sub sub lord" in t.lower() for t in terms)
    q = "What is the role of the sub sub lord in KP horary?"
    found = detect_phrase_terms(q, terms)
    assert found
    assert any("sub sub lord" in t.lower() for t in found)
    assert not any(t.lower() == "sub lord" for t in found)


def test_phrases_for_query_empty_when_no_domain_terms():
    terms = load_multiword_terms(LEXICON)
    assert phrases_for_query("hello world from the handbook", terms=terms) == []


def test_detects_soul_significator():
    terms = load_multiword_terms(LEXICON)
    found = detect_phrase_terms("who is the soul significator", terms)
    assert any("soul significator" in t.lower() for t in found)
