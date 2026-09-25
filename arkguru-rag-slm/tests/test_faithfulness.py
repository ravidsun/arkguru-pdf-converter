"""Faithfulness gate: sentences must overlap retrieved chunk text."""
from __future__ import annotations

from dataclasses import dataclass

from phase3_rag.faithfulness import (
    apply_gate,
    check_answer,
    format_refusal,
    run_gated_answer,
)


@dataclass
class Hit:
    chunk_id: str
    text: str
    source_id: str = "handbook.pdf"
    page: int | None = 1


CHUNK = (
    "Wear PPE before servicing a unit. Required gear includes safety glasses, "
    "insulated gloves, and steel-toe boots. Isolate electrical supply first."
)


def test_supported_sentence_from_chunk_is_ok():
    hits = [Hit("c1", CHUNK)]
    result = check_answer(
        "Wear PPE before servicing a unit. Isolate electrical supply first.",
        hits,
    )
    assert result.ok
    assert result.unsupported == ()
    assert "c1" in result.cited_chunk_ids


def test_invented_claim_is_rejected():
    hits = [Hit("c1", CHUNK)]
    result = check_answer(
        "The gearbox uses quantum unicorn bearings and never needs oil.",
        hits,
    )
    assert not result.ok
    assert result.unsupported
    assert "quantum unicorn" in result.unsupported[0].lower() or "unicorn" in result.unsupported[0].lower()


def test_mixed_answer_lists_only_unsupported():
    hits = [Hit("c1", CHUNK)]
    result = check_answer(
        "Wear PPE before servicing a unit. The unit is powered by cold fusion.",
        hits,
    )
    assert not result.ok
    assert len(result.unsupported) == 1
    assert "cold fusion" in result.unsupported[0].lower()


def test_empty_answer_is_not_ok():
    result = check_answer("", [Hit("c1", CHUNK)])
    assert not result.ok


def test_short_answer_is_not_ok():
    result = check_answer("Yes.", [Hit("c1", CHUNK)])
    assert not result.ok


def test_dict_hits_work_like_objects():
    hits = [{"chunk_id": "c1", "text": CHUNK}]
    result = check_answer("Wear PPE before servicing a unit.", hits)
    assert result.ok


def test_apply_gate_skips_extractive():
    hits = [Hit("c1", CHUNK)]
    extractive = f"[extractive] {CHUNK}"
    assert apply_gate(extractive, hits) == extractive


def test_apply_gate_refuses_hallucination():
    hits = [Hit("c1", CHUNK)]
    out = apply_gate("The gearbox uses quantum unicorn bearings in every model.", hits)
    assert out.startswith("I couldn't ground this answer")
    assert "unicorn" in out.lower()


def test_format_refusal_includes_unsupported():
    from phase3_rag.faithfulness import FaithfulnessResult
    text = format_refusal(FaithfulnessResult(
        ok=False, unsupported=("Invented fact.",), cited_chunk_ids=()))
    assert "Invented fact." in text


class _FakeRetr:
    def __init__(self, hits):
        self.cfg = {"retrieval": {"top_k_final": 2}}
        self.hits = hits
        self.calls: list[int | None] = []

    def search(self, query: str, *, top_k_final: int | None = None):
        self.calls.append(top_k_final)
        return self.hits


def test_run_gated_answer_returns_grounded_text_without_retry():
    hits = [Hit("c1", CHUNK)]
    retr = _FakeRetr(hits)

    def generate(_hits):
        return "Wear PPE before servicing a unit. Isolate electrical supply first."

    ans, out_hits = run_gated_answer(retr, "ppe?", generate)
    assert ans.startswith("Wear PPE")
    assert out_hits == hits
    assert retr.calls == [None]


def test_run_gated_answer_retries_then_refuses_hallucination():
    hits = [Hit("c1", CHUNK)]
    retr = _FakeRetr(hits)

    def generate(_hits):
        return "The gearbox uses quantum unicorn bearings and never needs oil."

    ans, _ = run_gated_answer(retr, "gearbox?", generate, {"max_retries": 1})
    assert ans.startswith("I couldn't ground this answer")
    assert retr.calls == [None, 4]  # default_k 2 * (1+1)


def test_run_gated_answer_no_hits():
    retr = _FakeRetr([])
    ans, hits = run_gated_answer(retr, "anything", lambda h: "should not run")
    assert ans == "No relevant context found."
    assert hits == []
    assert retr.calls == [None]
