"""
Lexical faithfulness gate for Phase 3 answers.

A generated sentence is supported if enough of its content tokens appear in
at least one retrieved chunk. Checking is a separate job from generation: the
same model does not grade its own answer. No extra LLM and no RAGAS on the
serve path — those stay in eval_ragas.py.

Extractive fallbacks ([extractive] ...) are already copied from a hit and
skip the gate.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from common.chunking import split_sentences

log = logging.getLogger("phase3.faithfulness")

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or of to in on for with from by is are was were be this that it "
    "as at if not no yes so".split()
)
_EXTRACTIVE_PREFIX = "[extractive]"
_NO_CONTEXT = "No relevant context found."

DEFAULT_MIN_OVERLAP = 0.4
DEFAULT_MIN_SENTENCE_TOKENS = 6
DEFAULT_MAX_RETRIES = 1


@dataclass(frozen=True)
class FaithfulnessResult:
    ok: bool
    unsupported: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]


def _hit_text(hit: Any) -> str:
    if isinstance(hit, dict):
        return str(hit.get("text") or "")
    return str(getattr(hit, "text", "") or "")


def _hit_id(hit: Any) -> str:
    if isinstance(hit, dict):
        return str(hit.get("chunk_id") or "")
    return str(getattr(hit, "chunk_id", "") or "")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.casefold()) if t not in _STOP}


def _is_passthrough(answer: str) -> bool:
    stripped = answer.strip()
    return stripped.startswith(_EXTRACTIVE_PREFIX) or stripped == _NO_CONTEXT


def check_answer(
    answer: str,
    hits: Sequence[Any],
    *,
    min_token_overlap: float = DEFAULT_MIN_OVERLAP,
    min_sentence_tokens: int = DEFAULT_MIN_SENTENCE_TOKENS,
) -> FaithfulnessResult:
    """Return whether every checkable sentence is supported by `hits`."""
    if not answer.strip():
        return FaithfulnessResult(ok=False, unsupported=(), cited_chunk_ids=())
    if not hits:
        return FaithfulnessResult(ok=False, unsupported=(answer.strip(),), cited_chunk_ids=())

    chunk_tokens = [(_hit_id(h), _content_tokens(_hit_text(h))) for h in hits]
    unsupported: list[str] = []
    cited: list[str] = []
    checked = 0

    for sentence in split_sentences(answer.strip()):
        raw = _WORD.findall(sentence.casefold())
        if len(raw) < min_sentence_tokens:
            continue
        stoks = {t for t in raw if t not in _STOP}
        if not stoks:
            continue
        checked += 1
        best_id = ""
        best_overlap = 0.0
        for cid, ctoks in chunk_tokens:
            overlap = len(stoks & ctoks) / len(stoks)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = cid
        if best_overlap >= min_token_overlap:
            if best_id:
                cited.append(best_id)
        else:
            unsupported.append(sentence)

    if checked == 0:
        # Only short sentences (or punctuation). Treat as ungrounded rather
        # than letting "Yes." slip through as a grounded domain answer.
        return FaithfulnessResult(ok=False, unsupported=(answer.strip(),), cited_chunk_ids=())

    return FaithfulnessResult(
        ok=not unsupported,
        unsupported=tuple(unsupported),
        cited_chunk_ids=tuple(dict.fromkeys(cited)),
    )


def format_refusal(result: FaithfulnessResult) -> str:
    lines = [
        "I couldn't ground this answer in the retrieved sources.",
        "",
        "Unsupported claims:",
    ]
    if result.unsupported:
        lines.extend(f"- {s}" for s in result.unsupported)
    else:
        lines.append("- (empty or too short to verify)")
    return "\n".join(lines)


def apply_gate(
    answer: str,
    hits: Sequence[Any],
    *,
    min_token_overlap: float = DEFAULT_MIN_OVERLAP,
    min_sentence_tokens: int = DEFAULT_MIN_SENTENCE_TOKENS,
) -> str:
    """Return `answer` if grounded, otherwise a refusal. Skips extractive text."""
    if _is_passthrough(answer):
        return answer
    result = check_answer(
        answer, hits,
        min_token_overlap=min_token_overlap,
        min_sentence_tokens=min_sentence_tokens,
    )
    if result.ok:
        return answer
    log.warning("faithfulness gate rejected answer; unsupported=%s", result.unsupported)
    return format_refusal(result)


def run_gated_answer(
    retr: Any,
    query: str,
    generate: Callable[[list], str],
    faithfulness_cfg: dict | None = None,
) -> tuple[str, list]:
    """Retrieve → generate → lexical check; one broader retrieve retry on fail."""
    fc = faithfulness_cfg or {}
    enabled = bool(fc.get("enabled", True))
    min_overlap = float(fc.get("min_token_overlap", DEFAULT_MIN_OVERLAP))
    min_sent = int(fc.get("min_sentence_tokens", DEFAULT_MIN_SENTENCE_TOKENS))
    max_retries = int(fc.get("max_retries", DEFAULT_MAX_RETRIES))
    cfg = getattr(retr, "cfg", None) or {}
    default_k = int(cfg.get("retrieval", {}).get("top_k_final", 6))

    hits = retr.search(query)
    if not hits:
        return _NO_CONTEXT, hits

    ans = generate(hits)
    if not enabled or _is_passthrough(ans):
        return ans, hits

    result = check_answer(
        ans, hits, min_token_overlap=min_overlap, min_sentence_tokens=min_sent,
    )
    attempt = 0
    while not result.ok and attempt < max_retries:
        attempt += 1
        wider = default_k * (attempt + 1)
        log.info("faithfulness retry %s with top_k_final=%s", attempt, wider)
        wider_hits = retr.search(query, top_k_final=wider)
        if not wider_hits:
            break
        hits = wider_hits
        ans = generate(hits)
        if _is_passthrough(ans):
            return ans, hits
        result = check_answer(
            ans, hits, min_token_overlap=min_overlap, min_sentence_tokens=min_sent,
        )

    if result.ok:
        return ans, hits
    log.warning("faithfulness gate failed after %s retries; unsupported=%s",
                attempt, result.unsupported)
    return format_refusal(result), hits
