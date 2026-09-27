"""Unicode-aware chunk quality scorer and post-chunk annotation.

A naive symbol-ratio check treats Croatian/Swedish letters, Sanskrit
diacritics, and KP Ezine prose as "garbled". This scorer uses:

  * Unicode ``str.isalpha()`` letter share (not ``[A-Za-z]``)
  * real-word share (vowel / Devanagari tokens)
  * longest punctuation/symbol run

Results are stored on ``chunk.extra["quality"]`` (persisted as ``meta.quality``).
Chunks that fail the gate are **flagged and kept** — they must never be
silently dropped, and they must never be embedded.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from .lang import DEFAULT_LANG, detect_lang, script_family
from .schema import Chunk, content_hash

# Latin + Indic vowel letters, including common diacritics (ā, č, å, …).
_VOWELS = set(
    "aeiouyAEIOUY"
    "àáâãäåèéêëìíîïòóôõöøùúûüýÿ"
    "ÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖØÙÚÛÜÝ"
    "āăąēėęīįōőūůűȳ"
    "ĀĂĄĒĖĘĪĮŌŐŪŮŰȲ"
    "æœÆŒ"
)
_WORD_SPLIT = re.compile(r"[^\W_]+", re.UNICODE)
_FORMULA_TOKEN = re.compile(
    r"^[\d°'\"" + r"=+\-*/^()[\]{}.,:;]+$"
)
_REPLACEMENT = "\ufffd"

DEFAULT_MIN_SCORE = 0.35
DEFAULT_MIN_ALPHA_RATIO = 0.28
DEFAULT_MAX_PUNCT_RUN = 0.18
DEFAULT_MIN_WORD_SHARE = 0.22


def _is_vowel_letter(ch: str) -> bool:
    if ch in _VOWELS:
        return True
    if "\u0900" <= ch <= "\u097F":
        # Independent vowels and matras in Devanagari.
        return unicodedata.category(ch) in {"Lo", "Mc", "Mn"} and (
            ch <= "\u0914" or "\u093A" <= ch <= "\u094F" or ch in "अआइईउऊऋॠऌॡएऐओऔ"
        )
    return False


def _is_real_word(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 3:
        return False
    if any("\u0900" <= c <= "\u097F" for c in letters):
        return True
    # Mojibake is often a run of Latin-1 supplement letters with no ASCII.
    # Croatian/Swedish mix ASCII with a few diacritics.
    ascii_letters = sum(1 for c in letters if "a" <= c.lower() <= "z")
    if ascii_letters / len(letters) < 0.35:
        return False
    return any(_is_vowel_letter(c) for c in letters)


def _looks_like_formula(text: str) -> bool:
    """Short, operator-heavy spans (house equations, degrees) are not garbage."""
    compact = "".join(text.split())
    if len(compact) < 8:
        return False
    ops = sum(1 for c in compact if c in "=/+-*^*°()[]{}")
    digits = sum(1 for c in compact if c.isdigit())
    return (ops + digits) / len(compact) >= 0.35 and ops >= 2


@dataclass
class QualityScore:
    alpha_ratio: float
    word_share: float
    punct_run: float
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    embed: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def score_quality(
    text: str,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    min_alpha_ratio: float = DEFAULT_MIN_ALPHA_RATIO,
    max_punct_run: float = DEFAULT_MAX_PUNCT_RUN,
    min_word_share: float = DEFAULT_MIN_WORD_SHARE,
) -> QualityScore:
    """Score ``text``. Empty input fails (flagged, not embeddable)."""
    raw = text or ""
    reasons: list[str] = []
    letters = digits = symbols = 0
    max_run = run = 0
    repl = 0
    for ch in raw:
        if ch == _REPLACEMENT:
            repl += 1
        if ch.isspace():
            run = 0
            continue
        cat = unicodedata.category(ch)
        if ch.isalpha() or cat in {"Mn", "Mc", "Me"}:
            letters += 1
            run = 0
        elif ch.isdigit():
            digits += 1
            run = 0
        else:
            symbols += 1
            if cat.startswith("P") or cat.startswith("S"):
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0

    nonempty = letters + digits + symbols
    alpha_ratio = (letters / nonempty) if nonempty else 0.0
    symbol_ratio = (symbols / nonempty) if nonempty else 1.0
    punct_run = (max_run / len(raw)) if raw else 1.0

    tokens = _WORD_SPLIT.findall(raw)
    scored_tokens = [
        t for t in tokens
        if not _FORMULA_TOKEN.match(t) and any(c.isalpha() for c in t)
    ]
    if scored_tokens:
        real = sum(1 for t in scored_tokens if _is_real_word(t))
        word_share = real / len(scored_tokens)
    else:
        word_share = 0.0
    # Devanagari prose is often lightly spaced; do not treat matras/dandas
    # as a missing-word / symbol failure.
    if script_family(raw) == "deva" and alpha_ratio >= 0.25:
        word_share = max(word_share, min_word_share)
        symbol_ratio = min(symbol_ratio, 0.25)

    formula = _looks_like_formula(raw)
    if formula:
        # Keep letter/punct checks; do not fail on low word_share.
        effective_word = max(word_share, min_word_share)
    else:
        effective_word = word_share

    score = (
        0.50 * alpha_ratio
        + 0.40 * effective_word
        + 0.10 * (1.0 - min(punct_run / max(max_punct_run, 1e-6), 1.0))
    )

    if not raw.strip():
        reasons.append("empty")
    if repl and (repl / max(len(raw), 1)) >= 0.02:
        reasons.append("replacement_chars")
    if alpha_ratio < min_alpha_ratio:
        reasons.append("low_alpha_ratio")
    if (not formula) and symbol_ratio > 0.40:
        reasons.append("high_symbol_ratio")
    if (not formula) and word_share < min_word_share:
        reasons.append("low_word_share")
    if punct_run > max_punct_run:
        reasons.append("long_punct_run")
    if score < min_score:
        reasons.append("low_score")

    passed = not reasons
    return QualityScore(
        alpha_ratio=round(alpha_ratio, 4),
        word_share=round(word_share, 4),
        punct_run=round(punct_run, 4),
        score=round(score, 4),
        passed=passed,
        reasons=reasons,
        embed=passed,
    )


def source_matches(source_id: str, patterns: Iterable[str] | None) -> bool:
    """Case-insensitive glob match on the full source_id or its basename."""
    import fnmatch

    name = source_id or ""
    base = name.rsplit("/", 1)[-1]
    for pat in patterns or []:
        if not pat:
            continue
        for candidate in (name, base, name.lower(), base.lower()):
            if fnmatch.fnmatch(candidate, pat) or fnmatch.fnmatch(
                candidate, str(pat).lower()
            ):
                return True
    return False


def annotate_chunks(
    chunks: list[Chunk],
    *,
    quality_cfg: Optional[dict] = None,
    lang_seed: int = 0,
) -> list[Chunk]:
    """Attach ``quality``, ``content_hash``, and a non-empty ``lang`` to each chunk.

    Failed quality chunks are kept, flagged, set to ``lang='und'``, and marked
    ``quality.embed = false`` so the datastore embedder skips them.
    """
    cfg = dict(quality_cfg or {})
    enabled = cfg.get("enabled", True)
    min_score = float(cfg.get("min_score", DEFAULT_MIN_SCORE))
    min_alpha = float(cfg.get("min_alpha_ratio", DEFAULT_MIN_ALPHA_RATIO))
    max_punct = float(cfg.get("max_punct_run", DEFAULT_MAX_PUNCT_RUN))
    min_words = float(cfg.get("min_word_share", DEFAULT_MIN_WORD_SHARE))
    allow = cfg.get("allow_sources") or []
    deny = cfg.get("deny_sources") or []

    for c in chunks:
        extra = dict(c.extra or {})
        digest = content_hash(c.text)
        c.content_hash = digest
        extra["content_hash"] = digest

        if not enabled:
            q = QualityScore(
                alpha_ratio=1.0, word_share=1.0, punct_run=0.0,
                score=1.0, passed=True, reasons=[], embed=True,
            )
        elif source_matches(c.source_id, allow):
            q = score_quality(
                c.text, min_score=0.0, min_alpha_ratio=0.0,
                max_punct_run=1.0, min_word_share=0.0,
            )
            q.passed = True
            q.embed = True
            q.reasons = []
        elif source_matches(c.source_id, deny):
            q = score_quality(
                c.text, min_score=min_score, min_alpha_ratio=min_alpha,
                max_punct_run=max_punct, min_word_share=min_words,
            )
            q.passed = False
            q.embed = False
            if "deny_source" not in q.reasons:
                q.reasons.append("deny_source")
        else:
            q = score_quality(
                c.text, min_score=min_score, min_alpha_ratio=min_alpha,
                max_punct_run=max_punct, min_word_share=min_words,
            )

        extra["quality"] = q.to_dict()
        if not q.passed:
            extra["quality_flagged"] = True
            c.lang = DEFAULT_LANG
        else:
            extra.pop("quality_flagged", None)
            detected = detect_lang(c.text, seed=lang_seed)
            c.lang = detected or DEFAULT_LANG
        if not (c.lang or "").strip():
            c.lang = DEFAULT_LANG
        c.extra = extra
    return chunks
