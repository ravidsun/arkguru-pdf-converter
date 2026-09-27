"""Detect multi-word domain terms in a query for the search_chunks phrase leg.

The phrase leg is driven by ``golden/domain_lexicon.json``. Only terms that
contain a space (after hyphen-normalisation) are used: ``phraseto_tsquery``
on a unigram is just a lexeme, which the websearch leg already covers.
Longer matches win so ``sub sub lord`` is preferred over ``sub lord``.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _norm_spaces(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", (s or "").strip().lower())


def is_multiword(term: str) -> bool:
    """True when ``term`` has two or more whitespace-separated tokens."""
    return len(_norm_spaces(term).split()) >= 2


def lexicon_multiword_terms(groups: Iterable[dict]) -> list[str]:
    """Unique multi-word canonicals + variants, longest first."""
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        candidates = [group.get("canonical") or ""]
        candidates.extend(group.get("variants") or [])
        for raw in candidates:
            text = (raw or "").strip()
            if not text or not is_multiword(text):
                continue
            key = _norm_spaces(text)
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            folded = _fold(text)
            if folded != text and is_multiword(folded):
                fkey = _norm_spaces(folded)
                if fkey not in seen:
                    seen.add(fkey)
                    out.append(folded)
    out.sort(key=lambda t: (-len(_norm_spaces(t)), t.lower()))
    return out


def load_lexicon_groups(path: Path | str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("groups") or [])


def load_multiword_terms(path: Path | str) -> list[str]:
    return lexicon_multiword_terms(load_lexicon_groups(path))


def detect_phrase_terms(
    query: str,
    terms: Sequence[str],
    *,
    max_terms: int = 8,
) -> list[str]:
    """Return lexicon multi-word terms found in ``query`` (longest first).

    Overlapping shorter terms are dropped when a longer term already covers
    the same span (``sub lord`` is skipped if ``sub sub lord`` matched).
    """
    hay = _norm_spaces(_fold(query))
    if not hay:
        return []
    occupied: list[tuple[int, int]] = []
    found: list[str] = []
    for term in terms:
        needle = _norm_spaces(_fold(term))
        if not needle or not is_multiword(needle):
            continue
        pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
        match = re.search(pattern, hay)
        if not match:
            continue
        start, end = match.span()
        if any(start < e and end > s for s, e in occupied):
            continue
        occupied.append((start, end))
        found.append(term)
        if len(found) >= max_terms:
            break
    return found


def phrases_for_query(
    query: str,
    lexicon_path: Optional[Path | str] = None,
    terms: Optional[Sequence[str]] = None,
    *,
    max_terms: int = 8,
) -> list[str]:
    """Convenience: load the lexicon (or use ``terms``) and detect."""
    if terms is None:
        if lexicon_path is None:
            raise ValueError("pass lexicon_path or terms")
        terms = load_multiword_terms(lexicon_path)
    return detect_phrase_terms(query, terms, max_terms=max_terms)
