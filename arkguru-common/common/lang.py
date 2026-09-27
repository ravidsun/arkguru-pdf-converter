"""Per-chunk, seeded, deterministic language detection.

Document-level ``langdetect`` on the first title-page blocks is what stamped
BPHS as ``ne`` and garbled J_KP readers as ``sv``/``sl``. This module:

  * runs on each chunk (caller seeds ``DetectorFactory`` every time)
  * rejects script/language contradictions (Devanagari cannot be ``sv``)
  * never returns empty — the fallback is ``und``
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger("common.lang")

DEFAULT_LANG = "und"
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# langdetect tags Latin-script noise as these codes. Devanagari text must not.
_LATIN_ONLY = frozenset({
    "af", "ca", "cs", "cy", "da", "de", "en", "es", "et", "fi", "fr",
    "hr", "hu", "id", "it", "lt", "lv", "nl", "no", "pl", "pt", "ro",
    "sk", "sl", "so", "sq", "sv", "sw", "tl", "tr", "vi",
})
_INDIC = frozenset({"hi", "mr", "ne", "bn", "sa"})


def script_family(text: str) -> str:
    """``deva`` if enough letters are Devanagari, else ``latin`` or ``und``."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return "und"
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    if (dev / len(letters)) >= 0.20:
        return "deva"
    return "latin"


def _seed_detector(seed: int) -> None:
    try:
        from langdetect import DetectorFactory
    except ImportError:
        return
    DetectorFactory.seed = int(seed)


def detect_lang(text: str, *, seed: int = 0) -> str:
    """Return a BCP-47-ish language code. Never empty; default ``und``.

    ``seed`` is applied to ``langdetect.DetectorFactory`` before every call so
    the same text always yields the same code.
    """
    sample = (text or "").strip()
    if not sample:
        return DEFAULT_LANG
    family = script_family(sample)
    raw = _detect_raw(sample, seed=seed)
    return reconcile_lang(raw, family)


def _detect_raw(text: str, *, seed: int) -> Optional[str]:
    try:
        from langdetect import detect
    except ImportError:
        return None
    _seed_detector(seed)
    try:
        code = detect(text[:4000])
    except Exception:
        return None
    if not code or not str(code).strip():
        return None
    return str(code).strip().lower()


def reconcile_lang(detected: Optional[str], family: str) -> str:
    """Apply the script check. Devanagari is never a Latin-only code."""
    code = (detected or "").strip().lower() or None
    if family == "deva":
        if code in _LATIN_ONLY or code is None:
            return "hi"
        # Garbled Latin was tagged ``ne`` in the live DB; real Nepali is rare
        # in this corpus. Prefer Hindi for Devanagari unless another Indic
        # language (not ``ne``) was returned.
        if code == "ne":
            return "hi"
        if code in _INDIC:
            return code
        return "hi"
    if not code:
        return DEFAULT_LANG
    return code


def lang_or_und(value: Optional[str]) -> str:
    v = (value or "").strip()
    return v if v else DEFAULT_LANG
