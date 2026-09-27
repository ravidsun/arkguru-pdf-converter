"""Shared text cleanup for PDF and web extractors.

pymupdf4llm wraps highlights in ``<mark>`` and also emits ``<u>``, ``<sup>``,
``<br>``, plus ``**`` / ``_`` emphasis. Those markers must not reach the
embedder. Cleanup is Unicode-safe: NFKC plus whitespace collapse, with
combining marks and Devanagari left intact.
"""
from __future__ import annotations

import re
import unicodedata
from html import unescape

# Block-ish tags become a newline so headings/paragraphs stay separable.
_BREAK_TAG = re.compile(
    r"(?is)</?(?:br|p|div|tr|li|h[1-6])(?:\s[^>]*)?/?>"
)
_HTML_COMMENT = re.compile(r"(?is)<!--.*?-->")
# Any remaining HTML tag, including <mark>, <u>, <sup>, leftovers.
_HTML_TAG = re.compile(r"(?is)</?[a-z][a-z0-9:-]*(?:\s[^>]*)?/?>")
# Markdown emphasis. Underscore form requires a non-word boundary so
# identifiers like sub_sub_lord are not shredded.
_MD_BOLD_ITAL = re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL)
_MD_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_BOLD_UNDER = re.compile(r"__(.+?)__", re.DOTALL)
_MD_ITAL_STAR = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_MD_ITAL_UNDER = re.compile(r"(?<!\w)_(?!_)([^_\n]+?)(?<!_)_(?!\w)")
_HORIZONTAL_WS = re.compile(r"[^\S\n\r]+")
_MULTI_NL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip inline HTML, unwrap markdown emphasis, NFKC-normalise, collapse ws.

    Empty / ``None`` input returns ``""``. The function is idempotent.
    """
    if not text:
        return ""
    out = unescape(str(text))
    out = _HTML_COMMENT.sub(" ", out)
    out = _BREAK_TAG.sub("\n", out)
    out = _HTML_TAG.sub("", out)
    out = unescape(out)
    out = _unwrap_markdown_emphasis(out)
    out = unicodedata.normalize("NFKC", out)
    out = _HORIZONTAL_WS.sub(" ", out)
    out = _MULTI_NL.sub("\n\n", out)
    # Preserve combining marks; only strip ASCII/Unicode space at the ends.
    return out.strip()


def _unwrap_markdown_emphasis(text: str) -> str:
    out = _MD_BOLD_ITAL.sub(r"\1", text)
    out = _MD_BOLD_STAR.sub(r"\1", out)
    out = _MD_BOLD_UNDER.sub(r"\1", out)
    out = _MD_ITAL_STAR.sub(r"\1", out)
    out = _MD_ITAL_UNDER.sub(r"\1", out)
    return out


def body_without_heading(text: str, heading: str | None) -> str:
    """Chunk body with a prefixed section heading removed."""
    body = (text or "").strip()
    head = (heading or "").strip()
    if not head:
        return body
    if body == head:
        return ""
    if body.startswith(head):
        rest = body[len(head):]
        return rest.lstrip("\n").lstrip()
    return body
