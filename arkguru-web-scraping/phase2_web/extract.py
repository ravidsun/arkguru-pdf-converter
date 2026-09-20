"""HTML -> main text. trafilatura when installed; regex fallback otherwise."""
from __future__ import annotations

import re
from html import unescape
from typing import Optional

from .fetch import FetchedPage


def _strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p>", "\n\n", html)
    html = re.sub(r"(?is)</h[1-6]>", "\n\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    text = unescape(html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_html(html: str) -> Optional[str]:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not m:
        return None
    return unescape(re.sub(r"\s+", " ", m.group(1))).strip() or None


def extract_page(page: FetchedPage) -> tuple[str, Optional[str]]:
    """Return ``(text, title)``. Empty text means drop the page."""
    html = page.html or ""
    title = _title_from_html(html)
    text = ""
    try:
        import trafilatura
        extracted = trafilatura.extract(
            html, url=page.url, include_comments=False, include_tables=True)
        if extracted:
            text = extracted.strip()
    except Exception:
        text = ""
    if not text:
        text = _strip_tags(html)
    return text, title
