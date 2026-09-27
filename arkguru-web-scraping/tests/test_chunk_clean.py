"""Web extract/chunk: clean_text + hard token cap."""
from __future__ import annotations

import re

from common.tokenizer import DEFAULT_MAX_TOKENS, count_tokens
from phase2_web.chunk import chunk_page
from phase2_web.extract import extract_page
from phase2_web.fetch import FetchedPage

_INLINE_TAG = re.compile(r"(?is)</?(?:mark|u|sup|br)\s*/?>")


def test_extract_page_strips_tags_and_keeps_devanagari():
    html = """<html><head><title>Dasha</title></head>
    <body><p>The <mark>sub lord</mark> is Saturn.<br>
    ग्रही <u>शुक्र</u> की दशा ।</p></body></html>"""
    page = FetchedPage(
        url="https://example.com/dasha", status=200,
        content_type="text/html", body=html.encode(), html=html,
    )
    text, title = extract_page(page)
    assert title == "Dasha"
    assert _INLINE_TAG.search(text) is None
    assert "sub lord" in text
    assert "शुक्र" in text
    assert "।" in text


def test_chunk_page_hard_cap_and_no_tags():
    body = (
        "The <mark>sub sub lord</mark> of the seventh house. " * 20 + "\n\n"
    ) * 400
    assert len(body) > 50_000
    chunks = chunk_page(
        body, "https://cafeastrology.com/saturn",
        title="Saturn",
        target_tokens=400,
        max_tokens=DEFAULT_MAX_TOKENS,
        min_content_chars=50,
        min_chunk_chars=80,
    )
    assert chunks
    joined = "\n".join(c.text for c in chunks)
    assert _INLINE_TAG.search(joined) is None
    assert all(count_tokens(c.text) <= DEFAULT_MAX_TOKENS for c in chunks)
    assert all(c.source_type == "web" for c in chunks)
