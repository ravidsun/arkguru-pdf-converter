"""Chunking + schema guard (no live crawl / Postgres)."""
from __future__ import annotations

import pytest

from common.schema import Chunk
from phase2_web.chunk import chunk_page
from phase2_web.dedup import exact_dedup, near_dedup
from phase2_web.pipeline import Phase2Config, _require_web_schema, pages_to_chunks
from phase2_web.fetch import CrawlResult, FetchedPage


def test_chunk_page_sets_web_source_type():
    text = "Saturn in the seventh house. " * 80
    chunks = chunk_page(text, "https://cosmicinsights.net/saturn", title="Saturn")
    assert chunks
    assert all(c.source_type == "web" for c in chunks)
    assert all(c.source_id == "https://cosmicinsights.net/saturn" for c in chunks)
    assert all(c.domain == "cosmicinsights.net" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_page_drops_thin_pages():
    assert chunk_page("too short", "https://x.com/a", min_content_chars=200) == []


def test_exact_dedup_keeps_first():
    a = Chunk(text="same", source_type="web", source_id="u", chunk_index=0)
    b = Chunk(text="same", source_type="web", source_id="u", chunk_index=1)
    c = Chunk(text="other", source_type="web", source_id="u", chunk_index=2)
    out = exact_dedup([a, b, c])
    assert [x.text for x in out] == ["same", "other"]


def test_near_dedup_exact_fallback():
    a = Chunk(text="alpha beta", source_type="web", source_id="u", chunk_index=0)
    b = Chunk(text="alpha beta", source_type="web", source_id="u", chunk_index=1)
    out = near_dedup([a, b], threshold=0.9)
    assert len(out) == 1


class _PublicStore:
    schema = "public"
    chunks = "chunks"
    is_public_schema = True


class _WebStore:
    schema = "web"
    chunks = "web.chunks"
    is_public_schema = False


def test_require_web_schema_refuses_public():
    with pytest.raises(SystemExit, match="refuses schema"):
        _require_web_schema(_PublicStore())
    _require_web_schema(_WebStore())  # does not raise


def test_pages_to_chunks_from_fixture_html():
    html = """<html><head><title>Nakshatra</title></head>
    <body><h1>Ashwini</h1><p>%s</p></body></html>""" % ("The first nakshatra. " * 40)
    result = CrawlResult(pages=[FetchedPage(
        url="https://cosmicinsights.net/ashwini", status=200,
        content_type="text/html", body=b"", html=html,
    )])
    chunks = pages_to_chunks(result, Phase2Config(min_content_chars=50))
    assert chunks
    assert chunks[0].title == "Nakshatra"
    assert chunks[0].source_type == "web"
