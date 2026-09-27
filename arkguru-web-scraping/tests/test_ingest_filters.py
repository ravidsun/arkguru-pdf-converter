"""Site-map / archive ingest deny + link-density; links are still followed."""
from __future__ import annotations

import httpx

from phase2_web.fetch import (
    crawl,
    ingest_path_denied,
    link_density,
)
from phase2_web.pipeline import Phase2Config, pages_to_chunks
from phase2_web.fetch import CrawlResult, FetchedPage


def test_ingest_path_denied_patterns():
    assert ingest_path_denied("https://cafeastrology.com/sitemap.html")
    assert ingest_path_denied("https://cafeastrology.com/site-map")
    assert ingest_path_denied("https://x.com/tag/saturn/")
    assert ingest_path_denied("https://x.com/category/houses/")
    assert ingest_path_denied("https://x.com/blog/page/2")
    assert ingest_path_denied("https://x.com/archive/2020")
    assert not ingest_path_denied("https://cafeastrology.com/saturn.html")
    assert not ingest_path_denied("https://x.com/lessons/page-of-houses")


def test_link_density_high_on_sitemap_html():
    sitemap = (
        "<html><body><h1>Sitemap</h1><ul>"
        + "".join(f'<li><a href="/p{i}">Page {i} title</a></li>' for i in range(40))
        + "</ul></body></html>"
    )
    article = (
        "<html><body><h1>Saturn</h1><p>"
        + ("The seventh house and Saturn delay marriage. " * 20)
        + '</p><p>See also <a href="/other">other</a>.</p></body></html>'
    )
    assert link_density(sitemap) > 0.60
    assert link_density(article) < 0.30


def test_sitemap_yields_zero_chunks_but_links_are_followed():
    index = b"""<!DOCTYPE html><html><head><title>Home</title></head>
    <body><h1>Home</h1><p>Welcome to the lesson hub.</p>
    <a href="/sitemap.html">sitemap</a>
    <a href="/lesson.html">lesson</a>
    </body></html>"""
    sitemap = b"""<!DOCTYPE html><html><head><title>Site Map</title></head>
    <body><h1>Site Map</h1><ul>""" + b"".join(
        f'<li><a href="/item{i}.html">Item {i}</a></li>'.encode() for i in range(20)
    ) + b"""<li><a href="/lesson.html">Lesson</a></li></ul></body></html>"""
    lesson = (
        b"""<!DOCTYPE html><html><head><title>Lesson</title></head><body><p>"""
        + (b"The seventh house and Saturn. " * 25)
        + b"""</p></body></html>"""
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if path in ("/", "/index.html"):
            return httpx.Response(200, headers={"content-type": "text/html"}, content=index)
        if path == "/sitemap.html":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=sitemap)
        if path == "/lesson.html":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=lesson)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = crawl(
        ["http://fixture.test/index.html"],
        max_pages=20,
        max_pages_per_seed=20,
        delay_seconds=0,
        client=client,
    )
    urls = [p.url for p in result.pages]
    assert not any("sitemap" in u for u in urls)
    assert any("lesson.html" in u for u in urls)
    chunks = pages_to_chunks(result, Phase2Config(min_content_chars=50, dedup=False))
    assert chunks
    assert all("sitemap" not in (c.source_id or "") for c in chunks)
    assert all(c.source_type == "web" for c in chunks)
    assert all(c.lang for c in chunks)


def test_pages_to_chunks_drops_sitemap_even_if_present():
    html = "<html><body><p>" + ("Saturn in the seventh house. " * 20) + "</p></body></html>"
    result = CrawlResult(pages=[
        FetchedPage(
            url="https://cafeastrology.com/sitemap.html", status=200,
            content_type="text/html", body=b"", html=html,
        ),
    ])
    assert pages_to_chunks(result, Phase2Config(min_content_chars=20, dedup=False)) == []
