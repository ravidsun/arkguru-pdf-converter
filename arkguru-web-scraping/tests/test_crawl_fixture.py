"""Crawl against an in-memory HTTP fixture (no public network)."""
from __future__ import annotations

import httpx

from phase2_web.fetch import crawl

INDEX = b"""<!DOCTYPE html><html><head><title>Demo</title></head>
<body><h1>Nakshatra</h1>
<p>Ashwini is the first nakshatra. PPE includes insulated gloves.</p>
<a href="/lesson.html">lesson</a>
<a href="/login">skip me</a>
<a href="https://evil.example/x">offsite</a>
</body></html>"""

LESSON = b"""<!DOCTYPE html><html><head><title>Lesson</title></head>
<body><p>The seventh house and Saturn. """ + (b"More teaching text. " * 20) + b"""</p>
</body></html>"""


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path in ("/robots.txt",):
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    if path in ("/", "/index.html"):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=INDEX)
    if path == "/lesson.html":
        return httpx.Response(200, headers={"content-type": "text/html"}, content=LESSON)
    if path == "/login":
        return httpx.Response(200, text="should not fetch")
    return httpx.Response(404)


def test_crawl_local_fixture_respects_deny_and_same_site():
    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport, follow_redirects=True)
    result = crawl(
        ["http://127.0.0.1:8899/index.html"],
        max_pages=10,
        max_pages_per_seed=10,
        delay_seconds=0,
        client=client,
    )
    urls = {p.url.rstrip("/") for p in result.pages}
    assert any("index.html" in u or u.endswith(":8899") for u in urls) or result.pages
    bodies = " ".join(p.html for p in result.pages)
    assert "Ashwini" in bodies
    assert "seventh house" in bodies
    assert "should not fetch" not in bodies
    assert all("evil.example" not in p.url for p in result.pages)


def test_crawl_counts_http_errors_toward_per_seed_budget():
    """Broken sitemaps must not fetch unbounded 404s after the seed cap."""
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if path == "/":
            anchors = "".join(f'<a href="/missing/{i}">x</a>' for i in range(40))
            html = f"<!DOCTYPE html><html><body><p>Seed page.</p>{anchors}</body></html>"
            return httpx.Response(200, headers={"content-type": "text/html"}, content=html.encode())
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, follow_redirects=True)
    result = crawl(
        ["http://fixture.test/"],
        max_pages=100,
        max_pages_per_seed=5,
        delay_seconds=0,
        client=client,
    )
    assert len(result.pages) == 1
    assert result.skipped >= 4
    # robots + seed + 4 errors (budget 5), not 40 missing links
    assert hits["n"] <= 8
