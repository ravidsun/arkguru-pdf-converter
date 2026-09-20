"""Fetch helpers: same-site, denylist, kundli generators, link extraction."""
from __future__ import annotations

from phase2_web.fetch import (
    extract_links,
    is_binary_url,
    is_kundli_generator,
    is_pdf_url,
    normalize_url,
    path_denied,
    same_site,
)


def test_same_site_strips_www():
    seed = "https://www.cosmicinsights.net/"
    assert same_site("https://cosmicinsights.net/nakshatra", seed)
    assert same_site("https://www.cosmicinsights.net/a", seed)
    assert not same_site("https://evil.example/x", seed)


def test_path_denied():
    assert path_denied("https://x.com/login")
    assert path_denied("https://x.com/chat/room")
    assert not path_denied("https://x.com/lessons/intro")


def test_kundli_generator_query():
    assert is_kundli_generator(
        "https://appliedjyotish.com/kundli?dob=1990-01-01&lat=12")
    assert not is_kundli_generator("https://appliedjyotish.com/kundli-fundamentals")


def test_pdf_and_binary():
    assert is_pdf_url("https://vedicastrologer.org/lessons/foo.pdf")
    assert is_binary_url("https://vedicastrologer.org/jhora/setup.exe")
    assert is_binary_url("https://vedicastrologer.org/download/JHora.zip")
    assert not is_binary_url("https://vedicastrologer.org/lessons/foo.pdf")


def test_extract_links_resolves_relative():
    html = '<a href="/a">A</a><a href="https://other.com/x">x</a>'
    links = extract_links(html, "https://astrolearn.co/lessons")
    assert "https://astrolearn.co/a" in links
    assert "https://other.com/x" in links


def test_normalize_strips_fragment_and_trailing_slash():
    assert normalize_url("https://x.com/a/#frag") == "https://x.com/a"
