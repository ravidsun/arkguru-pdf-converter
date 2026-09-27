"""clean_text: strip tags/emphasis, keep Devanagari and diacritics."""
from __future__ import annotations

import re

from common.text import body_without_heading, clean_text

_INLINE_TAG = re.compile(r"(?is)</?(?:mark|u|sup|br|span|em|strong)\b[^>]*/?>")


def test_strips_inline_html_and_unwraps_emphasis():
    raw = (
        "The <mark>sub sub lord</mark> is <u>critical</u> in KP. "
        "x<sup>2</sup><br>**bold** and _italic_ stay readable."
    )
    out = clean_text(raw)
    assert _INLINE_TAG.search(out) is None
    assert "<" not in out and ">" not in out
    assert "**" not in out
    assert "sub sub lord" in out
    assert "critical" in out
    assert "bold" in out
    assert "italic" in out
    assert "x2" in out or "x 2" in out


def test_preserves_devanagari_and_diacritics():
    raw = "ग्रही <mark>शुक्र</mark> की **दशा** । Vimśottarī and a\u0301cute."
    out = clean_text(raw)
    assert "ग्रही" in out
    assert "शुक्र" in out
    assert "दशा" in out
    assert "।" in out
    assert "Vimśottarī" in out
    # Combining acute on "a" is composed (á), not stripped.
    assert "á" in out
    assert _INLINE_TAG.search(out) is None


def test_does_not_shred_snake_case_or_repeat():
    raw = "sub_sub_lord of the cusp"
    assert clean_text(raw) == raw
    tagged = "once <mark>only</mark>"
    assert clean_text(clean_text(tagged)) == clean_text(tagged)


def test_body_without_heading_detects_heading_only():
    assert body_without_heading("Safety\n\n", "Safety") == ""
    assert body_without_heading("Safety", "Safety") == ""
    assert "Lockout" in body_without_heading("Safety\n\nLockout tagout.", "Safety")
