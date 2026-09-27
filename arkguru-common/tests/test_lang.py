"""Per-chunk seeded language detection + Devanagari script check."""
from __future__ import annotations

from common.lang import DEFAULT_LANG, detect_lang, reconcile_lang, script_family
from common.quality import annotate_chunks
from common.schema import Chunk


def test_empty_is_und():
    assert detect_lang("") == DEFAULT_LANG
    assert detect_lang("   ") == DEFAULT_LANG


def test_devanagari_is_not_sv_or_ne():
    text = (
        "बृहत्पाराशरहोराशास्त्र विंशोत्तरी दशा शनि शुक्र "
        "लग्न सप्तम भाव कारक ग्रह गोचर"
    )
    assert script_family(text) == "deva"
    assert reconcile_lang("sv", "deva") == "hi"
    assert reconcile_lang("ne", "deva") == "hi"
    assert reconcile_lang("hr", "deva") == "hi"
    code = detect_lang(text)
    assert code not in {"sv", "sl", "hr", "ne", "und", ""}
    assert code == "hi" or code in {"hi", "sa", "mr"}


def test_detect_lang_is_deterministic():
    text = (
        "Saturn in the seventh house delays marriage when the sub lord "
        "is afflicted by malefics in KP horary judgment."
    )
    a = detect_lang(text, seed=0)
    b = detect_lang(text, seed=0)
    assert a == b
    assert a != ""


def test_annotate_never_leaves_empty_lang():
    child = Chunk(
        text="Saturn in the seventh house. " * 8,
        source_type="web", source_id="https://example.com/a", chunk_index=0,
        lang="",
    )
    parent = Chunk(
        text="Saturn in the seventh house. " * 8,
        source_type="pdf", source_id="a.pdf", chunk_index=0,
        is_parent=True, lang=None,
    )
    out = annotate_chunks([child, parent])
    assert all(c.lang and c.lang.strip() for c in out)


def test_garbled_latin_does_not_keep_false_lang():
    junk = "ÿþÅÆØØ §±±± |||| ¤¤¤ ÃÃÃ ###@@@ " * 10
    c = Chunk(text=junk, source_type="pdf", source_id="jkp4.pdf", chunk_index=0)
    out = annotate_chunks([c])
    assert out[0].lang == DEFAULT_LANG
    assert out[0].extra["quality"]["passed"] is False
