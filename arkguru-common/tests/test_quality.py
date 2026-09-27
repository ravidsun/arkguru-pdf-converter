"""Quality gate: flag garbage, keep Croatian/Swedish/diacritics/formulas/Ezine."""
from __future__ import annotations

from common.lang import DEFAULT_LANG
from common.quality import annotate_chunks, score_quality
from common.schema import Chunk


def _chunk(text: str, source_id: str = "doc.pdf", **kw) -> Chunk:
    return Chunk(
        text=text, source_type="pdf", source_id=source_id, chunk_index=0, **kw,
    )


_EZINE = (
    "The sub sub lord of the 7th cusp is Saturn. In KP horary the star lord "
    "must be judged with the significators of marriage. Vimśottarī daśā of "
    "Śani running in the bhukti of Śukra delays the event when afflicted. "
    "This is the same prose style used in KP Ezine issues."
)

_CROATIAN = (
    "Ljubav i brak u horoskopu. Saturn u sedmom domu znači kašnjenje i "
    "odgovornost u vezi. Horoskop braka gleda kuću partnerstva i njezine "
    "vladare. Često se gleda i Venera u aspektu sa Saturnom."
)

_SWEDISH = (
    "Saturnus i sjunde huset betyder fördröjning i äktenskapet. "
    "Astrologin tittar på härskaren över huset och aspekterna till Venus. "
    "En transiterande Saturnus kan också förklara en sen förlovning."
)

_FORMULA = (
    "Lagna = 15° Aries; 7th cusp = 20° Libra; Ārūḍha Lagna = 4th from "
    "lagna lord. SSL(7) = Saturn. P(A∩B)/P(B) is the conditional used in "
    "the example. x^2 + y^2 = r^2 for the chart wheel."
)

_GARBLED = (
    "ÿþÅÆØØ §±±± |||| ¤¤¤ ÃÃÃ ###@@@ ~~~~ `` `` "
    "\ufffd\ufffd\ufffd %%%% ^^^^ "
) * 8


def test_ezine_croatian_swedish_formula_pass():
    for text in (_EZINE, _CROATIAN, _SWEDISH, _FORMULA):
        q = score_quality(text)
        assert q.passed, (text[:40], q)
        assert q.embed


def test_garbled_fails_and_is_not_dropped():
    q = score_quality(_GARBLED)
    assert not q.passed
    assert q.embed is False
    chunks = annotate_chunks([_chunk(_GARBLED, "bphs_sharma.pdf")])
    assert len(chunks) == 1
    assert chunks[0].extra["quality"]["passed"] is False
    assert chunks[0].extra["quality_flagged"] is True
    assert chunks[0].extra["quality"]["embed"] is False
    assert chunks[0].lang == DEFAULT_LANG


def test_devanagari_readable_passes():
    text = (
        "बृहत्पाराशरहोराशास्त्र में दशा विचार का वर्णन है। "
        "विंशोत्तरी दशा में शनि की महादशा विवाह में विलम्ब देती है। "
        "लग्न से सप्तम भाव के कारक शुक्र को देखना चाहिए।"
    )
    q = score_quality(text)
    assert q.passed, q


def test_allow_and_deny_source_lists():
    deny = annotate_chunks(
        [_chunk(_EZINE, "drop-me.pdf")],
        quality_cfg={"deny_sources": ["*drop-me*"]},
    )
    assert deny[0].extra["quality"]["passed"] is False
    allow = annotate_chunks(
        [_chunk(_GARBLED, "keep-me.pdf")],
        quality_cfg={"allow_sources": ["*keep-me*"]},
    )
    assert allow[0].extra["quality"]["passed"] is True
    assert allow[0].extra["quality"]["embed"] is True


def test_annotate_sets_content_hash_and_nonempty_lang():
    chunks = annotate_chunks([_chunk(_EZINE)])
    assert chunks[0].content_hash
    assert chunks[0].extra["content_hash"] == chunks[0].content_hash
    assert chunks[0].lang
    assert chunks[0].lang != ""
