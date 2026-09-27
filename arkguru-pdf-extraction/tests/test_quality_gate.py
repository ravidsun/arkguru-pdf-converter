"""Step 6 done-criterion: garbled is flagged and never silently embedded."""
from __future__ import annotations

from common.quality import annotate_chunks
from phase1_pdf.chunk import chunk_document
from phase1_pdf.extract import Block, Document


def test_garbled_bphs_like_chunk_is_kept_but_not_embeddable():
    junk = ("ÿþÅÆ ØØ §±±± |||| ¤¤¤ ÃÃÃ ###@@@ ~~~~ " * 12)
    doc = Document(
        source_id="BPHS_Sharma_vol1.pdf",
        blocks=[
            Block(text="BPHS", heading="BPHS", heading_level=1, is_heading=True),
            Block(text=junk, heading="BPHS", page=1),
        ],
    )
    chunks = chunk_document(
        doc, strategy="parent_child", min_chunk_chars=1, min_tokens=1,
    )
    chunks = annotate_chunks(chunks)
    children = [c for c in chunks if not c.is_parent]
    assert children
    flagged = [c for c in children if c.extra.get("quality_flagged")]
    assert flagged
    assert all(c.extra["quality"]["embed"] is False for c in flagged)
    assert all(c.lang == "und" for c in flagged)


def test_hr_sv_prose_is_not_flagged():
    hr = (
        "Saturn u sedmom domu znači kašnjenje braka. Horoskop gleda vladara "
        "kuće i aspekte Venere. To je uobičajeno tumačenje u zapadnoj astrologiji. "
    ) * 2
    sv = (
        "Saturnus i sjunde huset betyder fördröjning. Astrologin tittar på "
        "härskaren och aspekterna till Venus för äktenskapet i horoskopet. "
    ) * 2
    for text, src in ((hr, "hr.pdf"), (sv, "sv.pdf")):
        doc = Document(
            source_id=src,
            blocks=[
                Block(text="H", heading="H", heading_level=1, is_heading=True),
                Block(text=text, heading="H", page=1),
            ],
        )
        chunks = annotate_chunks(chunk_document(
            doc, min_chunk_chars=20, min_tokens=1,
        ))
        children = [c for c in chunks if not c.is_parent]
        assert children
        assert all(c.extra["quality"]["passed"] for c in children)
        assert all(c.extra["quality"]["embed"] for c in children)
