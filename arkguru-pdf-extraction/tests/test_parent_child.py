"""parent_child default: every PDF child has a parent; parents are not embedded."""
from __future__ import annotations

from common.datastore import ChunkStore
from phase1_pdf.chunk import chunk_document
from phase1_pdf.extract import Block, Document
from phase1_pdf.pipeline import Phase1Config


_PROSE = (
    "The seventh house and Saturn describe partnership karma in KP horary. "
    "The sub sub lord of the 7th cusp must be judged with the star lord. "
) * 3


def _doc() -> Document:
    return Document(
        source_id="manual.pdf",
        blocks=[
            Block(text="Marriage", heading="Marriage", heading_level=1,
                  is_heading=True, page=1),
            Block(text=_PROSE, heading="Marriage", page=1),
            Block(
                text="Planet: Saturn; House: 7",
                heading="Marriage", page=1, block_type="table",
            ),
            Block(text="Transits", heading="Transits", heading_level=1,
                  is_heading=True, page=2),
            Block(text=_PROSE.replace("seventh", "tenth"), heading="Transits",
                  page=2),
        ],
    )


def test_config_and_chunk_default_is_parent_child():
    assert Phase1Config().strategy == "parent_child"
    chunks = chunk_document(_doc(), min_chunk_chars=20, min_table_chars=1)
    children = [c for c in chunks if not c.is_parent]
    parents = {c.chunk_id: c for c in chunks if c.is_parent}
    assert parents
    assert children
    for c in children:
        assert c.parent_id
        assert c.parent_id in parents
        assert parents[c.parent_id].is_parent


def test_table_only_section_still_gets_a_parent():
    doc = Document(
        source_id="tables.pdf",
        blocks=[
            Block(text="Planets", heading="Planets", heading_level=1,
                  is_heading=True, page=1),
            Block(
                text="Planet: Saturn; House: 7\nPlanet: Jupiter; House: 9",
                heading="Planets", page=1, block_type="table",
            ),
        ],
    )
    chunks = chunk_document(
        doc, strategy="parent_child", min_chunk_chars=1, min_table_chars=1,
    )
    children = [c for c in chunks if not c.is_parent]
    parents = {c.chunk_id: c for c in chunks if c.is_parent}
    assert children and parents
    assert all(c.parent_id in parents for c in children)


def test_embedding_work_clause_excludes_parents():
    where, _ = ChunkStore(dsn="postgresql://unused")._embedding_work_clause(
        model=None, reembed=False)
    assert "is_parent = false" in where
