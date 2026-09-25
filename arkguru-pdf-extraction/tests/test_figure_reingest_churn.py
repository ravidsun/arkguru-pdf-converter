"""Enabling extract_figures renumbers chunk_index, so a re-ingest must delete first.

``Chunk.chunk_id`` is ``hash(source_type, source_id, chunk_index, role)`` and does
not depend on the text. Figure blocks are emitted alongside tables in
``special_blocks``, so turning figure extraction on shifts ``chunk_index`` for
everything after the first figure. The same chunk_id then names *different* text,
which means a bare ``ChunkStore.upsert`` rewrites rows while their existing
embeddings survive: the vector no longer describes the row.

Callers must therefore ``DELETE FROM chunks WHERE source_id = ...`` (the vectors
FK is ON DELETE CASCADE) and re-embed, rather than re-ingesting over the top.
"""
from phase1_pdf.chunk import chunk_document
from phase1_pdf.extract import Block, Document

_PROSE = ("The tenth house governs profession and karma in the natal chart. "
          "Its lord placed with benefics indicates steady advancement. " * 6)

_KW = dict(strategy="structure", target_tokens=400, overlap_pct=0.15,
           min_tokens=80, parent_max_tokens=2000)


def _doc(with_figures: bool, sections: int = 3) -> Document:
    """A document whose pages each carry prose, optionally a figure, then a table."""
    blocks: list[Block] = []
    for s in range(1, sections + 1):
        heading = f"Section {s} Heading"
        blocks.append(Block(text=heading, page=s, heading=heading,
                            heading_level=2, is_heading=True))
        blocks.append(Block(text=_PROSE, page=s, heading=heading))
        if with_figures:
            blocks.append(Block(text=f"Figure on page {s}: dasha wheel labels",
                                page=s, heading=heading, block_type="figure"))
        blocks.append(Block(text=f"| planet | degree |\n| Saturn | {s}0 |",
                            page=s, heading=heading, block_type="table"))
    return Document(source_id="trial.pdf", title="Trial", lang="en", blocks=blocks)


def _rows(doc):
    return [(c.chunk_index, c.extra.get("block_type", "text"), c.chunk_id, c.text)
            for c in chunk_document(doc, **_KW)]


def test_figures_add_chunks():
    without = _rows(_doc(False))
    with_figs = _rows(_doc(True))
    assert len(with_figs) == len(without) + 3
    assert [t for _, t, _, _ in with_figs].count("figure") == 3


def test_enabling_figures_shifts_table_and_prose_indexes():
    without = {t: [i for i, bt, _, _ in _rows(_doc(False)) if bt == t]
               for t in ("text", "table")}
    with_figs = {t: [i for i, bt, _, _ in _rows(_doc(True)) if bt == t]
                 for t in ("text", "table")}
    # tables and prose both move: the figure consumes an index ahead of them
    assert without["table"] != with_figs["table"]
    assert without["text"] != with_figs["text"]


def test_chunk_id_is_reused_for_different_text_so_upsert_would_corrupt():
    """The regression this guards: same chunk_id, different content.

    A bare upsert would rewrite these rows and leave their old embeddings in
    place, silently decoupling each vector from the text it describes.
    """
    before = {cid: (bt, text) for _, bt, cid, text in _rows(_doc(False))}
    after = {cid: (bt, text) for _, bt, cid, text in _rows(_doc(True))}

    reused_with_new_text = [cid for cid in before
                            if cid in after and before[cid][1] != after[cid][1]]
    assert reused_with_new_text, (
        "expected chunk_ids to be reused for different text; if this now fails, "
        "chunk_id derivation changed and the delete-then-reingest rule can be revisited")

    # the strongest form: an id that held a table now holds a figure
    assert any(before[cid][0] != after[cid][0] for cid in reused_with_new_text), (
        "expected at least one chunk_id to change block_type across the re-ingest")


def test_ids_are_stable_when_figures_stay_disabled():
    """Re-ingesting with the same settings is genuinely idempotent."""
    assert [c[2] for c in _rows(_doc(False))] == [c[2] for c in _rows(_doc(False))]
