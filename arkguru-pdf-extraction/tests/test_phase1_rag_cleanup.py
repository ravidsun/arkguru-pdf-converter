"""Phase 1 done-criteria for clean_text, hard cap, tables, and min-size."""
from __future__ import annotations

import re

from common.tables import is_pipe_heavy
from common.text import body_without_heading
from common.tokenizer import DEFAULT_MAX_TOKENS, count_tokens
from phase1_pdf.chunk import chunk_document
from phase1_pdf.extract import Block, Document, _markdown_to_blocks

_INLINE_TAG = re.compile(r"(?is)</?(?:mark|u|sup|br)\s*/?>")
_COLN = re.compile(r"\|Col\s*\d+\|", re.IGNORECASE)


def _chunks(doc: Document, **kwargs):
    defaults = dict(
        strategy="structure",
        target_tokens=400,
        overlap_pct=0.15,
        min_tokens=80,
        max_tokens=DEFAULT_MAX_TOKENS,
        min_chunk_chars=80,
        min_table_chars=40,
        min_figure_chars=40,
    )
    defaults.update(kwargs)
    return chunk_document(doc, **defaults)


def test_fixture_chunks_have_zero_inline_tags():
    doc = Document(
        source_id="kp-ezine.pdf",
        blocks=[
            Block(text="KP Horary", heading="KP Horary", heading_level=1, is_heading=True),
            Block(
                text=(
                    "The <mark>sub sub lord</mark> of the querent is <u>Saturn</u>. "
                    "Strength is x<sup>2</sup>.<br>Also ** benefic ** and _malefic_ "
                    "yogas. " * 8
                    + "ग्रही शुक्र की दशा । " * 6
                ),
                heading="KP Horary",
                page=1,
            ),
        ],
    )
    chunks = _chunks(doc)
    assert chunks
    joined = "\n".join(c.text for c in chunks)
    assert _INLINE_TAG.search(joined) is None
    assert "<mark>" not in joined
    assert "</u>" not in joined
    assert "शुक्र" in joined


def test_markdown_to_blocks_strips_tags_and_tables():
    md = (
        "# Vimshottari\n\n"
        "The <mark>dasha</mark> lord.\n\n"
        "| Col1 | Col2 |\n| --- | --- |\n| a | b |\n\n"
        "More **prose** after."
    )
    blocks = _markdown_to_blocks(md, page=1, table_mode="strip")
    prose = [b for b in blocks if not b.is_heading]
    assert all(b.block_type == "text" for b in prose)
    joined = "\n".join(b.text for b in prose)
    assert _INLINE_TAG.search(joined) is None
    assert "| Col1 |" not in joined
    assert all(not is_pipe_heavy(b.text) for b in prose)
    assert "prose after" in joined.lower() or "More prose after" in joined


def test_markdown_to_blocks_promotes_docling_tables():
    md = "# Houses\n\n| Planet | House |\n| --- | --- |\n| Saturn | 7 |\n"
    blocks = _markdown_to_blocks(md, page=None, table_mode="promote")
    tables = [b for b in blocks if b.block_type == "table"]
    assert tables
    assert "Planet: Saturn" in tables[0].text
    assert "Col1" not in tables[0].text
    assert all(not is_pipe_heavy(b.text) for b in blocks if not b.is_heading)


def test_hard_cap_on_190k_headingless_section():
    blob = ("nakshatra pada and the sub lord of the cusp " * 90 + "\n") * 80
    assert len(blob) > 190_000
    doc = Document(
        source_id="giant.pdf",
        blocks=[Block(text=blob, page=1)],
    )
    chunks = _chunks(doc, min_chunk_chars=1, min_tokens=1)
    assert chunks
    assert all((c.token_count or count_tokens(c.text)) <= DEFAULT_MAX_TOKENS for c in chunks)
    assert all(count_tokens(c.text) <= DEFAULT_MAX_TOKENS for c in chunks)


def test_tables_are_typed_linearised_without_coln_or_pipes():
    doc = Document(
        source_id="tables.pdf",
        blocks=[
            Block(text="Planets", heading="Planets", heading_level=1, is_heading=True),
            Block(
                text="Saturn as the 7th lord indicates delay in marriage when afflicted.",
                heading="Planets",
                page=1,
            ),
            Block(
                text="| Col1 | Planet | Degree |\n| --- | --- | --- |\n| x | Saturn | 10 |\n| y | Jupiter | 20 |",
                heading="Planets",
                page=1,
            ),
            Block(
                text=(
                    "Columns: Planet, Sign\n"
                    + "\n".join(f"Planet: P{i}; Sign: Aries" for i in range(60))
                ),
                heading="Planets",
                page=1,
                block_type="table",
            ),
        ],
    )
    chunks = _chunks(doc, min_chunk_chars=20, min_table_chars=10)
    tables = [c for c in chunks if (c.extra or {}).get("block_type") == "table"]
    prose = [c for c in chunks if not (c.extra or {}).get("block_type")]
    assert tables
    assert all(not is_pipe_heavy(c.text) for c in prose)
    joined = "\n".join(c.text for c in chunks)
    assert _COLN.search(joined) is None
    assert "Planet: Saturn" in joined or "Saturn" in joined
    assert all(c.extra.get("block_type") == "table" for c in tables)
    assert len(tables) >= 2  # long table split by rows
    assert any(t.text.count("Columns: Planet, Sign") >= 1 for t in tables)


def test_keep_tables_false_drops_all_tables():
    doc = Document(
        source_id="notables.pdf",
        blocks=[
            Block(text="Intro", heading="Intro", heading_level=1, is_heading=True),
            Block(text="Prose about dasha periods that is long enough to keep. " * 4,
                  heading="Intro", page=1),
            Block(
                text="| Planet | Degree |\n| --- | --- |\n| Saturn | 10 |",
                heading="Intro",
                page=1,
            ),
        ],
    )
    chunks = _chunks(doc, keep_tables=False, min_chunk_chars=20)
    assert chunks
    assert all((c.extra or {}).get("block_type") != "table" for c in chunks)
    assert all(not is_pipe_heavy(c.text) for c in chunks)
    assert _COLN.search("\n".join(c.text for c in chunks)) is None


def test_min_size_merges_and_drops_heading_only():
    blocks: list[Block] = []
    for i, title in enumerate(("A", "B", "C", "D", "E"), start=1):
        blocks.append(Block(
            text=title, heading=title, heading_level=2, is_heading=True, page=1,
        ))
        blocks.append(Block(text="ok.", heading=title, page=1))
    blocks.append(Block(
        text="Main", heading="Main", heading_level=1, is_heading=True, page=2,
    ))
    blocks.append(Block(
        text=(
            "The seventh house and Saturn describe partnership karma in KP horary. "
            "The sub sub lord of the 7th cusp must be judged with the star lord. "
        ) * 3,
        heading="Main",
        page=2,
    ))
    doc = Document(source_id="tiny.pdf", blocks=blocks)
    chunks = _chunks(doc, min_chunk_chars=80, min_tokens=20)
    child = [c for c in chunks if not c.is_parent]
    assert child
    heading_only = [
        c for c in child
        if not body_without_heading(c.text, c.section).strip()
    ]
    assert heading_only == []
    under_100 = [c for c in child if len(c.text) < 100]
    assert len(under_100) / len(child) < 0.005
