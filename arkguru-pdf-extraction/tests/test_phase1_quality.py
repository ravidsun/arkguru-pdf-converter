"""Folder sink splits and extract-quality helpers — no real PDFs required."""

from __future__ import annotations

from pathlib import Path

from common.schema import Chunk, read_jsonl

from phase1_pdf.chunk import chunk_document
from phase1_pdf.extract import (
    Block,
    Document,
    _figure_text_is_usable,
    _n_nonempty_table_rows,
    _ocr_to_searchable,
    _should_ocr_figures,
    strip_repeating_headers_footers,
)
from phase1_pdf.pipeline import (
    partition_file_sink_chunks,
    relative_source_id,
    relative_stem,
    write_file_sink,
)


def _chunk(**kwargs) -> Chunk:
    defaults = dict(
        text="body",
        source_type="pdf",
        source_id="hvac/manual.pdf",
        chunk_index=0,
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def test_relative_source_id_keeps_subfolder(tmp_path: Path) -> None:
    ingest = tmp_path / "raw_pdfs"
    pdf = ingest / "hvac" / "manual.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    assert relative_source_id(pdf, ingest) == "hvac/manual.pdf"
    assert relative_stem("hvac/manual.pdf") == "hvac/manual"
    other = tmp_path / "outside.pdf"
    other.write_bytes(b"%PDF")
    assert relative_source_id(other, ingest) == "outside.pdf"


def test_partition_is_exclusive() -> None:
    parent = _chunk(text="section parent", is_parent=True, chunk_index=0)
    child = _chunk(text="Heading\n\nprose", parent_id=parent.chunk_id, chunk_index=1)
    table = _chunk(
        text="Heading\n\n| a | b |",
        extra={"block_type": "table"},
        chunk_index=2,
    )
    figure = _chunk(
        text="Heading\n\ndiagram text",
        extra={"block_type": "figure"},
        chunk_index=3,
    )
    parts = partition_file_sink_chunks([parent, child, table, figure])
    assert parts["parents"] == [parent]
    assert parts["chunks"] == [child]
    assert parts["tables"] == [table]
    assert parts["figures"] == [figure]
    ids = [c.chunk_id for rows in parts.values() for c in rows]
    assert len(ids) == len(set(ids))


def test_write_file_sink_omits_empty_splits(tmp_path: Path) -> None:
    parent = _chunk(text="parent", is_parent=True, chunk_index=0)
    child = _chunk(text="Safety\n\nLockout tagout.", chunk_index=1)
    table = _chunk(
        text="Safety\n\n| col |",
        extra={"block_type": "table"},
        chunk_index=2,
    )
    folder = write_file_sink(
        [parent, child, table], tmp_path, "hvac/manual", "jsonl",
    )
    assert folder == tmp_path / "hvac" / "manual"
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["chunks.jsonl", "parents.jsonl", "tables.jsonl"]
    assert not (folder / "figures.jsonl").exists()
    loaded = read_jsonl(folder / "tables.jsonl")
    assert loaded[0].extra["block_type"] == "table"
    assert all(not c.is_parent for c in read_jsonl(folder / "chunks.jsonl"))


def test_strip_repeating_headers_footers() -> None:
    header = "ACME Field Manual — Confidential"
    doc = Document(
        source_id="hvac/manual.pdf",
        blocks=[
            Block(text=f"{header}\nTorque the flange bolts.", page=1),
            Block(text=f"{header}\nBleed the manifold slowly.", page=2),
            Block(text="Unique body on page three only.", page=3),
        ],
    )
    strip_repeating_headers_footers(doc)
    joined = "\n".join(b.text for b in doc.blocks)
    assert header not in joined
    assert "Torque the flange bolts." in joined
    assert "Bleed the manifold slowly." in joined
    assert "Unique body on page three only." in joined


def test_strip_headers_noop_single_page() -> None:
    line = "Running header"
    doc = Document(
        source_id="one.pdf",
        blocks=[Block(text=f"{line}\nOnly one page.", page=1)],
    )
    strip_repeating_headers_footers(doc)
    assert line in doc.blocks[0].text


def test_child_and_table_text_prefixed_with_heading() -> None:
    doc = Document(
        source_id="hvac/manual.pdf",
        blocks=[
            Block(text="Safety", heading="Safety", heading_level=1, is_heading=True, page=1),
            Block(text="Disconnect mains before service.", heading="Safety", page=1),
            Block(
                text="| A | B |\n| --- | --- |\n| 1 | 2 |",
                heading="Safety",
                page=1,
                block_type="table",
            ),
        ],
    )
    chunks = chunk_document(doc, strategy="structure", min_tokens=1, target_tokens=400)
    prose = [c for c in chunks if not c.extra.get("block_type")]
    tables = [c for c in chunks if c.extra.get("block_type") == "table"]
    assert prose and prose[0].text.startswith("Safety\n\n")
    assert prose[0].section == "Safety"
    assert tables and tables[0].text.startswith("Safety\n\n")
    assert tables[0].section == "Safety"


def test_drop_tables_with_fewer_than_two_rows() -> None:
    class OneRow:
        def extract(self):
            return [["only"]]

    class TwoRows:
        def extract(self):
            return [["h1", "h2"], ["a", "b"]]

    assert _n_nonempty_table_rows(OneRow()) == 1
    assert _n_nonempty_table_rows(TwoRows()) == 2


def test_figure_ocr_junk_filter() -> None:
    assert not _figure_text_is_usable("@@@###")
    assert not _figure_text_is_usable("ab")
    usable = "Compressor discharge temperature sensor reading"
    assert _figure_text_is_usable(usable)


class _FakePage:
    def __init__(self, images: list, native: str) -> None:
        self._images = images
        self._native = native

    def get_images(self, full=True):
        return self._images

    def get_text(self, _kind: str) -> str:
        return self._native


def test_figure_ocr_skipped_on_text_native_pages() -> None:
    assert not _should_ocr_figures(_FakePage([(1,)], "n" * 200))
    assert not _should_ocr_figures(_FakePage([], "fig"))
    assert _should_ocr_figures(_FakePage([(1,)], "fig"))


def test_ocr_to_searchable_uses_skip_text(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    called: dict = {}
    fake = types.ModuleType("ocrmypdf")

    def ocr(_src, dest, skip_text=False, force_ocr=True, progress_bar=True):
        called["skip_text"] = skip_text
        called["force_ocr"] = force_ocr
        Path(dest).write_bytes(b"%PDF")

    fake.ocr = ocr
    monkeypatch.setitem(sys.modules, "ocrmypdf", fake)
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    out = _ocr_to_searchable(pdf, tmp_path / "interim")
    assert out is not None
    assert out.name == "ocr_scan.pdf"
    assert called["skip_text"] is True
    assert called["force_ocr"] is False


def test_ocr_to_searchable_none_without_ocrmypdf(tmp_path: Path, monkeypatch) -> None:
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ocrmypdf":
            raise ImportError("no ocrmypdf")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "ocrmypdf", raising=False)
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    assert _ocr_to_searchable(pdf, tmp_path / "interim") is None
