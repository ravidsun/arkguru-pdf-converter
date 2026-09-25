"""
Phase 1, step 1: PDF -> clean text with page + heading structure.

Extraction backends (auto-selected, override via config):
  - "pymupdf4llm"  : fast, native-text PDFs, emits Markdown with headings.  [default]
  - "docling"      : slower but far better tables / complex layout (IBM).
  - "pymupdf"      : raw text fallback, no markdown structure.

Scanned / image-only pages are detected (near-zero extractable text) and, if
`ocr.enabled`, routed through an OCR pass (ocrmypdf preferred; pytesseract as a
per-page fallback). Everything degrades gracefully: a missing optional library
disables that backend rather than crashing the run.

Output of `extract_document()` is a `Document` of `Block`s -- backend-neutral,
so `chunk.py` never needs to know which extractor ran.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

log = logging.getLogger("phase1.extract")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

_EMPH_RE = re.compile(r"[*_`#]+")
_PLACEHOLDER_TITLES = {"", "(anonymous)", "untitled", "unknown"}


def _clean_heading(text: str) -> str:
    return _EMPH_RE.sub("", text).strip()


def _norm_title(t):
    if t is None:
        return None
    t = str(t).strip()
    return None if t.lower() in _PLACEHOLDER_TITLES else t




@dataclass
class Block:
    text: str
    page: Optional[int] = None
    heading: Optional[str] = None     # nearest section heading above this block
    heading_level: int = 0            # 0 = body text, 1-6 = heading itself
    is_heading: bool = False
    block_type: str = "text"          # "text" | "table" | "figure"


@dataclass
class Document:
    source_id: str
    title: Optional[str] = None
    lang: Optional[str] = None
    blocks: list[Block] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Markdown -> Blocks (shared by markdown-producing backends)
# ---------------------------------------------------------------------------
def _markdown_to_blocks(md: str, page: Optional[int]) -> list[Block]:
    blocks: list[Block] = []
    current_heading: Optional[str] = None
    buf: list[str] = []

    def flush():
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(Block(text=text, page=page, heading=current_heading))
            buf.clear()

    for line in md.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            htext = _clean_heading(m.group(2))
            current_heading = htext
            blocks.append(Block(text=htext, page=page, heading=htext,
                                heading_level=level, is_heading=True))
        else:
            buf.append(line)
    flush()
    return blocks


# ---------------------------------------------------------------------------
# Tables & figures (diagrams/charts/graphs) -- shared by pymupdf-based backends
# ---------------------------------------------------------------------------
_MIN_FIGURE_OCR_CHARS = 40          # ignore OCR noise / near-empty images
_MIN_FIGURE_LETTER_RATIO = 0.35     # drop OCR that is mostly non-letters
_MIN_TABLE_ROWS = 2                 # one-row "tables" are layout junk
_FIGURE_OCR_MAX_NATIVE_CHARS = 80   # skip figure OCR on text-native pages
_HEADER_MAX_LINE_LEN = 80
_HEADER_MIN_PAGE_FRACTION = 0.5


def _n_nonempty_table_rows(table) -> int | None:
    """Count non-empty rows, or None if the table object cannot be extracted."""
    try:
        rows = table.extract()
    except Exception:
        return None
    if not rows:
        return 0
    n = 0
    for row in rows:
        cells = row if isinstance(row, (list, tuple)) else [row]
        if any(c is not None and str(c).strip() for c in cells):
            n += 1
    return n


def _table_to_markdown(table) -> Optional[str]:
    """Best-effort conversion of a pymupdf Table object to a markdown table."""
    try:
        return table.to_markdown()
    except Exception:
        pass
    try:
        rows = table.extract()
    except Exception:
        return None
    if not rows:
        return None
    rows = [[("" if c is None else str(c).strip()) for c in row] for row in rows]
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def _extract_tables(page, page_num: int, heading: Optional[str]) -> list["Block"]:
    """Extract tables on a page as markdown blocks (block_type='table')."""
    blocks: list[Block] = []
    try:
        finder = page.find_tables()
    except Exception:
        return blocks
    for table in getattr(finder, "tables", []) or []:
        n_rows = _n_nonempty_table_rows(table)
        if n_rows is not None and n_rows < _MIN_TABLE_ROWS:
            continue
        md = _table_to_markdown(table)
        if md and md.strip():
            blocks.append(Block(text=md.strip(), page=page_num, heading=heading,
                                block_type="table"))
    return blocks


@lru_cache(maxsize=1)
def _ocr_engine():
    """Return a (pytesseract, PIL.Image) tuple, or None if unavailable."""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract, Image
    except ImportError:
        return None


def _extract_figures(pdf, page, page_num: int, heading: Optional[str]) -> list["Block"]:
    """OCR embedded images (diagrams/charts/graphs) so their text is captured.

    Skipped silently if pytesseract/Pillow are not installed -- this is an
    optional enrichment, not a hard dependency.
    """
    engine = _ocr_engine()
    if engine is None:
        return []
    pytesseract, Image = engine
    import io
    blocks: list[Block] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            base = pdf.extract_image(xref)
            im = Image.open(io.BytesIO(base["image"]))
            text = pytesseract.image_to_string(im).strip()
        except Exception:
            continue
        if _figure_text_is_usable(text):
            blocks.append(Block(text=text, page=page_num, heading=heading,
                                block_type="figure"))
    return blocks


def _figure_text_is_usable(text: str) -> bool:
    if len(text) < _MIN_FIGURE_OCR_CHARS:
        return False
    letters = sum(1 for c in text if c.isalpha())
    return (letters / len(text)) >= _MIN_FIGURE_LETTER_RATIO


def _should_ocr_figures(page) -> bool:
    """Figure OCR is expensive; only run on image-heavy pages with little native text."""
    try:
        images = page.get_images(full=True)
    except Exception:
        return False
    if not images:
        return False
    try:
        native = page.get_text("text").strip()
    except Exception:
        native = ""
    return len(native) < _FIGURE_OCR_MAX_NATIVE_CHARS


def _augment_with_tables_and_figures(
    doc: "Document", path: Path, extract_tables: bool, extract_figures: bool,
) -> None:
    """Insert table/figure blocks (in page order) into an already-extracted doc."""
    if not extract_tables and not extract_figures:
        return
    try:
        import pymupdf
    except ImportError:
        return
    # nearest heading seen so far per page, so tables/figures attach to a section
    heading_by_page: dict[int, Optional[str]] = {}
    cur_heading: Optional[str] = None
    for b in doc.blocks:
        if b.is_heading:
            cur_heading = b.heading
        if b.page is not None and b.page not in heading_by_page:
            heading_by_page[b.page] = cur_heading

    extra_blocks: list[Block] = []
    with pymupdf.open(str(path)) as pdf:
        for i, page in enumerate(pdf, start=1):
            heading = heading_by_page.get(i)
            if extract_tables:
                extra_blocks.extend(_extract_tables(page, i, heading))
            if extract_figures and _should_ocr_figures(page):
                extra_blocks.extend(_extract_figures(pdf, page, i, heading))
    doc.blocks.extend(extra_blocks)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _extract_pymupdf4llm(path: Path, extract_tables: bool = True,
                         extract_figures: bool = True) -> Document:
    import pymupdf4llm
    # page_chunks=True returns one dict per page with 'text' (markdown) + meta.
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    doc = Document(source_id=path.name)
    for i, pg in enumerate(pages, start=1):
        md = pg.get("text", "") if isinstance(pg, dict) else str(pg)
        doc.blocks.extend(_markdown_to_blocks(md, page=i))
    if pages and isinstance(pages[0], dict):
        doc.meta = pages[0].get("metadata", {}) or {}
        doc.title = _norm_title(doc.meta.get("title"))
    _augment_with_tables_and_figures(doc, path, extract_tables, extract_figures)
    return doc


def _extract_docling(path: Path, extract_tables: bool = True,
                     extract_figures: bool = True) -> Document:
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    result = conv.convert(str(path))
    md = result.document.export_to_markdown()
    doc = Document(source_id=path.name)
    # Docling markdown is not per-page; treat whole doc as page None but keep
    # heading structure (which is what chunking actually relies on).
    doc.blocks = _markdown_to_blocks(md, page=None)
    doc.title = getattr(result.document, "name", None)
    # Docling already models tables/figures natively in its markdown export,
    # so no separate pymupdf augmentation pass is needed here.
    return doc


def _extract_pymupdf(path: Path, extract_tables: bool = True,
                     extract_figures: bool = True) -> Document:
    import pymupdf  # aka fitz
    doc = Document(source_id=path.name)
    with pymupdf.open(str(path)) as pdf:
        doc.meta = pdf.metadata or {}
        doc.title = _norm_title(doc.meta.get("title"))
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                doc.blocks.append(Block(text=text, page=i))
    _augment_with_tables_and_figures(doc, path, extract_tables, extract_figures)
    return doc


# ---------------------------------------------------------------------------
# OCR detection + fallback
# ---------------------------------------------------------------------------
def _scanned_page_ratio(path: Path, min_chars_per_page: int = 40) -> float:
    """Fraction of pages with almost no extractable text (i.e. scanned/image)."""
    try:
        import pymupdf
    except ImportError:
        return 0.0
    empty = total = 0
    with pymupdf.open(str(path)) as pdf:
        for page in pdf:
            total += 1
            if len(page.get_text("text").strip()) < min_chars_per_page:
                empty += 1
    return (empty / total) if total else 0.0


def _needs_ocr(path: Path, min_chars_per_page: int = 40) -> bool:
    """True if ANY page looks scanned -- covers both fully-scanned PDFs and
    mixed documents (e.g. a native-text report with a few scanned appendix
    pages). `_ocr_to_searchable` uses `skip_text=True` so pages that already
    have text are left untouched, making it safe to OCR eagerly here."""
    return _scanned_page_ratio(path, min_chars_per_page) > 0.0


def _ocr_to_searchable(path: Path, out_dir: Path) -> Optional[Path]:
    """Run ocrmypdf to add a text layer to scanned/image-only pages.

    Uses `skip_text=True` (rather than `force_ocr`) so pages that already
    contain a native text layer are left byte-for-byte alone -- this is what
    makes OCR safe to run on mixed scanned/native documents.
    """
    try:
        import ocrmypdf
    except ImportError:
        log.warning("ocrmypdf not installed; skipping OCR for %s", path.name)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ocr_{path.name}"
    try:
        ocrmypdf.ocr(str(path), str(out_path), skip_text=True,
                     force_ocr=False, progress_bar=False)
        return out_path
    except Exception as e:  # pragma: no cover
        log.warning("OCR failed for %s: %s", path.name, e)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_BACKENDS = {
    "pymupdf4llm": _extract_pymupdf4llm,
    "docling": _extract_docling,
    "pymupdf": _extract_pymupdf,
}


def extract_document(
    path: str | Path,
    backend: str = "pymupdf4llm",
    ocr_enabled: bool = True,
    interim_dir: str | Path = "data/interim",
    extract_tables: bool = True,
    extract_figures: bool = True,
    source_id: str | None = None,
) -> Document:
    path = Path(path)
    original = path
    if ocr_enabled and _needs_ocr(path):
        ratio = _scanned_page_ratio(path)
        log.info("%s looks scanned (%.0f%% of pages) -> OCR", path.name, ratio * 100)
        ocred = _ocr_to_searchable(path, Path(interim_dir))
        if ocred is not None:
            path = ocred

    used = backend
    fn = _BACKENDS.get(backend)
    if fn is None:
        raise ValueError(f"Unknown backend '{backend}'. Options: {list(_BACKENDS)}")

    try:
        doc = fn(path, extract_tables=extract_tables, extract_figures=extract_figures)
    except ImportError as e:
        log.warning("Backend '%s' unavailable (%s); falling back to pymupdf.",
                    backend, e)
        used = "pymupdf"
        doc = _extract_pymupdf(path, extract_tables=extract_tables,
                               extract_figures=extract_figures)

    doc.source_id = source_id if source_id else original.name
    strip_repeating_headers_footers(doc)
    _detect_lang(doc)
    _attach_provenance(doc, original, used)
    return doc


def _line_key(line: str) -> str:
    return " ".join(line.split()).casefold()


def strip_repeating_headers_footers(
    doc: Document,
    *,
    min_page_fraction: float = _HEADER_MIN_PAGE_FRACTION,
    max_line_len: int = _HEADER_MAX_LINE_LEN,
) -> Document:
    """Remove short lines that appear on a large fraction of pages (running headers/footers).

    Mutates ``doc.blocks`` in place and returns ``doc``. No-op when fewer than
    two pages are present (page-less backends such as docling).
    """
    short_by_page: dict[int, set[str]] = {}
    for b in doc.blocks:
        if b.page is None or b.is_heading or b.block_type != "text":
            continue
        keys = short_by_page.setdefault(b.page, set())
        for line in b.text.splitlines():
            s = line.strip()
            if s and len(s) <= max_line_len:
                keys.add(_line_key(s))

    n_pages = len(short_by_page)
    if n_pages < 2:
        return doc

    page_hits: Counter[str] = Counter()
    for keys in short_by_page.values():
        page_hits.update(keys)
    repeating = {
        k for k, n in page_hits.items()
        if (n / n_pages) >= min_page_fraction
    }
    if not repeating:
        return doc

    kept: list[Block] = []
    for b in doc.blocks:
        if b.is_heading or b.block_type != "text":
            kept.append(b)
            continue
        new_lines = [
            line for line in b.text.splitlines()
            if not (
                line.strip()
                and len(line.strip()) <= max_line_len
                and _line_key(line) in repeating
            )
        ]
        text = "\n".join(new_lines).strip()
        if text:
            kept.append(replace(b, text=text))
    doc.blocks = kept
    return doc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _attach_provenance(doc: Document, path: Path, backend: str) -> None:
    meta = dict(doc.meta or {})
    meta["backend"] = backend
    try:
        meta["sha256"] = _file_sha256(path)
    except OSError:
        pass
    pages = [b.page for b in doc.blocks if b.page is not None]
    if pages:
        meta["page_count"] = max(pages)
    else:
        try:
            import pymupdf
            with pymupdf.open(str(path)) as pdf:
                meta["page_count"] = pdf.page_count
        except Exception:
            meta["page_count"] = 0
    doc.meta = meta


def _detect_lang(doc: Document) -> None:
    try:
        from langdetect import detect
        sample = " ".join(b.text for b in doc.blocks[:5])[:2000]
        if sample.strip():
            doc.lang = detect(sample)
    except Exception:
        doc.lang = None
