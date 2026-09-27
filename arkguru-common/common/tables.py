"""Table linearisation shared by Phase 1 extract and chunk.

One extraction path, typed ``table`` chunks only:

- Markdown pipe tables are stripped from prose (pymupdf4llm emits them inline).
- ``find_tables`` / parsed rows are linearised as ``header: value``.
- ``ColN`` placeholder headers from pymupdf ``to_markdown()`` are dropped.
- Long tables are split by rows; real headers are repeated on each window.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from .text import clean_text
from .tokenizer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TARGET_TOKENS,
    count_tokens,
    effective_max_tokens,
    split_to_max_tokens,
)

_COLN_RE = re.compile(r"^Col\s*\d+$", re.IGNORECASE)
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$"
)


def is_placeholder_header(cell: str) -> bool:
    return bool(_COLN_RE.match((cell or "").strip()))


def is_pipe_heavy(text: str) -> bool:
    """True when text looks like a leftover markdown / pipe table dump."""
    s = (text or "").strip()
    if not s:
        return False
    if s.startswith("|") and "|" in s[1:]:
        return True
    pipes = s.count("|")
    return pipes > 0 and pipes > 0.10 * len(s)


def parse_markdown_table_row(line: str) -> list[str]:
    s = (line or "").strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [clean_text(c) for c in s.split("|")]


def is_markdown_separator(line: str) -> bool:
    return bool(_MD_TABLE_SEP.match(line or ""))


def is_markdown_table_row(line: str) -> bool:
    return bool(_MD_TABLE_ROW.match(line or ""))


def rows_from_markdown(text: str) -> list[list[str]]:
    """Parse the first markdown table in ``text`` into cells."""
    _, tables = strip_markdown_tables(text)
    if tables:
        return tables[0]
    return []


def strip_markdown_tables(md: str) -> tuple[str, list[list[list[str]]]]:
    """Remove markdown tables from ``md``. Return (prose, parsed_tables).

    Parsed tables are available for backends (docling) whose only table path
    is the markdown export. pymupdf backends discard them and use find_tables.
    """
    if not md:
        return "", []
    prose_lines: list[str] = []
    tables: list[list[list[str]]] = []
    buf: list[str] = []

    def flush_table() -> None:
        rows = _rows_from_md_lines(buf)
        buf.clear()
        if rows:
            tables.append(rows)

    for line in md.splitlines():
        if _MD_TABLE_ROW.match(line) or (buf and is_markdown_separator(line)):
            buf.append(line)
            continue
        if buf:
            flush_table()
        prose_lines.append(line)
    if buf:
        flush_table()

    prose = "\n".join(prose_lines)
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
    return prose, tables


def _rows_from_md_lines(lines: Sequence[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if is_markdown_separator(line):
            continue
        if _MD_TABLE_ROW.match(line):
            cells = parse_markdown_table_row(line)
            if any(cells):
                rows.append(cells)
    return _normalize_row_widths(rows)


def cells_from_pymupdf_table(table) -> Optional[list[list[str]]]:
    """Extract raw cells from a pymupdf Table. Never calls ``to_markdown()``."""
    try:
        raw = table.extract()
    except Exception:
        return None
    if not raw:
        return []
    rows: list[list[str]] = []
    for row in raw:
        cells = row if isinstance(row, (list, tuple)) else [row]
        rows.append([
            clean_text("" if c is None else str(c))
            for c in cells
        ])
    return _normalize_row_widths(rows)


def _normalize_row_widths(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def split_header_and_body(
    rows: Sequence[Sequence[str]],
) -> tuple[list[str], list[list[str]]]:
    """First row is the header unless every cell is empty or ``ColN``."""
    if not rows:
        return [], []
    first = [clean_text(c) for c in rows[0]]
    rest = [[clean_text(c) for c in r] for r in rows[1:]]
    if first and any(c and not is_placeholder_header(c) for c in first):
        headers = ["" if is_placeholder_header(c) else c for c in first]
        return headers, rest
    # ColN-only (or empty) first row is not real header text — drop it.
    return [""] * len(first), rest


def linearize_row(headers: Sequence[str], row: Sequence[str]) -> str:
    """One data row as ``header: value`` pairs (semicolon-separated)."""
    parts: list[str] = []
    width = max(len(headers), len(row))
    for i in range(width):
        header = headers[i] if i < len(headers) else ""
        value = row[i] if i < len(row) else ""
        value = clean_text(value)
        if not value:
            continue
        if header and not is_placeholder_header(header):
            parts.append(f"{header}: {value}")
        else:
            parts.append(value)
    return "; ".join(parts)


def columns_line(headers: Sequence[str]) -> str:
    real = [h for h in headers if h and not is_placeholder_header(h)]
    if not real:
        return ""
    return "Columns: " + ", ".join(real)


def linearize_table(rows: Sequence[Sequence[str]]) -> str:
    """Full table as header-context line plus linearised rows."""
    headers, body = split_header_and_body(rows)
    lines: list[str] = []
    ctx = columns_line(headers)
    if ctx:
        lines.append(ctx)
    for row in body:
        line = linearize_row(headers, row)
        if line:
            lines.append(line)
    return "\n".join(lines)


def pack_table_windows(
    text: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_rows: int = 1,
) -> list[str]:
    """Split a linearised table by rows, repeating the ``Columns:`` header."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0] if lines[0].startswith("Columns:") else ""
    data = lines[1:] if header else lines
    cap = effective_max_tokens(max_tokens)
    if not data:
        return [header] if header and count_tokens(header) <= cap else []
    target = min(max(1, target_tokens), cap)
    windows: list[str] = []
    i = 0
    n = len(data)
    while i < n:
        cur: list[str] = []
        prefix = [header] if header else []
        tok = count_tokens(header) if header else 0
        j = i
        while j < n:
            t = count_tokens(data[j])
            # A single oversized row is kept only if it is the whole window;
            # otherwise close before adding it. The row itself is then split
            # by the caller via the hard-cap fallback if needed.
            if cur and tok + t > target:
                break
            if not cur and prefix and tok + t > cap:
                break
            cur.append(data[j])
            tok += t
            j += 1
        if not cur:
            # Row (or header+row) exceeds the cap: emit the row alone.
            cur = [data[i]]
            j = i + 1
        piece = "\n".join(prefix + cur).strip()
        if piece:
            if count_tokens(piece) > cap:
                windows.extend(split_to_max_tokens(piece, cap, overlap_tokens=0))
            else:
                windows.append(piece)
        if j >= n:
            break
        i = max(i + 1, j - max(0, overlap_rows))
    return windows


def iter_linearized_tables(tables: Iterable[Sequence[Sequence[str]]]) -> list[str]:
    out: list[str] = []
    for rows in tables:
        text = linearize_table(rows)
        if text.strip():
            out.append(text)
    return out
