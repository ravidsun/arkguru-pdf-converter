"""Table linearisation: no ColN, header:value, row windows."""
from __future__ import annotations

from common.tables import (
    cells_from_pymupdf_table,
    is_pipe_heavy,
    linearize_table,
    pack_table_windows,
    split_header_and_body,
    strip_markdown_tables,
)
from common.tokenizer import count_tokens


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def extract(self):
        return self._rows


def test_strips_markdown_tables_from_prose():
    md = (
        "Intro paragraph about dashas.\n\n"
        "| Col1 | Col2 |\n| --- | --- |\n| Saturn | 10 |\n\n"
        "After the table, more prose."
    )
    prose, tables = strip_markdown_tables(md)
    assert "Intro paragraph" in prose
    assert "After the table" in prose
    assert "| Col1 |" not in prose
    assert tables and tables[0][0][0] == "Col1"


def test_linearize_drops_coln_and_uses_header_value():
    rows = [
        ["Col1", "Planet", "Col3"],
        ["ignored-placeholder", "Saturn", "10°"],
    ]
    text = linearize_table(rows)
    assert "|Col" not in text and "| Col" not in text
    assert "Col1" not in text
    assert "Planet: Saturn" in text
    assert "10°" in text


def test_placeholder_only_headers_become_values():
    rows = [["Col1", "Col2"], ["Saturn", "Libra"]]
    headers, body = split_header_and_body(rows)
    assert all(h == "" or not h for h in headers)
    text = linearize_table(rows)
    assert "Col1" not in text
    assert "Saturn" in text and "Libra" in text
    assert "Col1:" not in text


def test_pack_table_windows_repeats_header_under_cap():
    headers = ["Planet", "Sign", "Degree"]
    rows = [headers] + [[f"P{i}", "Aries", str(i)] for i in range(80)]
    text = linearize_table(rows)
    windows = pack_table_windows(text, target_tokens=40, max_tokens=64, overlap_rows=1)
    assert len(windows) > 1
    assert all(count_tokens(w) <= 64 for w in windows)
    assert all(w.startswith("Columns: Planet, Sign, Degree") for w in windows)
    assert all("|Col" not in w for w in windows)


def test_cells_from_pymupdf_never_needs_to_markdown():
    table = _FakeTable([["Planet", "Lord"], ["Saturn", "Shani"]])
    rows = cells_from_pymupdf_table(table)
    assert rows[0][0] == "Planet"
    text = linearize_table(rows)
    assert "Planet: Saturn" in text


def test_is_pipe_heavy():
    assert is_pipe_heavy("| a | b |\n| 1 | 2 |")
    assert not is_pipe_heavy("The sub sub lord is the finest unit.")
