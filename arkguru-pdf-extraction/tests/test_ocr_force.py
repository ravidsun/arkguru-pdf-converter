"""OCR languages + per-source force_ocr (no real Tesseract / PDFs)."""
from __future__ import annotations

from pathlib import Path

from phase1_pdf.extract import (
    DEFAULT_FORCE_OCR,
    DEFAULT_OCR_LANGUAGES,
    _ocr_to_searchable,
    extract_document,
    should_force_ocr,
)


def test_default_force_ocr_lists_bphs_sharma_and_jkp_4_5():
    assert should_force_ocr(
        "BPHS_Girish_Chand_Sharma_vol1.pdf", DEFAULT_FORCE_OCR)
    assert should_force_ocr(
        "books/J_KP_reader_4_Marriage.pdf", DEFAULT_FORCE_OCR)
    assert should_force_ocr(
        "J_KP reader 5 Transits.pdf", DEFAULT_FORCE_OCR)
    assert not should_force_ocr("KP_Ezine_2019_03.pdf", DEFAULT_FORCE_OCR)


def test_force_ocr_passes_language_and_force_flag(tmp_path: Path, monkeypatch):
    import sys
    import types

    called: dict = {}
    fake = types.ModuleType("ocrmypdf")

    def ocr(_src, dest, **kw):
        called.update(kw)
        Path(dest).write_bytes(b"%PDF")

    fake.ocr = ocr
    monkeypatch.setitem(sys.modules, "ocrmypdf", fake)
    pdf = tmp_path / "BPHS_Sharma.pdf"
    pdf.write_bytes(b"%PDF")
    out = _ocr_to_searchable(
        pdf, tmp_path / "interim",
        language=DEFAULT_OCR_LANGUAGES, force_ocr=True,
    )
    assert out is not None
    assert called["language"] == "eng+hin+san"
    assert called["force_ocr"] is True
    assert called["skip_text"] is False


def test_extract_document_force_ocr_even_when_text_layer_exists(
    tmp_path: Path, monkeypatch,
):
    import phase1_pdf.extract as ext

    called = {"ocr": 0}

    def fake_needs(_path, min_chars_per_page=40):
        return False

    def fake_ocr(path, out_dir, *, language="eng+hin+san", force_ocr=False):
        called["ocr"] += 1
        called["language"] = language
        called["force_ocr"] = force_ocr
        return Path(path)

    def fake_backend(path, extract_tables=True, extract_figures=True,
                     ocr_languages="eng+hin+san"):
        return ext.Document(source_id=str(path))

    monkeypatch.setattr(ext, "_needs_ocr", fake_needs)
    monkeypatch.setattr(ext, "_ocr_to_searchable", fake_ocr)
    monkeypatch.setitem(ext._BACKENDS, "pymupdf4llm", fake_backend)
    monkeypatch.setattr(ext, "strip_repeating_headers_footers", lambda doc: doc)
    monkeypatch.setattr(ext, "_attach_provenance", lambda *a, **k: None)
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    extract_document(pdf, force_ocr=True, ocr_languages="eng+hin+san")
    assert called["ocr"] == 1
    assert called["force_ocr"] is True
    assert called["language"] == "eng+hin+san"
