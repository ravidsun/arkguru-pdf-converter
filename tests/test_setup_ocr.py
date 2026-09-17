"""setup_dev_env.sh must install the ocrmypdf stack (no live apt)."""
from pathlib import Path

_SETUP = Path(__file__).resolve().parents[1] / "scripts" / "setup_dev_env.sh"


def test_setup_installs_ocr_system_packages():
    text = _SETUP.read_text()
    assert "ensure_ocr_packages" in text
    assert "tesseract-ocr" in text
    assert "tesseract-ocr-eng" in text
    assert "ghostscript" in text


def test_setup_pips_ocrmypdf_stack_for_phase1():
    text = _SETUP.read_text()
    assert "ocrmypdf>=16.0" in text
    assert "pytesseract>=0.3.10" in text
    assert "pillow>=10.0" in text
