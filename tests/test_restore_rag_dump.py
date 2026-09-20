"""restore_rag_dump scripts must pin the portable dump checksum and counts."""
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SHA = "90b21d209c211b48378cf951e349c8ea954cd65f4804385d654659c4d222968c"
_COUNTS = ("109163", "60")


def test_bash_restore_pins_complete_dump():
    text = (_SCRIPTS / "restore_rag_dump.sh").read_text()
    assert _SHA in text
    assert "EXPECTED_CHUNKS=109163" in text
    assert "EXPECTED_EMBEDDINGS=109163" in text
    assert "EXPECTED_SOURCES=60" in text
    assert "filebin.net/arkguru-complete" in text
    assert "--force-remote" in text
    assert "GRANT ALL ON ALL TABLES IN SCHEMA public" in text


def test_windows_restore_pins_complete_dump():
    text = (_SCRIPTS / "restore_rag_dump.ps1").read_text()
    assert _SHA in text
    assert "ExpectedChunks = 109163" in text
    assert "ExpectedEmbeddings = 109163" in text
    assert "ExpectedSources = 60" in text
    assert "restore_rag_dump.cmd" in text
    assert "GRANT ALL ON ALL TABLES IN SCHEMA public" in text
    assert "Get-PortableDump" in text
    assert "curl.exe" in text
    assert "user-agent" in text
    assert "6ba1004fd99133af779ba7ce1c66465a0cbc2834b755145c4a8e088250c872db" in text
    assert "Install-PgvectorWindows" in text
    assert "vector.v0.8.6-pg16.zip" in text
    assert "vector.control" in text


def test_windows_cmd_launcher_bypasses_execution_policy():
    text = (_SCRIPTS / "restore_rag_dump.cmd").read_text()
    assert "ExecutionPolicy Bypass" in text
    assert "restore_rag_dump.ps1" in text
    assert "raw.githubusercontent.com/ravidsun/arkguru-pdf-converter" in text
    assert "URL_MAIN" in text
    assert "URL_BRANCH" in text
    assert "restore_rag_dump.ps1 missing; downloading" in text
    assert "user-agent curl/8.5.0" in text
