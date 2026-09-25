"""Phase 1 CLI accepts --datastore-config for the schema-web PDF harvest."""
from __future__ import annotations

from phase1_pdf.pipeline import main as phase1_main


def test_datastore_config_flag_is_accepted(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cfg):
        seen["path"] = cfg.datastore_config
        return []

    monkeypatch.setattr("phase1_pdf.pipeline.run", fake_run)
    rc = phase1_main(["--input", str(tmp_path), "--sink", "file",
                      "--datastore-config", "/tmp/web-datastore.yaml"])
    assert rc == 0
    assert seen["path"] == "/tmp/web-datastore.yaml"
