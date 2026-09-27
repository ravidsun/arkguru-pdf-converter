"""Phase 3 test hooks."""
from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "pgvector: needs a local Postgres 16 + pgvector "
        "(skipped without a loopback PG_DSN)",
    )


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
