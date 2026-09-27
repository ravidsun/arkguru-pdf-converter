"""Force tiktoken in Phase 2 tests so CI never touches Hugging Face."""
from __future__ import annotations

import pytest

from common.tokenizer import reset_tokenizer_cache


@pytest.fixture(autouse=True)
def _tokenizer_tiktoken(monkeypatch):
    monkeypatch.setenv("ARKGURU_TOKENIZER", "tiktoken")
    reset_tokenizer_cache()
    yield
    reset_tokenizer_cache()
