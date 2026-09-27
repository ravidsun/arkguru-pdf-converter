"""Keep suite token counts on tiktoken unless a test opts into auto-resolve."""
from __future__ import annotations

import pytest

from common.tokenizer import reset_tokenizer_cache


@pytest.fixture(autouse=True)
def _tokenizer_backend(request, monkeypatch):
    if request.node.get_closest_marker("tokenizer_autoresolve"):
        monkeypatch.delenv("ARKGURU_TOKENIZER", raising=False)
    else:
        monkeypatch.setenv("ARKGURU_TOKENIZER", "tiktoken")
    reset_tokenizer_cache()
    yield
    reset_tokenizer_cache()
