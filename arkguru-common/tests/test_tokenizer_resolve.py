"""Default tokenizer resolution: bge-m3 when available, tiktoken fallback."""
from __future__ import annotations

import logging

import pytest

from common.chunking import pack_windows
from common.tokenizer import (
    DEFAULT_HF_TOKENIZER,
    DEFAULT_MAX_TOKENS,
    MODEL_MAX_SEQ_LENGTH,
    count_tokens,
    effective_max_tokens,
    reset_tokenizer_cache,
    resolve_tokenizer,
    special_token_reserve,
)


class _FakeBgeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # One id per whitespace-separated token; ignores specials (body count).
        return [1] * max(1, len(str(text).split()))

    def decode(self, ids):
        return " ".join("tok" for _ in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


@pytest.mark.tokenizer_autoresolve
def test_default_resolution_prefers_bge_m3_when_available(monkeypatch):
    monkeypatch.delenv("ARKGURU_TOKENIZER", raising=False)
    fake = _FakeBgeTokenizer()
    monkeypatch.setattr("common.tokenizer._load_hf_tokenizer", lambda name: fake)
    reset_tokenizer_cache()

    backend, name = resolve_tokenizer()
    assert backend == "hf"
    assert name == DEFAULT_HF_TOKENIZER
    assert count_tokens("one two three") == 3
    assert special_token_reserve() == 2
    assert effective_max_tokens() == MODEL_MAX_SEQ_LENGTH - 2
    assert effective_max_tokens(MODEL_MAX_SEQ_LENGTH) == 510


@pytest.mark.tokenizer_autoresolve
def test_default_resolution_falls_back_to_tiktoken(monkeypatch, caplog):
    monkeypatch.delenv("ARKGURU_TOKENIZER", raising=False)

    def _missing(_name: str):
        raise OSError("local_files_only: BAAI/bge-m3 not in cache")

    monkeypatch.setattr("common.tokenizer._load_hf_tokenizer", _missing)
    reset_tokenizer_cache()

    with caplog.at_level(logging.WARNING, logger="common.tokenizer"):
        backend, name = resolve_tokenizer()
        count_tokens("hello world")

    assert backend == "tiktoken"
    assert name == "cl100k_base"
    assert "tiktoken" in caplog.text.lower()
    assert "bge-m3" in caplog.text.lower() or "xlm" in caplog.text.lower()
    # One-time: a second count does not emit another warning.
    n_warn = sum(1 for r in caplog.records if r.levelno >= logging.WARNING)
    count_tokens("again")
    n_warn_after = sum(1 for r in caplog.records if r.levelno >= logging.WARNING)
    assert n_warn_after == n_warn


def test_arkguru_tokenizer_tiktoken_skips_hf(monkeypatch):
    monkeypatch.setenv("ARKGURU_TOKENIZER", "tiktoken")
    called = []

    def _boom(name: str):
        called.append(name)
        raise AssertionError("HF must not be loaded when tiktoken is forced")

    monkeypatch.setattr("common.tokenizer._load_hf_tokenizer", _boom)
    reset_tokenizer_cache()
    backend, name = resolve_tokenizer()
    assert backend == "tiktoken"
    assert name == "cl100k_base"
    assert called == []
    assert count_tokens("hello world") >= 1


@pytest.mark.tokenizer_autoresolve
def test_explicit_hf_override_uses_that_id(monkeypatch):
    monkeypatch.setenv("ARKGURU_TOKENIZER", "org/custom-xlm")
    fake = _FakeBgeTokenizer()
    seen: list[str] = []

    def _load(name: str):
        seen.append(name)
        return fake

    monkeypatch.setattr("common.tokenizer._load_hf_tokenizer", _load)
    reset_tokenizer_cache()
    backend, name = resolve_tokenizer()
    assert backend == "hf"
    assert name == "org/custom-xlm"
    assert seen == ["org/custom-xlm"]


def test_effective_max_tokens_reserves_specials_only_on_model_window():
    assert DEFAULT_MAX_TOKENS == 510
    assert effective_max_tokens(64) == 64
    assert effective_max_tokens(MODEL_MAX_SEQ_LENGTH) == 510
    assert effective_max_tokens(None) == 510


def test_pack_windows_treats_512_as_model_window():
    sentences = ["word " * 40 for _ in range(30)]
    windows = pack_windows(
        sentences, target_tokens=400, overlap_tokens=0, max_tokens=512,
    )
    assert windows
    assert all(count_tokens(text) <= 510 for text, _ov in windows)
