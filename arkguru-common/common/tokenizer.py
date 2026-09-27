"""
Token counting used for chunk sizing against bge-m3 (XLM-R, max 512).

Default resolution (``ARKGURU_TOKENIZER`` unset or ``auto``):

1. ``BAAI/bge-m3`` tokenizer files via ``transformers``, ``local_files_only``
   (tokenizer only — never the embedding model, never a network download).
2. tiktoken ``cl100k_base``, with a one-time warning that counts can differ
   from XLM-R (especially Devanagari / diacritics).
3. A ~4 chars/token heuristic if tiktoken is missing too.

``ARKGURU_TOKENIZER`` overrides: a Hugging Face id, or ``tiktoken`` /
``cl100k_base`` to force the proxy (tests/CI). Tests should set that or
use a stub so CI never downloads.

The default *body* cap is the model window minus special tokens
(CLS/SEP → 510) so the embedder does not truncate.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

log = logging.getLogger("common.tokenizer")

DEFAULT_HF_TOKENIZER = "BAAI/bge-m3"
MODEL_MAX_SEQ_LENGTH = 512
DEFAULT_SPECIAL_TOKEN_RESERVE = 2
DEFAULT_TARGET_TOKENS = 400
# Body cap after reserving XLM-R specials. pack_windows still subtracts
# specials when the caller passes the raw model window (512).
DEFAULT_MAX_TOKENS = MODEL_MAX_SEQ_LENGTH - DEFAULT_SPECIAL_TOKEN_RESERVE

_TIKTOKEN_ALIASES = frozenset({"tiktoken", "cl100k_base", "proxy"})
_AUTO_ALIASES = frozenset({"", "auto"})

_fallback_warned = False


def tokenizer_name() -> str:
    return (os.environ.get("ARKGURU_TOKENIZER") or "auto").strip() or "auto"


def reset_tokenizer_cache() -> None:
    """Drop cached backend / HF objects. Used by tests after env changes."""
    global _fallback_warned
    _fallback_warned = False
    resolve_tokenizer.cache_clear()
    _hf_tokenizer.cache_clear()
    _tiktoken_encoder.cache_clear()


def _load_hf_tokenizer(name: str):
    """Load tokenizer *files* only. Never downloads, never loads the model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, local_files_only=True)


@lru_cache(maxsize=4)
def _hf_tokenizer(name: str):
    return _load_hf_tokenizer(name)


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _warn_tiktoken_fallback(reason: str) -> None:
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    log.warning(
        "Chunk sizing fell back to tiktoken cl100k_base (%s). "
        "cl100k counts differ from BAAI/bge-m3 (XLM-R SentencePiece), "
        "especially on Devanagari and diacritic-heavy text, so a chunk "
        "at the tiktoken cap may still be truncated at embed time. "
        "Install transformers and cache the %s tokenizer files "
        "(no model download needed). Set ARKGURU_TOKENIZER=tiktoken to "
        "force the proxy (tests/CI).",
        reason, DEFAULT_HF_TOKENIZER,
    )


@lru_cache(maxsize=1)
def resolve_tokenizer() -> tuple[str, str]:
    """Return ``(backend, name)``.

    ``backend`` is ``hf``, ``tiktoken``, or ``char``.
    """
    requested = tokenizer_name()
    key = requested.lower()
    if key in _TIKTOKEN_ALIASES:
        if _tiktoken_encoder() is not None:
            return "tiktoken", "cl100k_base"
        return "char", "char"

    hf_name = DEFAULT_HF_TOKENIZER if key in _AUTO_ALIASES else requested
    try:
        _hf_tokenizer(hf_name)
        return "hf", hf_name
    except Exception as exc:
        if key in _AUTO_ALIASES:
            _warn_tiktoken_fallback(f"{DEFAULT_HF_TOKENIZER} unavailable: {exc}")
        else:
            _warn_tiktoken_fallback(
                f"ARKGURU_TOKENIZER={requested} unavailable: {exc}"
            )
        if _tiktoken_encoder() is not None:
            return "tiktoken", "cl100k_base"
        return "char", "char"


def special_token_reserve() -> int:
    """Tokens the embedder adds around the chunk body (CLS/SEP for XLM-R)."""
    backend, name = resolve_tokenizer()
    if backend == "hf":
        try:
            tok = _hf_tokenizer(name)
            n = tok.num_special_tokens_to_add(pair=False)
            return max(0, int(n))
        except Exception:
            pass
    return DEFAULT_SPECIAL_TOKEN_RESERVE


def effective_max_tokens(max_tokens: int | None = None) -> int:
    """Body-token cap. Subtract specials when using the model window (512)."""
    if max_tokens is None:
        return max(1, MODEL_MAX_SEQ_LENGTH - special_token_reserve())
    cap = max(1, int(max_tokens))
    if cap >= MODEL_MAX_SEQ_LENGTH:
        return max(1, cap - special_token_reserve())
    return cap


def count_tokens(text: str) -> int:
    if not text:
        return 0
    backend, name = resolve_tokenizer()
    if backend == "hf":
        tok = _hf_tokenizer(name)
        return len(tok.encode(text, add_special_tokens=False))
    if backend == "tiktoken":
        enc = _tiktoken_encoder()
        if enc is not None:
            return len(enc.encode(text))
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    pieces = split_to_max_tokens(text, max_tokens, overlap_tokens=0)
    return pieces[0] if pieces else ""


def split_to_max_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """Hard-split ``text`` so every piece is at most ``max_tokens``.

    Uses the active encoder when available; otherwise a character budget.
    ``overlap_tokens`` steps the window back so adjacent pieces share a tail.
    ``max_tokens`` is the body cap (already special-reserved by the caller).
    """
    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    n = count_tokens(text)
    if n <= max_tokens:
        return [text]

    overlap = max(0, min(overlap_tokens, max_tokens - 1))
    step = max(1, max_tokens - overlap)

    backend, name = resolve_tokenizer()
    if backend == "hf":
        try:
            tok = _hf_tokenizer(name)
            ids = tok.encode(text, add_special_tokens=False)
            return _decode_id_windows(ids, max_tokens, step, tok.decode)
        except Exception:
            pass
    enc = _tiktoken_encoder()
    if enc is not None:
        ids = enc.encode(text)
        return _decode_id_windows(ids, max_tokens, step, enc.decode)

    width = max(1, max_tokens * 4)
    ov_chars = overlap * 4
    step_chars = max(1, width - ov_chars)
    out: list[str] = []
    i = 0
    while i < len(text):
        piece = text[i:i + width]
        if piece:
            out.append(piece)
        if i + width >= len(text):
            break
        i += step_chars
    return out


def _decode_id_windows(ids, max_tokens: int, step: int, decode) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(ids)
    while i < n:
        piece = decode(ids[i:i + max_tokens])
        if piece:
            out.append(piece)
        if i + max_tokens >= n:
            break
        i += step
    return out
