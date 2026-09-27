"""
Token counting used for chunk sizing.

bge-m3 (XLM-R) has ``max_seq_length`` 512. When ``ARKGURU_TOKENIZER`` is set
to a Hugging Face id (e.g. ``BAAI/bge-m3``) and ``transformers`` is
installed, that tokenizer is used. Otherwise we keep tiktoken
``cl100k_base`` — a fast proxy that is **not** equivalent to XLM-R,
especially on Devanagari. Production ingest that must match the embedder
should set ``ARKGURU_TOKENIZER=BAAI/bge-m3``.
"""
from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_TARGET_TOKENS = 400
DEFAULT_MAX_TOKENS = 512  # bge-m3 max_seq_length


@lru_cache(maxsize=1)
def tokenizer_name() -> str:
    return (os.environ.get("ARKGURU_TOKENIZER") or "tiktoken").strip()


@lru_cache(maxsize=2)
def _hf_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _hf_name() -> str | None:
    name = tokenizer_name()
    if not name or name.lower() in {"tiktoken", "cl100k_base", "auto", "proxy"}:
        return None
    return name


def count_tokens(text: str) -> int:
    if not text:
        return 0
    hf = _hf_name()
    if hf:
        try:
            tok = _hf_tokenizer(hf)
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    enc = _tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Fallback heuristic: ~4 chars/token for English prose.
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

    hf = _hf_name()
    if hf:
        try:
            tok = _hf_tokenizer(hf)
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
