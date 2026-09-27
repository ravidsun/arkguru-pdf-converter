"""Page text -> ``Chunk`` records with ``source_type='web'``."""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from common.chunking import pack_windows, split_for_packing
from common.schema import Chunk
from common.text import body_without_heading, clean_text
from common.tokenizer import DEFAULT_MAX_TOKENS, count_tokens


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def chunk_page(
    text: str,
    url: str,
    *,
    title: Optional[str] = None,
    target_tokens: int = 550,
    overlap_pct: float = 0.15,
    min_tokens: int = 80,
    min_content_chars: int = 200,
    lang: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_chunk_chars: int = 80,
) -> list[Chunk]:
    body = clean_text(text)
    if len(body) < min_content_chars:
        return []
    overlap_tokens = max(1, int(target_tokens * overlap_pct))
    units = split_for_packing(body) or [body]
    windows = pack_windows(
        units, target_tokens=target_tokens,
        overlap_tokens=overlap_tokens, min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    domain = domain_of(url)
    heading = (title or "").strip() or None
    kept: list[tuple[str, int]] = []
    for window, ov in windows:
        if min_chunk_chars > 0:
            rest = body_without_heading(window, heading)
            if not rest or len(rest) < min_chunk_chars:
                continue
        kept.append((window, ov))
    chunks: list[Chunk] = []
    for i, (window, ov) in enumerate(kept):
        chunks.append(Chunk(
            text=window,
            source_type="web",
            source_id=url,
            chunk_index=i,
            title=title,
            url=url,
            domain=domain,
            lang=lang,
            token_count=count_tokens(window),
            overlap_tokens=ov,
            extra={"block_type": "web"},
        ))
    return chunks
