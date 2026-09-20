"""Page text -> ``Chunk`` records with ``source_type='web'``."""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from common.chunking import pack_windows, split_sentences
from common.schema import Chunk
from common.tokenizer import count_tokens


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
) -> list[Chunk]:
    body = (text or "").strip()
    if len(body) < min_content_chars:
        return []
    overlap_tokens = max(1, int(target_tokens * overlap_pct))
    sentences = split_sentences(body) or [body]
    windows = pack_windows(
        sentences, target_tokens=target_tokens,
        overlap_tokens=overlap_tokens, min_tokens=min_tokens,
    )
    domain = domain_of(url)
    chunks: list[Chunk] = []
    for i, (window, ov) in enumerate(windows):
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
