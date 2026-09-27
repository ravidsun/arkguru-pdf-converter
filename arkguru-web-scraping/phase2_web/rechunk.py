"""Re-chunk existing ``web_chunks.csv`` through ``clean_text`` and the current chunker.

The committed CSV was imported as-is (including a cafeastrology site-map page
and empty ``lang``). Never re-import those rows unchanged: stitch per URL,
run ``clean_text`` + ``chunk_page``, then ``annotate_chunks`` (quality + lang).

Site-map / tag / category / pagination / archive URLs yield zero chunks.
Web rows stay ``source_type='web'`` for the shared ``chunks`` table.

Usage:
    python -m phase2_web.rechunk --input web_chunks.csv --out-dir data/processed
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from common.quality import annotate_chunks
from common.schema import Chunk

from .chunk import chunk_page
from .fetch import ingest_path_denied
from .pipeline import Phase2Config, write_file_sink

log = logging.getLogger("phase2.rechunk")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def stitch_chunk_texts(texts: list[str]) -> str:
    """Join ordered window texts, dropping a duplicated overlap prefix."""
    kept: list[str] = []
    for raw in texts:
        piece = (raw or "").strip()
        if not piece:
            continue
        if not kept:
            kept.append(piece)
            continue
        prev = kept[-1]
        max_overlap = min(len(prev), len(piece), 400)
        cut = 0
        for n in range(max_overlap, 24, -1):
            if piece.startswith(prev[-n:]):
                cut = n
                break
        kept.append(piece[cut:].lstrip() if cut else piece)
    return "\n\n".join(kept)


def _rows_by_source(path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row.get("source_id") or row.get("url") or "").strip()
            if not key:
                continue
            groups[key].append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r.get("chunk_index") or 0))
    return groups


def rechunk_web_csv(
    path: str | Path,
    *,
    cfg: Optional[Phase2Config] = None,
    quality_cfg: Optional[dict] = None,
) -> list[Chunk]:
    """Read a web_chunks.csv and emit freshly chunked ``source_type='web'`` rows."""
    cfg = cfg or Phase2Config()
    src = Path(path)
    groups = _rows_by_source(src)
    chunks: list[Chunk] = []
    skipped = 0
    for url, rows in groups.items():
        if ingest_path_denied(url) or any(
            ingest_path_denied(r.get("url") or "") for r in rows
        ):
            skipped += 1
            continue
        title = next((r.get("title") for r in rows if r.get("title")), None)
        body = stitch_chunk_texts([r.get("text") or "" for r in rows])
        chunks.extend(chunk_page(
            body, url, title=title,
            target_tokens=cfg.target_tokens,
            overlap_pct=cfg.overlap_pct,
            min_tokens=cfg.min_tokens,
            min_content_chars=cfg.min_content_chars,
            max_tokens=cfg.max_tokens,
            min_chunk_chars=cfg.min_chunk_chars,
        ))
    log.info(
        "rechunk %s: %d source(s), skipped_ingest_deny=%d, chunks=%d",
        src, len(groups), skipped, len(chunks),
    )
    return annotate_chunks(chunks, quality_cfg=quality_cfg or cfg.quality)


def upsert_web_same_table(chunks: Iterable[Chunk], store) -> int:
    """Delete+upsert per URL into whatever schema the store is pointed at."""
    by_url: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_url.setdefault(c.source_id, []).append(c)
    n = 0
    for source_id, rows in by_url.items():
        n += store.replace_source(source_id, rows)
    return n


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Re-chunk web_chunks.csv via clean_text + the current chunker")
    ap.add_argument("--input", default="web_chunks.csv",
                    help="Path to the committed (or exported) web_chunks.csv")
    ap.add_argument("--config", help="Phase 2 YAML (token / quality settings)")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--format", dest="out_format", choices=["jsonl", "parquet"])
    ap.add_argument("--sink", choices=["file", "postgres"], default="file")
    ap.add_argument("--datastore-config", dest="datastore_config")
    args = ap.parse_args(argv)

    if args.config:
        from .pipeline import _load_config
        cfg = _load_config(args.config)
    else:
        cfg = Phase2Config()
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.out_format:
        cfg.out_format = args.out_format
    if args.datastore_config:
        cfg.datastore_config = args.datastore_config

    chunks = rechunk_web_csv(args.input, cfg=cfg)
    write_file_sink(chunks, cfg)
    if args.sink == "postgres":
        from common.datastore_config import open_chunk_store
        store = open_chunk_store(cfg.datastore_config)
        store.ensure_schema()
        n = upsert_web_same_table(chunks, store)
        log.info("upserted %d re-chunked web row(s) into %s", n, store.chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
