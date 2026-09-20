"""
Phase 2 orchestrator: seeds -> HTML chunks (+ optional PDF harvest).

Usage:
    python -m phase2_web.pipeline --seeds https://example.com --max-pages 5
    python -m phase2_web.pipeline --config config/config.yaml --sink postgres

The postgres sink **refuses** schema ``public``. Web harvest belongs in
``web.chunks`` (see config/datastore.yaml ``postgres.schema: web``).
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common.schema import Chunk, write_jsonl, write_parquet

from .chunk import chunk_page
from .dedup import near_dedup
from .extract import extract_page
from .fetch import CrawlResult, crawl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("phase2.pipeline")


@dataclass
class Phase2Config:
    seeds: list[str] = field(default_factory=list)
    out_dir: str = "data/processed"
    out_format: str = "jsonl"
    max_pages: int = 320
    max_pages_per_seed: int = 80
    delay_seconds: float = 0.75
    same_domain_only: bool = True
    backend: str = "local"          # local | firecrawl
    target_tokens: int = 550
    overlap_pct: float = 0.15
    min_tokens: int = 80
    min_content_chars: int = 200
    dedup: bool = True
    sink: str = "file"              # file | postgres
    datastore_config: str = "config/datastore.yaml"
    download_pdfs: bool = True
    ingest_pdfs: bool = True
    pdf_dir: str = "data/raw_pdfs"


def _load_config(path: str) -> Phase2Config:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    p2 = (raw.get("phase2") or {})
    return Phase2Config(**{k: v for k, v in p2.items()
                           if k in Phase2Config.__dataclass_fields__})


def _require_web_schema(store) -> None:
    """Refuse to write Phase 2 rows into the book corpus (public.chunks)."""
    if getattr(store, "is_public_schema", True):
        raise SystemExit(
            "Phase 2 postgres sink refuses schema 'public'. "
            "Set postgres.schema: web in config/datastore.yaml so harvest "
            "lands in web.chunks / web.search_chunks, not public.chunks."
        )


def open_web_store(datastore_config: str):
    from common.datastore_config import open_chunk_store
    store = open_chunk_store(datastore_config)
    _require_web_schema(store)
    return store


def pages_to_chunks(result: CrawlResult, cfg: Phase2Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in result.pages:
        text, title = extract_page(page)
        chunks.extend(chunk_page(
            text, page.url, title=title,
            target_tokens=cfg.target_tokens,
            overlap_pct=cfg.overlap_pct,
            min_tokens=cfg.min_tokens,
            min_content_chars=cfg.min_content_chars,
        ))
    if cfg.dedup:
        before = len(chunks)
        chunks = near_dedup(chunks)
        log.info("dedup %d -> %d chunks", before, len(chunks))
    return chunks


def write_file_sink(chunks: list[Chunk], cfg: Phase2Config) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("web_chunks.parquet" if cfg.out_format == "parquet"
                      else "web_chunks.jsonl")
    if cfg.out_format == "parquet":
        write_parquet(chunks, path)
    else:
        write_jsonl(chunks, path)
    log.info("wrote %d chunks -> %s", len(chunks), path)
    return path


def upsert_web(chunks: list[Chunk], store) -> int:
    """Delete-then-reingest per URL so chunk_index shifts cannot corrupt."""
    seen: set[str] = set()
    n = 0
    by_url: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_url.setdefault(c.source_id, []).append(c)
    for source_id, rows in by_url.items():
        if source_id not in seen:
            store.delete_by_source_id(source_id)
            seen.add(source_id)
        n += store.upsert(rows)
    return n


def harvest_pdfs(result: CrawlResult, cfg: Phase2Config, store=None) -> int:
    if not cfg.download_pdfs or not result.pdf_urls:
        return 0
    from .pdfs import download_pdfs, ingest_pdfs_with_phase1
    saved = download_pdfs(result.pdf_urls, cfg.pdf_dir)
    log.info("downloaded %d pdf(s) -> %s", len(saved), cfg.pdf_dir)
    if not cfg.ingest_pdfs or not saved:
        return 0
    if cfg.sink != "postgres":
        log.info("pdf ingest skipped (sink=%s); files are in %s",
                 cfg.sink, cfg.pdf_dir)
        return 0
    if store is None:
        store = open_web_store(cfg.datastore_config)
        store.ensure_schema()
    n = ingest_pdfs_with_phase1(cfg.pdf_dir, cfg.datastore_config, sink="postgres")
    log.info("phase1 ingested %d pdf-derived chunk(s) into %s", n, store.chunks)
    return n


def run(cfg: Phase2Config) -> list[Chunk]:
    if not cfg.seeds:
        log.warning("No seeds configured")
        return []
    store = None
    if cfg.sink == "postgres":
        store = open_web_store(cfg.datastore_config)
        store.ensure_schema()
        log.info("postgres sink %s / %s", store.schema, store.chunks)

    result = crawl(
        cfg.seeds,
        max_pages=cfg.max_pages,
        max_pages_per_seed=cfg.max_pages_per_seed,
        delay_seconds=cfg.delay_seconds,
        same_domain_only=cfg.same_domain_only,
    )
    chunks = pages_to_chunks(result, cfg)
    write_file_sink(chunks, cfg)
    if cfg.sink == "postgres":
        upserted = upsert_web(chunks, store)
        log.info("upserted %d web chunks into %s", upserted, store.chunks)

    harvest_pdfs(result, cfg, store=store)
    return chunks


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2: URLs -> web chunks")
    ap.add_argument("--config", help="YAML config")
    ap.add_argument("--seeds", nargs="*", help="seed URLs")
    ap.add_argument("--max-pages", dest="max_pages", type=int)
    ap.add_argument("--max-pages-per-seed", dest="max_pages_per_seed", type=int)
    ap.add_argument("--delay", dest="delay_seconds", type=float)
    ap.add_argument("--sink", choices=["file", "postgres"])
    ap.add_argument("--datastore-config", dest="datastore_config")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--no-pdfs", dest="download_pdfs", action="store_false",
                    default=None)
    ap.add_argument("--no-ingest-pdfs", dest="ingest_pdfs", action="store_false",
                    default=None)
    ap.add_argument("--init-db", dest="init_db", action="store_true",
                    help="create web schema/tables then exit")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config) if args.config else Phase2Config()
    for k, v in vars(args).items():
        if k not in ("config", "init_db") and v is not None:
            setattr(cfg, k, v)

    if args.init_db:
        store = open_web_store(cfg.datastore_config)
        store.ensure_schema()
        log.info("datastore ready schema=%s tables '%s', '%s'",
                 store.schema, store.chunks, store.vectors)
        return 0

    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
