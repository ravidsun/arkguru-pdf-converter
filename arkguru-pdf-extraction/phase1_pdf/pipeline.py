"""
Phase 1 orchestrator: a folder of PDFs -> per-PDF folders of chunk files.

Usage:
    python -m phase1_pdf.pipeline --input data/raw_pdfs --out-dir data/processed
    python -m phase1_pdf.pipeline --config config/config.yaml

The file sink writes exclusive JSONL (or Parquet) splits under
``out_dir / {relative_stem}/`` (PDF path relative to ``--input``, without .pdf):

    chunks.jsonl    prose children
    parents.jsonl   parent_child parents (omitted if empty)
    tables.jsonl    extra.block_type == table (omitted if empty)
    figures.jsonl   extra.block_type == figure (omitted if empty)

Postgres still upserts every chunk. Records use the shared ``Chunk`` schema.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from common.schema import Chunk, write_jsonl, write_parquet
from .extract import extract_document
from .chunk import chunk_document, reindex_chunks

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("phase1.pipeline")


@dataclass
class Phase1Config:
    input_dir: str = "data/raw_pdfs"
    out_dir: str = "data/processed"
    out_format: str = "jsonl"          # "jsonl" | "parquet"
    backend: str = "pymupdf4llm"       # pymupdf4llm | docling | pymupdf
    strategy: str = "structure"        # structure | parent_child | semantic
    target_tokens: int = 400
    overlap_pct: float = 0.15
    min_tokens: int = 80
    parent_max_tokens: int = 2000
    ocr_enabled: bool = True
    extract_tables: bool = True
    extract_figures: bool = True
    dedup: bool = True
    # --- scaling ---
    workers: int = 1                   # 0 or <0 => auto (cpu_count-1)
    # --- datastore sink ---
    sink: str = "file"                 # file | postgres
    datastore_config: str = "config/datastore.yaml"   # all DB/table settings live here


def _load_config(path: str) -> Phase1Config:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    p1 = (raw.get("phase1") or {})
    return Phase1Config(**{k: v for k, v in p1.items()
                           if k in Phase1Config.__dataclass_fields__})


def _dedup(chunks: list[Chunk]) -> list[Chunk]:
    """Drop exact-duplicate child text (keeps first). Parents always kept."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.is_parent:
            out.append(c)
            continue
        key = hashlib.md5(c.text.strip().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _resolve_workers(n: int) -> int:
    if n and n > 0:
        return n
    import os
    return max(1, (os.cpu_count() or 2) - 1)


def relative_source_id(pdf: Path, input_dir: Path) -> str:
    """Path of ``pdf`` relative to the ingest folder, POSIX separators, with suffix.

    Two manuals that share a basename (``hvac/manual.pdf`` vs ``elec/manual.pdf``)
    stay distinct. Falls back to the basename if ``pdf`` is outside ``input_dir``.
    """
    try:
        rel = pdf.resolve().relative_to(input_dir.resolve())
    except ValueError:
        rel = Path(pdf.name)
    return rel.as_posix()


def relative_stem(source_id: str) -> str:
    """``hvac/manual.pdf`` -> ``hvac/manual`` (POSIX)."""
    return Path(source_id).with_suffix("").as_posix()


def partition_file_sink_chunks(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Exclusive splits for the per-PDF folder sink. Keys: chunks, parents, tables, figures."""
    prose: list[Chunk] = []
    parents: list[Chunk] = []
    tables: list[Chunk] = []
    figures: list[Chunk] = []
    for c in chunks:
        kind = (c.extra or {}).get("block_type")
        if c.is_parent:
            parents.append(c)
        elif kind == "table":
            tables.append(c)
        elif kind == "figure":
            figures.append(c)
        else:
            prose.append(c)
    return {
        "chunks": prose,
        "parents": parents,
        "tables": tables,
        "figures": figures,
    }


def write_file_sink(
    chunks: list[Chunk],
    out_dir: Path,
    stem: str,
    out_format: str = "jsonl",
) -> Path:
    """Write non-empty exclusive split files under ``out_dir / stem``."""
    folder = out_dir / Path(stem)
    parts = partition_file_sink_chunks(chunks)
    ext = ".parquet" if out_format == "parquet" else ".jsonl"
    writer = write_parquet if out_format == "parquet" else write_jsonl
    for name, rows in parts.items():
        if not rows:
            continue
        writer(rows, folder / f"{name}{ext}")
    return folder


def _process_one(args) -> tuple[str, str, list[Chunk]]:
    """Top-level (picklable) worker: one PDF -> (name, relative_stem, chunks)."""
    pdf_str, cfg = args
    pdf = Path(pdf_str)
    src_id = relative_source_id(pdf, Path(cfg.input_dir))
    try:
        doc = extract_document(
            pdf, backend=cfg.backend, ocr_enabled=cfg.ocr_enabled,
            extract_tables=cfg.extract_tables, extract_figures=cfg.extract_figures,
            source_id=src_id,
        )
        chunks = chunk_document(
            doc, strategy=cfg.strategy,
            target_tokens=cfg.target_tokens, overlap_pct=cfg.overlap_pct,
            parent_max_tokens=cfg.parent_max_tokens, min_tokens=cfg.min_tokens,
        )
        if cfg.dedup:
            chunks = _dedup(chunks)
        chunks = reindex_chunks(chunks)
        return pdf.name, relative_stem(src_id), chunks
    except Exception as e:
        log.exception("  FAILED %s: %s", pdf.name, e)
        return pdf.name, relative_stem(src_id), []


def run(cfg: Phase1Config, pdfs: Optional[list] = None) -> list[Chunk]:
    in_dir = Path(cfg.input_dir)
    if pdfs is None:
        pdfs = sorted(in_dir.glob("**/*.pdf"))
    else:
        pdfs = [Path(p) for p in pdfs]
    if not pdfs:
        log.warning("No PDFs to process under %s", in_dir.resolve())
        return []

    out_dir = Path(cfg.out_dir)
    workers = _resolve_workers(cfg.workers)

    store = None
    if cfg.sink == "postgres":
        from common.datastore_config import open_chunk_store
        store = open_chunk_store(cfg.datastore_config)
        store.ensure_schema()

    all_chunks: list[Chunk] = []
    upserted = 0

    def handle(name: str, stem: str, chunks: list[Chunk]) -> None:
        nonlocal upserted
        if cfg.sink == "postgres":
            upserted += store.upsert(chunks)
            try:
                counts = store.chunk_index_counts(name)
            except Exception:
                counts = []
            log.info("  %s -> %d chunks -> datastore chunk_index=%s",
                     name, len(chunks), counts)
        else:
            folder = write_file_sink(chunks, out_dir, stem, cfg.out_format)
            log.info("  %s -> %d chunks -> %s", name, len(chunks), folder)
        all_chunks.extend(chunks)

    log.info("Processing %d PDF(s) with %d worker(s), sink=%s",
             len(pdfs), workers, cfg.sink)
    tasks = [(str(p), cfg) for p in pdfs]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for name, stem, chunks in pool.imap_unordered(_process_one, tasks):
                handle(name, stem, chunks)
    else:
        for t in tasks:
            handle(*_process_one(t))

    if cfg.sink == "postgres":
        log.info("Upserted %d chunks into the pgvector datastore", upserted)
    else:
        log.info("Wrote %d PDF(s) -> %d total chunks in %s",
                 len(pdfs), len(all_chunks), out_dir.resolve())
    _print_stats(all_chunks)
    return all_chunks


def _print_stats(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    child = [c for c in chunks if not c.is_parent]
    toks = [c.token_count or 0 for c in child]
    toks.sort()
    n = len(toks)
    def pct(p): return toks[min(n - 1, int(p * n))] if n else 0
    idxs = [c.chunk_index for c in child]
    missing = sum(1 for i in idxs if i is None)
    log.info("Stats: %d child chunks | tokens p10=%d p50=%d p90=%d | %d parents "
             "| chunk_index min=%s max=%s nulls=%d",
             n, pct(0.1), pct(0.5), pct(0.9),
             sum(1 for c in chunks if c.is_parent),
             min(idxs) if idxs else None,
             max(idxs) if idxs else None,
             missing)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1: PDF -> structured chunks")
    ap.add_argument("--config", help="YAML config; overrides other flags")
    ap.add_argument("--input", dest="input_dir")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--format", dest="out_format", choices=["jsonl", "parquet"])
    ap.add_argument("--backend", choices=["pymupdf4llm", "docling", "pymupdf"])
    ap.add_argument("--strategy", choices=["structure", "parent_child", "semantic"])
    ap.add_argument("--target-tokens", dest="target_tokens", type=int)
    ap.add_argument("--overlap-pct", dest="overlap_pct", type=float)
    ap.add_argument("--min-tokens", dest="min_tokens", type=int)
    ap.add_argument("--no-ocr", dest="ocr_enabled", action="store_false", default=None)
    ap.add_argument("--no-tables", dest="extract_tables", action="store_false", default=None)
    ap.add_argument("--no-figures", dest="extract_figures", action="store_false", default=None)
    ap.add_argument("--workers", type=int, help="parallel worker processes (0=auto)")
    ap.add_argument("--sink", choices=["file", "postgres"])
    ap.add_argument("--datastore-config", dest="datastore_config",
                    help="path to datastore.yaml (default: config/datastore.yaml). "
                         "Phase 2 PDF harvest points this at schema: web.")
    ap.add_argument("--init-db", dest="init_db", action="store_true",
                    help="create datastore tables (chunks, chunk_embeddings) if "
                         "missing, then exit")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config) if args.config else Phase1Config()
    for k, v in vars(args).items():
        if k not in ("config", "init_db") and v is not None:
            setattr(cfg, k, v)

    if args.init_db:
        from common.datastore_config import open_chunk_store
        store = open_chunk_store(cfg.datastore_config)
        store.ensure_schema()
        log.info("datastore ready (tables '%s', '%s')", store.chunks, store.vectors)
        return 0

    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
