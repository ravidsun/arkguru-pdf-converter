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

Postgres uses ``replace_source`` (delete + upsert in one transaction).
Records use the shared ``Chunk`` schema. Duplicate file sha256s are skipped
and recorded on the canonical source as ``meta.also_in``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common.quality import annotate_chunks
from common.schema import Chunk, write_jsonl, write_parquet
from common.tokenizer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TARGET_TOKENS,
    effective_max_tokens,
)
from .extract import (
    DEFAULT_FORCE_OCR,
    DEFAULT_OCR_LANGUAGES,
    _file_sha256,
    extract_document,
    should_force_ocr,
)
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
    strategy: str = "parent_child"     # parent_child | structure | semantic
    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_pct: float = 0.15
    min_tokens: int = 80
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_chunk_chars: int = 80
    min_table_chars: int = 40
    min_figure_chars: int = 40
    parent_max_tokens: int = 2000
    ocr_enabled: bool = True
    ocr_languages: str = DEFAULT_OCR_LANGUAGES
    force_ocr: list = field(default_factory=lambda: list(DEFAULT_FORCE_OCR))
    extract_tables: bool = True  # false = drop tables; true = keep, linearised
    extract_figures: bool = True
    dedup: bool = True
    quality: dict = field(default_factory=dict)
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
    kwargs = {k: v for k, v in p1.items()
              if k in Phase1Config.__dataclass_fields__ and v is not None}
    if "force_ocr" in kwargs and not isinstance(kwargs["force_ocr"], list):
        kwargs["force_ocr"] = list(kwargs["force_ocr"] or [])
    if "quality" in kwargs and not isinstance(kwargs["quality"], dict):
        kwargs["quality"] = {}
    return Phase1Config(**kwargs)


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


def _process_one(args) -> tuple[str, str, str, str, list[Chunk]]:
    """Top-level (picklable) worker: one PDF -> (name, stem, source_id, sha256, chunks)."""
    pdf_str, cfg = args
    pdf = Path(pdf_str)
    src_id = relative_source_id(pdf, Path(cfg.input_dir))
    try:
        sha = _file_sha256(pdf)
    except OSError:
        sha = ""
    try:
        force = should_force_ocr(src_id, cfg.force_ocr)
        doc = extract_document(
            pdf, backend=cfg.backend, ocr_enabled=cfg.ocr_enabled,
            extract_tables=cfg.extract_tables, extract_figures=cfg.extract_figures,
            source_id=src_id,
            ocr_languages=cfg.ocr_languages,
            force_ocr=force,
        )
        chunks = chunk_document(
            doc, strategy=cfg.strategy,
            target_tokens=cfg.target_tokens, overlap_pct=cfg.overlap_pct,
            parent_max_tokens=cfg.parent_max_tokens, min_tokens=cfg.min_tokens,
            max_tokens=cfg.max_tokens,
            min_chunk_chars=cfg.min_chunk_chars,
            min_table_chars=cfg.min_table_chars,
            min_figure_chars=cfg.min_figure_chars,
            keep_tables=cfg.extract_tables,
        )
        if cfg.dedup:
            chunks = _dedup(chunks)
        chunks = reindex_chunks(chunks)
        chunks = annotate_chunks(chunks, quality_cfg=cfg.quality)
        return pdf.name, relative_stem(src_id), src_id, sha, chunks
    except Exception as e:
        log.exception("  FAILED %s: %s", pdf.name, e)
        return pdf.name, relative_stem(src_id), src_id, sha, []


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

    pending: list[Path] = []
    seen_sha: dict[str, str] = {}
    for pdf in pdfs:
        src_id = relative_source_id(pdf, in_dir)
        try:
            sha = _file_sha256(pdf)
        except OSError:
            sha = ""
        if sha and sha in seen_sha:
            log.info("skip %s: sha256 already queued as %s", src_id, seen_sha[sha])
            if store:
                store.record_source_alias(seen_sha[sha], src_id)
            continue
        if sha and store:
            canon = store.find_canonical_source_by_sha256(sha)
            if canon and canon != src_id:
                log.info("skip %s: sha256 already ingested as %s", src_id, canon)
                store.record_source_alias(canon, src_id)
                continue
        if sha:
            seen_sha[sha] = src_id
        pending.append(pdf)
    pdfs = pending

    all_chunks: list[Chunk] = []
    upserted = 0

    def handle(name: str, stem: str, source_id: str, sha256: str,
               chunks: list[Chunk]) -> None:
        nonlocal upserted
        if cfg.sink == "postgres":
            if not chunks:
                log.info("  %s -> 0 chunks (left existing rows untouched)", name)
                return
            upserted += store.replace_source(source_id, chunks)
            try:
                counts = store.chunk_index_counts(source_id)
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
            for name, stem, source_id, sha256, chunks in pool.imap_unordered(
                    _process_one, tasks):
                handle(name, stem, source_id, sha256, chunks)
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
    over = sum(1 for t in toks if t > effective_max_tokens(DEFAULT_MAX_TOKENS))
    log.info("Stats: %d child chunks | tokens p10=%d p50=%d p90=%d | %d parents "
             "| chunk_index min=%s max=%s nulls=%d | over_cap=%d",
             n, pct(0.1), pct(0.5), pct(0.9),
             sum(1 for c in chunks if c.is_parent),
             min(idxs) if idxs else None,
             max(idxs) if idxs else None,
             missing, over)


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
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    help="body token cap per chunk (default 510 = 512 minus specials)")
    ap.add_argument("--min-chunk-chars", dest="min_chunk_chars", type=int)
    ap.add_argument("--min-table-chars", dest="min_table_chars", type=int)
    ap.add_argument("--min-figure-chars", dest="min_figure_chars", type=int)
    ap.add_argument("--no-ocr", dest="ocr_enabled", action="store_false", default=None)
    ap.add_argument("--ocr-languages", dest="ocr_languages",
                    help="Tesseract languages for ocrmypdf/pytesseract "
                         "(default eng+hin+san)")
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
