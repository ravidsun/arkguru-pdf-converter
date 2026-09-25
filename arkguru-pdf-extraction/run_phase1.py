#!/usr/bin/env python
"""
run_phase1.py — Convenience runner for Phase 1: PDF Extraction Pipeline.

Assumes raw PDF files are already placed in the input folder (data/raw_pdfs by
default, or whatever is set in config/config.yaml).

Usage examples
--------------
# Use defaults from config/config.yaml
    python run_phase1.py

# Override input/output paths
    python run_phase1.py --input data/raw_pdfs --out-dir data/processed

# Choose a different backend or chunking strategy
    python run_phase1.py --backend docling --strategy parent_child

# Output as Parquet instead of JSONL
    python run_phase1.py --format parquet

# Disable OCR (faster, text-native PDFs only)
    python run_phase1.py --no-ocr

# Disable table/figure extraction (faster, prose-only)
    python run_phase1.py --no-tables --no-figures

# Point at a custom config file
    python run_phase1.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_phase1")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 runner — PDF extraction & chunking pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        help="Folder that contains raw PDFs (overrides config).",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="Output directory; one folder per PDF under "
             "{relative_stem}/ (chunks.jsonl plus optional parents/tables/"
             "figures.jsonl) (overrides config).",
    )
    parser.add_argument(
        "--format",
        dest="out_format",
        choices=["jsonl", "parquet"],
        help="Output format (overrides config).",
    )
    parser.add_argument(
        "--backend",
        choices=["pymupdf4llm", "docling", "pymupdf"],
        help="PDF extraction backend (overrides config).",
    )
    parser.add_argument(
        "--strategy",
        choices=["structure", "parent_child", "semantic"],
        help="Chunking strategy (overrides config).",
    )
    parser.add_argument(
        "--target-tokens",
        dest="target_tokens",
        type=int,
        help="Target tokens per chunk (overrides config).",
    )
    parser.add_argument(
        "--overlap-pct",
        dest="overlap_pct",
        type=float,
        help="Overlap percentage between chunks (overrides config).",
    )
    parser.add_argument(
        "--min-tokens",
        dest="min_tokens",
        type=int,
        help="Merge trailing chunks smaller than this into the previous one "
             "(overrides config).",
    )
    parser.add_argument(
        "--no-ocr",
        dest="ocr_enabled",
        action="store_false",
        default=None,
        help="Disable OCR for scanned pages.",
    )
    parser.add_argument(
        "--no-tables",
        dest="extract_tables",
        action="store_false",
        default=None,
        help="Disable table extraction (page.find_tables()).",
    )
    parser.add_argument(
        "--no-figures",
        dest="extract_figures",
        action="store_false",
        default=None,
        help="Disable OCR of embedded diagrams/charts/graphs.",
    )
    parser.add_argument(
        "--sink",
        choices=["file", "postgres"],
        help="Where to write chunks: files (default) or the Postgres datastore "
             "(connection/tables in config/datastore.yaml).",
    )
    parser.add_argument(
        "--init-db",
        dest="init_db",
        action="store_true",
        help="Create datastore tables (chunks, chunk_embeddings) if missing, "
             "then exit. Run once after setting up the database.",
    )
    return parser.parse_args(argv)


def _check_input(input_dir: str) -> None:
    """Validate that the input directory exists and contains PDFs."""
    p = Path(input_dir)
    if not p.exists():
        log.error("Input directory not found: %s", p.resolve())
        sys.exit(1)
    pdfs = list(p.glob("**/*.pdf"))
    if not pdfs:
        log.error("No PDF files found in: %s", p.resolve())
        sys.exit(1)
    log.info("Found %d PDF(s) in '%s':", len(pdfs), p.resolve())
    for pdf in sorted(pdfs):
        log.info("  - %s  (%.1f KB)", pdf.name, pdf.stat().st_size / 1024)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # We must run from the project root so that relative paths resolve.
    # ------------------------------------------------------------------
    script_dir = Path(__file__).parent.resolve()
    import os
    os.chdir(script_dir)
    log.info("Working directory: %s", script_dir)

    # ------------------------------------------------------------------
    # Import pipeline (after chdir so local packages are findable)
    # ------------------------------------------------------------------
    try:
        from phase1_pdf.pipeline import Phase1Config, _load_config, run
    except ImportError as exc:
        log.error(
            "Could not import phase1_pdf package. "
            "Make sure dependencies are installed: pip install -r requirements.txt\n"
            "Error: %s",
            exc,
        )
        return 1

    # ------------------------------------------------------------------
    # Build config: start from YAML, then apply CLI overrides
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        log.info("Loading config from: %s", config_path)
        cfg = _load_config(str(config_path))
    else:
        log.warning("Config file not found (%s), using built-in defaults.", config_path)
        cfg = Phase1Config()

    # Apply any CLI overrides (skip None / unset values)
    overrides = {
        "input_dir": args.input_dir,
        "out_dir": args.out_dir,
        "out_format": args.out_format,
        "backend": args.backend,
        "strategy": args.strategy,
        "target_tokens": args.target_tokens,
        "overlap_pct": args.overlap_pct,
        "min_tokens": args.min_tokens,
        "sink": args.sink,
    }
    if args.ocr_enabled is not None:
        overrides["ocr_enabled"] = args.ocr_enabled
    if args.extract_tables is not None:
        overrides["extract_tables"] = args.extract_tables
    if args.extract_figures is not None:
        overrides["extract_figures"] = args.extract_figures

    for key, val in overrides.items():
        if val is not None:
            setattr(cfg, key, val)
            log.info("Override: %s = %s", key, val)

    # ------------------------------------------------------------------
    # --init-db: ensure datastore tables exist, then exit (no PDFs needed)
    # ------------------------------------------------------------------
    if args.init_db:
        try:
            from common.datastore_config import open_chunk_store
            store = open_chunk_store(cfg.datastore_config)
            store.ensure_schema()
            log.info("Datastore ready: tables '%s' and '%s' ensured.",
                     store.chunks, store.vectors)
            return 0
        except Exception:
            log.exception("Failed to initialize the datastore. "
                          "See docs/DATABASE_SETUP.md.")
            return 1

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Phase 1 Configuration:")
    log.info("  input_dir       : %s", cfg.input_dir)
    log.info("  out_dir         : %s", cfg.out_dir)
    log.info("  out_format      : %s", cfg.out_format)
    log.info("  backend         : %s", cfg.backend)
    log.info("  strategy        : %s", cfg.strategy)
    log.info("  target_tokens   : %s", cfg.target_tokens)
    log.info("  overlap_pct     : %s", cfg.overlap_pct)
    log.info("  min_tokens      : %s", cfg.min_tokens)
    log.info("  ocr_enabled     : %s", cfg.ocr_enabled)
    log.info("  extract_tables  : %s", cfg.extract_tables)
    log.info("  extract_figures : %s", cfg.extract_figures)
    log.info("  dedup           : %s", cfg.dedup)
    log.info("=" * 60)

    _check_input(cfg.input_dir)

    # Ensure output directory exists
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run the pipeline
    # ------------------------------------------------------------------
    log.info("Starting Phase 1 pipeline...")
    t0 = time.perf_counter()
    try:
        chunks = run(cfg)
    except Exception:
        log.exception("Pipeline failed with an unexpected error.")
        return 1
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if chunks:
        log.info("=" * 60)
        log.info("Phase 1 completed successfully in %.1f s", elapsed)
        log.info("Total chunks produced : %d", len(chunks))
        log.info("Output written to     : %s", Path(cfg.out_dir).resolve())
        log.info("=" * 60)
    else:
        log.warning("Pipeline finished but produced 0 chunks — check your PDFs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

