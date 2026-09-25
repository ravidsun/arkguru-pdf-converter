"""Wrap ``python -m phase1_pdf.pipeline`` and ``scripts/make_sample_pdf.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..corpus import count_jsonl_files
from ..jobs import Job, RUNNER
from ..models import Phase1Request
from ..paths import Layout, require
from ..settings import jobs_dir


def default_input_dir(layout: Layout) -> Path:
    if layout.phase1:
        return layout.phase1 / "data" / "raw_pdfs"
    return layout.ui / "data" / "uploads"


def default_out_dir(layout: Layout) -> Path:
    if layout.phase1:
        return layout.phase1 / "data" / "processed"
    return layout.ui / "data" / "processed" / "phase1"


def start_phase1(layout: Layout, req: Phase1Request, *, input_dir: Path | None = None) -> Job:
    phase1 = require(layout.phase1, "Phase 1 (arkguru-pdf-extraction)")
    in_dir = Path(input_dir or req.input_dir or default_input_dir(layout)).expanduser()
    out_dir = Path(req.out_dir or default_out_dir(layout)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "phase1_pdf.pipeline",
        "--input",
        str(in_dir),
        "--out-dir",
        str(out_dir),
        "--backend",
        req.backend,
        "--strategy",
        req.strategy,
        "--target-tokens",
        str(req.target_tokens),
        "--overlap-pct",
        str(req.overlap_pct),
        "--min-tokens",
        str(req.min_tokens),
        "--workers",
        str(req.workers),
        "--sink",
        req.sink,
    ]
    if not req.ocr_enabled:
        cmd.append("--no-ocr")
    if not req.extract_tables:
        cmd.append("--no-tables")
    if not req.extract_figures:
        cmd.append("--no-figures")

    job_dir = jobs_dir() / "pending"
    job_dir.mkdir(parents=True, exist_ok=True)

    def _result(_job: Job) -> dict[str, Any]:
        counts = count_jsonl_files(out_dir)
        return {
            "input_dir": str(in_dir),
            "out_dir": str(out_dir),
            "sink": req.sink,
            "backend": req.backend,
            "strategy": req.strategy,
            "chunk_files": counts["files"],
            "chunk_rows": counts["rows"],
        }

    return RUNNER.start(
        "phase1",
        cmd,
        cwd=phase1,
        env={"PYTHONPATH": layout.pythonpath(phase1)},
        log_path=job_dir / "phase1.log",
        on_complete=_result,
    )


def start_sample_pdf(layout: Layout) -> Job:
    phase1 = require(layout.phase1, "Phase 1 (arkguru-pdf-extraction)")
    script = phase1 / "scripts" / "make_sample_pdf.py"
    if not script.exists():
        raise FileNotFoundError(f"sample PDF script missing: {script}")
    out = phase1 / "data" / "raw_pdfs" / "sample_handbook.pdf"

    def _result(_job: Job) -> dict[str, Any]:
        return {
            "pdf": str(out),
            "input_dir": str(out.parent),
            "exists": out.exists(),
            "bytes": out.stat().st_size if out.exists() else 0,
        }

    return RUNNER.start(
        "sample_pdf",
        [sys.executable, str(script)],
        cwd=phase1,
        env={"PYTHONPATH": layout.pythonpath(phase1)},
        on_complete=_result,
    )
