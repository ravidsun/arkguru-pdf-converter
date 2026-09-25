"""Map a unified UI job request to an existing phase CLI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import PhaseName, Sink, UnifiedJobRequest
from .paths import Layout, require


@dataclass(frozen=True)
class JobSpec:
    kind: str
    argv: list[str]
    cwd: Path


def _python() -> str:
    return sys.executable


def build_job_spec(
    layout: Layout,
    req: UnifiedJobRequest,
    *,
    python: Optional[str] = None,
) -> JobSpec:
    """Return argv + cwd for the requested phase. Does not start a process."""
    py = python or _python()
    phase = req.phase
    if phase is PhaseName.phase1:
        return _phase1(layout, req, py)
    if phase is PhaseName.phase2:
        return _phase2(layout, req, py)
    if phase is PhaseName.phase3_embed:
        return _phase3_embed(layout, py)
    if phase is PhaseName.phase3_ask:
        return _phase3_ask(layout, req, py)
    unused: PhaseName = phase
    raise ValueError(f"unhandled phase: {unused}")


def _phase1(layout: Layout, req: UnifiedJobRequest, py: str) -> JobSpec:
    phase1 = require(layout.phase1, "Phase 1 (arkguru-pdf-extraction)")
    default_in = phase1 / "data" / "raw_pdfs"
    input_dir = Path(req.input_dir).expanduser() if req.input_dir else default_in
    if not input_dir.is_absolute():
        input_dir = (layout.umbrella / input_dir).resolve()
    sink: Sink = req.sink
    argv = [
        py,
        "-m",
        "phase1_pdf.pipeline",
        "--input",
        str(input_dir),
        "--sink",
        sink,
    ]
    return JobSpec(kind=PhaseName.phase1.value, argv=argv, cwd=phase1)


def _phase2(layout: Layout, req: UnifiedJobRequest, py: str) -> JobSpec:
    phase2 = require(layout.phase2, "Phase 2 (arkguru-web-scraping)")
    seeds = [s.strip() for s in (req.seeds or []) if s and s.strip()]
    if not seeds:
        raise ValueError("Phase 2 requires at least one seed URL")
    argv = [
        py,
        "-m",
        "phase2_web.pipeline",
        "--config",
        "config/config.yaml",
        "--seeds",
        *seeds,
        "--sink",
        req.sink,
    ]
    if req.max_pages is not None:
        argv.extend(["--max-pages", str(req.max_pages)])
    return JobSpec(kind=PhaseName.phase2.value, argv=argv, cwd=phase2)


def _phase3_embed(layout: Layout, py: str) -> JobSpec:
    phase3 = require(layout.phase3, "Phase 3 (arkguru-rag-slm)")
    argv = [
        py,
        "-m",
        "phase3_rag.embed_datastore",
        "--embedder",
        "hashing",
        "--dim",
        "1024",
    ]
    return JobSpec(kind=PhaseName.phase3_embed.value, argv=argv, cwd=phase3)


def _phase3_ask(layout: Layout, req: UnifiedJobRequest, py: str) -> JobSpec:
    phase3 = require(layout.phase3, "Phase 3 (arkguru-rag-slm)")
    question = (req.question or "").strip()
    if not question:
        raise ValueError("Phase 3 ask requires a question")
    argv = [
        py,
        "-m",
        "phase3_rag.serve",
        "--ask",
        question,
    ]
    return JobSpec(kind=PhaseName.phase3_ask.value, argv=argv, cwd=phase3)
