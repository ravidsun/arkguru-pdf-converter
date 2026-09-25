"""Wrap ``python -m phase2_web.pipeline`` via a per-job YAML config.

The Phase 2 CLI exposes seeds / max_pages / sink but not ``backend`` or
``same_domain_only``. Writing the existing ``Phase2Config`` YAML is the
thinnest way to pass those fields without forking ``pipeline.py``.

``backend: firecrawl`` is accepted and written through, but the current
``phase2_web.fetch.crawl`` implementation is local httpx + trafilatura only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from ..corpus import count_jsonl_files
from ..jobs import Job, RUNNER
from ..models import Phase2Request
from ..paths import Layout, require
from ..settings import jobs_dir


def default_out_dir(layout: Layout) -> Path:
    if layout.phase2:
        return layout.phase2 / "data" / "processed"
    return layout.ui / "data" / "processed" / "phase2"


def default_seeds(layout: Layout) -> list[str]:
    if not layout.phase2:
        return []
    cfg = layout.phase2 / "config" / "config.yaml"
    if not cfg.exists():
        return []
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    return list((raw.get("phase2") or {}).get("seeds") or [])


def start_phase2(layout: Layout, req: Phase2Request) -> Job:
    phase2 = require(layout.phase2, "Phase 2 (arkguru-web-scraping)")
    out_dir = Path(req.out_dir or default_out_dir(layout)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in req.seeds if s and s.strip()]
    if not seeds:
        raise ValueError("At least one seed URL is required")

    job_id_dir = jobs_dir() / "phase2-pending"
    job_id_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = job_id_dir / "config.yaml"
    payload = {
        "phase2": {
            "seeds": seeds,
            "out_dir": str(out_dir),
            "out_format": "jsonl",
            "max_pages": req.max_pages,
            "max_pages_per_seed": req.max_pages_per_seed,
            "delay_seconds": req.delay_seconds,
            "same_domain_only": req.same_domain_only,
            "backend": req.backend,
            "sink": req.sink,
            "download_pdfs": req.download_pdfs,
            "ingest_pdfs": False,
            "dedup": True,
        }
    }
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "phase2_web.pipeline",
        "--config",
        str(cfg_path),
        "--sink",
        req.sink,
        "--out-dir",
        str(out_dir),
    ]
    if not req.download_pdfs:
        cmd.append("--no-pdfs")

    notes: list[str] = []
    if req.backend == "firecrawl":
        notes.append(
            "Phase 2 crawl() is local httpx/trafilatura today; "
            "backend=firecrawl is stored on the config but not executed."
        )

    def _result(_job: Job) -> dict[str, Any]:
        counts = count_jsonl_files(out_dir)
        web = out_dir / "web_chunks.jsonl"
        return {
            "out_dir": str(out_dir),
            "web_chunks": str(web) if web.exists() else None,
            "sink": req.sink,
            "backend": req.backend,
            "seeds": seeds,
            "chunk_files": counts["files"],
            "chunk_rows": counts["rows"],
            "notes": notes,
        }

    job = RUNNER.start(
        "phase2",
        cmd,
        cwd=phase2,
        env={"PYTHONPATH": layout.pythonpath(phase2)},
        log_path=job_id_dir / "phase2.log",
        on_complete=_result,
    )
    for note in notes:
        job.logs.append(f"[wizard] {note}")
    return job
