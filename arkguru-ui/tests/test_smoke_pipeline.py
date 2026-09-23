"""Optional end-to-end smoke: sample PDF → Phase 1 → index → chat.

Skipped when Phase 1/3 checkouts or extraction deps are missing.
"""

from __future__ import annotations

import time

import pytest

from backend.adapters import phase1 as phase1_ad
from backend.adapters import phase3 as phase3_ad
from backend.jobs import RUNNER
from backend.models import ChatRequest, Phase1Request, Phase3Request
from backend.paths import discover


def _wait(job, seconds: float = 120):
    deadline = time.time() + seconds
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.2)
    return job


def test_sample_pdf_to_extractive_chat():
    layout = discover()
    if layout.phase1 is None or layout.phase3 is None:
        pytest.skip("phase1/phase3 checkouts not present")
    try:
        import pymupdf  # noqa: F401
        import reportlab  # noqa: F401
    except ImportError:
        pytest.skip("Phase 1 extraction extras not installed")

    sample = _wait(phase1_ad.start_sample_pdf(layout), 60)
    assert sample.status == "succeeded", sample.logs[-20:]
    input_dir = sample.result["input_dir"]

    extract = _wait(
        phase1_ad.start_phase1(
            layout,
            Phase1Request(
                input_dir=input_dir,
                backend="pymupdf",
                strategy="structure",
                ocr_enabled=False,
                extract_figures=False,
                extract_tables=False,
                workers=1,
                sink="file",
            ),
            input_dir=input_dir,
        ),
        90,
    )
    assert extract.status == "succeeded", "\n".join(extract.logs[-40:])
    assert extract.result.get("chunk_rows", 0) > 0

    index = _wait(
        phase3_ad.start_phase3(
            layout,
            Phase3Request(
                sink="file",
                embedder="hashing",
                include_phase1=True,
                include_phase2=False,
                skip_finetune=True,
            ),
        ),
        90,
    )
    assert index.status == "succeeded", "\n".join(index.logs[-40:])

    reply = phase3_ad.chat(
        layout,
        ChatRequest(
            question="What PPE is required before servicing a unit?",
            sink="file",
            embedder="hashing",
        ),
    )
    blob = (reply["answer"] + " " + " ".join(s.get("text") or "" for s in reply["sources"])).lower()
    assert reply["sources"], reply
    assert any(token in blob for token in ("glove", "ppe", "eyewear", "lockout"))
    # Job runner still has the jobs we started (sanity for the UI poll path)
    assert RUNNER.get(extract.id) is not None
