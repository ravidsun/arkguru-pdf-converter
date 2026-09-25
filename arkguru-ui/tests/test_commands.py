import sys

from backend.commands import build_job_spec
from backend.jobs import JobBusy, JobRunner
from backend.models import PhaseName, UnifiedJobRequest
from backend.paths import discover


def test_phase1_spec_uses_pipeline_and_extraction_cwd():
    layout = discover()
    req = UnifiedJobRequest(phase=PhaseName.phase1, sink="postgres")
    spec = build_job_spec(layout, req, python=sys.executable)
    assert spec.kind == "phase1"
    assert spec.argv[1:3] == ["-m", "phase1_pdf.pipeline"]
    assert "--input" in spec.argv
    assert "--sink" in spec.argv and "postgres" in spec.argv
    assert spec.cwd.name == "arkguru-pdf-extraction"


def test_phase2_spec_requires_seeds_and_sets_cwd():
    layout = discover()
    try:
        build_job_spec(
            layout,
            UnifiedJobRequest(phase=PhaseName.phase2, seeds=[]),
            python=sys.executable,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "seed" in str(exc).lower()
    spec = build_job_spec(
        layout,
        UnifiedJobRequest(
            phase=PhaseName.phase2,
            seeds=["https://example.com/"],
            max_pages=4,
            sink="file",
        ),
        python=sys.executable,
    )
    assert spec.argv[1:3] == ["-m", "phase2_web.pipeline"]
    assert "--config" in spec.argv
    assert "https://example.com/" in spec.argv
    assert "--max-pages" in spec.argv and "4" in spec.argv
    assert spec.cwd.name == "arkguru-web-scraping"


def test_phase3_embed_and_ask_specs():
    layout = discover()
    embed = build_job_spec(
        layout,
        UnifiedJobRequest(phase=PhaseName.phase3_embed),
        python=sys.executable,
    )
    assert embed.argv[1:3] == ["-m", "phase3_rag.embed_datastore"]
    assert "--embedder" in embed.argv and "hashing" in embed.argv
    assert embed.cwd.name == "arkguru-rag-slm"
    try:
        build_job_spec(
            layout,
            UnifiedJobRequest(phase=PhaseName.phase3_ask, question="  "),
            python=sys.executable,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "question" in str(exc).lower()
    ask = build_job_spec(
        layout,
        UnifiedJobRequest(phase=PhaseName.phase3_ask, question="What is PPE?"),
        python=sys.executable,
    )
    assert ask.argv[1:3] == ["-m", "phase3_rag.serve"]
    assert "--ask" in ask.argv and "What is PPE?" in ask.argv


def test_second_job_rejected_while_running():
    runner = JobRunner()
    first = runner.start(
        "sleep",
        [sys.executable, "-c", "import time; time.sleep(8)"],
        cwd=".",
        env={},
    )
    try:
        runner.start(
            "echo",
            [sys.executable, "-c", "print('nope')"],
            cwd=".",
            env={},
        )
        raise AssertionError("expected JobBusy")
    except JobBusy as exc:
        assert first.id in str(exc)
    assert runner.current() is first
    if first._proc:
        first._proc.terminate()
