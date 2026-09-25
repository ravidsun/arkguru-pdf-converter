import time

from backend.adapters import phase1 as phase1_ad
from backend.models import Phase1Request
from backend.paths import discover


def test_phase1_cmd_flags_when_repo_present():
    layout = discover()
    if layout.phase1 is None:
        return
    req = Phase1Request(
        backend="pymupdf",
        strategy="parent_child",
        ocr_enabled=False,
        extract_figures=False,
        sink="file",
        workers=1,
        input_dir=str(phase1_ad.default_input_dir(layout)),
    )
    # Default input dir may be empty; pipeline still starts and logs a warning.
    job = phase1_ad.start_phase1(layout, req)
    assert job.cmd[1:3] == ["-m", "phase1_pdf.pipeline"]
    assert "--backend" in job.cmd and "pymupdf" in job.cmd
    assert "--strategy" in job.cmd and "parent_child" in job.cmd
    assert "--no-ocr" in job.cmd
    assert "--no-figures" in job.cmd
    deadline = time.time() + 20
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.1)
    assert job.status in ("succeeded", "failed")
