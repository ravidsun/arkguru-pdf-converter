import sys
import time

from backend.jobs import JobRunner


def test_job_runner_captures_echo():
    runner = JobRunner()
    job = runner.start(
        "echo",
        [sys.executable, "-c", "print('hello-wizard')"],
        cwd=".",
        env={},
    )
    deadline = time.time() + 5
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.05)
    assert job.status == "succeeded"
    assert any("hello-wizard" in line for line in job.logs)
    payload = job.to_dict()
    assert payload["log_total"] >= 1
    assert payload["returncode"] == 0
