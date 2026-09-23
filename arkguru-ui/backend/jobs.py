"""In-memory job runner with captured logs.

Long OCR / crawl / index runs stay off the request thread. The browser polls
``GET /api/jobs/{id}`` or streams ``GET /api/jobs/{id}/stream``.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
OnComplete = Callable[["Job"], dict[str, Any]]

_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    kind: str
    cmd: list[str]
    cwd: str
    env: dict[str, str]
    status: JobStatus = "queued"
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    returncode: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    on_complete: Optional[OnComplete] = None
    pid: Optional[int] = None
    log_path: Optional[str] = None
    _proc: Optional[subprocess.Popen[str]] = field(default=None, repr=False)

    def to_dict(self, *, log_offset: int = 0, log_limit: int = 2000) -> dict[str, Any]:
        lines = self.logs[log_offset : log_offset + log_limit]
        return {
            "id": self.id,
            "kind": self.kind,
            "cmd": self.cmd,
            "cwd": self.cwd,
            "status": self.status,
            "logs": lines,
            "log_offset": log_offset,
            "log_total": len(self.logs),
            "result": self.result,
            "error": self.error,
            "returncode": self.returncode,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pid": self.pid,
        }


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Optional[Job]:
        with _LOCK:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with _LOCK:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def start(
        self,
        kind: str,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path | None = None,
        on_complete: OnComplete | None = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            cmd=list(cmd),
            cwd=str(cwd),
            env=env,
            on_complete=on_complete,
            log_path=str(log_path) if log_path else None,
        )
        with _LOCK:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        proc = job._proc
        if proc and proc.poll() is None:
            proc.terminate()
            job.status = "canceled"
            job.logs.append("[wizard] sent SIGTERM")
        return job

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        log_file = None
        if job.log_path:
            Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(job.log_path, "w", encoding="utf-8")
        try:
            merged_env = os.environ.copy()
            merged_env.update(job.env)
            proc = subprocess.Popen(
                job.cmd,
                cwd=job.cwd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            job._proc = proc
            job.pid = proc.pid
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                job.logs.append(line)
                if log_file:
                    log_file.write(line + "\n")
                    log_file.flush()
            job.returncode = proc.wait()
            if job.status == "canceled":
                job.error = "canceled"
            elif job.returncode != 0:
                job.status = "failed"
                job.error = f"exit {job.returncode}"
            else:
                job.status = "succeeded"
                if job.on_complete:
                    try:
                        job.result = job.on_complete(job) or {}
                    except Exception as exc:  # pragma: no cover - result enrichment
                        job.logs.append(f"[wizard] result hook failed: {exc}")
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.logs.append(f"[wizard] {exc}")
        finally:
            job.finished_at = time.time()
            if job.status == "running":
                job.status = "failed"
                job.error = job.error or "ended without a status"
            if log_file:
                log_file.close()


RUNNER = JobRunner()
