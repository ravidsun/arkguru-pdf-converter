"""FastAPI entrypoint for the local arkguru wizard."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .adapters import phase1 as phase1_ad
from .adapters import phase2 as phase2_ad
from .adapters import phase3 as phase3_ad
from .corpus import count_jsonl_files
from .jobs import RUNNER
from .models import ChatRequest, Phase1Request, Phase2Request, Phase3Request
from .paths import discover
from .settings import load_env, ollama_host, pg_dsn_set, uploads_dir

load_env()

app = FastAPI(title="arkguru-ui", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _layout():
    return discover()


def _repo_info(path: Path | None) -> dict[str, Any]:
    return {"found": path is not None, "path": str(path) if path else None}


@app.get("/api/health")
def health() -> dict[str, Any]:
    layout = _layout()
    ollama = False
    try:
        r = httpx.get(f"{ollama_host()}/api/tags", timeout=1.5)
        ollama = r.status_code < 500
    except Exception:
        ollama = False
    return {
        "ok": True,
        "umbrella": str(layout.umbrella),
        "common": _repo_info(layout.common),
        "phase1": _repo_info(layout.phase1),
        "phase2": _repo_info(layout.phase2),
        "phase3": _repo_info(layout.phase3),
        "pg_dsn_set": pg_dsn_set(),
        "ollama": ollama,
        "notes": [
            "arkguru-pdf-extraction may be a git submodule and is also listed "
            "in the umbrella .gitignore (sibling clones). Either layout works.",
            "Phase 2 currently lives in this umbrella (./arkguru-web-scraping).",
            "Fine-tune is CLI-only: cd arkguru-rag-slm && make finetune.",
        ],
    }


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    layout = _layout()
    return {
        "phase1_input": str(phase1_ad.default_input_dir(layout)),
        "phase1_out": str(phase1_ad.default_out_dir(layout)),
        "phase2_out": str(phase2_ad.default_out_dir(layout)),
        "phase2_seeds": phase2_ad.default_seeds(layout),
        "phase3_store": str(phase3_ad.default_store(layout)),
        "phase3_processed": str(phase3_ad.default_processed(layout)),
        "uploads": str(uploads_dir()),
        "pg_dsn_set": pg_dsn_set(),
    }


@app.get("/api/corpus")
def corpus() -> dict[str, Any]:
    layout = _layout()
    p1 = layout.phase1 / "data" / "processed" if layout.phase1 else None
    p2 = layout.phase2 / "data" / "processed" if layout.phase2 else None
    p3 = layout.phase3 / "data" / "processed" if layout.phase3 else None
    store = phase3_ad.default_store(layout)
    store_jsonl = store.with_suffix(".jsonl")
    indexed = 0
    if store_jsonl.exists():
        indexed = sum(1 for line in store_jsonl.open() if line.strip())
    return {
        "phase1": count_jsonl_files(p1),
        "phase2": count_jsonl_files(p2),
        "phase3": count_jsonl_files(p3),
        "store": str(store),
        "indexed_rows": indexed,
    }


@app.post("/api/uploads")
async def uploads(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    dest = uploads_dir()
    saved: list[str] = []
    for item in files:
        name = Path(item.filename or "upload.pdf").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF uploads are accepted ({name})")
        path = dest / name
        with path.open("wb") as fh:
            shutil.copyfileobj(item.file, fh)
        saved.append(str(path))
    return {"dir": str(dest), "files": saved}


@app.post("/api/jobs/sample-pdf")
def start_sample_pdf() -> dict[str, Any]:
    try:
        job = phase1_ad.start_sample_pdf(_layout())
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job": job.to_dict()}


@app.post("/api/jobs/phase1")
def start_phase1(req: Phase1Request) -> dict[str, Any]:
    layout = _layout()
    input_dir = Path(req.input_dir).expanduser() if req.input_dir else None
    if input_dir is None:
        input_dir = phase1_ad.default_input_dir(layout)
    if not input_dir.exists():
        raise HTTPException(400, f"PDF input path does not exist: {input_dir}")
    try:
        job = phase1_ad.start_phase1(layout, req, input_dir=input_dir)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job": job.to_dict()}


@app.post("/api/jobs/phase2")
def start_phase2(req: Phase2Request) -> dict[str, Any]:
    try:
        job = phase2_ad.start_phase2(_layout(), req)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job": job.to_dict()}


@app.post("/api/jobs/phase3")
def start_phase3(req: Phase3Request) -> dict[str, Any]:
    try:
        job = phase3_ad.start_phase3(_layout(), req)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job": job.to_dict()}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_dict(log_limit=40) for j in RUNNER.list()]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, log_offset: int = 0) -> dict[str, Any]:
    job = RUNNER.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.to_dict(log_offset=log_offset)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        job = RUNNER.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, "unknown job") from exc
    return {"job": job.to_dict()}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    if RUNNER.get(job_id) is None:
        raise HTTPException(404, "unknown job")

    async def events():
        offset = 0
        while True:
            job = RUNNER.get(job_id)
            if job is None:
                yield "event: error\ndata: missing\n\n"
                return
            if offset < len(job.logs):
                chunk = job.logs[offset:]
                offset = len(job.logs)
                yield f"data: {json.dumps(chunk)}\n\n"
            if job.status in ("succeeded", "failed", "canceled"):
                yield f"event: done\ndata: {json.dumps(job.to_dict(log_limit=0))}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(400, "question is required")
    try:
        return phase3_ad.chat(_layout(), req)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"chat failed: {exc}") from exc
