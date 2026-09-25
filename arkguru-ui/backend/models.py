"""Pydantic request/response models for the wizard API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Phase1Backend = Literal["pymupdf4llm", "docling", "pymupdf"]
Phase1Strategy = Literal["structure", "parent_child", "semantic"]
Phase2Backend = Literal["local", "firecrawl"]
Sink = Literal["file", "postgres"]
EmbedderName = Literal["hashing", "sentence_transformer"]


class Phase1Request(BaseModel):
    input_dir: str | None = None
    out_dir: str | None = None
    backend: Phase1Backend = "pymupdf4llm"
    strategy: Phase1Strategy = "structure"
    ocr_enabled: bool = True
    extract_figures: bool = True
    extract_tables: bool = True
    target_tokens: int = 400
    overlap_pct: float = 0.15
    min_tokens: int = 80
    workers: int = 1
    sink: Sink = "file"


class Phase2Request(BaseModel):
    seeds: list[str] = Field(default_factory=list)
    max_pages: int = 8
    max_pages_per_seed: int = 4
    delay_seconds: float = 0.25
    same_domain_only: bool = True
    backend: Phase2Backend = "local"
    sink: Sink = "file"
    out_dir: str | None = None
    download_pdfs: bool = False


class Phase3Request(BaseModel):
    sink: Sink = "file"
    embedder: EmbedderName = "hashing"
    model_name: str = "BAAI/bge-m3"
    dim: int = 1024
    include_phase1: bool = True
    include_phase2: bool = True
    skip_finetune: bool = True
    store: str | None = None
    processed_dir: str | None = None


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    embedder: EmbedderName | None = None
    sink: Sink | None = None
    store: str | None = None
    model: str | None = None
    dim: int = 1024


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    extractive: bool
    ollama: bool
    store: str | None = None
    embedder: str | None = None


class JobAck(BaseModel):
    job: dict[str, Any]
