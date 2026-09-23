"""Load gitignored env files without echoing secrets."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .paths import UI_ROOT, umbrella_root


def load_env() -> None:
    """Load umbrella then UI `.env`. Existing process env wins."""
    for candidate in (umbrella_root() / ".env", UI_ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def pg_dsn_set() -> bool:
    return bool(os.environ.get("PG_DSN", "").strip())


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def data_dir() -> Path:
    path = UI_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir() -> Path:
    path = data_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_dir() -> Path:
    path = data_dir() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path
