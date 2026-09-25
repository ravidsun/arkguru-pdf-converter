"""Remember the last Phase 3 index settings for chat defaults.

``PG_DSN`` being present must not yank a file-mode wizard run onto Postgres.
The user (or the last successful index job) chooses the sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Session:
    sink: str = "file"
    embedder: str = "hashing"
    store: Optional[str] = None
    dim: int = 1024


SESSION = Session()
