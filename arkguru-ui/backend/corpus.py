"""Inspect JSONL outputs produced by the existing phase file sinks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def count_jsonl_files(root: Path | None) -> dict[str, Any]:
    if root is None or not root.exists():
        return {"root": str(root) if root else None, "files": [], "rows": 0}
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*.jsonl")):
        if not path.is_file():
            continue
        n = 0
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        n += 1
        except OSError:
            continue
        files.append({"path": str(path), "rows": n})
        total += n
    return {"root": str(root), "files": files, "rows": total}


def collect_chunk_jsonl(roots: list[Path]) -> list[Path]:
    """JSONL files that look like Chunk dumps (skip train/val pairs)."""
    skip_names = {"train.jsonl", "val.jsonl"}
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".jsonl":
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if path.name in skip_names:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return found


def peek_chunk_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict) and ("text" in data or "chunk_id" in data):
                    return data
                return None
    except (OSError, json.JSONDecodeError):
        return None
    return None
