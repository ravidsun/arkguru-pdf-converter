"""Locate the umbrella repo and each phase checkout.

Cloud Agents and laptop clones disagree on layout:

* Cloud / sibling: ``../arkguru-pdf-extraction`` next to this umbrella
* Nested: ``./arkguru-pdf-extraction`` inside the umbrella (submodule or clone)
* Phase 2 currently lives **in this umbrella** (``./arkguru-web-scraping``)

``arkguru-pdf-extraction`` is listed in ``.gitmodules`` **and** in the
umbrella ``.gitignore`` (nested sibling clones). The wizard accepts either
and never assumes the submodule was initialized.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

UI_ROOT = Path(__file__).resolve().parents[1]
UMBRELLA_DEFAULT = UI_ROOT.parent


def umbrella_root() -> Path:
    override = os.environ.get("ARKGURU_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return UMBRELLA_DEFAULT


def _first_existing(candidates: list[Path], marker: str) -> Optional[Path]:
    seen: set[Path] = set()
    for raw in candidates:
        if not raw:
            continue
        path = raw.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / marker).exists():
            return path
    return None


def find_checkout(name: str, marker: str, env_var: str | None = None) -> Optional[Path]:
    root = umbrella_root()
    parent = root.parent
    candidates: list[Path] = []
    if env_var and os.environ.get(env_var):
        candidates.append(Path(os.environ[env_var]))
    candidates.extend(
        [
            parent / name,
            root / name,
        ]
    )
    return _first_existing(candidates, marker)


def find_common() -> Optional[Path]:
    return find_checkout("arkguru-common", "pyproject.toml", "COMMON_REPO")


def find_phase1() -> Optional[Path]:
    return find_checkout(
        "arkguru-pdf-extraction",
        "phase1_pdf/pipeline.py",
        "PHASE1_REPO",
    )


def find_phase2() -> Optional[Path]:
    """Phase 2 is vendored in the umbrella; still allow a sibling override."""
    return find_checkout(
        "arkguru-web-scraping",
        "phase2_web/pipeline.py",
        "PHASE2_REPO",
    )


def find_phase3() -> Optional[Path]:
    return find_checkout("arkguru-rag-slm", "phase3_rag/run_pdfs.py", "PHASE3_REPO")


@dataclass(frozen=True)
class Layout:
    umbrella: Path
    ui: Path
    common: Optional[Path]
    phase1: Optional[Path]
    phase2: Optional[Path]
    phase3: Optional[Path]

    def pythonpath(self, *extra: Path) -> str:
        parts = [str(p) for p in extra if p]
        if self.common:
            parts.append(str(self.common))
        existing = os.environ.get("PYTHONPATH")
        if existing:
            parts.append(existing)
        # de-dupe, keep order
        out, seen = [], set()
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return os.pathsep.join(out)


def discover() -> Layout:
    return Layout(
        umbrella=umbrella_root(),
        ui=UI_ROOT,
        common=find_common(),
        phase1=find_phase1(),
        phase2=find_phase2(),
        phase3=find_phase3(),
    )


def require(path: Optional[Path], label: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"{label} checkout not found. Clone it as a sibling of the "
            "umbrella repo or nest it under the umbrella (see docs/LOCAL_RUN.md)."
        )
    return path
