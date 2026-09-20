"""Download publicly linked PDFs and Phase-1 them into schema ``web``."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse

from .fetch import USER_AGENT, is_binary_url, is_pdf_url

log = logging.getLogger("phase2.pdfs")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def pdf_filename(url: str) -> str:
    path = unquote(urlparse(url).path or "")
    name = path.rsplit("/", 1)[-1] or "download.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    name = _SAFE.sub("_", name).strip("._") or "download.pdf"
    return name[:120]


def download_pdfs(
    urls: Iterable[str],
    dest_dir: str | Path,
    *,
    client=None,
    timeout: float = 60.0,
) -> list[Path]:
    import httpx

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    own = client is None
    if own:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout,
                              follow_redirects=True)
    written: list[Path] = []
    try:
        for url in urls:
            if is_binary_url(url) or not is_pdf_url(url):
                continue
            host = (urlparse(url).hostname or "pdf").lower().removeprefix("www.")
            folder = dest / host
            folder.mkdir(parents=True, exist_ok=True)
            out = folder / pdf_filename(url)
            try:
                r = client.get(url, timeout=timeout)
                r.raise_for_status()
                ctype = r.headers.get("content-type", "")
                if "pdf" not in ctype.lower() and not r.content[:8].startswith(b"%PDF"):
                    log.info("skip non-pdf %s (%s)", url, ctype)
                    continue
                out.write_bytes(r.content)
                written.append(out)
                log.info("saved %s <- %s", out, url)
            except Exception as e:
                log.warning("pdf download failed %s: %s", url, e)
    finally:
        if own:
            client.close()
    return written


def ingest_pdfs_with_phase1(
    input_dir: str | Path,
    datastore_config: str,
    *,
    sink: str = "postgres",
) -> int:
    """Run Phase 1 against downloaded PDFs using Phase 2's datastore.yaml.

    Requires ``arkguru-pdf-extraction`` on ``sys.path`` (sibling checkout).
    """
    from pathlib import Path as P
    import sys
    here = P(__file__).resolve()
    candidates = [parent / "arkguru-pdf-extraction" for parent in here.parents]
    cwd = P.cwd()
    candidates.extend([
        cwd / "arkguru-pdf-extraction",
        cwd.parent / "arkguru-pdf-extraction",
        P("/tmp/arkguru-repos/arkguru-pdf-extraction"),
        P("/workspace/arkguru-pdf-extraction"),
    ])
    for sibling in candidates:
        if (sibling / "phase1_pdf" / "pipeline.py").exists():
            sys.path.insert(0, str(sibling))
            break
    else:
        raise RuntimeError(
            "arkguru-pdf-extraction not found; clone it as a sibling to ingest PDFs"
        )
    from phase1_pdf.pipeline import Phase1Config, run

    cfg = Phase1Config(
        input_dir=str(input_dir),
        sink=sink,
        datastore_config=datastore_config,
        workers=1,
    )
    chunks = run(cfg)
    return len(chunks)
