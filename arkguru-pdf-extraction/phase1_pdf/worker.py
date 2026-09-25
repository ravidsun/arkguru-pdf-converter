"""
Phase 1 autonomous worker: watch the input folder and ingest new/changed PDFs.

Deterministic (no LLM): a file appearing in `input_dir` is the trigger. Only
new or changed PDFs are processed each cycle (tracked in a small state file), and
they're written to whatever sink the config selects (files or the Postgres
datastore).

    python -m phase1_pdf.worker --once                 # single pass
    python -m phase1_pdf.worker --interval 30          # watch forever
    python -m phase1_pdf.worker --sink postgres --interval 60
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path

from common.worker import Worker, FolderState
from .pipeline import Phase1Config, _load_config, run

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("phase1.worker")


def make_run_once(cfg: Phase1Config, state_path: str):
    state = FolderState(state_path)

    def run_once():
        pdfs = sorted(Path(cfg.input_dir).glob("**/*.pdf"))
        todo = state.new_or_changed(pdfs)
        if not todo:
            return "no new PDFs"
        log.info("ingesting %d new/changed PDF(s)", len(todo))
        run(cfg, pdfs=todo)
        state.mark(todo)
        return f"ingested {len(todo)} PDF(s)"
    return run_once


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 folder-watch worker")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--input", dest="input_dir")
    ap.add_argument("--sink", choices=["file", "postgres"])
    ap.add_argument("--state", default="data/.phase1_worker_state.json")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="run a single pass and exit")
    a = ap.parse_args(argv)

    cfg = _load_config(a.config) if Path(a.config).exists() else Phase1Config()
    if a.input_dir: cfg.input_dir = a.input_dir
    if a.sink: cfg.sink = a.sink

    worker = Worker("phase1", make_run_once(cfg, a.state), interval=a.interval)
    worker.run_once() if a.once else worker.run_forever()


if __name__ == "__main__":
    main()
