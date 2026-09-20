"""Phase 2 worker: one crawl pass, optionally on an interval.

    python -m phase2_web.worker --once
    python -m phase2_web.worker --interval 3600 --sink postgres
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.worker import Worker
from .pipeline import Phase2Config, _load_config, run

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("phase2.worker")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2 crawl worker")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sink", choices=["file", "postgres"])
    ap.add_argument("--interval", type=float, default=3600.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)

    cfg = _load_config(a.config) if Path(a.config).exists() else Phase2Config()
    if a.sink:
        cfg.sink = a.sink

    def run_once():
        chunks = run(cfg)
        return f"{len(chunks)} chunks"

    worker = Worker("phase2", run_once, interval=a.interval)
    worker.run_once() if a.once else worker.run_forever()


if __name__ == "__main__":
    main()
