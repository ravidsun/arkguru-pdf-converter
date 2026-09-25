"""
Autonomous orchestrator (no LLM): run the three phase workers as a
dependency-ordered pipeline, once or on a schedule.

It's a tiny DAG runner. Each step is a shell command (a phase's `--once` worker)
with an optional `depends_on`. Steps run in topological order; if a dependency
failed or was skipped, dependents are skipped (so you never embed chunks that
weren't ingested). Everything funnels through the shared datastore, so the
orchestrator only has to run the phases in the right order.

    python orchestrator.py --once                 # one full pipeline pass
    python orchestrator.py --interval 300          # loop every 5 min
    python orchestrator.py --config orchestrator.yaml

Config: orchestrator.yaml (see that file). No external deps.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

from common.worker import Worker   # reuse the resilient loop + signal handling

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("orchestrator")


def _load(path: str) -> dict:
    import yaml
    return (yaml.safe_load(Path(path).read_text()) or {}).get("orchestrator", {})


def _topo_order(steps: list[dict]) -> list[dict]:
    """Kahn topological sort by depends_on; stable on declaration order."""
    by_name = {s["name"]: s for s in steps}
    indeg = {s["name"]: 0 for s in steps}
    for s in steps:
        for d in s.get("depends_on", []) or []:
            if d in by_name:
                indeg[s["name"]] += 1
    order, ready = [], [s["name"] for s in steps if indeg[s["name"]] == 0]
    while ready:
        n = ready.pop(0)
        order.append(by_name[n])
        for s in steps:
            if n in (s.get("depends_on", []) or []):
                indeg[s["name"]] -= 1
                if indeg[s["name"]] == 0:
                    ready.append(s["name"])
    if len(order) != len(steps):
        raise ValueError("orchestrator: dependency cycle detected")
    return order


def run_pipeline(cfg: dict) -> dict:
    steps = [s for s in cfg.get("steps", []) if s.get("enabled", True)]
    if not steps:
        log.warning("no enabled steps"); return {}
    order = _topo_order(steps)
    stop_on_failure = cfg.get("stop_on_failure", False)
    status: dict[str, str] = {}

    for step in order:
        name = step["name"]
        deps = step.get("depends_on", []) or []
        bad = [d for d in deps if status.get(d) not in ("ok", None)]
        if bad:
            status[name] = "skipped"
            log.warning("[%s] skipped (dependency %s not ok)", name, bad)
            continue
        cwd = step.get("cwd", ".")
        cmd = step["command"]
        log.info("[%s] running: %s  (cwd=%s)", name, cmd, cwd)
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, shell=True, cwd=cwd)
            ok = r.returncode == 0
        except Exception as e:
            log.exception("[%s] launch error: %s", name, e); ok = False
        status[name] = "ok" if ok else "failed"
        log.info("[%s] %s in %.1fs", name, status[name], time.perf_counter() - t0)
        if not ok and stop_on_failure:
            log.error("stop_on_failure -> aborting remaining steps"); break

    log.info("pipeline summary: %s", status)
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description="Autonomous phase orchestrator (DAG)")
    ap.add_argument("--config", default="orchestrator.yaml")
    ap.add_argument("--interval", type=float, help="loop every N seconds (else use config)")
    ap.add_argument("--once", action="store_true", help="run one pass and exit")
    a = ap.parse_args(argv)

    cfg = _load(a.config)
    interval = a.interval or cfg.get("interval_seconds", 300)
    worker = Worker("orchestrator", lambda: run_pipeline(cfg), interval=interval)
    worker.run_once() if a.once else worker.run_forever()


if __name__ == "__main__":
    main()
