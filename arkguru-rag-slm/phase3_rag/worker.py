"""
Phase 3 autonomous worker: keep the vectors table in sync with the chunks table.

Deterministic: the trigger is "there are chunks without embeddings." Each cycle
polls the datastore; when Phase 1/2 have added new chunks, it embeds only those
(idempotent -- fills the separate chunk_embeddings table in place).

    python -m phase3_rag.worker --once
    python -m phase3_rag.worker --interval 60
    python -m phase3_rag.worker --embedder hashing --once
    python -m phase3_rag.worker --embedder sentence_transformer --model BAAI/bge-m3
"""
from __future__ import annotations
import argparse, logging

from common.worker import Worker
from common.datastore_config import open_chunk_store
from phase3_rag.embedder import Embedder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("phase3.worker")


def make_run_once(datastore_config, embedder_backend, model, dim, batch):
    emb = Embedder(backend=embedder_backend, model_name=model, dim=dim)

    def run_once():
        store = open_chunk_store(datastore_config, dim=emb.dim)
        store.ensure_schema()
        todo = store.count(only_missing_embedding=True)
        if not todo:
            return "nothing to embed"
        done = 0
        for b in store.iter_missing_embeddings(batch=batch):
            store.update_embeddings([x[0] for x in b],
                                    emb.encode([x[1] for x in b]), model=model)
            done += len(b)
        return f"embedded {done} chunk(s)"
    return run_once


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 3 embedding worker")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"],
                    default=None,
                    help="Default: hashing with --once (Cloud-safe); "
                         "sentence_transformer when running as a daemon.")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)

    backend = a.embedder or ("hashing" if a.once else "sentence_transformer")
    run_once = make_run_once(a.datastore_config, backend, a.model, a.dim, a.batch)
    worker = Worker("phase3", run_once, interval=a.interval)
    worker.run_once() if a.once else worker.run_forever()


if __name__ == "__main__":
    main()
