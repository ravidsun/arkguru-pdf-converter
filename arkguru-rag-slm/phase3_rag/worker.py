"""
Phase 3 autonomous worker: keep the vectors table in sync with the chunks table.

Deterministic: the trigger is "there are chunks without embeddings." Each cycle
polls the datastore; when Phase 1/2 have added new chunks, it embeds only those
(idempotent -- fills the separate chunk_embeddings table in place).

    python -m phase3_rag.worker --once
    python -m phase3_rag.worker --interval 60
    python -m phase3_rag.worker --embedder hashing --allow-hashing --once
    python -m phase3_rag.worker --embedder sentence_transformer --model BAAI/bge-m3
"""
from __future__ import annotations

import argparse
import logging

from common.worker import Worker
from common.datastore_config import open_chunk_store
from phase3_rag.embedder import (
    DEFAULT_BACKEND,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIM,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL,
    Embedder,
    HashingDatastoreForbidden,
    assert_datastore_backend_allowed,
    write_embeddings,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("phase3.worker")


def make_run_once(datastore_config, embedder_backend, model, dim, batch,
                  device="auto", max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                  schema=None):
    emb = Embedder(
        backend=embedder_backend,
        model_name=model,
        dim=dim,
        device=device,
        max_seq_length=max_seq_length,
        batch_size=batch,
    )

    def run_once():
        store = open_chunk_store(datastore_config, dim=emb.dim, schema=schema)
        store.ensure_schema()
        todo = store.count_embeddings_to_fill(reembed=False)
        if not todo:
            return "nothing to embed"
        done = 0
        skipped_total = 0
        for b in store.iter_chunks_for_embedding(batch=batch, reembed=False):
            wrote, skipped = write_embeddings(
                store, [x[0] for x in b], [x[1] for x in b], emb)
            done += wrote
            skipped_total += len(skipped)
        return f"embedded {done} chunk(s), skipped_invalid={skipped_total}"
    return run_once


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 3 embedding worker")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"],
                    default=DEFAULT_BACKEND,
                    help="Default: sentence_transformer (including --once). "
                         "hashing requires --allow-hashing.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--device", default="auto",
                    help="auto (CUDA if available, else CPU), cpu, or cuda")
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--allow-hashing", action="store_true",
                    help="Permit hashing writes to Postgres (tests/dev only).")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)

    try:
        assert_datastore_backend_allowed(a.embedder, allow_hashing=a.allow_hashing)
    except HashingDatastoreForbidden as e:
        raise SystemExit(str(e)) from e

    run_once = make_run_once(
        a.datastore_config, a.embedder, a.model, a.dim, a.batch,
        device=a.device, max_seq_length=a.max_seq_length, schema=a.schema,
    )
    worker = Worker("phase3", run_once, interval=a.interval)
    worker.run_once() if a.once else worker.run_forever()


if __name__ == "__main__":
    main()
