"""
Phase 3: fill embeddings for chunks already in the Postgres datastore.

Phases 1/2 upsert chunks with no vectors. This script reads the rows that
still need vectors, embeds them locally (BAAI/bge-m3 by default), and writes
the vectors back IN PLACE with ``chunk_embeddings.model`` set to the real
embedder label.

Idempotent: re-running only touches rows that are still missing. ``--reembed``
also rewrites rows whose stored model is blank or different from ``--model``;
already-labelled rows are skipped, so an interrupted job resumes.

    # Windows PowerShell (RTX 3050 6 GB) — re-embed public with bge-m3 on CUDA:
    $env:PG_DSN = "postgresql://user:pass@localhost:5432/rag"
    python -m phase3_rag.embed_datastore --schema public --reembed --device cuda --model BAAI/bge-m3 --dim 1024

    # POSIX equivalent:
    export PG_DSN=postgresql://user:pass@localhost:5432/rag
    python -m phase3_rag.embed_datastore --schema public --reembed --device cuda --model BAAI/bge-m3 --dim 1024

Hashing is refused for this datastore path unless ``--allow-hashing`` is set.
Zero / NaN / Inf vectors are never stored (skipped and counted).
"""
from __future__ import annotations

import argparse
import logging
import sys

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.embed_datastore")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fill or re-embed chunk_embeddings in the datastore")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"],
                    default=DEFAULT_BACKEND,
                    help="Production default is sentence_transformer. "
                         "hashing requires --allow-hashing.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                    help="Encode batch size (default 16, sized for a 6 GB GPU).")
    ap.add_argument("--device", default="auto",
                    help="auto (CUDA if available, else CPU), cpu, or cuda")
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    ap.add_argument("--schema", default=None,
                    help="Postgres schema (overrides datastore.yaml).")
    ap.add_argument("--reembed", action="store_true",
                    help="Rewrite rows whose model is missing or differs from "
                         "--model. Already-matching rows are skipped (resumable).")
    ap.add_argument("--allow-hashing", action="store_true",
                    help="Permit hashing writes to Postgres (tests/dev only).")
    a = ap.parse_args(argv)

    try:
        assert_datastore_backend_allowed(a.embedder, allow_hashing=a.allow_hashing)
    except HashingDatastoreForbidden as e:
        raise SystemExit(str(e)) from e

    emb = Embedder(
        backend=a.embedder,
        model_name=a.model,
        dim=a.dim,
        device=a.device,
        max_seq_length=a.max_seq_length,
        batch_size=a.batch,
    )
    store = open_chunk_store(a.datastore_config, dim=emb.dim, schema=a.schema)
    todo = store.count_embeddings_to_fill(model=emb.label, reembed=a.reembed)
    log.info(
        "chunks needing embeddings: %d (embedder=%s model=%s dim=%d "
        "device=%s batch=%d reembed=%s schema=%s)",
        todo, a.embedder, emb.label, emb.dim, emb.device, a.batch,
        a.reembed, store.schema,
    )

    done = 0
    wrote_total = 0
    skipped_total = 0
    for batch in store.iter_chunks_for_embedding(
            batch=a.batch, model=emb.label, reembed=a.reembed):
        ids = [cid for cid, _ in batch]
        texts = [txt for _, txt in batch]
        wrote, skipped = write_embeddings(store, ids, texts, emb)
        done += len(ids)
        wrote_total += wrote
        skipped_total += len(skipped)
        pct = (100.0 * done / todo) if todo else 100.0
        log.info(
            "  %d / %d (%.1f%%) wrote=%d skipped_invalid=%d",
            done, todo, pct, wrote_total, skipped_total,
        )
    remaining = store.count_embeddings_to_fill(model=emb.label, reembed=a.reembed)
    log.info(
        "done. wrote=%d skipped_invalid=%d remaining=%d",
        wrote_total, skipped_total, remaining,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
