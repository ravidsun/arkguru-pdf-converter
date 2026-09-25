"""
Phase 3: fill embeddings for chunks already in the Postgres datastore.

Phases 1/2 upsert chunks with embedding=NULL. This script reads the rows that
still need vectors, embeds them locally (bge-m3 by default), and writes the
vectors back IN PLACE. Idempotent: re-running only touches rows that are still
NULL, so it resumes cleanly and supports the iterative "add more PDFs later"
workflow.

    export PG_DSN=postgresql://user:pass@localhost:5432/rag
    python -m phase3_rag.embed_datastore --model BAAI/bge-m3 --dim 1024
    # Cloud-safe (no sentence-transformers download):
    python -m phase3_rag.embed_datastore --embedder hashing --dim 1024

After this, retrieve.py / serve.py query the same table for hybrid search.
"""
from __future__ import annotations
import argparse, logging
from common.datastore_config import open_chunk_store
from phase3_rag.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.embed_datastore")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fill missing embeddings in the datastore")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"],
                    default="sentence_transformer")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args(argv)

    emb = Embedder(backend=a.embedder, model_name=a.model, dim=a.dim)
    store = open_chunk_store(a.datastore_config, dim=emb.dim)
    todo = store.count(only_missing_embedding=True)
    log.info("chunks needing embeddings: %d (embedder=%s dim=%d)", todo, a.embedder, emb.dim)

    done = 0
    for batch in store.iter_missing_embeddings(batch=a.batch):
        ids = [cid for cid, _ in batch]
        vecs = emb.encode([txt for _, txt in batch])
        done += store.update_embeddings(ids, vecs)
        log.info("  embedded %d / %d", done, todo)
    log.info("done. %d rows now have embeddings; %d still missing",
             done, store.count(only_missing_embedding=True))


if __name__ == "__main__":
    main()
