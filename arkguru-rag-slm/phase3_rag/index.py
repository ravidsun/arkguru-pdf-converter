"""
Phase 3, step 2: build the hybrid retrieval index over the corpus.

Two separate Postgres tables (see common/datastore.py):
  - chunks            : text + metadata (source of truth) + full-text (tsvector)
  - chunk_embeddings  : the vectors only, keyed by chunk_id, HNSW index

This script can upsert a corpus JSONL (from Phase 1/2) then embed, OR skip the
JSONL path when chunks are already in the datastore (Phases 1/2 ran with
`--sink postgres`) and only fill missing vectors — same as
`phase3_rag.embed_datastore`.

    export PG_DSN=postgresql://user:pass@localhost:5432/rag
    python -m phase3_rag.index --config config/config.yaml --corpus data/processed/corpus.jsonl
    python -m phase3_rag.index --embed-only --embedder hashing
"""
from __future__ import annotations
import argparse, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.index")


def _fill_missing(store, emb, model: str, batch: int = 128) -> int:
    todo = store.count(only_missing_embedding=True)
    log.info("chunks needing embeddings: %d (embedder=%s dim=%d)",
             todo, emb.backend, emb.dim)
    done = 0
    for rows in store.iter_missing_embeddings(batch=batch):
        ids = [cid for cid, _ in rows]
        vecs = emb.encode([txt for _, txt in rows])
        done += store.update_embeddings(ids, vecs, model=model)
        log.info("  embedded %d / %d", done, todo)
    return done


def build(cfg, corpus_path, *, embed_only=False, embedder="sentence_transformer"):
    from common.schema import read_jsonl
    from common.datastore_config import open_chunk_store
    from phase3_rag.embedder import Embedder

    try:
        emb = Embedder(backend=embedder, model_name=cfg["embedding_model"])
        store = open_chunk_store(cfg.get("datastore_config"), dim=emb.dim)
        store.ensure_schema()
    except Exception as e:
        log.warning("Cannot initialize datastore (%s) – skipping pgvector index build. "
                    "Retrieval will fall back to JsonlRetriever on data/processed/.", e)
        return

    if embed_only:
        log.info("embed-only: filling missing vectors in '%s' (skip JSONL upsert)",
                 store.vectors)
        done = _fill_missing(store, emb, cfg["embedding_model"])
        log.info("done. chunks=%d  vectors=%d  newly_embedded=%d",
                 store.count(), store.count_vectors(), done)
        return

    chunks = [c for c in read_jsonl(corpus_path) if not c.is_parent]

    log.info("upserting %d chunks into '%s'", len(chunks), store.chunks)
    store.upsert(chunks)

    log.info("embedding with %s (dim=%d) into '%s'", cfg["embedding_model"], emb.dim, store.vectors)
    ids = [c.chunk_id for c in chunks]
    vecs = emb.encode([c.text for c in chunks])
    store.update_embeddings(ids, vecs, model=cfg["embedding_model"])
    log.info("done. chunks=%d  vectors=%d", store.count(), store.count_vectors())


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--embed-only", action="store_true",
                    help="skip JSONL upsert; embed chunks already in the datastore "
                         "(same as phase3_rag.embed_datastore)")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"],
                    default="sentence_transformer",
                    help="hashing is Cloud-safe; sentence_transformer is the NUC default")
    a = ap.parse_args(argv)
    build(yaml.safe_load(open(a.config))["phase3"], a.corpus,
          embed_only=a.embed_only, embedder=a.embedder)


if __name__ == "__main__":
    main()
