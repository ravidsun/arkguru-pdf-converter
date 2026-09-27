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
    python -m phase3_rag.index --embed-only
    python -m phase3_rag.index --embed-only --embedder hashing --allow-hashing
"""
from __future__ import annotations
import argparse, logging
from phase3_rag.embedder import (
    DEFAULT_BACKEND,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LENGTH,
    Embedder,
    HashingDatastoreForbidden,
    assert_datastore_backend_allowed,
    write_embeddings,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.index")


def _fill_missing(store, emb, batch: int = DEFAULT_BATCH_SIZE) -> tuple[int, int]:
    todo = store.count_embeddings_to_fill(reembed=False)
    log.info("chunks needing embeddings: %d (embedder=%s model=%s dim=%d)",
             todo, emb.backend, emb.label, emb.dim)
    done = 0
    skipped_total = 0
    for rows in store.iter_chunks_for_embedding(batch=batch, reembed=False):
        wrote, skipped = write_embeddings(
            store, [cid for cid, _ in rows], [txt for _, txt in rows], emb)
        done += wrote
        skipped_total += len(skipped)
        log.info("  embedded %d / %d skipped_invalid=%d", done, todo, skipped_total)
    return done, skipped_total


def build(cfg, corpus_path, *, embed_only=False, embedder=DEFAULT_BACKEND,
          allow_hashing=False, device=None, batch=DEFAULT_BATCH_SIZE,
          max_seq_length=DEFAULT_MAX_SEQ_LENGTH):
    from common.schema import read_jsonl
    from common.datastore_config import open_chunk_store

    try:
        assert_datastore_backend_allowed(embedder, allow_hashing=allow_hashing)
        emb = Embedder(
            backend=embedder,
            model_name=cfg["embedding_model"],
            device=device or cfg.get("embedding_device"),
            max_seq_length=max_seq_length,
            batch_size=batch,
        )
        store = open_chunk_store(cfg.get("datastore_config"), dim=emb.dim)
        store.ensure_schema()
    except HashingDatastoreForbidden:
        raise
    except Exception as e:
        log.warning("Cannot initialize datastore (%s) – skipping pgvector index build. "
                    "Retrieval will fall back to JsonlRetriever on data/processed/.", e)
        return

    if embed_only:
        log.info("embed-only: filling missing vectors in '%s' (skip JSONL upsert)",
                 store.vectors)
        done, skipped = _fill_missing(store, emb, batch=batch)
        log.info("done. chunks=%d  vectors=%d  newly_embedded=%d skipped_invalid=%d",
                 store.count(), store.count_vectors(), done, skipped)
        return

    chunks = [c for c in read_jsonl(corpus_path) if not c.is_parent]

    log.info("upserting %d chunks into '%s'", len(chunks), store.chunks)
    store.upsert(chunks)

    log.info("embedding with %s (dim=%d device=%s) into '%s'",
             emb.label, emb.dim, emb.device, store.vectors)
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    wrote, skipped = write_embeddings(store, ids, texts, emb)
    log.info("done. chunks=%d  vectors=%d wrote=%d skipped_invalid=%d",
             store.count(), store.count_vectors(), wrote, len(skipped))


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--embed-only", action="store_true",
                    help="skip JSONL upsert; embed chunks already in the datastore "
                         "(same as phase3_rag.embed_datastore)")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"],
                    default=DEFAULT_BACKEND,
                    help="Default: sentence_transformer. hashing requires --allow-hashing.")
    ap.add_argument("--allow-hashing", action="store_true",
                    help="Permit hashing writes to Postgres (tests/dev only).")
    ap.add_argument("--device", default=None,
                    help="auto|cpu|cuda (default: config embedding_device, else auto)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    a = ap.parse_args(argv)
    build(yaml.safe_load(open(a.config))["phase3"], a.corpus,
          embed_only=a.embed_only, embedder=a.embedder,
          allow_hashing=a.allow_hashing, device=a.device,
          batch=a.batch, max_seq_length=a.max_seq_length)


if __name__ == "__main__":
    main()
