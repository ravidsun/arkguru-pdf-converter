"""
Phase 3, step 3: hybrid retrieval + cross-encoder rerank + parent expansion.

Backed by the TWO-table datastore (common/datastore.py):
  - dense search  -> chunk_embeddings (vectors), JOINed to chunks for text/meta
  - lexical search-> chunks.ts full-text (BM25-like)

Per query:
  1. Dense NN (top_k_vector) + lexical (top_k_bm25)
  2. Reciprocal Rank Fusion
  3. Cross-encoder rerank (bge-reranker) -> top_k_final
  4. Optional parent expansion (swap a matched child for its larger parent)

If a DSN is configured, `create_retriever()` raises on connect failure
(fail-closed). Hybrid retrieve is one SQL call (`ChunkStore.search_chunks` →
`search_chunks()`), then the same rerank + parent expansion. JSONL fallback
is only used when no DSN is set (offline smoke tests): `JsonlRetriever`
reads `**/*.jsonl` under data/processed/.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.retrieve")

# datastore search_* return columns, in order:
# chunk_id, text, section, source_id, page, url, parent_id, chunk_index, title, lang, score
_C = {"chunk_id": 0, "text": 1, "section": 2, "source_id": 3, "page": 4, "url": 5,
      "parent_id": 6, "chunk_index": 7, "title": 8, "lang": 9, "score": 10}


CONNECT_TIMEOUT_S = 15
_PARENT_CTX_CHARS_PER_TOKEN = 3


@dataclass
class Hit:
    chunk_id: str; text: str; score: float
    section: str = ""; source_id: str = ""; page: int | None = None; url: str = ""
    parent_id: str | None = None
    chunk_index: int | None = None


class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        from common.datastore_config import open_chunk_store
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.embedder = SentenceTransformer(cfg["embedding_model"], device="cpu")
        self.store = open_chunk_store(
            cfg.get("datastore_config"),
            dim=self.embedder.get_sentence_embedding_dimension())
        self.reranker = CrossEncoder(cfg["reranker_model"], device="cpu")
        from phase3_rag.rewrite import build_rewriter
        self.rewriter = build_rewriter(
            cfg.get("retrieval", {}).get("rewriter"), cfg)

    @staticmethod
    def _rrf(runs, k=60):
        from common.rrf import reciprocal_rank_fusion
        return reciprocal_rank_fusion(runs, k=k)

    def search(self, query: str, *, top_k_final: int | None = None) -> list[Hit]:
        rc = self.cfg["retrieval"]
        k_final = rc["top_k_final"] if top_k_final is None else top_k_final
        k_dense = rc["top_k_vector"]
        k_lexical = rc["top_k_bm25"]
        # The dense and lexical legs overlap only partially, so capping the SQL
        # LIMIT at max(k_dense, k_lexical) threw away candidates the fused set
        # had already found. The cross-encoder is the accurate stage, so give it
        # the whole union and let it sort; k_final then trims after reranking.
        k_candidates = rc.get("top_k_candidates", k_dense + k_lexical)

        variants = [query]
        rewriter = getattr(self, "rewriter", None)
        if rewriter is not None:
            variants = rewriter.rewrite(query)

        runs = []
        for v in variants:
            qvec = self.embedder.encode(v, normalize_embeddings=True).tolist()
            runs.append(self.store.search_chunks(
                v, qvec,
                k_dense=k_dense,
                k_lexical=k_lexical,
                k_final=k_candidates,
            ))
        # One variant is the common case and must stay byte-identical to before.
        fused = runs[0] if len(runs) == 1 else self._rrf(runs)
        if not fused:
            return []
        pairs = [(query, r[_C["text"]]) for r in fused]
        rr = self.reranker.predict(pairs)
        order = sorted(range(len(fused)), key=lambda i: rr[i], reverse=True)[: k_final]
        hits = []
        for i in order:
            r = fused[i]
            hits.append(Hit(
                chunk_id=r[_C["chunk_id"]], text=r[_C["text"]],
                section=r[_C["section"]] or "", source_id=r[_C["source_id"]] or "",
                page=r[_C["page"]], url=r[_C["url"]] or "", score=float(rr[i]),
                parent_id=r[_C["parent_id"]] or None,
                chunk_index=r[_C["chunk_index"]],
            ))
        return self._expand_parents(hits)

    def _expand_parents(self, hits: list[Hit]) -> list[Hit]:
        if not _parent_expansion_enabled(self.cfg):
            return hits
        pids = [h.parent_id for h in hits if h.parent_id]
        parents = self.store.fetch_by_ids(pids)
        return expand_hits_with_parents(hits, parents, max_chars=_parent_char_budget(self.cfg))


class JsonlRetriever:
    """
    Fallback retriever for when Postgres/pgvector is unavailable.

    Loads child chunks from ``**/*.jsonl`` under `processed_dir` (nested
    per-PDF folders and legacy flat files). Parent rows are kept for
    expansion, not indexed for search.
    """

    def __init__(self, cfg, processed_dir: str = "data/processed"):
        self.cfg = cfg
        self.processed_dir = Path(processed_dir)
        self._chunks: list[dict] = []
        self._parents: dict[str, dict] = {}
        self._vecs = None          # np.ndarray | None
        self._embedder = None
        self._reranker = None
        self._load_chunks()
        self._build_index()

    def _load_chunks(self):
        from common.schema import read_jsonl
        if not self.processed_dir.exists():
            log.warning("processed_dir %s does not exist – no chunks loaded", self.processed_dir)
            return
        jsonl_files = list_jsonl_files(self.processed_dir)
        if not jsonl_files:
            log.warning("No *.jsonl files found under %s", self.processed_dir)
            return
        for f in jsonl_files:
            try:
                chunks = read_jsonl(f)
                n_child = 0
                for c in chunks:
                    row = _chunk_row(c)
                    if c.is_parent:
                        self._parents[c.chunk_id] = {
                            "text": c.text,
                            "section": c.section or "",
                            "source_id": c.source_id,
                            "page": c.page,
                            "url": c.url or "",
                            "chunk_index": c.chunk_index,
                        }
                    else:
                        self._chunks.append(row)
                        n_child += 1
                log.info("JsonlRetriever: loaded %d child chunks from %s", n_child, f)
            except Exception as e:
                log.warning("Could not load %s: %s", f, e)
        log.info("JsonlRetriever: %d chunks total from %s", len(self._chunks), self.processed_dir)

    def _build_index(self):
        if not self._chunks:
            return
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer, CrossEncoder
            self._embedder = SentenceTransformer(self.cfg["embedding_model"], device="cpu")
            self._reranker = CrossEncoder(self.cfg["reranker_model"], device="cpu")
            texts = [c["text"] for c in self._chunks]
            self._vecs = self._embedder.encode(
                texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
            ).astype("float32")
            log.info("JsonlRetriever: pre-computed embeddings shape %s", self._vecs.shape)
        except Exception as e:
            log.warning("Could not load sentence-transformer models (%s) – BM25 only", e)

    @staticmethod
    def _rrf(runs, k=60):
        from common.rrf import reciprocal_rank_fusion
        return reciprocal_rank_fusion(runs, k=k)

    def search(self, query: str, *, top_k_final: int | None = None) -> list[Hit]:
        if not self._chunks:
            return []
        rc = self.cfg["retrieval"]
        top_k_vector = rc["top_k_vector"]
        top_k_bm25 = rc["top_k_bm25"]
        k_final = rc["top_k_final"] if top_k_final is None else top_k_final

        # --- dense ---
        dense: list[tuple] = []
        if self._vecs is not None and self._embedder is not None:
            import numpy as np
            qvec = self._embedder.encode(query, normalize_embeddings=True).astype("float32")
            sims = self._vecs @ qvec
            idx = int_top_k(sims, top_k_vector)
            dense = [(*_row(self._chunks[i]), float(sims[i])) for i in idx]

        # --- lexical BM25 ---
        lex: list[tuple] = []
        try:
            from rank_bm25 import BM25Okapi
            corpus = [c["text"].lower().split() for c in self._chunks]
            scores = BM25Okapi(corpus).get_scores(query.lower().split())
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_bm25]
            lex = [(*_row(self._chunks[i]), float(scores[i])) for i in idx]
        except Exception as e:
            log.debug("BM25 unavailable: %s", e)

        fused = self._rrf([dense, lex])[: max(top_k_vector, top_k_bm25)]
        if not fused:
            return []

        # --- rerank ---
        if self._reranker is not None:
            pairs = [(query, r[1]) for r in fused]
            rr = self._reranker.predict(pairs)
            order = sorted(range(len(fused)), key=lambda i: rr[i], reverse=True)
            top = [fused[i] for i in order[:k_final]]
            hits = [_hit_from_row(r, score=float(rr[order[j]])) for j, r in enumerate(top)]
        else:
            hits = [_hit_from_row(r, score=r[8]) for r in fused[:k_final]]
        return self._expand_parents(hits)

    def _expand_parents(self, hits: list[Hit]) -> list[Hit]:
        if not _parent_expansion_enabled(self.cfg):
            return hits
        return expand_hits_with_parents(
            hits, self._parents, max_chars=_parent_char_budget(self.cfg))


# --- helpers -----------------------------------------------------------------

def list_jsonl_files(processed_dir: Path) -> list[Path]:
    """All JSONL files under ``processed_dir``, including per-PDF subfolders."""
    if not processed_dir.exists():
        return []
    return sorted(processed_dir.rglob("*.jsonl"))


def _chunk_row(c) -> dict:
    return {
        "chunk_id": c.chunk_id, "text": c.text,
        "section": c.section or "", "source_id": c.source_id,
        "page": c.page, "url": c.url or "",
        "parent_id": c.parent_id or "",
        "chunk_index": c.chunk_index,
    }


def _row(c: dict) -> tuple:
    return (c["chunk_id"], c["text"], c["section"], c["source_id"], c["page"],
            c["url"], c.get("parent_id") or "", c.get("chunk_index"))


def _hit_from_row(r, *, score: float) -> Hit:
    return Hit(
        chunk_id=r[0], text=r[1], section=r[2], source_id=r[3],
        page=r[4], url=r[5], score=score, parent_id=r[6] or None,
        chunk_index=r[7],
    )


def _parent_expansion_enabled(cfg: dict) -> bool:
    return bool((cfg.get("retrieval") or {}).get("use_parent_expansion"))


def _parent_char_budget(cfg: dict) -> int:
    ctx = int((cfg.get("serve") or {}).get("ctx") or 4096)
    return max(ctx, ctx * _PARENT_CTX_CHARS_PER_TOKEN)


def expand_hits_with_parents(
    hits: list[Hit],
    parents: dict[str, dict],
    *,
    max_chars: int,
) -> list[Hit]:
    """Swap child text for parent text; collapse siblings; cap total chars."""
    if not hits:
        return []
    out: list[Hit] = []
    seen_parents: set[str] = set()
    total = 0
    for h in hits:
        pid = h.parent_id or ""
        parent = parents.get(pid) if pid else None
        if parent:
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            expanded = Hit(
                chunk_id=h.chunk_id,
                text=parent.get("text") or h.text,
                score=h.score,
                section=parent.get("section") or h.section,
                source_id=parent.get("source_id") or h.source_id,
                page=parent.get("page") if parent.get("page") is not None else h.page,
                url=parent.get("url") or h.url,
                parent_id=pid,
                chunk_index=parent.get("chunk_index", h.chunk_index),
            )
        else:
            expanded = h
        n = len(expanded.text or "")
        if out and total + n > max_chars:
            break
        out.append(expanded)
        total += n
    return out


def _probe_postgres(dsn: str) -> None:
    import psycopg
    conn = psycopg.connect(dsn, connect_timeout=CONNECT_TIMEOUT_S)
    conn.close()


def int_top_k(arr, k: int):
    """Return indices of the top-k values (descending) without full sort."""
    import numpy as np
    k = min(k, len(arr))
    idx = np.argpartition(-arr, k - 1)[:k]
    return idx[np.argsort(-arr[idx])]


def create_retriever(cfg, processed_dir: str = "data/processed"):
    """Return HybridRetriever when a DSN is set (raise if connect fails).

    JSONL fallback is used only when no DSN is configured.
    """
    from common.datastore_config import load_datastore_config, resolve_dsn
    ds_cfg = load_datastore_config(cfg.get("datastore_config"))
    dsn = resolve_dsn(ds_cfg.get("postgres", {}))
    if dsn:
        try:
            _probe_postgres(dsn)
        except Exception as e:
            raise RuntimeError(
                "Postgres DSN is set but the connection failed. Fix PG_DSN / "
                "network / sslmode=require (session pooler on port 5432). "
                f"Underlying error: {e}"
            ) from e
        log.info("Postgres available – using HybridRetriever")
        return HybridRetriever(cfg)
    log.warning("No Postgres DSN configured – falling back to JsonlRetriever")
    return JsonlRetriever(cfg, processed_dir)


if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("query")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["phase3"]
    r = create_retriever(cfg, a.processed_dir)
    for h in r.search(a.query):
        print(f"[{h.score:.3f}] {h.source_id} p{h.page} #{h.chunk_index} :: {h.section}\n  {h.text[:160]}...\n")
