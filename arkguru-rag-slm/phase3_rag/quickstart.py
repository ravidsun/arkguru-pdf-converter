"""
Phase 3 QUICKSTART -- test the RAG system on a couple of PDFs, no Postgres,
no fine-tuning required.

Flow:
  1. INGEST   chunk JSONL files (produced by arkguru-pdf-extraction /
              arkguru-web-scraping) into a local, incremental vector store.
              Re-run anytime with more files -- only new chunks are embedded.
  2. RETRIEVE hybrid: dense (vector store) + lexical (BM25), fused with RRF.
  3. ANSWER   via local Ollama if available; otherwise print the retrieved
              context (extractive) so you can see retrieval working with zero
              model setup.

Embedder backends:
  --embedder sentence_transformer   REAL (bge-m3) -- use on your NUC
  --embedder hashing                offline, no downloads -- for quick tests

Examples:
  # ingest two PDFs' chunks and ask one question (offline embedder)
  python -m phase3_rag.quickstart --add data/processed/pdf_chunks.jsonl \
      --embedder hashing --ask "What PPE is required before servicing a unit?"

  # later: add more, then chat (real embedder + Ollama on your machine)
  python -m phase3_rag.quickstart --add data/processed/new_batch.jsonl \
      --embedder sentence_transformer --model domain-slm --chat
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from common.schema import read_jsonl, read_parquet
from phase3_rag.embedder import Embedder
from phase3_rag.faithfulness import apply_gate
from phase3_rag.vector_store import LocalVectorStore


def _load_chunks(path: str):
    chunks = read_parquet(path) if path.endswith(".parquet") else read_jsonl(path)
    # retrieval matches on children; keep parents aside for context expansion
    return [c for c in chunks if not c.is_parent], {c.chunk_id: c for c in chunks if c.is_parent}


def ingest(store: LocalVectorStore, emb: Embedder, files: list[str]) -> None:
    for f in files:
        if not Path(f).exists():
            print(f"  ! missing: {f}"); continue
        children, _ = _load_chunks(f)
        metas = [{"chunk_id": c.chunk_id, "text": c.text, "section": c.section,
                  "source_id": c.source_id, "page": c.page, "url": c.url,
                  "parent_id": c.parent_id} for c in children]
        vecs = emb.encode([m["text"] for m in metas]) if metas else None
        added = store.add(metas, vecs) if metas else 0
        print(f"  {f}: {len(children)} chunks read, {added} new added "
              f"(store now {len(store)})")
    store.save()


def _rrf(runs, k=60):
    scores, rows = {}, {}
    for run in runs:
        for rank, (m, _s) in enumerate(run):
            cid = m["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            rows[cid] = m
    return [rows[c] for c in sorted(scores, key=scores.get, reverse=True)]


def retrieve(store, emb, query, top_k=5):
    qvec = emb.encode([query])[0]
    dense = store.search(qvec, k=20)
    # lexical BM25 over the same corpus
    try:
        from rank_bm25 import BM25Okapi
        corpus = [m["text"].lower().split() for m in store.meta]
        bm25 = BM25Okapi(corpus)
        sc = bm25.get_scores(query.lower().split())
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:20]
        lexical = [(store.meta[i], float(sc[i])) for i in order]
    except Exception:
        lexical = []
    fused = _rrf([dense, lexical]) if lexical else [m for m, _ in dense]
    return fused[:top_k]


def build_prompt(query, hits):
    ctx = "\n\n".join(f"[{h['source_id']} p.{h['page']}] {h['text']}" for h in hits)
    return ("You are a domain expert. Answer ONLY from the context and cite "
            f"[source_id p.page].\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}\n\nANSWER:")


def answer(query, hits, model=None):
    prompt = build_prompt(query, hits)
    if model:
        try:
            import requests
            r = requests.post("http://localhost:11434/api/generate",
                              json={"model": model, "prompt": prompt, "stream": False},
                              timeout=300)
            return apply_gate(r.json()["response"], hits)
        except Exception as e:
            print(f"  (Ollama unavailable: {e} -- showing retrieved context)")
    # extractive fallback: show what WOULD be sent to the model
    top = hits[0] if hits else None
    lead = f"[extractive] Top match: {top['text'][:300]}..." if top else "No matches."
    return lead


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 3 quickstart RAG demo")
    ap.add_argument("--store", default="data/store/index")
    ap.add_argument("--add", nargs="*", default=[], help="chunk JSONL/Parquet files to ingest")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"], default="hashing")
    ap.add_argument("--model-name", default="BAAI/bge-m3")
    ap.add_argument("--ask", help="single question")
    ap.add_argument("--chat", action="store_true", help="interactive loop")
    ap.add_argument("--model", help="Ollama model tag for generation (optional)")
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args(argv)

    emb = Embedder(backend=a.embedder, model_name=a.model_name)
    store = LocalVectorStore(a.store)
    if a.add:
        print(f"Ingesting (embedder={a.embedder}, dim={emb.dim}) ...")
        ingest(store, emb, a.add)
    print(f"Store: {len(store)} chunks indexed at {a.store}\n")

    def handle(q):
        hits = retrieve(store, emb, q, a.top_k)
        print(f"\nQ: {q}")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. [{h['source_id']} p{h['page']}] {h['section']} :: {h['text'][:110]}...")
        print("\nA:", answer(q, hits, a.model), "\n" + "-" * 70)

    if a.ask:
        handle(a.ask)
    if a.chat:
        print("Interactive RAG. Ctrl-C to exit.")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                handle(q)
    if not a.ask and not a.chat and not a.add:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
