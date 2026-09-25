"""
One-command wrapper: PDFs -> Phase 1 extraction -> Phase 3 ingest -> ask.

Saves you from manually copying JSONL between the two phase folders. It shells
out to Phase 1 (arkguru-pdf-extraction) to turn a folder of PDFs into chunks,
drops the result into this folder's data/processed/, then ingests it into the
local incremental vector store and (optionally) answers a question.

Assumes both folders live in this repo:
    arkguru-pdf-converter/
      arkguru-pdf-extraction/
      arkguru-rag-slm/           <- you run from here
Override the Phase 1 location with --phase1-repo if they live elsewhere.

Examples:
    # offline embedder, just prove retrieval on a couple of PDFs
    python -m phase3_rag.run_pdfs --pdfs ~/my_pdfs --embedder hashing \
        --ask "What PPE is required before servicing a unit?"

    # on your NUC: real embeddings + Ollama answer, interactive
    python -m phase3_rag.run_pdfs --pdfs ~/my_pdfs \
        --embedder sentence_transformer --model domain-slm --chat

    # add more PDFs later -> only new chunks get embedded (idempotent)
    python -m phase3_rag.run_pdfs --pdfs ~/more_pdfs --embedder hashing

    # Postgres handoff: Phase 1 --sink postgres then fill chunk_embeddings
    python -m phase3_rag.run_pdfs --pdfs ~/my_pdfs --sink postgres --embedder hashing
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

from phase3_rag.embedder import Embedder
from phase3_rag.vector_store import LocalVectorStore
from phase3_rag import quickstart

# datastore search_* columns (common.datastore._HIT_COLS + score)
_HIT = {"chunk_id": 0, "text": 1, "section": 2, "source_id": 3, "page": 4,
        "url": 5, "parent_id": 6, "chunk_index": 7, "title": 8, "lang": 9,
        "score": 10}

def _find_phase1(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("PHASE1_REPO")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parents[1]           # arkguru-rag-slm/
    candidates += [here.parent / "arkguru-pdf-extraction",
                   here.parent / "phase1-pdf-extraction"]
    for c in candidates:
        if c and (c / "phase1_pdf" / "pipeline.py").exists():
            return c
    raise SystemExit(
        "Could not locate Phase 1 (arkguru-pdf-extraction).\n"
        "Pass --phase1-repo /path/to/arkguru-pdf-extraction or set PHASE1_REPO.")


def collect_processed_jsonl(out_dir: Path, pdfs: Path) -> list[Path]:
    """JSONL for each ingested PDF: nested per-PDF folder and/or legacy flat file."""
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        found.append(path)

    for pdf in sorted(pdfs.rglob("*.pdf")):
        try:
            rel_stem = pdf.relative_to(pdfs).with_suffix("")
        except ValueError:
            rel_stem = Path(pdf.stem)
        nested = out_dir / rel_stem
        if nested.is_dir():
            for p in sorted(nested.rglob("*.jsonl")):
                _add(p)
        _add(out_dir / f"{pdf.stem}.jsonl")
    return found


def run_phase1(phase1_repo: Path, pdfs: Path, out_dir: Path, strategy: str,
               sink: str = "file") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "phase1_pdf.pipeline",
           "--input", str(pdfs.resolve()),
           "--out-dir", str(out_dir.resolve()),
           "--strategy", strategy]
    if sink == "postgres":
        cmd += ["--sink", "postgres"]
    print(f"[phase1] {' '.join(cmd)}  (cwd={phase1_repo})")
    env = dict(os.environ)
    common_root = Path(__file__).resolve().parents[2] / "arkguru-common"
    parts = [str(phase1_repo), str(common_root)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    r = subprocess.run(cmd, cwd=str(phase1_repo), env=env)
    if r.returncode != 0:
        raise SystemExit(f"Phase 1 failed (exit {r.returncode}).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PDFs -> Phase 1 -> Phase 3 ingest, in one command")
    ap.add_argument("--pdfs", required=True, help="folder of PDF files")
    ap.add_argument("--phase1-repo", help="path to arkguru-pdf-extraction (auto-detected next to this folder)")
    ap.add_argument("--strategy", default="parent_child",
                    choices=["structure", "parent_child", "semantic"])
    ap.add_argument("--store", default="data/store/index")
    ap.add_argument("--embedder", choices=["sentence_transformer", "hashing"], default="hashing")
    ap.add_argument("--model-name", default="BAAI/bge-m3")
    ap.add_argument("--ask")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--model", help="Ollama model tag for generated answers (optional)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sink", choices=["file", "postgres"], default="file",
                    help="file: local JSONL + LocalVectorStore. "
                         "postgres: Phase 1 --sink postgres then embed_datastore.")
    ap.add_argument("--dim", type=int, default=1024)
    a = ap.parse_args(argv)

    pdfs = Path(a.pdfs)
    if not pdfs.exists():
        raise SystemExit(f"--pdfs folder not found: {pdfs}")
    if a.sink == "postgres" and not os.environ.get("PG_DSN"):
        raise SystemExit("--sink postgres requires PG_DSN")
    phase1_repo = _find_phase1(a.phase1_repo)

    out_dir = Path("data/processed")
    run_phase1(phase1_repo, pdfs, out_dir, a.strategy, sink=a.sink)

    if a.sink == "postgres":
        return _run_postgres_embed(a)

    outputs = [str(p) for p in collect_processed_jsonl(out_dir, pdfs)]
    if not outputs:
        raise SystemExit(
            f"Phase 1 produced no JSONL files in {out_dir.resolve()} "
            f"matching PDFs under {pdfs}."
        )

    emb = Embedder(backend=a.embedder, model_name=a.model_name)
    store = LocalVectorStore(a.store)
    print(f"[phase3] ingesting {len(outputs)} file(s) "
          f"(embedder={a.embedder}, dim={emb.dim}) ...")
    quickstart.ingest(store, emb, outputs)
    print(f"[phase3] store now holds {len(store)} chunks at {a.store}\n")

    if a.ask:
        hits = quickstart.retrieve(store, emb, a.ask, a.top_k)
        print(f"Q: {a.ask}")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. [{h['source_id']} p{h['page']}] {h['section']} :: {h['text'][:110]}...")
        print("\nA:", quickstart.answer(a.ask, hits, a.model))
    if a.chat:
        quickstart.main(["--store", a.store, "--embedder", a.embedder,
                         "--model-name", a.model_name, "--top-k", str(a.top_k),
                         "--chat"] + (["--model", a.model] if a.model else []))
    return 0


def _run_postgres_embed(a) -> int:
    """Fill chunk_embeddings for chunks already upserted by Phase 1."""
    from phase3_rag.embed_datastore import main as embed_main

    print(f"[phase3] filling chunk_embeddings "
          f"(embedder={a.embedder}, dim={a.dim}) ...")
    embed_main(["--embedder", a.embedder, "--model", a.model_name,
                "--dim", str(a.dim)])

    if a.ask:
        _ask_postgres(a)
    if a.chat:
        raise SystemExit(
            "--chat with --sink postgres is not supported; use "
            "`python -m phase3_rag.serve` against the datastore.")
    return 0


def _ask_postgres(a) -> None:
    from common.datastore_config import open_chunk_store

    emb = Embedder(backend=a.embedder, model_name=a.model_name, dim=a.dim)
    store = open_chunk_store(dim=emb.dim)
    qvec = emb.encode([a.ask])[0].tolist()
    rows = store.search_dense(qvec, k=a.top_k)
    print(f"Q: {a.ask}")
    hits = []
    for i, r in enumerate(rows, 1):
        text = r[_HIT["text"]] or ""
        hits.append({"text": text, "source_id": r[_HIT["source_id"]],
                     "page": r[_HIT["page"]], "section": r[_HIT["section"]]})
        print(f"  {i}. [{r[_HIT['source_id']]} p{r[_HIT['page']]} "
              f"#{r[_HIT['chunk_index']]}] {r[_HIT['section']]} :: {text[:110]}...")
    print("\nA:", quickstart.answer(a.ask, hits, a.model))


if __name__ == "__main__":
    sys.exit(main())
