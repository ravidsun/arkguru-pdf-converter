"""
Phase 3, step 0: combine Phase 1 + Phase 2 outputs into training data.

Reads any number of JSONL/Parquet chunk files (the combined corpus) and produces
an instruction-tuning JSONL for QLoRA. Two generation modes:

  - "self_supervised" : template Q&A from chunk sections (no LLM needed). Fast,
                        deterministic, good for domain vocabulary adaptation.
  - "synthetic_qa"    : call a local model (Ollama) to draft Q&A per chunk.
                        Higher quality; requires Ollama running. [hook provided]

Also emits `corpus.jsonl` = the flat, deduped retrieval corpus used by index.py.
"""
from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path
from common.schema import read_jsonl, read_parquet, Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.prepare")


def _load_any(path: str) -> list[Chunk]:
    return read_parquet(path) if path.endswith(".parquet") else read_jsonl(path)


def load_combined(paths: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in paths:
        if Path(p).exists():
            c = _load_any(p); chunks.extend(c)
            log.info("loaded %d from %s", len(c), p)
        else:
            log.warning("missing input (skipped): %s", p)
    # global exact-dedup across sources
    seen, out = set(), []
    for c in chunks:
        if c.is_parent:
            out.append(c); continue
        k = c.text.strip()
        if k not in seen:
            seen.add(k); out.append(c)
    log.info("combined corpus: %d chunks (deduped)", len(out))
    return out


def make_self_supervised(chunks: list[Chunk]) -> list[dict]:
    """Template instruction pairs grounded in each chunk's section/source."""
    rows = []
    for c in chunks:
        if c.is_parent:
            continue
        src = c.section or c.title or "the source material"
        rows.append({
            "instruction": f"Based on the documentation, explain: {src}.",
            "input": "",
            "output": c.text,
            "meta": {"chunk_id": c.chunk_id, "source_id": c.source_id,
                     "page": c.page, "url": c.url},
        })
    return rows


def make_synthetic_qa(chunks: list[Chunk], model: str = "domain-slm") -> list[dict]:
    """Draft Q&A per chunk with a local Ollama model. HOOK: fill in the call."""
    import requests
    rows = []
    for c in chunks:
        if c.is_parent:
            continue
        prompt = ("Write ONE question a user might ask that is fully answered by "
                  "the text below, then answer it using only that text.\n\n"
                  f"TEXT:\n{c.text}\n\nReturn JSON {{\"q\":..., \"a\":...}}.")
        try:
            r = requests.post("http://localhost:11434/api/generate",
                              json={"model": model, "prompt": prompt, "stream": False},
                              timeout=120)
            qa = json.loads(r.json()["response"])
            rows.append({"instruction": qa["q"], "input": "", "output": qa["a"],
                         "meta": {"chunk_id": c.chunk_id}})
        except Exception as e:
            log.warning("synthetic QA failed for %s: %s", c.chunk_id, e)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-train", default="data/processed/train.jsonl")
    ap.add_argument("--out-corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--mode", choices=["self_supervised", "synthetic_qa"],
                    default="self_supervised")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args(argv)

    chunks = load_combined(args.inputs)
    from common.schema import write_jsonl
    write_jsonl(chunks, args.out_corpus)
    log.info("wrote retrieval corpus -> %s", args.out_corpus)

    rows = (make_synthetic_qa(chunks) if args.mode == "synthetic_qa"
            else make_self_supervised(chunks))
    random.Random(0).shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val, train = rows[:n_val], rows[n_val:]
    for name, data in [("train", train), ("val", val)]:
        p = Path(args.out_train).with_name(f"{name}.jsonl")
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data))
        log.info("wrote %d %s rows -> %s", len(data), name, p)


if __name__ == "__main__":
    main()
