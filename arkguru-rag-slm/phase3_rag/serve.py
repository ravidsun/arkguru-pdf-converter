"""
Phase 3, step 4: local RAG answering over your fine-tuned model. No external APIs.

Two modes:
  - one-shot:   python -m phase3_rag.serve --ask "your question"
  - chat loop:  python -m phase3_rag.serve

Generation backend = Ollama. If Ollama is running, the answer is generated and
grounded in the retrieved context with citations, then checked by a lexical
faithfulness gate (see phase3_rag.faithfulness). If Ollama is NOT reachable, it
degrades gracefully to an *extractive* answer (the top retrieved passage), so the
pipeline still produces useful output offline. Retrieval uses Postgres when
``PG_DSN`` is set (fail-closed on connect errors) and JSONL only when no DSN
is configured (see retrieve.create_retriever).

To register a fine-tuned model with Ollama:
  1. Merge adapter:   python -m phase3_rag.merge_and_export   (see finetune header)
  2. Convert to GGUF: llama.cpp/convert_hf_to_gguf.py ./merged --outfile domain.gguf
  3. Quantize:        llama-quantize domain.gguf domain-Q4_K_M.gguf Q4_K_M
  4. Modelfile:       printf 'FROM ./domain-Q4_K_M.gguf' > Modelfile
  5. ollama create domain-slm -f Modelfile
"""
from __future__ import annotations
import argparse, logging, yaml
from phase3_rag.faithfulness import run_gated_answer
from phase3_rag.retrieve import create_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.serve")

SYS = ("You are a domain expert assistant. Answer ONLY from the provided context. "
       "Cite sources as [source_id p.page #chunk_index]. If the context is insufficient, say so.")


def _cite(h) -> str:
    idx = f" #{h.chunk_index}" if getattr(h, "chunk_index", None) is not None else ""
    return f"{h.source_id} p.{h.page}{idx}"


def build_prompt(query, hits):
    ctx = "\n\n".join(f"[{_cite(h)}] {h.text}" for h in hits)
    return f"{SYS}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}\n\nANSWER:"


def ollama_up(host: str = "http://localhost:11434") -> bool:
    try:
        import requests
        requests.get(f"{host}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def ask_ollama(model, prompt, ctx=4096, host="http://localhost:11434"):
    import requests
    r = requests.post(f"{host}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"num_ctx": ctx}}, timeout=300)
    return r.json()["response"]


def answer(retr, model, query, ctx=4096, faithfulness=None):
    """Return (answer_text, hits). Generated via Ollama if up, else extractive.

    Generated answers pass a lexical faithfulness gate: each sentence must
    overlap retrieved chunk text. Failures retry once with a wider top-k,
    then refuse rather than return an ungrounded answer.
    """
    def generate(hits):
        if ollama_up():
            try:
                return ask_ollama(model, build_prompt(query, hits), ctx)
            except Exception as e:
                log.warning("Ollama call failed (%s); using extractive answer", e)
        else:
            log.info("Ollama not reachable; returning extractive answer "
                     "(start Ollama for a generated response).")
        top = hits[0]
        return f"[extractive] {top.text}"

    return run_gated_answer(retr, query, generate, faithfulness)


def _print(query, ans, hits):
    print(f"\nQ: {query}\nassistant> {ans}")
    print("  sources:", ", ".join(_cite(h) for h in hits), "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--processed-dir", default="data/processed",
                    help="fallback JSONL directory if Postgres is unavailable")
    ap.add_argument("--ask", help="answer a single question and exit")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["phase3"]
    retr = create_retriever(cfg, a.processed_dir)
    model = cfg["serve"]["model_tag"]
    cx = cfg["serve"]["ctx"]
    faith = cfg["serve"].get("faithfulness")

    if a.ask:
        ans, hits = answer(retr, model, a.ask, cx, faith)
        _print(a.ask, ans, hits)
        return

    print(f"Local RAG chat over '{model}'. Ctrl-C to exit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            ans, hits = answer(retr, model, q, cx, faith)
            _print(q, ans, hits)


if __name__ == "__main__":
    main()
