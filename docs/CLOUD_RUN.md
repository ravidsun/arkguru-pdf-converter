# Cloud run (Cursor Cloud Agent)

Run the 3-phase arkguru pipeline inside a **Cursor Cloud Agent** VM. This path is for agents (and humans using Cloud Agents) that start from `.cursor/environment.json` in this repository.

The companion document is [LOCAL_RUN.md](LOCAL_RUN.md).

## How the environment is defined

This repo is **repository-managed**. A committed `.cursor/environment.json` is the source of truth and overrides dashboard personal/team environments:

```json
{
  "name": "arkguru RAG dev",
  "install": "bash scripts/setup_dev_env.sh",
  "repositoryDependencies": [
    "https://github.com/ravidsun/arkguru-pdf-extraction",
    "https://github.com/ravidsun/arkguru-web-scraping",
    "https://github.com/ravidsun/arkguru-rag-slm"
  ]
}
```

| Field | Role |
|---|---|
| `install` | After checkout, creates `.venv`, installs vendored `arkguru-common`, and installs phase dependencies. Idempotent. Does **not** start servers. |
| `repositoryDependencies` | Puts the three phase repos in the Cloud Agent GitHub token scope so they can be checked out as siblings of this repo. |

There is no `start` command and no `terminals` entry: the pipeline is CLI-driven. Long-running workers (`make worker`, `orchestrator.py`) are started on demand.

Changes to `.cursor/environment.json` apply to **newly started** agents, not an already-running session.

## What a Cloud Agent sees on disk

Typical layout after checkout + `install`:

```
/agent/repos/
  arkguru-pdf-converter/     # this repo; contains arkguru-common + .venv
  arkguru-pdf-extraction/
  arkguru-web-scraping/
  arkguru-rag-slm/
```

`scripts/setup_dev_env.sh` looks for each phase repo as a sibling (`../arkguru-pdf-extraction`) or nested under this repo. If a phase is missing, install logs a warning and continues.

Activate the venv created by `install`:

```bash
source /agent/repos/arkguru-pdf-converter/.venv/bin/activate
# or, from this repo root:
source .venv/bin/activate
```

## What is installed (and what is not)

The Cloud `install` script matches the **offline** local bootstrap:

- `arkguru-common` (schema, tokenizer, chunking, datastore, worker, RRF) + pytest
- Phase 1: pymupdf / pymupdf4llm, tiktoken, reportlab (sample PDF)
- Phase 2: httpx, trafilatura, datasketch, …
- Phase 3 core only: numpy + rank-bm25

**Not** installed by default (keep the image lean; no multi-GB model downloads):

- `sentence-transformers`, transformers, peft, ragas, llama-index, FlagEmbedding
- System OCR: tesseract, ghostscript
- Postgres / pgvector
- Ollama / llama.cpp

Retrieval still works: `--embedder hashing` plus BM25, with extractive answers when Ollama is absent.

To add the full Phase 3 stack in a Cloud session:

```bash
source .venv/bin/activate
pip install -r ../arkguru-rag-slm/requirements.txt
```

That is slow and network-heavy; only do it when you need real embeddings or generation.

## Smoke test in a Cloud Agent

```bash
source .venv/bin/activate

# Shared package
pytest arkguru-common/tests -q

# Phase 1
cd ../arkguru-pdf-extraction
python scripts/make_sample_pdf.py
make phase1

# Phase 2 against a tiny local site (no external crawl)
mkdir -p /tmp/arkguru-demo-site
printf '<!DOCTYPE html><html><head><title>Demo</title></head><body><h1>Safety</h1><p>PPE includes insulated gloves rated to 1000V and safety eyewear. Lockout-tagout is mandatory before servicing any unit.</p></body></html>' \
  > /tmp/arkguru-demo-site/index.html
python3 -m http.server 8899 --directory /tmp/arkguru-demo-site &
cd ../arkguru-web-scraping
python -m phase2_web.pipeline --seeds http://127.0.0.1:8899/index.html --max-pages 5

# Phase 3 end-to-end (offline orchestrator)
cd ../arkguru-rag-slm
make e2e
```

`make e2e` should ingest the sample PDF and answer *"What does error code E14 mean?"* from the JSONL corpus (BM25 + extractive fallback).

One-shot RAG without the orchestrator:

```bash
cd ../arkguru-rag-slm
python -m phase3_rag.run_pdfs \
  --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
  --embedder hashing \
  --ask "What PPE is required before servicing a unit?"
```

## Cloud-specific constraints

| Topic | Behavior in Cloud Agents |
|---|---|
| GPU / Intel NUC features | Not available. Stay on hashing embeddings and extractive answers unless you explicitly install the heavy stack. |
| Ollama | Not started by `install`. `phase3_rag.serve` logs that Ollama is unreachable and returns the top passage. |
| Postgres | Optional. File sink (`data/processed/*.jsonl`) is the default. |
| Egress | Crawling public sites in Phase 2 depends on the environment network policy. Prefer a local `http.server` fixture when egress is restricted. |
| Secrets | Do not put API keys in `environment.json` or committed scripts. Use Cursor environment secrets for `FIRECRAWL_API_KEY` / `PG_DSN` if needed. |
| `install` vs `start` | Dependency install belongs in `install`. Do not put `ollama serve` or a crawl worker in `install` — they would block snapshotting. Put long-running processes in `start` or `terminals` only if you add them later. |

## Updating the Cloud environment

1. Edit `.cursor/environment.json` and/or `scripts/setup_dev_env.sh` on a branch.
2. Push. New Cloud Agents that start from that revision pick up the committed file.
3. To bake a faster boot, use Cursor environment **builds** from the dashboard after the `install` script succeeds. Builds snapshot post-install disk state; `install` is not re-run on later boots from that build.

Do not combine a Dockerfile, an explicit image, and a snapshot in the same config. This environment uses Cursor's default image plus `install`.

## Related files

- [`.cursor/environment.json`](../.cursor/environment.json)
- [`scripts/setup_dev_env.sh`](../scripts/setup_dev_env.sh)
- [`arkguru-common/README.md`](../arkguru-common/README.md)
