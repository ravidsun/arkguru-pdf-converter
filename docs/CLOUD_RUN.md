# Cloud run (Cursor Cloud Agent)

Run the 3-phase arkguru pipeline inside a **Cursor Cloud Agent** VM.

The companion document is [LOCAL_RUN.md](LOCAL_RUN.md).

## How the environment is defined

The personal Cloud Agent environment for this workspace is **dashboard-managed**. Its repos are:

- `arkguru-common`
- `arkguru-pdf-converter` (this repo; `install` script + optional vendored common)
- `arkguru-pdf-extraction`
- `arkguru-rag-slm`

`install` is empty until the environment is Saved with a working script. After Save, new agents run that `install` on boot (or from an environment build snapshot). There is no `start` command and no `terminals` entry: the pipeline is CLI-driven. Long-running workers (`make worker`, `orchestrator.py`) are started on demand.

This repo also commits [`.cursor/environment.json`](../.cursor/environment.json). That file is the source of truth when a Cloud Agent **starts from this repository revision** (it overrides dashboard personal/team environments):

```json
{
  "name": "arkguru RAG dev",
  "install": "bash scripts/setup_dev_env.sh",
  "repositoryDependencies": [
    "https://github.com/ravidsun/arkguru-common",
    "https://github.com/ravidsun/arkguru-pdf-extraction",
    "https://github.com/ravidsun/arkguru-rag-slm"
  ]
}
```

| Field | Role |
|---|---|
| `install` | After checkout, creates `.venv`, installs `arkguru-common`, and installs phase dependencies. Idempotent. Does **not** start servers. |
| `repositoryDependencies` | Puts `arkguru-common` and the Phase 1 / Phase 3 repos in the Cloud Agent GitHub token scope so they can be checked out as siblings. |

Phase 2 (`arkguru-web-scraping`) is optional and is **not** part of this environment. If that repo is present as a sibling, `install` will pick it up; otherwise it logs a warning and continues.

Changes to the dashboard environment or to `.cursor/environment.json` apply to **newly started** agents, not an already-running session.

## What a Cloud Agent sees on disk

Typical layout after checkout + `install`:

```
/agent/repos/
  arkguru-common/            # first-class shared package (preferred)
  arkguru-pdf-converter/     # this repo; .venv lives here
    arkguru-common/          # vendored fallback if the sibling is missing
  arkguru-pdf-extraction/
  arkguru-rag-slm/
```

`scripts/setup_dev_env.sh` looks for each repo as a sibling (`../arkguru-common`) or nested under this repo. It prefers the sibling `arkguru-common` checkout.

Debian/Ubuntu images need `python3-venv` (and `python3-pip` if missing). The install script installs those packages when `ensurepip` is unavailable.

Activate the venv created by `install`:

```bash
source /agent/repos/arkguru-pdf-converter/.venv/bin/activate
# or, from this repo root:
source .venv/bin/activate
```

## Bring your PDFs (Cloud Agent)

The agent runs on a **remote Linux VM**. It cannot read folders on your PC
(for example `C:\Users\ravid\Downloads\...`). Copying into a local git clone of
`data/raw_pdfs/` also does nothing for this agent.

**Ingest directory on the VM:**

```
/agent/repos/arkguru-pdf-extraction/data/raw_pdfs/
```

Subfolders are searched recursively (`hvac/manual.pdf` → source id `hvac/manual.pdf`).
That tree is **gitignored** (`data/raw_pdfs/*` except `.gitkeep`) — do not commit the corpus.

**Send a batch (recommended):** zip the folder on Windows, then attach the `.zip` to a
Cloud Agent prompt or follow-up.

```powershell
Compress-Archive -Path "C:\Users\ravid\Downloads\AstrologyBooks-20260908T054904Z-1-001\AstrologyBooks\VedicAstro_potdar" `
  -DestinationPath "$env:USERPROFILE\Downloads\VedicAstro_potdar.zip"
```

In the agent chat: attach `VedicAstro_potdar.zip` and ask it to unpack into
`/agent/repos/arkguru-pdf-extraction/data/raw_pdfs/VedicAstro_potdar` and run Phase 1.

Attaching individual PDFs works for a handful of files; a zip is better for a bookshelf.

**After files are on the VM:**

```bash
source /agent/repos/arkguru-pdf-converter/.venv/bin/activate
cd /agent/repos/arkguru-pdf-extraction
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres --workers 1
```

Phase 1 fills `chunks` only. Then:

```bash
cd /agent/repos/arkguru-rag-slm
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024
```

Needs runtime secret `PG_DSN` (Supabase **Session pooler**, port 5432) for `--sink postgres`.
Omit `--sink postgres` to write JSONL under `data/processed/` instead.

## What is installed (and what is not)

The Cloud `install` script matches the **offline** local bootstrap:

- `arkguru-common` (schema, tokenizer, chunking, datastore/`search_chunks`, worker, Python RRF for file mode) + pytest + **postgres extra** (`psycopg`, `pgvector`)
- Phase 1: pymupdf / pymupdf4llm, tiktoken, reportlab (sample PDF), `psycopg` / `pgvector` for `--sink postgres`
- Phase 2 (only if that repo is checked out): httpx, trafilatura, datasketch, …
- Phase 3 core only: numpy + rank-bm25

**Not** installed by default (keep the image lean; no multi-GB model downloads):

- `sentence-transformers`, transformers, peft, ragas, llama-index, FlagEmbedding
- System OCR: tesseract, ghostscript
- A Postgres **server** (Python `psycopg` / `pgvector` clients **are** installed via `arkguru-common[postgres]`)
- Ollama / llama.cpp

Retrieval still works: `--embedder hashing` plus file-mode BM25, or SQL
`search_chunks()` when `PG_DSN` is set, with extractive answers when Ollama is absent.

To add the full Phase 3 stack in a Cloud session:

```bash
source .venv/bin/activate
pip install -r ../arkguru-rag-slm/requirements.txt
```

That is slow and network-heavy; only do it when you need real embeddings or generation.

**32 PDFs:** default Cloud path (Phase 1 + hashing, no OCR packages, no GPU) is about **15–45 minutes** for native-text manuals. CPU `bge-m3` after installing the full stack is about **1–2.5 hours**. Scanned PDFs need Tesseract installed in the session or OCR will not run.

## Smoke test in a Cloud Agent

```bash
source /agent/repos/arkguru-pdf-converter/.venv/bin/activate

# Shared package (sibling checkout)
python -m pytest /agent/repos/arkguru-common/tests -q

# Phase 1
cd /agent/repos/arkguru-pdf-extraction
python scripts/make_sample_pdf.py
make phase1

# Phase 3 end-to-end (offline orchestrator)
cd /agent/repos/arkguru-rag-slm
make e2e
```

`make e2e` should ingest the sample PDF and answer *"What does error code E14 mean?"* from the JSONL corpus (BM25 + extractive fallback).

One-shot RAG without the orchestrator:

```bash
cd /agent/repos/arkguru-rag-slm
python -m phase3_rag.run_pdfs \
  --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
  --embedder hashing \
  --ask "What PPE is required before servicing a unit?"
```

### Two-table Postgres smoke (optional)

Phase 1 writes **`chunks`** only (`chunk_index` is 0-based on that table). Phase 3 `embed_datastore` fills **`chunk_embeddings`**. In the Table Editor, look at `chunks.chunk_index`, not `chunk_embeddings` (that table has no `chunk_index` column; a `0` can look blank).

Use a **Session pooler** DSN (`aws-<region>.pooler.supabase.com:5432`, user `postgres.<ref>`, `sslmode=require`) as runtime secret `PG_DSN`:

```bash
source /agent/repos/arkguru-pdf-converter/.venv/bin/activate
cd /agent/repos/arkguru-pdf-extraction
python scripts/make_sample_pdf.py
python -m phase1_pdf.pipeline --config config/config.yaml --init-db
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres --workers 1 --no-figures

cd /agent/repos/arkguru-rag-slm
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024
# or: python -m phase3_rag.run_pdfs --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
#        --sink postgres --embedder hashing
```

Confirm with SQL (not Table Editor guesswork):

```sql
SELECT chunk_id, chunk_index, source_id, page FROM chunks ORDER BY source_id, chunk_index;
SELECT count(*) FROM chunk_embeddings;
```

Phase 2 is skipped unless `arkguru-web-scraping` is checked out. When it is, a local HTTP fixture avoids public crawls:

```bash
mkdir -p /tmp/arkguru-demo-site
printf '<!DOCTYPE html><html><head><title>Demo</title></head><body><h1>Safety</h1><p>PPE includes insulated gloves rated to 1000V and safety eyewear. Lockout-tagout is mandatory before servicing any unit.</p></body></html>' \
  > /tmp/arkguru-demo-site/index.html
python3 -m http.server 8899 --directory /tmp/arkguru-demo-site &
cd /agent/repos/arkguru-web-scraping
python -m phase2_web.pipeline --seeds http://127.0.0.1:8899/index.html --max-pages 5
```

## Cloud-specific constraints

| Topic | Behavior in Cloud Agents |
|---|---|
| GPU / Intel NUC features | Not available. Stay on hashing embeddings and extractive answers unless you explicitly install the heavy stack. |
| Ollama | Not started by `install`. `phase3_rag.serve` logs that Ollama is unreachable and returns the top passage. |
| Postgres | Optional. File sink (`data/processed/*.jsonl`) is the default. For a two-table smoke against Supabase, set **environment-scoped** secret `PG_DSN` to the **Session pooler** URI (port **5432**, user `postgres.<ref>`, `sslmode=require`). Do not use Direct `db.<ref>.supabase.co` (IPv6-only) or transaction pooler port 6543. |
| Egress | Crawling public sites in Phase 2 depends on the environment network policy. Prefer a local `http.server` fixture when egress is restricted. |
| Secrets | Do not put API keys in `environment.json` or committed scripts. Add `PG_DSN` (and `FIRECRAWL_API_KEY` if needed) as an **environment-scoped Runtime Secret or Environment Variable** on this Cloud Agent environment so every agent receives it. `start` (`arkguru-common/scripts/start_services.sh`) copies `PG_DSN` into gitignored `.env` files in each repo; it must not overwrite an injected secret with the local `127.0.0.1` DSN. |
| `python3-venv` | Required to create `.venv`. The install script installs `python3-venv` / `python3-pip` via apt when `ensurepip` is missing. |
| `install` vs `start` | Dependency install belongs in `install`. Do not put `ollama serve` or a crawl worker in `install` — they would block snapshotting. Put long-running processes in `start` or `terminals` only if you add them later. |

## Updating the Cloud environment

**Dashboard-managed personal environment** (this workspace):

1. Change `scripts/setup_dev_env.sh` (and this document) on a branch and merge.
2. Re-run install, snapshot, and propose `install` from a Cloud Agent so the Environment panel can be Saved. Save is required; a draft proposal does not activate the environment.
3. To bake a faster boot, use Cursor environment **builds** after `install` succeeds. Builds snapshot post-install disk state; `install` is not re-run on later boots from that build.

**When starting from this repo's committed `.cursor/environment.json`:**

1. Edit `.cursor/environment.json` and/or `scripts/setup_dev_env.sh` on a branch.
2. Push. New Cloud Agents that start from that revision pick up the committed file.

Do not combine a Dockerfile, an explicit image, and a snapshot in the same config. This environment uses Cursor's default image plus `install`.

## Related files

- [`.cursor/environment.json`](../.cursor/environment.json)
- [`DATABASE_SETUP.md`](DATABASE_SETUP.md) — native, Docker, hosted `PG_DSN`
- [`scripts/setup_dev_env.sh`](../scripts/setup_dev_env.sh)
- [`../arkguru-common/scripts/start_services.sh`](https://github.com/ravidsun/arkguru-common/blob/develop/scripts/start_services.sh) — Cloud `start`; publishes `PG_DSN` into gitignored `.env` files
- [`../arkguru-common/README.md`](https://github.com/ravidsun/arkguru-common) (sibling) or the vendored [arkguru-common/README.md](../arkguru-common/README.md)
