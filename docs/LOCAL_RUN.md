# Local run

Run the 3-phase arkguru pipeline on a machine you control (laptop, NUC, or workstation). This path uses a Python virtualenv and file-based outputs. Postgres, Ollama, OCR, and the heavy Phase 3 ML stack are optional.

The companion document is [CLOUD_RUN.md](CLOUD_RUN.md).

## Layout

Clone the shared package, this umbrella repo, and the phase repos as **siblings**. `arkguru-common` is a first-class GitHub repo; this umbrella also vendors a copy under `./arkguru-common` as a fallback when the sibling is missing. Phase 2 (`arkguru-web-scraping`) is optional.

```
some-dir/
  arkguru-common/            # shared package (preferred)
  arkguru-pdf-converter/     # this repo (env setup + vendored common fallback)
    arkguru-common/
  arkguru-pdf-extraction/    # Phase 1
  arkguru-web-scraping/      # Phase 2 (optional)
  arkguru-rag-slm/           # Phase 3
```

```bash
mkdir -p ~/arkguru && cd ~/arkguru
git clone https://github.com/ravidsun/arkguru-common.git
git clone https://github.com/ravidsun/arkguru-pdf-converter.git
git clone https://github.com/ravidsun/arkguru-pdf-extraction.git
git clone https://github.com/ravidsun/arkguru-rag-slm.git
# optional Phase 2:
# git clone https://github.com/ravidsun/arkguru-web-scraping.git
```

Python 3.9+ is required. On Debian/Ubuntu also install `python3-venv` (and `python3-pip` if needed).

Optional system packages for scanned PDFs and figure OCR:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr ghostscript

# macOS
brew install tesseract ghostscript
```

## Install

From the umbrella repo:

```bash
cd arkguru-pdf-converter
bash scripts/setup_dev_env.sh
source .venv/bin/activate
```

The script is idempotent. It:

1. Creates `.venv` in this repo (override with `ARKGURU_VENV`). On Debian/Ubuntu it installs `python3-venv` if `ensurepip` is missing.
2. Installs `arkguru-common` from the sibling checkout when present, otherwise the vendored copy (`import common` works in every phase), including the **`postgres` extra** (`psycopg`, `pgvector`).
3. Installs Phase 1 requirements plus `reportlab` (sample PDF fixture) and Postgres clients, and Phase 2 requirements if that repo is checked out.
4. Installs the **offline** Phase 3 core (`numpy`, `rank-bm25`) so retrieval works without downloading models.

The virtualenv lives in `arkguru-pdf-converter/.venv`. Activate it before running any phase.

### Full Phase 3 stack (optional)

Real embeddings, LoRA fine-tuning, RAGAS eval, and Ollama generation:

```bash
source .venv/bin/activate
pip install -r ../arkguru-rag-slm/requirements.txt
```

That pull is large (transformers, sentence-transformers, FlagEmbedding, ragas, llama-index). Skip it until you need it.

## Smoke test (offline, no database)

With the venv activated:

```bash
# Shared package (sibling checkout; fall back to ./arkguru-common/tests)
python -m pytest ../arkguru-common/tests -q
```

### Phase 1 — PDF → chunks

```bash
cd ../arkguru-pdf-extraction
python scripts/make_sample_pdf.py     # data/raw_pdfs/sample_handbook.pdf
make phase1                           # data/processed/sample_handbook/chunks.jsonl
```

Drop your own PDFs in `data/raw_pdfs/` and re-run `make phase1`.

### Phase 2 — URLs → chunks

Edit `config/config.yaml` `phase2.seeds`, or pass seeds on the CLI:

```bash
cd ../arkguru-web-scraping
python -m phase2_web.pipeline --seeds https://example.com/docs --max-pages 20
# → data/processed/web_chunks.jsonl
```

For a JS-heavy site, set `backend: firecrawl` and `FIRECRAWL_API_KEY`.

### Phase 3 — ingest + retrieve (hashing embedder)

One command: Phase 1 extract → local vector store → extractive answer (Ollama is optional):

```bash
cd ../arkguru-rag-slm
python -m phase3_rag.run_pdfs \
  --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
  --embedder hashing \
  --ask "What PPE is required before servicing a unit?"
```

Or the autonomous file-mode orchestrator (ingest PDFs, then answer from JSONL + BM25):

```bash
make e2e
```

`make e2e` asks *"What does error code E14 mean?"* against the sample handbook. If Ollama is not running, the answer is extractive (top retrieved passage).

## Switch database (`PG_DSN`)

The datastore is one Postgres + pgvector client. Switch hosts by changing `PG_DSN` (or copying [`.env.example`](../.env.example) → `.env`). No provider-specific backend.

**Local / VPS Docker** — [compose.yaml](../compose.yaml) (`pgvector/pgvector:pg16`):

```bash
docker compose up -d
export PG_DSN=postgresql://rag:change-me@localhost:5432/rag
```

**Hosted Supabase** — session pooler (port **5432**, not transaction **6543**), user `postgres.<ref>`:

```bash
export PG_DSN='postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require'
```

**Other hosted Postgres + pgvector** (Neon, Crunchy Bridge, Timescale/Tiger, Aiven, RDS/Aurora, Cloud SQL, Azure Database, DigitalOcean/Railway/Render): paste their session/direct URI into `PG_DSN`. Remote hosts get `sslmode=require` when omitted. Optional `PG_SSLMODE=require|prefer|disable`. Prefer a session/direct connection — named cursors used by `iter_missing_embeddings` break on transaction poolers.

**Not drop-ins:** Baserow, NocoDB, Directus, Appwrite, PocketBase, Firebase, or dedicated vector DBs (Pinecone, Qdrant, Weaviate, Milvus, Chroma). Use `pgvector/pgvector` (or any Postgres with the `vector` extension), not `baserow/baserow`.

## Two-table Postgres smoke

Phase 1 `--sink postgres` fills **`chunks`** (`chunk_index` is 0-based). It does **not** write vectors. Phase 3 `embed_datastore` fills **`chunk_embeddings`**. In the Table Editor, inspect `chunks.chunk_index` — `chunk_embeddings` has no such column, and `0` can look blank.

Local Docker/native DSN (`docker compose up -d` in this repo), or a Supabase **Session pooler** URI (`aws-<region>.pooler.supabase.com:5432`, user `postgres.<ref>`, `sslmode=require`):

```bash
export PG_DSN=postgresql://rag:change-me@localhost:5432/rag   # or Session pooler URI
cd ../arkguru-pdf-extraction
python scripts/make_sample_pdf.py
python -m phase1_pdf.pipeline --init-db
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres --workers 1

cd ../arkguru-rag-slm
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024
# equivalent one-liner:
# python -m phase3_rag.run_pdfs --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
#     --sink postgres --embedder hashing
```

```sql
SELECT chunk_id, chunk_index, source_id, page FROM chunks ORDER BY source_id, chunk_index;
SELECT count(*) FROM chunk_embeddings;
```

See [DATABASE_SETUP.md](https://github.com/ravidsun/arkguru-pdf-extraction/blob/main/docs/DATABASE_SETUP.md) Option D for Supabase Session pooler details.

## Production-style local run

Postgres + pgvector, sentence-transformer embeddings, and a local LLM via Ollama.

1. Install the full Phase 3 requirements (see above).
2. Start Postgres with the `vector` extension (`docker compose up -d` here),
   **or** point `PG_DSN` at hosted Supabase / Neon / any Postgres + pgvector.
   See [DATABASE_SETUP.md](https://github.com/ravidsun/arkguru-pdf-extraction/blob/main/docs/DATABASE_SETUP.md)
   (**Option D — Supabase**): session pooler URI (port **5432**, not transaction
   **6543**), `sslmode=require`, `PG_DSN` only in `.env` (never commit it).
   Retrieve is **fail-closed** when `PG_DSN` is set but connect fails. Daily
   dump: from `arkguru-rag-slm`, `python -m phase3_rag.backup --once` or
   `make backup`.

```bash
# Local Docker (compose.yaml in this repo) / native (Option A–C)
docker compose up -d
export PG_DSN=postgresql://rag:change-me@localhost:5432/rag
psql "$PG_DSN" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Hosted Supabase (Option D) or other hosted pgvector: copy URI into .env as PG_DSN
# cp .env.example .env
```

3. Ingest to files or `--sink postgres`:

```bash
cd ../arkguru-pdf-extraction
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres

cd ../arkguru-web-scraping
python -m phase2_web.pipeline --config config/config.yaml --sink postgres
```

4. Fine-tune (optional, CPU hours), merge/quantize, register with Ollama as `domain-slm` (see the header of `phase3_rag/serve.py`).

5. Embed (if chunks are already in Postgres) and chat:

```bash
cd ../arkguru-rag-slm
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024   # Cloud-safe
# NUC:
python -m phase3_rag.embed_datastore --embedder sentence_transformer \
    --model BAAI/bge-m3 --dim 1024
# or: make index / python -m phase3_rag.index --embed-only --embedder hashing
# JSONL corpus → Postgres (not the local .npz store): make index
ollama serve &
python -m phase3_rag.serve
# file-mode alternative (no DSN): python -m phase3_rag.run_pdfs --pdfs ... --embedder hashing --chat
```

## Useful make targets

| Repo | Target | What it does |
|---|---|---|
| `arkguru-pdf-extraction` | `make phase1` | Extract + chunk PDFs |
| `arkguru-pdf-extraction` | `make worker` | Watch `data/raw_pdfs` |
| `arkguru-web-scraping` | `make phase2` | Crawl seeds in config |
| `arkguru-rag-slm` | `make from-pdfs PDFS=... ASK="..."` | PDF → ingest → ask |
| `arkguru-rag-slm` | `make from-pdfs PDFS=... SINK=postgres` | PDF → `chunks` + `chunk_embeddings` |
| `arkguru-rag-slm` | `make embed-db` | Fill `chunk_embeddings` (hashing) |
| `arkguru-rag-slm` | `make e2e` | Offline orchestrator pass |
| `arkguru-rag-slm` | `make serve` | Interactive RAG chat |
| `arkguru-rag-slm` | `make backup` | Dump `chunks` + embeddings if watermark moved |
| `arkguru-rag-slm` | `make index` | JSONL → **Postgres** chunks + embeddings |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: common` | Activate `.venv` created by `scripts/setup_dev_env.sh`, or `pip install -e arkguru-common` |
| Phase repo not found during setup | Clone it as a **sibling** of `arkguru-pdf-converter` |
| Empty Phase 1 output | Scanned PDF — install tesseract/ghostscript and set `ocr_enabled: true` |
| Blank Phase 2 pages | JS site — switch to `backend: firecrawl` |
| Ollama errors | Start `ollama serve`; extractive answers still work without it |
| Slow / OOM on fine-tune | Stay on `--embedder hashing` for plumbing tests; reduce LoRA rank / batch size for real training |
