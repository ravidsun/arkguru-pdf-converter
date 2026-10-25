# arkguru — A 3-Phase Local RAG System

**arkguru** is an end-to-end retrieval-augmented generation (RAG) pipeline that extracts knowledge from PDFs and websites, optionally fine-tunes a small language model, and serves a fully local chat system — no external APIs at inference time.

This repository (`arkguru-pdf-converter`) is the source of truth. The four Python packages live as folders here; they are not separate checkouts.

**Target hardware:** Asus NUC with Intel Core Ultra 9 185H, 96 GB RAM, GPU-optional
**CPU-only fine-tuning:** ✓ Yes, with Intel Extension for Transformers or ipex-llm
**GPU migration:** ✓ Yes, config + uncommented lines only
**Local inference:** ✓ Yes, via Ollama (llama.cpp)

**How to run:**

- [arkguru-ui/README.md](arkguru-ui/README.md) — local web wizard (FastAPI + React) over the same phase CLIs
- [docs/LOCAL_RUN.md](docs/LOCAL_RUN.md) — laptop / NUC / workstation (venv, file sink, optional Postgres + Ollama)
- [docs/CLOUD_RUN.md](docs/CLOUD_RUN.md) — Cursor Cloud Agent (`.cursor/environment.json` + `scripts/setup_dev_env.sh`)
- Hybrid retrieve implementation: [`arkguru-common/common/datastore.py`](arkguru-common/common/datastore.py) — `ensure_schema()` installs SQL `search_chunks()`

YouTube playlist ingestion and a unified ingest-manifest CLI are **not implemented**. That work is tracked in epic [#39](https://github.com/ravidsun/arkguru-pdf-converter/issues/39).

---

## Repository layout

```
arkguru-pdf-converter/          # this git clone
  arkguru-common/               # shared Chunk schema, datastore, chunking
  arkguru-pdf-extraction/       # Phase 1 — PDFs → Chunks
  arkguru-web-scraping/         # Phase 2 — URLs → Chunks (schema web)
  arkguru-rag-slm/              # Phase 3 — embed, retrieve, serve, optional LoRA
  arkguru-ui/                   # optional local wizard over the same CLIs
  compose.yaml                  # local Docker Postgres + pgvector (host 5433)
  scripts/                      # setup_dev_env.sh, setup_docker_pg.sh, detect_local_pg.sh
  docs/                         # LOCAL_RUN.md, CLOUD_RUN.md, DATABASE_SETUP.md
```

| Folder | Role |
|---|---|
| `arkguru-common` | Shared `Chunk` schema, tokenizer, `pack_windows`, `ChunkStore`, `PG_DSN` resolution |
| `arkguru-pdf-extraction` | Extract → chunk → write (files or `public.chunks`) |
| `arkguru-web-scraping` | Allowlisted crawl → chunk → write (`web_chunks.jsonl` and/or `web.chunks`) |
| `arkguru-rag-slm` | Combine, embed, hybrid retrieve, Ollama serve, optional LoRA / RAGAS |
| `arkguru-ui` | FastAPI + React wizard that shells out to the phase CLIs |

---

## Prerequisites and setup

- Python 3.9+
- On Debian/Ubuntu: `python3-venv` (and `python3-pip` if needed)
- Optional: Docker Engine (local Postgres + pgvector)
- Optional: [Ollama](https://ollama.com) for generated answers
- Optional: system Tesseract + Ghostscript for scanned-PDF OCR (installed by setup on Debian/Ubuntu)

```bash
git clone https://github.com/ravidsun/arkguru-pdf-converter.git
cd arkguru-pdf-converter
bash scripts/setup_dev_env.sh
source .venv/bin/activate
```

`scripts/setup_dev_env.sh` is idempotent. It:

1. Creates `.venv` here (override with `ARKGURU_VENV`).
2. Installs `arkguru-common` editable with extras `parquet`, `postgres`, and `test`.
3. Installs Phase 1 and Phase 2 requirements, plus `reportlab` (sample PDF), `psycopg` / `pgvector`, and `ocrmypdf` / `pytesseract` / `pillow`. On Debian/Ubuntu it also installs `tesseract-ocr`, `tesseract-ocr-eng`, and `ghostscript`.
4. Installs only the **offline** Phase 3 core (`numpy`, `rank-bm25`) so hashing retrieve works without downloading models.

The virtualenv lives at `.venv`. Activate it before running any phase.

**OCR on macOS** (setup does not brew-install these):

```bash
brew install tesseract ghostscript
```

**Windows:** install Tesseract and Ghostscript and add them to `PATH`.

**Full Phase 3 stack** (real embeddings, LoRA, RAGAS, sentence-transformers) is a large extra install:

```bash
source .venv/bin/activate
pip install -r arkguru-rag-slm/requirements.txt
```

**Optional UI:**

```bash
pip install -r arkguru-ui/requirements.txt
cd arkguru-ui/frontend && npm install
cd .. && make dev          # API :8765 + Vite :5173
```

Copy env files; never commit secrets:

```bash
cp .env.example .env
# optional: cp arkguru-ui/.env.example arkguru-ui/.env
```

`.env` is gitignored. `PG_DSN` is the only connection string the app reads.

---

## Postgres + pgvector

Postgres is **optional**. File/JSONL mode works without it. For a shared two-table datastore (`chunks` + `chunk_embeddings`) you need PostgreSQL 14+ (16 recommended) with pgvector. Switch hosts by changing `PG_DSN` only.

**One-click local Docker** ([compose.yaml](compose.yaml), image `pgvector/pgvector:pg16`, host port **5433** so native 5432 can coexist):

```bash
bash scripts/setup_docker_pg.sh
# writes gitignored .env
# PG_DSN=postgresql://rag:change-me@127.0.0.1:5433/rag
set -a && . .env && set +a
```

**Already running** (native 5432 or Docker 5433) — probe and write `.env` (never prints the password):

```bash
bash scripts/detect_local_pg.sh
# or: PYTHONPATH=arkguru-common python -m common.local_pg
```

**Manual compose:**

```bash
docker compose up -d
export PG_DSN=postgresql://rag:change-me@127.0.0.1:5433/rag
```

Override the published port with `ARKGURU_PG_PORT`. The compose password is the documented local default `change-me` — change it for anything that is not a throwaway laptop DB, and never commit a real DSN.

**Hosted** (Supabase session pooler, Neon, RDS, …): paste a session/direct URI into `PG_DSN`. Remote hosts get `sslmode=require` when omitted. Transaction poolers break the named cursor in `iter_missing_embeddings`.

```bash
# Create the extension once; the app also does this on first write
psql "$PG_DSN" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Tables + search_chunks() are created by ensure_schema()
cd arkguru-pdf-extraction
python -m phase1_pdf.pipeline --init-db
```

See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md).

---

## System architecture

```
┌─────────────────────┐
│  PDFs on disk       │
│  Websites (crawl)   │
└──────────┬──────────┘
           │
      ┌────▼─────────────────────────────────┐
      │  Phase 1: arkguru-pdf-extraction     │
      │  + Phase 2: arkguru-web-scraping     │
      │  → Common Chunk schema               │
      └──────────┬──────────────────────────┘
                 │
      ┌──────────▼──────────────────────────┐
      │  Phase 3: arkguru-rag-slm            │
      │  - Combine + deduplicate             │
      │  - Fine-tune small LM (LoRA, optional)│
      │  - Index + SQL search_chunks()       │
      │  - Serve via Ollama                  │
      └──────────┬──────────────────────────┘
                 │
         ┌───────▼────────┐
         │  Local chat    │
         │  Grounded QA   │
         │  with citations│
         └────────────────┘
```

Optional: `arkguru-ui` starts the same CLIs, streams logs, and chats with citations.

---

## Phase 1: PDF Extraction (`arkguru-pdf-extraction`)

**Input:** Folder of PDFs (native, scanned, or mixed) — default `data/raw_pdfs/` (recursive)
**Output (file sink):** Per-PDF folders under `data/processed/{relative_stem}/`
**Output (postgres sink):** `public.chunks` on `PG_DSN` (no files, no vectors)
**Speed:** native ~3–5 MB/s; scanned ~0.1 MB/s (OCR). See [Hardware](#hardware--performance).

**Key features:**
- **Three extraction backends:** `pymupdf4llm` (default, fast), `docling` (complex tables; still commented in `requirements.txt`), `pymupdf` (fallback)
- **Scanned PDF OCR:** if *any* page looks image-only, the file is routed through `ocrmypdf` with `skip_text=True` (not `force_ocr`). Native text layers are left untouched.
- **Table extraction:** `page.find_tables()` → standalone markdown chunks (`extra.block_type: table`)
- **Figure OCR:** `pytesseract` on embedded images (`extra.block_type: figure`); skipped if the libs are missing
- **Three chunking strategies:** `structure` (config default, heading windows), `parent_child` (long parent + small children). `semantic` is a CLI/config value; Phase 1 never passes `semantic_embedder`, so it uses the same windows as `structure`.

### File-mode smoke

```bash
cd arkguru-pdf-extraction
python scripts/make_sample_pdf.py     # data/raw_pdfs/sample_handbook.pdf
make phase1                           # python -m phase1_pdf.pipeline --config config/config.yaml
# → data/processed/sample_handbook/chunks.jsonl
#    (+ parents.jsonl / tables.jsonl / figures.jsonl when non-empty)
```

Equivalent:

```bash
python -m phase1_pdf.pipeline --input data/raw_pdfs --out-dir data/processed --workers 1
# or the convenience runner (same flags except --workers / --datastore-config):
python run_phase1.py --backend pymupdf4llm --strategy structure --no-figures
```

### CLI flags (`python -m phase1_pdf.pipeline`)

| Flag | Values / default | Notes |
|---|---|---|
| `--config` | `config/config.yaml` via `make phase1` | YAML `phase1:` block; CLI overrides YAML |
| `--input` | `data/raw_pdfs` | Recursive `**/*.pdf` |
| `--out-dir` | `data/processed` | File sink only |
| `--format` | `jsonl` \| `parquet` | File sink only |
| `--backend` | `pymupdf4llm` \| `docling` \| `pymupdf` | Missing optional lib falls back to `pymupdf` |
| `--strategy` | `structure` \| `parent_child` \| `semantic` | `semantic` == `structure` today |
| `--target-tokens` / `--overlap-pct` / `--min-tokens` | 400 / 0.15 / 80 | From config |
| `--no-ocr` / `--no-tables` / `--no-figures` | off by default | Config defaults are all `true` |
| `--workers` | `0` = auto (`cpu_count-1`) | `run_phase1.py` does **not** expose this |
| `--sink` | `file` \| `postgres` | Postgres writes `chunks` only |
| `--datastore-config` | `config/datastore.yaml` | DSN from `PG_DSN` |
| `--init-db` | | `ensure_schema()` then exit |

Worker (watch `data/raw_pdfs`):

```bash
python -m phase1_pdf.worker --once
python -m phase1_pdf.worker --interval 30 --sink postgres
# make worker  →  --interval 30
```

**Configuration** lives in `arkguru-pdf-extraction/config/config.yaml` (`phase1:`). Datastore table names and `dim` live in `config/datastore.yaml`.

**Postgres handoff** (Phase 1 does **not** write `chunk_embeddings`):

```bash
export PG_DSN=postgresql://rag:change-me@127.0.0.1:5433/rag
python -m phase1_pdf.pipeline --init-db
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres --workers 1
```

Nested inputs stay nested: `hvac/manual.pdf` → `data/processed/hvac/manual/chunks.jsonl` (file sink) or `source_id=hvac/manual.pdf` (postgres). Do not commit PDFs; `data/raw_pdfs/*` is gitignored.

---

## Phase 2: Web Scraping (`arkguru-web-scraping`)

**Input:** Seed URLs in `config/config.yaml` (`phase2.seeds`)
**Output (always):** `data/processed/web_chunks.jsonl` (or `.parquet`)
**Output (`--sink postgres`):** also `web.chunks` / `web.search_chunks()` on the same `PG_DSN`. It **refuses** schema `public`.
**Speed:** ~200–500 ms/page for the implemented local fetcher.

**Key features:**
- **Schema `web`:** `config/datastore.yaml` has `postgres.schema: web`. `--sink postgres` exits if the schema is `public` or unset.
- **Allowlisted BFS:** astrology article hubs in `config/config.yaml` (Vedic + traditional Western; no commercial Kundli portals). `same_domain_only`, robots.txt, path denylist, Kundli-generator query skip.
- **Fetch backend in code:** `phase2_web.fetch.crawl` is local `httpx` + `trafilatura` (regex fallback). `backend: firecrawl` is a config field and is **not executed**. `firecrawl-py` is commented in `requirements.txt`.
- **Structure-aware chunking:** same `pack_windows` as Phase 1 (`target_tokens: 550`)
- **Near-dedup:** MinHash LSH when `datasketch` is installed (Jaccard ≥ 0.9); exact-text fallback otherwise
- **PDF harvest:** publicly linked `.pdf` files are saved under `data/raw_pdfs/{domain}/`. Phase-1 ingest into schema `web` runs only when `--sink postgres` and `ingest_pdfs: true`. Jagannath Hora installer binaries are skipped.

### Run

```bash
cd arkguru-web-scraping
python -m phase2_web.pipeline --config config/config.yaml
# → data/processed/web_chunks.jsonl
# make phase2  is the same command

# isolated schema on the same PG_DSN as the books:
python -m phase2_web.pipeline --config config/config.yaml --sink postgres
# → rag.web.chunks  (DBeaver: schema web)
python -m phase2_web.pipeline --init-db    # web schema/tables only
```

Crawling the default seed list hits live sites (`max_pages: 2000`, `max_pages_per_seed: 80`, `delay_seconds: 0.75`). For a first try, pass a tiny allowlist:

```bash
python -m phase2_web.pipeline --config config/config.yaml \
  --seeds https://docs.python.org/3/tutorial/ \
  --max-pages 6 --max-pages-per-seed 6 --no-pdfs
```

### CLI flags (`python -m phase2_web.pipeline`)

| Flag | Notes |
|---|---|
| `--config` | YAML `phase2:` block |
| `--seeds` | Override seed URLs |
| `--max-pages` / `--max-pages-per-seed` / `--delay` | Caps and politeness |
| `--sink` | `file` \| `postgres` (file JSONL is still written) |
| `--datastore-config` / `--out-dir` | Defaults from YAML |
| `--no-pdfs` | Skip PDF download |
| `--no-ingest-pdfs` | Download PDFs but do not run Phase 1 |
| `--init-db` | Create `web` schema/tables, then exit |

There is **no** `--backend` CLI flag. `backend` is YAML-only and unused by `crawl()`.

Worker:

```bash
python -m phase2_web.worker --once
python -m phase2_web.worker --interval 3600 --sink postgres
# make worker  →  --once
```

Orchestrator step `crawl_web` stays **disabled** in `arkguru-rag-slm/orchestrator.yaml` until you turn it on.

---

## Phase 3: RAG + Fine-tuning (`arkguru-rag-slm`)

**Input:** Phase 1/2 JSONL (or rows already in Postgres)
**Output:** local `.npz` store, or `chunk_embeddings` + grounded answers
**Speed:** Fine-tune 3–50h (CPU), Inference 5–20 tok/s (CPU), 100+ tok/s (GPU)

### Quickstart (no database)

```bash
cd arkguru-rag-slm

# One command: Phase 1 extract → local store → ask
python -m phase3_rag.run_pdfs \
  --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
  --embedder hashing \
  --ask "What PPE is required before servicing a unit?"

# Real embeddings + Ollama (after pip install -r requirements.txt):
python -m phase3_rag.run_pdfs --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
  --embedder sentence_transformer --model domain-slm --chat
```

`run_pdfs` flags: `--pdfs` (required folder), `--phase1-repo`, `--strategy` (`parent_child` default), `--store` (`data/store/index`), `--embedder` (`hashing` \| `sentence_transformer`), `--model-name` (`BAAI/bge-m3`), `--ask`, `--chat`, `--model` (Ollama tag), `--top-k`, `--sink` (`file` \| `postgres`), `--dim` (1024). `--sink postgres` requires `PG_DSN` and does not support `--chat` (use `phase3_rag.serve`).

Makefile wrapper:

```bash
make from-pdfs PDFS=../arkguru-pdf-extraction/data/raw_pdfs ASK="What PPE is required before servicing a unit?"
make from-pdfs PDFS=../arkguru-pdf-extraction/data/raw_pdfs SINK=postgres
```

Manual two-step (no Phase 1 shell-out):

```bash
python -m phase3_rag.quickstart \
  --add ../arkguru-pdf-extraction/data/processed/sample_handbook/chunks.jsonl \
  --embedder hashing --store data/store/index \
  --ask "What PPE is required before servicing a unit?"
```

The local store is `data/store/index.{npz,jsonl}`. Incremental and keyed by `chunk_id`. This path is for small corpora; production uses Postgres.

### Combine + fine-tune (optional)

Phase 1 writes **per-PDF folders**, not `data/processed/pdf_chunks.jsonl`. `make combine` still passes those two legacy filenames (`pdf_chunks.jsonl` + `web_chunks.jsonl`); missing files are skipped with a warning. Prefer the real paths:

```bash
python -m phase3_rag.prepare_dataset \
  --inputs ../arkguru-pdf-extraction/data/processed/sample_handbook/chunks.jsonl \
           ../arkguru-web-scraping/data/processed/web_chunks.jsonl \
  --out-corpus data/processed/corpus.jsonl \
  --out-train data/processed/train.jsonl \
  --mode self_supervised \
  --val-frac 0.1
# → data/processed/corpus.jsonl, train.jsonl, val.jsonl
```

`--mode synthetic_qa` drafts Q&A via local Ollama. There is no `train_ratio` / `max_samples` config key.

```bash
make finetune
# python -m phase3_rag.finetune_qlora --config config/config.yaml
# → artifacts/lora-adapter/
```

`phase3_rag/serve.py` sketches merge → GGUF → `ollama create domain-slm`. There is **no** `phase3_rag.merge_and_export` module in this repo.

### Embed + retrieve + serve (Postgres)

```bash
export PG_DSN=postgresql://rag:change-me@127.0.0.1:5433/rag

# Chunks already in Postgres (Phase 1 --sink postgres):
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024
# make embed-db  is the same (hashing, dim 1024)
# NUC / production:
python -m phase3_rag.embed_datastore --embedder sentence_transformer \
  --model BAAI/bge-m3 --dim 1024

# JSONL corpus → Postgres chunks + embeddings (not the local .npz store):
# make index  →  python -m phase3_rag.index --config config/config.yaml
python -m phase3_rag.index --embed-only --embedder hashing
```

`embed_datastore` flags: `--datastore-config`, `--embedder` (default `sentence_transformer`), `--model`, `--dim`, `--batch`.

```bash
# Inspect retrieval (no --verbose flag)
python -m phase3_rag.retrieve "What PPE is required before servicing a unit?"
python -m phase3_rag.retrieve --processed-dir data/processed "your question"

# Chat. With PG_DSN set: SQL search_chunks() (fail-closed). Without: JSONL under --processed-dir.
python -m phase3_rag.serve
python -m phase3_rag.serve --ask "What PPE is required before servicing a unit?"
# make serve
```

`serve.py` generation backend is **Ollama only**. If Ollama is down, the top hit is returned extractive (`[extractive] …`). There is no `serve.backend: api`, `temperature`, or `top_p` setting in the serve CLI.

### Evaluate

```bash
make eval                 # python -m phase3_rag.eval_ragas --config config/config.yaml
make eval-retrieval       # no Ollama; hashing by default (EMB= / OUT= / BASE=)
make golden               # rebuild golden set (PER=25)
python -m phase3_rag.sources    # PDF source_id vs embedded counts
```

Golden file: `phase3_rag/golden/golden_qa.jsonl`.

### Workers and orchestrator

```bash
python -m phase3_rag.worker --once                 # hashing
python -m phase3_rag.worker --interval 60          # sentence_transformer (make worker)
python -m phase3_rag.worker --once --embedder hashing

python orchestrator.py --once                      # make orchestrate-once
python orchestrator.py --interval 300              # make orchestrate
python orchestrator.py --once --config orchestrator.filemode.yaml   # make e2e
```

`orchestrator.yaml`: `ingest_pdfs` → `index_embeddings` (`crawl_web` is `enabled: false`). `ingest_pdfs` uses `--sink postgres` when `PG_DSN` is set. File-mode e2e asks *"What does error code E14 mean?"* against Phase 1 JSONL.

Daily dump when the `created_at` watermark moved: `make backup` (`python -m phase3_rag.backup --once`) → `data/backups/*.dump`.

### Key components

| Module | Purpose | Input | Output |
|---|---|---|---|
| `prepare_dataset.py` | Combine + dedup chunks, generate training pairs | JSONL/Parquet files | train.jsonl, val.jsonl, corpus.jsonl |
| `finetune_qlora.py` | Fine-tune small LM with LoRA (CPU or GPU) | train.jsonl | artifacts/lora-adapter/ |
| `embed_datastore.py` | Embed chunks already in Postgres | `chunks` missing vectors | `chunk_embeddings` + HNSW |
| `index.py` | Upsert a JSONL corpus **into Postgres**, then embed | corpus.jsonl + `PG_DSN` | `chunks` + `chunk_embeddings` |
| `vector_store.py` / `run_pdfs` / `quickstart` | Local incremental index (no DB) | Phase 1 JSONL | `data/store/index.{npz,jsonl}` |
| `retrieve.py` | Hybrid retrieve: SQL `search_chunks()` when `PG_DSN` is set; JSONL otherwise | query | top-k ranked chunks |
| `serve.py` | Chat via Ollama, or extractive fallback + faithfulness gate | query + index | grounded answer + sources |
| `backup.py` | `pg_dump` of both tables when watermark moved | Postgres | `data/backups/*.dump` |
| `eval_ragas.py` | Evaluate retrieval + generation | golden_qa.jsonl | RAGAS metrics |
| `eval_retrieval.py` | Retrieval-only metrics (no Ollama) | golden + datastore | JSON report |
| `rewrite.py` | Query variants: `noop` \| `lexicon` \| `ollama` | query | variant list |

### Configuration

**Main config:** `arkguru-rag-slm/config/config.yaml` (`phase3:`)
**Database config:** `config/datastore.yaml` (DSN via `PG_DSN`, table names, `dim`, `hnsw_ef_search`)
**Golden eval set:** `phase3_rag/golden/golden_qa.jsonl`

Verified knobs from the YAML that exists today:

```yaml
base_model: "Qwen/Qwen2.5-3B-Instruct"
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
embedding_model: "BAAI/bge-m3"
reranker_model: "BAAI/bge-reranker-v2-m3"
retrieval:
  top_k_vector: 60          # keep datastore.yaml hnsw_ef_search >= this
  top_k_bm25: 60
  top_k_candidates: 120
  top_k_final: 6
  use_parent_expansion: true
  rewriter: noop            # noop | lexicon | ollama
serve:
  backend: "ollama"         # documented in YAML; serve.py always calls Ollama
  model_tag: "domain-slm"
  ctx: 4096
  faithfulness:
    enabled: true
    min_token_overlap: 0.4
    min_sentence_tokens: 6
    max_retries: 1
```

**Embedder choices:**
- `hashing`: Offline testing (no downloads, deterministic, low quality)
- `sentence_transformer`: Production (SOTA, ~100–300 queries/s CPU)

### Hybrid retrieve (`search_chunks`)

When `PG_DSN` is set, Phase 3 retrieve is **one SQL function**, `search_chunks()`, created by `ensure_schema()` in `common/datastore.py`. It fuses:

- **Dense:** HNSW cosine on `chunk_embeddings`
- **Lexical:** GIN `ts` on `chunks` (`ts @@ plainto_tsquery` + `ts_rank`)
- **RRF:** `1 / (rrf_k + rank)` with `rrf_k=60`, both legs `is_parent = false`

Python embeds the query (`HybridRetriever` uses `sentence_transformers`), optionally rewrites it, reranks with `BAAI/bge-reranker-v2-m3`, and expands parent chunks. File/npz mode (`run_pdfs` / `quickstart`) keeps Python `reciprocal_rank_fusion`. JSONL retrieve (`JsonlRetriever`) is used only when **no** DSN is configured. See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md#hybrid-retrieve).

---

## Inputs and outputs

| Stage | Inputs | Outputs |
|---|---|---|
| Phase 1 file | `arkguru-pdf-extraction/data/raw_pdfs/**/*.pdf` | `data/processed/{stem}/chunks.jsonl` (+ `parents` / `tables` / `figures` when non-empty); OCR scratch in `data/interim/` |
| Phase 1 postgres | same PDFs + `PG_DSN` | `public.chunks` (no vectors, no JSONL) |
| Phase 2 | `config/config.yaml` seeds | `data/processed/web_chunks.jsonl`; optional `data/raw_pdfs/{domain}/*.pdf` |
| Phase 2 postgres | same + `PG_DSN` | also `web.chunks` / `web.chunk_embeddings` (vectors still empty until Phase 3) |
| Phase 3 file | Phase 1/2 JSONL | `data/store/index.{npz,jsonl}`; optional `data/processed/{corpus,train,val}.jsonl` |
| Phase 3 postgres | `PG_DSN` | `chunk_embeddings`; dumps in `data/backups/` |
| Fine-tune | `train.jsonl` / `val.jsonl` | `artifacts/lora-adapter/` |
| UI | same CLIs | job logs under `arkguru-ui/data/jobs/` |

`data/raw_pdfs/*`, `data/processed/*`, `data/interim/*`, `.env`, and `artifacts/` are gitignored.

---

## Data flow and schema

All three phases share a common **`Chunk` schema** (in `common/schema.py`):

```python
text: str                    # chunk body
source_type: "pdf" | "web"   # origin
source_id: str               # PDF relative path or URL
chunk_index: int             # sequence in source (reindexed 0-based in Phase 1)
title: str | None
section: str | None
page: int | None             # PDF page (1-based)
url: str | None
domain: str | None
lang: str | None
token_count: int | None
overlap_tokens: int
parent_id: str | None
is_parent: bool
embedding: list[float] | None # nullable; filled by Phase 3
extra: dict                  # block_type: "table" | "figure" | "web"
chunk_id: str                # SHA256(source_type + source_id + index + role)[:16]
schema_version: str          # "1.0"
created_at: float
```

**Why one schema?**
- Phases 1 and 2 outputs concatenate with zero conversion
- One `chunks` table for all sources; vectors live in `chunk_embeddings`
- Re-embedding is truncating/refilling the embeddings table (vectors are disposable)

**Idempotency:**
- Same inputs → same `chunk_id`
- Upserts update text/metadata and **do not** reset `created_at` (backup watermark stays put)
- Adding new PDFs only appends new chunks; Phase 3 embeds rows still missing vectors

---

## Hardware & performance

### Tested on: Intel Core i7 (16 GB RAM, 8 cores)

| Task | Time | Throughput | Bottleneck |
|---|---|---|---|
| Phase 1: Native PDF (10 MB) | 2–3 s | ~3–5 MB/s | Parsing |
| Phase 1: Scanned PDF (5 MB) | 30–60 s | ~0.1 MB/s | OCR |
| Phase 2: Static site (100 pages) | 20–50 s | ~2–5 pages/s | Network + parsing |
| Phase 3: Fine-tune 3B model (5K pairs) | 12–24 h | ~400–600 pairs/h | Compute (CPU) |
| Phase 3: Inference (per token) | ~50–200 ms | 5–20 tok/s | Memory + compute (CPU) |
| 32 native-text PDFs, hashing embed (Cursor Cloud Agent) | ~15–45 min | — | Phase 1 + CPU, no GPU |
| 32 native-text PDFs, CPU `bge-m3` (after model cached) | ~1–2.5 h | — | Embed dominates |

### Scaling on GPU (future)
Fine-tuning: ~1–2 h (with `bitsandbytes` + GTX 1080)
Inference: 50–100+ tok/s (with vLLM + RTX 3090)

---

## Deployment paths

### Path A: Local files (development / small corpus)

```
Phase 1 PDFs → per-PDF .jsonl folders → Phase 3 (local index)
Phase 2 URLs → web_chunks.jsonl ↗
                          ↓
                   data/store/*.jsonl
                   (vector index in memory/disk)
```

**Pros:** Simple, offline, no database setup
**Cons:** Single-machine only, limited to ~100K chunks (memory-bound)
**Best for:** Prototyping, <1GB corpus

### Path B: Postgres + pgvector (production / large corpus)

```
Phase 1 PDFs → Postgres public.chunks
Phase 2 URLs → Postgres web.chunks
       ↓
Phase 3 fills embeddings (chunk_embeddings)
       ↓
Retrieval: search_chunks() (SQL RRF of HNSW + GIN FTS)
       ↓
Serve via Ollama (local LLM)
```

**Pros:** Scales to millions of chunks, transactional, shareable
**Cons:** Database setup, network latency (usually <10 ms)
**Best for:** Production, large corpora, team sharing

### Path C: Hybrid (start file-based, migrate to DB)

1. Develop locally with files (Path A)
2. Spin up Postgres + pgvector (Path B)
3. Phase 1/2 re-run with `--sink postgres` → upsert to DB
4. Phase 3 `embed_datastore` fills vectors
5. `search_chunks()` hybrid retrieve scales, no reprocessing of PDFs/web

---

## Tests

With the umbrella venv activated:

```bash
source .venv/bin/activate

# Root helper (setup_dev_env.sh OCR pins)
python -m pytest tests -q

# Shared package
python -m pytest arkguru-common/tests -q

# Phase 1 (make test here only runs ../arkguru-common/tests — use pytest directly)
python -m pytest arkguru-pdf-extraction/tests -q

# Phase 2
cd arkguru-web-scraping && make test     # pytest tests -q

# Phase 3
cd ../arkguru-rag-slm && make test       # pytest tests -q

# UI
cd ../arkguru-ui
pip install -r requirements.txt
make test
```

One-shot from the repo root (after UI deps are installed):

```bash
python -m pytest tests arkguru-common/tests \
  arkguru-pdf-extraction/tests arkguru-web-scraping/tests \
  arkguru-rag-slm/tests arkguru-ui/tests -q
```

---

## Troubleshooting by phase

### Setup / Postgres

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: common` | venv not active | `source .venv/bin/activate` or `pip install -e arkguru-common` |
| Can't connect to Postgres | DSN wrong or DB down | Check `PG_DSN`; `bash scripts/detect_local_pg.sh`; retrieve **raises** if DSN is set but connect fails |
| Docker port clash | Native Postgres on 5432 | Compose publishes **5433**; override with `ARKGURU_PG_PORT` |

### Phase 1 (PDFs)

| Issue | Cause | Fix |
|---|---|---|
| Empty output | Scanned PDF | Confirm `tesseract --version`; leave `ocr_enabled: true`; install Ghostscript |
| Garbled tables | Simple grid extraction | `--backend docling` (uncomment `docling` in `requirements.txt` first) |
| Missing image text | Figures not OCR'd | `extract_figures: true` + system Tesseract |
| Chunks too big/small | Tokenizer miscalibrated | Tune `target_tokens`, `overlap_pct`, `min_tokens` |
| Out of memory | Huge PDF + many workers | Reduce `--workers` or split the input folder |

### Phase 2 (Web)

| Issue | Cause | Fix |
|---|---|---|
| Blank content | JS-rendered site | The local fetcher cannot run Firecrawl. Narrow seeds or save a static HTML snapshot; do not set `backend: firecrawl` expecting it to run |
| Crawl wandered off-site | `same_domain_only: false` | Keep it `true` or lower `max_pages` / `max_pages_per-seed` |
| Slow crawl | Rate-limited / `delay_seconds: 0.75` | Lower `--max-pages` or raise `--delay` |
| `--sink postgres` exits | Schema is `public` | Keep `postgres.schema: web` in `config/datastore.yaml` |
| Duplicate content | Syndicated pages | `datasketch` is in requirements; near-dedup needs it |

### Phase 3 (RAG)

| Issue | Cause | Fix |
|---|---|---|
| Can't connect to Ollama | Ollama not running | `ollama serve` (extractive answer still works) |
| Generated answer refused | Faithfulness gate | Inspect `python -m phase3_rag.retrieve "question"`; gate retries once then refuses |
| Slow inference (CPU) | Normal on CPU | Smaller model or GPU |
| Poor answer quality | Bad chunks or retrieval | `python -m phase3_rag.retrieve "question"` |
| Fine-tune OOM | Model + data too large | Reduce `lora.rank` or `train.batch_size` |
| Re-embed with new model | Need other embeddings | `TRUNCATE chunk_embeddings;` then re-run `embed_datastore` (match `dim`) |
| `make combine` empty | Legacy filenames | Pass the real Phase 1 folder JSONL paths to `prepare_dataset` |

---

## Contributing

This repository is the source of truth. Each phase is a folder here:

- [`arkguru-common`](arkguru-common) — shared schema, datastore, chunking
- [`arkguru-pdf-extraction`](arkguru-pdf-extraction) — PDF → Chunks
- [`arkguru-web-scraping`](arkguru-web-scraping) — URLs → Chunks
- [`arkguru-rag-slm`](arkguru-rag-slm) — Chunks → Fine-tune + RAG
- [`arkguru-ui`](arkguru-ui) — local wizard

Contributions welcome. Open issues and pull requests against this repo.

---

## License

MIT (see `arkguru-common/pyproject.toml`)

---

## FAQ

**Q: Do I need Postgres?**
A: No. Start with local files (`data/store/*.jsonl` or Phase 1 folders). Postgres is optional for scale and team sharing. File mode uses Python RRF / BM25; Postgres mode uses SQL `search_chunks()`.

**Q: Where do Phase 2 web chunks go?**
A: Always `data/processed/web_chunks.jsonl`. With `--sink postgres`, also schema `web` on the same database (`web.chunks`, `web.search_chunks()`). They never land in `public.chunks`. `--sink postgres` refuses `schema: public`.

**Q: How does hybrid retrieve work with Postgres?**
A: `ensure_schema()` installs `search_chunks()`. With `PG_DSN` set, `retrieve.py` / `serve.py` embed the query, call that function (HNSW + GIN FTS fused with RRF), then rerank and expand parents in Python. If the function is missing, `ChunkStore.search_chunks` runs `ensure_schema()` once and retries.

**Q: Can I use Neon / RDS / a local Docker DB instead of Supabase?**
A: Yes, if it is PostgreSQL 14+ with pgvector. Native local is port 5432; Docker compose is port 5433. Run `bash scripts/setup_docker_pg.sh` or `bash scripts/detect_local_pg.sh`, or set `PG_DSN` (see [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md)).

**Q: Can I use a different LLM?**
A: Phase 3 fine-tune defaults to `Qwen/Qwen2.5-3B-Instruct`. Serving uses whatever Ollama tag is in `serve.model_tag` (default `domain-slm`). Swap `base_model` and re-fine-tune if you want a new adapter.

**Q: Can I add a GPU later?**
A: Yes. CPU fine-tuning code is isolated in `finetune_qlora.py`; the Intel stacks are commented. Inference via Ollama is already GPU-capable.

**Q: How do I update my corpus?**
A: Re-run Phase 1/2 on new PDFs/URLs. Upserts key by `chunk_id` and do not bump `created_at`. Phase 3 embeds only rows still missing from `chunk_embeddings`. No re-fine-tune required to retrieve the new PDFs.

**Q: Can I use this for non-English?**
A: The chunk schema includes `lang`. Phase 1 tags language with `langdetect` when installed. Embedder `bge-m3` supports 100+ languages. The reranker in config is `BAAI/bge-reranker-v2-m3` (multilingual).

**Q: What's the difference between `parent_child` and `semantic` chunking?**
A:
- `parent_child`: Each section gets a large parent chunk plus small children. Retrieval hits children; serve can expand to the parent.
- `semantic`: *Intended* to split where adjacent-sentence similarity drops. **Not wired in Phase 1 today** (`pipeline` never passes `semantic_embedder`), so choosing `semantic` behaves like packed sentence windows.

Start with `structure` (Phase 1 config default). `run_pdfs` defaults to `parent_child`.

**Q: How do I evaluate my system?**
A: `eval_ragas.py` computes context precision/recall/faithfulness/answer relevancy over `phase3_rag/golden/golden_qa.jsonl`. `eval_retrieval.py` is retrieval-only (no Ollama). `make golden` rebuilds the stratified set.

**Q: Is YouTube ingestion available?**
A: No. Planned work is tracked in epic [#39](https://github.com/ravidsun/arkguru-pdf-converter/issues/39) (playlist acquire → transcript → PDF → existing Phase 1 path).

**Q: Is Firecrawl available?**
A: No. Phase 2 `backend` may say `firecrawl` in YAML, but `crawl()` is local httpx + trafilatura only.

---

## Quick links

- [Local run](docs/LOCAL_RUN.md) — clone layout, venv, smoke test, production-style local stack
- [Cloud run](docs/CLOUD_RUN.md) — Cursor Cloud Agent install, layout, constraints
- [Phase 1 README](arkguru-pdf-extraction/README.md) — detailed extraction, OCR, chunking strategies
- [Phase 2 README](arkguru-web-scraping/README.md) — crawl, extract, near-dedup logic
- [Phase 3 README](arkguru-rag-slm/README.md) — fine-tuning, retrieval, serving, evaluation
- [Common README](arkguru-common/README.md) — shared package
- [UI README](arkguru-ui/README.md) — local wizard
- [Database setup](docs/DATABASE_SETUP.md) — Postgres + pgvector: native, Docker, hosted
- [Hybrid retrieve](docs/DATABASE_SETUP.md#hybrid-retrieve) — SQL `search_chunks()` from `ensure_schema()`
- [`arkguru-common/common/datastore.py`](arkguru-common/common/datastore.py) — `ensure_schema()` + `CREATE FUNCTION search_chunks`
- [compose.yaml](compose.yaml) — local / VPS `pgvector/pgvector:pg16`
- [Epic #39](https://github.com/ravidsun/arkguru-pdf-converter/issues/39) — planned PDF + YouTube ingest automation
