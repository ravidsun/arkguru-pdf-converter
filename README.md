# arkguru — A 3-Phase Local RAG System

**arkguru** is an end-to-end retrieval-augmented generation (RAG) pipeline that extracts knowledge from PDFs and websites, fine-tunes a small language model, and serves a fully local chat system — no external APIs at inference time.

**Target hardware:** Asus NUC with Intel Core Ultra 9 185H, 96 GB RAM, GPU-optional  
**CPU-only fine-tuning:** ✓ Yes, with Intel Extension for Transformers or ipex-llm  
**GPU migration:** ✓ Yes, config + uncommented lines only  
**Local inference:** ✓ Yes, via Ollama (llama.cpp)

**How to run:**

- [docs/LOCAL_RUN.md](docs/LOCAL_RUN.md) — laptop / NUC / workstation (venv, file sink, optional Postgres + Ollama)
- [docs/CLOUD_RUN.md](docs/CLOUD_RUN.md) — Cursor Cloud Agent (`.cursor/environment.json` + `scripts/setup_dev_env.sh`)
- Hybrid retrieve implementation: [`arkguru-common/common/datastore.py`](arkguru-common/common/datastore.py) — `ensure_schema()` installs SQL `search_chunks()`

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
      │  - Fine-tune small LM (LoRA)         │
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

---

## Phase 1: PDF Extraction (`arkguru-pdf-extraction`)

**Input:** Folder of PDFs (native, scanned, or mixed)  
**Output:** Per-PDF folders of JSONL (or Parquet) under `data/processed/{stem}/`  
**Speed:** native ~3–5 MB/s; scanned ~0.1 MB/s (OCR). See [Hardware](#hardware--performance).

**Key features:**
- **Three extraction backends:** pymupdf4llm (default, fast), docling (complex tables), pymupdf (fallback)
- **Scanned PDF OCR:** Detects image-only pages and OCRs them in-place; native text layers untouched
- **Table extraction:** Preserves structure (rows, columns) as standalone markdown chunks
- **Figure OCR:** Extracts text from diagrams/charts (axis labels, legends)
- **Three chunking strategies:** `structure` (default, heading windows), `parent_child` (long parent + small children). `semantic` is a CLI/config value but Phase 1 does not pass an embedder, so it uses the same windows as `structure`.

**Example workflow:**
```bash
cd arkguru-pdf-extraction
python scripts/make_sample_pdf.py     # create test PDF
make phase1                           # extract & chunk
# → data/processed/sample_handbook/chunks.jsonl
#    (+ parents.jsonl / tables.jsonl / figures.jsonl when non-empty)
```

**Configuration highlights:**
```yaml
backend: "pymupdf4llm"    # fast native-text PDFs
backend: "docling"        # complex tables (slower)
strategy: "parent_child"  # retrieve small chunks, expand to parent for context
target_tokens: 400        # ~300–500 tokens per chunk (tuned for retrieval)
ocr_enabled: true         # safe on mixed documents
extract_figures: true     # OCR text in images
```

**System dependencies (optional OCR):**
```bash
# macOS
brew install tesseract ghostscript

# Ubuntu/Debian
sudo apt-get install tesseract-ocr ghostscript

# Windows: Download from GitHub + Ghostscript website, add to PATH
```

---

## Phase 2: Web Scraping (`arkguru-web-scraping`)

**Input:** List of seed URLs  
**Output:** Chunks from crawled pages (JSONL/Parquet, same schema as Phase 1)  
**Speed:** ~200–500 ms/page (local backend), ~1–3 s/page (firecrawl)

**Key features:**
- **Two fetch backends:** local (trafilatura + httpx, free), firecrawl (JS-heavy sites, paid)
- **Breadth-first crawl:** BFS frontier with link discovery up to max_pages
- **Structure-aware chunking:** Same `pack_windows` logic as Phase 1 → identical chunk sizes
- **Near-dedup:** MinHash LSH collapses syndicated/similar pages (Jaccard ≥ 0.9)

**Example workflow:**
```bash
cd arkguru-web-scraping
# Point at your docs site
python -m phase2_web.pipeline --seeds https://your.site/docs --max-pages 200
# → data/processed/web_chunks.jsonl
```

**Backend decision:**
| Use | If |
|---|---|
| `local` | Static sites, traditional CMS (WordPress), internal wikis, documentation |
| `firecrawl` | Single-page apps, JavaScript frameworks, React/Vue, dynamic content |

**Configuration highlights:**
```yaml
seeds: ["https://docs.example.com"]
max_pages: 200              # hard cap on crawl size
same_domain_only: true      # prevent crawl explosion
backend: "local"            # local | firecrawl
target_tokens: 550          # slightly larger for web (more boilerplate)
min_content_chars: 200      # skip nav-only pages
```

---

## Phase 3: RAG + Fine-tuning (`arkguru-rag-slm`)

**Input:** Combined chunks from Phase 1 + Phase 2, (optional) Postgres + pgvector  
**Output:** Fine-tuned model + vector index, grounded answers via local LLM  
**Speed:** Fine-tune 3–50h (CPU), Inference 5–20 tok/s (CPU), 100+ tok/s (GPU)

### Quickstart (no database, minimal setup)

Test the full pipeline locally in minutes:

```bash
cd arkguru-rag-slm
pip install -r requirements.txt

# One command: extract Phase 1 PDFs, ingest, then chat
python -m phase3_rag.run_pdfs --pdfs /path/to/your_pdfs --embedder hashing \
    --ask "your question here"

# On production hardware (with local LLM), switch embedder + model:
python -m phase3_rag.run_pdfs --pdfs /path/to/pdfs --embedder sentence_transformer \
    --model domain-slm --chat
```

### Full production setup

#### 1. Prepare data
```bash
# Copy Phase 1 per-PDF folders (or concatenated jsonl) and Phase 2 web_chunks.jsonl:
#   data/processed/<stem>/chunks.jsonl
#   data/processed/web_chunks.jsonl

# Combine, deduplicate, generate training pairs:
make combine
# → data/processed/train.jsonl, val.jsonl, corpus.jsonl
```

#### 2. Set up Postgres + pgvector (optional, for scale)

For corpora >10K chunks or to share with Phase 1/2:

```bash
# One-click Docker Postgres (port 5433) + write .env:
bash scripts/setup_docker_pg.sh
# Native already running (port 5432) or Docker already up:
# bash scripts/detect_local_pg.sh

# Hosted: paste a Supabase session-pooler URI, or Neon / Crunchy / RDS / …
# export PG_DSN='postgresql://user:pass@HOST:5432/DB?sslmode=require'

# Create extension (run once; the app also does this):
psql "$PG_DSN" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Tables + search_chunks() are created by ensure_schema() on first write
```

Switch hosts by changing `PG_DSN` only. The host must be PostgreSQL 14+ with pgvector. `ensure_schema()` also installs the SQL function `search_chunks()` used for hybrid retrieve.

See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md), [docs/LOCAL_RUN.md](docs/LOCAL_RUN.md#switch-database-pg_dsn), or [compose.yaml](compose.yaml).

#### 3. Fine-tune (optional but recommended)

```bash
make finetune
# → artifacts/lora-adapter/ (12–24h on NUC CPU, depends on dataset size + model)
```

Then merge, convert to GGUF, quantize to Q4_K_M, and register with Ollama:
```bash
# Commands in phase3_rag/serve.py header; produces `domain-slm` model tag
```

#### 4. Index for retrieval

```bash
# Option A: chunks already in Postgres (Phase 1 --sink postgres)
python -m phase3_rag.embed_datastore --embedder sentence_transformer \
    --model BAAI/bge-m3 --dim 1024

# Option B: JSONL corpus → Postgres chunks + embeddings
make index
# → python -m phase3_rag.index (not the local .npz store)
```

#### 5. Chat locally

```bash
# Start Ollama (if not already running)
ollama serve &

make serve
# you> What PPE is required before servicing a unit?
# assistant> ... [sample_handbook.pdf p.1]
```

With `PG_DSN` set, `serve.py` / `retrieve.py` call `ChunkStore.search_chunks` (SQL RRF of HNSW cosine + GIN full-text, `is_parent = false`), then rerank and expand parents in Python. Without a DSN, file/npz mode keeps Python RRF.

#### 6. Evaluate quality (RAGAS)

```bash
make eval
# Reports context precision, recall, faithfulness, answer relevancy
```

### Key components

| Module | Purpose | Input | Output |
|---|---|---|---|
| `prepare_dataset.py` | Combine + dedup chunks, generate training pairs | JSONL/Parquet from Phase 1/2 | train.jsonl, val.jsonl, corpus.jsonl |
| `finetune_qlora.py` | Fine-tune small LM with LoRA (CPU or GPU) | train.jsonl | lora-adapter/ |
| `embed_datastore.py` | Embed chunks already in Postgres | `chunks` missing vectors | `chunk_embeddings` + HNSW |
| `index.py` | Upsert a JSONL corpus **into Postgres**, then embed | corpus.jsonl + `PG_DSN` | `chunks` + `chunk_embeddings` |
| `vector_store.py` / `run_pdfs` | Local incremental index (no DB) | Phase 1 JSONL | `data/store/index.{npz,jsonl}` |
| `retrieve.py` | Hybrid retrieve: SQL `search_chunks()` (HNSW + FTS RRF) when `PG_DSN` is set; Python RRF for files; then rerank + parent expand | corpus/pgvector + query | top-k ranked chunks |
| `serve.py` | Chat via Ollama, or extractive fallback + faithfulness gate | query + index | grounded answer + sources |
| `backup.py` | `pg_dump` of both tables when `created_at` watermark moved | Postgres | `data/backups/*.dump` |
| `eval_ragas.py` | Evaluate retrieval + generation quality | golden_qa.jsonl + system | RAGAS metrics |

### Configuration

**Main config:** `config/config.yaml`  
**Database config:** `config/datastore.yaml` (DSN, table names, embedding dim)  
**Golden eval set:** `phase3_rag/golden/golden_qa.jsonl` (50+ Q&A pairs)

**Key tuning knobs:**

```yaml
base_model: "Qwen/Qwen2.5-3B-Instruct"
lora:
  rank: 16
  alpha: 32
embedding_model: "BAAI/bge-m3"
reranker_model: "BAAI/bge-reranker-base"
retrieval:
  top_k_vector: 20
  top_k_bm25: 20
  top_k_final: 6
  use_parent_expansion: true
serve:
  faithfulness:
    enabled: true
    min_token_overlap: 0.4
```

**Embedder choices:**
- `hashing`: Offline testing (no downloads, deterministic, low quality)
- `sentence_transformer`: Production (SOTA, ~100–300 queries/s CPU)

### Hybrid retrieve (`search_chunks`)

When `PG_DSN` is set, Phase 3 retrieve is **one SQL function**, `search_chunks()`, created by `ensure_schema()` in `common/datastore.py` (not a hand-written schema file). It fuses:

- **Dense:** HNSW cosine on `chunk_embeddings` (`ORDER BY embedding <=> q LIMIT k_dense`)
- **Lexical:** GIN `ts` on `chunks` (`ts @@ plainto_tsquery` + `ts_rank`)
- **RRF:** `1 / (rrf_k + rank)` with `rrf_k=60`, both legs `is_parent = false`

Python still embeds the query, reranks with `BAAI/bge-reranker-base`, and expands parent chunks. File/npz mode (`run_pdfs` / `quickstart`) keeps Python `reciprocal_rank_fusion`. See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md#hybrid-retrieve).

---

## Data flow and schema

All three phases share a common **`Chunk` schema** (in `common/schema.py`):

```python
text: str                    # chunk body
source_type: "pdf" | "web"   # origin
source_id: str               # PDF filename or URL
chunk_index: int             # sequence in source
title: str                   # document title
section: str                 # section heading
page: int | None             # page number (PDFs)
url: str | None              # full URL (web)
domain: str | None           # domain (web)
lang: str                    # detected language
token_count: int             # chunk size
overlap_tokens: int          # overlap with next chunk
parent_id: str | None        # link to larger context (parent_child strategy)
is_parent: bool              # true if this is a parent chunk
embedding: list[float] | None # nullable; filled by Phase 3
extra: dict                  # block_type: "table" | "figure" (Phase 1)
chunk_id: str                # SHA256(source + index + role); idempotent
```

**Why one schema?**
- Phases 1 and 2 outputs concatenate with zero conversion
- One `chunks` table for all sources; vectors live in `chunk_embeddings`
- Re-embedding is truncating/refilling the embeddings table (vectors are disposable)

**Idempotency:**
All outputs are keyed by `chunk_id`, so re-running is safe:
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
Phase 1 PDFs → .jsonl files → Phase 3 (local index)
Phase 2 URLs → .jsonl files ↗
                          ↓
                   data/store/*.jsonl
                   (vector index in memory/disk)
```

**Pros:** Simple, offline, no database setup  
**Cons:** Single-machine only, limited to ~100K chunks (memory-bound)  
**Best for:** Prototyping, <1GB corpus

### Path B: Postgres + pgvector (production / large corpus)

```
Phase 1 PDFs → Postgres (chunks table)
Phase 2 URLs ↗
       ↓
Phase 3 fills embeddings (chunk_embeddings table)
       ↓
Retrieval: search_chunks() (SQL RRF of HNSW + GIN FTS)
       ↓
Serve via Ollama (local LLM)
```

**Pros:** Scales to millions of chunks, transactional, shareable  
**Cons:** Database setup, network latency (usually <10 ms)  
**Best for:** Production, >1 million chunks, team sharing

### Path C: Hybrid (start file-based, migrate to DB)

1. Develop locally with files (Path A)
2. Once satisfied, spin up Postgres + pgvector (Path B)
3. Phase 1/2 re-run with `--sink postgres` → upsert to DB
4. Phase 3 switches config to read from DB
5. `search_chunks()` hybrid retrieve scales, no reprocessing of PDFs/web

---

## Troubleshooting by phase

### Phase 1 (PDFs)

| Issue | Cause | Fix |
|---|---|---|
| Empty output | Scanned PDF | Enable OCR: `ocr_enabled: true` |
| Garbled tables | Simple grid extraction | Switch backend: `backend: docling` |
| Missing image text | Figures not OCR'd | Enable: `extract_figures: true` |
| Chunks too big/small | Tokenizer miscalibrated | Tune `target_tokens`, `overlap_pct`, `min_tokens` |
| Out of memory | Huge PDF + single worker | Reduce `workers` or split input folder |

### Phase 2 (Web)

| Issue | Cause | Fix |
|---|---|---|
| Blank content | JavaScript-rendered site | Switch backend: `backend: firecrawl` |
| Crawl wandered off-site | `same_domain_only: false` | Set to `true` or limit `max_pages` |
| Slow crawl | Rate-limited by site | Reduce `max_pages` or add delays in `fetch.py` |
| Duplicate content | Syndicated pages + pagination | Ensure `datasketch` installed for MinHash dedup |

### Phase 3 (RAG)

| Issue | Cause | Fix |
|---|---|---|
| Can't connect to Postgres | DSN wrong or DB down | Check `PG_DSN`; retrieve **raises** if DSN is set but connect fails |
| Can't connect to Ollama | Ollama not running | Start: `ollama serve` (extractive answer still works) |
| Generated answer refused | Faithfulness gate | Check retrieved context; gate retries once then refuses ungrounded text |
| Slow inference (CPU) | Normal on CPU | Add GPU, use smaller model (Phi-3, MiniChat), or use vLLM |
| Poor answer quality | Bad chunks or retrieval | Inspect: `python -m phase3_rag.retrieve "question"` |
| Fine-tune OOM | Model + data too large | Reduce `lora.rank`, `batch_size`, or `max_samples` |
| Re-embed with new model | Need to test other embeddings | Truncate `chunk_embeddings`, re-run `embed_datastore` |

---

## Contributing

Each phase is a self-contained repo:
- [`arkguru-pdf-extraction`](https://github.com/ravidsun/arkguru-pdf-extraction) — PDF → Chunks
- [`arkguru-web-scraping`](https://github.com/ravidsun/arkguru-web-scraping) — URLs → Chunks
- [`arkguru-rag-slm`](https://github.com/ravidsun/arkguru-rag-slm) — Chunks → Fine-tune + RAG

Contributions welcome. See individual repos for issue tracking and pull requests.

---

## License

MIT (each repo independently licensed)

---

## FAQ

**Q: Do I need Postgres?**  
A: No. Start with local files (`data/store/*.jsonl`). Postgres is optional for scale (>100K chunks) and team sharing. File mode uses Python RRF; Postgres mode uses SQL `search_chunks()`.

**Q: How does hybrid retrieve work with Postgres?**  
A: `ensure_schema()` installs `search_chunks()`. With `PG_DSN` set, `retrieve.py` / `serve.py` embed the query, call that function (HNSW + GIN FTS fused with RRF), then rerank and expand parents in Python. If the function is missing, `ChunkStore.search_chunks` runs `ensure_schema()` once and retries.

**Q: Can I use Neon / RDS / a local Docker DB instead of Supabase?**  
A: Yes, if it is PostgreSQL 14+ with pgvector. Native local is port 5432; Docker compose is port 5433. Run `bash scripts/setup_docker_pg.sh` (one-click Docker) or `bash scripts/detect_local_pg.sh`, or set `PG_DSN` (see [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md)).

**Q: Can I use a different LLM?**  
A: Yes. Phase 3 defaults to `Qwen2.5-3B-Instruct`, but supports any GGUF in Ollama (Mistral, Llama, Phi, etc.). Swap `base_model` in config and re-fine-tune.

**Q: Can I add a GPU later?**  
A: Yes. CPU fine-tuning code is isolated in `finetune_qlora.py`; swap one backend (Intel Extension → bitsandbytes) and uncomment GPU lines. Inference via Ollama already GPU-capable.

**Q: How do I update my corpus?**  
A: Re-run Phase 1/2 on new PDFs/URLs. Upserts key by `chunk_id` and do not bump `created_at`. Phase 3 embeds only rows still missing from `chunk_embeddings`. No re-fine-tune required to retrieve the new PDFs.

**Q: Can I use this for non-English?**  
A: Yes. Chunk schema includes `lang` field. Phase 1/2 auto-detect language. Phase 3 fine-tuning is language-agnostic (LoRA adapts any base model). Embedder (`bge-m3`) supports 100+ languages.

**Q: What's the difference between `parent_child` and `semantic` chunking?**  
A: 
- `parent_child`: Each section gets a large parent chunk plus small children. Retrieval hits children; serve can expand to the parent.
- `semantic`: *Intended* to split where adjacent-sentence similarity drops. **Not wired in Phase 1 today** (`pipeline` never passes `semantic_embedder`), so choosing `semantic` behaves like packed sentence windows.

Start with `structure` (Phase 1 config default). `run_pdfs` defaults to `parent_child`.

**Q: How do I evaluate my system?**  
A: Phase 3 includes `eval_ragas.py`, which computes context precision/recall/faithfulness over a golden Q&A set. Bootstrap with 50–100 expert-curated pairs, then run before/after fine-tuning to measure improvement.

---

## Quick links

- [Local run](docs/LOCAL_RUN.md) — clone layout, venv, smoke test, production-style local stack
- [Cloud run](docs/CLOUD_RUN.md) — Cursor Cloud Agent install, layout, constraints
- [Phase 1 README](arkguru-pdf-extraction/README.md) — detailed extraction, OCR, chunking strategies
- [Phase 2 README](arkguru-web-scraping/README.md) — crawl, extract, near-dedup logic
- [Phase 3 README](arkguru-rag-slm/README.md) — fine-tuning, retrieval, serving, evaluation
- [Database setup](docs/DATABASE_SETUP.md) — Postgres + pgvector: native, Docker, hosted
- [Hybrid retrieve](docs/DATABASE_SETUP.md#hybrid-retrieve) — SQL `search_chunks()` from `ensure_schema()`
- [`arkguru-common/common/datastore.py`](arkguru-common/common/datastore.py) — `ensure_schema()` + `CREATE FUNCTION search_chunks`
- [compose.yaml](compose.yaml) — local / VPS `pgvector/pgvector:pg16`
