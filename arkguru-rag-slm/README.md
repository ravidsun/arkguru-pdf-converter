# arkguru-rag-slm — Phase 3

Combine the Phase 1 (PDF) and Phase 2 (web) chunk datasets, CPU-QLoRA fine-tune a
small language model, and serve a **fully local** RAG system — no external APIs at
inference. Target hardware: **Asus NUC · Intel Core Ultra 9 185H · 96 GB RAM ·
GPU-ready**. This folder is Phase 3 of the **arkguru-pdf-converter** monorepo.
It consumes output from
[`arkguru-pdf-extraction`](../arkguru-pdf-extraction) and
[`arkguru-web-scraping`](../arkguru-web-scraping).

## Quickstart — test on a couple of PDFs (no Postgres, no fine-tuning)

Want to see the RAG working before setting up Postgres, Ollama, or fine-tuning?
`phase3_rag/quickstart.py` ingests chunk files into a **local, incremental
vector store** and answers questions with hybrid retrieval.

```bash
pip install -r requirements.txt

# ---- EASIEST: one command does Phase 1 + ingest (no manual copying) ----
# Phase 1 is the sibling folder in this monorepo (override with --phase1-repo).
python -m phase3_rag.run_pdfs --pdfs /path/to/your_pdfs --embedder hashing \
    --ask "your question here"
# add more PDFs anytime with the same command (only new chunks are embedded):
python -m phase3_rag.run_pdfs --pdfs /path/to/more_pdfs --embedder hashing
# Postgres two-table handoff (Phase 1 fills `chunks`, then embed_datastore):
#   export PG_DSN=...   # Session pooler on Cloud; see docs/DATABASE_SETUP.md
#   python -m phase3_rag.run_pdfs --pdfs /path/to/your_pdfs --sink postgres \
#       --embedder hashing
# on your NUC, real embeddings + generated answers:
#   ... --embedder sentence_transformer --model domain-slm --chat

# ---- OR the manual two-step (more control) ----
# 1) Produce chunks with Phase 1 (per-PDF folders, not a single pdf_chunks.jsonl):
#    ../arkguru-pdf-extraction/data/processed/<stem>/chunks.jsonl

# 2) Ingest + ask — offline embedder, no model downloads, runs anywhere:
python -m phase3_rag.quickstart \
    --add ../arkguru-pdf-extraction/data/processed/sample_handbook/chunks.jsonl \
    --embedder hashing --ask "your question here"

# 3) Add MORE PDFs later — only new chunks are embedded (idempotent by chunk_id):
python -m phase3_rag.quickstart --add data/processed/new_batch.jsonl \
    --embedder hashing --ask "another question"

# 4) On your NUC: switch to REAL embeddings + a local LLM for generated answers:
python -m phase3_rag.quickstart \
    --add ../arkguru-pdf-extraction/data/processed/sample_handbook/chunks.jsonl \
    --embedder sentence_transformer --model-name BAAI/bge-m3 \
    --model domain-slm --chat
```

**Two embedder backends** (`phase3_rag/embedder.py`): `hashing` is
dependency-free (numpy only), deterministic, fixed-dimension — perfect for
offline testing of the pipeline; `sentence_transformer` uses real `bge-m3`
embeddings on your machine. **Answers:** if a local Ollama model is given via
`--model`, the quickstart generates a grounded answer; otherwise it prints the
retrieved context (extractive) so you can verify retrieval with zero model setup.

**Verified end-to-end** (offline embedder, 3 sample PDFs): ingest 2 PDFs → ask →
correct chunk ranks #1; add a 3rd PDF later → 2 new chunks appended, its content
retrievable; re-ingest the first batch → 0 added (idempotent). The store persists
to `data/store/index.{npz,jsonl}`.

This local store is for testing and small corpora; for production/scale, fill
**pgvector** with `embed_datastore` (chunks already in DB) or `index.py` (JSONL
corpus → Postgres). `make index` is that Postgres path, not `data/store/*.npz`.

---

## Postgres + pgvector datastore (two separate tables)

> **Setting up the database?** See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md) for step-by-step Postgres + pgvector setup on a local machine (Docker/Homebrew/apt) or a VPS (remote access, firewall, SSL/SSH tunnel).

The datastore separates chunk data from vectors into **two tables**:

- `chunks` — text + metadata (the JSONL/source-of-truth) + a full-text
  (`tsvector`) column for lexical search. **No vectors here.**
- `chunk_embeddings` — the embeddings only, keyed by `chunk_id`
  (FK -> `chunks`, `ON DELETE CASCADE`), with the HNSW cosine index.

**All datastore settings live in one place — `config/datastore.yaml`** (DSN via
the `PG_DSN` env var, table names, and embedding dim). No connection details are
hardcoded in code. To switch database or tables, edit that one file; to switch
the connection, set `PG_DSN` (or copy `.env.example` → `.env`). Env vars
(`PG_DSN`, `DATASTORE_BACKEND`, `DATASTORE_CONFIG`) override the file.

Phases 1/2 upsert into `chunks`; Phase 3 fills `chunk_embeddings`; retrieve is
one SQL function, `search_chunks()` (installed by `ensure_schema()`): HNSW cosine
on `chunk_embeddings` fused with GIN `ts` on `chunks` via RRF (`is_parent =
false`). Python still embeds the query, then reranks and expands parents.
`search_dense` / `search_lexical` remain for debugging a single leg. Because
vectors live apart, you can **re-embed with a different model by truncating just
`chunk_embeddings`** — the chunk rows are never touched.

```bash
export PG_DSN=postgresql://user:pass@localhost:5432/rag
# Phase 1/2 already wrote the `chunks` table (`chunk_index` lives there, 0-based).
# Fill the `chunk_embeddings` table (vectors only — no chunk_index column):
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024          # Cloud-safe
# NUC / production embeddings:
python -m phase3_rag.embed_datastore --embedder sentence_transformer --model BAAI/bge-m3 --dim 1024
# retrieve.py / serve.py then call ChunkStore.search_chunks (SQL RRF).
```
`embed_datastore.py` is idempotent — it only embeds child rows still missing from
`chunk_embeddings`, so adding more PDFs later just means re-running it. Embeddings
are disposable and model-specific; the chunks are the durable source of truth.

## How it works (the logic)

Six stages, each its own module in `phase3_rag/`.

**0. Combine (`prepare_dataset.py`).**
Loads any number of Phase 1/2 JSONL/Parquet files, globally deduplicates, and
writes two things: `corpus.jsonl` (the flat retrieval corpus) and
`train.jsonl` / `val.jsonl` (instruction pairs). Two generation modes:
`self_supervised` (deterministic template pairs, no model needed) and
`synthetic_qa` (drafts Q&A per chunk via a local Ollama model).

**1. Fine-tune (`finetune_qlora.py`).**
A standard PEFT/transformers LoRA loop. The **only** CPU-specific part is 4-bit
model loading; pick one stack:

| Stack | Notes |
|---|---|
| **Intel Extension for Transformers (ITREX)** | first to ship CPU QLoRA (INT4/NF4); mature |
| **ipex-llm** *(recommended)* | Intel's actively-maintained LLM library; CPU + Intel GPU/NPU |

A portable bf16 fallback (no special deps) is included for smoke tests. Default
base model `Qwen2.5-3B-Instruct`; LoRA rank 16 / alpha 32 on the attention
projections. Expect ~12–24 h for a 3B model on this NUC; a GPU later cuts that to
1–2 h with a one-line switch to `bitsandbytes` + `device_map="cuda"`.

> **Fine-tuning teaches style, format, and vocabulary — not facts reliably.**
> Retrieval (below) supplies the facts. A cheaper first move is **RAG-only on the
> off-the-shelf 3B**, then add fine-tuning where RAGAS shows gaps. This repo
> supports both.

**2. Index (`index.py`).**
Upserts a JSONL corpus into the **`chunks` table**, embeds with a local model
(`bge-m3` default), and writes vectors to **`chunk_embeddings`** (HNSW). Lexical
search uses the `tsvector` on `chunks`. `--embed-only` fills missing vectors when
chunks are already in Postgres (same as `embed_datastore`). Use `--embedder
hashing` on Cloud; `sentence_transformer` on the NUC. Missing embedding rows are
streamed with a server-side cursor.

**3. Retrieve (`retrieve.py`).** Per query when `PG_DSN` is set: embed in Python
→ one SQL call `search_chunks()` (HNSW + GIN FTS, RRF, `is_parent = false`) →
**cross-encoder rerank** (`BAAI/bge-reranker-v2-m3` in `config.yaml`) → `top_k_final` → optional
**parent-chunk expansion** (swap a matched child for its larger parent so the LLM
gets full surrounding context). Connect failures **raise** (fail-closed). File/npz
mode still fuses with Python `reciprocal_rank_fusion`; JSONL retrieve is only used
when no DSN is configured.

**4. Serve (`serve.py`).**
Local chat over **Ollama**. Retrieves context, builds a grounded prompt that
instructs the model to answer only from context and cite `[source_id p.page]`.
Generated answers pass a lexical **faithfulness** gate (retry once, then refuse).
If Ollama is down, the top retrieved passage is returned extractive.

**5. Evaluate (`eval_ragas.py`).**
Runs **RAGAS** — context precision, context recall, faithfulness, answer
relevancy — over a golden Q&A set (`phase3_rag/golden/golden_qa.jsonl`, seed with
50+ pairs). Run it before and after fine-tuning to prove the change earned its keep.

---

## How-to guide

### Prerequisites
- From the umbrella repo, `scripts/setup_dev_env.sh` only installs Phase 3
  `numpy` + `rank-bm25`. For embeddings, LoRA, RAGAS, or `HybridRetriever`,
  run `pip install -r requirements.txt` (CPU QLoRA stacks stay commented).
- **Ollama** installed and running (`https://ollama.com`) for generated answers / eval.
- **PostgreSQL + pgvector — optional.** Retrieval picks the mode from `PG_DSN`:
  - **Database mode** — if `PG_DSN` is set, `create_retriever()` uses
    `HybridRetriever` (SQL `search_chunks`, then rerank). Connect failures **raise**
    (fail-closed). `HybridRetriever` imports `sentence_transformers`.
    ```bash
    export PG_DSN=postgresql://rag:change-me@127.0.0.1:5433/rag
    ```
  - **File mode** — if **no** DSN is configured, `JsonlRetriever` reads
    `**/*.jsonl` under `--processed-dir` (default `data/processed`). A set-but-down
    DSN does **not** fall back.

  See [docs/DATABASE_SETUP.md](../docs/DATABASE_SETUP.md) (canonical) or the
  copy in this folder.

### 1. Bring in the data
**Option A — files.** Point at the sibling Phase 1/2 outputs (or copy them):
```
../arkguru-pdf-extraction/data/processed/<stem>/chunks.jsonl
../arkguru-web-scraping/data/processed/web_chunks.jsonl
```
Phase 1 writes **per-PDF folders**, not a single `pdf_chunks.jsonl`.
**Option B — shared datastore (no copying).** If Phases 1/2 ran with
`--sink postgres`, the chunks are already in the Postgres + pgvector table; skip
the copy and go to step 4's datastore path. See *Postgres + pgvector datastore*
above.

### 2. Build the training + retrieval datasets
```bash
# make combine still passes data/processed/pdf_chunks.jsonl and web_chunks.jsonl
# (missing files are skipped). Prefer the real Phase 1 folder paths:
python -m phase3_rag.prepare_dataset \
  --inputs ../arkguru-pdf-extraction/data/processed/sample_handbook/chunks.jsonl \
           ../arkguru-web-scraping/data/processed/web_chunks.jsonl \
  --out-corpus data/processed/corpus.jsonl \
  --out-train data/processed/train.jsonl \
  --mode self_supervised --val-frac 0.1
# -> data/processed/train.jsonl, val.jsonl, corpus.jsonl
```

### 3. Fine-tune (optional but recommended after RAG works)
```bash
make finetune          # writes artifacts/lora-adapter/
```
`finetune_qlora.py` writes `artifacts/lora-adapter/`. The header of
`phase3_rag/serve.py` sketches merge → GGUF → `ollama create domain-slm`.
There is **no** `phase3_rag.merge_and_export` module in this repo.

### 4. Index for retrieval
```bash
export PG_DSN=postgresql://user:pass@localhost:5432/rag

# Option A: JSONL corpus → Postgres chunks + embeddings (not the local .npz store)
make index
# Cloud-safe hashing, or skip JSONL when Phase 1 already used --sink postgres:
python -m phase3_rag.index --embed-only --embedder hashing

# Option B (shared datastore): fill embeddings for chunks already in the table
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024   # make embed-db
# NUC:
python -m phase3_rag.embed_datastore --embedder sentence_transformer \
    --model BAAI/bge-m3 --dim 1024
```
`embed_datastore` only embeds child rows still missing from `chunk_embeddings`.

**File mode:** skip Postgres. `run_pdfs` / `quickstart` write `data/store/index.{npz,jsonl}`.
`JsonlRetriever` is used when **no** `PG_DSN` is set.

Daily dump when the watermark moved: `make backup` (`python -m phase3_rag.backup --once`).

### 5. Chat locally
```bash
make serve                     # uses Postgres if available, else data/processed/*.jsonl
# you> What PPE is required before servicing a unit?
# assistant> ... [sample_handbook.pdf p.1]
```
`serve.py` calls `create_retriever()`, so the same command works with or without
a database — it logs which mode it chose on startup.

### 6. Evaluate quality
```bash
make eval              # RAGAS metrics over the golden set
```
Edit `phase3_rag/golden/golden_qa.jsonl` to hold 50+ `{"question","answer"}`
pairs from your domain.

---

## Autonomous workers & orchestrator

Run the whole system as deterministic background workers (no LLM) coordinated by
a small DAG orchestrator.

**Phase 3 worker** — polls the datastore and embeds any chunks that lack vectors.
`--once` (Cloud / orchestrator) defaults to the hashing embedder;
a long-running daemon defaults to `sentence_transformer` for the NUC:
```bash
python -m phase3_rag.worker --once                 # hashing
python -m phase3_rag.worker --interval 60          # sentence_transformer (make worker)
python -m phase3_rag.worker --once --embedder hashing
```

**Orchestrator** (`orchestrator.py` + `orchestrator.yaml`) — runs the phase
workers as a dependency-ordered pipeline: `ingest_pdfs` → `index_embeddings`
(and `crawl_web` when enabled). `index_embeddings` depends only on PDF ingest
while web crawl is disabled, so embeddings still run after a PDF-only pass.
`ingest_pdfs` uses `--sink postgres` when `PG_DSN` is set. `index_embeddings`
uses `--once --embedder hashing`. Steps run in topological order; a step is
skipped if a dependency failed, so you never embed chunks that weren't ingested.
It shells out to the nested phase folders (`arkguru-pdf-extraction`,
`arkguru-web-scraping`). Adjust `cwd` in `orchestrator.yaml` if you move them.

```bash
python orchestrator.py --once                  # one full pass   (make orchestrate-once)
python orchestrator.py --interval 300          # loop every 5 min (make orchestrate)
```
Everything funnels through the shared datastore, so the orchestrator only has to
run the phases in the right order.

**File mode (no database):** the `index_embeddings` step is Postgres-only — set
`enabled: false` on it in `orchestrator.yaml`. Phases 1/2 then just write JSONL to
`data/processed/`, and `serve.py` reads those directly via the file fallback.

### End-to-end on a single PDF

Drop a PDF in `arkguru-pdf-extraction/data/raw_pdfs/`, then run the pipeline one
of two ways (both coordinated by the orchestrator, no LLM):

**Production (Postgres):** with `PG_DSN` set and `sink: postgres`, run
`python orchestrator.py --once`. Phase 1 upserts chunks to the shared datastore,
the Phase 3 worker embeds them, and `make serve` answers with pgvector hybrid
retrieval — every phase funnels through the one database.

**Offline / no database (file mode):**
```bash
python orchestrator.py --once --config orchestrator.filemode.yaml   # make e2e
```
Phase 1 ingests the PDF to `data/processed/*.jsonl`; Phase 3 (`serve.py --ask`)
retrieves and **generates a grounded, cited answer via Ollama** — and if Ollama
isn't running it degrades gracefully to an *extractive* answer (the top passage),
so the pipeline still produces output offline. Retrieval auto-selects Postgres or
the JSONL file fallback. *(Phase 2 is the independent web agent and stays disabled
for a PDF-only input.)* Verified end-to-end (Ollama absent -> extractive):

```
[ingest_pdfs] field_manual.pdf -> 6 chunks -> data/processed/field_manual.jsonl
[answer_demo] assistant> Error code E14 indicates a communication fault between the
              main board and display module. Reseat the ribbon cable; if it
              persists, replace harness 55-2210.   sources: field_manual.pdf p1
pipeline summary: {'ingest_pdfs': 'ok', 'answer_demo': 'ok'}
```
Start Ollama (and set `serve.model_tag`) to get a written answer instead of the
extractive passage. The file-mode BM25 fallback is a lightweight convenience; for
best ranking use Postgres + embeddings.

## Running across machines

The phases are decoupled by two things — the portable `Chunk` schema and the
shared Postgres datastore — so you can spread them across machines. Three common
topologies:

### A. Single machine (dev / all-in-one)
This monorepo already contains all three phase folders. From `arkguru-rag-slm`:
```bash
cd arkguru-rag-slm
make e2e                                   # offline: one PDF -> answer
# or with a local Postgres:
export PG_DSN=postgresql://rag:change-me@127.0.0.1:5433/rag
python orchestrator.py --once
```

### B. Cross-machine via a shared datastore (recommended for production)
Run ingestion wherever it's convenient (a cloud VM with a fast CPU for OCR) and
serving on your NUC. Both point `PG_DSN` at **one** Postgres — on a VPS or the NUC
(see [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md)).

Ingestion machine (cloud / other) — from the Phase 1 / Phase 2 folders in this repo:
```bash
export PG_DSN="postgresql://rag:pass@db-host:5432/rag?sslmode=require"
cd arkguru-pdf-extraction
python -m phase1_pdf.pipeline --init-db        # first time: create tables
python -m phase1_pdf.worker  --sink postgres --interval 60      # watch + ingest
cd ../arkguru-web-scraping
python -m phase2_web.worker  --sink postgres --interval 3600    # scheduled crawl
```
Serving machine (NUC) — from `arkguru-rag-slm`:
```bash
export PG_DSN="postgresql://rag:pass@db-host:5432/rag?sslmode=require"
python -m phase3_rag.worker --interval 60      # embed new chunks as they arrive
python -m phase3_rag.serve                     # answer (Ollama)
```
Everything funnels through the datastore, so **no files move between machines** —
each side just runs its own worker(s) on a schedule (systemd / cron / Docker).

> The `orchestrator.py` shells out to the nested phase folders, so it's meant
> for the single-machine case. In a distributed setup the **shared database is
> the coordinator** — run each phase's worker on its own host.

### C. Portable files (no shared database)
Phase 1/2 emit JSONL/Parquet; copy them to the serving machine and use file mode.
```bash
# machine A (ingestion) — per-PDF folders, not a flat *.jsonl
cd ../arkguru-pdf-extraction
python run_phase1.py --out-dir data/processed
scp -r data/processed/*  nuc:~/arkguru-pdf-converter/arkguru-rag-slm/data/processed/
# NUC (serving) — file-mode fallback, no DB
cd ~/arkguru-pdf-converter/arkguru-rag-slm
python -m phase3_rag.serve --ask "What does error code E14 mean?" --processed-dir data/processed
```

### Hardware notes
- **Ingestion (Phases 1–2)** is CPU-bound; OCR is the heavy part, so a larger
  cloud CPU helps. No GPU needed. Outputs are portable (JSONL/Parquet or the DB).
- **Serving / fine-tuning (Phase 3)** targets the NUC (Core Ultra 9 185H, 96 GB).
  Embeddings + reranker run on CPU today; a future GPU speeds them up
  (`device="cuda"`, and vLLM for serving).
- **Postgres** can live on the NUC (simplest) or a small VPS so cloud ingestion
  can reach it — prefer an **SSH tunnel** over exposing 5432 (see DATABASE_SETUP.md).

## Configuration (`config/config.yaml`)
Key knobs: `base_model`, `lora.{rank,alpha}`, `embedding_model`,
`reranker_model`, `datastore_config`, `retrieval.{top_k_vector,top_k_bm25,
top_k_final,use_parent_expansion}`, `serve.{backend,model_tag,ctx,faithfulness}`.

### Fine-tuning configuration

**LoRA parameters** (from `config/config.yaml`):
- `rank`: 16; `alpha`: 32; `dropout`: 0.05
- `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj"]`

**Model selection:**
- `Qwen/Qwen2.5-3B-Instruct` *(default in config)* — balanced, ~12–24h on NUC
- Any other Hugging Face causal LM you set as `base_model` (you must have the weights)

**Dataset split:**
- `prepare_dataset --val-frac 0.1` (default). There is no `train_ratio` or `max_samples` key.

**Generation strategy for synthetic QA pairs:**
- `self_supervised`: Deterministic template pairs (`instruction` = section/title, `output` = chunk text). No model needed.
- `synthetic_qa`: Drafts Q&A per chunk via local Ollama. Slower; failures are logged and skipped.

### Retrieval configuration

**Hybrid search parameters** (from `config/config.yaml`):
- `top_k_vector`: dense hits passed to SQL `search_chunks` as `k_dense` (default **60**)
- `top_k_bm25`: lexical hits passed as `k_lexical` (default **60**)
- `top_k_candidates`: union size the cross-encoder reranks (default **120**)
- `top_k_final`: after SQL RRF + Python rerank (default **6**)
- `rewriter`: `noop` (default) \| `lexicon` \| `ollama`
- Keep `datastore.yaml` `hnsw_ef_search` (200) ≥ `top_k_vector`

**Ranking:**
- Postgres: SQL RRF inside `search_chunks()` (`rrf_k=60`), then cross-encoder rerank (`BAAI/bge-reranker-v2-m3`)
- File/npz (`quickstart` / `run_pdfs`): Python `reciprocal_rank_fusion` in the local store
- JSONL fallback (`JsonlRetriever`, no `PG_DSN`): in-process BM25 + vectors, then the same reranker
- There is no `use_cross_encoder_rerank` flag in `config.yaml`
- `use_parent_expansion`: swap a retrieved child for its parent section when present
- `serve.faithfulness`: lexical overlap gate (`enabled`, `min_token_overlap`, `min_sentence_tokens`, `max_retries`)

**Example: high-precision vs. high-recall:**
```yaml
# High precision
retrieval:
  top_k_vector: 10
  top_k_bm25: 8
  top_k_final: 3

# High recall
retrieval:
  top_k_vector: 30
  top_k_bm25: 25
  top_k_final: 10
```

### Serving configuration

- `serve.backend` in YAML is `"ollama"`. `serve.py` always calls Ollama at `http://localhost:11434`; there is no API/OpenAI backend and no `temperature` / `top_p` flags.
- `model_tag`: Ollama model name (default `domain-slm`)
- `ctx`: context window in tokens (default **4096**)
- CLI: `python -m phase3_rag.serve [--config] [--processed-dir] [--ask]`

### Embedder options

**`sentence_transformer`:**
- Model: `BAAI/bge-m3` (default, 1024-dim), `mixedbread-ai/mxbai-embed-large` (1024-dim)
- Speed: ~100–300 queries/s on CPU, ~1000+/s on GPU
- Quality: SOTA for dense retrieval
- Cost: One-time download (~2GB), then zero inference cost

**`hashing` (offline testing):**
- No model download, runs anywhere
- Speed: 100K+ queries/s
- Deterministic and reproducible
- Good for pipeline testing; NOT for production (low quality)

## Troubleshooting

### `index` / `serve` can't connect
→ check `PG_DSN` and that pgvector extension is created; confirm Ollama is running on `:11434`.

```bash
# Test PG connection:
psql $PG_DSN -c "SELECT 1"

# Test Ollama:
curl http://localhost:11434/api/tags

# Check pgvector extension:
psql $PG_DSN -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Slow inference
→ expected on CPU (~5–20 tok/s); use a smaller base model or add a GPU. Fine-tuning can't speed this up (inference speed is model size + hardware). Workarounds:
- Switch to a 1.5B model (Phi-2, MiniChat)
- Add a GPU (even an old GTX 1060 = 10x speedup)
- Use a shorter context window (`ctx: 1024`)
- For high throughput, use vLLM (once you add GPU)

### Answers not grounded
→ increase `top_k_final`, enable `use_parent_expansion`, or improve chunk quality upstream in Phase 1/2. Also check:
```bash
# Inspect what's being retrieved (no --verbose flag):
python -m phase3_rag.retrieve "your question here"
```

This prints the ranked chunks so you can see if good chunks are even available.

### Embedding query times out
→ `bge-m3` is large (~1GB); first query downloads the model. Subsequent queries cache it. If timeout happens repeatedly, increase `sentence_transformer` batch size in `index.py` or switch to `hashing` for testing.

### Fine-tuning crashes with OOM
→ reduce `lora.rank` or `train.batch_size` in `config/config.yaml`. Or enable gradient checkpointing in `finetune_qlora.py`:
```python
model.gradient_checkpointing_enable()  # trade compute for memory
```

### Re-embed with a different model
→ truncate only the `chunk_embeddings` table; chunks are untouched and reusable:
```bash
psql $PG_DSN -c "TRUNCATE TABLE chunk_embeddings;"
python -m phase3_rag.embed_datastore --embedder sentence_transformer --model mixedbread-ai/mxbai-embed-large --dim 1024
```

## Advanced: Evaluation metrics (RAGAS)

The `eval_ragas.py` module computes:

- **Context Precision**: What % of retrieved chunks actually contain answer nuggets? (lower noise)
- **Context Recall**: What % of chunks needed for a correct answer were retrieved? (lower miss rate)
- **Faithfulness**: Does the generated answer contradict the context? (lower hallucination)
- **Answer Relevancy**: Is the answer on-topic? (lower drift)

Run before and after fine-tuning to prove the change earned its keep:
```bash
python -m phase3_rag.eval_ragas --golden phase3_rag/golden/golden_qa.jsonl
```

**Bootstrapping the golden set:**
1. Start with 5–10 high-value question–answer pairs (domain expert curated)
2. Run `prepare_dataset.py` with `synthetic_qa` to generate 50+ pairs
3. Manually review and fix any hallucinations
4. Expand to 100+ pairs for more stable metrics

## GPU migration (future)

Fine-tune → `bitsandbytes` 4-bit + `device_map="cuda"`; serving → **vLLM** for
much higher throughput; embeddings → same libs with `device="cuda"`. Because the
CPU-specific lines are isolated and commented, migration is config + a few
uncommented lines, not a rewrite.

## Tests

```bash
source ../.venv/bin/activate
make test          # python -m pytest tests -q
```

Retrieval-only eval (no Ollama): `make eval-retrieval`. RAGAS eval: `make eval`
(needs the full `requirements.txt` stack + a reachable generator).
