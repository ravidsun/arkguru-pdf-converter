# arkguru-pdf-extraction — Phase 1

Turn a folder of PDFs into clean, structured, portable **`Chunk`** records
(JSONL or Parquet) ready for retrieval and fine-tuning. Runs **local, CPU-only,
no GPU**. This is Phase 1 of a 3-repo system; its output feeds
[`arkguru-rag-slm`](https://github.com/ravidsun/arkguru-rag-slm) (Phase 3)
alongside [`arkguru-web-scraping`](https://github.com/ravidsun/arkguru-web-scraping)
(Phase 2). All three share one record schema, so their outputs concatenate with
no glue.

---

## How it works (the logic)

The pipeline is three stages: **extract → chunk → write**, orchestrated by
`phase1_pdf/pipeline.py`.

**1. Extract (`phase1_pdf/extract.py`).**
Each PDF is converted into a backend-neutral `Document` of `Block`s, where a
block carries `text`, its `page`, `block_type` (`text` | `table` | `figure`),
and the nearest `heading` above it. Three extraction backends produce the same
`Block` shape, so nothing downstream cares which one ran:

| Backend | Best for | Notes |
|---|---|---|
| `pymupdf4llm` *(default)* | native-text PDFs, speed | MuPDF C engine, emits Markdown with headings |
| `docling` | complex tables / layout | IBM table-transformer; slower, heavier deps, best fidelity |
| `pymupdf` | raw fallback | plain per-page text, no structure |

Robustness for messy real-world PDFs is layered on top of every backend
(except `docling`, which already models this natively):

- **Scanned pages, including mixed documents.** Each page is checked for
  extractable text; if *any* page looks image-only, the whole file is routed
  through `ocrmypdf` with `skip_text=True` (not `force_ocr`), so pages that
  already have a native text layer are left untouched while only the
  scanned/image pages get OCR'd. This makes it safe to run on a report that's
  mostly native text but has a few scanned appendix pages.
- **Tables.** `pymupdf`'s `page.find_tables()` extracts each table as a
  markdown block (`block_type: "table"`), attached to the nearest section
  heading, so rows/columns survive instead of being flattened into garbled
  prose.
- **Diagrams / charts / graphs.** Embedded images on each page are OCR'd
  (via `pytesseract`, optional) to pull out any text baked into a figure
  (axis labels, legends, callouts); results become `block_type: "figure"`
  blocks. Silently skipped if `pytesseract`/`Pillow` aren't installed.

A missing optional library degrades gracefully (backend falls back to
`pymupdf`; table/figure extraction is skipped) rather than crashing the run.

**2. Chunk (`phase1_pdf/chunk.py`).**
Prose blocks are grouped under their nearest heading into *sections*, then
packed into windows of ~`target_tokens` (default **400**, tuned smaller than
before for retrieval precision) with `overlap_pct` overlap (default **15%**).
A `min_tokens` floor (default 80) merges any tiny trailing window into its
predecessor instead of shipping a low-context sliver. Sentence splitting
tolerates common abbreviations (`Fig.`, `e.g.`, `Dr.`, ...) so headings and
captions aren't chopped mid-thought. Chunking **never crosses a heading
boundary**, so every chunk is a coherent unit. `table`/`figure` blocks are
never sentence-packed -- each becomes its own standalone chunk (tagged
`extra.block_type`) so structured content stays intact. Three strategies:

- **`structure`** *(default)* — heading-grouped, sentence-packed windows.
- **`parent_child`** — additionally emits one large *parent* chunk per section
  (capped at `parent_max_tokens`). Retrieval later matches small, precise
  *children* but can expand to the parent for full context via `parent_id`.
- **`semantic`** — splits where adjacent-sentence embedding similarity drops into
  the lowest quartile. Phase 1 **does not pass an embedder**, so this falls back
  to the same packed windows as `structure`.

Token sizing uses a fast proxy tokenizer (`tiktoken cl100k_base`) — good enough
for "is this ~300–500 tokens?"; Phase 3 does exact accounting with the base
model's own tokenizer.

**3. Write (`common/schema.py`).**
Chunks are deduplicated per source PDF (exact-text for children; parents
always kept) and written under **`out_dir / {relative_stem} /`** as exclusive
splits (`chunks.jsonl`, plus `parents.jsonl` / `tables.jsonl` / `figures.jsonl`
when non-empty). Nested inputs stay nested (`hvac/manual.pdf` →
`data/processed/hvac/manual/chunks.jsonl`). Each record gets a deterministic
`chunk_id` (SHA-256 of source+index+role) so re-runs are idempotent.

---

## How-to guide

### 1. Install
```bash
git clone https://github.com/ravidsun/arkguru-pdf-extraction.git
cd arkguru-pdf-extraction
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```
The default backend needs only `pymupdf4llm`, `tiktoken`, `pyyaml`. Scanned-PDF
OCR is on by default: `pip install -r requirements.txt` pulls `ocrmypdf`,
`pytesseract`, and `Pillow`. You still need system **Tesseract** and
**Ghostscript**. `docling` stays commented (heavier tables/layout backend).

### 2. Try it on the built-in sample
```bash
python scripts/make_sample_pdf.py     # writes data/raw_pdfs/sample_handbook.pdf
make phase1                            # -> data/processed/sample_handbook/chunks.jsonl
```
You'll see per-file chunk counts and a token-distribution summary
(`p10/p50/p90`).

### 3. Run on your own PDFs
Drop files into `data/raw_pdfs/` (subfolders are searched recursively) and:
```bash
make phase1
# or with explicit options:
python -m phase1_pdf.pipeline \
  --input data/raw_pdfs \
  --out-dir data/processed \
  --strategy parent_child \
  --backend pymupdf4llm \
  --format parquet \
  --workers 0            # 0 = auto (all cores); processes PDFs in parallel
```
Each PDF gets its own output folder, e.g.
`data/raw_pdfs/quarterly_report.pdf` -> `data/processed/quarterly_report/chunks.jsonl`.

**Cloud Agent:** that folder is on the remote VM
(`/agent/repos/arkguru-pdf-converter/arkguru-pdf-extraction/data/raw_pdfs/`), not on your laptop.
Zip the books, attach the zip to the agent chat, and ask it to unpack there.
See [`docs/CLOUD_RUN.md`](../docs/CLOUD_RUN.md) section **Bring your PDFs**.
Do not commit PDFs; `data/raw_pdfs/*` is gitignored.

`run_phase1.py` is an equivalent convenience runner with pre-flight checks
(verifies the input folder exists and lists the PDFs found) and a run summary;
it accepts the same flags:
```bash
python run_phase1.py --backend docling --strategy parent_child --no-figures
```

### 4. Configure (`config/config.yaml`)
```yaml
phase1:
  input_dir: "data/raw_pdfs"
  out_dir:   "data/processed"   # per-PDF folder: {stem}/chunks.jsonl (+ parents/tables/figures)
  out_format: "jsonl"        # jsonl | parquet
  backend:   "pymupdf4llm"   # pymupdf4llm | docling | pymupdf
  strategy:  "structure"     # structure | parent_child | semantic
  target_tokens: 400         # aim inside 300-500 for retrieval precision
  overlap_pct: 0.15          # 12-15%
  min_tokens: 80             # merge trailing slivers smaller than this
  parent_max_tokens: 2000
  ocr_enabled: true          # OCRs scanned pages; safe on mixed scanned/native PDFs
  extract_tables: true       # tables -> standalone markdown chunks
  extract_figures: true      # OCR diagrams/charts/graphs (needs pytesseract)
  dedup: true
  # --- scaling ---
  workers: 0                 # 0 = auto (cpu_count-1); 1 = single process
  # --- datastore sink (optional; default writes files to out_dir) ---
  sink: "file"               # file | postgres
  datastore_config: "config/datastore.yaml"
```
Run with a config file via `python -m phase1_pdf.pipeline --config config/config.yaml`.
CLI flags override config values.

### 5. Hand off to Phase 3
Two options:

- **Files (default).** Copy the per-PDF **folders** from `data/processed/` into
  the `arkguru-rag-slm` repo's `data/processed/`. They merge with Phase 2
  `web_chunks.jsonl` if you have it.
- **Shared datastore (no copying).** Run with `--sink postgres` so chunks land
  directly in the Postgres + pgvector **`chunks`** table. Phase 1 does **not**
  write `chunk_embeddings`. Phase 3 fills vectors in place:

  `python -m phase3_rag.embed_datastore --embedder hashing --dim 1024`

  Look at `chunks.chunk_index` (0-based), not the empty embeddings table. See
  *Scaling & datastore* below.

---

## Output schema (`common.schema.Chunk`)
`text`, `source_type` (`"pdf"`), `source_id`, `chunk_index`, `title`, `section`,
`page`, `lang`, `parent_id`, `is_parent`, `token_count`, `overlap_tokens`,
`embedding` (optional), `extra` (e.g. `{"block_type": "table"|"figure"}`),
`chunk_id`. Primitives only — fully portable between a cloud box and your
local machine.

## Layout
```
common/        shared Chunk schema, tokenizer, chunking helpers (vendored)
phase1_pdf/    extract.py · chunk.py · pipeline.py
config/        config.yaml
scripts/       make_sample_pdf.py (offline test fixture)
data/          raw_pdfs/ (input) · interim/ (OCR) · processed/{stem}/chunks.jsonl
```

## Scaling & datastore

> **Setting up the database?** See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md) for step-by-step Postgres + pgvector setup on a local machine (Docker/Homebrew/apt) or a VPS (remote access, firewall, SSL/SSH tunnel).

**Parallel + streaming.** Phase 1 processes PDFs across worker processes and
writes each PDF's output as soon as it's ready (folder `processed/{stem}/`).
That's ~10-16x faster on a multi-core CPU, memory stays flat, and a crash
mid-run keeps the files already written. No hard cap on PDFs per run; throughput
(OCR especially) is the ceiling, not memory.

```bash
python -m phase1_pdf.pipeline --input data/raw_pdfs --workers 0   # 0 = auto (cpu_count-1)
```

**Write straight to a Postgres + pgvector datastore** instead of files. The
datastore uses **two separate tables**: `chunks` (text + metadata — the
source-of-truth) and `chunk_embeddings` (vectors only, keyed by `chunk_id`).
Phase 1 writes **only the `chunks` table**; Phase 3 fills `chunk_embeddings`
later. Upserts are idempotent by `chunk_id` and **do not reset `created_at`**
(so a no-op re-ingest does not trip the daily backup watermark). Re-running a
growing corpus only adds new ids.

```bash
export PG_DSN=postgresql://user:pass@localhost:5432/rag
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres --workers 0
```

**Tables are created automatically.** The first `--sink postgres` run creates the
`chunks` and `chunk_embeddings` tables (and indexes) if they don't exist — the
run logs whether each was *created* or *already present*. To create them
explicitly without processing any PDFs (e.g. right after setting up the DB):

```bash
python -m phase1_pdf.pipeline --init-db      # or: make init-db
python run_phase1.py --init-db               # equivalent via the runner
```
You never write any SQL DDL yourself. If the database is unreachable, the run
fails with a clear message pointing to docs/DATABASE_SETUP.md.
**All datastore settings live in one place — `config/datastore.yaml`** (DSN via
the `PG_DSN` env var, table names, and embedding dim). No connection details are
hardcoded in code. To switch database or tables, edit that one file; to switch
the connection, set `PG_DSN` (or copy `.env.example` → `.env`). Env vars
(`PG_DSN`, `DATASTORE_BACKEND`, `DATASTORE_CONFIG`) override the file.

Requires `pip install "psycopg[binary]" pgvector` (commented in requirements.txt)
and a running Postgres with the `vector` extension. Files remain the default and
the portable interchange format.

## Dependencies explained

### Core dependencies (always required)
- **pymupdf4llm** — fast, native-text PDF extraction with built-in Markdown formatting
- **tiktoken** — lightweight tokenizer for chunk sizing (cl100k_base for GPT models)
- **pyyaml** — configuration file parsing

### OCR and figure extraction
- **ocrmypdf** — scans image-only pages and preserves native text layers; requires system `tesseract` + `ghostscript` (`pip` dependency; `ocr_enabled: true` by default)
- **pytesseract** + **Pillow** — extracts text from embedded diagrams/charts; requires system `tesseract`

### Optional extraction backends
- **docling** — IBM's table-transformer for complex layouts; significantly slower but superior table fidelity; requires `transformer` and `pdf2image` deps (still commented in `requirements.txt`)
- **pymupdf** — fallback plain text extraction; zero structure but always works

### Optional datastore
- **psycopg[binary]** + **pgvector** — PostgreSQL upsert with vector extension for streaming large corpora to a shared database

## Troubleshooting

### Empty / tiny output
→ PDFs may be scanned. `ocr_enabled` defaults to true; install system `tesseract` + `ghostscript` (the converter `setup_dev_env.sh` does this on Debian/Ubuntu). OCR only touches pages that actually lack text, so it is safe to leave on for native PDFs.

**System dependencies on macOS:**
```bash
brew install tesseract ghostscript
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr ghostscript
```

**On Windows:** Download from [Tesseract GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki) and [Ghostscript website](https://www.ghostscript.com/download/gsdnld.html), then add to PATH.

### Tables come out garbled
→ they're extracted natively via `page.find_tables()` (`extract_tables: true`); for very complex layouts, switch `backend: docling`. Docling's table-transformer is slower but handles merged cells, rotated text, and multi-column headers far better than pymupdf's basic grid detection.

Example switch:
```bash
python -m phase1_pdf.pipeline --backend docling --input data/raw_pdfs --out-dir data/processed
```

### Diagrams/charts missing text
→ set `extract_figures: true` and install `pytesseract` + `pillow` + system `tesseract`. This extracts axis labels, legends, and callout text baked into images via OCR.

### Chunks too big/small
→ tune `target_tokens` / `overlap_pct` / `min_tokens`.
- **Retrieval precision** prefers smaller chunks (300–400 tokens); increase `overlap_pct` to 20–25% for better context preservation across chunk boundaries.
- **Fine-tuning** prefers larger chunks (500–700 tokens) for richer training signal; reduce `overlap_pct` to 10–15% to avoid duplicate training pairs.

### Out of memory on large corpora
→ reduce `workers` or set to 1, or split input PDFs across multiple runs. Phase 1 processes PDFs in separate processes; each inherits a fresh memory context, so parallelism doesn't accumulate heap. If a single PDF is huge, the backend itself may spike memory (docling especially); no workaround except reducing `max_pages` per input file.

### Embeddings are NULL in the chunks table
→ embeddings are optional; Phase 1 writes `NULL` and Phase 3 fills them. This is intentional: the chunks are the durable source, and embeddings are model-specific and can be regenerated by truncating the `chunk_embeddings` table.

## Performance benchmarks

On a 2023 Intel Core i7 laptop (16GB RAM, 8 cores), with default config:

| Task | File size | Time | Throughput |
|---|---|---|---|
| Native PDF (no OCR) | 10 MB | 2–3 s | ~3–5 MB/s |
| Native PDF + figure extraction | 10 MB | 5–8 s | ~1–2 MB/s |
| Scanned PDF (OCR'd) | 5 MB | 30–60 s | ~0.1 MB/s |
| Complex tables (docling) | 2 MB | 10–15 s | ~0.2 MB/s |

OCR is the slowest step. For production corpora with many scanned PDFs, consider:
- Parallel workers (default: `workers=0` uses all cores)
- A GPU-capable server for OCR batching
- Pre-processing PDFs to remove unnecessary pages

## Advanced: Custom extraction backend

Extend `phase1_pdf/extract.py` with a new backend by:
1. Implementing `def extract_blocks(pdf_path: str) -> list[Block]`
2. Registering it in the `BACKENDS` dict
3. Passing `--backend your_name` at runtime

The `Block` schema (text, page, block_type, heading) is the only contract; the extraction engine is pluggable.
