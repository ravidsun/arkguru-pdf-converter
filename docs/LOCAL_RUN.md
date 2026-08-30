# Local run

Run the 3-phase arkguru pipeline on a machine you control (laptop, NUC, or workstation). This path uses a Python virtualenv and file-based outputs. Postgres, Ollama, OCR, and the heavy Phase 3 ML stack are optional.

The companion document is [CLOUD_RUN.md](CLOUD_RUN.md).

## Layout

Clone the umbrella repo and the three phase repos as **siblings**. `arkguru-common` is vendored inside this repo (not a separate GitHub project).

```
some-dir/
  arkguru-pdf-converter/     # this repo (common + env setup)
    arkguru-common/
  arkguru-pdf-extraction/    # Phase 1
  arkguru-web-scraping/      # Phase 2
  arkguru-rag-slm/           # Phase 3
```

```bash
mkdir -p ~/arkguru && cd ~/arkguru
git clone https://github.com/ravidsun/arkguru-pdf-converter.git
git clone https://github.com/ravidsun/arkguru-pdf-extraction.git
git clone https://github.com/ravidsun/arkguru-web-scraping.git
git clone https://github.com/ravidsun/arkguru-rag-slm.git
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

1. Creates `.venv` in this repo (override with `ARKGURU_VENV`).
2. Installs the vendored `arkguru-common` package (`import common` works in every phase).
3. Installs Phase 1 and Phase 2 requirements, plus `reportlab` (sample PDF fixture).
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
# Shared package
pytest arkguru-common/tests -q
```

### Phase 1 — PDF → chunks

```bash
cd ../arkguru-pdf-extraction
python scripts/make_sample_pdf.py     # data/raw_pdfs/sample_handbook.pdf
make phase1                           # data/processed/sample_handbook.jsonl
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

## Production-style local run

Postgres + pgvector, sentence-transformer embeddings, and a local LLM via Ollama.

1. Install the full Phase 3 requirements (see above).
2. Start Postgres with the `vector` extension. See [DATABASE_SETUP.md](https://github.com/ravidsun/arkguru-pdf-extraction/blob/main/docs/DATABASE_SETUP.md).

```bash
export PG_DSN=postgresql://user:pass@localhost:5432/rag
psql "$PG_DSN" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

3. Ingest to files or `--sink postgres`:

```bash
cd ../arkguru-pdf-extraction
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres

cd ../arkguru-web-scraping
python -m phase2_web.pipeline --config config/config.yaml --sink postgres
```

4. Fine-tune (optional, CPU hours), merge/quantize, register with Ollama as `domain-slm` (see the header of `phase3_rag/serve.py`).

5. Index and chat:

```bash
cd ../arkguru-rag-slm
make index
ollama serve &
python -m phase3_rag.run_pdfs --pdfs ../arkguru-pdf-extraction/data/raw_pdfs \
  --embedder sentence_transformer --model domain-slm --chat
# or: make serve
```

## Useful make targets

| Repo | Target | What it does |
|---|---|---|
| `arkguru-pdf-extraction` | `make phase1` | Extract + chunk PDFs |
| `arkguru-pdf-extraction` | `make worker` | Watch `data/raw_pdfs` |
| `arkguru-web-scraping` | `make phase2` | Crawl seeds in config |
| `arkguru-rag-slm` | `make from-pdfs PDFS=... ASK="..."` | PDF → ingest → ask |
| `arkguru-rag-slm` | `make e2e` | Offline orchestrator pass |
| `arkguru-rag-slm` | `make serve` | Interactive RAG chat |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: common` | Activate `.venv` created by `scripts/setup_dev_env.sh`, or `pip install -e arkguru-common` |
| Phase repo not found during setup | Clone it as a **sibling** of `arkguru-pdf-converter` |
| Empty Phase 1 output | Scanned PDF — install tesseract/ghostscript and set `ocr_enabled: true` |
| Blank Phase 2 pages | JS site — switch to `backend: firecrawl` |
| Ollama errors | Start `ollama serve`; extractive answers still work without it |
| Slow / OOM on fine-tune | Stay on `--embedder hashing` for plumbing tests; reduce LoRA rank / batch size for real training |
