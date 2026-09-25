# arkguru-ui — local 3-phase RAG wizard

A **local-first** web UI for the existing arkguru pipeline. It does not
reimplement PDF extraction, crawling, or retrieval. FastAPI starts the same
CLIs the Makefiles already call, streams their logs, and React walks you
through:

1. Phase 1 — PDF extraction (`python -m phase1_pdf.pipeline`)
2. Phase 2 — web scraping (`python -m phase2_web.pipeline`)
3. Phase 3 — combine + index (`phase3_rag.prepare_dataset` + `quickstart` / `embed_datastore`)
4. Chat — `phase3_rag.quickstart.retrieve` + `answer`, with citations

Fine-tune is **CLI-only** in this package (`cd arkguru-rag-slm && make finetune`).
The wizard always takes the skip path: combine → index → chat.

## Layout this UI understands

Common, Phase 1, Phase 2, and Phase 3 are regular folders in this repo
(`docs/LOCAL_RUN.md`). `backend/paths.py` still accepts a sibling checkout
if you point `PHASE1_REPO` / `PHASE2_REPO` / `PHASE3_REPO` / `COMMON_REPO`
or `ARKGURU_ROOT` at one (see `.env.example`).

```
arkguru-pdf-converter/       # the only git clone
  arkguru-common/
  arkguru-pdf-extraction/    # Phase 1 (in-tree)
  arkguru-web-scraping/      # Phase 2 (in-tree)
  arkguru-rag-slm/           # Phase 3 (in-tree)
  arkguru-ui/                # this package
```

## Install

From the umbrella repo, with the existing venv (Python 3.9+):

```bash
cd arkguru-pdf-converter
bash scripts/setup_dev_env.sh
source .venv/bin/activate

pip install -r arkguru-ui/requirements.txt
cd arkguru-ui/frontend && npm install
```

Copy env files; never commit secrets:

```bash
cp .env.example .env                 # umbrella PG_DSN, if you use Postgres
cp arkguru-ui/.env.example arkguru-ui/.env
```

`PG_DSN` is read from the process environment / `.env`. The API only reports
whether it is **set**, never the value.

## Run

From the umbrella repo (no npm):

```bash
source .venv/bin/activate
make ui
# http://127.0.0.1:8080
```

The page posts `POST /api/jobs` (`phase1` | `phase2` | `phase3_embed` |
`phase3_ask`) and streams `GET /api/jobs/{id}/log`. One job at a time.

Optional React wizard (needs Node):

```bash
cd arkguru-ui
make dev          # API :8765 + Vite :5173
# or separately:
make api          # http://127.0.0.1:8765/api/health
make frontend     # http://127.0.0.1:5173  (proxies /api)
```

## Smoke path (offline, no Postgres, no Ollama)

1. Phase 1 → **Generate sample PDF** (runs `scripts/make_sample_pdf.py`)
2. **Run extraction** with backend `pymupdf4llm`, strategy `structure`, sink `file`
3. Skip Phase 2 (or crawl a tiny allowlist with `max_pages` ≤ 6)
4. Phase 3 → embedder **hashing**, sink **file**, fine-tune skipped
5. Chat: *What PPE is required before servicing a unit?*

Expected: extractive answer mentioning insulated gloves / lockout-tagout, plus
a citation for `sample_handbook.pdf`.

Equivalent CLIs (what the wizard shells out to):

```bash
cd ../arkguru-pdf-extraction
python scripts/make_sample_pdf.py
python -m phase1_pdf.pipeline --input data/raw_pdfs --out-dir data/processed --workers 1

cd ../arkguru-rag-slm
python -m phase3_rag.prepare_dataset \
  --inputs ../arkguru-pdf-extraction/data/processed/sample_handbook/chunks.jsonl \
  --out-corpus data/processed/corpus.jsonl
python -m phase3_rag.quickstart --add data/processed/corpus.jsonl \
  --embedder hashing --store data/store/index \
  --ask "What PPE is required before servicing a unit?"
```

## How each phase is invoked

| Step | Existing entrypoint | Wizard extras |
|---|---|---|
| Phase 1 | `python -m phase1_pdf.pipeline` from the Phase 1 cwd | CLI flags for backend / strategy / OCR / figures / sink |
| Sample PDF | `scripts/make_sample_pdf.py` | Sets the Phase 1 input folder |
| Phase 2 | `python -m phase2_web.pipeline --config <temp.yaml>` | Temp YAML carries `same_domain_only` and `backend` (not all of these are CLI flags) |
| Phase 3 file | `phase3_rag.prepare_dataset` then `phase3_rag.quickstart --add` | Deduped `corpus.jsonl` + local `.npz` store |
| Phase 3 postgres | `python -m phase3_rag.embed_datastore` | Requires `PG_DSN`; does not invent a new schema |
| Chat file | imports `quickstart.retrieve` / `answer` | Hashing embedder + extractive fallback if Ollama is down |
| Chat postgres | `ChunkStore.search_dense` + `quickstart.answer` | Same `PG_DSN` the CLIs already use |
| Fine-tune | **not started** | Banner + `make finetune` |

`PYTHONPATH` is the phase checkout plus `arkguru-common` (sibling preferred,
vendored fallback).

## Known limitations

- **Fine-tune UI is out of scope.** Hours-long LoRA stays on the Phase 3 CLI.
- **Phase 2 `backend: firecrawl` is not executed.** `Phase2Config` has the field
  and the wizard writes it, but `phase2_web.fetch.crawl` is local httpx +
  trafilatura. Keep `FIRECRAWL_API_KEY` in `.env` if you later wire it; do not
  put keys in the repo.
- **`semantic` chunking** is a Phase 1 CLI value; without an embedder it
  behaves like `structure` (existing Phase 1 behavior).
- **Postgres chat with hashing** uses dense search only (same idea as
  `phase3_rag.run_pdfs --sink postgres --ask`). Full hybrid SQL retrieve +
  rerank still wants `sentence_transformer` / `make serve`.
- **Ollama is optional.** Without it, answers are `[extractive]` top-hit text.
- **Single-user, in-memory jobs.** Restarting the API drops the job list;
  logs under `arkguru-ui/data/jobs/` remain.
- Phase trees are in this repo. If a checkout is missing, `GET /api/health`
  reports `found: false` until the folder is present.

## Tests

```bash
cd arkguru-ui
source ../.venv/bin/activate
pip install -r requirements.txt
make test
```
