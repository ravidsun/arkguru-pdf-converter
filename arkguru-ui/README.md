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

The umbrella repo (`arkguru-pdf-converter`) and the phase checkouts disagree
on disk layout. `backend/paths.py` uses the same sibling-or-nested search as
`scripts/setup_dev_env.sh`:

```
# Laptop / docs/LOCAL_RUN.md (siblings)
some-dir/
  arkguru-common/
  arkguru-pdf-converter/          # this umbrella; .venv lives here
    arkguru-common/               # vendored fallback
    arkguru-web-scraping/         # Phase 2 source of truth (in-tree)
    arkguru-ui/                   # this package
  arkguru-pdf-extraction/         # Phase 1
  arkguru-rag-slm/                # Phase 3

# Also accepted: nested under the umbrella
arkguru-pdf-converter/
  arkguru-pdf-extraction/         # submodule and/or gitignored clone
  arkguru-rag-slm/
```

**Submodule quirk:** `arkguru-pdf-extraction` is listed in `.gitmodules`
(pinned commit) **and** in the umbrella `.gitignore` (nested sibling clones).
A fresh clone of the umbrella often has an empty Phase 1 directory until you:

```bash
git submodule update --init arkguru-pdf-extraction
# or
git clone https://github.com/ravidsun/arkguru-pdf-extraction.git
```

Phase 3 may also appear as a gitlink in some umbrella checkouts even when it
is absent from `.gitmodules`. Clone `arkguru-rag-slm` as a sibling (Cloud Agent
`repositoryDependencies`) or nest it under the umbrella. Override any path
with `PHASE1_REPO` / `PHASE2_REPO` / `PHASE3_REPO` / `COMMON_REPO` / `ARKGURU_ROOT`
in `.env` (see `.env.example`). Do not commit a changed gitlink SHA from a
local clone of those dirs.

Phase 2 currently lives **in this umbrella** (`./arkguru-web-scraping`).

## Install

From the umbrella repo, with the existing venv (Python 3.9+):

```bash
cd arkguru-pdf-converter
bash scripts/setup_dev_env.sh
source .venv/bin/activate

# Phase 1 + Phase 3 checkouts (if the directories are empty)
# git submodule update --init arkguru-pdf-extraction
# git clone https://github.com/ravidsun/arkguru-rag-slm.git

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

```bash
cd arkguru-ui
make dev          # API :8765 + Vite :5173
# or separately:
make api          # http://127.0.0.1:8765/api/health
make ui           # http://127.0.0.1:5173  (proxies /api)
```

Open `http://127.0.0.1:5173`. Long jobs (OCR, crawl, index) run in a
background thread; the UI polls `/api/jobs/{id}` and can stream
`/api/jobs/{id}/stream`.

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
- Empty Phase 1 / Phase 3 directories in a fresh umbrella clone are expected
  until those repos are checked out.

## Tests

```bash
cd arkguru-ui
source ../.venv/bin/activate
pip install -r requirements.txt
make test
```
