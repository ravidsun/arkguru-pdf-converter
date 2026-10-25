# arkguru-web-scraping — Phase 2

Turn an allowlist of URLs into portable **`Chunk`** records (JSONL or Postgres
schema `web`). This folder is Phase 2 of the **arkguru-pdf-converter** monorepo.
Output concatenates with Phase 1 via the shared schema in
[`arkguru-common`](../arkguru-common).

Web harvest is **isolated** from the book corpus:

| | Books (Phase 1 dump) | This folder |
|---|---|---|
| Schema | `public` | `web` |
| Tables | `public.chunks` | `web.chunks` |
| SQL | `public.search_chunks()` | `web.search_chunks()` |

`--sink postgres` **exits** if `postgres.schema` is `public` or unset.

YouTube / playlist ingest is **not implemented** (epic
[#39](https://github.com/ravidsun/arkguru-pdf-converter/issues/39)).

---

## How it works

`phase2_web/pipeline.py` runs **crawl → extract → chunk → write** (+ optional PDF harvest).

1. **Fetch (`phase2_web/fetch.py`).** BFS per seed with `httpx`. Honors robots.txt,
   `same_domain_only`, a path denylist (`/login`, `/cart`, `/chat`, …), Kundli-generator
   query strings, and binary/asset suffixes. Jagannath Hora installer URLs are skipped.
   HTML pages go to `pages`; `.pdf` URLs are listed, not fetched as HTML.

   `Phase2Config.backend` accepts `local` | `firecrawl`, but **`crawl()` is local only**.
   There is no Firecrawl client call. `firecrawl-py` is commented in `requirements.txt`.

2. **Extract (`phase2_web/extract.py`).** `trafilatura.extract` when installed;
   regex tag-strip fallback otherwise. Pages under `min_content_chars` (default 200)
   produce no chunks.

3. **Chunk (`phase2_web/chunk.py`).** Same `pack_windows` helper as Phase 1
   (`target_tokens: 550`, `overlap_pct: 0.15`, `min_tokens: 80`).
   `source_type="web"`, `extra.block_type="web"`.

4. **Dedup (`phase2_web/dedup.py`).** MinHash LSH (Jaccard ≥ 0.9) when `datasketch`
   is installed; exact-text MD5 otherwise.

5. **Write.** Always writes `data/processed/web_chunks.jsonl` (or `.parquet`).
   `--sink postgres` also delete-then-upserts per URL into `web.chunks`.

6. **PDF harvest (`phase2_web/pdfs.py`).** Public `.pdf` links are saved under
   `data/raw_pdfs/{domain}/`. Phase-1 extraction into schema `web` runs only when
   `sink=postgres` and `ingest_pdfs` is true (`--datastore-config` points Phase 1
   at this folder’s `config/datastore.yaml`).

---

## Install

From the umbrella repo:

```bash
cd arkguru-pdf-converter
bash scripts/setup_dev_env.sh
source .venv/bin/activate
cd arkguru-web-scraping
```

Or this folder only:

```bash
cd arkguru-web-scraping
pip install -r requirements.txt   # pins `-e ../arkguru-common`
```

`--sink postgres` also needs `psycopg` / `pgvector` (umbrella setup installs
`arkguru-common[postgres]`; they are commented in this folder’s requirements).

---

## Run

```bash
# JSONL (default). make phase2 == this command.
python -m phase2_web.pipeline --config config/config.yaml
# → data/processed/web_chunks.jsonl

# Postgres schema web (same PG_DSN as the books). JSONL is still written.
python -m phase2_web.pipeline --config config/config.yaml --sink postgres
# → rag.web.chunks (DBeaver: schema web)

python -m phase2_web.pipeline --config config/config.yaml --init-db
```

Default seeds in `config/config.yaml` are Vedic learning sites plus traditional
Western article hubs. Commercial Kundli portals stay out. `max_pages` is 2000
and `max_pages_per_seed` is 80 — that is a real crawl. For a first try:

```bash
python -m phase2_web.pipeline --config config/config.yaml \
  --seeds https://docs.python.org/3/tutorial/ \
  --max-pages 6 --max-pages-per-seed 6 --delay 0.75 --no-pdfs
```

Worker:

```bash
python -m phase2_web.worker --once              # make worker
python -m phase2_web.worker --interval 3600 --sink postgres
```

Orchestrator step `crawl_web` in `arkguru-rag-slm/orchestrator.yaml` stays
**disabled** until you turn it on.

### CLI flags (`python -m phase2_web.pipeline`)

| Flag | Notes |
|---|---|
| `--config` | YAML `phase2:` block |
| `--seeds` | Override seed URLs |
| `--max-pages` / `--max-pages-per-seed` | Hard caps |
| `--delay` | Seconds between fetches (`delay_seconds` in YAML, default 0.75) |
| `--sink` | `file` \| `postgres` |
| `--datastore-config` | Default `config/datastore.yaml` (`schema: web`) |
| `--out-dir` | Default `data/processed` |
| `--no-pdfs` | Do not download linked PDFs |
| `--no-ingest-pdfs` | Download but skip Phase 1 |
| `--init-db` | Create `web` schema/tables, then exit |

There is no `--backend` CLI flag. `same_domain_only` is YAML-only.

---

## Config

`config/config.yaml` (`phase2:`):

```yaml
seeds: [...]
out_dir: "data/processed"
out_format: "jsonl"          # jsonl | parquet
max_pages: 2000
max_pages_per_seed: 80
delay_seconds: 0.75
same_domain_only: true
backend: "local"             # accepted; crawl() does not branch on this
target_tokens: 550
overlap_pct: 0.15
min_tokens: 80
min_content_chars: 200
dedup: true
sink: "file"
datastore_config: "config/datastore.yaml"
download_pdfs: true
ingest_pdfs: true
pdf_dir: "data/raw_pdfs"
```

`config/datastore.yaml` must keep `postgres.schema: web`.

---

## Inputs and outputs

| | Path |
|---|---|
| Seeds | `config/config.yaml` → `phase2.seeds` |
| HTML chunks | `data/processed/web_chunks.jsonl` |
| Harvested PDFs | `data/raw_pdfs/{domain}/` |
| Postgres | `web.chunks` on `PG_DSN` (vectors filled later by Phase 3) |

---

## Tests

```bash
source ../.venv/bin/activate
make test          # pip install -e ../arkguru-common[test] && pytest tests -q
# or: python -m pytest tests -q
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `--sink postgres` exits | Keep `schema: web` in `config/datastore.yaml` |
| Blank pages | Local fetcher cannot execute JS. Do not expect `backend: firecrawl` to run. |
| Crawl too large | Lower `--max-pages` / `--max-pages-per-seed` |
| `ModuleNotFoundError: common` | Activate the umbrella `.venv` or `pip install -e ../arkguru-common` |
