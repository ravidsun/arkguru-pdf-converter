# arkguru-web-scraping — Phase 2

Turn an allowlist of URLs into portable **`Chunk`** records (JSONL or Postgres
schema ``web``). This is Phase 2 of the arkguru stack. Output concatenates with
Phase 1 via the shared schema in
[`arkguru-common`](https://github.com/ravidsun/arkguru-common).

Web harvest is **isolated** from the book corpus:

| | Books (Phase 1 dump) | This repo |
|---|---|---|
| Schema | `public` | `web` |
| Tables | `public.chunks` | `web.chunks` |
| SQL | `public.search_chunks()` | `web.search_chunks()` |

`--sink postgres` **exits** if `postgres.schema` is `public` or unset.

## Run

```bash
# JSONL (default)
python -m phase2_web.pipeline --config config/config.yaml
# → data/processed/web_chunks.jsonl

# Postgres schema web (same PG_DSN as the books)
python -m phase2_web.pipeline --config config/config.yaml --sink postgres
```

First-pass seeds (Vedic learning sites) live in `config/config.yaml`.
`max_pages_per_seed` defaults to 80. robots.txt is honored. Login/chat/cart
paths and Kundli-generator query strings are skipped.

Public `.pdf` links discovered during the crawl are saved under
`data/raw_pdfs/{domain}/` and Phase-1 extracted **into schema web** (not
`public.chunks`). Jagannath Hora installer binaries are skipped.

Orchestrator step `crawl_web` stays **disabled** until you turn it on.

## Config

See `config/config.yaml` and `config/datastore.yaml` (`schema: web`).
