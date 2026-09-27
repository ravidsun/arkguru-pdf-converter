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

`--sink postgres` **exits** if `postgres.schema` is `public` or unset
(legacy isolation). Re-chunked web rows belong in the **same** `chunks`
table with `source_type='web'`:

```bash
python -m phase2_web.rechunk --input web_chunks.csv --out-dir data/processed
# optional: --sink postgres --datastore-config config/datastore.yaml
```

Site-map, `/tag/`, `/category/`, `/page/N`, and archive pages are fetched
so their links can be followed, but they are not ingested. A link-density
filter drops listing pages that are mostly anchors.

## Run

```bash
# JSONL (default)
python -m phase2_web.pipeline --config config/config.yaml
# → data/processed/web_chunks.jsonl

# Postgres schema web (same PG_DSN as the books)
python -m phase2_web.pipeline --config config/config.yaml --sink postgres
```

First-pass seeds (Vedic learning sites plus traditional Western article hubs)
live in `config/config.yaml`. Commercial Kundli portals stay out.
`max_pages_per_seed` defaults to 80. robots.txt is honored. Login/chat/cart
paths and Kundli-generator query strings are skipped.

Public `.pdf` links discovered during the crawl are saved under
`data/raw_pdfs/{domain}/` and Phase-1 extracted **into schema web** (not
`public.chunks`). Jagannath Hora installer binaries are skipped.

Orchestrator step `crawl_web` stays **disabled** until you turn it on.

## Config

See `config/config.yaml` and `config/datastore.yaml` (`schema: web`).

Chunking keys: `target_tokens` (550), `max_tokens` (510 body / 512 window),
`min_tokens` (80), `min_chunk_chars` (80), `min_content_chars` (200).
Counts use the cached `BAAI/bge-m3` tokenizer when available; otherwise
tiktoken (one-time warning). `ARKGURU_TOKENIZER=tiktoken` forces the proxy.
