# arkguru-common

Shared, framework-free primitives used by all three phases of the **arkguru**
local RAG system:

- [`arkguru-pdf-extraction`](../arkguru-pdf-extraction) (Phase 1)
- [`arkguru-web-scraping`](../arkguru-web-scraping) (Phase 2)
- [`arkguru-rag-slm`](../arkguru-rag-slm) (Phase 3)

Each phase folder installs this package as an editable dependency
(`pip install -e ../arkguru-common`), so the `common` package is importable as a
top-level module.

## Modules

| Module | Purpose |
|---|---|
| `common.schema` | The shared `Chunk` dataclass + JSONL/Parquet I/O (`read_jsonl`, `write_jsonl`, `read_parquet`, `write_parquet`). |
| `common.tokenizer` | `count_tokens` / `truncate_to_tokens` / `split_to_max_tokens` / `resolve_tokenizer`. Default: `BAAI/bge-m3` tokenizer files via `transformers` (`local_files_only`, never the model). Falls back to tiktoken `cl100k_base` with a one-time warning. `ARKGURU_TOKENIZER` overrides (`tiktoken` for tests/CI). Body cap default is **510** (512 minus XLM-R specials). |
| `common.chunking` | `split_paragraphs` / `split_sentences` / `split_for_packing` / `pack_windows`. Sentences split on `.!?`, danda `।`/`॥`, and newlines. `pack_windows` never exceeds `max_tokens`. |
| `common.text` | `clean_text` — strip inline HTML (`<mark>`, `<u>`, `<sup>`, `<br>`, …), unwrap `**`/`_` emphasis, NFKC, collapse whitespace. Applied to PDF and web text. |
| `common.quality` | Unicode quality scorer (`score_quality`) + `annotate_chunks` (`meta.quality`, `content_hash`, `lang`). Failed chunks are flagged and never embedded. |
| `common.lang` | Per-chunk seeded `detect_lang` with a Devanagari script check. Never empty (`und`). |
| `common.tables` | Linearise tables as `header: value`, drop `ColN` placeholders, strip markdown pipe tables from prose, pack long tables by row. |
| `common.datastore` | `ChunkStore` — two-table Postgres + pgvector (`chunks` text + `chunk_embeddings` vectors). Host-agnostic `PG_DSN` (Supabase, Neon, RDS, native local, Docker). `search_chunks()` v2 is the hybrid retrieve (SQL RRF of dense + websearch + phrase). `ensure_schema()` applies `migrations/`. Upsert does not reset `created_at` on conflict. `iter_missing_embeddings` uses a named server-side cursor. |
| `common.migrate` | Numbered SQL migrations + CLI (`python -m common.migrate --schema v2 --dry-run` / `--apply`). Refuses `public` on `--apply` unless `--allow-public`. |
| `common.validate_schema` | Read-only Phase 4 validation SELECTs (`python -m common.validate_schema --schema v2`). |
| `common.phrase` | Multi-word domain-term detection for the `phraseto_tsquery` leg. |
| `common.local_pg` | Detect native (5432) vs Docker (5433) local Postgres and keep an injected remote `PG_DSN`. |
| `common.datastore_config` | `open_chunk_store`, `load_datastore_config`, `resolve_dsn` — config-driven datastore factory (no hardcoded DSNs). |
| `common.worker` | `Worker` (resilient run-loop with signal handling) + `FolderState` (new/changed file tracking). |
| `common.rrf` | `reciprocal_rank_fusion` — Python fusion for file/npz retrieve. Postgres hybrid is SQL `search_chunks()`. |

## Datastore contracts

- Two tables: `chunks` (source of truth) and `chunk_embeddings` (model-specific, ON DELETE CASCADE).
- Optional **`schema`** (default `public`): Phase 2 uses `schema: web` so harvest is `web.chunks` / `web.search_chunks()` and cannot replace `public.search_chunks()`. Identifiers are validated before interpolation.
- Connection is DSN-only: remote hosts get `sslmode=require` when omitted; loopback/Docker is left alone. Override with `PG_SSLMODE` or an explicit URI `sslmode`.
- `upsert()` updates payload columns on `chunk_id` conflict and **leaves `created_at` unchanged** so backup watermarks ignore no-op re-ingests. Duplicate child `content_hash` values from another source update `meta.also_in` instead of inserting. `replace_source()` deletes-by-source and upserts in **one transaction**. `delete_by_source_id()` remains for callers that only need the delete. `ensure_schema()` applies the numbered files in repo-root `migrations/` (baseline + `search_chunks` v2) and **refuses `public`** unless `allow_public_schema` / `--allow-public` is set.
- Retrieval hits include `chunk_id`, `text`, `section`, `source_id`, `page`, `url`, `parent_id`, `chunk_index`, `title`, `lang`, plus `rrf_score`, `dense_rank`, `lexical_rank`, `phrase_rank` from `search_chunks()` v2.
- `search_chunks(query_text, query_embedding, …)` is the Phase 3 retrieve API when `PG_DSN` is set. Caller embeds; SQL fuses HNSW cosine, `websearch_to_tsquery` (OR fallback), and `phraseto_tsquery` on multi-word lexicon terms (`is_parent = false`, quality-gate failures dropped). If the function is missing, `ensure_schema()` runs once and the SELECT is retried. `search_dense` / `search_lexical` remain for single-leg debugging. File/npz retrieve still uses `reciprocal_rank_fusion`.
- `iter_missing_embeddings(batch=…)` uses a named (server-side) cursor and `itersize` so the client does not buffer the full result set. `iter_chunks_for_embedding(..., reembed=True, model=…)` also selects rows whose stored model is NULL, blank, or different from the target (resumable re-embed). `update_embeddings` requires a non-blank `model` label. `distinct_embedding_models()` is the retrieve startup check.

## Install

```bash
pip install -e .            # core (pyyaml, tiktoken)
pip install -e ".[xlm]"     # transformers; uses cached BAAI/bge-m3 tokenizer files
pip install -e ".[parquet]" # + pyarrow for Parquet interchange
pip install -e ".[postgres]"# + psycopg/pgvector for the DB datastore
pip install -e ".[test]"    # + pytest
```

## Test

```bash
pip install -e ".[test]"
pytest tests -q
```

MIT licensed.
