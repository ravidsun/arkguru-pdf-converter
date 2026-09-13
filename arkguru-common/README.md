# arkguru-common

Shared, framework-free primitives used by all three phases of the **arkguru**
local RAG system:

- [`arkguru-pdf-extraction`](https://github.com/ravidsun/arkguru-pdf-extraction) (Phase 1)
- [`arkguru-web-scraping`](https://github.com/ravidsun/arkguru-web-scraping) (Phase 2)
- [`arkguru-rag-slm`](https://github.com/ravidsun/arkguru-rag-slm) (Phase 3)

Each phase repo installs this package as an editable sibling dependency
(`pip install -e ../arkguru-common`), so the `common` package is importable as a
top-level module.

## Modules

| Module | Purpose |
|---|---|
| `common.schema` | The shared `Chunk` dataclass + JSONL/Parquet I/O (`read_jsonl`, `write_jsonl`, `read_parquet`, `write_parquet`). |
| `common.tokenizer` | `count_tokens` / `truncate_to_tokens` (tiktoken `cl100k_base` proxy, with a char-based fallback). |
| `common.chunking` | `split_sentences` + `pack_windows` — the structure-aware windowing shared by Phase 1 and Phase 2. |
| `common.datastore` | `ChunkStore` — two-table Postgres + pgvector (`chunks` text + `chunk_embeddings` vectors). Host-agnostic `PG_DSN` (Supabase, Neon, RDS, local Docker). Upsert does not reset `created_at` on conflict. `iter_missing_embeddings` uses a named server-side cursor. |
| `common.datastore_config` | `open_chunk_store`, `load_datastore_config`, `resolve_dsn` — config-driven datastore factory (no hardcoded DSNs). |
| `common.worker` | `Worker` (resilient run-loop with signal handling) + `FolderState` (new/changed file tracking). |
| `common.rrf` | `reciprocal_rank_fusion` — fuse dense + lexical ranked runs (Phase 3 retrieval). |

## Datastore contracts

- Two tables: `chunks` (source of truth) and `chunk_embeddings` (model-specific, ON DELETE CASCADE).
- Connection is DSN-only: remote hosts get `sslmode=require` when omitted; loopback/Docker is left alone. Override with `PG_SSLMODE` or an explicit URI `sslmode`.
- `upsert()` updates payload columns on `chunk_id` conflict and **leaves `created_at` unchanged** so backup watermarks ignore no-op re-ingests.
- `iter_missing_embeddings(batch=…)` uses a named (server-side) cursor and `itersize` so the client does not buffer the full result set.

## Install

```bash
pip install -e .            # core (pyyaml, tiktoken)
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
