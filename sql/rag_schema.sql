-- Schema only (no row data). Extracted from the 2026-09-15 rag dump
-- (Postgres 16.15 + pgvector, dim 1024). Matches ensure_schema() in
-- common/datastore.py.
--
--   psql "$PG_DSN" -v ON_ERROR_STOP=1 -f sql/rag_schema.sql
--
-- From a custom dump:
--   pg_restore --schema-only --no-owner --no-acl -f rag_schema.sql arkguru-rag.dump

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE public.chunks (
    chunk_id text NOT NULL,
    text text NOT NULL,
    source_type text,
    source_id text,
    chunk_index integer DEFAULT 0 NOT NULL,
    title text,
    section text,
    page integer,
    url text,
    domain text,
    lang text,
    parent_id text,
    is_parent boolean DEFAULT false,
    token_count integer,
    overlap_tokens integer DEFAULT 0,
    meta jsonb DEFAULT '{}'::jsonb,
    ts tsvector GENERATED ALWAYS AS (
        to_tsvector('english'::regconfig, COALESCE(text, ''::text))
    ) STORED,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chunks_pkey PRIMARY KEY (chunk_id)
);

CREATE TABLE public.chunk_embeddings (
    chunk_id text NOT NULL,
    embedding public.vector(1024),
    model text,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chunk_embeddings_pkey PRIMARY KEY (chunk_id),
    CONSTRAINT chunk_embeddings_chunk_id_fkey
        FOREIGN KEY (chunk_id) REFERENCES public.chunks(chunk_id) ON DELETE CASCADE
);

CREATE INDEX chunks_source_idx ON public.chunks USING btree (source_type, source_id);
CREATE INDEX chunks_ts_idx ON public.chunks USING gin (ts);
CREATE INDEX chunk_embeddings_hnsw_idx
    ON public.chunk_embeddings USING hnsw (embedding public.vector_cosine_ops);
