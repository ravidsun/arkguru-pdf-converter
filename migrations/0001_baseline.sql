-- 0001_baseline: full current RAG schema (PR #37).
-- Placeholders: {{schema}} {{dim}} {{chunks}} {{vectors}} {{function}}
-- {{chunks_bare}} {{vectors_bare}}
-- Idempotent. Safe on an empty schema and on a schema that already has tables.

CREATE TABLE IF NOT EXISTS {{schema}}.schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {{chunks}} (
    chunk_id       text PRIMARY KEY,
    text           text NOT NULL,
    source_type    text,
    source_id      text,
    chunk_index    int NOT NULL DEFAULT 0,
    title          text,
    section        text,
    page           int,
    url            text,
    domain         text,
    lang           text NOT NULL DEFAULT 'und',
    parent_id      text,
    is_parent      boolean DEFAULT false,
    token_count    int,
    overlap_tokens int DEFAULT 0,
    content_hash   text,
    meta           jsonb DEFAULT '{}'::jsonb,
    ts             tsvector GENERATED ALWAYS AS
                   (to_tsvector('english', coalesce(text, ''))) STORED,
    created_at     timestamptz DEFAULT now()
);

ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS chunk_id text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS text text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS source_type text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS source_id text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS chunk_index int;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS title text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS section text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS page int;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS url text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS domain text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS lang text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS parent_id text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS is_parent boolean;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS token_count int;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS overlap_tokens int;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS content_hash text;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS meta jsonb;
ALTER TABLE {{chunks}} ADD COLUMN IF NOT EXISTS created_at timestamptz;

UPDATE {{chunks}} SET chunk_index = 0 WHERE chunk_index IS NULL;
ALTER TABLE {{chunks}} ALTER COLUMN chunk_index SET DEFAULT 0;
ALTER TABLE {{chunks}} ALTER COLUMN chunk_index SET NOT NULL;
UPDATE {{chunks}} SET lang = 'und' WHERE lang IS NULL OR btrim(lang) = '';
ALTER TABLE {{chunks}} ALTER COLUMN lang SET DEFAULT 'und';
ALTER TABLE {{chunks}} ALTER COLUMN lang SET NOT NULL;

DO $m$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = '{{schema}}'
          AND c.relname = '{{chunks_bare}}'
          AND i.indisprimary
    ) THEN
        ALTER TABLE {{chunks}} ADD PRIMARY KEY (chunk_id);
    END IF;
END
$m$;

CREATE INDEX IF NOT EXISTS {{chunks_bare}}_ts_idx
    ON {{chunks}} USING gin (ts);
CREATE INDEX IF NOT EXISTS {{chunks_bare}}_source_idx
    ON {{chunks}} (source_type, source_id);
CREATE UNIQUE INDEX IF NOT EXISTS {{chunks_bare}}_content_hash_child_uidx
    ON {{chunks}} (content_hash)
    WHERE is_parent IS NOT TRUE
      AND content_hash IS NOT NULL
      AND btrim(content_hash) <> '';

CREATE TABLE IF NOT EXISTS {{vectors}} (
    chunk_id   text PRIMARY KEY
               REFERENCES {{chunks}} (chunk_id) ON DELETE CASCADE,
    embedding  vector({{dim}}),
    model      text,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE {{vectors}} ADD COLUMN IF NOT EXISTS chunk_id text;
ALTER TABLE {{vectors}} ADD COLUMN IF NOT EXISTS embedding vector({{dim}});
ALTER TABLE {{vectors}} ADD COLUMN IF NOT EXISTS model text;
ALTER TABLE {{vectors}} ADD COLUMN IF NOT EXISTS created_at timestamptz;

DO $m$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = '{{schema}}'
          AND c.relname = '{{vectors_bare}}'
          AND i.indisprimary
    ) THEN
        ALTER TABLE {{vectors}} ADD PRIMARY KEY (chunk_id);
    END IF;
END
$m$;

DO $m$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint con
        JOIN pg_class c ON c.oid = con.conrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = '{{schema}}'
          AND c.relname = '{{vectors_bare}}'
          AND con.contype = 'f'
    ) THEN
        ALTER TABLE {{vectors}}
            ADD CONSTRAINT {{vectors_bare}}_chunk_id_fkey
            FOREIGN KEY (chunk_id) REFERENCES {{chunks}} (chunk_id)
            ON DELETE CASCADE;
    END IF;
END
$m$;

CREATE INDEX IF NOT EXISTS {{vectors_bare}}_hnsw_idx
    ON {{vectors}} USING hnsw (embedding vector_cosine_ops);

-- search_chunks v1 (dense HNSW + lexical plainto/OR fallback, RRF).
DROP FUNCTION IF EXISTS {{function}}(text, vector, integer, integer, integer, integer, text, text);
DROP FUNCTION IF EXISTS {{function}}(text, vector, integer, integer, integer, integer, text, text, integer, text, text[]);

CREATE OR REPLACE FUNCTION {{function}}(
  query_text      text,
  query_embedding vector,
  k_dense         int  DEFAULT 20,
  k_lexical       int  DEFAULT 20,
  k_final         int  DEFAULT 6,
  rrf_k           int  DEFAULT 60,
  filter_source_type text DEFAULT NULL,
  filter_source_id   text DEFAULT NULL
)
RETURNS TABLE (
  chunk_id text, text text, section text, source_id text,
  page int, url text, parent_id text, chunk_index int,
  title text, lang text,
  rrf_score float4, dense_rank int, lexical_rank int
)
LANGUAGE sql
STABLE
AS $search$
WITH dense AS (
  SELECT v.chunk_id,
         ROW_NUMBER() OVER (ORDER BY v.embedding <=> query_embedding)::int AS rnk
  FROM {{vectors}} v
  ORDER BY v.embedding <=> query_embedding
  LIMIT k_dense
),
dense_hits AS (
  SELECT d.chunk_id, d.rnk
  FROM dense d
  JOIN {{chunks}} c ON c.chunk_id = d.chunk_id
  WHERE c.is_parent = false
    AND (filter_source_type IS NULL OR c.source_type = filter_source_type)
    AND (filter_source_id IS NULL OR c.source_id = filter_source_id)
),
lex_and AS (
  SELECT plainto_tsquery('english', query_text) AS q
),
lex_or AS (
  SELECT to_tsquery('english',
           array_to_string(
             ARRAY(SELECT quote_literal(lx)
                   FROM unnest(tsvector_to_array(
                          to_tsvector('english', query_text))) AS lx),
             ' | ')) AS q
),
lex_and_n AS (
  SELECT count(*) AS n FROM (
    SELECT 1
    FROM {{chunks}} c, lex_and
    WHERE c.ts @@ lex_and.q
      AND c.is_parent = false
      AND (filter_source_type IS NULL OR c.source_type = filter_source_type)
      AND (filter_source_id IS NULL OR c.source_id = filter_source_id)
    LIMIT k_lexical
  ) probe
),
lex_q AS (
  SELECT CASE WHEN (SELECT n FROM lex_and_n) >= k_lexical
              THEN (SELECT q FROM lex_and)
              ELSE (SELECT q FROM lex_or)
         END AS q
),
lexical AS (
  SELECT c.chunk_id,
         ROW_NUMBER() OVER (
           ORDER BY ts_rank(c.ts, lex_q.q) DESC
         )::int AS rnk
  FROM {{chunks}} c, lex_q
  WHERE c.ts @@ lex_q.q
    AND c.is_parent = false
    AND (filter_source_type IS NULL OR c.source_type = filter_source_type)
    AND (filter_source_id IS NULL OR c.source_id = filter_source_id)
  ORDER BY ts_rank(c.ts, lex_q.q) DESC
  LIMIT k_lexical
),
fused AS (
  SELECT
    COALESCE(d.chunk_id, l.chunk_id) AS chunk_id,
    (COALESCE(1.0 / (rrf_k + d.rnk), 0.0)
     + COALESCE(1.0 / (rrf_k + l.rnk), 0.0)) AS score,
    d.rnk AS dense_rank,
    l.rnk AS lexical_rank
  FROM dense_hits d
  FULL OUTER JOIN lexical l ON d.chunk_id = l.chunk_id
)
SELECT
  c.chunk_id, c.text, c.section, c.source_id, c.page, c.url,
  c.parent_id, c.chunk_index, c.title, c.lang,
  f.score::float4 AS rrf_score,
  f.dense_rank,
  f.lexical_rank
FROM fused f
JOIN {{chunks}} c ON c.chunk_id = f.chunk_id
ORDER BY f.score DESC
LIMIT k_final;
$search$;
