-- 0002_search_chunks_v2: hybrid retrieve with a phrase leg.
-- Dense HNSW cosine + websearch_to_tsquery (AND, OR fallback) +
-- phraseto_tsquery on multi-word domain terms, fused with RRF.
-- Excludes parent chunks, quality-gate failures, and (dense only)
-- chunks that have no embedding. Filters: source_type, source_id, lang.
-- Placeholders: {{schema}} {{dim}} {{chunks}} {{vectors}} {{function}}

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
  filter_source_id   text DEFAULT NULL,
  k_phrase        int  DEFAULT 20,
  filter_lang     text DEFAULT NULL,
  phrase_terms    text[] DEFAULT NULL
)
RETURNS TABLE (
  chunk_id text, text text, section text, source_id text,
  page int, url text, parent_id text, chunk_index int,
  title text, lang text,
  rrf_score float4, dense_rank int, lexical_rank int, phrase_rank int
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
    AND (c.meta->'quality'->>'embed' IS NULL
         OR c.meta->'quality'->>'embed' IN ('true', '1'))
    AND (c.meta->'quality'->>'passed' IS NULL
         OR c.meta->'quality'->>'passed' IN ('true', '1'))
    AND (filter_source_type IS NULL OR c.source_type = filter_source_type)
    AND (filter_source_id IS NULL OR c.source_id = filter_source_id)
    AND (filter_lang IS NULL OR c.lang = filter_lang)
),
lex_and AS (
  SELECT websearch_to_tsquery('english', query_text) AS q
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
      AND (c.meta->'quality'->>'embed' IS NULL
           OR c.meta->'quality'->>'embed' IN ('true', '1'))
      AND (c.meta->'quality'->>'passed' IS NULL
           OR c.meta->'quality'->>'passed' IN ('true', '1'))
      AND (filter_source_type IS NULL OR c.source_type = filter_source_type)
      AND (filter_source_id IS NULL OR c.source_id = filter_source_id)
      AND (filter_lang IS NULL OR c.lang = filter_lang)
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
    AND (c.meta->'quality'->>'embed' IS NULL
         OR c.meta->'quality'->>'embed' IN ('true', '1'))
    AND (c.meta->'quality'->>'passed' IS NULL
         OR c.meta->'quality'->>'passed' IN ('true', '1'))
    AND (filter_source_type IS NULL OR c.source_type = filter_source_type)
    AND (filter_source_id IS NULL OR c.source_id = filter_source_id)
    AND (filter_lang IS NULL OR c.lang = filter_lang)
  ORDER BY ts_rank(c.ts, lex_q.q) DESC
  LIMIT k_lexical
),
phrase_q AS (
  SELECT NULLIF(
    (SELECT string_agg(phraseto_tsquery('english', btrim(p))::text, ' & ')
     FROM unnest(COALESCE(phrase_terms, ARRAY[]::text[])) AS p
     WHERE btrim(p) <> ''),
    ''
  )::tsquery AS q
),
phrase AS (
  SELECT c.chunk_id,
         ROW_NUMBER() OVER (
           ORDER BY ts_rank(c.ts, phrase_q.q) DESC
         )::int AS rnk
  FROM {{chunks}} c, phrase_q
  WHERE phrase_q.q IS NOT NULL
    AND c.ts @@ phrase_q.q
    AND c.is_parent = false
    AND (c.meta->'quality'->>'embed' IS NULL
         OR c.meta->'quality'->>'embed' IN ('true', '1'))
    AND (c.meta->'quality'->>'passed' IS NULL
         OR c.meta->'quality'->>'passed' IN ('true', '1'))
    AND (filter_source_type IS NULL OR c.source_type = filter_source_type)
    AND (filter_source_id IS NULL OR c.source_id = filter_source_id)
    AND (filter_lang IS NULL OR c.lang = filter_lang)
  ORDER BY ts_rank(c.ts, phrase_q.q) DESC
  LIMIT k_phrase
),
fused AS (
  SELECT
    COALESCE(d.chunk_id, l.chunk_id, p.chunk_id) AS chunk_id,
    (COALESCE(1.0 / (rrf_k + d.rnk), 0.0)
     + COALESCE(1.0 / (rrf_k + l.rnk), 0.0)
     + COALESCE(1.0 / (rrf_k + p.rnk), 0.0)) AS score,
    d.rnk AS dense_rank,
    l.rnk AS lexical_rank,
    p.rnk AS phrase_rank
  FROM dense_hits d
  FULL OUTER JOIN lexical l ON d.chunk_id = l.chunk_id
  FULL OUTER JOIN phrase p ON COALESCE(d.chunk_id, l.chunk_id) = p.chunk_id
)
SELECT
  c.chunk_id, c.text, c.section, c.source_id, c.page, c.url,
  c.parent_id, c.chunk_index, c.title, c.lang,
  f.score::float4 AS rrf_score,
  f.dense_rank,
  f.lexical_rank,
  f.phrase_rank
FROM fused f
JOIN {{chunks}} c ON c.chunk_id = f.chunk_id
ORDER BY f.score DESC
LIMIT k_final;
$search$;
