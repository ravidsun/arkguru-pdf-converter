"""Read-only Phase 4 validation checks against a RAG schema.

    python -m common.validate_schema --schema v2

Each check is a SELECT. The CLI never issues DDL/DML. ``public`` is allowed
(read-only). Exit code is 1 when any check misses its expected bound.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

from .datastore import _sql_ident

# Phase 4 plan SQL, schema-qualified. ``expect`` is ("eq", n) or ("ge", n).
CHECKS: tuple[tuple[str, str, tuple[str, int]], ...] = (
    (
        "embedding_model",
        "SELECT count(*) FROM {vectors} "
        "WHERE model IS DISTINCT FROM 'BAAI/bge-m3'",
        ("eq", 0),
    ),
    (
        "vector_dim_norm",
        "SELECT count(*) FROM {vectors} "
        "WHERE vector_dims(embedding) <> 1024 "
        "OR vector_norm(embedding) NOT BETWEEN 0.99 AND 1.01",
        ("eq", 0),
    ),
    (
        "child_missing_embedding",
        "SELECT count(*) FROM {chunks} c "
        "LEFT JOIN {vectors} v USING (chunk_id) "
        "WHERE NOT coalesce(c.is_parent, false) AND v.chunk_id IS NULL",
        ("eq", 0),
    ),
    (
        "empty_lang",
        "SELECT count(*) FROM {chunks} WHERE lang IS NULL OR lang = ''",
        ("eq", 0),
    ),
    (
        "lang_ne",
        "SELECT count(*) FROM {chunks} WHERE lang = 'ne'",
        ("eq", 0),
    ),
    (
        "oversized_child",
        "SELECT count(*) FROM {chunks} "
        "WHERE NOT is_parent AND length(text) > 4000",
        ("eq", 0),
    ),
    (
        "pipe_heavy_nontable",
        "SELECT count(*) FROM {chunks} "
        "WHERE NOT is_parent AND coalesce(meta->>'block_type', 'text') <> 'table' "
        "AND (text LIKE '|%%' OR length(text) - length(replace(text, '|', '')) "
        "> 0.10 * length(text))",
        ("eq", 0),
    ),
    (
        "coln_headers",
        "SELECT count(*) FROM {chunks} WHERE text ~ '\\|Col[0-9]+\\|'",
        ("eq", 0),
    ),
    (
        "inline_html_tags",
        "SELECT count(*) FROM {chunks} "
        "WHERE text ~* '</?(mark|u|sup|br)\\s*/?>'",
        ("eq", 0),
    ),
    (
        "tiny_child",
        "SELECT count(*) FROM {chunks} "
        "WHERE NOT is_parent AND length(text) < 100",
        ("eq", 0),
    ),
    (
        "duplicate_child_text",
        "SELECT count(*) FROM ("
        "SELECT md5(btrim(text)) FROM {chunks} WHERE NOT is_parent "
        "GROUP BY 1 HAVING count(*) > 1) d",
        ("eq", 0),
    ),
    (
        "duplicate_pdf_sha256",
        "SELECT count(*) FROM ("
        "SELECT meta->>'sha256' FROM {chunks} WHERE source_type = 'pdf' "
        "GROUP BY 1 HAVING count(DISTINCT source_id) > 1) d",
        ("eq", 0),
    ),
    (
        "orphan_parent_id",
        "SELECT count(*) FROM {chunks} c "
        "WHERE coalesce(c.parent_id, '') <> '' AND NOT EXISTS ("
        "SELECT 1 FROM {chunks} p "
        "WHERE p.chunk_id = c.parent_id AND p.is_parent)",
        ("eq", 0),
    ),
    (
        "pdf_child_without_parent",
        "SELECT count(*) FROM {chunks} "
        "WHERE source_type = 'pdf' AND NOT is_parent "
        "AND coalesce(parent_id, '') = ''",
        ("eq", 0),
    ),
    (
        "sitemap_web_rows",
        "SELECT count(*) FROM {chunks} "
        "WHERE source_type = 'web' "
        "AND (source_id ~* 'site-?map' OR url ~* 'site-?map')",
        ("eq", 0),
    ),
    (
        "search_chunks_function",
        "SELECT count(*) FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE p.proname = 'search_chunks' AND n.nspname = '{schema}'",
        ("ge", 1),
    ),
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    sql: str
    value: Optional[int]
    expect: tuple[str, int]
    ok: bool
    error: Optional[str] = None

    @property
    def expected_label(self) -> str:
        op, n = self.expect
        if op == "eq":
            return f"== {n}"
        if op == "ge":
            return f">= {n}"
        never: str = op
        raise ValueError(f"unknown expect op: {never}")


def _assert_readonly(sql: str) -> str:
    stripped = sql.lstrip().lstrip("(").lstrip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError(f"validation SQL must be a SELECT, got: {sql[:80]!r}")
    return sql


def render_checks(schema: str, *, chunks_table: str = "chunks",
                  vectors_table: str = "chunk_embeddings"
                  ) -> list[tuple[str, str, tuple[str, int]]]:
    ident = _sql_ident(schema.strip())
    chunks = f"{ident}.{_sql_ident(chunks_table)}"
    vectors = f"{ident}.{_sql_ident(vectors_table)}"
    out = []
    for name, sql, expect in CHECKS:
        rendered = sql.format(schema=ident, chunks=chunks, vectors=vectors)
        _assert_readonly(rendered)
        out.append((name, rendered, expect))
    return out


def _meets(value: int, expect: tuple[str, int]) -> bool:
    op, n = expect
    if op == "eq":
        return value == n
    if op == "ge":
        return value >= n
    never: str = op
    raise ValueError(f"unknown expect op: {never}")


def run_checks(store) -> list[CheckResult]:
    """Execute the Phase 4 SELECTs. Read-only: no commit of writes."""
    rendered = render_checks(
        store.schema,
        chunks_table=store.chunks.split(".")[-1],
        vectors_table=store.vectors.split(".")[-1],
    )
    results: list[CheckResult] = []
    with store._connect() as conn, conn.cursor() as cur:
        # Make the session read-only so a mistaken write cannot commit.
        cur.execute("SET TRANSACTION READ ONLY")
        for name, sql, expect in rendered:
            try:
                cur.execute(sql)
                row = cur.fetchone()
                value = int(row[0]) if row and row[0] is not None else 0
                results.append(CheckResult(
                    name=name, sql=sql, value=value, expect=expect,
                    ok=_meets(value, expect),
                ))
            except Exception as exc:
                results.append(CheckResult(
                    name=name, sql=sql, value=None, expect=expect,
                    ok=False, error=str(exc),
                ))
    return results


def format_report(results: list[CheckResult], schema: str) -> str:
    lines = [f"Phase 4 validation  schema={schema}  checks={len(results)}"]
    width = max(len(r.name) for r in results)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        if r.error:
            detail = f"error: {r.error}"
        else:
            detail = f"got {r.value} (expect {r.expected_label})"
        lines.append(f"  {status:4}  {r.name:<{width}}  {detail}")
    failed = sum(1 for r in results if not r.ok)
    lines.append(f"{failed} failed, {len(results) - failed} passed")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Read-only Phase 4 validation SQL against a schema.")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--datastore-config", default=None)
    ap.add_argument(
        "--sql-only", action="store_true",
        help="Print the SELECTs and exit (no connection).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    _sql_ident(args.schema.strip())
    if args.sql_only:
        for name, sql, expect in render_checks(args.schema):
            op, n = expect
            print(f"-- {name}  expect {op} {n}")
            print(sql + ";")
            print()
        return 0

    from .datastore import ChunkStore
    from .datastore_config import load_datastore_config, resolve_dsn

    cfg = load_datastore_config(args.datastore_config)
    pg = dict(cfg.get("postgres") or {})
    dsn = args.dsn or pg.get("dsn") or resolve_dsn(pg)
    if not dsn:
        raise SystemExit("No Postgres DSN. Set PG_DSN or pass --dsn.")
    store = ChunkStore(
        dsn=dsn,
        table=pg.get("chunks_table", "chunks"),
        vectors_table=pg.get("vectors_table", "chunk_embeddings"),
        dim=int(pg.get("dim") or 1024),
        schema=args.schema,
    )
    results = run_checks(store)
    sys.stdout.write(format_report(results, args.schema))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
