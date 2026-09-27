"""Ordered, schema-parametrised SQL migrations for the RAG datastore.

SQL files live in the repo-root ``migrations/`` folder (``{{placeholders}}``).
The runner applies them in filename order and records versions in
``<schema>.schema_migrations``.

    python -m common.migrate --schema v2 --dry-run
    python -m common.migrate --schema v2 --apply

``--dry-run`` never connects; it prints the exact SQL that ``--apply`` would
run on a fresh schema. ``--apply`` requires a DSN and refuses the ``public``
schema unless ``--allow-public`` is passed (live legacy data).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .datastore import (
    _PUBLIC_SCHEMAS,
    _ensure_vector_extension,
    _sql_ident,
    index_ident,
    qualify_ident,
)

PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
_MIGRATION_NAME = re.compile(r"^(\d{4})_.+\.sql$")
_MIGRATIONS_ENV = "ARKGURU_MIGRATIONS_DIR"

# CREATE TABLE for the tracking table is issued by the runner before files run.
_TRACKING_DDL = """\
CREATE TABLE IF NOT EXISTS {schema}.schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
"""


class PublicSchemaForbidden(RuntimeError):
    """Raised when a write would touch ``public`` without an explicit opt-in."""


class MigrationError(RuntimeError):
    """Invalid migration files or placeholders."""


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    sql: str

    @property
    def name(self) -> str:
        return self.path.name


def find_migrations_dir(start: Optional[Path] = None) -> Path:
    """Locate ``migrations/*.sql``.

    Order: ``$ARKGURU_MIGRATIONS_DIR``, then walk up from ``start`` (default:
    this file) looking for a ``migrations/`` directory that contains ``*.sql``,
    stopping at a git root.
    """
    env = (os.environ.get(_MIGRATIONS_ENV) or "").strip()
    if env:
        p = Path(env)
        if not p.is_dir():
            raise FileNotFoundError(f"{_MIGRATIONS_ENV}={env!r} is not a directory")
        return p
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(10):
        cand = cur / "migrations"
        if cand.is_dir() and any(cand.glob("*.sql")):
            return cand
        if (cur / ".git").exists() or cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        "migrations/ not found (set ARKGURU_MIGRATIONS_DIR or run from the repo)"
    )


def list_migration_files(directory: Optional[Path] = None) -> list[Path]:
    d = directory or find_migrations_dir()
    files = sorted(p for p in d.glob("*.sql") if _MIGRATION_NAME.match(p.name))
    if not files:
        raise MigrationError(f"no numbered *.sql files in {d}")
    versions = [p.stem.split("_", 1)[0] for p in files]
    if len(versions) != len(set(versions)):
        raise MigrationError(f"duplicate migration versions in {d}")
    return files


def load_migrations(directory: Optional[Path] = None) -> list[Migration]:
    out: list[Migration] = []
    for path in list_migration_files(directory):
        out.append(Migration(
            version=path.stem,
            path=path,
            sql=path.read_text(encoding="utf-8"),
        ))
    return out


def _required_placeholders(sql: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(sql))


def render_sql(sql: str, values: dict[str, str]) -> str:
    """Replace ``{{name}}`` placeholders. Extra values are ignored."""

    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise MigrationError(
                f"unresolved placeholder {{{{{key}}}}} "
                f"(have {sorted(values)})"
            )
        return values[key]

    rendered = PLACEHOLDER_RE.sub(repl, sql)
    leftover = _required_placeholders(rendered)
    if leftover:
        raise MigrationError(f"unresolved placeholders: {sorted(leftover)}")
    return rendered


def split_sql(sql: str) -> list[str]:
    """Split a script into statements, respecting dollar-quotes and strings."""
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    in_single = False
    dollar: Optional[str] = None
    while i < n:
        ch = sql[i]
        if dollar is not None:
            if sql.startswith(dollar, i):
                buf.append(dollar)
                i += len(dollar)
                dollar = None
                continue
            buf.append(ch)
            i += 1
            continue
        if in_single:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and sql[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            end = sql.find("\n", i)
            if end == -1:
                buf.append(sql[i:])
                break
            buf.append(sql[i:end + 1])
            i = end + 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == "$":
            m = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql[i:])
            if m:
                dollar = m.group(0)
                buf.append(dollar)
                i += len(dollar)
                continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return [s for s in statements if s and not _comment_only(s)]


def _comment_only(sql: str) -> bool:
    for line in sql.splitlines():
        t = line.strip()
        if t and not t.startswith("--"):
            return False
    return True


def placeholder_values(
    *,
    schema: str,
    dim: int = 1024,
    chunks_table: str = "chunks",
    vectors_table: str = "chunk_embeddings",
    function_name: str = "search_chunks",
) -> dict[str, str]:
    """Build the substitution map. Identifiers are validated first."""
    schema_ident = _sql_ident(schema.strip())
    chunks_bare = _sql_ident(chunks_table)
    vectors_bare = _sql_ident(vectors_table)
    fn_bare = _sql_ident(function_name)
    # Always schema-qualify, including public, so dry-run SQL is unambiguous.
    chunks = f"{schema_ident}.{chunks_bare}"
    vectors = f"{schema_ident}.{vectors_bare}"
    function = f"{schema_ident}.{fn_bare}"
    try:
        dim_i = int(dim)
    except (TypeError, ValueError) as e:
        raise MigrationError(f"dim must be an integer, got {dim!r}") from e
    if dim_i < 1:
        raise MigrationError(f"dim must be >= 1, got {dim_i}")
    return {
        "schema": schema_ident,
        "dim": str(dim_i),
        "chunks": chunks,
        "vectors": vectors,
        "function": function,
        "chunks_bare": chunks_bare,
        "vectors_bare": vectors_bare,
        "chunks_index": index_ident(chunks, ""),
        "vectors_index": index_ident(vectors, ""),
        "function_unqual": fn_bare,
        "qualify_chunks": qualify_ident(chunks_bare, schema_ident),
    }


def render_migration(migration: Migration, values: dict[str, str]) -> str:
    return render_sql(migration.sql, values)


def render_named_migration(
    filename: str,
    *,
    schema: str,
    dim: int = 1024,
    chunks_table: str = "chunks",
    vectors_table: str = "chunk_embeddings",
    function_name: str = "search_chunks",
    directory: Optional[Path] = None,
    extra: Optional[dict[str, str]] = None,
) -> str:
    """Render one file (used by ``_search_chunks_ddl`` / tests)."""
    values = placeholder_values(
        schema=schema, dim=dim, chunks_table=chunks_table,
        vectors_table=vectors_table, function_name=function_name,
    )
    if extra:
        values.update(extra)
    path = (directory or find_migrations_dir()) / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return render_sql(path.read_text(encoding="utf-8"), values)


def _is_public(schema: str) -> bool:
    return (schema or "").strip().lower() in _PUBLIC_SCHEMAS or (
        (schema or "").strip().lower() == "public"
    )


def assert_schema_writable(schema: str, *, allow_public: bool) -> None:
    _sql_ident(schema.strip())
    if _is_public(schema) and not allow_public:
        raise PublicSchemaForbidden(
            f"refusing to modify schema {schema!r} (live legacy data). "
            "Pass --allow-public / allow_public=True after reviewing --dry-run, "
            "or apply to a new schema (e.g. --schema v2)."
        )


def prelude_sql(schema: str) -> str:
    """SQL the runner executes before numbered files (schema + tracking table)."""
    ident = _sql_ident(schema.strip())
    parts = []
    if not _is_public(ident):
        parts.append(f"CREATE SCHEMA IF NOT EXISTS {ident};")
    parts.append(_TRACKING_DDL.format(schema=ident))
    return "\n".join(parts) + "\n"


def dry_run_sql(
    schema: str,
    *,
    dim: int = 1024,
    chunks_table: str = "chunks",
    vectors_table: str = "chunk_embeddings",
    directory: Optional[Path] = None,
) -> str:
    """Exact SQL ``--apply`` would run on a fresh schema. No connection."""
    values = placeholder_values(
        schema=schema, dim=dim, chunks_table=chunks_table,
        vectors_table=vectors_table,
    )
    blocks = [
        "-- dry-run: no connection; this is the full set for a fresh schema.",
        "-- --apply skips versions already in <schema>.schema_migrations.",
        prelude_sql(schema).rstrip(),
    ]
    for mig in load_migrations(directory):
        blocks.append(f"-- === {mig.name} ===")
        blocks.append(render_migration(mig, values).rstrip())
        blocks.append(
            f"INSERT INTO {values['schema']}.schema_migrations (version) "
            f"VALUES ('{mig.version}') ON CONFLICT (version) DO NOTHING;"
        )
    return "\n\n".join(blocks) + "\n"


def _applied_versions(cur, schema: str) -> set[str]:
    ident = _sql_ident(schema)
    cur.execute(f"SELECT version FROM {ident}.schema_migrations")
    rows = cur.fetchall() or []
    return {r[0] for r in rows if r and r[0]}


def apply_migrations(
    store,
    *,
    allow_public: bool = False,
    dim: Optional[int] = None,
    directory: Optional[Path] = None,
    conn=None,
) -> list[str]:
    """Apply pending migrations to ``store.schema``. Returns versions applied.

    ``conn`` is an optional open connection (tests). Otherwise ``store._connect()``.
    """
    schema = store.schema
    assert_schema_writable(schema, allow_public=allow_public)
    values = placeholder_values(
        schema=schema,
        dim=int(dim if dim is not None else store.dim),
        chunks_table=store.chunks.split(".")[-1],
        vectors_table=store.vectors.split(".")[-1],
    )
    migrations = load_migrations(directory)
    owns_conn = conn is None
    if owns_conn:
        conn = store._connect()
    applied: list[str] = []
    try:
        with conn.cursor() as cur:
            _ensure_vector_extension(cur)
            for stmt in split_sql(prelude_sql(schema)):
                cur.execute(stmt)
            already = _applied_versions(cur, schema)
            for mig in migrations:
                if mig.version in already:
                    continue
                for stmt in split_sql(render_migration(mig, values)):
                    cur.execute(stmt)
                cur.execute(
                    f"INSERT INTO {values['schema']}.schema_migrations (version) "
                    f"VALUES (%s) ON CONFLICT (version) DO NOTHING",
                    (mig.version,),
                )
                applied.append(mig.version)
        if hasattr(conn, "commit"):
            conn.commit()
    finally:
        if owns_conn and conn is not None and hasattr(conn, "close"):
            conn.close()
    return applied


def _store_from_args(args):
    from .datastore import ChunkStore
    from .datastore_config import load_datastore_config, resolve_dsn

    cfg = load_datastore_config(args.datastore_config)
    pg = dict(cfg.get("postgres") or {})
    dsn = args.dsn or pg.get("dsn") or resolve_dsn(pg)
    dim = args.dim if args.dim is not None else int(pg.get("dim") or 1024)
    return ChunkStore(
        dsn=dsn,
        table=pg.get("chunks_table", "chunks"),
        vectors_table=pg.get("vectors_table", "chunk_embeddings"),
        dim=dim,
        schema=args.schema,
        allow_public_schema=bool(args.allow_public),
    ), dim


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Apply RAG schema migrations (or print them with --dry-run).")
    ap.add_argument(
        "--schema", required=True,
        help="Target Postgres schema (required). Use v2 for the new corpus.")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the exact SQL and exit. Never connects.")
    ap.add_argument(
        "--apply", action="store_true",
        help="Apply pending migrations. Requires PG_DSN / --dsn.")
    ap.add_argument(
        "--allow-public", action="store_true",
        help="Permit --apply against the public schema (live legacy data).")
    ap.add_argument("--dsn", default=None, help="Override PG_DSN.")
    ap.add_argument("--dim", type=int, default=None,
                    help="vector width (default: datastore.yaml / 1024).")
    ap.add_argument("--datastore-config", default=None)
    ap.add_argument(
        "--migrations-dir", default=None,
        help="Override the migrations/ folder.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.dry_run and args.apply:
        ap.error("choose one of --dry-run or --apply")
    if not args.dry_run and not args.apply:
        ap.error("choose --dry-run (print SQL) or --apply (write)")

    _sql_ident(args.schema.strip())
    directory = Path(args.migrations_dir) if args.migrations_dir else None
    dim = args.dim if args.dim is not None else 1024

    if args.dry_run:
        print(dry_run_sql(args.schema, dim=dim, directory=directory), end="")
        return 0

    try:
        assert_schema_writable(args.schema, allow_public=args.allow_public)
    except PublicSchemaForbidden as exc:
        raise SystemExit(str(exc)) from exc
    try:
        store, dim = _store_from_args(args)
    except ValueError as exc:
        raise SystemExit(
            "No Postgres DSN. Set PG_DSN or pass --dsn before --apply. "
            f"({exc})"
        ) from exc
    if not store.dsn:
        raise SystemExit(
            "No Postgres DSN. Set PG_DSN or pass --dsn before --apply.")
    applied = apply_migrations(
        store, allow_public=args.allow_public, dim=dim, directory=directory)
    if applied:
        print(f"applied {len(applied)} migration(s) to {args.schema}: "
              + ", ".join(applied))
    else:
        print(f"schema {args.schema}: already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
