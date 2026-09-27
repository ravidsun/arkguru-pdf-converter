"""Migration runner: dry-run SQL, public guard, placeholders, splitter."""
from __future__ import annotations

import pytest

from common.datastore import ChunkStore
from common.migrate import (
    PublicSchemaForbidden,
    assert_schema_writable,
    dry_run_sql,
    find_migrations_dir,
    load_migrations,
    main as migrate_main,
    placeholder_values,
    render_sql,
    split_sql,
)


def test_migrations_dir_has_ordered_files():
    files = load_migrations()
    names = [m.name for m in files]
    assert names[0] == "0001_baseline.sql"
    assert names[1] == "0002_search_chunks_v2.sql"
    assert names == sorted(names)
    assert find_migrations_dir().is_dir()


def test_placeholder_values_qualify_including_public():
    v = placeholder_values(schema="v2", dim=1024)
    assert v["schema"] == "v2"
    assert v["chunks"] == "v2.chunks"
    assert v["vectors"] == "v2.chunk_embeddings"
    assert v["function"] == "v2.search_chunks"
    assert v["dim"] == "1024"
    pub = placeholder_values(schema="public")
    assert pub["chunks"] == "public.chunks"
    assert pub["function"] == "public.search_chunks"


def test_render_sql_rejects_unknown_placeholder():
    with pytest.raises(Exception, match="unresolved"):
        render_sql("SELECT {{nope}}", {"schema": "v2"})


def test_split_sql_keeps_statement_after_leading_comment():
    stmts = split_sql("-- header\nCREATE TABLE t (id int);\n-- only comment;\n")
    assert len(stmts) == 1
    assert "CREATE TABLE t" in stmts[0]


def test_split_sql_respects_dollar_quotes():
    script = """
    CREATE TABLE t (id int);
    CREATE FUNCTION f() RETURNS int AS $search$
    BEGIN
      RETURN 1;
    END;
    $search$;
    SELECT 2;
    """
    stmts = split_sql(script)
    assert len(stmts) == 3
    assert "CREATE TABLE t" in stmts[0]
    assert "$search$" in stmts[1]
    assert "SELECT 2" in stmts[2]


def test_dry_run_prints_exact_rendered_sql():
    sql = dry_run_sql("v2", dim=1024)
    assert "CREATE SCHEMA IF NOT EXISTS v2" in sql
    assert "CREATE TABLE IF NOT EXISTS v2.chunks" in sql
    assert "CREATE TABLE IF NOT EXISTS v2.chunk_embeddings" in sql
    assert "vector(1024)" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "USING gin (ts)" in sql
    assert "chunks_content_hash_child_uidx" in sql
    assert "CREATE OR REPLACE FUNCTION v2.search_chunks" in sql
    assert "phraseto_tsquery" in sql
    assert "websearch_to_tsquery" in sql
    assert "INSERT INTO v2.schema_migrations" in sql
    assert "{{schema}}" not in sql
    assert "{{dim}}" not in sql


def test_dry_run_cli_does_not_need_dsn(capsys):
    rc = migrate_main(["--schema", "v2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CREATE TABLE IF NOT EXISTS v2.chunks" in out
    assert "phraseto_tsquery" in out


def test_cli_requires_dry_run_or_apply():
    with pytest.raises(SystemExit):
        migrate_main(["--schema", "v2"])


def test_apply_refuses_public_without_flag():
    store = ChunkStore(dsn="postgresql://unused", schema="public")
    with pytest.raises(PublicSchemaForbidden, match="public"):
        assert_schema_writable(store.schema, allow_public=False)
    with pytest.raises(SystemExit):
        migrate_main(["--schema", "public", "--apply"])


def test_apply_cli_without_dsn_fails(monkeypatch):
    monkeypatch.delenv("PG_DSN", raising=False)
    with pytest.raises(SystemExit, match="DSN"):
        migrate_main(["--schema", "v2", "--apply"])


def test_invalid_schema_rejected():
    with pytest.raises(ValueError):
        placeholder_values(schema="v2; DROP TABLE chunks")
