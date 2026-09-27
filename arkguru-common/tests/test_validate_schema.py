"""Phase 4 validation CLI is read-only and schema-parametrised."""
from __future__ import annotations

from common.validate_schema import (
    CHECKS,
    _assert_readonly,
    format_report,
    render_checks,
    CheckResult,
)


def test_all_checks_are_selects():
    assert len(CHECKS) >= 16
    for name, sql, expect in render_checks("v2"):
        _assert_readonly(sql)
        assert "v2.chunks" in sql or "nspname = 'v2'" in sql or name == "search_chunks_function"
        assert "{{" not in sql
        assert expect[0] in {"eq", "ge"}


def test_sql_only_cli(capsys):
    from common.validate_schema import main
    rc = main(["--schema", "v2", "--sql-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SELECT count(*)" in out
    assert "v2.chunks" in out
    assert "INSERT" not in out.upper().replace("SELECT", "")
    assert "UPDATE" not in out
    assert "DELETE" not in out


def test_format_report_fail_exit_shape():
    results = [
        CheckResult("ok", "SELECT 0", 0, ("eq", 0), True),
        CheckResult("bad", "SELECT 2", 2, ("eq", 0), False),
    ]
    text = format_report(results, "v2")
    assert "PASS" in text
    assert "FAIL" in text
    assert "1 failed" in text
