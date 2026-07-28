"""Tests for SQL safety guardrails (read-only enforcement, LIMIT injection)."""

from __future__ import annotations

import pytest

from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.safety import (
    SQLSafetyError,
    StatementType,
    classify_statement,
    classify_statement_ast,
    enforce_read_only,
    inject_limit,
    run_with_timeout,
)
from universal_text2sql.utils.demo_data import seed_demo_database


@pytest.fixture()
def connector() -> DatabaseConnector:
    """In-memory SQLite connector seeded with demo data, read-only by default."""
    conn = DatabaseConnector("sqlite:///:memory:", read_only=False)
    seed_demo_database(conn)
    conn.read_only = True
    return conn


class TestClassifyStatement:
    @pytest.mark.parametrize(
        "sql,expected",
        [
            ("SELECT * FROM customers", StatementType.READ),
            ("  select * from customers  ", StatementType.READ),
            ("WITH x AS (SELECT 1) SELECT * FROM x", StatementType.READ),
            ("EXPLAIN SELECT * FROM customers", StatementType.READ),
            ("INSERT INTO customers VALUES (1)", StatementType.WRITE),
            ("UPDATE customers SET email = 'x'", StatementType.WRITE),
            ("DELETE FROM customers", StatementType.WRITE),
            ("DROP TABLE customers", StatementType.WRITE),
            ("ALTER TABLE customers ADD COLUMN x TEXT", StatementType.WRITE),
            ("PRAGMA table_info(customers)", StatementType.WRITE),
        ],
    )
    def test_single_statement_classification(self, sql, expected):
        assert classify_statement(sql) == expected

    def test_multi_statement_is_blocked(self):
        assert classify_statement("SELECT 1; DROP TABLE customers;") == StatementType.WRITE

    def test_empty_sql_is_unknown(self):
        assert classify_statement("   ") == StatementType.UNKNOWN


class TestClassifyStatementAst:
    def test_plain_select(self):
        pytest.importorskip("sqlglot")
        assert classify_statement_ast("SELECT * FROM customers", "SQLite") == StatementType.READ

    def test_cte_that_looks_readonly_but_writes(self):
        """The regex baseline can't see past the leading WITH; the AST mode can."""
        pytest.importorskip("sqlglot")
        sql = "WITH x AS (SELECT * FROM customers) DELETE FROM x"
        assert classify_statement(sql) == StatementType.READ  # known regex limitation
        assert classify_statement_ast(sql, "SQLite") == StatementType.WRITE

    def test_multi_statement(self):
        pytest.importorskip("sqlglot")
        assert (
            classify_statement_ast("SELECT 1; DROP TABLE customers;", "SQLite")
            == StatementType.WRITE
        )

    def test_falls_back_to_none_on_unparseable_syntax(self):
        pytest.importorskip("sqlglot")
        # EXPLAIN isn't fully modelled by sqlglot's generic grammar -> None,
        # deferring to the regex baseline (which does classify it as READ).
        assert classify_statement_ast("EXPLAIN SELECT * FROM customers", "SQLite") is None


class TestEnforceReadOnly:
    def test_blocks_write(self):
        with pytest.raises(SQLSafetyError):
            enforce_read_only("DROP TABLE customers", "SQLite")

    def test_allows_select(self):
        enforce_read_only("SELECT * FROM customers", "SQLite")  # should not raise

    def test_allow_writes_bypasses_check(self):
        enforce_read_only("DROP TABLE customers", "SQLite", allow_writes=True)

    def test_blocks_cte_write_via_ast(self):
        pytest.importorskip("sqlglot")
        with pytest.raises(SQLSafetyError):
            enforce_read_only("WITH x AS (SELECT * FROM customers) DELETE FROM x", "SQLite")


class TestInjectLimit:
    def test_adds_limit(self):
        result = inject_limit("SELECT * FROM customers", 10)
        assert "LIMIT 10" in result

    def test_noop_when_limit_present(self):
        sql = "SELECT * FROM customers LIMIT 5"
        assert inject_limit(sql, 10) == sql

    def test_rejects_non_positive_limit(self):
        with pytest.raises(ValueError):
            inject_limit("SELECT * FROM customers", 0)


class TestRunWithTimeout:
    def test_no_timeout_runs_normally(self):
        assert run_with_timeout(lambda: 42, None) == 42

    def test_timeout_raises_safety_error(self):
        import time

        with pytest.raises(SQLSafetyError):
            run_with_timeout(lambda: time.sleep(1), 0.01)


class TestDatabaseConnectorReadOnly:
    def test_read_only_blocks_write(self, connector: DatabaseConnector):
        with pytest.raises(SQLSafetyError):
            connector.execute_query("DELETE FROM customers")

    def test_read_only_allows_select(self, connector: DatabaseConnector):
        df = connector.execute_query("SELECT COUNT(*) AS cnt FROM customers")
        assert df.iloc[0]["cnt"] == 10

    def test_writable_connector_allows_write(self):
        conn = DatabaseConnector("sqlite:///:memory:", read_only=False)
        seed_demo_database(conn)
        conn.execute_query("DELETE FROM customers WHERE customer_id = 1")
        df = conn.execute_query("SELECT COUNT(*) AS cnt FROM customers")
        assert df.iloc[0]["cnt"] == 9

    def test_per_call_override(self):
        conn = DatabaseConnector("sqlite:///:memory:", read_only=True)
        seed_demo_database(conn)
        # Read-only by default, but this call explicitly opts out.
        conn.execute_query("DELETE FROM customers WHERE customer_id = 1", enforce_read_only=False)
        df = conn.execute_query("SELECT COUNT(*) AS cnt FROM customers")
        assert df.iloc[0]["cnt"] == 9

    def test_limit_is_applied(self, connector: DatabaseConnector):
        df = connector.execute_query("SELECT * FROM customers", limit=3)
        assert len(df) == 3

    def test_default_is_read_only(self):
        conn = DatabaseConnector("sqlite:///:memory:")
        assert conn.read_only is True


class TestBootstrapRegression:
    def test_bootstrap_agent_still_answers_select_questions(self, tmp_path, monkeypatch):
        """The new read_only=True default must not affect normal question-answering."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("QUERY_MEMORY_PATH", str(tmp_path / "memory.json"))
        monkeypatch.setenv("METADATA_CACHE_DIR", str(tmp_path / "cache"))

        from universal_text2sql.bootstrap import bootstrap

        ctx = bootstrap(database_url="sqlite:///:memory:")
        assert ctx.connector.read_only is True
        df = ctx.connector.execute_query("SELECT COUNT(*) AS cnt FROM customers")
        assert df.iloc[0]["cnt"] == 10
