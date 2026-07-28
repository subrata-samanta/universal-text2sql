"""Universal database connector supporting SQLite, PostgreSQL, and MySQL."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from universal_text2sql.database.safety import (
    SQLSafetyError,
    inject_limit,
    run_with_timeout,
)
from universal_text2sql.database.safety import (
    enforce_read_only as _enforce_read_only,
)

logger = logging.getLogger(__name__)

__all__ = ["DatabaseConnector", "SQLSafetyError"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float value for %s=%r", name, raw)
        return None


class DatabaseConnector:
    """Universal database connector that works with any SQLAlchemy-supported database."""

    def __init__(
        self,
        connection_url: str,
        *,
        read_only: bool | None = None,
        statement_timeout_seconds: float | None = None,
    ) -> None:
        """Initialise the connector.

        Args:
            connection_url: SQLAlchemy connection URL, e.g.
                - ``sqlite:///./mydb.sqlite3``
                - ``postgresql://user:pw@localhost/db``
                - ``mysql+pymysql://user:pw@localhost/db``
            read_only: Whether :meth:`execute_query` enforces read-only SQL
                by default. Falls back to the ``SQL_READ_ONLY`` env var
                (default ``True``) -- writes require an explicit opt-in via
                ``execute_query(..., enforce_read_only=False)`` or
                ``DatabaseConnector(read_only=False)``. See
                :mod:`universal_text2sql.database.safety`.
            statement_timeout_seconds: Best-effort default query timeout.
                Falls back to ``SQL_STATEMENT_TIMEOUT_SECONDS`` (unset = no
                timeout).
        """
        self.connection_url = connection_url
        self.db_type = self._detect_db_type(connection_url)
        self.read_only = read_only if read_only is not None else _env_bool("SQL_READ_ONLY", True)
        self.statement_timeout_seconds = (
            statement_timeout_seconds
            if statement_timeout_seconds is not None
            else _env_float("SQL_STATEMENT_TIMEOUT_SECONDS")
        )
        self.engine: Engine = create_engine(
            connection_url,
            echo=False,
            future=True,
            # Pool settings comfortable for all supported backends
            pool_pre_ping=True,
        )
        logger.info("Connected to %s database (read_only=%s)", self.db_type, self.read_only)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_db_type(url: str) -> str:
        scheme = urlparse(url).scheme.split("+")[0]
        mapping = {"sqlite": "SQLite", "postgresql": "PostgreSQL", "mysql": "MySQL"}
        return mapping.get(scheme, scheme.capitalize())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_query(
        self,
        sql: str,
        *,
        enforce_read_only: bool | None = None,
        limit: int | None = None,
        timeout_seconds: float | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: The SQL statement to execute.
            enforce_read_only: Overrides the connector's ``read_only``
                setting for this call only. When enforced, non-``SELECT``
                statements raise :class:`~universal_text2sql.database.safety.SQLSafetyError`
                *before* touching the database.
            limit: When set, a ``LIMIT`` clause is appended (unless the
                query already has one) -- useful for capping the cost of
                exploratory/preview executions (e.g. self-consistency
                candidate voting) without truncating a final answer that
                wasn't asked to be limited.
            timeout_seconds: Best-effort wall-clock timeout for this call,
                overriding the connector's default. See
                :func:`~universal_text2sql.database.safety.run_with_timeout`
                for the guarantees (and non-guarantees) this provides.

        Returns:
            A :class:`pandas.DataFrame` with the query results.

        Raises:
            SQLSafetyError: If read-only enforcement is active and *sql* is
                not classified as a single read-only statement, or if the
                query exceeds its timeout.
            Exception: Re-raises any SQLAlchemy/database exception so that the
                agent's reflection loop can inspect the error message.
        """
        read_only = self.read_only if enforce_read_only is None else enforce_read_only
        if read_only:
            _enforce_read_only(sql, self.db_type, allow_writes=False)

        if limit is not None:
            sql = inject_limit(sql, limit, self.db_type)

        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self.statement_timeout_seconds
        )

        def _run() -> pd.DataFrame:
            # engine.begin() (rather than .connect()) so that a write
            # statement — permitted when read-only enforcement is off —
            # actually commits; a no-op wrapper for read-only SELECTs.
            with self.engine.begin() as conn:
                if effective_timeout and self.db_type == "PostgreSQL":
                    # Best-effort server-side enforcement; the thread-based
                    # timeout below is the portable fallback for backends
                    # (like SQLite) with no native statement timeout.
                    conn.execute(text(f"SET statement_timeout = {int(effective_timeout * 1000)}"))
                result = conn.execute(text(sql))
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = list(result.keys())
                    return pd.DataFrame(rows, columns=columns)
                # DML/DDL executed via the read_only=False escape hatch:
                # no rows to return, but report how many were affected.
                return pd.DataFrame({"rows_affected": [result.rowcount]})

        return run_with_timeout(_run, effective_timeout)

    def execute_ddl(self, sql: str) -> None:
        """Execute a DDL or DML statement (CREATE, INSERT, …).

        Unlike :meth:`execute_query`, this is an intentional escape hatch for
        trusted, developer-authored setup/migration SQL and is never subject
        to read-only enforcement.
        """
        with self.engine.begin() as conn:
            conn.execute(text(sql))

    def get_table_names(self) -> list[str]:
        """Return all table names in the database."""
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(self.engine)
        return inspector.get_table_names()

    def get_column_info(self, table_name: str) -> list[dict[str, Any]]:
        """Return column metadata for *table_name*.

        Each entry is a dict with keys ``name``, ``type``, ``nullable``,
        ``primary_key``.

        Raises:
            ValueError: If *table_name* is not present in the database.
        """
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(self.engine)
        # Validate table_name against the known list to prevent injection
        known_tables = inspector.get_table_names()
        if table_name not in known_tables:
            raise ValueError(
                f"Table '{table_name}' does not exist. Known tables: {known_tables}"
            )
        columns = inspector.get_columns(table_name)
        pk_constraint = inspector.get_pk_constraint(table_name)
        pk_cols = set(pk_constraint.get("constrained_columns", []))

        result: list[dict[str, Any]] = []
        for col in columns:
            result.append(
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "primary_key": col["name"] in pk_cols,
                }
            )
        return result

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        """Return foreign key metadata for *table_name*."""
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(self.engine)
        known_tables = inspector.get_table_names()
        if table_name not in known_tables:
            raise ValueError(
                f"Table '{table_name}' does not exist. Known tables: {known_tables}"
            )
        return inspector.get_foreign_keys(table_name)

    def get_sample_rows(self, table_name: str, n: int = 3) -> pd.DataFrame:
        """Return up to *n* sample rows from *table_name*.

        *table_name* is validated against the live schema to prevent injection.
        """
        known_tables = self.get_table_names()
        if table_name not in known_tables:
            raise ValueError(
                f"Table '{table_name}' does not exist. Known tables: {known_tables}"
            )
        # n is cast to int to prevent injection via numeric parameter
        limit = max(1, int(n))
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"  # noqa: S608
        return self.execute_query(sql)

    def test_connection(self) -> bool:
        """Return ``True`` if the database connection is alive."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
