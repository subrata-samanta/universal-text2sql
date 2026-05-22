"""Universal database connector supporting SQLite, PostgreSQL, and MySQL."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Universal database connector that works with any SQLAlchemy-supported database."""

    def __init__(self, connection_url: str) -> None:
        """Initialise the connector.

        Args:
            connection_url: SQLAlchemy connection URL, e.g.
                - ``sqlite:///./mydb.sqlite3``
                - ``postgresql://user:pw@localhost/db``
                - ``mysql+pymysql://user:pw@localhost/db``
        """
        self.connection_url = connection_url
        self.db_type = self._detect_db_type(connection_url)
        self.engine: Engine = create_engine(
            connection_url,
            echo=False,
            future=True,
            # Pool settings comfortable for all supported backends
            pool_pre_ping=True,
        )
        logger.info("Connected to %s database", self.db_type)

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

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: The SQL statement to execute.

        Returns:
            A :class:`pandas.DataFrame` with the query results.

        Raises:
            Exception: Re-raises any SQLAlchemy/database exception so that the
                agent's reflection loop can inspect the error message.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = list(result.keys())
        return pd.DataFrame(rows, columns=columns)

    def execute_ddl(self, sql: str) -> None:
        """Execute a DDL or DML statement (CREATE, INSERT, …)."""
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
