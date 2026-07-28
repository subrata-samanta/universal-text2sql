"""SQL statement safety guardrails.

The agent executes LLM-generated SQL against a live database with no human
in the loop. Before this module existed, :meth:`DatabaseConnector.execute_query`
ran *any* SQL string unconditionally -- ``SELECT``, ``INSERT``, ``DROP``, all
identical (this gap was already documented, but not enforced, in
``SECURITY.md``). This module makes "only read-only SQL runs unless the
caller explicitly opts in" an enforced default rather than a suggestion.

Two layers, in increasing order of robustness:

1. :func:`classify_statement` -- a zero-dependency regex baseline. Always
   available, always runs. Splits on ``;`` to reject multi-statement
   payloads (the classic "harmless SELECT; DROP TABLE ...;" injection
   shape) and classifies by the leading keyword.
2. :func:`classify_statement_ast` -- a stronger, *optional* AST-based check
   via `sqlglot <https://github.com/tobymao/sqlglot>`_, lazily imported.
   Correctly handles cases the regex baseline can't, e.g.
   ``WITH x AS (SELECT ...) DELETE FROM x`` (a CTE that *looks* read-only
   from its first keyword but ends in a write). Returns ``None`` (falling
   back to the regex baseline) when sqlglot isn't installed or can't parse
   the statement -- it never raises and never silently downgrades safety.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from enum import Enum
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StatementType(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    UNKNOWN = "UNKNOWN"


class SQLSafetyError(Exception):
    """Raised when a SQL statement violates the configured safety policy."""


# ---------------------------------------------------------------------------
# Regex baseline (zero dependency, always available)
# ---------------------------------------------------------------------------

_WRITE_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA", "REPLACE", "MERGE",
    "VACUUM", "REINDEX", "COPY",
}
_READ_KEYWORDS = {"SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "DESC"}

_LEADING_COMMENT_RE = re.compile(r"^\s*(--[^\n]*\n\s*|/\*.*?\*/\s*)*", re.DOTALL)
_FIRST_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _strip_leading_comments(sql: str) -> str:
    return _LEADING_COMMENT_RE.sub("", sql).strip()


def _split_statements(sql: str) -> list[str]:
    """Split on ``;`` outside of single/double-quoted string literals."""
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    for ch in sql:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == ";" and not in_single and not in_double:
            statements.append("".join(current))
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return [s.strip() for s in statements if s.strip()]


def classify_statement(sql: str) -> StatementType:
    """Zero-dependency regex-based statement classifier.

    A multi-statement payload is always classified ``WRITE`` (blocked by
    default), regardless of what the individual statements are -- a
    trailing statement smuggled in after a harmless ``SELECT`` is exactly
    the shape a prompt-injected or hallucinated write would take.
    """
    statements = _split_statements(sql)
    if not statements:
        return StatementType.UNKNOWN
    if len(statements) > 1:
        logger.warning(
            "Multi-statement SQL classified as WRITE (unsafe): %d statements", len(statements)
        )
        return StatementType.WRITE

    stmt = _strip_leading_comments(statements[0])
    match = _FIRST_WORD_RE.match(stmt)
    if not match:
        return StatementType.UNKNOWN

    keyword = match.group(0).upper()
    if keyword in _READ_KEYWORDS:
        return StatementType.READ
    if keyword in _WRITE_KEYWORDS:
        return StatementType.WRITE
    return StatementType.UNKNOWN


# ---------------------------------------------------------------------------
# Optional AST-based classification (sqlglot, lazily imported)
# ---------------------------------------------------------------------------

_SQLGLOT_DIALECT_MAP = {
    "SQLite": "sqlite",
    "PostgreSQL": "postgres",
    "MySQL": "mysql",
}

_AST_WRITE_TYPE_NAMES = (
    "Insert", "Update", "Delete", "Drop", "Alter", "Create",
    "TruncateTable", "Grant", "Merge", "Pragma", "Copy",
)
_AST_READ_TYPE_NAMES = ("Select", "Union", "Except", "Intersect")


def classify_statement_ast(sql: str, dialect: str = "sqlite") -> StatementType | None:
    """Stronger AST-based classification via the optional ``sqlglot`` package.

    Returns ``None`` (triggering the :func:`classify_statement` fallback)
    when sqlglot isn't installed, fails to parse the statement, or maps it
    to a node type this function doesn't recognise (e.g. sqlglot's generic
    ``Command`` fallback for syntax it can't fully model) -- it never
    raises and never guesses.
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return None

    # sqlglot logs a warning to its own logger for syntax it can't fully
    # model (falling back to a generic Command node) -- we already handle
    # that case explicitly below, so silence the duplicate noise.
    logging.getLogger("sqlglot").setLevel(logging.ERROR)

    sqlglot_dialect = _SQLGLOT_DIALECT_MAP.get(dialect, dialect.lower())
    try:
        expressions = [e for e in sqlglot.parse(sql, read=sqlglot_dialect) if e is not None]
    except Exception as exc:  # pragma: no cover - defensive, sqlglot internals vary
        logger.debug("sqlglot failed to parse SQL (%s); falling back to regex classifier", exc)
        return None

    if not expressions:
        return StatementType.UNKNOWN
    if len(expressions) > 1:
        logger.warning(
            "Multi-statement SQL classified as WRITE (unsafe): %d statements", len(expressions)
        )
        return StatementType.WRITE

    node = expressions[0]
    write_types = tuple(t for t in (getattr(exp, n, None) for n in _AST_WRITE_TYPE_NAMES) if t)
    read_types = tuple(t for t in (getattr(exp, n, None) for n in _AST_READ_TYPE_NAMES) if t)

    if isinstance(node, write_types):
        return StatementType.WRITE
    if isinstance(node, read_types):
        return StatementType.READ
    return None


# ---------------------------------------------------------------------------
# Public enforcement API
# ---------------------------------------------------------------------------


def enforce_read_only(
    sql: str,
    dialect: str = "sqlite",
    *,
    allow_writes: bool = False,
    use_ast: bool = True,
) -> None:
    """Raise :class:`SQLSafetyError` unless *sql* is a single read-only statement.

    Fails closed: anything that isn't confidently classified ``READ``
    (including ``UNKNOWN``) is blocked, matching a least-privilege default.
    """
    if allow_writes:
        return

    stmt_type = classify_statement_ast(sql, dialect) if use_ast else None
    if stmt_type is None:
        stmt_type = classify_statement(sql)

    if stmt_type != StatementType.READ:
        raise SQLSafetyError(
            f"Refusing to execute a non-read-only statement (classified as "
            f"{stmt_type.value}). Pass allow_writes=True on execute_query(), or "
            f"construct DatabaseConnector(read_only=False), to permit this."
        )


_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)
_FETCH_RE = re.compile(r"\bFETCH\s+(FIRST|NEXT)\b", re.IGNORECASE)
_TOP_RE = re.compile(r"^\s*SELECT\s+TOP\s+\d+", re.IGNORECASE)


def inject_limit(sql: str, limit: int, dialect: str = "sqlite") -> str:
    """Append ``LIMIT {limit}`` to *sql* unless it already caps rows.

    *limit* is always coerced through :func:`int`, so this is safe to call
    with any caller-supplied value -- never string-interpolated as-is.
    A no-op when the query already has a top-level ``LIMIT``/``FETCH``/``TOP``
    clause (a redundant tighter limit would be harmless anyway, but detecting
    the common case avoids visibly odd double-``LIMIT`` SQL in traces/logs).
    """
    limit = int(limit)
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    stripped = sql.rstrip().rstrip(";").rstrip()
    if _LIMIT_RE.search(stripped) or _FETCH_RE.search(stripped) or _TOP_RE.search(stripped):
        return sql
    return f"{stripped}\nLIMIT {limit}"


def run_with_timeout(fn: Callable[[], T], timeout_seconds: float | None) -> T:
    """Best-effort wall-clock timeout around *fn*.

    This bounds how long the caller waits for a result -- it does **not**
    guarantee the underlying query is actually cancelled on every backend
    (e.g. SQLite has no portable way to interrupt an in-flight query from
    another thread). Treat it as a safety net against a hung client, not a
    substitute for server-side statement timeouts where the backend
    supports them (see ``DatabaseConnector``'s Postgres ``statement_timeout``
    handling).
    """
    if timeout_seconds is None:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            raise SQLSafetyError(f"Query exceeded timeout of {timeout_seconds}s") from exc
