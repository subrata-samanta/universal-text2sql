"""Shared result-comparison utilities.

Used by :mod:`universal_text2sql.agent.nodes` (self-consistency candidate
voting) and the ``eval`` harness (execution-accuracy scoring) — both need
the same thing: "do two query results represent the same answer, ignoring
row/column order."
"""

from __future__ import annotations

import pandas as pd


def result_signature(df: pd.DataFrame | None) -> str:
    """Order-independent signature of a query result, for equality grouping."""
    if df is None:
        return "<empty>"
    try:
        rows = sorted(tuple(str(v) for v in row) for row in df.itertuples(index=False))
        cols = sorted(str(c) for c in df.columns)
        return f"{len(rows)}::{cols}::" + "|".join(",".join(r) for r in rows)
    except Exception:
        return str(df)


def results_match(actual: pd.DataFrame | None, expected: pd.DataFrame | None) -> bool:
    """Whether *actual* and *expected* represent the same result set."""
    return result_signature(actual) == result_signature(expected)
