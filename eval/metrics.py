"""Execution-accuracy metrics for the evaluation harness.

Deliberately does **not** score exact SQL text as a pass/fail gate — two
semantically identical queries can differ arbitrarily in formatting, alias
names, or join order. The primary metric is execution accuracy: does
running the generated SQL produce the same result set as the golden
``expected_result``, compared order-independently (reusing
:func:`universal_text2sql.agent.scoring.result_signature`, the same logic
the agent's self-consistency voting uses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from universal_text2sql.agent.scoring import result_signature


def dataframe_from_expected(expected: dict[str, Any]) -> pd.DataFrame:
    """Build a DataFrame from a golden record's ``expected_result`` block."""
    return pd.DataFrame(expected.get("rows", []), columns=expected.get("columns", []))


def execution_accuracy(actual: pd.DataFrame | None, expected: dict[str, Any]) -> bool:
    """Whether *actual* matches the golden *expected* result, order-independently."""
    return result_signature(actual) == result_signature(dataframe_from_expected(expected))


@dataclass
class QuestionResult:
    """Outcome of running one golden question."""

    id: str
    question: str
    tags: list[str] = field(default_factory=list)
    generated_sql: str = ""
    executed: bool = False
    execution_error: str = ""
    correct: bool = False


@dataclass
class EvalSummary:
    """Aggregate results across a full golden-set run."""

    results: list[QuestionResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def execution_success_rate(self) -> float:
        """Fraction of questions whose SQL ran without error (correctness not required)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.executed) / self.total

    @property
    def execution_accuracy(self) -> float:
        """Fraction of questions whose result matched the golden expectation."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / self.total
