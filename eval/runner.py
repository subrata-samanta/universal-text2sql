"""Golden-question evaluation loop.

Two run modes, sharing the same scoring logic (:mod:`eval.metrics`):

- :func:`run_golden_set` — pluggable ``question -> SQL`` generator plus a
  ``sql -> DataFrame`` executor. Used for ``--mock`` mode (a canned
  per-question SQL lookup, no LLM/API key needed) but works with any
  generator function.
- :func:`run_golden_set_via_agent` — drives a real
  :class:`~universal_text2sql.bootstrap.AgentContext`, i.e. the actual
  agent graph (schema linking, self-reflection, etc.), for ``--live`` mode.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from eval.metrics import EvalSummary, QuestionResult, execution_accuracy

if TYPE_CHECKING:
    from universal_text2sql.bootstrap import AgentContext

DEFAULT_DATASET = Path(__file__).parent / "golden" / "demo_db.jsonl"

GeneratorFn = Callable[[str], str]
ExecutorFn = Callable[[str], pd.DataFrame]


def load_golden_set(path: Path | str = DEFAULT_DATASET, tag: str | None = None) -> list[dict[str, Any]]:
    """Load a JSONL golden dataset, optionally filtered to a single tag."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if tag and tag not in record.get("tags", []):
                continue
            records.append(record)
    return records


def mock_generator(dataset: list[dict[str, Any]]) -> GeneratorFn:
    """Build a ``question -> SQL`` function from each record's canned ``mock_sql``."""
    lookup = {record["question"]: record["mock_sql"] for record in dataset}

    def _generate(question: str) -> str:
        try:
            return lookup[question]
        except KeyError:
            raise ValueError(f"No mock SQL registered for question: {question!r}") from None

    return _generate


def run_golden_set(
    dataset: list[dict[str, Any]],
    generate_sql: GeneratorFn,
    execute_sql: ExecutorFn,
) -> EvalSummary:
    """Run every question in *dataset* through *generate_sql* then *execute_sql*."""
    summary = EvalSummary()
    for record in dataset:
        result = QuestionResult(
            id=record["id"], question=record["question"], tags=record.get("tags", [])
        )

        try:
            result.generated_sql = generate_sql(record["question"])
        except Exception as exc:
            result.execution_error = f"generation failed: {exc}"
            summary.results.append(result)
            continue

        try:
            df = execute_sql(result.generated_sql)
            result.executed = True
        except Exception as exc:
            result.execution_error = str(exc)
            summary.results.append(result)
            continue

        result.correct = execution_accuracy(df, record["expected_result"])
        summary.results.append(result)

    return summary


def run_golden_set_via_agent(dataset: list[dict[str, Any]], ctx: AgentContext) -> EvalSummary:
    """Run every question through a real :class:`AgentContext` (``--live`` mode)."""
    summary = EvalSummary()
    for record in dataset:
        result = QuestionResult(
            id=record["id"], question=record["question"], tags=record.get("tags", [])
        )

        try:
            agent_result = ctx.ask(record["question"])
        except Exception as exc:
            result.execution_error = f"agent error: {exc}"
            summary.results.append(result)
            continue

        result.generated_sql = agent_result.get("generated_sql", "")
        error = agent_result.get("execution_error")
        df = agent_result.get("execution_result")
        result.executed = df is not None and not error
        if error:
            result.execution_error = error
        if result.executed:
            result.correct = execution_accuracy(df, record["expected_result"])
        summary.results.append(result)

    return summary
