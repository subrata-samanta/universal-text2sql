"""Evaluation CLI: ``python -m eval.cli``.

Zero-API-key mode (``--mock``, the default) is the CI-required path: it
scores the harness's plumbing and metrics using a canned per-question SQL
lookup, no LLM call involved. ``--live`` mode drives the real agent (needs
an LLM configured, e.g. ``GROQ_API_KEY``) for actual benchmarking — this is
intentionally never wired into required CI, to keep the test suite free of
API-key/network dependencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.metrics import EvalSummary
from eval.runner import (
    DEFAULT_DATASET,
    load_golden_set,
    mock_generator,
    run_golden_set,
    run_golden_set_via_agent,
)
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.utils.demo_data import seed_demo_database


def _build_demo_connector() -> DatabaseConnector:
    connector = DatabaseConnector("sqlite:///:memory:", read_only=False)
    seed_demo_database(connector)
    return connector


def _print_summary(summary: EvalSummary, verbose: bool) -> None:
    for result in summary.results:
        if result.execution_error:
            status = "ERROR"
        elif result.correct:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"[{status}] {result.id}: {result.question}")
        if verbose or status != "PASS":
            if result.generated_sql:
                print(f"    SQL: {result.generated_sql}")
            if result.execution_error:
                print(f"    Error: {result.execution_error}")

    correct = sum(1 for r in summary.results if r.correct)
    executed = sum(1 for r in summary.results if r.executed)
    print()
    print(f"Execution accuracy:      {summary.execution_accuracy:.1%} ({correct}/{summary.total})")
    print(f"Execution success rate:  {summary.execution_success_rate:.1%} ({executed}/{summary.total})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the text-to-SQL agent against a golden question set."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        help="Use canned SQL per question -- no LLM/API key needed (default).",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Run the real bootstrap()ped agent -- requires an LLM configured (e.g. GROQ_API_KEY).",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Path to a golden JSONL dataset."
    )
    parser.add_argument("--tag", default=None, help="Only run questions with this tag.")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit non-zero if execution accuracy is below this threshold (0.0-1.0).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print generated SQL for every question, not just failures."
    )
    args = parser.parse_args(argv)

    dataset = load_golden_set(args.dataset, tag=args.tag)
    if not dataset:
        print(f"No golden questions found in {args.dataset} (tag={args.tag!r}).", file=sys.stderr)
        return 2

    if args.live:
        from universal_text2sql.bootstrap import bootstrap

        ctx = bootstrap(database_url="sqlite:///:memory:")
        summary = run_golden_set_via_agent(dataset, ctx)
    else:
        connector = _build_demo_connector()
        summary = run_golden_set(dataset, mock_generator(dataset), connector.execute_query)

    _print_summary(summary, args.verbose)

    if args.fail_under is not None and summary.execution_accuracy < args.fail_under:
        print(
            f"\nFAILED: execution accuracy {summary.execution_accuracy:.1%} "
            f"is below --fail-under {args.fail_under:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
