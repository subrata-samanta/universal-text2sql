"""Tests for the eval/ golden-question harness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eval.cli import main as cli_main
from eval.metrics import EvalSummary, QuestionResult, dataframe_from_expected, execution_accuracy
from eval.runner import DEFAULT_DATASET, load_golden_set, mock_generator, run_golden_set


@pytest.fixture()
def small_dataset(tmp_path: Path) -> list[dict]:
    records = [
        {
            "id": "t1",
            "question": "how many rows",
            "mock_sql": "SELECT 1 AS n",
            "tags": ["simple"],
            "expected_result": {"columns": ["n"], "rows": [[1]]},
        },
        {
            "id": "t2",
            "question": "give me nothing useful",
            "mock_sql": "SELECT 'wrong' AS n",
            "tags": ["simple"],
            "expected_result": {"columns": ["n"], "rows": [["right"]]},
        },
    ]
    path = tmp_path / "small.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return records


class TestMetrics:
    def test_dataframe_from_expected(self):
        df = dataframe_from_expected({"columns": ["a", "b"], "rows": [[1, "x"]]})
        assert list(df.columns) == ["a", "b"]
        assert df.iloc[0]["a"] == 1

    def test_execution_accuracy_match(self):
        actual = pd.DataFrame({"cnt": [10]})
        assert execution_accuracy(actual, {"columns": ["cnt"], "rows": [[10]]}) is True

    def test_execution_accuracy_order_independent(self):
        actual = pd.DataFrame({"a": [2, 1]})
        assert execution_accuracy(actual, {"columns": ["a"], "rows": [[1], [2]]}) is True

    def test_execution_accuracy_mismatch(self):
        actual = pd.DataFrame({"cnt": [5]})
        assert execution_accuracy(actual, {"columns": ["cnt"], "rows": [[10]]}) is False

    def test_eval_summary_empty(self):
        summary = EvalSummary()
        assert summary.total == 0
        assert summary.execution_accuracy == 0.0
        assert summary.execution_success_rate == 0.0

    def test_eval_summary_aggregates(self):
        summary = EvalSummary(
            results=[
                QuestionResult(id="a", question="q", executed=True, correct=True),
                QuestionResult(id="b", question="q", executed=True, correct=False),
                QuestionResult(id="c", question="q", executed=False, correct=False),
            ]
        )
        assert summary.total == 3
        assert summary.execution_accuracy == pytest.approx(1 / 3)
        assert summary.execution_success_rate == pytest.approx(2 / 3)


class TestRunner:
    def test_load_golden_set_default(self):
        dataset = load_golden_set()
        assert len(dataset) >= 10
        assert all({"id", "question", "mock_sql", "expected_result"} <= d.keys() for d in dataset)

    def test_load_golden_set_filters_by_tag(self):
        dataset = load_golden_set(tag="complex")
        assert len(dataset) >= 1
        assert all("complex" in d["tags"] for d in dataset)

    def test_load_golden_set_from_custom_path(self, small_dataset, tmp_path: Path):
        dataset = load_golden_set(tmp_path / "small.jsonl")
        assert len(dataset) == 2

    def test_mock_generator_returns_canned_sql(self, small_dataset):
        gen = mock_generator(small_dataset)
        assert gen("how many rows") == "SELECT 1 AS n"

    def test_mock_generator_unknown_question_raises(self, small_dataset):
        gen = mock_generator(small_dataset)
        with pytest.raises(ValueError):
            gen("an unregistered question")

    def test_run_golden_set_scores_correct_and_incorrect(self, small_dataset):
        def execute_sql(sql: str) -> pd.DataFrame:
            if "wrong" in sql:
                return pd.DataFrame({"n": ["wrong"]})
            return pd.DataFrame({"n": [1]})

        summary = run_golden_set(small_dataset, mock_generator(small_dataset), execute_sql)
        assert summary.total == 2
        by_id = {r.id: r for r in summary.results}
        assert by_id["t1"].correct is True
        assert by_id["t2"].correct is False

    def test_run_golden_set_captures_generation_failure(self, small_dataset):
        def failing_generator(question: str) -> str:
            raise RuntimeError("boom")

        def execute_sql(sql: str) -> pd.DataFrame:
            return pd.DataFrame()

        summary = run_golden_set(small_dataset, failing_generator, execute_sql)
        assert all(not r.executed and r.execution_error for r in summary.results)

    def test_run_golden_set_captures_execution_failure(self, small_dataset):
        def execute_sql(sql: str) -> pd.DataFrame:
            raise RuntimeError("no such table")

        summary = run_golden_set(small_dataset, mock_generator(small_dataset), execute_sql)
        assert all(not r.executed and not r.correct for r in summary.results)


class TestEndToEndMockAgainstDemoDb:
    """Regression guard: if demo_data.py's seed data ever changes, this fails."""

    def test_full_demo_golden_set_passes_via_mock_cli(self):
        exit_code = cli_main(["--mock"])
        assert exit_code == 0

    def test_fail_under_threshold_triggers_nonzero_exit(self):
        exit_code = cli_main(["--mock", "--fail-under", "1.5"])
        assert exit_code == 1

    def test_missing_dataset_returns_error_code(self, tmp_path: Path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        exit_code = cli_main(["--mock", "--dataset", str(empty)])
        assert exit_code == 2


def test_default_dataset_path_exists():
    assert DEFAULT_DATASET.exists()
