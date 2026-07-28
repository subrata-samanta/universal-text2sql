"""Tests for complexity classification and self-consistency (multi-candidate voting)."""

from __future__ import annotations

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from universal_text2sql.agent.nodes import classify_complexity, generate_sql, select_best_candidate
from universal_text2sql.agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "question": "How many customers are there?",
        "relevant_tables": [],
        "schema_context": "CREATE TABLE customers (customer_id INTEGER);",
        "column_samples": "",
        "kg_context": "",
        "business_glossary": "",
        "complexity": "",
        "sql_candidates": [],
        "few_shot_examples": [],
        "generated_sql": "",
        "execution_result": None,
        "execution_error": "",
        "validation_verdict": "",
        "final_answer": "",
        "retry_count": 0,
        "max_retries": 3,
        "messages": [],
        "success": False,
    }
    state.update(overrides)
    return state


class TestClassifyComplexity:
    def test_classifies_simple(self):
        llm = RunnableLambda(lambda _: AIMessage(content="SIMPLE"))
        update = classify_complexity(_base_state(), llm=llm)
        assert update["complexity"] == "SIMPLE"

    def test_classifies_complex_with_extra_text(self):
        llm = RunnableLambda(lambda _: AIMessage(content="Classification: COMPLEX (multiple joins)"))
        update = classify_complexity(_base_state(), llm=llm)
        assert update["complexity"] == "COMPLEX"

    def test_unrecognised_label_defaults_to_moderate(self):
        llm = RunnableLambda(lambda _: AIMessage(content="not sure"))
        update = classify_complexity(_base_state(), llm=llm)
        assert update["complexity"] == "MODERATE"


class TestGenerateSqlSelfConsistency:
    def test_single_sample_by_default(self):
        llm = RunnableLambda(lambda _: AIMessage(content="SELECT COUNT(*) FROM customers"))
        update = generate_sql(_base_state(), llm=llm, db_type="SQLite")
        assert update["sql_candidates"] == ["SELECT COUNT(*) FROM customers"]

    def test_simple_complexity_forces_single_sample(self):
        calls = {"n": 0}

        def _llm(_input):
            calls["n"] += 1
            return AIMessage(content=f"SELECT {calls['n']}")

        update = generate_sql(
            _base_state(complexity="SIMPLE"),
            llm=RunnableLambda(_llm),
            db_type="SQLite",
            self_consistency_samples=5,
        )
        assert calls["n"] == 1
        assert len(update["sql_candidates"]) == 1

    def test_moderate_complexity_samples_multiple_candidates(self):
        calls = {"n": 0}

        def _llm(_input):
            calls["n"] += 1
            return AIMessage(content=f"SELECT {calls['n']} FROM customers")

        update = generate_sql(
            _base_state(complexity="MODERATE"),
            llm=RunnableLambda(_llm),
            db_type="SQLite",
            self_consistency_samples=3,
        )
        assert calls["n"] == 3
        assert len(update["sql_candidates"]) == 3
        assert update["generated_sql"] == update["sql_candidates"][0]

    def test_duplicate_candidates_are_deduplicated(self):
        llm = RunnableLambda(lambda _: AIMessage(content="SELECT 1"))
        update = generate_sql(
            _base_state(complexity="COMPLEX"),
            llm=llm,
            db_type="SQLite",
            self_consistency_samples=4,
        )
        assert update["sql_candidates"] == ["SELECT 1"]


class _FakeConnector:
    """Minimal stand-in exposing only what select_best_candidate needs."""

    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses

    def execute_query(self, sql: str) -> pd.DataFrame:
        result = self._responses[sql]
        if isinstance(result, Exception):
            raise result
        return result


class TestSelectBestCandidate:
    def test_single_candidate_is_noop(self):
        state = _base_state(sql_candidates=["SELECT 1"])
        update = select_best_candidate(state, connector=_FakeConnector({}))
        assert update == {}

    def test_majority_result_wins(self):
        df_a = pd.DataFrame({"cnt": [10]})
        df_b = pd.DataFrame({"cnt": [10]})
        df_c = pd.DataFrame({"cnt": [99]})
        connector = _FakeConnector(
            {
                "SELECT A": df_a,
                "SELECT B": df_b,
                "SELECT C": df_c,
            }
        )
        state = _base_state(sql_candidates=["SELECT A", "SELECT B", "SELECT C"])
        update = select_best_candidate(state, connector=connector)
        assert update["generated_sql"] in ("SELECT A", "SELECT B")

    def test_falls_back_to_first_error_when_all_fail(self):
        connector = _FakeConnector(
            {
                "SELECT BAD1": Exception("no such table"),
                "SELECT BAD2": Exception("syntax error"),
            }
        )
        state = _base_state(sql_candidates=["SELECT BAD1", "SELECT BAD2"])
        update = select_best_candidate(state, connector=connector)
        assert update["generated_sql"] == "SELECT BAD1"

    def test_failed_candidates_excluded_from_voting(self):
        df_ok = pd.DataFrame({"cnt": [5]})
        connector = _FakeConnector(
            {
                "SELECT OK": df_ok,
                "SELECT BAD": Exception("boom"),
            }
        )
        state = _base_state(sql_candidates=["SELECT OK", "SELECT BAD"])
        update = select_best_candidate(state, connector=connector)
        assert update["generated_sql"] == "SELECT OK"
