"""Tests for query decomposition (DIN-SQL-style, agent/nodes.py::decompose_sql)."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from universal_text2sql.agent.graph import _should_decompose, build_graph
from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.agent.nodes import _compose_cte_query, decompose_sql
from universal_text2sql.agent.state import AgentState
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import SchemaDiscovery
from universal_text2sql.utils.demo_data import seed_demo_database


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "question": "Which customer has spent the most money overall?",
        "relevant_tables": [],
        "schema_context": "CREATE TABLE orders (customer_id INTEGER, total_amount REAL);",
        "column_samples": "",
        "kg_context": "",
        "business_glossary": "",
        "grounding_hints": "",
        "complexity": "COMPLEX",
        "sql_candidates": [],
        "sub_questions": [],
        "decomposition_steps": [],
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


class TestComposeCteQuery:
    def test_single_step_returns_bare_fragment(self):
        steps = [{"sub_question": "q", "sql_fragment": "SELECT 1"}]
        assert _compose_cte_query(steps) == "SELECT 1"

    def test_multi_step_composes_with_cte(self):
        steps = [
            {"sub_question": "a", "sql_fragment": "SELECT 1 AS x"},
            {"sub_question": "b", "sql_fragment": "SELECT 2 AS y"},
            {"sub_question": "c", "sql_fragment": "SELECT * FROM step_1, step_2"},
        ]
        composed = _compose_cte_query(steps)
        assert composed.startswith("WITH step_1 AS (")
        assert "step_2 AS (" in composed
        assert composed.strip().endswith("SELECT * FROM step_1, step_2")


class TestShouldDecompose:
    def test_disabled_always_routes_to_generate_sql(self):
        state = _base_state(complexity="COMPLEX")
        assert _should_decompose(state, enable_query_decomposition=False) == "generate_sql"

    def test_enabled_but_not_complex_routes_to_generate_sql(self):
        state = _base_state(complexity="MODERATE")
        assert _should_decompose(state, enable_query_decomposition=True) == "generate_sql"

    def test_enabled_and_complex_routes_to_decompose(self):
        state = _base_state(complexity="COMPLEX")
        assert _should_decompose(state, enable_query_decomposition=True) == "decompose_sql"


class TestDecomposeSql:
    def test_happy_path_composes_and_records_steps(self):
        responses = iter(
            [
                json.dumps(["compute per-customer spend", "return the top spender"]),
                "SELECT customer_id, SUM(total_amount) AS total FROM orders GROUP BY customer_id",
                "SELECT customer_id FROM step_1 ORDER BY total DESC LIMIT 1",
            ]
        )

        def _llm(_input):
            return AIMessage(content=next(responses))

        update = decompose_sql(_base_state(), llm=RunnableLambda(_llm), db_type="SQLite")

        assert update["sub_questions"] == [
            "compute per-customer spend",
            "return the top spender",
        ]
        assert len(update["decomposition_steps"]) == 2
        assert update["generated_sql"].startswith("WITH step_1 AS (")
        assert update["sql_candidates"] == [update["generated_sql"]]

    def test_malformed_plan_falls_back_to_single_shot(self):
        responses = iter(
            [
                "not json at all",  # decomposition plan fails to parse
                "SELECT customer_id FROM orders ORDER BY total_amount DESC LIMIT 1",  # fallback
            ]
        )

        def _llm(_input):
            return AIMessage(content=next(responses))

        update = decompose_sql(_base_state(), llm=RunnableLambda(_llm), db_type="SQLite")

        assert "sub_questions" not in update  # fallback delegates to generate_sql, which doesn't set it
        assert update["generated_sql"] == "SELECT customer_id FROM orders ORDER BY total_amount DESC LIMIT 1"
        assert update["sql_candidates"] == [update["generated_sql"]]

    def test_empty_plan_list_falls_back_to_single_shot(self):
        responses = iter(
            [
                "[]",  # valid JSON but no sub-questions
                "SELECT 1",  # fallback single-shot
            ]
        )

        def _llm(_input):
            return AIMessage(content=next(responses))

        update = decompose_sql(_base_state(), llm=RunnableLambda(_llm), db_type="SQLite")
        assert update["generated_sql"] == "SELECT 1"

    def test_empty_fragment_falls_back_to_single_shot(self):
        responses = iter(
            [
                json.dumps(["step one", "step two"]),
                "",  # empty fragment for step one -> triggers fallback
                "SELECT 1",  # fallback single-shot generate_sql call
            ]
        )

        def _llm(_input):
            return AIMessage(content=next(responses))

        update = decompose_sql(_base_state(), llm=RunnableLambda(_llm), db_type="SQLite")
        assert update["generated_sql"] == "SELECT 1"


class TestBuildGraphWithDecomposition:
    def test_compiles_with_decomposition_enabled(self):
        conn = DatabaseConnector("sqlite:///:memory:")
        seed_demo_database(conn)
        schema = SchemaDiscovery(conn).discover()
        memory = QueryMemory(enabled=False)

        mock_llm = RunnableLambda(lambda _: AIMessage(content="SELECT 1"))

        graph = build_graph(
            connector=conn,
            schema=schema,
            memory=memory,
            llm=mock_llm,
            enable_query_decomposition=True,
        )
        assert graph is not None
