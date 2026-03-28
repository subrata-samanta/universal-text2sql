"""Tests for LangGraph graph construction and routing logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from universal_text2sql.agent.graph import (
    _should_reflect_on_validation,
    _should_reflect_or_continue,
)
from universal_text2sql.agent.state import AgentState


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "question": "Test question",
        "relevant_tables": [],
        "schema_context": "",
        "column_samples": "",
        "few_shot_examples": [],
        "generated_sql": "SELECT 1",
        "execution_result": None,
        "execution_error": "",
        "validation_verdict": "",
        "final_answer": "",
        "retry_count": 0,
        "max_retries": 3,
        "messages": [],
        "success": False,
    }
    base.update(overrides)
    return base


class TestRoutingLogic:
    # _should_reflect_or_continue
    def test_no_error_routes_to_validate(self):
        state = _make_state(execution_error="")
        assert _should_reflect_or_continue(state) == "validate_result"

    def test_error_routes_to_reflect(self):
        state = _make_state(execution_error="table not found", retry_count=0)
        assert _should_reflect_or_continue(state) == "reflect"

    def test_max_retries_reached_routes_to_format(self):
        state = _make_state(execution_error="still broken", retry_count=3, max_retries=3)
        assert _should_reflect_or_continue(state) == "format_answer"

    # _should_reflect_on_validation
    def test_valid_verdict_routes_to_format(self):
        state = _make_state(validation_verdict="VALID")
        assert _should_reflect_on_validation(state) == "format_answer"

    def test_invalid_verdict_routes_to_reflect(self):
        state = _make_state(
            validation_verdict="INVALID: result is empty", retry_count=0
        )
        assert _should_reflect_on_validation(state) == "reflect"

    def test_invalid_max_retries_routes_to_format(self):
        state = _make_state(
            validation_verdict="INVALID: still wrong", retry_count=3, max_retries=3
        )
        assert _should_reflect_on_validation(state) == "format_answer"


class TestBuildGraph:
    def test_graph_compiled_without_errors(self):
        from unittest.mock import MagicMock, patch

        from universal_text2sql.agent.graph import build_graph
        from universal_text2sql.agent.memory import QueryMemory
        from universal_text2sql.database.connector import DatabaseConnector
        from universal_text2sql.database.schema import DatabaseSchema, SchemaDiscovery
        from universal_text2sql.utils.demo_data import seed_demo_database

        conn = DatabaseConnector("sqlite:///:memory:")
        seed_demo_database(conn)
        schema = SchemaDiscovery(conn).discover()
        memory = QueryMemory(enabled=False)

        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock(return_value=mock_llm)

        # Should not raise
        graph = build_graph(
            connector=conn,
            schema=schema,
            memory=memory,
            llm=mock_llm,
        )
        assert graph is not None
