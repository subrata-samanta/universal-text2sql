"""Tests for agent node functions (using mock LLM to avoid Groq API calls)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.agent.nodes import (
    execute_sql,
    generate_sql,
    reflect,
    select_schema,
    store_memory,
)
from universal_text2sql.agent.state import AgentState
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import SchemaDiscovery
from universal_text2sql.utils.demo_data import seed_demo_database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def connector() -> DatabaseConnector:
    conn = DatabaseConnector("sqlite:///:memory:")
    seed_demo_database(conn)
    return conn


@pytest.fixture(scope="module")
def schema(connector: DatabaseConnector):
    return SchemaDiscovery(connector).discover()


@pytest.fixture
def memory() -> QueryMemory:
    return QueryMemory(enabled=False)


def _make_mock_llm(response_text: str):
    """Return a RunnableLambda that always emits *response_text* as an AIMessage.

    Using ``RunnableLambda`` ensures the mock works correctly with LangChain's
    LCEL pipe operator (``prompt | llm``), where the *prompt's* ``__or__``
    method is invoked – not the LLM's.
    """
    return RunnableLambda(lambda _: AIMessage(content=response_text))


def _base_state() -> AgentState:
    return {
        "question": "How many customers are there?",
        "relevant_tables": [],
        "schema_context": "",
        "column_samples": "",
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSelectSchema:
    def test_returns_relevant_tables(self, schema, memory):
        state = _base_state()
        state["question"] = "How many customers are from the USA?"
        update = select_schema(state, schema=schema, memory=memory)
        assert "customers" in update["relevant_tables"]
        assert "schema_context" in update
        assert len(update["schema_context"]) > 0

    def test_column_samples_populated(self, schema, memory):
        state = _base_state()
        state["question"] = "Show all products"
        update = select_schema(state, schema=schema, memory=memory)
        # column_samples should be a non-empty string
        assert isinstance(update["column_samples"], str)

    def test_grounding_hints_empty_without_index(self, schema, memory):
        state = _base_state()
        state["question"] = "How many customers are from the USA?"
        update = select_schema(state, schema=schema, memory=memory)
        assert update["grounding_hints"] == ""

    def test_grounding_hints_populated_with_index(self, schema, connector, memory, tmp_path):
        from universal_text2sql.knowledge.grounding import ValueGroundingIndex

        grounding_index = ValueGroundingIndex.build(schema, connector, cache_dir=tmp_path)
        state = _base_state()
        state["question"] = "How many customers are from the USA?"
        update = select_schema(state, schema=schema, memory=memory, grounding_index=grounding_index)
        assert "USA" in update["grounding_hints"]


class TestExecuteSQL:
    def test_successful_execution(self, connector: DatabaseConnector):
        state = _base_state()
        state["generated_sql"] = "SELECT COUNT(*) AS cnt FROM customers"
        update = execute_sql(state, connector=connector)
        assert update["execution_error"] == ""
        assert update["execution_result"] is not None
        assert update["execution_result"].iloc[0]["cnt"] == 10

    def test_failed_execution_captures_error(self, connector: DatabaseConnector):
        state = _base_state()
        state["generated_sql"] = "SELECT * FROM nonexistent_table_xyz"
        update = execute_sql(state, connector=connector)
        assert update["execution_error"] != ""
        assert update["execution_result"] is None
        assert update["success"] is False


class TestGenerateSQL:
    def test_sql_extracted_from_response(self, schema, memory):
        sql = "SELECT COUNT(*) FROM customers"
        mock_llm = _make_mock_llm(f"```sql\n{sql}\n```")
        state = _base_state()
        state.update(select_schema(state, schema=schema, memory=memory))

        update = generate_sql(state, llm=mock_llm, db_type="SQLite")
        assert update["generated_sql"] == sql

    def test_sql_extracted_without_fences(self, schema, memory):
        sql = "SELECT * FROM products WHERE price > 100"
        mock_llm = _make_mock_llm(sql)
        state = _base_state()
        state.update(select_schema(state, schema=schema, memory=memory))

        update = generate_sql(state, llm=mock_llm, db_type="SQLite")
        assert update["generated_sql"] == sql


class TestReflect:
    def test_reflect_increments_retry_count(self, schema, memory):
        corrected = "SELECT COUNT(*) FROM customers WHERE country = 'USA'"
        mock_llm = _make_mock_llm(corrected)
        state = _base_state()
        state.update(select_schema(state, schema=schema, memory=memory))
        state["generated_sql"] = "SELECT * FROM customer"  # wrong table
        state["execution_error"] = "no such table: customer"
        state["retry_count"] = 0

        update = reflect(state, llm=mock_llm, db_type="SQLite")
        assert update["retry_count"] == 1
        assert update["generated_sql"] == corrected


class TestStoreMemory:
    def test_successful_query_stored(self, tmp_path):
        mem = QueryMemory(persist_path=tmp_path / "test_mem.json", enabled=True)
        state = _base_state()
        state["success"] = True
        state["generated_sql"] = "SELECT COUNT(*) FROM customers"
        state["question"] = "How many customers?"
        state["retry_count"] = 0

        store_memory(state, memory=mem)
        results = mem.get_similar("How many customers?")
        assert any(r["sql"] == "SELECT COUNT(*) FROM customers" for r in results)

    def test_failed_query_not_stored(self, tmp_path):
        mem = QueryMemory(persist_path=tmp_path / "test_mem2.json", enabled=True)
        state = _base_state()
        state["success"] = False
        state["generated_sql"] = "SELECT * FROM bad_table"

        store_memory(state, memory=mem)
        assert mem._entries == []
