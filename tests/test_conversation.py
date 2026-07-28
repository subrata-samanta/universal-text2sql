"""Tests for multi-turn conversational follow-ups."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from universal_text2sql.agent.graph import run_query
from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.agent.nodes import generate_sql
from universal_text2sql.agent.state import AgentState
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import SchemaDiscovery
from universal_text2sql.prompts.templates import build_conversation_context_block
from universal_text2sql.utils.demo_data import seed_demo_database


class TestBuildConversationContextBlock:
    def test_empty_history_returns_empty_string(self):
        assert build_conversation_context_block([]) == ""

    def test_single_turn_includes_question_and_sql(self):
        history = [{"question": "How many customers?", "sql": "SELECT COUNT(*) FROM customers", "answer": "10"}]
        block = build_conversation_context_block(history)
        assert "How many customers?" in block
        assert "SELECT COUNT(*) FROM customers" in block
        assert "10" in block

    def test_caps_at_max_turns(self):
        history = [{"question": f"q{i}", "sql": f"sql{i}", "answer": f"a{i}"} for i in range(5)]
        block = build_conversation_context_block(history, max_turns=2)
        assert "q3" in block and "q4" in block
        assert "q0" not in block and "q1" not in block and "q2" not in block

    def test_missing_optional_keys_do_not_raise(self):
        block = build_conversation_context_block([{"question": "q"}])
        assert "q" in block


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "question": "Now filter that by country",
        "relevant_tables": [],
        "schema_context": "CREATE TABLE customers (customer_id INTEGER, country TEXT);",
        "column_samples": "",
        "kg_context": "",
        "business_glossary": "",
        "grounding_hints": "",
        "conversation_context": "",
        "complexity": "",
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


class TestGenerateSqlThreadsConversationContext:
    def test_conversation_context_reaches_the_prompt(self):
        captured = {}

        def _llm(prompt_value):
            captured["text"] = prompt_value.to_string()
            return AIMessage(content="SELECT * FROM customers WHERE country = 'USA'")

        state = _base_state(conversation_context="## Previous Turn(s)\nPrevious question: How many customers?")
        generate_sql(state, llm=RunnableLambda(_llm), db_type="SQLite")

        assert "Previous Turn(s)" in captured["text"]
        assert "How many customers?" in captured["text"]

    def test_missing_conversation_context_defaults_to_empty(self):
        state = _base_state()
        del state["conversation_context"]
        # Should not raise despite the key being absent from state.
        llm = RunnableLambda(lambda _: AIMessage(content="SELECT 1"))
        update = generate_sql(state, llm=llm, db_type="SQLite")
        assert update["generated_sql"] == "SELECT 1"


class TestRunQueryConversationHistory:
    @pytest.fixture()
    def connector(self) -> DatabaseConnector:
        conn = DatabaseConnector("sqlite:///:memory:")
        seed_demo_database(conn)
        return conn

    def test_run_query_without_history_behaves_as_before(self, connector):
        schema = SchemaDiscovery(connector).discover()
        memory = QueryMemory(enabled=False)
        responses = iter(["SIMPLE", "SELECT COUNT(*) AS cnt FROM customers", "VALID", "answer"])
        llm = RunnableLambda(lambda _: AIMessage(content=next(responses)))

        result = run_query(
            question="How many customers are there?",
            connector=connector, schema=schema, memory=memory, llm=llm,
        )
        assert result["conversation_context"] == ""
        assert result["success"] is True

    def test_run_query_with_history_populates_conversation_context(self, connector):
        schema = SchemaDiscovery(connector).discover()
        memory = QueryMemory(enabled=False)
        responses = iter(["SIMPLE", "SELECT * FROM customers WHERE country = 'USA'", "VALID", "answer"])
        llm = RunnableLambda(lambda _: AIMessage(content=next(responses)))

        history = [
            {
                "question": "How many customers are there?",
                "sql": "SELECT COUNT(*) FROM customers",
                "answer": "There are 10 customers.",
            }
        ]
        result = run_query(
            question="Now filter that by country",
            connector=connector, schema=schema, memory=memory, llm=llm,
            conversation_history=history,
        )
        assert "How many customers are there?" in result["conversation_context"]
        assert result["success"] is True
