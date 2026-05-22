"""LangGraph workflow for the Universal Text-to-SQL agent.

Graph topology
--------------

    START
      │
      ▼
  select_schema          ← identify relevant tables, build context
      │
      ▼
  generate_sql           ← CoT + few-shot SQL generation via Groq
      │
      ▼
  execute_sql            ← run against live database
      │
      ├─ error ──► reflect ──► execute_sql  (up to MAX_RETRIES)
      │
      ▼
  validate_result        ← ask LLM if result answers the question
      │
      ├─ invalid ► reflect ──► execute_sql  (up to MAX_RETRIES)
      │
      ▼
  format_answer          ← produce natural language response
      │
      ▼
  store_memory           ← RL-inspired: save success for future few-shots
      │
      ▼
    END
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.agent.nodes import (
    LLMRunnable,
    execute_sql,
    format_answer,
    generate_sql,
    reflect,
    select_schema,
    store_memory,
    validate_result,
)
from universal_text2sql.agent.state import AgentState
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import DatabaseSchema
from universal_text2sql.llm.groq_client import get_groq_llm

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


# ---------------------------------------------------------------------------
# Conditional routing helpers
# ---------------------------------------------------------------------------


def _should_reflect_or_continue(state: AgentState) -> str:
    """After execute_sql: route to reflect on error or proceed to validate."""
    if state.get("execution_error"):
        if state.get("retry_count", 0) >= state.get("max_retries", _DEFAULT_MAX_RETRIES):
            logger.warning("Max retries reached. Moving to format_answer with error.")
            return "format_answer"
        return "reflect"
    return "validate_result"


def _should_reflect_on_validation(state: AgentState) -> str:
    """After validate_result: route to reflect when the result is invalid."""
    verdict = state.get("validation_verdict", "")
    if verdict.upper().startswith("INVALID"):
        if state.get("retry_count", 0) >= state.get("max_retries", _DEFAULT_MAX_RETRIES):
            logger.warning("Max retries reached after validation. Returning best effort answer.")
            return "format_answer"
        return "reflect"
    return "format_answer"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    connector: DatabaseConnector,
    schema: DatabaseSchema,
    memory: QueryMemory,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    llm: LLMRunnable | None = None,
) -> Any:
    """Build and compile the LangGraph StateGraph.

    Args:
        connector: Live database connection.
        schema: Pre-discovered database schema.
        memory: RL-inspired query memory.
        max_retries: Maximum self-reflection iterations.
        llm: Optional pre-built LLM; if ``None`` a default Groq LLM is used.

    Returns:
        A compiled LangGraph runnable.
    """
    if llm is None:
        llm = get_groq_llm()

    db_type = connector.db_type

    # Bind dependencies into each node via functools.partial
    _select_schema = functools.partial(select_schema, schema=schema, memory=memory)
    _generate_sql = functools.partial(generate_sql, llm=llm, db_type=db_type)
    _execute_sql = functools.partial(execute_sql, connector=connector)
    _validate_result = functools.partial(validate_result, llm=llm)
    _reflect = functools.partial(reflect, llm=llm, db_type=db_type)
    _format_answer = functools.partial(format_answer, llm=llm)
    _store_memory = functools.partial(store_memory, memory=memory)

    # Build the StateGraph
    builder = StateGraph(AgentState)

    builder.add_node("select_schema", _select_schema)
    builder.add_node("generate_sql", _generate_sql)
    builder.add_node("execute_sql", _execute_sql)
    builder.add_node("validate_result", _validate_result)
    builder.add_node("reflect", _reflect)
    builder.add_node("format_answer", _format_answer)
    builder.add_node("store_memory", _store_memory)

    # Edges
    builder.add_edge(START, "select_schema")
    builder.add_edge("select_schema", "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_conditional_edges(
        "execute_sql",
        _should_reflect_or_continue,
        {
            "reflect": "reflect",
            "validate_result": "validate_result",
            "format_answer": "format_answer",
        },
    )
    builder.add_edge("reflect", "execute_sql")
    builder.add_conditional_edges(
        "validate_result",
        _should_reflect_on_validation,
        {
            "reflect": "reflect",
            "format_answer": "format_answer",
        },
    )
    builder.add_edge("format_answer", "store_memory")
    builder.add_edge("store_memory", END)

    graph = builder.compile()
    logger.info("LangGraph compiled successfully.")
    return graph


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_query(
    question: str,
    connector: DatabaseConnector,
    schema: DatabaseSchema,
    memory: QueryMemory,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run a single natural language question through the agent.

    Returns the final :class:`AgentState` as a dict.
    """
    graph = build_graph(
        connector=connector,
        schema=schema,
        memory=memory,
        max_retries=max_retries,
        llm=llm,
    )

    initial_state: AgentState = {
        "question": question,
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
        "max_retries": max_retries,
        "messages": [],
        "success": False,
    }

    result = graph.invoke(initial_state)
    return result
