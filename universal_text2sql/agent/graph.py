"""LangGraph workflow for the Universal Text-to-SQL agent.

Graph topology
--------------

    START
      │
      ▼
  select_schema          ← semantic + knowledge-graph schema linking, glossary lookup
      │
      ▼
  classify_complexity    ← SIMPLE / MODERATE / COMPLEX routing (DIN-SQL-style)
      │
      ▼
  generate_sql           ← CoT + few-shot SQL generation (multi-candidate if complex)
      │
      ▼
  select_best_candidate  ← self-consistency: execute candidates, majority vote
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
    classify_complexity,
    execute_sql,
    format_answer,
    generate_sql,
    reflect,
    select_best_candidate,
    select_schema,
    store_memory,
    validate_result,
)
from universal_text2sql.agent.state import AgentState
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import DatabaseSchema
from universal_text2sql.knowledge.graph import SchemaKnowledgeGraph
from universal_text2sql.knowledge.grounding import ValueGroundingIndex
from universal_text2sql.knowledge.metadata import MetadataEnricher
from universal_text2sql.llm.base import LLMRunnable
from universal_text2sql.llm.factory import get_llm

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
_DEFAULT_SELF_CONSISTENCY_SAMPLES = int(os.getenv("SELF_CONSISTENCY_SAMPLES", "1"))


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
    knowledge_graph: SchemaKnowledgeGraph | None = None,
    metadata_enricher: MetadataEnricher | None = None,
    grounding_index: ValueGroundingIndex | None = None,
    self_consistency_samples: int = _DEFAULT_SELF_CONSISTENCY_SAMPLES,
) -> Any:
    """Build and compile the LangGraph StateGraph.

    Args:
        connector: Live database connection.
        schema: Pre-discovered database schema.
        memory: RL-inspired query memory.
        max_retries: Maximum self-reflection iterations.
        llm: Optional pre-built LLM; if ``None``, one is built via
            :func:`~universal_text2sql.llm.factory.get_llm` (Groq by
            default, or another provider via the ``LLM_PROVIDER`` env var).
        knowledge_graph: Optional auto-built schema knowledge graph, used for
            join-path context during SQL generation/reflection. See
            :mod:`universal_text2sql.knowledge.graph`.
        metadata_enricher: Optional auto-generated business glossary, used to
            resolve business terms that don't literally match column names.
            See :mod:`universal_text2sql.knowledge.metadata`.
        grounding_index: Optional value/entity grounding index, used to match
            question terms against real column values for filter literals.
            See :mod:`universal_text2sql.knowledge.grounding`.
        self_consistency_samples: How many SQL candidates to sample for
            questions classified as at least ``MODERATE`` complexity (``1``
            disables self-consistency and matches the original single-shot
            behaviour).

    Returns:
        A compiled LangGraph runnable.
    """
    if llm is None:
        # Defaults to Groq (matching the original behaviour exactly) unless
        # LLM_PROVIDER selects a different provider. See universal_text2sql.llm.factory.
        llm = get_llm()

    db_type = connector.db_type

    # Bind dependencies into each node via functools.partial
    _select_schema = functools.partial(
        select_schema,
        schema=schema,
        memory=memory,
        knowledge_graph=knowledge_graph,
        metadata_enricher=metadata_enricher,
        grounding_index=grounding_index,
    )
    _classify_complexity = functools.partial(classify_complexity, llm=llm)
    _generate_sql = functools.partial(
        generate_sql,
        llm=llm,
        db_type=db_type,
        self_consistency_samples=self_consistency_samples,
    )
    _select_best_candidate = functools.partial(select_best_candidate, connector=connector)
    _execute_sql = functools.partial(execute_sql, connector=connector)
    _validate_result = functools.partial(validate_result, llm=llm)
    _reflect = functools.partial(reflect, llm=llm, db_type=db_type)
    _format_answer = functools.partial(format_answer, llm=llm)
    _store_memory = functools.partial(store_memory, memory=memory)

    # Build the StateGraph
    builder = StateGraph(AgentState)

    builder.add_node("select_schema", _select_schema)
    builder.add_node("classify_complexity", _classify_complexity)
    builder.add_node("generate_sql", _generate_sql)
    builder.add_node("select_best_candidate", _select_best_candidate)
    builder.add_node("execute_sql", _execute_sql)
    builder.add_node("validate_result", _validate_result)
    builder.add_node("reflect", _reflect)
    builder.add_node("format_answer", _format_answer)
    builder.add_node("store_memory", _store_memory)

    # Edges
    builder.add_edge(START, "select_schema")
    builder.add_edge("select_schema", "classify_complexity")
    builder.add_edge("classify_complexity", "generate_sql")
    builder.add_edge("generate_sql", "select_best_candidate")
    builder.add_edge("select_best_candidate", "execute_sql")
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
    knowledge_graph: SchemaKnowledgeGraph | None = None,
    metadata_enricher: MetadataEnricher | None = None,
    grounding_index: ValueGroundingIndex | None = None,
    self_consistency_samples: int = _DEFAULT_SELF_CONSISTENCY_SAMPLES,
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
        knowledge_graph=knowledge_graph,
        metadata_enricher=metadata_enricher,
        grounding_index=grounding_index,
        self_consistency_samples=self_consistency_samples,
    )

    initial_state: AgentState = {
        "question": question,
        "relevant_tables": [],
        "schema_context": "",
        "column_samples": "",
        "kg_context": "",
        "business_glossary": "",
        "grounding_hints": "",
        "complexity": "",
        "sql_candidates": [],
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
