"""LangGraph node implementations for the Text-to-SQL agent.

Each node is a plain callable that receives the current :class:`AgentState`
and returns a *partial* state update (only the keys being modified).

Node overview
-------------
1. ``select_schema``        – semantic + knowledge-graph schema linking.
2. ``classify_complexity``  – route simple vs. complex questions.
3. ``generate_sql``         – produce (optionally multiple) SQL candidates via CoT reasoning.
4. ``select_best_candidate``– self-consistency voting across candidates.
5. ``execute_sql``          – run the query against the live database.
6. ``validate_result``      – check whether the result actually answers the question.
7. ``reflect``               – diagnose errors and rewrite the SQL.
8. ``format_answer``        – turn the DataFrame result into a human answer.
9. ``store_memory``         – persist successful queries for future few-shot use.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.agent.scoring import result_signature
from universal_text2sql.agent.state import AgentState
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import DatabaseSchema
from universal_text2sql.knowledge.graph import SchemaKnowledgeGraph
from universal_text2sql.knowledge.grounding import ValueGroundingIndex
from universal_text2sql.knowledge.metadata import MetadataEnricher
from universal_text2sql.llm.base import LLMRunnable
from universal_text2sql.prompts.templates import (
    ANSWER_GENERATION_PROMPT,
    QUERY_COMPLEXITY_PROMPT,
    RESULT_VALIDATION_PROMPT,
    SQL_GENERATION_PROMPT,
    SQL_REFLECTION_PROMPT,
    build_column_samples_block,
    build_few_shot_block,
)

logger = logging.getLogger(__name__)


def _extract_sql(text: str) -> str:
    """Strip markdown code fences and extra whitespace from LLM output."""
    # Remove ```sql ... ``` fences
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    # Strip leading "SQL:" label the LLM sometimes emits
    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _df_preview(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Return a compact string preview of a DataFrame."""
    if df is None or df.empty:
        return "(empty result)"
    try:
        return df.head(max_rows).to_string(index=False)
    except Exception:
        return str(df.head(max_rows))


# ---------------------------------------------------------------------------
# Node: select_schema
# ---------------------------------------------------------------------------


def select_schema(
    state: AgentState,
    schema: DatabaseSchema,
    memory: QueryMemory,
    knowledge_graph: SchemaKnowledgeGraph | None = None,
    metadata_enricher: MetadataEnricher | None = None,
    grounding_index: ValueGroundingIndex | None = None,
) -> dict[str, Any]:
    """Identify relevant tables and build prompt context.

    Schema linking prefers TF-IDF semantic similarity over the table's
    name/columns/auto-generated glossary (so a question can match a table
    even without literal keyword overlap); it falls back to plain keyword
    overlap when the semantic index can't distinguish any table. When a
    :class:`SchemaKnowledgeGraph` is supplied, the relevant tables' join
    relationships (declared, inferred, or multi-hop) are rendered into
    ``kg_context``; when a :class:`MetadataEnricher` is supplied, the
    business-glossary block is rendered into ``business_glossary``; when a
    :class:`ValueGroundingIndex` is supplied, question terms matched
    against real column values are rendered into ``grounding_hints``.
    """
    question = state["question"]

    relevant = schema.get_relevant_tables_semantic(question)

    # Fallback: use all tables if nothing matched
    if not relevant:
        relevant = list(schema.tables.keys())

    schema_ddl = schema.subset_ddl(relevant)
    column_samples = build_column_samples_block(
        {t: schema.tables[t] for t in relevant if t in schema.tables}
    )

    kg_context = ""
    if knowledge_graph is not None:
        kg_context = knowledge_graph.describe(relevant)

    business_glossary = ""
    if metadata_enricher is not None:
        business_glossary = metadata_enricher.glossary_block(schema, relevant)

    grounding_hints = ""
    if grounding_index is not None:
        grounding_hints = grounding_index.hints_block(grounding_index.match(question, relevant))

    # Retrieve few-shot examples from memory
    few_shot = memory.get_similar(question, top_k=3)

    logger.info("Selected tables: %s", relevant)

    return {
        "relevant_tables": relevant,
        "schema_context": schema_ddl,
        "column_samples": column_samples,
        "kg_context": kg_context,
        "business_glossary": business_glossary,
        "grounding_hints": grounding_hints,
        "few_shot_examples": few_shot,
        "messages": [HumanMessage(content=f"Processing question: {question}")],
    }


# ---------------------------------------------------------------------------
# Node: generate_sql
# ---------------------------------------------------------------------------


def generate_sql(
    state: AgentState,
    llm: LLMRunnable,
    db_type: str,
    self_consistency_samples: int = 1,
) -> dict[str, Any]:
    """Generate SQL using CoT prompting, few-shot examples, and (optionally) self-consistency.

    When ``self_consistency_samples`` > 1 and the question was classified as
    at least ``MODERATE`` complexity, multiple candidate queries are sampled
    from the LLM (mirroring the self-consistency technique from CHASE-SQL /
    DIN-SQL-style pipelines). All candidates are stored in
    ``sql_candidates``; the first one is also set as ``generated_sql`` so the
    rest of the graph behaves identically whether or not self-consistency
    produced more than one candidate — the ``select_best_candidate`` node
    decides the final winner by execution voting.
    """
    few_shot_block = build_few_shot_block(state.get("few_shot_examples", []))
    prompt_input = {
        "db_type": db_type,
        "schema_ddl": state["schema_context"],
        "column_samples": state["column_samples"],
        "kg_context": state.get("kg_context", ""),
        "business_glossary": state.get("business_glossary", ""),
        "grounding_hints": state.get("grounding_hints", ""),
        "few_shot_block": few_shot_block,
        "question": state["question"],
    }

    n_samples = 1 if state.get("complexity", "").upper() == "SIMPLE" else max(1, self_consistency_samples)

    chain = SQL_GENERATION_PROMPT | llm
    candidates: list[str] = []
    for _ in range(n_samples):
        response = chain.invoke(prompt_input)
        raw_sql = response.content if hasattr(response, "content") else str(response)
        sql = _extract_sql(raw_sql)
        if sql and sql not in candidates:
            candidates.append(sql)
    if not candidates:
        candidates = [""]

    logger.info("Generated %d SQL candidate(s); primary: %s", len(candidates), candidates[0])

    return {
        "generated_sql": candidates[0],
        "sql_candidates": candidates,
        "execution_error": "",
        "messages": [AIMessage(content=f"Generated SQL:\n```sql\n{candidates[0]}\n```")],
    }


# ---------------------------------------------------------------------------
# Node: classify_complexity
# ---------------------------------------------------------------------------


def classify_complexity(
    state: AgentState,
    llm: LLMRunnable,
) -> dict[str, Any]:
    """Classify question difficulty to route self-consistency sampling.

    DIN-SQL-style difficulty routing: cheap single-shot generation for
    simple lookups, multi-candidate self-consistency reserved for questions
    that actually need multiple joins/aggregation/subqueries.
    """
    chain = QUERY_COMPLEXITY_PROMPT | llm
    response = chain.invoke(
        {
            "schema_ddl": state.get("schema_context", ""),
            "question": state["question"],
        }
    )
    raw = response.content if hasattr(response, "content") else str(response)
    label = raw.strip().upper()
    complexity = next((c for c in ("SIMPLE", "MODERATE", "COMPLEX") if c in label), "MODERATE")
    logger.info("Classified query complexity: %s", complexity)
    return {"complexity": complexity}


# ---------------------------------------------------------------------------
# Node: select_best_candidate (self-consistency voting)
# ---------------------------------------------------------------------------


_result_signature = result_signature  # local alias, kept for call-site brevity below

_SELF_CONSISTENCY_PREVIEW_LIMIT = 200


def select_best_candidate(
    state: AgentState,
    connector: DatabaseConnector,
) -> dict[str, Any]:
    """Execute every self-consistency candidate and pick the majority result.

    A no-op (returns ``{}``) when only one candidate was generated — the
    overwhelmingly common case — so the ``execute_sql``/``reflect`` loop
    behaves exactly as it did before self-consistency sampling existed.

    Candidate executions are capped with ``limit=`` (they're only used to
    vote on which SQL text wins, not returned to the user) — the winning
    query is re-executed unbounded by ``execute_sql`` afterwards, so this
    never truncates the actual answer.
    """
    candidates = state.get("sql_candidates") or []
    if len(candidates) <= 1:
        return {}

    groups: dict[str, list[str]] = {}
    first_error: tuple[str, str] | None = None
    for sql in candidates:
        try:
            df = connector.execute_query(sql, limit=_SELF_CONSISTENCY_PREVIEW_LIMIT)
            groups.setdefault(_result_signature(df), []).append(sql)
        except Exception as exc:
            if first_error is None:
                first_error = (sql, str(exc))

    if not groups:
        logger.info("Self-consistency: all %d candidates failed execution.", len(candidates))
        return {"generated_sql": first_error[0]}

    winning_group = max(groups.values(), key=len)
    logger.info(
        "Self-consistency: %d/%d candidates agreed on the winning SQL.",
        len(winning_group),
        len(candidates),
    )
    return {"generated_sql": winning_group[0]}


# ---------------------------------------------------------------------------
# Node: execute_sql
# ---------------------------------------------------------------------------


def execute_sql(
    state: AgentState,
    connector: DatabaseConnector,
) -> dict[str, Any]:
    """Execute the current SQL query against the database."""
    sql = state["generated_sql"]
    try:
        df = connector.execute_query(sql)
        logger.info("Query executed successfully, %d rows returned.", len(df))
        return {
            "execution_result": df,
            "execution_error": "",
            "success": True,
        }
    except Exception as exc:
        error_msg = str(exc)
        logger.warning("SQL execution error: %s", error_msg)
        return {
            "execution_result": None,
            "execution_error": error_msg,
            "success": False,
        }


# ---------------------------------------------------------------------------
# Node: validate_result
# ---------------------------------------------------------------------------


def validate_result(
    state: AgentState,
    llm: LLMRunnable,
) -> dict[str, Any]:
    """Validate that the query result actually answers the question."""
    # If there was an execution error, skip LLM validation
    if state.get("execution_error"):
        return {"validation_verdict": f"INVALID: {state['execution_error']}"}

    df = state.get("execution_result")
    result_preview = _df_preview(df)

    chain = RESULT_VALIDATION_PROMPT | llm
    response = chain.invoke(
        {
            "question": state["question"],
            "sql": state["generated_sql"],
            "result_preview": result_preview,
        }
    )

    verdict = response.content.strip() if hasattr(response, "content") else str(response).strip()
    logger.info("Validation verdict: %s", verdict)

    return {
        "validation_verdict": verdict,
        "success": verdict.upper().startswith("VALID"),
    }


# ---------------------------------------------------------------------------
# Node: reflect
# ---------------------------------------------------------------------------


def reflect(
    state: AgentState,
    llm: LLMRunnable,
    db_type: str,
) -> dict[str, Any]:
    """Reflect on errors and generate a corrected SQL query."""
    retry_count = state.get("retry_count", 0) + 1

    error_message = state.get("execution_error") or state.get("validation_verdict", "Unknown error")

    chain = SQL_REFLECTION_PROMPT | llm
    response = chain.invoke(
        {
            "db_type": db_type,
            "schema_ddl": state["schema_context"],
            "kg_context": state.get("kg_context", ""),
            "grounding_hints": state.get("grounding_hints", ""),
            "previous_sql": state["generated_sql"],
            "error_message": error_message,
            "question": state["question"],
        }
    )

    raw_sql = response.content if hasattr(response, "content") else str(response)
    corrected_sql = _extract_sql(raw_sql)
    logger.info("Reflected SQL (attempt %d): %s", retry_count, corrected_sql)

    return {
        "generated_sql": corrected_sql,
        "retry_count": retry_count,
        "execution_error": "",
        "messages": [
            AIMessage(
                content=(
                    f"Self-reflection (attempt {retry_count}):\n"
                    f"Error: {error_message}\n"
                    f"Corrected SQL:\n```sql\n{corrected_sql}\n```"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: format_answer
# ---------------------------------------------------------------------------


def format_answer(
    state: AgentState,
    llm: LLMRunnable,
) -> dict[str, Any]:
    """Convert the DataFrame result into a human-readable answer."""
    df = state.get("execution_result")
    result_preview = _df_preview(df, max_rows=20)

    chain = ANSWER_GENERATION_PROMPT | llm
    response = chain.invoke(
        {
            "question": state["question"],
            "sql": state["generated_sql"],
            "result_preview": result_preview,
        }
    )

    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
    logger.info("Final answer generated.")

    return {
        "final_answer": answer,
        "success": True,
        "messages": [AIMessage(content=f"Answer: {answer}")],
    }


# ---------------------------------------------------------------------------
# Node: store_memory
# ---------------------------------------------------------------------------


def store_memory(
    state: AgentState,
    memory: QueryMemory,
) -> dict[str, Any]:
    """Persist a successful query to the RL-inspired memory store."""
    if state.get("success") and state.get("generated_sql"):
        memory.add(
            question=state["question"],
            sql=state["generated_sql"],
            retry_count=state.get("retry_count", 0),
        )
        logger.info("Stored query in memory.")
    return {}
