"""LangGraph node implementations for the Text-to-SQL agent.

Each node is a plain callable that receives the current :class:`AgentState`
and returns a *partial* state update (only the keys being modified).

Node overview
-------------
1. ``select_schema``   – identify relevant tables from the question.
2. ``generate_sql``    – produce a SQL query with CoT reasoning.
3. ``execute_sql``     – run the query against the live database.
4. ``validate_result`` – check whether the result actually answers the question.
5. ``reflect``         – diagnose errors and rewrite the SQL.
6. ``format_answer``   – turn the DataFrame result into a human answer.
7. ``store_memory``    – persist successful queries for future few-shot use.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.agent.state import AgentState
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import DatabaseSchema
from universal_text2sql.prompts.templates import (
    ANSWER_GENERATION_PROMPT,
    RESULT_VALIDATION_PROMPT,
    SQL_GENERATION_PROMPT,
    SQL_REFLECTION_PROMPT,
    build_column_samples_block,
    build_few_shot_block,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMRunnable(Protocol):
    """Minimal protocol for a LangChain-compatible chat LLM.

    Both :class:`~langchain_groq.ChatGroq` and
    :class:`~langchain_core.runnables.RunnableLambda` satisfy this protocol.
    """

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        ...

    def __or__(self, other: Any) -> Any:
        ...


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
) -> dict[str, Any]:
    """Identify relevant tables and build prompt context."""
    question = state["question"]

    # Keyword extraction: split question and use word tokens as search terms
    keywords = [w for w in question.lower().split() if len(w) > 3]
    relevant = schema.get_relevant_tables(keywords)

    # Fallback: use all tables if nothing matched
    if not relevant:
        relevant = list(schema.tables.keys())

    schema_ddl = schema.subset_ddl(relevant)
    column_samples = build_column_samples_block(
        {t: schema.tables[t] for t in relevant if t in schema.tables}
    )

    # Retrieve few-shot examples from memory
    few_shot = memory.get_similar(question, top_k=3)

    logger.info("Selected tables: %s", relevant)

    return {
        "relevant_tables": relevant,
        "schema_context": schema_ddl,
        "column_samples": column_samples,
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
) -> dict[str, Any]:
    """Generate a SQL query using CoT prompting and optional few-shot examples."""
    few_shot_block = build_few_shot_block(state.get("few_shot_examples", []))

    chain = SQL_GENERATION_PROMPT | llm
    response = chain.invoke(
        {
            "db_type": db_type,
            "schema_ddl": state["schema_context"],
            "column_samples": state["column_samples"],
            "few_shot_block": few_shot_block,
            "question": state["question"],
        }
    )

    raw_sql = response.content if hasattr(response, "content") else str(response)
    sql = _extract_sql(raw_sql)
    logger.info("Generated SQL: %s", sql)

    return {
        "generated_sql": sql,
        "execution_error": "",
        "messages": [AIMessage(content=f"Generated SQL:\n```sql\n{sql}\n```")],
    }


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
