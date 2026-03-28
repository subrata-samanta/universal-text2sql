"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state that flows through every node in the LangGraph workflow."""

    # The original natural language question
    question: str

    # Relevant tables identified by the schema selector
    relevant_tables: list[str]

    # DDL fragment for relevant tables (used in prompts)
    schema_context: str

    # Column sample values block (used in prompts)
    column_samples: str

    # Few-shot examples retrieved from the query memory
    few_shot_examples: list[dict[str, Any]]

    # Generated SQL query (current attempt)
    generated_sql: str

    # Result of executing the SQL
    execution_result: Any  # pandas DataFrame or None

    # Error message if execution failed
    execution_error: str

    # Validation verdict ("VALID" or "INVALID: <reason>")
    validation_verdict: str

    # Final natural language answer
    final_answer: str

    # How many retry attempts have been made
    retry_count: int

    # Maximum retries allowed (injected at graph invocation time)
    max_retries: int

    # Internal message history for traceability
    messages: Annotated[list, add_messages]

    # Whether the overall run succeeded
    success: bool


class QueryMemoryEntry(BaseModel):
    """A single entry in the RL-inspired query memory."""

    question: str
    sql: str
    success: bool = True
    reward: float = Field(default=1.0, ge=0.0, le=1.0)
