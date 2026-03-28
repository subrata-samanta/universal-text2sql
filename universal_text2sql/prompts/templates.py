"""Prompt templates for the Text-to-SQL agent.

Design principles
-----------------
* Chain-of-thought (CoT) reasoning is embedded in every generation prompt.
* Few-shot examples are dynamically injected from the RL-inspired memory store.
* Self-reflection prompts guide the model to diagnose and fix SQL errors.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

# ---------------------------------------------------------------------------
# System context shared by all prompts
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """You are an expert SQL engineer. Your job is to translate natural language \
questions into correct, efficient SQL queries.

Database type: {db_type}

Guidelines:
- Always use the exact table and column names from the schema (case-sensitive where required).
- Prefer ANSI SQL unless a database-specific optimisation is clearly needed.
- Use JOINs instead of sub-selects where appropriate for readability.
- Never include DML (INSERT/UPDATE/DELETE/DROP) unless explicitly asked.
- Return ONLY the SQL statement – no markdown fences, no explanation.
"""

# ---------------------------------------------------------------------------
# SQL generation (initial attempt)
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_SYSTEM_BASE),
        (
            "human",
            """## Database Schema
{schema_ddl}

## Column Samples
{column_samples}

{few_shot_block}
## Question
{question}

Think step-by-step:
1. Identify which tables and columns are relevant.
2. Determine what aggregations, filters, or joins are needed.
3. Write the final SQL query.

SQL:""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Self-reflection / error correction
# ---------------------------------------------------------------------------

SQL_REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_SYSTEM_BASE),
        (
            "human",
            """## Database Schema
{schema_ddl}

## Previous SQL Attempt
```sql
{previous_sql}
```

## Error Encountered
{error_message}

## Question
{question}

Analyse the error carefully, then rewrite the SQL to fix the problem.
Return ONLY the corrected SQL statement.

Corrected SQL:""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Result validation (does the answer make sense?)
# ---------------------------------------------------------------------------

RESULT_VALIDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """You are validating whether a SQL query result correctly answers the user question.

Question: {question}
SQL used: {sql}
Result (first 5 rows):
{result_preview}

Does this result correctly answer the question? Reply with exactly one of:
- "VALID" if the result answers the question.
- "INVALID: <reason>" if the result is wrong or empty and you can suggest a fix.

Response:""",
        )
    ]
)

# ---------------------------------------------------------------------------
# Natural language answer generation
# ---------------------------------------------------------------------------

ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Convert the SQL query result into a clear, concise natural language answer.

Original question: {question}
SQL query: {sql}
Result:
{result_preview}

Natural language answer:""",
        )
    ]
)

# ---------------------------------------------------------------------------
# Few-shot block builder
# ---------------------------------------------------------------------------


def build_few_shot_block(examples: list[dict]) -> str:
    """Build a few-shot examples block from *examples*.

    Each example must have ``question`` and ``sql`` keys.
    Returns an empty string when the list is empty.
    """
    if not examples:
        return ""
    lines = ["## Successful Query Examples (for reference)\n"]
    for i, ex in enumerate(examples[:5], 1):  # cap at 5 examples
        lines.append(f"### Example {i}")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"SQL: {ex['sql']}\n")
    return "\n".join(lines) + "\n"


def build_column_samples_block(schema_tables: dict) -> str:
    """Build a compact column sample block from schema table metadata."""
    lines: list[str] = []
    for table_name, meta in schema_tables.items():
        for col in meta.columns:
            if col.sample_values:
                samples = ", ".join(str(v) for v in col.sample_values[:3])
                lines.append(f"  {table_name}.{col.name}: [{samples}]")
    return "\n".join(lines) if lines else "No sample values available."
