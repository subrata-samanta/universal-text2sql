#!/usr/bin/env python3
"""Command-line interface for the Universal Text-to-SQL Agent.

Usage
-----
    python main.py                        # interactive mode
    python main.py "How many customers?"  # single query mode

Environment
-----------
    GROQ_API_KEY   – required
    DATABASE_URL   – optional (defaults to in-memory SQLite demo)
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)

from universal_text2sql.agent.graph import run_query
from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import SchemaDiscovery
from universal_text2sql.utils.demo_data import seed_demo_database


def _print_banner() -> None:
    print("\n" + "=" * 60)
    print("   Universal Text-to-SQL Agent  (LangGraph + Groq)")
    print("=" * 60 + "\n")


def _setup() -> tuple[DatabaseConnector, object, QueryMemory]:
    db_url = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    print(f"📡 Connecting to: {db_url}")
    connector = DatabaseConnector(db_url)
    seed_demo_database(connector)

    print("🔍 Discovering schema …")
    schema = SchemaDiscovery(connector).discover()
    table_list = ", ".join(schema.tables.keys())
    print(f"✅ Schema ready – tables: {table_list}\n")

    enabled = os.getenv("ENABLE_QUERY_MEMORY", "true").lower() == "true"
    memory = QueryMemory(enabled=enabled)
    return connector, schema, memory


def _ask(question: str, connector, schema, memory) -> None:
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    print(f"\n❓ {question}")
    print("⏳ Processing …")

    result = run_query(
        question=question,
        connector=connector,
        schema=schema,
        memory=memory,
        max_retries=max_retries,
    )

    print(f"\n📝 SQL:\n{result.get('generated_sql', '')}")
    retries = result.get("retry_count", 0)
    if retries:
        print(f"🔁 Self-reflected {retries} time(s)")
    df = result.get("execution_result")
    if df is not None and not df.empty:
        print(f"\n📊 Results ({len(df)} rows):")
        print(df.to_string(index=False))
    elif result.get("execution_error"):
        print(f"\n❌ Error: {result['execution_error']}")
    print(f"\n💬 Answer: {result.get('final_answer', '')}")
    print("-" * 60)


def main() -> None:
    _print_banner()

    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  GROQ_API_KEY is not set. Export it or create a .env file.")
        print("   Example: export GROQ_API_KEY=your_key_here\n")

    connector, schema, memory = _setup()

    # Single-shot mode (argument provided)
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _ask(question, connector, schema, memory)
        return

    # Interactive mode
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye! 👋")
            break
        _ask(question, connector, schema, memory)


if __name__ == "__main__":
    print("main")


    print("new")
    main()
