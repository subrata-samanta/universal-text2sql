#!/usr/bin/env python3
"""Command-line interface for the Universal Text-to-SQL Agent.

Usage
-----
    python main.py                        # interactive mode
    python main.py "How many customers?"  # single query mode
    python main.py --graph                # print the auto-built knowledge graph and exit
    python main.py --glossary             # print the auto-generated business glossary and exit

Environment
-----------
    GROQ_API_KEY   – required for SQL generation (optional for --graph)
    DATABASE_URL   – optional (defaults to in-memory SQLite demo)
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)

from universal_text2sql.bootstrap import AgentContext, bootstrap


def _print_banner() -> None:
    print("\n" + "=" * 60)
    print("   Universal Text-to-SQL Agent  (LangGraph + Groq)")
    print("   Auto schema discovery · knowledge graph · glossary")
    print("=" * 60 + "\n")


def _setup() -> AgentContext:
    ctx = bootstrap()
    print(f"📡 Connected: {ctx.connector.connection_url}")
    table_list = ", ".join(ctx.schema.tables.keys())
    print(f"🔍 Schema discovered – tables: {table_list}")
    print(
        f"🕸️  Knowledge graph: {ctx.knowledge_graph.graph.number_of_nodes()} nodes, "
        f"{ctx.knowledge_graph.graph.number_of_edges()} edges"
    )
    if ctx.llm is None:
        print("⚠️  GROQ_API_KEY is not set. SQL generation will fail until it is configured.")
    print()
    return ctx


def _ask(question: str, ctx: AgentContext) -> None:
    print(f"\n❓ {question}")
    print("⏳ Processing …")

    result = ctx.ask(question)

    print(f"\n📝 SQL:\n{result.get('generated_sql', '')}")
    if result.get("complexity"):
        print(f"🧮 Complexity: {result['complexity']}")
    if len(result.get("sql_candidates", []) or []) > 1:
        print(f"🗳️  Self-consistency: {len(result['sql_candidates'])} candidates sampled")
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

    if "--graph" in sys.argv:
        ctx = bootstrap()
        print(ctx.describe_knowledge_graph())
        return

    if "--glossary" in sys.argv:
        ctx = bootstrap()
        print(ctx.describe_glossary())
        return

    ctx = _setup()

    # Single-shot mode (argument provided)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        _ask(" ".join(args), ctx)
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
        _ask(question, ctx)


if __name__ == "__main__":
    main()
