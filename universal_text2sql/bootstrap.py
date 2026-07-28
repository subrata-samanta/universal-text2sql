"""Single entry point that turns *any* database into a ready-to-query agent.

``bootstrap()`` is what makes this a *universal* text-to-SQL agent rather
than one configured for a specific dataset: point it at a connection URL
and it will, with zero manual setup:

1. Connect — any SQLAlchemy-supported backend (SQLite, PostgreSQL, MySQL, ...).
2. Discover the schema live (tables, columns, types, PK/FK, sample values).
3. Build a knowledge graph of the schema — declared foreign keys plus
   naming-convention-inferred relationships and cross-table semantic
   siblings (see :mod:`universal_text2sql.knowledge.graph`).
4. Auto-generate a business glossary — heuristic descriptions always,
   LLM-written descriptions/synonyms additionally when an LLM is available
   (see :mod:`universal_text2sql.knowledge.metadata`), cached to disk.
5. Load the RL-inspired query memory used for dynamic few-shot prompting.

The returned :class:`AgentContext` bundles all of it behind a single
``.ask(question)`` call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from universal_text2sql.agent.graph import run_query
from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import DatabaseSchema, SchemaDiscovery
from universal_text2sql.knowledge.graph import SchemaKnowledgeGraph
from universal_text2sql.knowledge.metadata import MetadataEnricher
from universal_text2sql.llm.base import LLMRunnable

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Everything the agent needs to answer questions against one database."""

    connector: DatabaseConnector
    schema: DatabaseSchema
    knowledge_graph: SchemaKnowledgeGraph
    metadata_enricher: MetadataEnricher
    memory: QueryMemory
    llm: LLMRunnable | None = None
    max_retries: int = 3
    self_consistency_samples: int = 1

    def ask(self, question: str) -> dict[str, Any]:
        """Run a single natural language question through the agent graph."""
        return run_query(
            question=question,
            connector=self.connector,
            schema=self.schema,
            memory=self.memory,
            max_retries=self.max_retries,
            llm=self.llm,
            knowledge_graph=self.knowledge_graph,
            metadata_enricher=self.metadata_enricher,
            self_consistency_samples=self.self_consistency_samples,
        )

    def describe_knowledge_graph(self) -> str:
        """Human-readable relationship summary, e.g. for a UI sidebar."""
        return self.knowledge_graph.describe(list(self.schema.tables.keys()))

    def describe_glossary(self) -> str:
        """Full auto-generated business glossary, e.g. for a UI sidebar."""
        return self.metadata_enricher.glossary_block(self.schema)


def bootstrap(
    database_url: str | None = None,
    llm: LLMRunnable | None = None,
    enable_metadata_enrichment: bool | None = None,
    enable_knowledge_graph: bool = True,
    enable_query_memory: bool | None = None,
    seed_demo: bool = True,
    max_retries: int | None = None,
    self_consistency_samples: int | None = None,
) -> AgentContext:
    """Connect to *any* SQLAlchemy-supported database and prepare the agent.

    Args:
        database_url: SQLAlchemy connection URL. Falls back to the
            ``DATABASE_URL`` env var, then an in-memory SQLite demo DB.
        llm: Pre-built chat LLM used both for SQL generation and for the
            LLM-assisted metadata pass. Falls back to a Groq client built
            from ``GROQ_API_KEY``/``GROQ_MODEL`` when available; metadata
            enrichment silently degrades to heuristics-only if no LLM can
            be constructed (e.g. no API key set), and SQL generation calls
            will fail if invoked without an LLM.
        enable_metadata_enrichment: Toggle the LLM-based glossary pass.
            Defaults to the ``ENABLE_METADATA_ENRICHMENT`` env var
            (``true`` by default). Heuristic descriptions are always
            generated regardless of this flag.
        enable_knowledge_graph: Build the schema knowledge graph (default
            ``True``; cheap, no LLM calls).
        enable_query_memory: Toggle the RL-inspired few-shot query memory.
            Defaults to the ``ENABLE_QUERY_MEMORY`` env var.
        seed_demo: When the target database has no tables, seed it with a
            small demo dataset so the agent is immediately usable.
        max_retries: Max self-reflection retries. Defaults to ``MAX_RETRIES``.
        self_consistency_samples: SQL candidates sampled for non-trivial
            questions. Defaults to ``SELF_CONSISTENCY_SAMPLES`` (``1``,
            i.e. self-consistency disabled, since it multiplies LLM calls).

    Returns:
        A ready-to-use :class:`AgentContext`.
    """
    db_url = database_url or os.getenv("DATABASE_URL", "sqlite:///:memory:")
    logger.info("Bootstrapping universal text-to-SQL agent for %s", db_url)

    connector = DatabaseConnector(db_url)

    if seed_demo and not connector.get_table_names():
        from universal_text2sql.utils.demo_data import seed_demo_database

        seed_demo_database(connector)

    schema = SchemaDiscovery(connector).discover()
    logger.info("Discovered %d table(s): %s", len(schema.tables), ", ".join(schema.tables))

    knowledge_graph = (
        SchemaKnowledgeGraph.build(schema) if enable_knowledge_graph else SchemaKnowledgeGraph()
    )

    if llm is None:
        llm = _default_llm()

    if enable_metadata_enrichment is None:
        enable_metadata_enrichment = (
            os.getenv("ENABLE_METADATA_ENRICHMENT", "true").lower() == "true"
        )
    metadata_enricher = MetadataEnricher(llm=llm, enabled=enable_metadata_enrichment)
    metadata_enricher.enrich(schema)

    if enable_query_memory is None:
        enable_query_memory = os.getenv("ENABLE_QUERY_MEMORY", "true").lower() == "true"
    memory = QueryMemory(enabled=enable_query_memory)

    resolved_max_retries = (
        max_retries if max_retries is not None else int(os.getenv("MAX_RETRIES", "3"))
    )
    resolved_samples = (
        self_consistency_samples
        if self_consistency_samples is not None
        else int(os.getenv("SELF_CONSISTENCY_SAMPLES", "1"))
    )

    return AgentContext(
        connector=connector,
        schema=schema,
        knowledge_graph=knowledge_graph,
        metadata_enricher=metadata_enricher,
        memory=memory,
        llm=llm,
        max_retries=resolved_max_retries,
        self_consistency_samples=resolved_samples,
    )


def _default_llm() -> LLMRunnable | None:
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        from universal_text2sql.llm.groq_client import get_groq_llm

        return get_groq_llm()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not initialise default LLM: %s", exc)
        return None
