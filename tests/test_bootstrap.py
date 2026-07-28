"""Tests for the bootstrap() orchestrator (AgentContext)."""

from __future__ import annotations

from pathlib import Path

import pytest

from universal_text2sql.bootstrap import AgentContext, bootstrap


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep bootstrap() from touching the repo working directory or a real API."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("QUERY_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("METADATA_CACHE_DIR", str(tmp_path / "cache"))


class TestBootstrap:
    def test_returns_fully_populated_context(self):
        ctx = bootstrap(database_url="sqlite:///:memory:")
        assert isinstance(ctx, AgentContext)
        assert "customers" in ctx.schema.tables
        assert ctx.llm is None  # no GROQ_API_KEY in this environment
        assert ctx.knowledge_graph.graph.number_of_edges() > 0
        assert ctx.memory is not None

    def test_heuristic_glossary_populated_without_llm(self):
        ctx = bootstrap(database_url="sqlite:///:memory:")
        glossary = ctx.describe_glossary()
        assert "customers" in glossary
        assert "primary key" in glossary.lower()

    def test_knowledge_graph_can_be_disabled(self):
        ctx = bootstrap(database_url="sqlite:///:memory:", enable_knowledge_graph=False)
        assert ctx.knowledge_graph.graph.number_of_edges() == 0

    def test_describe_knowledge_graph_mentions_declared_relationships(self):
        ctx = bootstrap(database_url="sqlite:///:memory:")
        text = ctx.describe_knowledge_graph()
        assert "declared FK" in text or "inferred relationship" in text

    def test_seed_demo_creates_tables_for_empty_database(self, tmp_path: Path):
        db_path = tmp_path / "fresh.sqlite3"
        ctx = bootstrap(database_url=f"sqlite:///{db_path}", seed_demo=True)
        assert len(ctx.schema.tables) > 0

    def test_env_var_defaults_are_respected(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MAX_RETRIES", "5")
        monkeypatch.setenv("SELF_CONSISTENCY_SAMPLES", "3")
        ctx = bootstrap(database_url="sqlite:///:memory:")
        assert ctx.max_retries == 5
        assert ctx.self_consistency_samples == 3

    def test_explicit_kwargs_override_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MAX_RETRIES", "5")
        ctx = bootstrap(database_url="sqlite:///:memory:", max_retries=1)
        assert ctx.max_retries == 1
