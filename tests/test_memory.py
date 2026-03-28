"""Tests for the agent memory (RL-inspired few-shot store)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from universal_text2sql.agent.memory import QueryMemory


@pytest.fixture
def tmp_memory(tmp_path: Path) -> QueryMemory:
    """Return a fresh memory store backed by a temp file."""
    return QueryMemory(persist_path=tmp_path / "memory.json", enabled=True)


class TestQueryMemory:
    def test_add_and_retrieve(self, tmp_memory: QueryMemory):
        tmp_memory.add("How many customers?", "SELECT COUNT(*) FROM customers")
        results = tmp_memory.get_similar("How many customers are there?", top_k=3)
        assert len(results) >= 1
        assert results[0]["sql"] == "SELECT COUNT(*) FROM customers"

    def test_reward_is_higher_for_first_try(self, tmp_memory: QueryMemory):
        tmp_memory.add("Count products", "SELECT COUNT(*) FROM products", retry_count=0)
        tmp_memory.add("Count items", "SELECT COUNT(*) FROM order_items", retry_count=2)
        # Both have "count" keyword – first-try entry should have higher reward
        entry_no_retry = next(
            e for e in tmp_memory._entries if "products" in e.sql
        )
        entry_with_retry = next(
            e for e in tmp_memory._entries if "order_items" in e.sql
        )
        assert entry_no_retry.reward > entry_with_retry.reward

    def test_persisted_to_disk(self, tmp_memory: QueryMemory, tmp_path: Path):
        tmp_memory.add("Test question", "SELECT 1", retry_count=0)
        path = tmp_path / "memory.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert any(e["sql"] == "SELECT 1" for e in data)

    def test_load_from_disk(self, tmp_path: Path):
        path = tmp_path / "mem2.json"
        path.write_text(
            json.dumps(
                [{"question": "loaded question", "sql": "SELECT 2", "success": True, "reward": 1.0}]
            )
        )
        mem = QueryMemory(persist_path=path, enabled=True)
        results = mem.get_similar("loaded question")
        assert len(results) >= 1

    def test_disabled_memory_returns_empty(self):
        mem = QueryMemory(enabled=False)
        mem.add("Will not be stored", "SELECT 1")
        assert mem.get_similar("Will not be stored") == []

    def test_top_k_respected(self, tmp_memory: QueryMemory):
        for i in range(10):
            tmp_memory.add(f"Count table {i}", f"SELECT COUNT(*) FROM t{i}")
        results = tmp_memory.get_similar("Count all tables", top_k=3)
        assert len(results) <= 3
