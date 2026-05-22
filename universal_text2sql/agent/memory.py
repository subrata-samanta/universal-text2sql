"""RL-inspired query memory for few-shot learning.

Successful queries are stored and replayed as few-shot examples for future
prompts.  A simple reward signal boosts queries that were accepted on the
first try and diminishes those that required multiple retries.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock

from universal_text2sql.agent.state import QueryMemoryEntry

logger = logging.getLogger(__name__)

_MEMORY_FILE = Path(os.getenv("QUERY_MEMORY_PATH", "./query_memory.json"))
_MAX_ENTRIES = 200


class QueryMemory:
    """Thread-safe, file-persisted query memory store."""

    def __init__(
        self,
        persist_path: Path | None = None,
        max_entries: int = _MAX_ENTRIES,
        enabled: bool = True,
    ) -> None:
        self._path = persist_path or _MEMORY_FILE
        self._max_entries = max_entries
        self._enabled = enabled
        self._lock = Lock()
        self._entries: list[QueryMemoryEntry] = []
        if self._enabled:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, question: str, sql: str, retry_count: int = 0) -> None:
        """Record a successful query.

        The reward is computed as ``1.0 / (1 + retry_count)`` so that queries
        solved on the first try get the highest weight and are surfaced most
        often as few-shot examples.
        """
        if not self._enabled:
            return
        reward = 1.0 / (1 + retry_count)
        entry = QueryMemoryEntry(question=question, sql=sql, reward=reward)
        with self._lock:
            self._entries.append(entry)
            # Keep the most rewarding entries when the store is full
            if len(self._entries) > self._max_entries:
                self._entries.sort(key=lambda e: e.reward, reverse=True)
                self._entries = self._entries[: self._max_entries]
            self._persist()
        logger.debug("Added query memory entry (reward=%.2f): %s", reward, question[:60])

    def get_similar(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the *top_k* most relevant past queries for the given question.

        Relevance is approximated by word-overlap score weighted by the stored
        reward, keeping this dependency-free.
        """
        if not self._enabled or not self._entries:
            return []
        question_tokens = set(question.lower().split())
        scored: list[tuple[float, QueryMemoryEntry]] = []
        for entry in self._entries:
            entry_tokens = set(entry.question.lower().split())
            overlap = len(question_tokens & entry_tokens)
            score = overlap * entry.reward
            if overlap > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"question": e.question, "sql": e.sql}
            for _, e in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._entries = [QueryMemoryEntry(**item) for item in data]
            logger.info("Loaded %d query memory entries from %s", len(self._entries), self._path)
        except Exception as exc:
            logger.warning("Could not load query memory: %s", exc)

    def _persist(self) -> None:
        try:
            self._path.write_text(
                json.dumps([e.model_dump() for e in self._entries], indent=2)
            )
        except Exception as exc:
            logger.warning("Could not persist query memory: %s", exc)
