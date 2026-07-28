"""Automatic metadata / business-glossary generation.

A recurring theme in recent text-to-SQL work (CHESS's "context retrieval",
the M-Schema representation used in several 2024 Spider/BIRD leaderboard
entries, and enterprise semantic-layer tools) is that raw DDL is a poor
proxy for what a column *means*. ``t.st`` tells an LLM nothing; "order
status, one of pending/shipped/completed/cancelled" does.

Since this agent must work against **any** unseen database with zero
manual configuration, it generates that semantic layer itself:

1. Heuristic pass (always runs, no LLM needed): expands identifier
   naming conventions (``snake_case`` / ``camelCase``), infers role from
   type + key constraints + sample values.
2. LLM pass (optional, one call per table): asks the model to write a
   short business description of the table and each column, plus
   synonyms end users might use instead of the raw column name (helps
   schema linking match "revenue" to a column named ``total_amount``).

Results are cached to disk keyed by a hash of the table/column
signature, so re-running against the same database is free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from universal_text2sql.database.schema import DatabaseSchema
from universal_text2sql.llm.base import LLMRunnable

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.getenv("METADATA_CACHE_DIR", "./.metadata_cache"))

_WORD_SPLIT_RE = re.compile(r"[_\-]+|(?<=[a-z0-9])(?=[A-Z])")


def _humanize(identifier: str) -> str:
    words = [w for w in _WORD_SPLIT_RE.split(identifier) if w]
    return " ".join(w.lower() for w in words)


def _schema_signature(schema: DatabaseSchema) -> str:
    parts = [schema.db_type]
    for table_name in sorted(schema.tables):
        meta = schema.tables[table_name]
        col_sig = ",".join(f"{c.name}:{c.data_type}" for c in meta.columns)
        parts.append(f"{table_name}[{col_sig}]")
    return "|".join(parts)


def _schema_hash(schema: DatabaseSchema) -> str:
    return hashlib.sha256(_schema_signature(schema).encode("utf-8")).hexdigest()[:16]


class MetadataEnricher:
    """Auto-generates and caches table/column business metadata."""

    def __init__(
        self,
        llm: LLMRunnable | None = None,
        cache_dir: Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.llm = llm
        self.cache_dir = cache_dir or _CACHE_DIR
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, schema: DatabaseSchema) -> DatabaseSchema:
        """Populate ``description``/``business_meaning``/``synonyms`` in place.

        Always applies the heuristic pass. Additionally applies an LLM pass
        (with disk caching) when ``self.llm`` is set and ``self.enabled``.
        """
        for meta in schema.tables.values():
            self._apply_heuristics(meta)

        if self.llm is not None and self.enabled:
            cached = self._load_cache(schema)
            if cached is not None:
                self._apply_llm_metadata(schema, cached)
            else:
                generated = self._generate_with_llm(schema)
                if generated:
                    self._apply_llm_metadata(schema, generated)
                    self._save_cache(schema, generated)

        return schema

    def glossary_block(self, schema: DatabaseSchema, tables: list[str] | None = None) -> str:
        """Render a compact business-glossary block for prompt injection."""
        lines = ["## Business Glossary (auto-generated)"]
        table_names = tables or list(schema.tables.keys())
        for table_name in table_names:
            meta = schema.tables.get(table_name)
            if meta is None:
                continue
            header = f"- {table_name}"
            if meta.description:
                header += f": {meta.description}"
            if meta.synonyms:
                header += f" (aka: {', '.join(meta.synonyms)})"
            lines.append(header)
            for col in meta.columns:
                if col.business_meaning:
                    syn = f" (aka: {', '.join(col.synonyms)})" if col.synonyms else ""
                    lines.append(f"    - {col.name}: {col.business_meaning}{syn}")
        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------------
    # Heuristic pass (no LLM, always available, fully deterministic)
    # ------------------------------------------------------------------

    def _apply_heuristics(self, meta) -> None:  # noqa: ANN001 - TableMetadata
        if not meta.description:
            meta.description = f"Table storing {_humanize(meta.name)} records."

        for col in meta.columns:
            if col.business_meaning:
                continue
            col.business_meaning = self._heuristic_column_meaning(meta.name, col)

    @staticmethod
    def _heuristic_column_meaning(table_name: str, col) -> str:  # noqa: ANN001
        human = _humanize(col.name)
        if col.primary_key:
            return f"Unique identifier for each {_humanize(table_name).rstrip('s')} row (primary key)."
        if col.name.lower().endswith("_id") or col.name.lower().endswith("id"):
            ref = _humanize(col.name.replace("_id", "").replace("Id", ""))
            if ref:
                return f"Reference to a {ref} record (foreign key)."
        if col.sample_values:
            sample_str = ", ".join(str(v) for v in col.sample_values[:3])
            return f"{human.capitalize()}. Example values: {sample_str}."
        return f"{human.capitalize()}."

    # ------------------------------------------------------------------
    # LLM pass
    # ------------------------------------------------------------------

    def _generate_with_llm(self, schema: DatabaseSchema) -> dict[str, Any] | None:
        from universal_text2sql.prompts.templates import METADATA_GENERATION_PROMPT

        result: dict[str, Any] = {}
        chain = METADATA_GENERATION_PROMPT | self.llm

        for table_name, meta in schema.tables.items():
            col_lines = "\n".join(
                f"  - {c.name} ({c.data_type}"
                f"{', primary key' if c.primary_key else ''})"
                f": samples={c.sample_values[:3]}"
                for c in meta.columns
            )
            try:
                response = chain.invoke(
                    {"table_name": table_name, "row_count": meta.row_count, "columns": col_lines}
                )
                raw = response.content if hasattr(response, "content") else str(response)
                parsed = self._parse_json_response(raw)
                if parsed:
                    result[table_name] = parsed
            except Exception as exc:  # pragma: no cover - network/LLM failures
                logger.warning("Metadata generation failed for table %s: %s", table_name, exc)

        return result or None

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    def _apply_llm_metadata(self, schema: DatabaseSchema, data: dict[str, Any]) -> None:
        for table_name, table_data in data.items():
            meta = schema.tables.get(table_name)
            if meta is None or not isinstance(table_data, dict):
                continue
            if table_data.get("description"):
                meta.description = table_data["description"]
            if table_data.get("synonyms"):
                meta.synonyms = list(table_data["synonyms"])
            col_data = table_data.get("columns", {})
            if not isinstance(col_data, dict):
                continue
            for col in meta.columns:
                info = col_data.get(col.name)
                if not isinstance(info, dict):
                    continue
                if info.get("meaning"):
                    col.business_meaning = info["meaning"]
                if info.get("synonyms"):
                    col.synonyms = list(info["synonyms"])

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, schema: DatabaseSchema) -> Path:
        return self.cache_dir / f"{_schema_hash(schema)}.json"

    def _load_cache(self, schema: DatabaseSchema) -> dict[str, Any] | None:
        path = self._cache_path(schema)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Could not read metadata cache %s: %s", path, exc)
            return None

    def _save_cache(self, schema: DatabaseSchema, data: dict[str, Any]) -> None:
        path = self._cache_path(schema)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Could not write metadata cache %s: %s", path, exc)
