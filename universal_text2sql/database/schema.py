"""Automatic schema discovery and metadata generation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from universal_text2sql.database.connector import DatabaseConnector

logger = logging.getLogger(__name__)


@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    nullable: bool
    primary_key: bool
    sample_values: list[Any] = field(default_factory=list)
    # Populated by knowledge.metadata.MetadataEnricher (LLM- or heuristic-generated)
    business_meaning: str = ""
    synonyms: list[str] = field(default_factory=list)


@dataclass
class TableMetadata:
    name: str
    columns: list[ColumnMetadata] = field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    description: str = ""
    # Populated by knowledge.metadata.MetadataEnricher
    synonyms: list[str] = field(default_factory=list)


@dataclass
class DatabaseSchema:
    db_type: str
    tables: dict[str, TableMetadata] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_type": self.db_type,
            "tables": {name: asdict(meta) for name, meta in self.tables.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_ddl(self) -> str:
        """Generate a CREATE TABLE DDL string for every table (for prompt context)."""
        lines: list[str] = []
        for table_name, meta in self.tables.items():
            col_defs: list[str] = []
            for col in meta.columns:
                parts = [f"    {col.name} {col.data_type}"]
                if col.primary_key:
                    parts.append("PRIMARY KEY")
                if not col.nullable:
                    parts.append("NOT NULL")
                col_defs.append(" ".join(parts))

            # Add FK references
            for fk in meta.foreign_keys:
                constrained = ", ".join(fk.get("constrained_columns", []))
                referred_table = fk.get("referred_table", "")
                referred_cols = ", ".join(fk.get("referred_columns", []))
                col_defs.append(
                    f"    FOREIGN KEY ({constrained}) REFERENCES {referred_table}({referred_cols})"
                )

            lines.append(f"CREATE TABLE {table_name} (")
            lines.append(",\n".join(col_defs))
            lines.append(");\n")
        return "\n".join(lines)

    def get_relevant_tables(self, keywords: list[str]) -> list[str]:
        """Return table names that are most relevant to the given keywords.

        A simple keyword-overlap scoring is used so that the LLM only sees the
        tables that matter, keeping prompts short.
        """
        keywords_lower = [k.lower() for k in keywords]
        scores: dict[str, int] = {}

        for table_name, meta in self.tables.items():
            score = 0
            # Table name match
            if any(kw in table_name.lower() for kw in keywords_lower):
                score += 3
            # Column name matches
            for col in meta.columns:
                if any(kw in col.name.lower() for kw in keywords_lower):
                    score += 1
            scores[table_name] = score

        # Sort by score descending; if no matches at all, return all tables
        sorted_tables = sorted(scores, key=lambda t: scores[t], reverse=True)
        relevant = [t for t in sorted_tables if scores[t] > 0]
        return relevant if relevant else sorted_tables

    def _table_document(self, meta: TableMetadata) -> str:
        """Build a text blob describing a table for semantic retrieval."""
        parts = [meta.name, meta.description, " ".join(meta.synonyms)]
        for col in meta.columns:
            parts.append(col.name)
            parts.append(col.business_meaning)
            parts.append(" ".join(col.synonyms))
            parts.extend(str(v) for v in col.sample_values[:3])
        return " ".join(p for p in parts if p)

    def get_relevant_tables_semantic(
        self, question: str, top_k: int | None = None, min_score: float = 0.0
    ) -> list[str]:
        """Rank tables by TF-IDF cosine similarity against the question.

        Falls back to :meth:`get_relevant_tables` (keyword overlap) when the
        semantic index cannot separate any tables (e.g. an empty schema, or
        a question that shares no vocabulary at all with any table/column
        name, description, or sample value).
        """
        from universal_text2sql.retrieval.semantic import SemanticIndex

        documents = {name: self._table_document(meta) for name, meta in self.tables.items()}
        index = SemanticIndex.from_documents(documents)
        ranked = index.rank(question, top_k=top_k)
        relevant = [t for t, score in ranked if score > min_score]

        if not relevant:
            keywords = [w for w in question.lower().split() if len(w) > 3]
            return self.get_relevant_tables(keywords)

        return relevant

    def subset_ddl(self, table_names: list[str]) -> str:
        """Return DDL for a subset of tables only."""
        original = self.tables
        self.tables = {k: v for k, v in original.items() if k in table_names}
        ddl = self.to_ddl()
        self.tables = original
        return ddl


class SchemaDiscovery:
    """Automatically discovers and builds schema metadata from a live database."""

    def __init__(self, connector: DatabaseConnector, sample_rows: int = 3) -> None:
        self.connector = connector
        self.sample_rows = sample_rows

    def discover(self) -> DatabaseSchema:
        """Introspect the database and return a fully-populated :class:`DatabaseSchema`."""
        schema = DatabaseSchema(db_type=self.connector.db_type)
        table_names = self.connector.get_table_names()

        if not table_names:
            logger.warning("No tables found in the database.")
            return schema

        logger.info("Discovering schema for tables: %s", table_names)

        for table_name in table_names:
            meta = self._discover_table(table_name)
            schema.tables[table_name] = meta

        return schema

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_table(self, table_name: str) -> TableMetadata:
        columns_raw = self.connector.get_column_info(table_name)
        fks = self.connector.get_foreign_keys(table_name)
        row_count = self._get_row_count(table_name)

        # Fetch sample data for value context
        try:
            sample_df = self.connector.get_sample_rows(table_name, self.sample_rows)
        except Exception:
            sample_df = None

        columns: list[ColumnMetadata] = []
        for col_info in columns_raw:
            sample_values: list[Any] = []
            if sample_df is not None and col_info["name"] in sample_df.columns:
                sample_values = [
                    v
                    for v in sample_df[col_info["name"]].tolist()
                    if v is not None
                ]
            columns.append(
                ColumnMetadata(
                    name=col_info["name"],
                    data_type=col_info["type"],
                    nullable=col_info["nullable"],
                    primary_key=col_info["primary_key"],
                    sample_values=sample_values,
                )
            )

        return TableMetadata(
            name=table_name,
            columns=columns,
            foreign_keys=fks,
            row_count=row_count,
        )

    def _get_row_count(self, table_name: str) -> int:
        try:
            # table_name is always sourced from get_table_names() (inspector),
            # so it is safe to interpolate; cast to int to avoid type confusion.
            df = self.connector.execute_query(
                f"SELECT COUNT(*) AS cnt FROM {table_name}"  # noqa: S608
            )
            return int(df.iloc[0]["cnt"])
        except Exception:
            return 0
