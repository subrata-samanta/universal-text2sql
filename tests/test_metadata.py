"""Tests for automatic metadata / business-glossary generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import SchemaDiscovery
from universal_text2sql.knowledge.metadata import MetadataEnricher
from universal_text2sql.utils.demo_data import seed_demo_database


@pytest.fixture()
def connector() -> DatabaseConnector:
    conn = DatabaseConnector("sqlite:///:memory:")
    seed_demo_database(conn)
    return conn


@pytest.fixture()
def schema(connector: DatabaseConnector):
    return SchemaDiscovery(connector).discover()


class TestHeuristicEnrichment:
    def test_no_llm_still_populates_descriptions(self, schema):
        enricher = MetadataEnricher(llm=None, enabled=True)
        enricher.enrich(schema)

        assert schema.tables["customers"].description
        for col in schema.tables["customers"].columns:
            assert col.business_meaning

    def test_primary_key_meaning_mentions_primary_key(self, schema):
        enricher = MetadataEnricher(llm=None)
        enricher.enrich(schema)
        pk_col = next(c for c in schema.tables["customers"].columns if c.primary_key)
        assert "primary key" in pk_col.business_meaning.lower()

    def test_foreign_key_style_column_meaning(self, schema):
        enricher = MetadataEnricher(llm=None)
        enricher.enrich(schema)
        fk_col = next(c for c in schema.tables["orders"].columns if c.name == "customer_id")
        assert "foreign key" in fk_col.business_meaning.lower() or "reference" in fk_col.business_meaning.lower()

    def test_glossary_block_contains_all_tables(self, schema):
        enricher = MetadataEnricher(llm=None)
        enricher.enrich(schema)
        block = enricher.glossary_block(schema)
        for table_name in schema.tables:
            assert table_name in block

    def test_glossary_block_can_be_filtered(self, schema):
        enricher = MetadataEnricher(llm=None)
        enricher.enrich(schema)
        block = enricher.glossary_block(schema, tables=["customers"])
        assert "customers" in block
        assert "products" not in block


class TestLLMEnrichment:
    def _mock_llm(self, response_json: dict):
        # _generate_with_llm calls the chain once per table and expects a flat
        # JSON object describing *that* table directly (not wrapped by name).
        return RunnableLambda(lambda _: AIMessage(content=json.dumps(response_json)))

    def test_llm_description_applied_and_cached(self, schema, tmp_path: Path):
        response = {
            "description": "People who purchase products from the store.",
            "synonyms": ["clients", "buyers"],
            "columns": {
                "customer_id": {"meaning": "Unique customer identifier.", "synonyms": ["client id"]},
                "first_name": {"meaning": "Customer's given name.", "synonyms": []},
                "last_name": {"meaning": "Customer's family name.", "synonyms": []},
                "email": {"meaning": "Customer's contact email address.", "synonyms": []},
                "country": {"meaning": "Customer's home country.", "synonyms": []},
                "created_at": {"meaning": "Signup date.", "synonyms": []},
            },
        }
        llm = self._mock_llm(response)
        enricher = MetadataEnricher(llm=llm, cache_dir=tmp_path, enabled=True)
        enricher.enrich(schema)

        assert schema.tables["customers"].description == "People who purchase products from the store."
        assert "clients" in schema.tables["customers"].synonyms
        cid = next(c for c in schema.tables["customers"].columns if c.name == "customer_id")
        assert cid.business_meaning == "Unique customer identifier."

        # A cache file should have been written for this schema signature.
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

    def test_cache_avoids_second_llm_call(self, schema, tmp_path: Path):
        call_count = {"n": 0}

        def _llm(_input):
            call_count["n"] += 1
            return AIMessage(
                content=json.dumps(
                    {
                        "description": "desc",
                        "synonyms": [],
                        "columns": {
                            c.name: {"meaning": "m", "synonyms": []}
                            for c in schema.tables["customers"].columns
                        },
                    }
                )
            )

        llm = RunnableLambda(_llm)
        enricher1 = MetadataEnricher(llm=llm, cache_dir=tmp_path, enabled=True)
        enricher1.enrich(schema)
        first_call_count = call_count["n"]
        assert first_call_count > 0

        # Re-enrich a fresh schema copy with a new enricher instance sharing the cache dir.
        connector = DatabaseConnector("sqlite:///:memory:")
        seed_demo_database(connector)
        schema2 = SchemaDiscovery(connector).discover()
        enricher2 = MetadataEnricher(llm=llm, cache_dir=tmp_path, enabled=True)
        enricher2.enrich(schema2)

        assert call_count["n"] == first_call_count  # no additional LLM calls
        assert schema2.tables["customers"].description == "desc"

    def test_malformed_llm_response_falls_back_gracefully(self, schema, tmp_path: Path):
        llm = RunnableLambda(lambda _: AIMessage(content="not json at all"))
        enricher = MetadataEnricher(llm=llm, cache_dir=tmp_path, enabled=True)
        # Should not raise, and heuristic descriptions still populated.
        enricher.enrich(schema)
        assert schema.tables["customers"].description

    def test_disabled_enrichment_skips_llm(self, schema, tmp_path: Path):
        call_count = {"n": 0}

        def _llm(_input):
            call_count["n"] += 1
            return AIMessage(content="{}")

        llm = RunnableLambda(_llm)
        enricher = MetadataEnricher(llm=llm, cache_dir=tmp_path, enabled=False)
        enricher.enrich(schema)
        assert call_count["n"] == 0
        # Heuristics still run even when LLM enrichment is disabled.
        assert schema.tables["customers"].description
