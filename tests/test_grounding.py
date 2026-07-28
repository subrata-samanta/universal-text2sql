"""Tests for value/entity grounding (knowledge/grounding.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import (
    ColumnMetadata,
    DatabaseSchema,
    SchemaDiscovery,
    TableMetadata,
)
from universal_text2sql.knowledge.grounding import ValueGroundingIndex
from universal_text2sql.utils.demo_data import seed_demo_database


@pytest.fixture(scope="module")
def connector() -> DatabaseConnector:
    conn = DatabaseConnector("sqlite:///:memory:")
    seed_demo_database(conn)
    return conn


@pytest.fixture(scope="module")
def schema(connector: DatabaseConnector) -> DatabaseSchema:
    return SchemaDiscovery(connector).discover()


@pytest.fixture()
def index(schema: DatabaseSchema, connector: DatabaseConnector, tmp_path: Path) -> ValueGroundingIndex:
    return ValueGroundingIndex.build(schema, connector, cache_dir=tmp_path)


class TestBuild:
    def test_profiles_expected_categorical_columns(self, index: ValueGroundingIndex):
        profiled = {(p.table, p.column) for p in index.profiles}
        assert ("customers", "country") in profiled
        assert ("orders", "status") in profiled
        assert ("products", "category") in profiled

    def test_excludes_high_cardinality_columns(self, index: ValueGroundingIndex):
        profiled = {(p.table, p.column) for p in index.profiles}
        assert ("customers", "email") not in profiled
        assert ("customers", "first_name") not in profiled
        assert ("orders", "order_id") not in profiled  # numeric, excluded by type filter too

    def test_excludes_primary_keys(self, index: ValueGroundingIndex):
        profiled = {(p.table, p.column) for p in index.profiles}
        assert ("products", "product_id") not in profiled

    def test_profile_values_are_correct(self, index: ValueGroundingIndex):
        country_profile = next(p for p in index.profiles if p.column == "country")
        assert set(country_profile.distinct_values) == {
            "USA", "UK", "Canada", "Germany", "France", "Japan", "Australia",
        }

    def test_skips_tables_above_row_ceiling(
        self, connector: DatabaseConnector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        schema = DatabaseSchema(db_type="SQLite")
        schema.tables["customers"] = TableMetadata(
            name="customers",
            row_count=999_999_999,
            columns=[
                ColumnMetadata(name="country", data_type="TEXT", nullable=False, primary_key=False)
            ],
        )
        index = ValueGroundingIndex.build(schema, connector, cache_dir=tmp_path)
        assert index.profiles == []

    def test_caches_to_disk(self, schema: DatabaseSchema, connector: DatabaseConnector, tmp_path: Path):
        ValueGroundingIndex.build(schema, connector, cache_dir=tmp_path)
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

    def test_second_build_uses_cache(
        self, schema: DatabaseSchema, connector: DatabaseConnector, tmp_path: Path
    ):
        index1 = ValueGroundingIndex.build(schema, connector, cache_dir=tmp_path)
        index2 = ValueGroundingIndex.build(schema, connector, cache_dir=tmp_path)
        assert {(p.table, p.column, tuple(p.distinct_values)) for p in index1.profiles} == {
            (p.table, p.column, tuple(p.distinct_values)) for p in index2.profiles
        }


class TestMatch:
    def test_matches_country(self, index: ValueGroundingIndex):
        matches = index.match("How many customers are from usa?", ["customers"])
        assert ("customers", "country", "USA") in matches

    def test_matches_category(self, index: ValueGroundingIndex):
        matches = index.match("products in the electronics category", ["products"])
        assert ("products", "category", "Electronics") in matches

    def test_matches_status(self, index: ValueGroundingIndex):
        matches = index.match("orders with status completed", ["orders"])
        assert ("orders", "status", "completed") in matches

    def test_no_match_for_unrelated_question(self, index: ValueGroundingIndex):
        matches = index.match("xyz totally unrelated qqq", ["customers", "orders", "products"])
        assert matches == []

    def test_stopwords_do_not_cause_false_positives(self, index: ValueGroundingIndex):
        # "many" (a stopword) is a substring of "Germany" -- must not match.
        matches = index.match("How many customers are there?", ["customers"])
        assert ("customers", "country", "Germany") not in matches

    def test_respects_table_filter(self, index: ValueGroundingIndex):
        matches = index.match("usa", ["orders"])  # country isn't in orders
        assert matches == []

    def test_top_k_limits_results(self, index: ValueGroundingIndex):
        matches = index.match(
            "usa uk canada germany france japan australia", ["customers"], top_k=2
        )
        assert len(matches) <= 2

    def test_empty_question_returns_no_matches(self, index: ValueGroundingIndex):
        assert index.match("", ["customers"]) == []


class TestHintsBlock:
    def test_empty_matches_returns_empty_string(self, index: ValueGroundingIndex):
        assert index.hints_block([]) == ""

    def test_renders_matches(self, index: ValueGroundingIndex):
        block = index.hints_block([("customers", "country", "USA")])
        assert "customers.country" in block
        assert "USA" in block
