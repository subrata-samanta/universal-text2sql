"""Tests for database connector and schema discovery."""

from __future__ import annotations

import pytest

from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import SchemaDiscovery
from universal_text2sql.utils.demo_data import seed_demo_database


@pytest.fixture(scope="module")
def connector() -> DatabaseConnector:
    """In-memory SQLite connector seeded with demo data."""
    conn = DatabaseConnector("sqlite:///:memory:")
    seed_demo_database(conn)
    return conn


@pytest.fixture(scope="module")
def schema(connector: DatabaseConnector):
    return SchemaDiscovery(connector).discover()


# ---------------------------------------------------------------------------
# Connector tests
# ---------------------------------------------------------------------------


class TestDatabaseConnector:
    def test_connection_alive(self, connector: DatabaseConnector):
        assert connector.test_connection() is True

    def test_db_type_detected(self, connector: DatabaseConnector):
        assert connector.db_type == "SQLite"

    def test_get_table_names(self, connector: DatabaseConnector):
        tables = connector.get_table_names()
        assert "customers" in tables
        assert "products" in tables
        assert "orders" in tables
        assert "order_items" in tables

    def test_execute_query_returns_dataframe(self, connector: DatabaseConnector):
        import pandas as pd

        df = connector.execute_query("SELECT COUNT(*) AS cnt FROM customers")
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["cnt"] == 10

    def test_execute_query_multiple_rows(self, connector: DatabaseConnector):
        df = connector.execute_query("SELECT * FROM products")
        assert len(df) == 10

    def test_get_column_info(self, connector: DatabaseConnector):
        cols = connector.get_column_info("customers")
        names = [c["name"] for c in cols]
        assert "customer_id" in names
        assert "email" in names

    def test_primary_key_detected(self, connector: DatabaseConnector):
        cols = connector.get_column_info("customers")
        pk_cols = [c["name"] for c in cols if c["primary_key"]]
        assert "customer_id" in pk_cols

    def test_get_sample_rows(self, connector: DatabaseConnector):
        df = connector.get_sample_rows("products", n=3)
        assert len(df) <= 3

    def test_bad_query_raises(self, connector: DatabaseConnector):
        with pytest.raises(Exception):
            connector.execute_query("SELECT * FROM nonexistent_table_xyz")


# ---------------------------------------------------------------------------
# Schema discovery tests
# ---------------------------------------------------------------------------


class TestSchemaDiscovery:
    def test_tables_discovered(self, schema):
        assert "customers" in schema.tables
        assert "orders" in schema.tables

    def test_column_metadata_populated(self, schema):
        table = schema.tables["customers"]
        col_names = [c.name for c in table.columns]
        assert "customer_id" in col_names
        assert "first_name" in col_names

    def test_sample_values_collected(self, schema):
        table = schema.tables["customers"]
        email_col = next(c for c in table.columns if c.name == "email")
        assert len(email_col.sample_values) > 0

    def test_row_count_correct(self, schema):
        assert schema.tables["customers"].row_count == 10
        assert schema.tables["products"].row_count == 10

    def test_ddl_generated(self, schema):
        ddl = schema.to_ddl()
        assert "CREATE TABLE customers" in ddl
        assert "customer_id" in ddl

    def test_subset_ddl(self, schema):
        ddl = schema.subset_ddl(["customers"])
        assert "CREATE TABLE customers" in ddl
        assert "CREATE TABLE products" not in ddl

    def test_get_relevant_tables_by_keyword(self, schema):
        relevant = schema.get_relevant_tables(["customer", "email"])
        assert "customers" in relevant

    def test_get_relevant_tables_fallback_all(self, schema):
        # Nonsense keywords → returns all tables
        relevant = schema.get_relevant_tables(["xyzabc999"])
        assert len(relevant) == len(schema.tables)

    def test_to_json_serialisable(self, schema):
        import json

        data = json.loads(schema.to_json())
        assert "tables" in data
        assert "customers" in data["tables"]

    def test_signature_hash_is_deterministic(self, schema):
        assert schema.signature_hash() == schema.signature_hash()
        assert len(schema.signature_hash()) == 16

    def test_signature_hash_changes_with_schema(self, schema, connector):
        from universal_text2sql.database.schema import ColumnMetadata, TableMetadata

        other = SchemaDiscovery(connector).discover()
        other.tables["extra_table"] = TableMetadata(
            name="extra_table",
            columns=[ColumnMetadata(name="id", data_type="INTEGER", nullable=False, primary_key=True)],
        )
        assert schema.signature_hash() != other.signature_hash()
