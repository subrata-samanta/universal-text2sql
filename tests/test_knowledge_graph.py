"""Tests for the auto-built schema knowledge graph."""

from __future__ import annotations

import pytest

from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import (
    ColumnMetadata,
    DatabaseSchema,
    SchemaDiscovery,
    TableMetadata,
)
from universal_text2sql.knowledge.graph import SchemaKnowledgeGraph
from universal_text2sql.utils.demo_data import seed_demo_database


@pytest.fixture(scope="module")
def connector() -> DatabaseConnector:
    conn = DatabaseConnector("sqlite:///:memory:")
    seed_demo_database(conn)
    return conn


@pytest.fixture(scope="module")
def schema(connector: DatabaseConnector) -> DatabaseSchema:
    return SchemaDiscovery(connector).discover()


@pytest.fixture(scope="module")
def kg(schema: DatabaseSchema) -> SchemaKnowledgeGraph:
    return SchemaKnowledgeGraph.build(schema)


class TestDeclaredForeignKeys:
    def test_declared_fk_edges_present(self, kg: SchemaKnowledgeGraph):
        hops = kg.find_join_path("orders", "customers")
        assert len(hops) == 1
        assert hops[0].kind == "foreign_key"
        assert hops[0].left_table == "orders"
        assert hops[0].right_table == "customers"
        assert hops[0].left_column == "customer_id"
        assert hops[0].right_column == "customer_id"

    def test_multi_hop_join_path(self, kg: SchemaKnowledgeGraph):
        # order_items has no direct FK to customers; must go through orders.
        hops = kg.find_join_path("order_items", "customers")
        assert len(hops) == 2
        touched_tables = {hops[0].left_table, hops[0].right_table, hops[1].left_table, hops[1].right_table}
        assert {"order_items", "orders", "customers"} <= touched_tables

    def test_no_path_between_unrelated_tables(self, kg: SchemaKnowledgeGraph):
        assert kg.find_join_path("nonexistent_a", "nonexistent_b") == []

    def test_same_table_has_no_path(self, kg: SchemaKnowledgeGraph):
        assert kg.find_join_path("customers", "customers") == []


class TestJoinPathsForTables:
    def test_connects_all_requested_tables(self, kg: SchemaKnowledgeGraph):
        hops = kg.get_join_paths_for_tables(["customers", "order_items", "products"])
        touched = set()
        for hop in hops:
            touched.add(hop.left_table)
            touched.add(hop.right_table)
        assert {"customers", "order_items", "products"} <= touched

    def test_single_table_returns_empty(self, kg: SchemaKnowledgeGraph):
        assert kg.get_join_paths_for_tables(["customers"]) == []


class TestDescribe:
    def test_describe_mentions_relationships(self, kg: SchemaKnowledgeGraph):
        text = kg.describe()
        assert "orders.customer_id = customers.customer_id" in text or (
            "customers.customer_id = orders.customer_id" in text
        )

    def test_describe_filters_by_tables(self, kg: SchemaKnowledgeGraph):
        text = kg.describe(["customers"])
        assert "products" not in text.lower().replace("no explicit", "")

    def test_describe_empty_graph(self):
        empty_kg = SchemaKnowledgeGraph()
        assert "No explicit or inferred relationships" in empty_kg.describe()


class TestInferredForeignKeys:
    def test_inferred_fk_without_declared_constraint(self):
        # Build a schema with no FK metadata but a naming-convention link.
        schema = DatabaseSchema(db_type="SQLite")
        schema.tables["customers"] = TableMetadata(
            name="customers",
            columns=[ColumnMetadata(name="customer_id", data_type="INTEGER", nullable=False, primary_key=True)],
        )
        schema.tables["orders"] = TableMetadata(
            name="orders",
            columns=[
                ColumnMetadata(name="order_id", data_type="INTEGER", nullable=False, primary_key=True),
                ColumnMetadata(name="customer_id", data_type="INTEGER", nullable=False, primary_key=False),
            ],
        )
        kg = SchemaKnowledgeGraph.build(schema)
        hops = kg.find_join_path("orders", "customers")
        assert len(hops) == 1
        assert hops[0].kind == "inferred_fk"


class TestSerialization:
    def test_to_dict_has_nodes_and_edges(self, kg: SchemaKnowledgeGraph):
        data = kg.to_dict()
        assert "nodes" in data and "edges" in data
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0

    def test_to_graphviz_is_valid_dot_skeleton(self, kg: SchemaKnowledgeGraph):
        dot = kg.to_graphviz()
        assert dot.startswith("digraph schema {")
        assert dot.strip().endswith("}")
