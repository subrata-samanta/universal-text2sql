"""Schema knowledge graph.

Recent text-to-SQL research (e.g. RESDSQL's ranking-enhanced schema
encoding, CHESS's entity/context retrieval, and graph-augmented schema
linking work such as RASAT/ SADGA) treats the database schema as a graph
of tables and columns rather than a flat list of ``CREATE TABLE``
statements. Explicit foreign keys only capture *declared* relationships;
real-world databases (especially ones exported without constraints, e.g.
CSV-backed SQLite dumps) are full of *undeclared* relationships that a
human would infer from naming conventions alone.

This module builds a directed graph over the discovered schema:

* ``table:<name>``  – one node per table
* ``column:<table>.<name>`` – one node per column

Edge kinds:

* ``has_column``   – table -> column (structural)
* ``foreign_key``  – column -> column, taken from the database's declared
  FK constraints
* ``inferred_fk``  – column -> column, recovered from naming conventions
  when no explicit constraint exists (e.g. ``orders.customer_id`` ->
  ``customers.customer_id`` / ``customers.id``)
* ``semantic_sibling`` – column <-> column, same/near-identical column
  name in two different tables (candidate join keys / duplicated
  concepts, e.g. two ``email`` columns)

The graph is then used for two things an LLM struggles with on wide
schemas: (1) telling it *which* columns to join on for tables that are
not directly connected (multi-hop join paths), and (2) pruning the
prompt to only the tables/columns relevant to a question.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from universal_text2sql.database.schema import DatabaseSchema

logger = logging.getLogger(__name__)

_ID_SUFFIX_RE = re.compile(r"_id$|Id$", re.IGNORECASE)


def _singularize(word: str) -> str:
    """Very small heuristic singularizer, good enough for table-name matching."""
    lower = word.lower()
    if lower.endswith("ies") and len(lower) > 3:
        return lower[:-3] + "y"
    if lower.endswith("ses") or lower.endswith("xes") or lower.endswith("ches"):
        return lower[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return lower[:-1]
    return lower


def _table_node(table: str) -> str:
    return f"table:{table}"


def _column_node(table: str, column: str) -> str:
    return f"column:{table}.{column}"


@dataclass
class JoinHop:
    """A single ``A.col = B.col`` join condition."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    kind: str  # "foreign_key" | "inferred_fk"

    def as_sql(self) -> str:
        return (
            f"{self.left_table}.{self.left_column} = "
            f"{self.right_table}.{self.right_column}"
        )


@dataclass
class SchemaKnowledgeGraph:
    """Directed graph of tables/columns plus join-path reasoning."""

    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    # table-level projection used for join-path search
    _table_graph: nx.MultiGraph = field(default_factory=nx.MultiGraph)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, schema: DatabaseSchema) -> "SchemaKnowledgeGraph":
        kg = cls()
        kg._add_structural_nodes(schema)
        kg._add_explicit_foreign_keys(schema)
        kg._add_inferred_foreign_keys(schema)
        kg._add_semantic_siblings(schema)
        logger.info(
            "Knowledge graph built: %d nodes, %d edges (%d table-level joins)",
            kg.graph.number_of_nodes(),
            kg.graph.number_of_edges(),
            kg._table_graph.number_of_edges(),
        )
        return kg

    def _add_structural_nodes(self, schema: DatabaseSchema) -> None:
        for table_name, meta in schema.tables.items():
            self.graph.add_node(_table_node(table_name), kind="table", name=table_name)
            self._table_graph.add_node(table_name)
            for col in meta.columns:
                self.graph.add_node(
                    _column_node(table_name, col.name),
                    kind="column",
                    table=table_name,
                    name=col.name,
                    data_type=col.data_type,
                    primary_key=col.primary_key,
                )
                self.graph.add_edge(
                    _table_node(table_name),
                    _column_node(table_name, col.name),
                    kind="has_column",
                )

    def _add_explicit_foreign_keys(self, schema: DatabaseSchema) -> None:
        for table_name, meta in schema.tables.items():
            for fk in meta.foreign_keys:
                referred_table = fk.get("referred_table")
                constrained_cols = fk.get("constrained_columns", [])
                referred_cols = fk.get("referred_columns", [])
                if not referred_table or referred_table not in schema.tables:
                    continue
                for c_col, r_col in zip(constrained_cols, referred_cols):
                    self._add_join_edge(
                        table_name, c_col, referred_table, r_col, kind="foreign_key"
                    )

    def _add_inferred_foreign_keys(self, schema: DatabaseSchema) -> None:
        """Recover undeclared FK relationships from naming conventions.

        Handles the two dominant conventions seen across real schemas:
        ``customer_id`` -> ``customers.customer_id`` or ``customers.id``.
        """
        pk_index: dict[str, list[tuple[str, str]]] = {}
        for table_name, meta in schema.tables.items():
            for col in meta.columns:
                if col.primary_key:
                    pk_index.setdefault(col.name.lower(), []).append((table_name, col.name))

        existing_pairs = {
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("kind") == "foreign_key"
        }

        for table_name, meta in schema.tables.items():
            for col in meta.columns:
                if col.primary_key or not _ID_SUFFIX_RE.search(col.name):
                    continue
                stem = _ID_SUFFIX_RE.sub("", col.name).lower()
                if not stem:
                    continue
                candidate_tables = {stem, _singularize(stem), stem + "s", stem + "es"}
                for cand_table in candidate_tables:
                    if cand_table == table_name.lower():
                        continue
                    match = next(
                        (t for t in schema.tables if t.lower() == cand_table), None
                    )
                    if match is None:
                        continue
                    target_col = None
                    for c in schema.tables[match].columns:
                        if c.primary_key and c.name.lower() in (col.name.lower(), "id"):
                            target_col = c.name
                            break
                    if target_col is None:
                        continue
                    pair = (_column_node(table_name, col.name), _column_node(match, target_col))
                    if pair in existing_pairs:
                        continue
                    self._add_join_edge(
                        table_name, col.name, match, target_col, kind="inferred_fk"
                    )
                    break

    def _add_semantic_siblings(self, schema: DatabaseSchema) -> None:
        """Link columns that share a name across tables (candidate join keys)."""
        by_name: dict[str, list[str]] = {}
        for table_name, meta in schema.tables.items():
            for col in meta.columns:
                by_name.setdefault(col.name.lower(), []).append(table_name)

        for col_name, tables in by_name.items():
            if len(tables) < 2:
                continue
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    t1, t2 = tables[i], tables[j]
                    self.graph.add_edge(
                        _column_node(t1, col_name),
                        _column_node(t2, col_name),
                        kind="semantic_sibling",
                    )
                    self.graph.add_edge(
                        _column_node(t2, col_name),
                        _column_node(t1, col_name),
                        kind="semantic_sibling",
                    )

    def _add_join_edge(
        self, left_table: str, left_col: str, right_table: str, right_col: str, kind: str
    ) -> None:
        self.graph.add_edge(
            _column_node(left_table, left_col),
            _column_node(right_table, right_col),
            kind=kind,
        )
        self.graph.add_edge(
            _column_node(right_table, right_col),
            _column_node(left_table, left_col),
            kind=kind,
        )
        self._table_graph.add_edge(
            left_table,
            right_table,
            left_table=left_table,
            left_column=left_col,
            right_table=right_table,
            right_column=right_col,
            kind=kind,
        )

    # ------------------------------------------------------------------
    # Join-path reasoning
    # ------------------------------------------------------------------

    def find_join_path(self, table_a: str, table_b: str) -> list[JoinHop]:
        """Shortest chain of join conditions connecting *table_a* to *table_b*.

        Returns ``[]`` if the tables are unrelated or unknown.
        """
        if table_a == table_b:
            return []
        if table_a not in self._table_graph or table_b not in self._table_graph:
            return []
        try:
            path = nx.shortest_path(self._table_graph, table_a, table_b)
        except nx.NetworkXNoPath:
            return []

        hops: list[JoinHop] = []
        for u, v in zip(path, path[1:]):
            edge_data = self._table_graph.get_edge_data(u, v)
            # MultiGraph with parallel edges keyed by int; take the first
            first = next(iter(edge_data.values()))
            if first["left_table"] == u:
                left_col, right_col = first["left_column"], first["right_column"]
            else:
                left_col, right_col = first["right_column"], first["left_column"]
            hops.append(
                JoinHop(
                    left_table=u,
                    left_column=left_col,
                    right_table=v,
                    right_column=right_col,
                    kind=first["kind"],
                )
            )
        return hops

    def get_join_paths_for_tables(self, tables: list[str]) -> list[JoinHop]:
        """Minimal set of join hops connecting *all* of *tables* (best effort).

        Grows a tree by repeatedly connecting the nearest not-yet-included
        table via its shortest path to the current tree (a cheap
        approximation of a Steiner tree — good enough for prompt context).
        """
        known = [t for t in tables if t in self._table_graph]
        if len(known) < 2:
            return []

        connected = {known[0]}
        remaining = set(known[1:])
        all_hops: list[JoinHop] = []
        seen_pairs: set[tuple[str, str]] = set()

        while remaining:
            best: tuple[int, str, list[JoinHop]] | None = None
            for target in remaining:
                for source in connected:
                    hops = self.find_join_path(source, target)
                    if hops and (best is None or len(hops) < best[0]):
                        best = (len(hops), target, hops)
            if best is None:
                # No path from the connected tree to any remaining table.
                connected |= remaining
                break
            _, target, hops = best
            for hop in hops:
                key = tuple(sorted((hop.left_table, hop.right_table)))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    all_hops.append(hop)
            connected.add(target)
            remaining.discard(target)

        return all_hops

    # ------------------------------------------------------------------
    # Prompt-facing rendering
    # ------------------------------------------------------------------

    def describe(self, tables: list[str] | None = None) -> str:
        """Render a compact, LLM-readable relationship block.

        When *tables* is given, only relationships touching those tables
        (plus the join paths needed to connect them) are included.
        """
        lines: list[str] = []

        edges_seen: set[tuple[str, str, str]] = set()
        for u, v, data in self.graph.edges(data=True):
            kind = data.get("kind")
            if kind not in ("foreign_key", "inferred_fk"):
                continue
            u_node, v_node = self.graph.nodes[u], self.graph.nodes[v]
            if u_node.get("kind") != "column" or v_node.get("kind") != "column":
                continue
            t1, c1 = u_node["table"], u_node["name"]
            t2, c2 = v_node["table"], v_node["name"]
            if tables and t1 not in tables and t2 not in tables:
                continue
            key = tuple(sorted((f"{t1}.{c1}", f"{t2}.{c2}"))) + (kind,)
            if key in edges_seen:
                continue
            edges_seen.add(key)
            label = "declared FK" if kind == "foreign_key" else "inferred relationship"
            lines.append(f"  {t1}.{c1} = {t2}.{c2}  ({label})")

        if tables and len(tables) > 1:
            hops = self.get_join_paths_for_tables(tables)
            bridging = [h for h in hops if h.left_table not in tables or h.right_table not in tables]
            if bridging:
                lines.append(
                    "\n  Suggested join path through an intermediate table "
                    "(no direct relationship exists between the relevant tables):"
                )
                for hop in bridging:
                    lines.append(f"    {hop.as_sql()}")

        if not lines:
            return "No explicit or inferred relationships detected between the relevant tables."

        return "## Table Relationships (Knowledge Graph)\n" + "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation, useful for UI visualisation."""
        nodes = [
            {"id": n, **{k: v for k, v in d.items()}} for n, d in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "kind": d.get("kind")}
            for u, v, d in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def to_graphviz(self) -> str:
        """Return a Graphviz ``dot`` string (renderable via ``st.graphviz_chart``)."""
        lines = ["digraph schema {", '  rankdir="LR";', "  node [shape=box, fontsize=10];"]
        for table in self._table_graph.nodes:
            lines.append(f'  "{table}" [style=filled, fillcolor="#e8f0fe"];')
        seen: set[tuple[str, str]] = set()
        for u, v, data in self._table_graph.edges(data=True):
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            label = f"{data['left_column']}={data['right_column']}"
            style = "solid" if data.get("kind") == "foreign_key" else "dashed"
            lines.append(f'  "{u}" -> "{v}" [label="{label}", style={style}, dir=none];')
        lines.append("}")
        return "\n".join(lines)
