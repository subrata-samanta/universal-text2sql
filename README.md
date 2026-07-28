# Universal Text-to-SQL Agent

[![CI](https://github.com/subrata-samanta/universal-text2sql/actions/workflows/ci.yml/badge.svg)](https://github.com/subrata-samanta/universal-text2sql/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code of Conduct](https://img.shields.io/badge/Code%20of%20Conduct-Contributor%20Covenant-blueviolet.svg)](CODE_OF_CONDUCT.md)

A fully agentic, **universal** Natural Language → SQL system that works against
**any** database with zero manual configuration: it discovers its own schema,
builds its own knowledge graph, and writes its own business glossary.

This is a community open-source project — contributions, issues, and ideas
are very welcome. See [Contributing](#contributing) below to get started.

- 🔌 **Works with any database** – SQLite, PostgreSQL, MySQL (anything SQLAlchemy supports)
- 🗺️ **Auto-discovers schema** – reads tables, columns, types, primary/foreign keys, and sample values at startup
- 🕸️ **Auto-builds a knowledge graph** – declared foreign keys *plus* naming-convention-inferred relationships and multi-hop join paths, so the LLM never has to guess how to join two tables that aren't directly connected
- 📖 **Auto-generates a business glossary** – heuristic + optional LLM-written table/column descriptions and synonyms, cached to disk, so "revenue" can resolve to a column literally named `total_amount`
- 🔍 **Semantic schema linking** – TF-IDF/cosine retrieval over the auto-generated metadata (not just literal keyword overlap) decides which tables matter for a question
- 🧮 **Complexity-aware routing** – classifies each question as SIMPLE/MODERATE/COMPLEX and only pays for expensive strategies when needed
- 🗳️ **Self-consistency SQL generation** – samples multiple SQL candidates for non-trivial questions, executes all of them, and lets the results vote on the winner
- 🤖 **LangGraph workflow** – multi-node agentic graph with conditional routing
- ⚡ **Groq LLM** – ultra-fast inference with `llama-3.3-70b-versatile` (or any Groq model)
- 🔄 **Self-reflection** – automatically detects SQL errors and retries with corrected queries
- 🧠 **RL-inspired query memory** – successful queries are stored with a reward signal and replayed as few-shot examples for future questions
- 💬 **Streamlit UI** – interactive web interface (with a knowledge-graph viewer + glossary panel) + CLI

---

## Table of Contents

- [Architecture](#architecture)
- [Component deep-dive](#component-deep-dive)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Supported Databases](#supported-databases)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Community](#community)
- [References](#references)
- [License](#license)

---

## Architecture

The system is organised into four layers. Layers 1–3 run **once per
database** (at bootstrap time) and are cached; layer 4 runs **once per
question**. This separation is what makes the agent "universal" — everything
dataset-specific is *derived automatically* instead of hand-configured.

```
┌───────────────────────────────────────────────────────────────────────┐
│ Layer 4 · Per-question agent graph      (LangGraph, universal_text2sql/agent/) │
│   schema linking → complexity routing → SQL generation → self-      │
│   consistency voting → execution → validation → reflection → answer  │
├───────────────────────────────────────────────────────────────────────┤
│ Layer 3 · Semantic layer                (universal_text2sql/knowledge/, retrieval/) │
│   knowledge graph (joins) · business glossary (meaning) · TF-IDF index │
├───────────────────────────────────────────────────────────────────────┤
│ Layer 2 · Schema discovery              (universal_text2sql/database/schema.py) │
│   tables, columns, types, PK/FK, row counts, sample values             │
├───────────────────────────────────────────────────────────────────────┤
│ Layer 1 · Universal connector           (universal_text2sql/database/connector.py) │
│   any SQLAlchemy engine — SQLite / PostgreSQL / MySQL / ...            │
└───────────────────────────────────────────────────────────────────────┘
```

All four layers are assembled by a single call, `bootstrap()` (see
`universal_text2sql/bootstrap.py`), which returns an `AgentContext` — the
object both the CLI and the Streamlit UI drive.

### System diagram

```mermaid
flowchart TB
    CLI["main.py (CLI)"]
    UI["app.py (Streamlit UI)"]
    LIB["Library caller<br/>your own script"]

    CLI --> CONN
    UI --> CONN
    LIB --> CONN

    CONN["DatabaseConnector<br/>any SQLAlchemy URL"] --> DISC["SchemaDiscovery<br/>tables / columns / PK-FK / samples"]
    DISC --> KG["SchemaKnowledgeGraph<br/>declared + inferred joins"]
    DISC --> META["MetadataEnricher<br/>heuristic + LLM glossary, cached"]
    MEM["QueryMemory<br/>RL-weighted few-shot store"]

    KG --> CTX["AgentContext.ask question"]
    META --> CTX
    MEM --> CTX
    CTX --> GRAPH["LangGraph agent<br/>see node diagram below"]
    GRAPH --> DB[("Target Database")]
    GRAPH --> LLMBOX["Groq LLM ChatGroq"]
    GRAPH -. writes back .-> MEM
    GRAPH --> ANSWER["final_answer + generated_sql<br/>+ execution_result + trace"]
```

### Agent graph (per-question execution)

```mermaid
flowchart TD
    QIN(["question in"]) --> SEL[select_schema]
    SEL -->|"TF-IDF semantic ranking<br/>+ KG join context<br/>+ glossary lookup"| CLS[classify_complexity]
    CLS -->|"SIMPLE / MODERATE / COMPLEX"| GEN[generate_sql]
    GEN -->|"1 candidate if SIMPLE,<br/>N candidates otherwise"| VOTE[select_best_candidate]
    VOTE -->|"executes every candidate,<br/>majority-result wins"| EXEC[execute_sql]
    EXEC -->|error| REFLECT[reflect]
    EXEC -->|ok| VALIDATE[validate_result]
    VALIDATE -->|INVALID| REFLECT
    REFLECT -->|"corrected SQL<br/>retry_count += 1"| EXEC
    VALIDATE -->|VALID| FORMAT[format_answer]
    EXEC -->|"max retries reached"| FORMAT
    VALIDATE -->|"max retries reached"| FORMAT
    FORMAT --> STORE[store_memory]
    STORE --> QOUT(["final_answer out"])

    style SEL fill:#e8f0fe,stroke:#4285f4
    style CLS fill:#e8f0fe,stroke:#4285f4
    style GEN fill:#fef7e0,stroke:#f9ab00
    style VOTE fill:#fef7e0,stroke:#f9ab00
    style REFLECT fill:#fce8e6,stroke:#ea4335
```

Every node is a plain function `(state, **deps) -> partial_state_update`
(`universal_text2sql/agent/nodes.py`); `agent/graph.py` binds dependencies via
`functools.partial` and wires the nodes into a LangGraph `StateGraph` over
the `AgentState` TypedDict (`agent/state.py`). State flows through the whole
graph as one dict that each node reads from and merges updates into —
there's no hidden state anywhere else.

### Request lifecycle (sequence)

```mermaid
sequenceDiagram
    participant U as User
    participant C as AgentContext
    participant G as LangGraph
    participant K as KnowledgeGraph/Glossary
    participant L as Groq LLM
    participant D as Database

    U->>C: ask("top 3 products by revenue")
    C->>G: run_query(question, ...)
    G->>K: rank tables (TF-IDF) + join paths + glossary
    K-->>G: relevant_tables, kg_context, business_glossary
    G->>L: classify complexity
    L-->>G: "MODERATE"
    G->>L: generate SQL (N candidates if non-trivial)
    L-->>G: candidate SQL(s)
    loop self-consistency (N > 1)
        G->>D: execute candidate
        D-->>G: result rows
    end
    G->>G: vote → winning SQL
    G->>D: execute winning SQL
    D-->>G: result rows / error
    alt execution error
        G->>L: reflect on error
        L-->>G: corrected SQL
        G->>D: re-execute
    end
    G->>L: validate result answers the question
    L-->>G: VALID / INVALID
    G->>L: format natural-language answer
    L-->>G: final_answer
    G->>C: store successful query in memory (reward-weighted)
    C-->>U: {final_answer, generated_sql, execution_result, trace}
```

### State-of-the-art techniques implemented

The agent doesn't just prompt an LLM with a schema dump — it builds a semantic
layer over the target database automatically, following techniques from
recent text-to-SQL research (CHESS, CHASE-SQL, DIN-SQL, MAC-SQL-style
pipelines) adapted to run against an arbitrary, previously-unseen database:

| Technique | Research direction | Implementation |
|---|---|---|
| **Schema knowledge graph** | Graph-augmented schema linking (RESDSQL/SADGA-style) | `knowledge/graph.py` builds a graph of tables/columns with declared FK, naming-convention-*inferred* FK, and cross-table "semantic sibling" edges; multi-hop join paths are computed with graph shortest-path search |
| **Automatic metadata / business glossary** | Context retrieval & semantic layers (CHESS) | `knowledge/metadata.py` generates table/column descriptions and synonyms — heuristically from naming conventions always, and via one cached LLM call per table when available — so questions using business terms (not raw column names) still resolve correctly |
| **Semantic schema linking** | Embedding-based retrieval (CHESS, DAIL-SQL) | `retrieval/semantic.py` is a dependency-free TF-IDF + cosine-similarity index used to rank tables/columns by relevance to the question, replacing literal keyword overlap |
| **Complexity-aware routing** | Difficulty classification & decomposition (DIN-SQL) | `classify_complexity` node labels each question SIMPLE/MODERATE/COMPLEX and gates expensive strategies (multi-candidate sampling) so trivial questions stay cheap |
| **Self-consistency generation** | Multi-path sampling + execution voting (CHASE-SQL, self-consistency) | `generate_sql` samples several SQL candidates for non-trivial questions; `select_best_candidate` executes all of them and picks the majority-agreeing result |
| **Self-reflection** | Execution-guided error correction | Error + previous SQL → corrected SQL loop (configurable retries), now with knowledge-graph join context in the correction prompt |
| **RL-inspired few-shot memory** | Reward-weighted example replay | `reward = 1/(1+retries)` weights memory entries; zero-retry queries surface first, retrieved via the same TF-IDF-flavoured word-overlap scoring |
| **Result validation** | Self-verification | Separate LLM call checks whether the answer actually makes sense |

---

## Component deep-dive

### Layer 1 — `database/connector.py`: universal connector

`DatabaseConnector` wraps a single SQLAlchemy `Engine` and exposes only the
primitives the rest of the system needs: `execute_query` (→ DataFrame),
`get_table_names`, `get_column_info`, `get_foreign_keys`, `get_sample_rows`.
Because everything above this layer only talks to these five methods, adding
support for a new SQL dialect is a connection-string change, not a code
change — anything SQLAlchemy has a dialect for (SQLite, PostgreSQL, MySQL,
Snowflake, BigQuery, ...) works without touching the agent.

### Layer 2 — `database/schema.py`: schema discovery

`SchemaDiscovery.discover()` introspects the live database once (via
SQLAlchemy's `inspect()`) and materialises a `DatabaseSchema`: every table's
columns (name, type, nullability, PK), declared foreign keys, row count, and
a handful of sample values per column. This is the *raw* metadata layer —
purely structural, no semantics yet. `DatabaseSchema.to_ddl()` / `subset_ddl()`
render it back into `CREATE TABLE` text for prompts.

### Layer 3 — the semantic layer (what makes it "universal")

This is the layer that lets the agent work on a database it has never seen,
without a human writing a data dictionary first.

**`knowledge/graph.py` — `SchemaKnowledgeGraph`**

A `networkx.MultiDiGraph` with two node kinds (`table:<name>`,
`column:<table>.<name>`) and four edge kinds:

| Edge kind | Meaning | How it's found |
|---|---|---|
| `has_column` | structural table → column edge | direct from schema |
| `foreign_key` | declared relationship | SQLAlchemy FK constraints |
| `inferred_fk` | *undeclared* relationship | naming convention: a non-PK column ending in `_id`/`Id` is matched against singular/plural table names and their primary key (handles the extremely common case of FK-less exports, e.g. denormalised CSV-backed SQLite dumps) |
| `semantic_sibling` | same column name in ≥2 tables | candidate join key / duplicated concept (e.g. two `email` columns) |

A second, undirected **table-level projection** (`_table_graph`) is
maintained alongside the column graph purely for path-finding:
`find_join_path(a, b)` runs `networkx.shortest_path` over it to produce the
exact `A.col = B.col` hops connecting two tables — including tables with
*no* direct relationship, by routing through an intermediate table (e.g.
`order_items` → `customers` via `orders`). `get_join_paths_for_tables(...)`
extends this to a whole set of relevant tables with a cheap greedy
Steiner-tree approximation. `describe()` renders the result as plain text
that gets injected straight into the SQL generation/reflection prompts, so
the LLM is told the exact join columns instead of having to guess them from
a DDL dump.

**`knowledge/metadata.py` — `MetadataEnricher`**

Two-tier metadata generation:

1. **Heuristic pass (always on, zero cost):** identifiers are split on
   `snake_case`/`camelCase` boundaries into human-readable phrases; role is
   inferred from primary-key/foreign-key status and sample values (e.g.
   `customer_id` → *"Reference to a customer record (foreign key)."*).
2. **LLM pass (optional, cached):** one call per table sends its DDL +
   sample values to the configured LLM and asks for a JSON object —
   table description, table synonyms, and per-column `{meaning, synonyms}`.
   Results are merged over the heuristic baseline and written to
   `METADATA_CACHE_DIR/<schema-signature-hash>.json`, so re-running against
   the same database is free after the first pass, and a schema change
   (new/renamed column) invalidates only that database's cache entry.

The combined result is rendered by `glossary_block()` into a "Business
Glossary" prompt section — this is what lets a question about *"revenue"*
resolve to a column literally named `total_amount`.

**`retrieval/semantic.py` — `SemanticIndex`**

A small, dependency-free TF-IDF vector space model (pure Python — no numpy
matrix, no sklearn, no embedding API): term frequencies × inverse document
frequency, cosine similarity via sparse dot products over `Counter` objects.
It's deliberately not a neural embedding model — schema/glossary text and
questions are short, so classic TF-IDF captures most of the useful signal
while staying deterministic (important for tests) and avoiding an extra API
dependency (Groq doesn't offer embeddings). It's used in two places:
`DatabaseSchema.get_relevant_tables_semantic()` (schema linking — ranks
tables by similarity of their name/description/columns/glossary/samples
against the question, falling back to plain keyword overlap if nothing
scores above zero) and, unchanged from the original design, `QueryMemory`'s
few-shot retrieval.

### Layer 4 — the per-question agent (`agent/`)

| Node | Reads | Produces | Notes |
|---|---|---|---|
| `select_schema` | question, schema, KG, glossary | `relevant_tables`, `schema_context`, `kg_context`, `business_glossary`, `few_shot_examples` | semantic ranking with keyword-overlap fallback |
| `classify_complexity` | question, `schema_context` | `complexity` | one LLM call, defaults to `MODERATE` on an unparseable response |
| `generate_sql` | all of the above | `generated_sql`, `sql_candidates` | 1 candidate unless `complexity != SIMPLE` **and** `self_consistency_samples > 1`; duplicate candidates are deduplicated |
| `select_best_candidate` | `sql_candidates` | `generated_sql` | no-op when there's only one candidate; otherwise executes every candidate and keeps the SQL whose *result* the largest group of candidates agree on (order-independent row/column signature) |
| `execute_sql` | `generated_sql` | `execution_result` / `execution_error` | the single source of truth for what actually ran |
| `reflect` | error or invalid verdict, previous SQL, KG context | corrected `generated_sql`, `retry_count += 1` | loops back to `execute_sql` up to `MAX_RETRIES` |
| `validate_result` | question, SQL, result preview | `validation_verdict` | second LLM call, catches "ran fine but answers the wrong question" |
| `format_answer` | result preview | `final_answer` | DataFrame → natural language |
| `store_memory` | `success`, `generated_sql` | — | writes to `QueryMemory` with `reward = 1/(1+retry_count)` |

### Why these design choices

- **Two-tier metadata (heuristic + optional LLM)** — the agent must be
  useful with *no* LLM configured yet (e.g. `main.py --graph` / `--glossary`
  work without `GROQ_API_KEY`), and must not force an LLM call per column on
  every bootstrap of a database it has already seen — hence heuristics
  always run, the LLM pass is opt-in and cached per table.
- **TF-IDF instead of embeddings** — keeps the project dependency-light
  (no torch/sentence-transformers) and deterministic for tests, at the cost
  of missing pure synonym matches the glossary doesn't already cover — an
  acceptable trade-off given the glossary pass exists specifically to close
  that gap.
- **Self-consistency gated by complexity, not always-on** — sampling N
  candidates multiplies LLM calls by N; gating it behind `classify_complexity`
  means simple lookups (the majority of real usage) stay single-shot, and the
  extra cost only applies where it measurably helps (multi-join / aggregation
  questions).
- **`select_best_candidate` only picks the winning SQL text, not its
  execution result** — `execute_sql` still re-runs the chosen query. This
  keeps `execute_sql` the single place that owns `execution_result`/
  `execution_error`, so the reflect/validate loop downstream doesn't need to
  know whether self-consistency ran at all.
- **Everything new is additive to `AgentState`** — every new field
  (`kg_context`, `business_glossary`, `complexity`, `sql_candidates`) is read
  with `.get(..., default)`, and every new node parameter has a safe default
  (`knowledge_graph=None`, `self_consistency_samples=1`). The graph behaves
  exactly like the original single-shot pipeline when the new features are
  left at their defaults, which is why all pre-existing tests pass unchanged.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

Get a free Groq API key at <https://console.groq.com>.

### 3a. Streamlit UI

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser. The sidebar includes a
**Knowledge Graph** panel (Graphviz join-path diagram) and an
**Auto-Generated Glossary** panel showing what the agent inferred about your
data before answering anything.

### 3b. CLI (interactive)

```bash
python main.py
```

### 3c. CLI (single query)

```bash
python main.py "What are the top 3 products by revenue?"
```

### 3d. CLI (inspect what the agent auto-discovered, no API key required)

```bash
python main.py --graph      # print the auto-built knowledge graph
python main.py --glossary   # print the auto-generated business glossary
```

### 3e. Use as a library

```python
from dotenv import load_dotenv
load_dotenv()

from universal_text2sql.bootstrap import bootstrap

# One call: connect, discover schema, build the knowledge graph,
# generate the business glossary, and load query memory.
ctx = bootstrap(database_url="sqlite:///./mydb.sqlite3")

result = ctx.ask("How many customers signed up last month?")
print(result["final_answer"])
print(result["generated_sql"])

print(ctx.describe_knowledge_graph())
print(ctx.describe_glossary())
```

`bootstrap()` returns an `AgentContext` bundling everything the lower-level
`universal_text2sql.agent.graph.run_query()` function needs, for callers that
want direct control over each dependency instead.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required for SQL generation)* | Your Groq API key |
| `DATABASE_URL` | `sqlite:///:memory:` | SQLAlchemy connection URL |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `MAX_RETRIES` | `3` | Max self-reflection retries |
| `ENABLE_QUERY_MEMORY` | `true` | Enable RL-inspired query memory |
| `QUERY_MEMORY_PATH` | `./query_memory.json` | Path to persist query memory |
| `ENABLE_METADATA_ENRICHMENT` | `true` | Enable the LLM-assisted glossary pass (heuristic descriptions always run) |
| `METADATA_CACHE_DIR` | `./.metadata_cache` | Where the auto-generated glossary cache is stored (keyed by schema signature) |
| `SELF_CONSISTENCY_SAMPLES` | `1` | SQL candidates sampled for non-trivial questions (`1` disables self-consistency) |

---

## Supported Databases

| Database | Connection URL format |
|---|---|
| SQLite | `sqlite:///./path/to/db.sqlite3` |
| PostgreSQL | `postgresql://user:password@host:5432/dbname` |
| MySQL | `mysql+pymysql://user:password@host:3306/dbname` |
| Any SQLAlchemy backend | See [SQLAlchemy docs](https://docs.sqlalchemy.org/en/20/core/engines.html) |

When no `DATABASE_URL` is set, an **in-memory SQLite demo database** with four tables (customers, products, orders, order_items) is automatically created and seeded.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests run against an in-memory SQLite database and mock the LLM where
needed, so **no API key is required**. Tests cover the original agent graph
plus the new knowledge graph, metadata enrichment, semantic retrieval,
self-consistency, and bootstrap modules.

---

## Project Structure

```
universal_text2sql/
├── agent/
│   ├── graph.py        # LangGraph StateGraph + routing logic
│   ├── memory.py       # RL-inspired query memory (few-shot store)
│   ├── nodes.py        # Individual graph node implementations
│   └── state.py        # AgentState TypedDict
├── database/
│   ├── connector.py    # Universal SQLAlchemy connector
│   └── schema.py       # Auto schema discovery, metadata, semantic table ranking
├── knowledge/
│   ├── graph.py         # SchemaKnowledgeGraph — declared/inferred FKs, join-path search
│   └── metadata.py       # MetadataEnricher — heuristic + LLM auto business glossary
├── retrieval/
│   └── semantic.py       # Dependency-free TF-IDF + cosine similarity index
├── llm/
│   └── groq_client.py  # Groq LLM factory
├── prompts/
│   └── templates.py    # CoT / reflection / validation / complexity / metadata prompts
└── utils/
    └── demo_data.py    # Demo database seeder
bootstrap.py            # Single entry point: connect → discover → KG → glossary → AgentContext
app.py                  # Streamlit web UI (knowledge graph + glossary panels)
main.py                 # CLI entry point (--graph / --glossary inspection flags)
tests/
├── test_schema.py            # Connector + schema discovery tests
├── test_memory.py            # Query memory tests
├── test_agent_nodes.py       # Node unit tests (mock LLM)
├── test_graph.py             # Graph routing + compile tests
├── test_knowledge_graph.py   # Knowledge graph construction + join-path tests
├── test_metadata.py          # Heuristic + LLM metadata enrichment tests
├── test_semantic_retrieval.py# TF-IDF retrieval tests
├── test_self_consistency.py  # Complexity classification + candidate voting tests
└── test_bootstrap.py         # AgentContext orchestration tests
```

Project tooling config lives in `pyproject.toml` (packaging, `ruff`, `pytest`,
`coverage`), with `.pre-commit-config.yaml`, `.editorconfig`, and a `Makefile`
providing convenience shortcuts (`make test`, `make lint`, `make format`,
`make ui`) for contributors — see [Contributing](#contributing).

---

## Roadmap

Ideas under consideration — see [open issues](https://github.com/subrata-samanta/universal-text2sql/issues)
for the current state and feel free to propose more:

- [ ] Pluggable LLM providers beyond Groq (OpenAI, Anthropic, local models via Ollama)
- [ ] Optional dense-embedding backend for semantic retrieval (as an upgrade path from TF-IDF)
- [ ] Query decomposition for COMPLEX questions (sub-question → sub-SQL → compose), DIN-SQL-style
- [ ] Benchmark harness against public text-to-SQL datasets (Spider, BIRD)
- [ ] Row-level security / read-only enforcement helpers for production deployments
- [ ] Additional dialect-specific prompt guidance (BigQuery, Snowflake, MSSQL)
- [ ] Multi-turn conversational follow-up questions ("now filter that by country")

## Contributing

Contributions are very welcome — bug fixes, new database dialect notes, new
research-backed techniques, tests, and documentation improvements alike.

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, test/lint
   commands, commit conventions, and PR process.
2. Check [open issues](https://github.com/subrata-samanta/universal-text2sql/issues)
   for something to work on, or open a new one to discuss your idea first for
   larger changes.
3. All contributors are expected to follow the
   [Code of Conduct](CODE_OF_CONDUCT.md).

Found a security issue? Please **do not** open a public issue — see
[SECURITY.md](SECURITY.md) for how to report it privately.

## Community

- 🐛 [Report a bug](https://github.com/subrata-samanta/universal-text2sql/issues/new?template=bug_report.yml)
- ✨ [Request a feature](https://github.com/subrata-samanta/universal-text2sql/issues/new?template=feature_request.yml)
- 💬 [Discussions](https://github.com/subrata-samanta/universal-text2sql/discussions) — usage questions and open-ended ideas
- 📜 [Changelog](CHANGELOG.md) — what shipped, and when

## References

This project implements ideas from (and owes credit to) recent text-to-SQL
research — see the [State-of-the-art techniques](#state-of-the-art-techniques-implemented)
table for how each is applied here:

- Pourreza & Rafiei, *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL
  with Self-Correction*, 2023.
- Talaei et al., *CHESS: Contextual Harnessing for Efficient SQL Synthesis*, 2024.
- Pourreza et al., *CHASE-SQL: Multi-Path Reasoning and Preference-Optimized
  Candidate Selection in Text-to-SQL*, 2024.
- Wang et al., *MAC-SQL: A Multi-Agent Collaborative Framework for
  Text-to-SQL*, 2024.
- Li et al., *RESDSQL: Decoupling Schema Linking and Skeleton Parsing for
  Text-to-SQL*, 2023.
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in
  Language Models*, 2022 (the general technique behind self-consistency
  SQL candidate voting).

If you use this project in academic work, please cite it via the repository
URL; a `CITATION.cff` is welcome as a contribution if there's demand for it.

## License

Licensed under the [MIT License](LICENSE) — free for personal, academic, and
commercial use.
