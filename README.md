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
- 🎯 **Value/entity grounding** – profiles low-cardinality columns and matches question terms against real stored values, so "customers from usa" resolves to the literal value actually in the database
- 🔍 **Semantic schema linking** – TF-IDF/cosine retrieval over the auto-generated metadata (not just literal keyword overlap) decides which tables matter for a question
- 🧮 **Complexity-aware routing** – classifies each question as SIMPLE/MODERATE/COMPLEX and only pays for expensive strategies when needed
- 🧩 **Query decomposition** – breaks COMPLEX questions into ordered sub-questions and composes their SQL via CTEs, DIN-SQL-style
- 🗳️ **Self-consistency SQL generation** – samples multiple SQL candidates for non-trivial questions, executes all of them, and lets the results vote on the winner
- 🗨️ **Multi-turn follow-ups** – "now filter that by country" reuses the previous question/SQL/answer as context
- 🤖 **LangGraph workflow** – multi-node agentic graph with conditional routing
- 🔀 **Pluggable LLM providers** – Groq (default, ultra-fast), OpenAI, Anthropic, or local Ollama, swappable via one env var
- 🛡️ **SQL safety guardrails** – read-only enforcement by default, blocking LLM-generated writes unless explicitly allowed
- 🔄 **Self-reflection** – automatically detects SQL errors and retries with corrected queries
- 🧠 **RL-inspired query memory** – successful queries are stored with a reward signal and replayed as few-shot examples for future questions
- 📏 **Evaluation harness** – golden-question execution-accuracy regression suite, zero-API-key by default
- 💬 **Streamlit UI** – interactive web interface (with knowledge-graph, glossary, and conversation panels) + CLI

---

## Table of Contents

- [Architecture](#architecture)
- [Component deep-dive](#component-deep-dive)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Supported Databases](#supported-databases)
- [Evaluation](#evaluation)
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
│   schema linking → complexity routing → SQL generation (or            │
│   decomposition) → self-consistency voting → execution (read-only     │
│   guarded) → validation → reflection → answer                          │
├───────────────────────────────────────────────────────────────────────┤
│ Layer 3 · Semantic layer                (universal_text2sql/knowledge/, retrieval/) │
│   knowledge graph (joins) · business glossary (meaning) · value       │
│   grounding (real literals) · TF-IDF index                             │
├───────────────────────────────────────────────────────────────────────┤
│ Layer 2 · Schema discovery              (universal_text2sql/database/schema.py) │
│   tables, columns, types, PK/FK, row counts, sample values             │
├───────────────────────────────────────────────────────────────────────┤
│ Layer 1 · Universal connector           (universal_text2sql/database/connector.py) │
│   any SQLAlchemy engine — SQLite / PostgreSQL / MySQL / ...            │
│   + safety.py read-only enforcement · llm/factory.py pluggable LLMs   │
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

    CONN["DatabaseConnector read_only=True<br/>any SQLAlchemy URL"] --> DISC["SchemaDiscovery<br/>tables / columns / PK-FK / samples"]
    DISC --> KG["SchemaKnowledgeGraph<br/>declared + inferred joins"]
    DISC --> META["MetadataEnricher<br/>heuristic + LLM glossary, cached"]
    DISC --> GROUND["ValueGroundingIndex<br/>profiled categorical values, cached"]
    MEM["QueryMemory<br/>RL-weighted few-shot store"]

    KG --> CTX["AgentContext.ask question"]
    META --> CTX
    GROUND --> CTX
    MEM --> CTX
    CTX --> GRAPH["LangGraph agent<br/>see node diagram below"]
    GRAPH --> DB[("Target Database")]
    GRAPH --> LLMBOX["Pluggable LLM via llm/factory.py<br/>Groq default, or OpenAI/Anthropic/Ollama"]
    GRAPH -. writes back .-> MEM
    GRAPH --> ANSWER["final_answer + generated_sql<br/>+ execution_result + trace"]
```

### Agent graph (per-question execution)

```mermaid
flowchart TD
    QIN(["question in"]) --> SEL[select_schema]
    SEL -->|"TF-IDF semantic ranking<br/>+ KG join context<br/>+ glossary + grounding hints<br/>+ conversation context"| CLS[classify_complexity]
    CLS -->|"COMPLEX + decomposition enabled"| DECOMP[decompose_sql]
    CLS -->|"otherwise"| GEN[generate_sql]
    DECOMP -->|"sub-questions -> CTEs -> composed SQL<br/>falls back to generate_sql on failure"| VOTE[select_best_candidate]
    GEN -->|"1 candidate if SIMPLE,<br/>N candidates otherwise"| VOTE
    VOTE -->|"executes every candidate,<br/>majority-result wins"| EXEC["execute_sql<br/>read-only enforced by default"]
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
    style DECOMP fill:#fef7e0,stroke:#f9ab00
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
    participant K as KG/Glossary/Grounding
    participant L as Pluggable LLM
    participant D as Database (read-only)

    U->>C: ask("now filter that by country", history)
    C->>G: run_query(question, conversation_history, ...)
    G->>K: rank tables (TF-IDF) + join paths + glossary + grounded values
    K-->>G: relevant_tables, kg_context, business_glossary, grounding_hints
    G->>L: classify complexity
    L-->>G: "COMPLEX"
    alt decomposition enabled
        G->>L: plan sub-questions, generate SQL per step
        L-->>G: sub_questions, composed CTE SQL
    else single-shot
        G->>L: generate SQL (N candidates if non-trivial)
        L-->>G: candidate SQL(s)
        loop self-consistency (N > 1)
            G->>D: execute candidate (bounded preview)
            D-->>G: result rows
        end
        G->>G: vote → winning SQL
    end
    G->>D: execute winning SQL (read-only enforced)
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
| **Value/entity grounding** | Literal grounding against real column values (CHESS) | `knowledge/grounding.py` profiles low-cardinality text columns, caches their distinct values, and n-gram-matches question terms against them so filter literals (e.g. `country = 'USA'`) come from what's actually stored, not an LLM guess |
| **Complexity-aware routing** | Difficulty classification & decomposition (DIN-SQL) | `classify_complexity` node labels each question SIMPLE/MODERATE/COMPLEX and gates expensive strategies (multi-candidate sampling, decomposition) so trivial questions stay cheap |
| **Query decomposition** | Sub-question planning + SQL composition (DIN-SQL) | `decompose_sql` plans an ordered list of sub-questions for COMPLEX queries, generates a SQL fragment per step, and composes them into a single `WITH step_1 AS (...), ...` query — falling back to single-shot generation if any step fails to parse |
| **Self-consistency generation** | Multi-path sampling + execution voting (CHASE-SQL, self-consistency) | `generate_sql` samples several SQL candidates for non-trivial questions; `select_best_candidate` executes all of them and picks the majority-agreeing result |
| **Self-reflection** | Execution-guided error correction | Error + previous SQL → corrected SQL loop (configurable retries), now with knowledge-graph join context and grounding hints in the correction prompt |
| **RL-inspired few-shot memory** | Reward-weighted example replay | `reward = 1/(1+retries)` weights memory entries; zero-retry queries surface first, retrieved via the same TF-IDF-flavoured word-overlap scoring |
| **Result validation** | Self-verification | Separate LLM call checks whether the answer actually makes sense |

### Production-readiness techniques implemented

Research-grade generation quality is only half of a "future ready" agent —
these close the gap toward something safe to point at a real database:

| Technique | Why it matters | Implementation |
|---|---|---|
| **SQL safety guardrails** | LLM-generated SQL is untrusted input | `database/safety.py` classifies every statement as READ/WRITE (regex baseline, optional `sqlglot` AST mode that also catches writes hidden inside a CTE) and blocks writes by default; `DatabaseConnector(read_only=True)` is the out-of-the-box default |
| **Pluggable LLM providers** | No single-vendor lock-in | `llm/factory.py` resolves `LLM_PROVIDER` (Groq/OpenAI/Anthropic/Ollama) to a common `LLMRunnable` protocol; each provider SDK is lazy-imported so the core install stays dependency-light |
| **Multi-turn conversational follow-ups** | Real usage is a conversation, not one-shot Q&A | Caller-owned `conversation_history` is rendered into a "Previous Turn(s)" prompt block so "now filter that by country" resolves against the prior question/SQL/answer |
| **Evaluation harness** | "It works on my question" isn't evidence | `eval/` runs a golden-question execution-accuracy regression suite in a zero-API-key `--mock` mode (wired into CI) plus a `--live` mode for real benchmarking |

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

`execute_query` also enforces the SQL safety policy from
`database/safety.py`: by default (`read_only=True`, driven by `SQL_READ_ONLY`)
every statement is classified READ/WRITE before it runs, and WRITE statements
raise `SQLSafetyError` instead of executing — see
[Production-readiness techniques implemented](#production-readiness-techniques-implemented).

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

**`knowledge/grounding.py` — `ValueGroundingIndex`**

Schema/glossary text tells the LLM *which* column to filter on; it doesn't
tell it what the values in that column actually look like. `ValueGroundingIndex.build()`
profiles `TEXT`/`VARCHAR`/`CHAR` columns, skips any table over
`GROUNDING_MAX_TABLE_ROWS` (never full-scans a huge table), and treats a
column as categorical only if its distinct-value count is both under
`GROUNDING_MAX_DISTINCT` and under `GROUNDING_CARDINALITY_RATIO` of the row
count — cheaply separating `customers.country` (low cardinality) from
`customers.email` (effectively unique) without any hardcoded column-name
rules. The resulting distinct values are cached to disk by schema-signature
hash (same pattern as the glossary cache). At query time, `.match()`
n-gram-matches the question's tokens against the cached values (substring +
`difflib` overlap, or optional `rapidfuzz` token-set scoring when installed)
and renders a "Value Grounding Hints" prompt block — so "customers from usa"
resolves to the literal `'USA'` actually stored in the table, not a guess.
This is explicitly *not* semantic matching: "USA" won't match a stored
`"United States"` without a shared substring — that gap is instead partly
covered by the LLM glossary's synonym pass.

### Layer 4 — the per-question agent (`agent/`)

| Node | Reads | Produces | Notes |
|---|---|---|---|
| `select_schema` | question, schema, KG, glossary, grounding index | `relevant_tables`, `schema_context`, `kg_context`, `business_glossary`, `grounding_hints`, `few_shot_examples` | semantic ranking with keyword-overlap fallback |
| `classify_complexity` | question, `schema_context` | `complexity` | one LLM call, defaults to `MODERATE` on an unparseable response |
| `decompose_sql` | all of the above | `generated_sql`, `sql_candidates`, `sub_questions`, `decomposition_steps` | only reached when `complexity == COMPLEX` **and** query decomposition is enabled; plans ordered sub-questions, generates a SQL fragment per step, composes a CTE query; falls back to `generate_sql` on any parse/generation failure |
| `generate_sql` | all of the above, plus `conversation_context` | `generated_sql`, `sql_candidates` | 1 candidate unless `complexity != SIMPLE` **and** `self_consistency_samples > 1`; duplicate candidates are deduplicated; the default branch when decomposition is off or the question isn't COMPLEX |
| `select_best_candidate` | `sql_candidates` | `generated_sql` | no-op when there's only one candidate; otherwise executes every candidate (bounded preview `LIMIT`) and keeps the SQL whose *result* the largest group of candidates agree on (order-independent row/column signature) |
| `execute_sql` | `generated_sql` | `execution_result` / `execution_error` | the single source of truth for what actually ran; read-only enforced by default (`SQLSafetyError` on a WRITE statement) |
| `reflect` | error or invalid verdict, previous SQL, KG context, grounding hints | corrected `generated_sql`, `retry_count += 1` | loops back to `execute_sql` up to `MAX_RETRIES` |
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
  (`kg_context`, `business_glossary`, `complexity`, `sql_candidates`,
  `grounding_hints`, `sub_questions`, `conversation_context`, ...) is read
  with `.get(..., default)`, and every new node/graph parameter has a safe
  default (`knowledge_graph=None`, `self_consistency_samples=1`,
  `enable_query_decomposition=False`, `read_only=True`). The graph behaves
  exactly like the original single-shot pipeline when the new features are
  left at their defaults, which is why all pre-existing tests pass unchanged.
- **Read-only is the default, not opt-in** — an LLM-generated `DELETE`/`DROP`
  reaching a real database is a much worse failure mode than a blocked
  legitimate write, so `SQL_READ_ONLY=true` ships as the out-of-the-box
  posture; callers that genuinely need writes (e.g. an agent-driven ETL
  step) opt out explicitly via `read_only=False`.
- **Decomposition and self-consistency are mutually exclusive per question** —
  both are LLM-call multipliers (`1 + N_subquestions` vs. `N_candidates`);
  stacking them would multiply cost further for unclear accuracy gain, so
  `decompose_sql` is a separate branch off `classify_complexity` rather than
  composing with `select_best_candidate`'s voting.
- **Grounding is cardinality-gated, not column-name-gated** — matching purely
  on column name (`status`, `category`, `country`, ...) doesn't generalise to
  an arbitrary schema; a distinct-value-count threshold does, at the cost of
  occasionally profiling a column that isn't semantically categorical (cheap
  and cached, so an acceptable trade).

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

Get a free Groq API key at <https://console.groq.com>. Prefer a different
provider? Set `LLM_PROVIDER=openai|anthropic|ollama` plus that provider's
API key/URL — see [Environment Variables](#environment-variables) — and
install the matching extra, e.g. `pip install -e ".[openai]"`.

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
python main.py --grounding  # print the auto-profiled value grounding index
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
| `GROQ_API_KEY` | *(required for SQL generation with the default provider)* | Your Groq API key |
| `DATABASE_URL` | `sqlite:///:memory:` | SQLAlchemy connection URL |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `MAX_RETRIES` | `3` | Max self-reflection retries |
| `ENABLE_QUERY_MEMORY` | `true` | Enable RL-inspired query memory |
| `QUERY_MEMORY_PATH` | `./query_memory.json` | Path to persist query memory |
| `ENABLE_METADATA_ENRICHMENT` | `true` | Enable the LLM-assisted glossary pass (heuristic descriptions always run) |
| `METADATA_CACHE_DIR` | `./.metadata_cache` | Where the auto-generated glossary cache is stored (keyed by schema signature) |
| `SELF_CONSISTENCY_SAMPLES` | `1` | SQL candidates sampled for non-trivial questions (`1` disables self-consistency) |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `anthropic` \| `ollama` — see `llm/factory.py` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | *(required if `LLM_PROVIDER=openai`)* / `gpt-4o-mini` | OpenAI credentials + model (needs `pip install -e ".[openai]"`) |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | *(required if `LLM_PROVIDER=anthropic`)* / `claude-3-5-sonnet-latest` | Anthropic credentials + model (needs `pip install -e ".[anthropic]"`) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.1` | Local Ollama endpoint + model, no API key needed (needs `pip install -e ".[ollama]"`) |
| `SQL_READ_ONLY` | `true` | Block LLM-generated `INSERT`/`UPDATE`/`DELETE`/DDL by default — see `database/safety.py` |
| `SQL_STATEMENT_TIMEOUT_SECONDS` | *(unset = no timeout)* | Best-effort per-query timeout |
| `ENABLE_VALUE_GROUNDING` | `true` | Profile low-cardinality text columns and match question terms against real values |
| `GROUNDING_CACHE_DIR` | `./.grounding_cache` | Where profiled distinct values are cached (keyed by schema signature) |
| `GROUNDING_MAX_TABLE_ROWS` | `200000` | Tables above this row count are never profiled |
| `GROUNDING_MAX_DISTINCT` | `500` | Max distinct values for a column to be treated as categorical |
| `GROUNDING_CARDINALITY_RATIO` | `0.75` | Max `distinct_count / row_count` for a column to be treated as categorical |
| `ENABLE_QUERY_DECOMPOSITION` | `false` | Route COMPLEX questions through sub-question decomposition + CTE composition instead of single-shot generation |

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

## Evaluation

"It answered my question correctly" isn't evidence a change actually helped —
the `eval/` package runs a golden-question **execution-accuracy** regression
suite against the bundled demo database: each question has a hand-computed
expected result, the generated SQL is executed, and the DataFrame is compared
order-independently (not an exact-SQL-text match, since many correct queries
are textually different).

```bash
python -m eval.cli --mock                    # zero API key, canned SQL — what CI runs
python -m eval.cli --live                     # real AgentContext.ask(), needs an LLM configured
python -m eval.cli --mock --fail-under 1.0    # non-zero exit code if accuracy drops below the threshold
make eval                                     # shortcut for --mock
```

`--mock` mode needs no network access or API key (it replays canned
`question -> SQL` mappings shipped alongside `eval/golden/demo_db.jsonl`), so
it's the required, always-on CI gate. `--live` mode drives the actual agent
end-to-end and is for manual/local benchmarking of real generation quality —
it is intentionally **not** wired into required CI, to avoid reintroducing an
API-key dependency into the test suite.

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
│   ├── graph.py         # LangGraph StateGraph + routing logic (incl. decompose_sql branch)
│   ├── memory.py        # RL-inspired query memory (few-shot store)
│   ├── nodes.py         # Individual graph node implementations
│   ├── scoring.py       # Shared order-independent result-signature comparison
│   └── state.py         # AgentState TypedDict
├── database/
│   ├── connector.py     # Universal SQLAlchemy connector (read-only enforcement, LIMIT injection)
│   ├── safety.py        # SQL statement classification (regex + optional sqlglot AST) and guardrails
│   └── schema.py        # Auto schema discovery, metadata, semantic table ranking
├── knowledge/
│   ├── graph.py          # SchemaKnowledgeGraph — declared/inferred FKs, join-path search
│   ├── grounding.py       # ValueGroundingIndex — profiled categorical column values
│   └── metadata.py        # MetadataEnricher — heuristic + LLM auto business glossary
├── retrieval/
│   └── semantic.py       # Dependency-free TF-IDF + cosine similarity index
├── llm/
│   ├── base.py           # Canonical LLMRunnable protocol
│   ├── factory.py        # get_llm(provider) dispatch (LLM_PROVIDER env var)
│   ├── groq_client.py    # Groq LLM factory (default provider)
│   ├── openai_client.py  # OpenAI LLM factory (optional extra)
│   ├── anthropic_client.py # Anthropic LLM factory (optional extra)
│   └── ollama_client.py  # Local Ollama LLM factory (optional extra)
├── prompts/
│   └── templates.py    # CoT / reflection / validation / complexity / decomposition / metadata prompts
└── utils/
    └── demo_data.py    # Demo database seeder
bootstrap.py            # Single entry point: connect → discover → KG → glossary → grounding → AgentContext
app.py                  # Streamlit web UI (knowledge graph, glossary, grounding, conversation panels)
main.py                 # CLI entry point (--graph / --glossary / --grounding inspection flags)
eval/
├── golden/demo_db.jsonl  # Golden question set with hand-computed expected results
├── metrics.py            # Execution-accuracy scoring
├── runner.py             # Evaluation loop (mock + live generators)
└── cli.py                # `python -m eval.cli` entry point
tests/
├── test_schema.py            # Connector + schema discovery tests
├── test_memory.py            # Query memory tests
├── test_agent_nodes.py       # Node unit tests (mock LLM)
├── test_graph.py             # Graph routing + compile tests
├── test_knowledge_graph.py   # Knowledge graph construction + join-path tests
├── test_metadata.py          # Heuristic + LLM metadata enrichment tests
├── test_semantic_retrieval.py# TF-IDF retrieval tests
├── test_self_consistency.py  # Complexity classification + candidate voting tests
├── test_bootstrap.py         # AgentContext orchestration tests
├── test_safety.py            # SQL statement classification + read-only enforcement tests
├── test_scoring.py           # Shared result-signature comparison tests
├── test_eval_harness.py      # Evaluation harness (metrics + runner + CLI) tests
├── test_llm_factory.py       # Pluggable LLM provider factory tests
├── test_grounding.py         # Value grounding profiling + matching tests
├── test_decomposition.py     # Query decomposition + CTE composition tests
└── test_conversation.py      # Multi-turn conversation context tests
```

Project tooling config lives in `pyproject.toml` (packaging, optional extras,
`ruff`, `pytest`, `coverage`), with `.pre-commit-config.yaml`, `.editorconfig`,
and a `Makefile` providing convenience shortcuts (`make test`, `make lint`,
`make format`, `make ui`, `make eval`) for contributors — see
[Contributing](#contributing).

---

## Roadmap

Ideas under consideration — see [open issues](https://github.com/subrata-samanta/universal-text2sql/issues)
for the current state and feel free to propose more:

- [ ] Optional dense-embedding backend for semantic retrieval (as an upgrade path from TF-IDF)
- [ ] Benchmark against public text-to-SQL datasets (Spider, BIRD) — the internal
      `eval/` golden-set harness covers regression testing today; a public-dataset
      leaderboard comparison is the next step
- [ ] Additional dialect-specific prompt guidance (BigQuery, Snowflake, MSSQL)
- [ ] Self-consistency voting over composed decomposition queries (currently
      decomposition and self-consistency are mutually exclusive per question)
- [ ] Semantic (not just substring) value grounding, e.g. "USA" ↔ "United States"

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
