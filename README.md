# Universal Text-to-SQL Agent

A fully agentic, **universal** Natural Language → SQL system that works against
**any** database with zero manual configuration: it discovers its own schema,
builds its own knowledge graph, and writes its own business glossary.

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

## Architecture

```
START
  │
  ▼
select_schema        ← semantic (TF-IDF) schema linking + knowledge-graph join context + glossary
  │
  ▼
classify_complexity  ← SIMPLE / MODERATE / COMPLEX routing
  │
  ▼
generate_sql         ← CoT + few-shot generation; samples multiple candidates for non-trivial questions
  │
  ▼
select_best_candidate← self-consistency: executes every candidate, majority vote wins
  │
  ▼
execute_sql          ← run against live database via SQLAlchemy
  │
  ├─ error ──► reflect ──► execute_sql   (up to MAX_RETRIES self-reflection loops)
  │
  ▼
validate_result      ← LLM checks whether the result actually answers the question
  │
  ├─ invalid ──► reflect ──► execute_sql
  │
  ▼
format_answer        ← convert DataFrame → natural language answer
  │
  ▼
store_memory         ← RL reward: reward = 1 / (1 + retry_count); save for few-shot
  │
  ▼
END
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
