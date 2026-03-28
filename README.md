# Universal Text-to-SQL Agent

A fully agentic, **universal** Natural Language → SQL system that:

- 🔌 **Works with any database** – SQLite, PostgreSQL, MySQL (anything SQLAlchemy supports)
- 🗺️ **Auto-discovers schema** – reads tables, columns, types, primary/foreign keys, and sample values at startup
- 🤖 **LangGraph workflow** – multi-node agentic graph with conditional routing
- ⚡ **Groq LLM** – ultra-fast inference with `llama-3.3-70b-versatile` (or any Groq model)
- 🔄 **Self-reflection** – automatically detects SQL errors and retries with corrected queries
- 🧠 **RL-inspired query memory** – successful queries are stored with a reward signal and replayed as few-shot examples for future questions
- 💬 **Streamlit UI** – interactive web interface + CLI

---

## Architecture

```
START
  │
  ▼
select_schema      ← keyword-based table relevance scoring + few-shot retrieval
  │
  ▼
generate_sql       ← Chain-of-Thought prompting + dynamic few-shot injection (Groq)
  │
  ▼
execute_sql        ← run against live database via SQLAlchemy
  │
  ├─ error ──► reflect ──► execute_sql   (up to MAX_RETRIES self-reflection loops)
  │
  ▼
validate_result    ← LLM checks whether the result actually answers the question
  │
  ├─ invalid ──► reflect ──► execute_sql
  │
  ▼
format_answer      ← convert DataFrame → natural language answer
  │
  ▼
store_memory       ← RL reward: reward = 1 / (1 + retry_count); save for few-shot
  │
  ▼
END
```

### Key techniques

| Technique | Implementation |
|---|---|
| Chain-of-Thought (CoT) | System prompt instructs step-by-step table→column→join reasoning |
| Few-shot prompting | Top-k similar past queries injected dynamically |
| Self-reflection | Error + previous SQL → corrected SQL loop (configurable retries) |
| RL-inspired reward | `reward = 1/(1+retries)` weights memory entries; zero-retry queries surface first |
| Schema linking | Keyword overlap scores tables so only relevant schema is in the prompt |
| Result validation | Separate LLM call checks whether the answer actually makes sense |

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

Open <http://localhost:8501> in your browser.

### 3b. CLI (interactive)

```bash
python main.py
```

### 3c. CLI (single query)

```bash
python main.py "What are the top 3 products by revenue?"
```

### 3d. Use as a library

```python
from dotenv import load_dotenv
load_dotenv()

from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import SchemaDiscovery
from universal_text2sql.agent.memory import QueryMemory
from universal_text2sql.agent.graph import run_query

connector = DatabaseConnector("sqlite:///./mydb.sqlite3")
schema = SchemaDiscovery(connector).discover()
memory = QueryMemory()

result = run_query(
    question="How many customers signed up last month?",
    connector=connector,
    schema=schema,
    memory=memory,
)
print(result["final_answer"])
print(result["generated_sql"])
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `DATABASE_URL` | `sqlite:///:memory:` | SQLAlchemy connection URL |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `MAX_RETRIES` | `3` | Max self-reflection retries |
| `ENABLE_QUERY_MEMORY` | `true` | Enable RL-inspired query memory |
| `QUERY_MEMORY_PATH` | `./query_memory.json` | Path to persist query memory |

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

All tests run against an in-memory SQLite database and mock the Groq LLM so **no API key is required**.

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
│   └── schema.py       # Auto schema discovery & metadata
├── llm/
│   └── groq_client.py  # Groq LLM factory
├── prompts/
│   └── templates.py    # CoT / reflection / validation prompts
└── utils/
    └── demo_data.py    # Demo database seeder
app.py                  # Streamlit web UI
main.py                 # CLI entry point
tests/
├── test_schema.py      # Connector + schema discovery tests
├── test_memory.py      # Query memory tests
├── test_agent_nodes.py # Node unit tests (mock LLM)
└── test_graph.py       # Graph routing + compile tests
```
