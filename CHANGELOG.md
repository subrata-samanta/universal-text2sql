# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) once a
`1.0.0` release is cut. Pre-1.0, minor versions may include breaking changes.

## [Unreleased]

## [0.3.0] — Safety, pluggable providers, grounding, decomposition, evaluation, conversation

### Added
- `database/safety.py` — SQL statement classification (regex baseline, plus
  an optional `sqlglot` AST mode that also catches writes hidden inside a
  CTE) and read-only enforcement. `DatabaseConnector` now defaults to
  `read_only=True` (`SQL_READ_ONLY`), blocking LLM-generated
  `INSERT`/`UPDATE`/`DELETE`/DDL with a `SQLSafetyError` unless explicitly
  disabled; `execute_query` also supports a `limit=`/`timeout_seconds=`
  bound for exploratory/self-consistency executions.
- `llm/` pluggable provider layer — `llm/base.py`'s canonical `LLMRunnable`
  protocol, `llm/factory.py::get_llm(provider)` resolving `LLM_PROVIDER`
  (`groq` default), and lazy-imported `openai_client.py`/`anthropic_client.py`/
  `ollama_client.py` clients, each behind its own `pyproject.toml` extra
  (`openai`, `anthropic`, `ollama`, `all-providers`).
- `knowledge/grounding.py` — `ValueGroundingIndex`: profiles low-cardinality
  text columns (cardinality- and row-count-gated, cached to disk by schema
  signature) and matches question terms against real stored values, so
  filter literals come from the database instead of an LLM guess. Wired
  into `select_schema` as `grounding_hints`, on by default
  (`ENABLE_VALUE_GROUNDING`).
- Query decomposition — `decompose_sql` agent node, a DIN-SQL-style
  alternate branch off `classify_complexity` for COMPLEX questions: plans
  ordered sub-questions, generates a SQL fragment per step, and composes a
  `WITH step_1 AS (...), ...` query, falling back to single-shot generation
  on any step failure. Opt-in via `ENABLE_QUERY_DECOMPOSITION`.
- `eval/` — a golden-question execution-accuracy regression harness
  (`python -m eval.cli --mock|--live [--tag] [--fail-under]`, `make eval`),
  zero-API-key `--mock` mode wired into CI, `--live` mode for real
  benchmarking against a configured LLM.
- Multi-turn conversational follow-ups — `AgentContext.ask(question,
  conversation_history=...)` threads prior `{question, sql, answer}` turns
  into the SQL generation prompt so e.g. "now filter that by country"
  resolves against the previous turn. Caller-owned history in both the CLI
  interactive loop and the Streamlit session state.
- `agent/scoring.py` — shared order-independent result-signature comparison,
  extracted from `select_best_candidate` and reused by the evaluation
  harness.
- `main.py --grounding` — prints the auto-profiled value grounding index.
- New environment variables: `LLM_PROVIDER`, `OPENAI_API_KEY`/`OPENAI_MODEL`,
  `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`/`OLLAMA_MODEL`,
  `SQL_READ_ONLY`, `SQL_STATEMENT_TIMEOUT_SECONDS`, `ENABLE_VALUE_GROUNDING`,
  `GROUNDING_CACHE_DIR`, `GROUNDING_MAX_TABLE_ROWS`, `GROUNDING_MAX_DISTINCT`,
  `GROUNDING_CARDINALITY_RATIO`, `ENABLE_QUERY_DECOMPOSITION`.
- Open-source project scaffolding: `LICENSE` (MIT), `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CONTRIBUTING.md`, issue/PR templates, CI workflow
  (lint + test matrix), `pyproject.toml` packaging/tooling config,
  `.pre-commit-config.yaml`, `.editorconfig`, `Makefile`, `CODEOWNERS`.
- 109 new tests (198 total) covering SQL safety, the LLM provider factory,
  value grounding, query decomposition, the evaluation harness, and
  multi-turn conversation context.

### Changed
- `SECURITY.md` updated to describe the now-enforced read-only default
  instead of just recommending it.
- All new `AgentState` fields (`grounding_hints`, `sub_questions`,
  `decomposition_steps`, `conversation_context`) and node/graph parameters
  are additive with safe defaults — zero-config behavior is unchanged.

## [0.2.0] — Knowledge graph, auto glossary, and self-consistency

### Added
- `knowledge/graph.py` — `SchemaKnowledgeGraph`: builds a graph of the
  schema from declared foreign keys, naming-convention-*inferred*
  relationships, and cross-table "semantic sibling" columns; supports
  multi-hop join-path search between tables.
- `knowledge/metadata.py` — `MetadataEnricher`: auto-generates a business
  glossary per table/column (heuristic descriptions always, optional
  LLM-assisted descriptions/synonyms cached to disk by schema signature).
- `retrieval/semantic.py` — dependency-free TF-IDF + cosine similarity
  index used for schema linking, replacing pure keyword overlap.
- `classify_complexity` agent node — SIMPLE/MODERATE/COMPLEX routing.
- `select_best_candidate` agent node — self-consistency: sample multiple
  SQL candidates for non-trivial questions, execute all of them, and pick
  the majority-agreeing result.
- `bootstrap.py` — single entry point (`connect → discover → knowledge
  graph → glossary → AgentContext`) wired into the CLI (`main.py --graph`
  / `--glossary`) and the Streamlit UI (knowledge graph + glossary panels).
- New environment variables: `ENABLE_METADATA_ENRICHMENT`,
  `METADATA_CACHE_DIR`, `SELF_CONSISTENCY_SAMPLES`.
- 49 new tests covering the knowledge graph, metadata enrichment, semantic
  retrieval, self-consistency voting, and bootstrap modules.

### Changed
- `select_schema` now ranks tables by TF-IDF semantic similarity (with a
  keyword-overlap fallback) instead of pure keyword overlap, and injects
  knowledge-graph join context and the auto-generated glossary into the SQL
  generation/reflection prompts.
- `ColumnMetadata`/`TableMetadata` gained `business_meaning`/`synonyms`
  fields (populated by `MetadataEnricher`).

### Removed
- Unused `sam.py` scratch file and stray debug prints in `main.py`.

## [0.1.0] — Initial agent

### Added
- Universal SQLAlchemy database connector (SQLite, PostgreSQL, MySQL, ...).
- Automatic schema discovery: tables, columns, types, PK/FK, row counts,
  sample values.
- LangGraph agentic workflow: `select_schema → generate_sql → execute_sql
  → validate_result → format_answer → store_memory`, with a
  `reflect`-powered self-correction loop on execution errors or failed
  validation (configurable `MAX_RETRIES`).
- Groq LLM integration (`llama-3.3-70b-versatile` by default).
- Chain-of-thought SQL generation prompting.
- RL-inspired query memory: successful queries are stored with
  `reward = 1 / (1 + retry_count)` and replayed as few-shot examples.
- Streamlit web UI and CLI entry point.
- Initial test suite (40 tests) covering the connector, schema discovery,
  agent nodes, and graph routing logic.

[Unreleased]: https://github.com/subrata-samanta/universal-text2sql/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/subrata-samanta/universal-text2sql/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/subrata-samanta/universal-text2sql/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/subrata-samanta/universal-text2sql/releases/tag/v0.1.0
