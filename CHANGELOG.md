# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) once a
`1.0.0` release is cut. Pre-1.0, minor versions may include breaking changes.

## [Unreleased]

### Added
- Open-source project scaffolding: `LICENSE` (MIT), `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CONTRIBUTING.md`, issue/PR templates, CI workflow
  (lint + test matrix), `pyproject.toml` packaging/tooling config,
  `.pre-commit-config.yaml`, `.editorconfig`, `Makefile`, `CODEOWNERS`.

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

[Unreleased]: https://github.com/subrata-samanta/universal-text2sql/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/subrata-samanta/universal-text2sql/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/subrata-samanta/universal-text2sql/releases/tag/v0.1.0
