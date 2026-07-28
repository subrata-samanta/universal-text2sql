# Contributing to Universal Text-to-SQL Agent

Thanks for your interest in contributing! This project aims to be a
community-driven, state-of-the-art text-to-SQL agent that works against any
database with zero manual configuration. Contributions of all sizes are
welcome — bug fixes, new database dialect support, new research techniques,
documentation, tests, and issue triage.

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running the test suite](#running-the-test-suite)
- [Code style and linting](#code-style-and-linting)
- [Commit messages](#commit-messages)
- [Branching and pull requests](#branching-and-pull-requests)
- [Adding a new agent node](#adding-a-new-agent-node)
- [Adding a new knowledge/retrieval technique](#adding-a-new-knowledgeretrieval-technique)
- [Reporting bugs](#reporting-bugs)
- [Proposing features](#proposing-features)
- [Maintainers and review process](#maintainers-and-review-process)

## Ways to contribute

- 🐛 **Bug reports** — see [Reporting bugs](#reporting-bugs).
- ✨ **Feature proposals** — see [Proposing features](#proposing-features).
- 🧪 **Tests** — every module in this project is unit-tested with mocked
  LLMs and an in-memory SQLite database, so no API key is required to
  contribute.
- 📚 **Documentation** — README clarity, docstrings, and examples are all
  fair game.
- 🔬 **Research-backed techniques** — this project explicitly tracks recent
  text-to-SQL research (see the README's "State-of-the-art techniques"
  table). PRs that implement or improve on a published technique, with a
  link to the paper/approach, are especially welcome.
- 🗄️ **Database dialect support** — the agent works through SQLAlchemy, so
  most dialects work out of the box; contributions that add
  dialect-specific SQL guidance (e.g. BigQuery, Snowflake, MSSQL quirks) to
  the prompts are valuable.

## Development setup

```bash
git clone https://github.com/<your-fork>/universal-text2sql.git
cd universal-text2sql

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"   # ruff, pytest-cov, pre-commit

cp .env.example .env
# GROQ_API_KEY is only needed to actually run SQL generation;
# the full test suite runs without it.
```

Some features are behind optional extras so the core install stays
dependency-light — install them only if you're working on that area:

```bash
pip install -e ".[safety]"        # sqlglot, for AST-based SQL statement classification
pip install -e ".[fuzzy]"         # rapidfuzz, for stronger value-grounding matching
pip install -e ".[openai]"        # langchain-openai, for the OpenAI LLM provider
pip install -e ".[anthropic]"     # langchain-anthropic, for the Anthropic LLM provider
pip install -e ".[ollama]"        # langchain-ollama, for the local Ollama LLM provider
pip install -e ".[all-providers]" # openai + anthropic + ollama together
```

Each optional dependency is lazy-imported (inside the function that needs
it, never at module import time) with a clear `ImportError` naming the
correct extra — the test suite must keep passing with **none** of these
installed, so don't move an optional import to module scope.

Optionally install the pre-commit hooks so formatting/linting issues are
caught before you push:

```bash
pre-commit install
```

## Running the test suite

```bash
pytest tests/ -v
```

All 198+ tests run against an in-memory SQLite database and mock the LLM
where one is needed, so the full suite runs with **no API key and no
network access**. Please add or update tests for any behavior change —
see the existing `tests/test_*.py` files for the patterns used (mocked
`ChatGroq`-compatible LLMs via `langchain_core.runnables.RunnableLambda`,
an in-memory seeded SQLite fixture, etc.). Tests that exercise an optional
extra (e.g. sqlglot-based AST classification) use
`pytest.importorskip(...)` so the suite still passes when that extra isn't
installed.

Run with coverage:

```bash
pytest tests/ --cov=universal_text2sql --cov-report=term-missing
```

If your change affects SQL generation quality (prompt tweaks, new context
injected into the prompt, grounding/decomposition logic), also run the
evaluation harness — a golden-question execution-accuracy regression suite
that catches quality regressions unit tests won't:

```bash
make eval   # equivalent to: python -m eval.cli --mock
```

This runs in mocked-LLM mode (no API key needed) against
`eval/golden/demo_db.jsonl` and is part of CI. If you add a genuinely new
kind of question the agent should handle, consider adding a case to the
golden set.

## Code style and linting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and
formatting (config lives in `pyproject.toml`).

```bash
ruff check .          # lint
ruff format .         # format
ruff check . --fix    # auto-fix what can be auto-fixed
```

General style notes, consistent with the existing codebase:

- Type hints on all public function signatures.
- Prefer `.get(key, default)` over `state[key]` when reading `AgentState` in
  a graph node — new optional fields must not break older callers.
- No commented-out code, no debug prints left behind.
- Keep node functions in `agent/nodes.py` as plain
  `(state, **deps) -> partial_state_update` callables — no hidden state,
  no side effects beyond what's documented (DB execution, LLM calls, memory
  writes).
- Docstrings should explain *why*, not restate the function name in prose.

## Commit messages

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add BigQuery dialect hints to SQL generation prompt
fix: correct join direction in inferred FK detection
docs: clarify SELF_CONSISTENCY_SAMPLES cost tradeoff in README
test: add coverage for empty schema knowledge graph
refactor: extract TF-IDF tokenizer into its own function
```

This isn't strictly enforced, but it makes the changelog and history much
easier to skim.

## Branching and pull requests

1. Fork the repo and create a branch off `main`:
   `git checkout -b feat/short-description`.
2. Make your change, with tests. Run `pytest` and `ruff check .` locally.
3. Open a PR against `main`. Fill in the PR template — link any related
   issue, describe *why* the change is needed (not just what changed), and
   include a test plan.
4. CI (lint + test matrix) must pass before review.
5. A maintainer will review, request changes if needed, and merge once
   approved. Small, focused PRs get reviewed faster than large ones.

If you're proposing a larger architectural change (a new agent layer, a
different LLM provider abstraction, etc.), please open an issue to discuss
the approach first — it saves everyone rework.

## Adding a new agent node

The LangGraph agent (`universal_text2sql/agent/`) is designed to make new
nodes cheap to add:

1. Write a plain function in `agent/nodes.py`:
   `def my_node(state: AgentState, **deps) -> dict[str, Any]`, returning only
   the keys it changes.
2. Add any new state field to `AgentState` in `agent/state.py` — give
   callers of your node a sensible default and read it elsewhere with
   `.get(...)` so existing graphs/tests that don't set it keep working.
3. Wire it into `agent/graph.py`: bind dependencies with
   `functools.partial`, `add_node(...)`, and connect edges (`add_edge` or
   `add_conditional_edges` for branching).
4. Add unit tests calling the node function directly with a mocked LLM
   (see `tests/test_self_consistency.py` for the pattern), plus, if it
   changes routing, a `build_graph()` compile-sanity test.

## Adding a new knowledge/retrieval technique

The `knowledge/` and `retrieval/` packages are where dataset-agnostic
"figure this database out automatically" logic lives:

- `knowledge/graph.py` — schema-graph construction and join-path reasoning.
- `knowledge/metadata.py` — auto-generated business glossary.
- `retrieval/semantic.py` — similarity ranking for schema linking / few-shot
  recall.

If you're implementing a technique from a paper, please reference it (a
short comment or a README table row is enough) so others can find the
original source. Keep new techniques **opt-in and gracefully degrading** —
the project's guiding constraint is that it must still work with zero
configuration and no LLM configured (see the heuristic fallback in
`MetadataEnricher` for the pattern to follow).

## Reporting bugs

Please use the **Bug report** issue template and include:

- What you expected to happen vs. what actually happened
- The database backend/URL scheme (no need to share credentials — just
  `postgresql://...` vs `sqlite://...`, etc.)
- The question asked and the SQL the agent generated, if relevant
- Whether it reproduces against the bundled demo database
  (`python main.py "..."` with no `DATABASE_URL` set)

## Proposing features

Please use the **Feature request** issue template. If your proposal is
inspired by a specific paper or existing tool, link it — it helps reviewers
evaluate the trade-offs quickly and keeps the project grounded in the
"state-of-the-art techniques" table in the README.

## Maintainers and review process

This project is currently maintained by [@subrata-samanta](https://github.com/subrata-samanta).
Maintainers are responsible for triaging issues, reviewing PRs, and cutting
releases (see [CHANGELOG.md](CHANGELOG.md)). As the contributor base grows,
active, consistent contributors may be invited to become maintainers.

Security issues should **not** go through the normal issue/PR flow — see
[SECURITY.md](SECURITY.md).
