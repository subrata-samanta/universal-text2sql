# Security Policy

## Supported Versions

This project is pre-1.0 and moves quickly. Security fixes are made against
the latest commit on `main` only.

| Version | Supported |
| --- | --- |
| `main` (latest) | ✅ |
| Older tags/releases | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report it privately by emailing **tsbrta@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce it (a minimal repro is very helpful)
- Any relevant logs, stack traces, or proof-of-concept code

You should receive an acknowledgement within **3 business days**. We'll work
with you to understand and validate the issue, develop a fix, and coordinate
a disclosure timeline before any public details are published. Please give us
a reasonable window to ship a fix before disclosing publicly.

## Scope Notes Specific to This Project

A few things worth knowing if you're evaluating this project's attack
surface:

- **SQL execution**: the agent executes LLM-generated SQL against the
  configured database. It does not sandbox or restrict statement types
  beyond what the underlying database user's permissions allow. **Always run
  this agent against a database user with read-only / least-privilege
  access** unless you have reviewed and trust the SQL generation path for
  your use case. Treat LLM output as untrusted input to your database.
- **Prompt injection**: table/column names, sample values, and stored query
  memory are interpolated into LLM prompts. A malicious or compromised
  database (e.g. attacker-controlled row data) could attempt to influence
  generated SQL via prompt injection. Report any reliable prompt-injection-
  to-SQL-injection chain as a vulnerability.
- **Secrets**: `GROQ_API_KEY` and database credentials are read from
  environment variables / `.env` and are never logged or persisted by the
  agent. If you find a code path that logs or caches a secret (including in
  `query_memory.json` or the metadata cache), please report it.
- **Dependencies**: this project depends on `langgraph`, `langchain`,
  `sqlalchemy`, and others. Vulnerabilities in those upstream packages should
  generally be reported to their respective maintainers, but feel free to
  flag them to us too if they materially affect this project.

Thank you for helping keep this project and its users safe.
