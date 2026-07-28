## Summary

<!-- What does this PR do, and why? Link any related issue (e.g. "Closes #123"). -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / internal improvement
- [ ] Test coverage
- [ ] Other (describe above)

## Test plan

<!-- How did you verify this? Prefer checkboxes over prose. -->

- [ ] `pytest tests/ -v` passes locally
- [ ] `ruff check .` passes locally
- [ ] Added/updated tests for the behavior change
- [ ] Manually verified against the bundled demo database (`python main.py "..."`) and/or a real database, if applicable

## Compatibility

- [ ] This does not require an API key / LLM to keep existing zero-config behavior working (or the trade-off is explained above)
- [ ] New `AgentState` fields (if any) are read with `.get(..., default)` and don't break callers that omit them
