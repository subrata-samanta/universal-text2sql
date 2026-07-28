"""Shared LLM protocol used across the agent, knowledge, and eval modules."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMRunnable(Protocol):
    """Minimal protocol for a LangChain-compatible chat LLM.

    Both :class:`~langchain_groq.ChatGroq` and
    :class:`~langchain_core.runnables.RunnableLambda` satisfy this protocol,
    as does every provider client under :mod:`universal_text2sql.llm`
    (:func:`~universal_text2sql.llm.openai_client.get_openai_llm`,
    :func:`~universal_text2sql.llm.anthropic_client.get_anthropic_llm`,
    :func:`~universal_text2sql.llm.ollama_client.get_ollama_llm`).
    """

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        ...

    def __or__(self, other: Any) -> Any:
        ...
