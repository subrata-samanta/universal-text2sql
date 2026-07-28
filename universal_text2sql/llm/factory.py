"""LLM provider factory: swap between Groq/OpenAI/Anthropic/Ollama via config.

The agent's node functions (:mod:`universal_text2sql.agent.nodes`) only
depend on the structural :class:`~universal_text2sql.llm.base.LLMRunnable`
protocol, so nothing about the graph itself needs to change to support a
different provider -- only which concrete client gets constructed at
bootstrap time.

Each provider's own SDK (``langchain-openai``, ``langchain-anthropic``,
``langchain-ollama``) is an *optional* dependency, lazily imported only when
that provider is actually selected -- ``from universal_text2sql.llm.factory
import get_llm`` never requires any of them to be installed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from universal_text2sql.llm.base import LLMRunnable

_DEFAULT_PROVIDER = "groq"


def _load_groq(**kwargs: Any) -> LLMRunnable:
    from universal_text2sql.llm.groq_client import get_groq_llm

    return get_groq_llm(**kwargs)


def _load_openai(**kwargs: Any) -> LLMRunnable:
    from universal_text2sql.llm.openai_client import get_openai_llm

    return get_openai_llm(**kwargs)


def _load_anthropic(**kwargs: Any) -> LLMRunnable:
    from universal_text2sql.llm.anthropic_client import get_anthropic_llm

    return get_anthropic_llm(**kwargs)


def _load_ollama(**kwargs: Any) -> LLMRunnable:
    from universal_text2sql.llm.ollama_client import get_ollama_llm

    return get_ollama_llm(**kwargs)


_PROVIDERS: dict[str, Callable[..., LLMRunnable]] = {
    "groq": _load_groq,
    "openai": _load_openai,
    "anthropic": _load_anthropic,
    "ollama": _load_ollama,
}


def get_llm(provider: str | None = None, **kwargs: Any) -> LLMRunnable:
    """Build a chat LLM for *provider* (or the ``LLM_PROVIDER`` env var, default ``groq``).

    Args:
        provider: One of ``"groq"``, ``"openai"``, ``"anthropic"``,
            ``"ollama"`` (case-insensitive). Falls back to the
            ``LLM_PROVIDER`` env var, then ``"groq"`` -- so calling
            ``get_llm()`` with no arguments and no env var set behaves
            identically to calling :func:`~universal_text2sql.llm.groq_client.get_groq_llm`
            directly.
        **kwargs: Forwarded to the resolved provider's client function
            (e.g. ``model=``, ``temperature=``, ``api_key=``/``base_url=``).

    Returns:
        A ready-to-use chat model satisfying
        :class:`~universal_text2sql.llm.base.LLMRunnable`.

    Raises:
        ValueError: If *provider* isn't one of the supported names.
        ImportError: If the resolved provider's optional SDK isn't
            installed (the error message names the correct ``pip install``
            extra).
    """
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER)).lower()
    loader = _PROVIDERS.get(resolved_provider)
    if loader is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{resolved_provider}'. Supported: {sorted(_PROVIDERS)}"
        )
    return loader(**kwargs)
