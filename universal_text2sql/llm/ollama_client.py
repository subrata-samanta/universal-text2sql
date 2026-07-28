"""Local Ollama LLM client wrapper (optional provider).

No API key required -- just a reachable Ollama server. Install with
``pip install universal-text2sql[ollama]``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_BASE_URL = "http://localhost:11434"


def get_ollama_llm(
    model: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    base_url: str | None = None,
) -> Any:
    """Return a configured :class:`~langchain_ollama.ChatOllama` instance.

    Args:
        model: Ollama model name (must already be pulled locally, e.g.
            ``ollama pull llama3.1``). Falls back to the ``OLLAMA_MODEL``
            env var, then :data:`_DEFAULT_MODEL`.
        temperature: Sampling temperature (``0.0`` for deterministic SQL).
        base_url: Ollama server URL. Falls back to the ``OLLAMA_BASE_URL``
            env var, then ``http://localhost:11434``.

    Raises:
        ImportError: If the optional ``langchain-ollama`` package isn't
            installed.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "The 'ollama' LLM provider requires the langchain-ollama package. "
            "Install it with: pip install universal-text2sql[ollama]"
        ) from exc

    resolved_model = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
    resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)

    llm = ChatOllama(model=resolved_model, temperature=temperature, base_url=resolved_base_url)
    logger.debug(
        "Ollama LLM initialised: model=%s base_url=%s temperature=%s",
        resolved_model,
        resolved_base_url,
        temperature,
    )
    return llm
