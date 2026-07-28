"""OpenAI LLM client wrapper (optional provider).

Install with ``pip install universal-text2sql[openai]``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TEMPERATURE = 0.0


def get_openai_llm(
    model: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    api_key: str | None = None,
) -> Any:
    """Return a configured :class:`~langchain_openai.ChatOpenAI` instance.

    Args:
        model: OpenAI model name. Falls back to the ``OPENAI_MODEL`` env
            var, then :data:`_DEFAULT_MODEL`.
        temperature: Sampling temperature (``0.0`` for deterministic SQL).
        api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY`` env var.

    Raises:
        ImportError: If the optional ``langchain-openai`` package isn't
            installed.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' LLM provider requires the langchain-openai package. "
            "Install it with: pip install universal-text2sql[openai]"
        ) from exc

    resolved_model = model or os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")

    if not resolved_key:
        logger.warning("OPENAI_API_KEY is not set. Set it in your environment or .env file.")

    llm = ChatOpenAI(model=resolved_model, temperature=temperature, api_key=resolved_key)
    logger.debug("OpenAI LLM initialised: model=%s temperature=%s", resolved_model, temperature)
    return llm
