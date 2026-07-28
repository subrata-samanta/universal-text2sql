"""Anthropic LLM client wrapper (optional provider).

Install with ``pip install universal-text2sql[anthropic]``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-3-5-sonnet-latest"
_DEFAULT_TEMPERATURE = 0.0


def get_anthropic_llm(
    model: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    api_key: str | None = None,
) -> Any:
    """Return a configured :class:`~langchain_anthropic.ChatAnthropic` instance.

    Args:
        model: Anthropic model name. Falls back to the ``ANTHROPIC_MODEL``
            env var, then :data:`_DEFAULT_MODEL`.
        temperature: Sampling temperature (``0.0`` for deterministic SQL).
        api_key: Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY``
            env var.

    Raises:
        ImportError: If the optional ``langchain-anthropic`` package isn't
            installed.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' LLM provider requires the langchain-anthropic package. "
            "Install it with: pip install universal-text2sql[anthropic]"
        ) from exc

    resolved_model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    if not resolved_key:
        logger.warning("ANTHROPIC_API_KEY is not set. Set it in your environment or .env file.")

    llm = ChatAnthropic(model=resolved_model, temperature=temperature, api_key=resolved_key)
    logger.debug("Anthropic LLM initialised: model=%s temperature=%s", resolved_model, temperature)
    return llm
