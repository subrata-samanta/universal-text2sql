"""Groq LLM client wrapper."""

from __future__ import annotations

import logging
import os

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_TEMPERATURE = 0.0


def get_groq_llm(
    model: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    api_key: str | None = None,
) -> ChatGroq:
    """Return a configured :class:`ChatGroq` instance.

    Args:
        model: Groq model name. Falls back to the ``GROQ_MODEL`` env var, then
            :data:`_DEFAULT_MODEL`.
        temperature: Sampling temperature (``0.0`` for deterministic SQL).
        api_key: Groq API key. Falls back to the ``GROQ_API_KEY`` env var.

    Returns:
        A ready-to-use :class:`~langchain_groq.ChatGroq` chat model.
    """
    resolved_model = model or os.getenv("GROQ_MODEL", _DEFAULT_MODEL)
    resolved_key = api_key or os.getenv("GROQ_API_KEY", "")

    if not resolved_key:
        logger.warning(
            "GROQ_API_KEY is not set. Set it in your environment or .env file."
        )

    llm = ChatGroq(
        model=resolved_model,
        temperature=temperature,
        api_key=resolved_key,
        max_retries=2,
    )
    logger.debug("Groq LLM initialised: model=%s temperature=%s", resolved_model, temperature)
    return llm
