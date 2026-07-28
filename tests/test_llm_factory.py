"""Tests for the pluggable LLM provider factory."""

from __future__ import annotations

import importlib.util

import pytest
from langchain_core.runnables import RunnableLambda

from universal_text2sql.llm.base import LLMRunnable
from universal_text2sql.llm.factory import get_llm


def _skip_if_installed(module_name: str) -> None:
    """Skip a test that only makes sense when *module_name* is NOT installed.

    Contributors who've installed a provider extra locally (e.g.
    ``pip install -e ".[all-providers]"``) shouldn't see these fail --
    they exercise the ImportError-guidance path, which only triggers when
    the optional package is genuinely absent (the default CI environment).
    """
    if importlib.util.find_spec(module_name) is not None:
        pytest.skip(f"{module_name} is installed in this environment")


class TestGetLlm:
    def test_unset_provider_defaults_to_groq(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        llm = get_llm()
        assert type(llm).__name__ == "ChatGroq"

    def test_explicit_groq_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        llm = get_llm(provider="groq")
        assert type(llm).__name__ == "ChatGroq"

    def test_provider_case_insensitive(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        llm = get_llm(provider="GROQ")
        assert type(llm).__name__ == "ChatGroq"

    def test_env_var_selects_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        llm = get_llm()
        assert type(llm).__name__ == "ChatGroq"

    def test_explicit_provider_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        # Explicit provider="groq" wins over LLM_PROVIDER=openai, and since
        # langchain-openai is never touched, this must not raise ImportError.
        llm = get_llm(provider="groq")
        assert type(llm).__name__ == "ChatGroq"

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm(provider="not-a-real-provider")

    def test_optional_provider_without_package_raises_clear_import_error(self):
        # This repo's default dev environment doesn't install langchain-openai,
        # so this exercises the real lazy-import failure path.
        _skip_if_installed("langchain_openai")
        with pytest.raises(ImportError, match=r"pip install universal-text2sql\[openai\]"):
            get_llm(provider="openai")

    def test_ollama_without_package_raises_clear_import_error(self):
        _skip_if_installed("langchain_ollama")
        with pytest.raises(ImportError, match=r"pip install universal-text2sql\[ollama\]"):
            get_llm(provider="ollama")

    def test_anthropic_without_package_raises_clear_import_error(self):
        _skip_if_installed("langchain_anthropic")
        with pytest.raises(ImportError, match=r"pip install universal-text2sql\[anthropic\]"):
            get_llm(provider="anthropic")

    def test_kwargs_forwarded_to_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        llm = get_llm(provider="groq", model="llama-3.1-8b-instant", temperature=0.5)
        assert llm.model_name == "llama-3.1-8b-instant"
        assert llm.temperature == 0.5


class TestLLMRunnableProtocol:
    def test_runnable_lambda_satisfies_protocol(self):
        assert isinstance(RunnableLambda(lambda x: x), LLMRunnable)

    def test_plain_object_does_not_satisfy_protocol(self):
        assert not isinstance(object(), LLMRunnable)
