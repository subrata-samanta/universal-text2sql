"""Tests for the dependency-free TF-IDF semantic retrieval used for
schema linking and few-shot query recall.
"""

from __future__ import annotations

from universal_text2sql.retrieval.semantic import SemanticIndex, rank_documents, tokenize


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Total Revenue") == ["total", "revenue"]

    def test_splits_camel_case(self):
        assert "customer" in tokenize("customerId")
        assert "id" in tokenize("customerId")

    def test_splits_snake_case(self):
        assert tokenize("total_amount") == ["total", "amount"]

    def test_removes_stopwords(self):
        tokens = tokenize("How many customers are there?")
        assert "how" not in tokens
        assert "are" not in tokens
        assert "customers" in tokens

    def test_empty_string(self):
        assert tokenize("") == []


class TestSemanticIndex:
    def test_ranks_matching_document_highest(self):
        docs = {
            "customers": "customers first_name last_name email country",
            "products": "products name category price stock",
            "orders": "orders order_date status total_amount",
        }
        index = SemanticIndex.from_documents(docs)
        ranked = index.rank("What countries are customers from?")
        assert ranked[0][0] == "customers"
        assert ranked[0][1] > 0

    def test_top_k_limits_results(self):
        docs = {"a": "alpha beta", "b": "beta gamma", "c": "gamma delta"}
        index = SemanticIndex.from_documents(docs)
        ranked = index.rank("beta", top_k=2)
        assert len(ranked) == 2

    def test_no_overlap_scores_zero(self):
        docs = {"a": "totally unrelated content here"}
        index = SemanticIndex.from_documents(docs)
        ranked = index.rank("zzz nonexistent qqq")
        assert ranked[0][1] == 0.0

    def test_empty_documents_returns_empty(self):
        index = SemanticIndex.from_documents({})
        assert index.rank("anything") == []

    def test_rank_documents_convenience_function(self):
        docs = {"a": "revenue total sales", "b": "shipping address city"}
        ranked = rank_documents("total revenue by sales", docs)
        assert ranked[0][0] == "a"
