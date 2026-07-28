"""Lightweight semantic retrieval used for schema linking and few-shot recall.

State-of-the-art text-to-SQL pipelines (CHESS, DIN-SQL, DAIL-SQL) lean on
embedding similarity for two jobs this agent also needs:

1. **Schema linking** — deciding which tables/columns are relevant to a
   question, using more than literal substring overlap (e.g. matching the
   word "revenue" to a column whose auto-generated business meaning says
   "total order amount").
2. **Few-shot retrieval** — finding the most similar past successful
   queries to inject as in-context examples (dynamic few-shot prompting).

Pulling in a real embedding model (sentence-transformers, OpenAI/Groq
embeddings, ...) would add a heavy dependency and, for Groq, an API this
project doesn't otherwise need. Since schema/glossary text and question
text are both short, a classic TF-IDF + cosine-similarity vector space
model gets ~90% of the benefit of dense embeddings for this use case
while staying dependency-free and fully deterministic (useful for tests).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "for", "to", "and", "or", "with", "by", "at", "from", "that",
    "this", "it", "as", "do", "does", "did", "how", "what", "which", "who",
    "many", "much", "show", "me", "please", "list", "get", "find",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenization with light camelCase/snake_case splitting."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class SemanticIndex:
    """A tiny in-memory TF-IDF index over a fixed set of documents."""

    doc_ids: list[str] = field(default_factory=list)
    _vectors: dict[str, Counter] = field(default_factory=dict)
    _idf: dict[str, float] = field(default_factory=dict)
    _norms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, documents: dict[str, str]) -> SemanticIndex:
        """Build an index from ``{doc_id: text}``."""
        index = cls()
        if not documents:
            return index

        index.doc_ids = list(documents.keys())
        term_doc_freq: Counter = Counter()
        raw_term_freqs: dict[str, Counter] = {}

        for doc_id, text in documents.items():
            tokens = tokenize(text)
            tf = Counter(tokens)
            raw_term_freqs[doc_id] = tf
            for term in tf:
                term_doc_freq[term] += 1

        n_docs = len(documents)
        idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in term_doc_freq.items()
        }
        index._idf = idf

        for doc_id, tf in raw_term_freqs.items():
            vec = Counter({term: count * idf[term] for term, count in tf.items()})
            index._vectors[doc_id] = vec
            index._norms[doc_id] = math.sqrt(sum(w * w for w in vec.values())) or 1.0

        return index

    def _query_vector(self, query: str) -> Counter:
        tf = Counter(tokenize(query))
        return Counter({term: count * self._idf.get(term, 0.0) for term, count in tf.items()})

    def rank(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score)]`` sorted by cosine similarity, descending.

        Documents that share zero terms with the query score ``0.0`` and are
        still included (callers typically want a fallback ranking).
        """
        if not self.doc_ids:
            return []

        q_vec = self._query_vector(query)
        q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0

        scores: list[tuple[str, float]] = []
        for doc_id in self.doc_ids:
            doc_vec = self._vectors.get(doc_id, Counter())
            if not doc_vec or not q_vec:
                scores.append((doc_id, 0.0))
                continue
            # Dot product over the smaller vector's keys.
            shorter, longer = (q_vec, doc_vec) if len(q_vec) < len(doc_vec) else (doc_vec, q_vec)
            dot = sum(w * longer.get(term, 0.0) for term, w in shorter.items())
            denom = q_norm * self._norms.get(doc_id, 1.0)
            scores.append((doc_id, dot / denom if denom else 0.0))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k] if top_k is not None else scores


def rank_documents(query: str, documents: dict[str, str], top_k: int | None = None) -> list[tuple[str, float]]:
    """Convenience one-shot ranking without keeping an index around."""
    return SemanticIndex.from_documents(documents).rank(query, top_k=top_k)
