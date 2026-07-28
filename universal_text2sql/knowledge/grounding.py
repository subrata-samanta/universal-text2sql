"""Value / entity grounding (CHESS-style literal grounding).

An LLM generating a `WHERE country = 'USA'` clause has to guess the exact
stored spelling — is it ``"USA"``, ``"United States"``, ``"US"``? The schema
DDL and even the auto-generated glossary don't tell it. This module closes
that gap by profiling low-cardinality text columns once (at bootstrap time,
cached to disk) and, per question, matching question terms against the real
distinct values so the SQL-generation prompt can say "the question mentions
'usa', which matches the real value 'USA' in customers.country" instead of
making the LLM guess.

Two stages:

1. **Profiling** (:meth:`ValueGroundingIndex.build`) — for every ``TEXT``/
   ``VARCHAR``/``CHAR``-family column below a row-count ceiling, probe
   ``COUNT(DISTINCT col)``; treat it as "categorical" only if the distinct
   count is both small in absolute terms and small relative to the row
   count, then cache its actual distinct values. Never scans a table above
   the row-count ceiling — cost stays bounded regardless of database size.
2. **Matching** (:meth:`ValueGroundingIndex.match`) — question terms (1-3
   word n-grams) are compared against each profiled column's cached values.
   The zero-dependency baseline uses substring containment plus
   :func:`difflib.SequenceMatcher` ratio; when the optional ``rapidfuzz``
   package is installed, its token-set-ratio scorer is used instead for
   better recall on reordered/abbreviated phrases.

**Known limitation**: neither matching mode understands true semantic
aliases with no shared substring (e.g. "USA" vs "United States") — that
class of gap is instead partly covered by the LLM-assisted business
glossary's synonym generation (:mod:`universal_text2sql.knowledge.metadata`).
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from universal_text2sql.database.connector import DatabaseConnector
from universal_text2sql.database.schema import DatabaseSchema
from universal_text2sql.retrieval.semantic import tokenize

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.getenv("GROUNDING_CACHE_DIR", "./.grounding_cache"))
_MAX_TABLE_ROWS = int(os.getenv("GROUNDING_MAX_TABLE_ROWS", "200000"))
_MAX_DISTINCT = int(os.getenv("GROUNDING_MAX_DISTINCT", "500"))
# A column is "categorical" when distinct_count/row_count is below this.
# 0.75 cleanly separates genuinely repeating values (status/category/country
# columns typically sit well under 0.5) from identifier-like columns
# (email/name columns sit at ~1.0, all-unique) even on small tables, while
# still bounding cost on large ones via GROUNDING_MAX_DISTINCT above.
_CARDINALITY_RATIO = float(os.getenv("GROUNDING_CARDINALITY_RATIO", "0.75"))

_TEXT_TYPE_RE = re.compile(r"TEXT|VARCHAR|CHAR", re.IGNORECASE)

_DEFAULT_MATCH_THRESHOLD = 0.72


@dataclass
class ColumnProfile:
    """Cached distinct-value profile for one categorical text column."""

    table: str
    column: str
    distinct_values: list[str] = field(default_factory=list)


@dataclass
class ValueGroundingIndex:
    """Profiled categorical columns for one database, with question matching."""

    profiles: list[ColumnProfile] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        schema: DatabaseSchema,
        connector: DatabaseConnector,
        cache_dir: Path | None = None,
    ) -> ValueGroundingIndex:
        """Profile categorical text columns, using a disk cache when available."""
        cache_dir = cache_dir or _CACHE_DIR

        cached = _load_cache(cache_dir, schema)
        if cached is not None:
            return cls(profiles=[ColumnProfile(**p) for p in cached])

        profiles: list[ColumnProfile] = []
        for table_name, meta in schema.tables.items():
            if meta.row_count <= 0 or meta.row_count > _MAX_TABLE_ROWS:
                continue
            for col in meta.columns:
                if col.primary_key or not _TEXT_TYPE_RE.search(col.data_type):
                    continue
                profile = _profile_column(connector, table_name, col.name, meta.row_count)
                if profile is not None:
                    profiles.append(profile)

        logger.info("Value grounding: profiled %d categorical column(s)", len(profiles))
        _save_cache(cache_dir, schema, profiles)
        return cls(profiles=profiles)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        question: str,
        tables: list[str],
        top_k: int = 10,
        threshold: float = _DEFAULT_MATCH_THRESHOLD,
    ) -> list[tuple[str, str, str]]:
        """Return up to *top_k* ``(table, column, value)`` hits for *question*."""
        scorer = _get_scorer()
        phrases = _question_ngrams(question)
        if not phrases:
            return []

        scored: list[tuple[float, str, str, str]] = []
        for profile in self.profiles:
            if profile.table not in tables:
                continue
            for value in profile.distinct_values:
                best = max((scorer(phrase, value) for phrase in phrases), default=0.0)
                if best >= threshold:
                    scored.append((best, profile.table, profile.column, value))

        scored.sort(key=lambda t: t[0], reverse=True)
        seen: set[tuple[str, str, str]] = set()
        results: list[tuple[str, str, str]] = []
        for _score, table, column, value in scored:
            key = (table, column, value)
            if key in seen:
                continue
            seen.add(key)
            results.append(key)
            if len(results) >= top_k:
                break
        return results

    def hints_block(self, matches: list[tuple[str, str, str]]) -> str:
        """Render matches as a prompt-ready "Value Grounding Hints" block."""
        if not matches:
            return ""
        lines = ["## Value Grounding Hints (matched against real column data)"]
        for table, column, value in matches:
            lines.append(f"  {table}.{column} contains the value: {value!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


def _profile_column(
    connector: DatabaseConnector, table: str, column: str, row_count: int
) -> ColumnProfile | None:
    try:
        # table/column are always sourced from SchemaDiscovery (ultimately
        # from live inspector introspection), never user/LLM input -- safe
        # to interpolate, matching the existing pattern in schema.py.
        count_df = connector.execute_query(
            f"SELECT COUNT(DISTINCT {column}) AS n FROM {table}"  # noqa: S608
        )
        distinct_count = int(count_df.iloc[0]["n"])
    except Exception as exc:
        logger.debug("Grounding cardinality probe failed for %s.%s: %s", table, column, exc)
        return None

    if distinct_count == 0 or distinct_count > _MAX_DISTINCT:
        return None
    if distinct_count / row_count >= _CARDINALITY_RATIO:
        return None

    try:
        values_df = connector.execute_query(
            f"SELECT DISTINCT {column} FROM {table} LIMIT {int(_MAX_DISTINCT)}"  # noqa: S608
        )
    except Exception as exc:
        logger.debug("Grounding value fetch failed for %s.%s: %s", table, column, exc)
        return None

    values = sorted({str(v) for v in values_df.iloc[:, 0].tolist() if v is not None})
    if not values:
        return None
    return ColumnProfile(table=table, column=column, distinct_values=values)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _question_ngrams(question: str, max_n: int = 3) -> list[str]:
    """1-3 word n-grams built from stopword-filtered question tokens.

    Reuses :func:`~universal_text2sql.retrieval.semantic.tokenize` (rather
    than a raw word split) specifically to exclude common words like "how"/
    "many"/"from" -- without that filtering, short stopwords can accidentally
    substring-match into unrelated proper nouns (e.g. "many" is a substring
    of "Germany").
    """
    words = tokenize(question)
    grams: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            grams.append(" ".join(words[i : i + n]))
    return grams


def _default_score(phrase: str, value: str) -> float:
    p, v = phrase.lower(), value.lower()
    if p == v:
        return 1.0
    if p in v or v in p:
        return 0.85
    return difflib.SequenceMatcher(None, p, v).ratio()


def _rapidfuzz_score(phrase: str, value: str) -> float:
    from rapidfuzz import fuzz

    return fuzz.token_set_ratio(phrase, value) / 100.0


def _get_scorer():
    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        return _default_score
    return _rapidfuzz_score


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, schema: DatabaseSchema) -> Path:
    return cache_dir / f"{schema.signature_hash()}.json"


def _load_cache(cache_dir: Path, schema: DatabaseSchema) -> list[dict] | None:
    path = _cache_path(cache_dir, schema)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not read grounding cache %s: %s", path, exc)
        return None


def _save_cache(cache_dir: Path, schema: DatabaseSchema, profiles: list[ColumnProfile]) -> None:
    path = _cache_path(cache_dir, schema)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(p) for p in profiles], indent=2))
    except Exception as exc:
        logger.warning("Could not write grounding cache %s: %s", path, exc)
