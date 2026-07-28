"""Tests for the shared result-comparison utility (agent/scoring.py)."""

from __future__ import annotations

import pandas as pd

from universal_text2sql.agent.scoring import result_signature, results_match


class TestResultSignature:
    def test_none_has_stable_signature(self):
        assert result_signature(None) == result_signature(None)

    def test_identical_frames_match(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert result_signature(df1) == result_signature(df2)

    def test_row_order_independent(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"a": [2, 1], "b": ["y", "x"]})
        assert result_signature(df1) == result_signature(df2)

    def test_different_values_differ(self):
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        assert result_signature(df1) != result_signature(df2)

    def test_different_row_counts_differ(self):
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [1, 1]})
        assert result_signature(df1) != result_signature(df2)


class TestResultsMatch:
    def test_matching_frames(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [2, 1]})
        assert results_match(df1, df2) is True

    def test_none_vs_frame(self):
        assert results_match(None, pd.DataFrame({"a": [1]})) is False

    def test_both_none(self):
        assert results_match(None, None) is True
