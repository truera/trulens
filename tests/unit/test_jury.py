"""Unit tests for trulens.feedback.jury.Jury."""

import inspect
import statistics
import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from trulens.feedback.jury import Jury


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_provider(model_engine: str, score: float):
    """Create a mock provider whose relevance() returns *score*."""
    provider = MagicMock()
    provider.model_engine = model_engine
    provider.relevance.side_effect = lambda prompt, response, **kw: score
    return provider


def _make_failing_provider(model_engine: str):
    """Create a mock provider whose relevance() always raises."""
    provider = MagicMock()
    provider.model_engine = model_engine
    provider.relevance.side_effect = RuntimeError("LLM unavailable")
    return provider


def _make_tuple_provider(model_engine: str, score: float):
    """Create a mock provider whose relevance() returns (score, metadata)."""
    provider = MagicMock()
    provider.model_engine = model_engine
    provider.relevance.side_effect = lambda prompt, response, **kw: (
        score,
        {"reason": "ok"},
    )
    return provider


# ---------------------------------------------------------------------------
# Signature shim — gives mock providers a realistic relevance signature so
# inspect.signature works the same as on a real LLMProvider.
# ---------------------------------------------------------------------------


def _mock_relevance(prompt: str, response: str) -> float:
    ...


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestJuryAggregation(unittest.TestCase):
    def _jury(self, scores, aggregation="mean", **kw):
        providers = [_make_provider(f"m{i}", s) for i, s in enumerate(scores)]
        # Give providers a realistic signature for the method
        for p in providers:
            p.relevance.__func__ = _mock_relevance
            p.relevance.__self__ = p
        j = Jury(providers, method="relevance", aggregation=aggregation, **kw)
        # Inject realistic __signature__ so tests reflect real usage
        j.__signature__ = inspect.signature(_mock_relevance)
        return j

    def _call(self, jury):
        return jury(prompt="What is TruLens?", response="An eval library.")

    def test_mean(self):
        j = self._jury([0.6, 0.8, 1.0], "mean")
        result = self._call(j)
        self.assertAlmostEqual(result, statistics.mean([0.6, 0.8, 1.0]))

    def test_median(self):
        j = self._jury([0.2, 0.9, 0.5], "median")
        result = self._call(j)
        self.assertAlmostEqual(result, statistics.median([0.2, 0.9, 0.5]))

    def test_trimmed_mean(self):
        j = self._jury([0.1, 0.5, 0.9], "trimmed_mean")
        result = self._call(j)
        self.assertAlmostEqual(result, 0.5)

    def test_trimmed_mean_fewer_than_three_falls_back_to_mean(self):
        j = self._jury([0.4, 0.8], "trimmed_mean")
        result = self._call(j)
        self.assertAlmostEqual(result, statistics.mean([0.4, 0.8]))

    def test_majority_vote_positive(self):
        # 3 above threshold (0.5), 1 below → majority positive → 1.0
        j = self._jury([0.6, 0.7, 0.8, 0.3], "majority_vote", threshold=0.5)
        result = self._call(j)
        self.assertAlmostEqual(result, 1.0)

    def test_majority_vote_negative(self):
        # 1 above threshold, 2 below → not majority → 0.0
        j = self._jury([0.8, 0.2, 0.3], "majority_vote", threshold=0.5)
        result = self._call(j)
        self.assertAlmostEqual(result, 0.0)

    def test_weighted_mean(self):
        j = self._jury(
            [1.0, 0.0, 0.5],
            "weighted_mean",
            weights=[0.5, 0.3, 0.2],
        )
        expected = 1.0 * 0.5 + 0.0 * 0.3 + 0.5 * 0.2
        result = self._call(j)
        self.assertAlmostEqual(result, expected)

    def test_custom_aggregation_callable(self):
        j = self._jury([0.3, 0.7], aggregation=lambda scores: max(scores))
        result = self._call(j)
        self.assertAlmostEqual(result, 0.7)


class TestJuryConstruction(unittest.TestCase):
    def _provider(self, name="gpt-4o", score=0.5):
        p = _make_provider(name, score)
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        return p

    def test_empty_jurors_raises(self):
        with self.assertRaises(ValueError):
            Jury([], method="relevance")

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            Jury([self._provider()], method="relevance", aggregation="harmonic")

    def test_weighted_mean_without_weights_raises(self):
        with self.assertRaises(ValueError):
            Jury([self._provider()], method="relevance", aggregation="weighted_mean")

    def test_weighted_mean_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            Jury(
                [self._provider(), self._provider()],
                method="relevance",
                aggregation="weighted_mean",
                weights=[0.5],
            )

    def test_missing_method_raises(self):
        p = MagicMock(spec=[])  # no attributes
        with self.assertRaises(AttributeError):
            Jury([p], method="relevance")

    def test_signature_matches_underlying_method(self):
        p = self._provider()
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p], method="relevance")
        sig = inspect.signature(j)
        self.assertIn("prompt", sig.parameters)
        self.assertIn("response", sig.parameters)

    def test_dunder_name_set(self):
        p = self._provider()
        j = Jury([p], method="relevance")
        self.assertEqual(j.__name__, "jury_relevance")


class TestJuryErrorHandling(unittest.TestCase):
    def _sig(self, provider):
        provider.relevance.__signature__ = inspect.signature(_mock_relevance)
        return provider

    def test_one_juror_fails_aggregates_rest(self):
        good = self._sig(_make_provider("good", 0.8))
        bad = self._sig(_make_failing_provider("bad"))
        j = Jury([good, bad], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        result = j(prompt="x", response="y")
        self.assertAlmostEqual(result, 0.8)

    def test_all_jurors_fail_raises(self):
        p1 = self._sig(_make_failing_provider("a"))
        p2 = self._sig(_make_failing_provider("b"))
        j = Jury([p1, p2], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        with self.assertRaises(RuntimeError):
            j(prompt="x", response="y")


class TestJuryReturnDetails(unittest.TestCase):
    def test_return_details_false_returns_float(self):
        p = _make_provider("gpt-4o", 0.75)
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p], method="relevance", return_details=False)
        j.__signature__ = inspect.signature(_mock_relevance)
        result = j(prompt="x", response="y")
        self.assertIsInstance(result, float)

    def test_return_details_true_returns_tuple(self):
        p = _make_provider("gpt-4o", 0.75)
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p], method="relevance", return_details=True)
        j.__signature__ = inspect.signature(_mock_relevance)
        result = j(prompt="x", response="y")
        self.assertIsInstance(result, tuple)
        score, details = result
        self.assertIsInstance(score, float)
        self.assertIsInstance(details, dict)
        self.assertAlmostEqual(score, 0.75)

    def test_details_dict_contains_per_juror_scores(self):
        providers = [
            _make_provider("gpt-4o-mini", 0.6),
            _make_provider("claude-haiku", 0.9),
        ]
        for p in providers:
            p.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury(providers, method="relevance", return_details=True)
        j.__signature__ = inspect.signature(_mock_relevance)
        _, details = j(prompt="x", response="y")
        self.assertIn("gpt-4o-mini", details)
        self.assertIn("claude-haiku", details)
        self.assertAlmostEqual(details["gpt-4o-mini"], 0.6)
        self.assertAlmostEqual(details["claude-haiku"], 0.9)

    def test_tuple_result_from_provider_unwrapped(self):
        p = _make_tuple_provider("gpt-4o", 0.7)
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        result = j(prompt="x", response="y")
        self.assertAlmostEqual(result, 0.7)


class TestJuryMetricIntegration(unittest.TestCase):
    """Verify Jury integrates with Metric without changes to Metric."""

    def test_metric_accepts_jury_as_implementation(self):
        from trulens.core import Metric

        p1 = _make_provider("gpt-4o-mini", 0.8)
        p1.relevance.__signature__ = inspect.signature(_mock_relevance)
        p2 = _make_provider("claude-haiku", 0.6)
        p2.relevance.__signature__ = inspect.signature(_mock_relevance)

        j = Jury([p1, p2], method="relevance", aggregation="mean")
        j.__signature__ = inspect.signature(_mock_relevance)

        m = Metric(implementation=j, name="Test Jury").on_input().on_output()
        self.assertIsNotNone(m)
        self.assertEqual(m.supplied_name, "Test Jury")

    def test_metric_selector_validation_passes(self):
        from trulens.core import Metric

        p = _make_provider("gpt-4o", 0.9)
        p.relevance.__signature__ = inspect.signature(_mock_relevance)

        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)

        # on_input() binds 'prompt', on_output() binds 'response'.
        # Metric validates both are in Jury's __signature__ — this must not raise.
        m = Metric(implementation=j).on_input().on_output()
        self.assertIn("prompt", m.selectors)
        self.assertIn("response", m.selectors)


class TestJuryDuplicateNames(unittest.TestCase):
    def test_duplicate_model_engine_names_disambiguated(self):
        p1 = _make_provider("gpt-4o-mini", 0.7)
        p1.relevance.__signature__ = inspect.signature(_mock_relevance)
        p2 = _make_provider("gpt-4o-mini", 0.9)
        p2.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p1, p2], method="relevance", return_details=True)
        j.__signature__ = inspect.signature(_mock_relevance)
        _, details = j(prompt="x", response="y")
        self.assertEqual(len(details), 2)
        # Names must be distinct (indexed).
        self.assertIn("gpt-4o-mini[0]", details)
        self.assertIn("gpt-4o-mini[1]", details)


if __name__ == "__main__":
    unittest.main()
