"""Unit tests for trulens.feedback.self_consistency.SelfConsistency."""

from __future__ import annotations

import inspect
import statistics
import unittest
from unittest.mock import MagicMock

from trulens.feedback.llm_provider import UNPARSABLE_SCORE
from trulens.feedback.self_consistency import SelfConsistency

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_relevance(prompt: str, response: str) -> float: ...


def _mock_relevance_with_kwargs(
    prompt: str, response: str, **kwargs
) -> float: ...


def _make_provider(score: float | list[float]):
    """Mock provider whose relevance() returns *score* (or cycles the list)."""
    provider = MagicMock()
    provider.model_engine = "mock-model"
    if isinstance(score, list):
        provider.relevance.side_effect = score
    else:
        provider.relevance.side_effect = lambda *a, **kw: score
    provider.relevance.__signature__ = inspect.signature(_mock_relevance)
    return provider


def _make_failing_provider():
    """Mock provider whose relevance() always raises."""
    provider = MagicMock()
    provider.model_engine = "mock-model"
    provider.relevance.side_effect = RuntimeError("LLM unavailable")
    provider.relevance.__signature__ = inspect.signature(_mock_relevance)
    return provider


def _make_cot_provider(score: float, reason: str):
    """Mock provider whose relevance() returns (score, {"reason": reason})."""
    provider = MagicMock()
    provider.model_engine = "mock-model"
    provider.relevance.side_effect = lambda *a, **kw: (
        score,
        {"reason": reason},
    )
    provider.relevance.__signature__ = inspect.signature(_mock_relevance)
    return provider


def _sc(scores: list[float], aggregation="mean", n: int | None = None, **kw):
    """Build a SelfConsistency whose trials return *scores* in order."""
    provider = _make_provider(scores)
    sc = SelfConsistency(
        provider,
        method="relevance",
        n=n if n is not None else len(scores),
        aggregation=aggregation,
        max_workers=1,  # serial for determinism
        **kw,
    )
    sc.__signature__ = inspect.signature(_mock_relevance)
    return sc


def _call(sc: SelfConsistency) -> float:
    score, _ = sc(prompt="What is TruLens?", response="An eval library.")
    return score


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestSelfConsistencyConstruction(unittest.TestCase):
    def _provider(self):
        return _make_provider(0.8)

    def test_n_less_than_2_raises(self):
        with self.assertRaises(ValueError):
            SelfConsistency(self._provider(), method="relevance", n=1)

    def test_missing_method_on_provider_raises(self):
        bad = MagicMock(spec=[])  # no attributes
        with self.assertRaises(AttributeError):
            SelfConsistency(bad, method="relevance", n=3)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            SelfConsistency(
                self._provider(),
                method="relevance",
                n=3,
                aggregation="harmonic",
            )

    def test_signature_copied_from_provider_method(self):
        sc = SelfConsistency(self._provider(), method="relevance", n=3)
        sig = inspect.signature(sc)
        self.assertIn("prompt", sig.parameters)
        self.assertIn("response", sig.parameters)

    def test_dunder_name_set(self):
        sc = SelfConsistency(self._provider(), method="relevance", n=3)
        self.assertEqual(sc.__name__, "self_consistency_relevance")


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------


class TestSelfConsistencyAggregation(unittest.TestCase):
    def test_mean(self):
        sc = _sc([0.2, 0.6, 1.0], "mean")
        self.assertAlmostEqual(_call(sc), statistics.mean([0.2, 0.6, 1.0]))

    def test_median(self):
        sc = _sc([0.1, 0.5, 0.9], "median")
        self.assertAlmostEqual(_call(sc), 0.5)

    def test_trimmed_mean(self):
        sc = _sc([0.1, 0.5, 0.9], "trimmed_mean")
        self.assertAlmostEqual(_call(sc), 0.5)

    def test_trimmed_mean_fewer_than_three_falls_back_to_mean(self):
        sc = _sc([0.4, 0.8], "trimmed_mean")
        self.assertAlmostEqual(_call(sc), statistics.mean([0.4, 0.8]))

    def test_majority_vote_positive(self):
        sc = _sc([0.6, 0.7, 0.8, 0.3], "majority_vote", threshold=0.5)
        self.assertAlmostEqual(_call(sc), 1.0)

    def test_majority_vote_negative(self):
        sc = _sc([0.8, 0.2, 0.3], "majority_vote", threshold=0.5)
        self.assertAlmostEqual(_call(sc), 0.0)

    def test_majority_vote_tie_falls_back_to_median(self):
        sc = _sc([0.8, 0.2], "majority_vote", threshold=0.5)
        self.assertAlmostEqual(_call(sc), statistics.median([0.8, 0.2]))

    def test_custom_aggregation_callable(self):
        sc = _sc([0.3, 0.7], aggregation=lambda scores: max(scores))
        self.assertAlmostEqual(_call(sc), 0.7)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSelfConsistencyErrorHandling(unittest.TestCase):
    def test_all_trials_fail_raises(self):
        provider = _make_failing_provider()
        sc = SelfConsistency(provider, method="relevance", n=3, max_workers=1)
        sc.__signature__ = inspect.signature(_mock_relevance)
        with self.assertRaises(RuntimeError):
            sc(prompt="x", response="y")

    def test_one_trial_fails_aggregates_rest(self):
        # Alternate: first call raises, second returns 0.8.
        provider = MagicMock()
        provider.model_engine = "mock"
        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("fluke")
            return 0.8

        provider.relevance.side_effect = side_effect
        provider.relevance.__signature__ = inspect.signature(_mock_relevance)

        sc = SelfConsistency(provider, method="relevance", n=2, max_workers=1)
        sc.__signature__ = inspect.signature(_mock_relevance)
        score, _ = sc(prompt="x", response="y")
        self.assertAlmostEqual(score, 0.8)


# ---------------------------------------------------------------------------
# UNPARSABLE_SCORE sentinel handling
# ---------------------------------------------------------------------------


class TestSelfConsistencyUnparsableSentinel(unittest.TestCase):
    def test_sentinel_dropped_from_mean(self):
        # Two valid scores and one sentinel → mean of the valid ones.
        sc = _sc([1.0, 1.0, UNPARSABLE_SCORE], "mean")
        self.assertAlmostEqual(_call(sc), 1.0)

    def test_all_sentinel_scores_raises(self):
        sc = _sc([UNPARSABLE_SCORE, UNPARSABLE_SCORE], "mean")
        with self.assertRaises(RuntimeError):
            sc(prompt="x", response="y")

    def test_sentinel_in_cot_tuple_is_dropped(self):
        provider = _make_cot_provider(UNPARSABLE_SCORE, "no score found")
        provider.relevance.side_effect = [
            (1.0, {"reason": "ok"}),
            (UNPARSABLE_SCORE, {"reason": "no score found"}),
        ]
        sc = SelfConsistency(provider, method="relevance", n=2, max_workers=1)
        sc.__signature__ = inspect.signature(_mock_relevance)
        score, _ = sc(prompt="x", response="y")
        self.assertAlmostEqual(score, 1.0)

    def test_sentinel_logged_as_warning(self):
        sc = _sc([0.8, UNPARSABLE_SCORE], "mean")
        with self.assertLogs(
            "trulens.feedback.self_consistency", level="WARNING"
        ) as logs:
            _call(sc)
        self.assertTrue(
            any("parsable" in msg for msg in logs.output),
            msg=logs.output,
        )

    def test_zero_score_is_not_a_failure(self):
        # 0.0 is a real verdict; only the sentinel value means "no score".
        sc = _sc([0.0, 1.0], "mean")
        self.assertAlmostEqual(_call(sc), 0.5)


# ---------------------------------------------------------------------------
# Return format
# ---------------------------------------------------------------------------


class TestSelfConsistencyReturnFormat(unittest.TestCase):
    def test_always_returns_tuple(self):
        sc = _sc([0.6, 0.8])
        result = sc(prompt="x", response="y")
        self.assertIsInstance(result, tuple)
        score, meta = result
        self.assertIsInstance(score, float)
        self.assertIsInstance(meta, dict)
        self.assertIn("reason", meta)

    def test_reason_contains_method_name(self):
        sc = _sc([0.6, 0.8])
        _, meta = sc(prompt="x", response="y")
        self.assertIn("relevance", meta["reason"])

    def test_reason_contains_trial_scores(self):
        sc = _sc([0.6, 0.8])
        _, meta = sc(prompt="x", response="y")
        self.assertIn("trials=", meta["reason"])

    def test_reason_contains_reliability_metrics(self):
        sc = _sc([0.6, 0.8])
        _, meta = sc(prompt="x", response="y")
        self.assertIn("std_dev=", meta["reason"])
        self.assertIn("flip_rate=", meta["reason"])
        self.assertIn("entropy=", meta["reason"])

    def test_cot_reason_passthrough(self):
        provider = _make_cot_provider(0.8, "Supporting Evidence: clear")
        sc = SelfConsistency(provider, method="relevance", n=2, max_workers=1)
        sc.__signature__ = inspect.signature(_mock_relevance)
        _, meta = sc(prompt="x", response="y")
        self.assertIn("Supporting Evidence: clear", meta["reason"])

    def test_accepts_positional_arguments(self):
        sc = _sc([0.75, 0.85])
        score, _ = sc("What is TruLens?", "An eval library.")
        self.assertIsInstance(score, float)


# ---------------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------------


class TestSelfConsistencyReliabilityMetrics(unittest.TestCase):
    def test_flip_rate_zero_when_all_agree(self):
        sc = _sc([1.0, 1.0, 1.0])
        self.assertAlmostEqual(sc._flip_rate([1.0, 1.0, 1.0]), 0.0)

    def test_flip_rate_maximum_on_50_50_split(self):
        sc = _sc([0.0, 1.0])
        self.assertAlmostEqual(sc._flip_rate([0.0, 1.0]), 0.5)

    def test_entropy_zero_when_unanimous(self):
        sc = _sc([1.0, 1.0])
        self.assertAlmostEqual(
            sc._binary_entropy([1.0, 1.0], threshold=0.5), 0.0
        )

    def test_entropy_one_at_50_50_split(self):
        sc = _sc([0.0, 1.0])
        self.assertAlmostEqual(
            sc._binary_entropy([0.0, 1.0], threshold=0.5), 1.0
        )

    def test_std_dev_zero_for_identical_scores(self):
        sc = _sc([0.7, 0.7, 0.7])
        score, meta = sc(prompt="x", response="y")
        self.assertIn("std_dev=0.000", meta["reason"])


# ---------------------------------------------------------------------------
# Metric integration
# ---------------------------------------------------------------------------


class TestSelfConsistencyMetricIntegration(unittest.TestCase):
    def test_metric_accepts_self_consistency_as_implementation(self):
        from trulens.core import Metric

        provider = _make_provider(0.8)
        sc = SelfConsistency(provider, method="relevance", n=3, max_workers=1)
        sc.__signature__ = inspect.signature(_mock_relevance)

        m = (
            Metric(implementation=sc, name="SC Relevance")
            .on_input()
            .on_output()
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.supplied_name, "SC Relevance")

    def test_metric_selector_validation_passes(self):
        from trulens.core import Metric

        provider = _make_provider(0.8)
        sc = SelfConsistency(provider, method="relevance", n=3, max_workers=1)
        sc.__signature__ = inspect.signature(_mock_relevance)

        m = Metric(implementation=sc).on_input().on_output()
        self.assertIn("prompt", m.selectors)
        self.assertIn("response", m.selectors)


if __name__ == "__main__":
    unittest.main()
