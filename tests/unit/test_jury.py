"""Unit tests for trulens.feedback.jury.Jury."""

from __future__ import annotations

import inspect
import statistics
import unittest
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


def _make_cot_provider(model_engine: str, score: float, reason: str):
    """Create a mock provider whose relevance() returns (score, {"reason": reason})."""
    provider = MagicMock()
    provider.model_engine = model_engine
    provider.relevance.side_effect = lambda prompt, response, **kw: (
        score,
        {"reason": reason},
    )
    return provider


# ---------------------------------------------------------------------------
# Signature shim — gives mock providers a realistic relevance signature so
# inspect.signature works the same as on a real LLMProvider.
# ---------------------------------------------------------------------------


def _mock_relevance(prompt: str, response: str) -> float: ...


def _mock_relevance_with_kwargs(
    prompt: str, response: str, **kwargs
) -> float: ...


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestJuryAggregation(unittest.TestCase):
    def _jury(self, scores, aggregation="mean", **kw):
        providers = [_make_provider(f"m{i}", s) for i, s in enumerate(scores)]
        for p in providers:
            p.relevance.__func__ = _mock_relevance
            p.relevance.__self__ = p
        j = Jury(providers, method="relevance", aggregation=aggregation, **kw)
        j.__signature__ = inspect.signature(_mock_relevance)
        return j

    def _call(self, jury) -> float:
        score, _ = jury(prompt="What is TruLens?", response="An eval library.")
        return score

    def test_mean(self):
        j = self._jury([0.6, 0.8, 1.0], "mean")
        self.assertAlmostEqual(self._call(j), statistics.mean([0.6, 0.8, 1.0]))

    def test_median(self):
        j = self._jury([0.2, 0.9, 0.5], "median")
        self.assertAlmostEqual(
            self._call(j), statistics.median([0.2, 0.9, 0.5])
        )

    def test_trimmed_mean(self):
        j = self._jury([0.1, 0.5, 0.9], "trimmed_mean")
        self.assertAlmostEqual(self._call(j), 0.5)

    def test_trimmed_mean_fewer_than_three_falls_back_to_mean(self):
        j = self._jury([0.4, 0.8], "trimmed_mean")
        self.assertAlmostEqual(self._call(j), statistics.mean([0.4, 0.8]))

    def test_majority_vote_positive(self):
        # 3 above threshold, 1 below → majority positive → 1.0
        j = self._jury([0.6, 0.7, 0.8, 0.3], "majority_vote", threshold=0.5)
        self.assertAlmostEqual(self._call(j), 1.0)

    def test_majority_vote_negative(self):
        # 1 above threshold, 2 below → not majority → 0.0
        j = self._jury([0.8, 0.2, 0.3], "majority_vote", threshold=0.5)
        self.assertAlmostEqual(self._call(j), 0.0)

    def test_majority_vote_tie_falls_back_to_median(self):
        # Exact tie: 1 above, 1 below → falls back to median
        j = self._jury([0.8, 0.2], "majority_vote", threshold=0.5)
        self.assertAlmostEqual(self._call(j), statistics.median([0.8, 0.2]))

    def test_weighted_mean(self):
        j = self._jury(
            [1.0, 0.0, 0.5],
            "weighted_mean",
            weights=[0.5, 0.3, 0.2],
        )
        expected = 1.0 * 0.5 + 0.0 * 0.3 + 0.5 * 0.2
        self.assertAlmostEqual(self._call(j), expected)

    def test_custom_aggregation_callable(self):
        j = self._jury([0.3, 0.7], aggregation=lambda scores: max(scores))
        self.assertAlmostEqual(self._call(j), 0.7)


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
            Jury(
                [self._provider()],
                method="relevance",
                aggregation="weighted_mean",
            )

    def test_weighted_mean_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            Jury(
                [self._provider(), self._provider()],
                method="relevance",
                aggregation="weighted_mean",
                weights=[0.5],
            )

    def test_missing_method_on_first_juror_raises(self):
        p = MagicMock(spec=[])  # no attributes
        with self.assertRaises(AttributeError):
            Jury([p], method="relevance")

    def test_missing_method_on_later_juror_raises(self):
        good = self._provider()
        bad = MagicMock(spec=[])  # no attributes
        with self.assertRaises(AttributeError):
            Jury([good, bad], method="relevance")

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
        score, _ = j(prompt="x", response="y")
        self.assertAlmostEqual(score, 0.8)

    def test_all_jurors_fail_raises(self):
        p1 = self._sig(_make_failing_provider("a"))
        p2 = self._sig(_make_failing_provider("b"))
        j = Jury([p1, p2], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        with self.assertRaises(RuntimeError):
            j(prompt="x", response="y")

    def test_weighted_mean_partial_failure_redistributes_weight(self):
        # juror[0]=1.0 weight=0.3, juror[1] fails weight=0.7
        # surviving total=0.3, result = 1.0 * 0.3 / 0.3 = 1.0
        good = self._sig(_make_provider("good", 1.0))
        bad = self._sig(_make_failing_provider("bad"))
        j = Jury(
            [good, bad],
            method="relevance",
            aggregation="weighted_mean",
            weights=[0.3, 0.7],
        )
        j.__signature__ = inspect.signature(_mock_relevance)
        score, _ = j(prompt="x", response="y")
        self.assertAlmostEqual(score, 1.0)

    def test_zero_total_weight_after_failures_raises(self):
        good = self._sig(_make_provider("good", 0.8))
        j = Jury(
            [good],
            method="relevance",
            aggregation="weighted_mean",
            weights=[0.0],
        )
        j.__signature__ = inspect.signature(_mock_relevance)
        with self.assertRaises(ValueError):
            j(prompt="x", response="y")


class TestJuryReturnFormat(unittest.TestCase):
    """Jury always returns (float, {"reason": ...}) — the _with_cot_reasons convention."""

    def _provider_with_sig(self, name, score):
        p = _make_provider(name, score)
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        return p

    def test_always_returns_tuple(self):
        p = self._provider_with_sig("gpt-4o", 0.75)
        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        result = j(prompt="x", response="y")
        self.assertIsInstance(result, tuple)
        score, meta = result
        self.assertIsInstance(score, float)
        self.assertIsInstance(meta, dict)
        self.assertIn("reason", meta)

    def test_reason_contains_aggregation_header(self):
        p = self._provider_with_sig("gpt-4o", 0.8)
        j = Jury([p], method="relevance", aggregation="mean")
        j.__signature__ = inspect.signature(_mock_relevance)
        _, meta = j(prompt="x", response="y")
        self.assertIn("Aggregation: mean", meta["reason"])

    def test_reason_contains_per_juror_scores(self):
        p1 = self._provider_with_sig("gpt-4o-mini", 0.6)
        p2 = self._provider_with_sig("claude-haiku", 0.9)
        j = Jury([p1, p2], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        _, meta = j(prompt="x", response="y")
        self.assertIn("gpt-4o-mini", meta["reason"])
        self.assertIn("claude-haiku", meta["reason"])
        self.assertIn("0.600", meta["reason"])
        self.assertIn("0.900", meta["reason"])

    def test_juror_cot_reason_passthrough(self):
        # When jurors return (score, {"reason": "..."}) the explanation
        # should appear indented in the aggregated reason string.
        p = _make_cot_provider(
            "gpt-4o", 0.8, "Supporting Evidence: directly answers"
        )
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        _, meta = j(prompt="x", response="y")
        self.assertIn("Supporting Evidence: directly answers", meta["reason"])

    def test_plain_float_juror_has_no_reason_lines(self):
        # Float-returning jurors produce no CoT lines — only the score line.
        p = self._provider_with_sig("gpt-4o", 0.7)
        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        _, meta = j(prompt="x", response="y")
        lines = meta["reason"].splitlines()
        # Header + one juror score line, no indented reason lines
        self.assertEqual(len(lines), 2)

    def test_tuple_result_score_extracted_correctly(self):
        p = _make_cot_provider("gpt-4o", 0.7, "ok")
        p.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        score, _ = j(prompt="x", response="y")
        self.assertAlmostEqual(score, 0.7)

    def test_accepts_positional_arguments_matching_provider_signature(self):
        p = self._provider_with_sig("gpt-4o", 0.75)
        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)

        score, _ = j("What is TruLens?", "An eval library.")

        self.assertAlmostEqual(score, 0.75)
        p.relevance.assert_called_once_with(
            "What is TruLens?", "An eval library."
        )

    def test_preserves_extra_keyword_arguments(self):
        p = _make_provider("gpt-4o", 0.8)
        p.relevance.__signature__ = inspect.signature(
            _mock_relevance_with_kwargs
        )
        j = Jury([p], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance_with_kwargs)

        score, _ = j("x", "y", custom_instructions="be strict")

        self.assertAlmostEqual(score, 0.8)
        p.relevance.assert_called_once_with(
            "x",
            "y",
            custom_instructions="be strict",
        )

    def test_each_juror_receives_independent_kwargs(self):
        class MutatingKwargsJury(Jury):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.seen_custom_instructions = []

            def _call_juror(self, juror, args, kwargs):
                self.seen_custom_instructions.append(
                    kwargs.get("custom_instructions")
                )
                kwargs.pop("custom_instructions", None)
                return super()._call_juror(juror, args, kwargs)

        p1 = _make_provider("gpt-4o-mini", 0.6)
        p1.relevance.__signature__ = inspect.signature(
            _mock_relevance_with_kwargs
        )
        p2 = _make_provider("claude-haiku", 0.8)
        p2.relevance.__signature__ = inspect.signature(
            _mock_relevance_with_kwargs
        )
        j = MutatingKwargsJury(
            [p1, p2],
            method="relevance",
            max_workers=1,
        )
        j.__signature__ = inspect.signature(_mock_relevance_with_kwargs)

        score, _ = j("x", "y", custom_instructions="be strict")

        self.assertAlmostEqual(score, 0.7)
        self.assertEqual(
            j.seen_custom_instructions,
            ["be strict", "be strict"],
        )


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

        m = Metric(implementation=j).on_input().on_output()
        self.assertIn("prompt", m.selectors)
        self.assertIn("response", m.selectors)


class TestJuryDuplicateNames(unittest.TestCase):
    def test_duplicate_model_engine_names_disambiguated(self):
        p1 = _make_provider("gpt-4o-mini", 0.7)
        p1.relevance.__signature__ = inspect.signature(_mock_relevance)
        p2 = _make_provider("gpt-4o-mini", 0.9)
        p2.relevance.__signature__ = inspect.signature(_mock_relevance)
        j = Jury([p1, p2], method="relevance")
        j.__signature__ = inspect.signature(_mock_relevance)
        _, meta = j(prompt="x", response="y")
        self.assertIn("gpt-4o-mini[0]", meta["reason"])
        self.assertIn("gpt-4o-mini[1]", meta["reason"])


if __name__ == "__main__":
    unittest.main()
