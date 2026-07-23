"""Unit tests for trulens.feedback.optimize."""

import pytest
from trulens.feedback.optimize import FewShotOptimizer
from trulens.feedback.optimize import OptimizeResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_feedback_fn(bias: float = 0.0):
    """Return a fake feedback_fn that returns ground_truth + bias + noise=0."""

    def feedback_fn(examples: str = "", **kwargs) -> float:
        # returns a deterministic score based on kwargs values length
        score = sum(len(v) for v in kwargs.values()) / 100.0
        return min(1.0, max(0.0, score + bias))

    return feedback_fn


CANDIDATES = [
    ({"input": "What is 2+2?", "output": "4"}, 1.0),
    ({"input": "Capital of France?", "output": "Paris"}, 0.9),
    ({"input": "Who wrote Hamlet?", "output": "Einstein"}, 0.1),
    ({"input": "Sky color?", "output": "Blue"}, 0.8),
]

EVAL_DATASET = [
    ({"input": "What is 3+3?", "output": "6"}, 1.0),
    ({"input": "Capital of Germany?", "output": "Berlin"}, 0.9),
    ({"input": "Water formula?", "output": "H2O"}, 0.8),
]


# ---------------------------------------------------------------------------
# _pearson_correlation
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    def setup_method(self):
        self.opt = FewShotOptimizer(
            feedback_fn=make_feedback_fn(),
            candidates=CANDIDATES,
            eval_dataset=EVAL_DATASET,
        )

    def test_perfect_positive_correlation(self):
        r = self.opt._pearson_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert r == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        r = self.opt._pearson_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert r == pytest.approx(-1.0)

    def test_zero_correlation(self):
        r = self.opt._pearson_correlation([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        assert r is None  # zero variance in ground truth

    def test_too_few_samples_returns_none(self):
        assert self.opt._pearson_correlation([0.5], [0.5]) is None

    def test_empty_lists_returns_none(self):
        assert self.opt._pearson_correlation([], []) is None

    def test_mismatched_lengths_returns_none(self):
        assert self.opt._pearson_correlation([1.0, 2.0], [1.0]) is None


# ---------------------------------------------------------------------------
# format_examples
# ---------------------------------------------------------------------------


class TestFormatExamples:
    def setup_method(self):
        self.opt = FewShotOptimizer(
            feedback_fn=make_feedback_fn(),
            candidates=CANDIDATES,
            eval_dataset=EVAL_DATASET,
        )

    def test_single_example_contains_key_and_score(self):
        result = self.opt.format_examples([CANDIDATES[0]])
        assert "input" in result
        assert "expected_score" in result
        assert "1.00" in result

    def test_multiple_examples_separated(self):
        result = self.opt.format_examples(CANDIDATES[:2])
        assert "Example 1" in result
        assert "Example 2" in result

    def test_custom_separator(self):
        opt = FewShotOptimizer(
            feedback_fn=make_feedback_fn(),
            candidates=CANDIDATES,
            eval_dataset=EVAL_DATASET,
            format_sep="---",
        )
        result = opt.format_examples(CANDIDATES[:2])
        assert "---" in result

    def test_empty_examples_returns_empty_string(self):
        result = self.opt.format_examples([])
        assert result == ""


# ---------------------------------------------------------------------------
# FewShotOptimizer.__init__ validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError, match="candidates"):
            FewShotOptimizer(
                feedback_fn=make_feedback_fn(),
                candidates=[],
                eval_dataset=EVAL_DATASET,
            )

    def test_empty_eval_dataset_raises(self):
        with pytest.raises(ValueError, match="eval_dataset"):
            FewShotOptimizer(
                feedback_fn=make_feedback_fn(),
                candidates=CANDIDATES,
                eval_dataset=[],
            )

    def test_n_examples_zero_raises(self):
        with pytest.raises(ValueError, match="n_examples"):
            FewShotOptimizer(
                feedback_fn=make_feedback_fn(),
                candidates=CANDIDATES,
                eval_dataset=EVAL_DATASET,
                n_examples=0,
            )


# ---------------------------------------------------------------------------
# optimize() end-to-end
# ---------------------------------------------------------------------------


class TestOptimize:
    def setup_method(self):
        self.opt = FewShotOptimizer(
            feedback_fn=make_feedback_fn(),
            candidates=CANDIDATES,
            eval_dataset=EVAL_DATASET,
            n_examples=2,
        )

    def test_returns_optimize_result(self):
        result = self.opt.optimize()
        assert isinstance(result, OptimizeResult)

    def test_respects_n_examples(self):
        result = self.opt.optimize()
        assert len(result.best_examples) <= 2

    def test_best_examples_are_subset_of_candidates(self):
        result = self.opt.optimize()
        for ex in result.best_examples:
            assert ex in CANDIDATES

    def test_candidate_scores_populated(self):
        result = self.opt.optimize()
        assert len(result.candidate_scores) > 0

    def test_no_duplicate_examples_selected(self):
        result = self.opt.optimize()
        seen = []
        for ex in result.best_examples:
            assert ex not in seen
            seen.append(ex)

    def test_feedback_fn_exception_handled_gracefully(self):
        def flaky_fn(examples: str = "", **kwargs) -> float:
            if "Capital" in kwargs.get("input", ""):
                raise RuntimeError("flaky!")
            return 0.8

        opt = FewShotOptimizer(
            feedback_fn=flaky_fn,
            candidates=CANDIDATES,
            eval_dataset=EVAL_DATASET,
            n_examples=2,
        )
        # should not raise even with a flaky feedback_fn
        result = opt.optimize()
        assert isinstance(result, OptimizeResult)
