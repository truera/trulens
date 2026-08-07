"""Unit tests for cross-model alignment. No model or API key needed."""

import pytest

cross_model_alignment = pytest.importorskip(
    "trulens.benchmark.cross_model_alignment"
)
CrossModelAlignment = cross_model_alignment.CrossModelAlignment
CrossModelAlignmentReport = cross_model_alignment.CrossModelAlignmentReport


def test_identical_judges_are_interchangeable():
    scores = [0.1, 0.4, 0.6, 0.9, 0.3]
    report = CrossModelAlignmentReport(["a", "b"], [scores, scores])
    assert report.spearman[0, 1] == pytest.approx(1.0)
    assert any("interchangeable" in r for r in report.recommendations())


def test_diagonal_spearman_is_one():
    report = CrossModelAlignmentReport(
        ["a", "b"], [[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]]
    )
    assert report.spearman[0, 0] == pytest.approx(1.0)
    assert report.spearman[1, 1] == pytest.approx(1.0)


def test_bias_detects_consistent_offset():
    high = [0.3, 0.5, 0.7]
    low = [0.1, 0.3, 0.5]
    report = CrossModelAlignmentReport(["high", "low"], [high, low])
    assert report.bias[0, 1] == pytest.approx(0.2, abs=1e-6)


def test_mean_abs_diff():
    report = CrossModelAlignmentReport(["a", "b"], [[0.0, 1.0], [1.0, 0.0]])
    assert report.mean_abs_diff[0, 1] == pytest.approx(1.0)


def test_outlier_detected():
    a = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    b = [0.12, 0.22, 0.32, 0.42, 0.52, 0.62]
    c = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    d = [0.50, 0.10, 0.60, 0.20, 0.40, 0.30]  # uncorrelated with a/b/c
    report = CrossModelAlignmentReport(["a", "b", "c", "d"], [a, b, c, d])
    recs = report.recommendations()
    assert any("d is an outlier" in r for r in recs)
    assert not any("a is an outlier" in r for r in recs)


def test_ground_truth_metrics():
    perfect = [0.2, 0.5, 0.8]
    flat = [0.5, 0.5, 0.5]
    report = CrossModelAlignmentReport(
        ["perfect", "flat"], [perfect, flat], expected=[0.2, 0.5, 0.8]
    )
    gt = report.ground_truth_metrics()
    assert gt["perfect"]["mae"] == pytest.approx(0.0)
    assert gt["perfect"]["spearman"] == pytest.approx(1.0)
    assert gt["flat"]["mae"] > 0.0


class _FakeProvider:
    def __init__(self, scores):
        self._scores = iter(scores)

    def relevance(self, query, response):
        return next(self._scores)


def test_run_with_fake_providers():
    golden = [
        {"query": "q1", "expected_response": "r1", "expected_score": 0.2},
        {"query": "q2", "expected_response": "r2", "expected_score": 0.8},
    ]
    judges = [
        {"provider": _FakeProvider([0.25, 0.75]), "name": "a"},
        {"provider": _FakeProvider([0.3, 0.7]), "name": "b"},
    ]
    report = CrossModelAlignment(judges, "relevance", golden).run()
    assert report.n == 2
    assert report.scores[0].tolist() == [0.25, 0.75]
    assert report.expected is not None


def test_requires_at_least_two_judges():
    with pytest.raises(ValueError):
        CrossModelAlignment(
            [{"provider": object(), "name": "a"}], "relevance", []
        )


def test_run_raises_when_all_rows_fail():
    class _Boom:
        def relevance(self, query, response):
            raise RuntimeError("judge down")

    golden = [{"query": "q", "expected_response": "r"}]
    judges = [
        {"provider": _Boom(), "name": "a"},
        {"provider": _Boom(), "name": "b"},
    ]
    with pytest.raises(ValueError):
        CrossModelAlignment(judges, "relevance", golden).run()
