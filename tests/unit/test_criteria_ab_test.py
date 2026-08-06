"""Unit tests for criteria A/B testing. No model or API key needed."""

import pytest

criteria_ab_test = pytest.importorskip("trulens.benchmark.criteria_ab_test")
CriteriaABTest = criteria_ab_test.CriteriaABTest
CriteriaABTestReport = criteria_ab_test.CriteriaABTestReport


def test_metrics_perfect_vs_flat():
    report = CriteriaABTestReport(
        "perfect",
        [0.2, 0.5, 0.8],
        "flat",
        [0.5, 0.5, 0.5],
        expected=[0.2, 0.5, 0.8],
    )
    metrics = report.metrics()
    assert metrics["perfect"]["mae"] == pytest.approx(0.0)
    assert metrics["perfect"]["spearman"] == pytest.approx(1.0)
    assert metrics["flat"]["mae"] > 0.0


def test_winner_is_lower_mae():
    report = CriteriaABTestReport(
        "good",
        [0.2, 0.5, 0.8],
        "bad",
        [0.9, 0.1, 0.4],
        expected=[0.2, 0.5, 0.8],
    )
    assert report.winner() == "good"


def test_significance_identical_is_not_significant():
    scores = [0.2, 0.5, 0.8, 0.4]
    report = CriteriaABTestReport("a", scores, "b", scores)
    sig = report.significance()
    assert sig["mean_difference"] == pytest.approx(0.0)
    assert sig["p_value"] == pytest.approx(1.0)


def test_significance_consistent_shift_is_significant():
    a = [0.6, 0.7, 0.8, 0.9, 0.5, 0.65, 0.75, 0.85]
    b = [0.3, 0.4, 0.5, 0.6, 0.2, 0.35, 0.45, 0.55]  # B is consistently -0.3
    report = CriteriaABTestReport("a", a, "b", b)
    sig = report.significance()
    assert sig["mean_difference"] == pytest.approx(0.3, abs=1e-6)
    assert sig["p_value"] < 0.05


def test_top_disagreements_orders_by_abs_difference():
    report = CriteriaABTestReport(
        "a",
        [0.1, 0.9, 0.5],
        "b",
        [0.1, 0.2, 0.45],
        queries=["q0", "q1", "q2"],
    )
    top = report.top_disagreements(k=1)
    assert top[0]["index"] == 1
    assert top[0]["query"] == "q1"


def test_run_with_fake_variants_and_kwargs():
    golden = [
        {"query": "q1", "expected_response": "r1", "expected_score": 0.2},
        {"query": "q2", "expected_response": "r2", "expected_score": 0.8},
    ]
    a_scores = iter([0.25, 0.75])
    b_scores = iter([0.5, 0.5])

    def fn_a(query, response):
        return next(a_scores)

    def fn_b(query, response, criteria=None):
        assert criteria == "strict"  # kwargs are forwarded
        return next(b_scores)

    report = CriteriaABTest(
        golden,
        {"fn": fn_a, "name": "default"},
        {"fn": fn_b, "kwargs": {"criteria": "strict"}, "name": "strict"},
    ).run()
    assert report.scores_a.tolist() == [0.25, 0.75]
    assert report.scores_b.tolist() == [0.5, 0.5]
    assert report.expected is not None


def test_variant_requires_fn_and_name():
    with pytest.raises(ValueError):
        CriteriaABTest([], {"name": "a"}, {"fn": lambda *a: 0.5, "name": "b"})


def test_run_raises_when_all_rows_fail():
    def boom(query, response):
        raise RuntimeError("judge down")

    golden = [{"query": "q", "expected_response": "r"}]
    with pytest.raises(ValueError):
        CriteriaABTest(
            golden, {"fn": boom, "name": "a"}, {"fn": boom, "name": "b"}
        ).run()
