"""Unit tests for the score distribution analyzer. No model or API key needed."""

import pytest

score_distribution = pytest.importorskip("trulens.benchmark.score_distribution")
ScoreDistributionAnalyzer = score_distribution.ScoreDistributionAnalyzer
ScoreDistributionReport = score_distribution.ScoreDistributionReport


def test_well_spread_scores_have_no_flags():
    predicted = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 0.1, 0.45, 0.9]
    expected = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 0.1, 0.5, 0.9]
    report = ScoreDistributionReport(predicted, expected)
    assert report.std > 0.2
    assert report.unique_count >= 3
    assert report.entropy > 0.5
    assert report.flags() == []


def test_constant_judge_flagged_low_discrimination():
    report = ScoreDistributionReport([0.5] * 20)
    assert report.std < 0.1
    assert any("Low discrimination" in f for f in report.flags())


def test_lenient_judge_flagged():
    report = ScoreDistributionReport([0.6, 0.7, 0.8, 0.9, 1.0, 0.75])
    assert any("leniency" in f.lower() for f in report.flags())


def test_bimodal_judge_flagged():
    report = ScoreDistributionReport([0.0, 0.05, 0.95, 1.0, 0.1, 0.9])
    assert any("Bimodal" in f for f in report.flags())


def test_histogram_counts_sum_to_n():
    predicted = [0.1, 0.2, 0.25, 0.9, 0.95, 0.5]
    report = ScoreDistributionReport(predicted, num_bins=10)
    assert sum(count for _, _, count in report.histogram()) == len(predicted)


def test_calibration_curve_reveals_miscalibration():
    # judge says ~0.9 for examples humans scored ~0.25
    predicted = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    expected = [0.2, 0.3, 0.25, 0.8, 0.7, 0.75]
    report = ScoreDistributionReport(predicted, expected)
    high_bin = [point for point in report.calibration_curve() if point[0] > 0.8]
    assert high_bin
    assert high_bin[0][1] == pytest.approx(0.25, abs=0.05)


def test_bucket_spread_separates_expected_levels():
    predicted = [0.1, 0.15, 0.5, 0.55, 0.9, 0.92]
    expected = [0.0, 0.1, 0.5, 0.5, 0.9, 1.0]
    spread = ScoreDistributionReport(predicted, expected).bucket_spread()
    assert spread["low"]["predicted_mean"] < spread["high"]["predicted_mean"]


def test_analyzer_runs_feedback_fn_over_golden_set():
    golden = [
        {"query": "q1", "expected_response": "r1", "expected_score": 0.2},
        {"query": "q2", "expected_response": "r2", "expected_score": 0.8},
    ]
    scores = iter([0.25, 0.75])

    def fake_fn(query, response):
        return next(scores)

    report = ScoreDistributionAnalyzer(fake_fn, golden).run()
    assert report.count == 2
    assert report.predicted.tolist() == [0.25, 0.75]


def test_analyzer_handles_tuple_and_dict_returns():
    golden = [{"query": "q", "expected_response": "r", "expected_score": 0.5}]

    def tuple_fn(query, response):
        return (0.6, {"reason": "because"})

    report = ScoreDistributionAnalyzer(tuple_fn, golden).run()
    assert report.predicted.tolist() == [0.6]


def test_analyzer_raises_when_no_scores_produced():
    def boom(query, response):
        raise RuntimeError("judge is down")

    golden = [{"query": "q", "expected_response": "r"}]
    with pytest.raises(ValueError):
        ScoreDistributionAnalyzer(boom, golden).run()
