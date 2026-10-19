"""Calibration and error metrics must not score a judge failure as a verdict."""

import contextlib
import io
import logging
import math

import pytest
from trulens.core.utils.constants import UNPARSABLE_SCORE
from trulens.feedback import groundtruth as feedback_groundtruth


def _scores_for(method, scores):
    """Shape `scores` the way `method` takes them: ece wants score/confidence."""
    if method == "ece":
        return [(score, 0.9) for score in scores]
    return scores


def _metric(method, labels, scores):
    aggregator = feedback_groundtruth.GroundTruthAggregator(true_labels=labels)
    return getattr(aggregator, method)(_scores_for(method, scores))


@pytest.mark.parametrize("method", ["brier_score", "mae", "ece"])
def test_perfect_judge_with_one_unparsable_reply_is_still_perfect(method):
    labels = [1, 0, 1, 0]
    perfect = _metric(method, labels, [1.0, 0.0, 1.0, 0.0])
    with_failure = _metric(method, labels, [1.0, 0.0, 1.0, UNPARSABLE_SCORE])

    assert with_failure == perfect


@pytest.mark.parametrize("method", ["brier_score", "mae"])
def test_unparsable_row_is_dropped_with_its_own_label(method):
    # The failure sits in the middle. Dropping only the score would pair the
    # last score (0.0) with the middle label (1) and report an error.
    result = _metric(method, [1, 1, 0], [1.0, UNPARSABLE_SCORE, 0.0])

    assert result == 0.0


@pytest.mark.parametrize("method", ["brier_score", "mae"])
def test_mean_covers_only_the_graded_rows(method):
    # Graded rows are (0.5 vs 1) and (1.0 vs 1): errors 0.5 and 0.0.
    result = _metric(method, [1, 0, 1], [0.5, UNPARSABLE_SCORE, 1.0])

    expected = {"brier_score": 0.25 / 2, "mae": 0.5 / 2}[method]
    assert result == pytest.approx(expected)


def test_ece_ignores_the_unparsable_row_and_keeps_labels_aligned():
    labels = [1, 1, 0, 0]
    with_failure = feedback_groundtruth.GroundTruthAggregator(
        true_labels=labels
    ).ece([(0.9, 0.9), (UNPARSABLE_SCORE, 0.9), (0.2, 0.9), (0.1, 0.9)])
    expected = feedback_groundtruth.GroundTruthAggregator(
        true_labels=[1, 0, 0]
    ).ece([(0.9, 0.9), (0.2, 0.9), (0.1, 0.9)])

    assert with_failure == expected


def test_ece_does_not_read_the_sentinel_as_a_prediction_of_zero():
    # -1.0 is below 0.5, so it used to count as "predicted 0". Against a label
    # of 1 that is a wrong prediction the judge never made.
    aggregator = feedback_groundtruth.GroundTruthAggregator(true_labels=[1, 1])

    result = aggregator.ece([(1.0, 1.0), (UNPARSABLE_SCORE, 1.0)])

    assert result == 0.0


@pytest.mark.parametrize("method", ["brier_score", "mae", "ece"])
def test_all_unparsable_is_undefined_not_a_number_that_looks_like_a_result(
    method,
):
    result = _metric(method, [1, 0], [UNPARSABLE_SCORE, UNPARSABLE_SCORE])

    assert math.isnan(result)


@pytest.mark.parametrize("method", ["brier_score", "mae", "ece"])
def test_empty_input_is_undefined_for_every_metric(method):
    aggregator = feedback_groundtruth.GroundTruthAggregator(true_labels=[])

    assert math.isnan(getattr(aggregator, method)([]))


def test_dropped_rows_are_reported(caplog):
    with caplog.at_level(logging.WARNING, logger=feedback_groundtruth.__name__):
        _metric("mae", [1, 0], [1.0, UNPARSABLE_SCORE])

    assert "ignored 1 of 2 scores" in caplog.text


def test_clean_scores_log_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=feedback_groundtruth.__name__):
        _metric("mae", [1, 0], [1.0, 0.0])

    assert caplog.text == ""


def test_mae_accepts_score_reason_pairs_with_an_unparsable_row():
    aggregator = feedback_groundtruth.GroundTruthAggregator(
        true_labels=[1, 0, 1]
    )

    result = aggregator.mae([[1.0, "ok"], [UNPARSABLE_SCORE, "?"], [1.0, "ok"]])

    assert result == 0.0


def test_mae_does_not_print():
    aggregator = feedback_groundtruth.GroundTruthAggregator(true_labels=[1, 0])
    out = io.StringIO()

    with contextlib.redirect_stdout(out):
        aggregator.mae([[1.0, "r"], [0.0, "r"]])

    assert out.getvalue() == ""


def test_mae_refuses_a_score_list_that_does_not_match_the_labels():
    aggregator = feedback_groundtruth.GroundTruthAggregator(
        true_labels=[1, 0, 1]
    )

    with pytest.raises(ValueError, match="3 labels and 2 scores"):
        aggregator.mae([1.0, 0.0])
