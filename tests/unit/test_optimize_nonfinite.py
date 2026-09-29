"""Non-finite judge predictions must not win few-shot selection."""

import pytest
from trulens.feedback import optimize as feedback_optimize


@pytest.mark.parametrize("metric", ["pearson", "spearman"])
@pytest.mark.parametrize(
    "bad_score", [float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("max_workers", [1, 2])
def test_nonfinite_candidate_does_not_displace_valid_candidate(
    metric, bad_score, max_workers
):
    candidates = [({"name": "invalid"}, 0), ({"name": "valid"}, 1)]

    def judge(value, examples):
        return bad_score if examples[0][0]["name"] == "invalid" else value

    optimizer = feedback_optimize.FewShotOptimizer(
        feedback_fn=judge,
        candidates=candidates,
        eval_dataset=[({"value": 0.0}, 0.0), ({"value": 1.0}, 1.0)],
        examples_format="structured",
        n_examples=1,
        metric=metric,
        max_workers=max_workers,
    )

    result = optimizer.optimize()

    assert result.best_examples == [candidates[1]]
    assert result.metric_score == pytest.approx(1.0)
    assert result.candidate_scores[0] == -1.0


@pytest.mark.parametrize("metric", ["pearson", "spearman"])
def test_nonfinite_prediction_skips_its_label_too(metric):
    candidates = [({"name": "example"}, 1)]

    def judge(value, examples):
        return value

    optimizer = feedback_optimize.FewShotOptimizer(
        feedback_fn=judge,
        candidates=candidates,
        eval_dataset=[
            ({"value": 0.0}, 0.0),
            ({"value": float("nan")}, 100.0),
            ({"value": 1.0}, 1.0),
        ],
        n_examples=1,
        metric=metric,
        max_workers=1,
    )

    result = optimizer.optimize()

    assert result.best_examples == candidates
    assert result.metric_score == pytest.approx(1.0)


@pytest.mark.parametrize("metric", ["pearson", "spearman"])
def test_all_nonfinite_predictions_leave_no_selected_examples(metric):
    def judge(value, examples):
        return float("nan")

    optimizer = feedback_optimize.FewShotOptimizer(
        feedback_fn=judge,
        candidates=[({"name": "example"}, 1)],
        eval_dataset=[({"value": 0.0}, 0.0), ({"value": 1.0}, 1.0)],
        metric=metric,
        max_workers=1,
    )

    result = optimizer.optimize()

    assert result.best_examples == []
    assert result.metric_score is None
