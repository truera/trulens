"""Guardrails must not interpret failed evaluations as passing scores."""

import pytest
from trulens.core.guardrails import base as guardrails_base
from trulens.core.metric import metric as core_metric


@pytest.mark.parametrize(
    "score", [-1.0, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("higher_is_better", [False, True])
@pytest.mark.parametrize("kind", ["block_input", "block_output"])
def test_invalid_feedback_score_raises(score, higher_is_better, kind):
    """Reject judge failures in either direction before returning any result."""
    calls = []

    def judge(query, context=None):
        return score

    feedback = core_metric.Metric(
        implementation=judge, higher_is_better=higher_is_better
    )

    def app(query):
        calls.append(query)
        return ["context"] if kind == "context_filter" else "response"

    decorator = getattr(guardrails_base, kind)(feedback, threshold=0.5)
    guarded = decorator(app)
    with pytest.raises(ValueError, match="finite.*unparsable"):
        guarded("query")
    if kind == "block_input":
        assert calls == []
    else:
        assert calls == ["query"]


@pytest.mark.parametrize(
    "score", [-1.0, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("higher_is_better", [False, True])
def test_context_filter_drops_only_invalid_context(
    score, higher_is_better, caplog
):
    """A failed judge for the middle context preserves other passing results."""

    def judge(query, context):
        if context == "invalid":
            return score
        return 1.0 if higher_is_better else 0.0

    feedback = core_metric.Metric(
        implementation=judge, higher_is_better=higher_is_better
    )

    @guardrails_base.context_filter(feedback, 0.5, "query")
    def retrieve(query):
        return ["first", "invalid", "last"]

    assert sorted(retrieve("query")) == ["first", "last"]
    assert "invalid guardrail score" in caplog.text


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("higher_is_better", [False, True])
@pytest.mark.parametrize(
    "kind", ["block_input", "block_output", "context_filter"]
)
def test_valid_feedback_score_thresholds(score, higher_is_better, kind):
    """Keep the existing strict/inclusive threshold behavior for valid scores."""

    def judge(query, context=None):
        return score

    feedback = core_metric.Metric(
        implementation=judge, higher_is_better=higher_is_better
    )

    def app(query):
        return ["context"] if kind == "context_filter" else "response"

    guarded = getattr(guardrails_base, kind)(feedback, threshold=0.5)(app)
    if kind == "context_filter":
        passed = score > 0.5 if higher_is_better else score < 0.5
        assert guarded("query") == (["context"] if passed else [])
    else:
        passed = score >= 0.5 if higher_is_better else score <= 0.5
        assert guarded("query") == ("response" if passed else None)
