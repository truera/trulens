"""Retrieval scores must line up with every retrieved chunk."""

import math

import pytest
from trulens.feedback import groundtruth as feedback_groundtruth
from trulens.feedback.dummy import provider as dummy_provider


@pytest.fixture
def agreement():
    return feedback_groundtruth.GroundTruthAgreement(
        ground_truth=[
            {"query": "q", "expected_chunks": [{"text": "relevant"}]}
        ],
        provider=dummy_provider.DummyProvider(),
    )


@pytest.mark.parametrize(
    "method", ["mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"]
)
@pytest.mark.parametrize("scores", [[], [0.9], [0.9, 0.1, 1.0]])
def test_mismatched_scores_raise(agreement, method, scores):
    with pytest.raises(ValueError, match=rf"2 chunks and {len(scores)} scores"):
        getattr(agreement, method)(
            "q", ["irrelevant", "relevant"], relevance_scores=scores
        )


@pytest.mark.parametrize(
    "method", ["mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"]
)
def test_scores_for_empty_retrieval_raise(agreement, method):
    with pytest.raises(ValueError, match="0 chunks and 1 scores"):
        getattr(agreement, method)("q", [], relevance_scores=[0.9])


@pytest.mark.parametrize(
    "method", ["mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"]
)
def test_equal_length_scores_still_sort_retrieved_chunks(agreement, method):
    chunks = ["irrelevant", "relevant"]
    scores = [0.1, 0.9]
    kwargs = {} if method == "mrr" else {"k": 1}

    result = getattr(agreement, method)(
        "q", chunks, relevance_scores=scores, **kwargs
    )

    assert result == 1.0
    assert chunks == ["irrelevant", "relevant"]
    assert scores == [0.1, 0.9]


@pytest.mark.parametrize(
    "method, expected",
    [
        ("mrr", 0.5),
        ("ndcg_at_k", 1 / math.log2(3)),
        ("precision_at_k", 0.5),
        ("recall_at_k", 1.0),
    ],
)
@pytest.mark.parametrize("scores", [None, [0.0, 0.0]])
def test_omitted_and_tied_scores_keep_existing_results(
    agreement, method, expected, scores
):
    result = getattr(agreement, method)(
        "q", ["irrelevant", "relevant"], relevance_scores=scores
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "method", ["mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"]
)
def test_empty_scores_are_valid_for_empty_retrieval(agreement, method):
    result = getattr(agreement, method)("q", [], relevance_scores=[])

    if method in {"ndcg_at_k", "precision_at_k"}:
        assert math.isnan(result)
    else:
        assert result == 0.0
