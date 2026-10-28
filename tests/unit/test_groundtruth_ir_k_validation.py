"""`k` must be positive, and NumPy relevance scores must be accepted."""

import numpy as np
import pytest
from trulens.feedback import groundtruth as feedback_groundtruth
from trulens.feedback.dummy import provider as dummy_provider

K_METHODS = ["ndcg_at_k", "precision_at_k", "recall_at_k", "ir_hit_rate"]
SCORED_METHODS = ["mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"]
CHUNKS = ["a", "x", "b", "y"]


@pytest.fixture
def agreement():
    return feedback_groundtruth.GroundTruthAgreement(
        ground_truth=[
            {
                "query": "q",
                "expected_chunks": [{"text": "a"}, {"text": "b"}],
            }
        ],
        provider=dummy_provider.DummyProvider(),
    )


@pytest.mark.parametrize("method", K_METHODS)
@pytest.mark.parametrize("k", [0, -1, -10])
def test_k_below_one_raises(agreement, method, k):
    # k=0 used to mean "every chunk", and a negative k sliced from the end.
    with pytest.raises(ValueError, match=rf"k must be .* got {k}"):
        getattr(agreement, method)("q", CHUNKS, k=k)


@pytest.mark.parametrize("method", K_METHODS)
def test_k_is_checked_even_without_a_golden_set_match(agreement, method):
    with pytest.raises(ValueError, match="k must be"):
        getattr(agreement, method)("unknown query", CHUNKS, k=0)


@pytest.mark.parametrize("method", K_METHODS)
def test_k_none_and_positive_k_still_work(agreement, method):
    assert not np.isnan(getattr(agreement, method)("q", CHUNKS))
    assert not np.isnan(getattr(agreement, method)("q", CHUNKS, k=1))
    assert not np.isnan(getattr(agreement, method)("q", CHUNKS, k=100))


def test_k_values_unchanged(agreement):
    assert agreement.precision_at_k("q", CHUNKS, k=2) == pytest.approx(0.5)
    assert agreement.recall_at_k("q", CHUNKS, k=1) == pytest.approx(0.5)
    assert agreement.ir_hit_rate("q", CHUNKS, k=1) == 1.0


@pytest.mark.parametrize("method", SCORED_METHODS)
def test_numpy_relevance_scores_match_a_list(agreement, method):
    scores = [0.9, 0.1, 0.8, 0.2]
    kwargs = {} if method == "mrr" else {"k": 2}
    as_list = getattr(agreement, method)(
        "q", CHUNKS, relevance_scores=scores, **kwargs
    )
    as_array = getattr(agreement, method)(
        "q", CHUNKS, relevance_scores=np.array(scores), **kwargs
    )
    assert as_array == pytest.approx(as_list)


@pytest.mark.parametrize("method", SCORED_METHODS)
def test_empty_scores_for_empty_retrieval_still_work(agreement, method):
    result = getattr(agreement, method)("q", [], relevance_scores=[])
    assert np.isnan(result) or result == 0.0
