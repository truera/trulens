"""Unit tests for trulens.apps.rl.reward."""

import pytest
from trulens.apps.rl import RewardFunction
from trulens.apps.rl import TRLRewardAdapter


def mock_relevance_feedback(
    prompt: str = "", response: str = "", **kwargs
) -> float:
    """Fake feedback function returning similarity between prompt and response lengths."""
    score = min(1.0, (len(prompt) + len(response)) / 20.0)
    return score


def mock_tuple_feedback(prompt: str = "", response: str = "", **kwargs):
    """Fake feedback function returning (score, metadata) tuple."""
    score = 0.8
    return score, {"reason": "Good relevance"}


class TestRewardFunction:
    def test_init_validation(self):
        with pytest.raises(
            ValueError, match="Must supply both app_name and app_version"
        ):
            RewardFunction(mock_relevance_feedback, app_name="test_app")

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transform string"):
            RewardFunction(
                mock_relevance_feedback, transform="invalid_transform"
            )

    def test_transform_2x_minus_1(self):
        adapter = RewardFunction(mock_relevance_feedback, transform="2x-1")
        # score = (4+6)/20 = 0.5 -> reward = 2*0.5 - 1 = 0.0
        reward = adapter.evaluate_sample("test", "output")
        assert reward == pytest.approx(0.0)

        # score = 1.0 -> reward = 1.0
        def perfect_feedback(*args, **kwargs):
            return 1.0

        adapter_perfect = RewardFunction(perfect_feedback, transform="2x-1")
        assert adapter_perfect.evaluate_sample("p", "c") == pytest.approx(1.0)

    def test_transform_identity(self):
        adapter = RewardFunction(mock_relevance_feedback, transform="identity")
        reward = adapter.evaluate_sample("test", "output")
        assert reward == pytest.approx(0.5)

    def test_custom_callable_transform(self):
        custom_transform = lambda s: s * 10.0
        adapter = RewardFunction(
            mock_relevance_feedback, transform=custom_transform
        )
        reward = adapter.evaluate_sample("test", "output")
        assert reward == pytest.approx(5.0)

    def test_batch_call(self):
        adapter = TRLRewardAdapter(
            mock_relevance_feedback, transform="identity"
        )
        prompts = ["p1", "p2"]
        completions = ["c1", "c2"]
        rewards = adapter(prompts, completions)
        assert len(rewards) == 2
        assert isinstance(rewards[0], float)

    def test_mismatched_batch_raises(self):
        adapter = TRLRewardAdapter(mock_relevance_feedback)
        with pytest.raises(ValueError, match="Mismatched batch size"):
            adapter(["p1"], ["c1", "c2"])

    def test_tuple_feedback_result(self):
        adapter = RewardFunction(mock_tuple_feedback, transform="identity")
        reward = adapter.evaluate_sample("p", "c")
        assert reward == pytest.approx(0.8)

    def test_from_metric(self):
        class MockMetric:
            def __init__(self):
                self.implementation = mock_relevance_feedback

        metric = MockMetric()
        adapter = RewardFunction.from_metric(metric, transform="identity")
        reward = adapter.evaluate_sample("test", "output")
        assert reward == pytest.approx(0.5)

    def test_canonical_trulens_signature(self):
        """Test with canonical TruLens provider function signature (keyword-only prompt & response)."""

        def trulens_canonical_feedback(prompt: str, response: str) -> float:
            assert prompt == "hello"
            assert response == "world"
            return 0.8

        adapter = RewardFunction(
            trulens_canonical_feedback, transform="identity"
        )
        assert adapter.evaluate_sample("hello", "world") == pytest.approx(0.8)

    def test_internal_typeerror_propagates(self):
        """Test that internal TypeErrors inside feedback_fn logic are not swallowed."""

        def buggy_feedback(prompt: str, response: str) -> float:
            raise TypeError("Internal provider argument error")

        adapter = RewardFunction(buggy_feedback, transform="identity")
        with pytest.raises(TypeError, match="Internal provider argument error"):
            adapter.evaluate_sample("p", "c")
