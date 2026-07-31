"""Reinforcement Learning (RL) reward function adapter for TruLens feedback functions.

Provides:
- :class:`RewardFunction`: Wraps any TruLens feedback function into a scalar reward function
  suitable for RL fine-tuning (e.g. Hugging Face TRL PPOTrainer / GRPOTrainer).
- :class:`TRLRewardAdapter`: High-level adapter specifically designed for TRL reward_funcs signature.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

logger = logging.getLogger(__name__)


def transform_2x_minus_1(score: float) -> float:
    """Transform score in [0, 1] to reward signal in [-1, 1]."""
    return 2.0 * float(score) - 1.0


def transform_identity(score: float) -> float:
    """Identity transform (returns score unchanged in [0, 1])."""
    return float(score)


class RewardFunction:
    """Adapts a TruLens feedback function into an RL reward signal.

    Args:
        feedback_fn: A TruLens feedback callable returning float or (float, dict).
        transform: Optional function or string name ("2x-1", "identity") to transform [0,1] score to reward.
        app_name: Optional virtual app name for logging.
        app_version: Optional virtual app version for logging.
    """

    def __init__(
        self,
        feedback_fn: Callable[..., Any],
        *,
        transform: str | Callable[[float], float] = "2x-1",
        app_name: str | None = None,
        app_version: str | None = None,
    ) -> None:
        self.feedback_fn = feedback_fn

        if isinstance(transform, str):
            if transform in ("2x-1", "2*score - 1", "2*s-1"):
                self._transform_fn = transform_2x_minus_1
            elif transform in ("identity", "none", "passthrough"):
                self._transform_fn = transform_identity
            else:
                raise ValueError(
                    f"Unknown transform string '{transform}'. Use '2x-1', 'identity', or pass a custom callable."
                )
        elif callable(transform):
            self._transform_fn = transform
        else:
            self._transform_fn = transform_identity

        self._recorder: Any | None = None
        if (app_name is None) != (app_version is None):
            raise ValueError(
                "Must supply both app_name and app_version to enable logging, or neither."
            )
        if app_name is not None and app_version is not None:
            from trulens.apps.virtual import TruVirtual

            self._recorder = TruVirtual(
                app_name=app_name, app_version=app_version
            )

    @classmethod
    def from_metric(
        cls,
        metric: Any,
        *,
        transform: str | Callable[[float], float] = "2x-1",
        app_name: str | None = None,
        app_version: str | None = None,
    ) -> RewardFunction:
        """Create a RewardFunction directly from a TruLens Metric object or callable."""
        if hasattr(metric, "implementation") and callable(
            metric.implementation
        ):
            feedback_fn = metric.implementation
        elif callable(metric):
            feedback_fn = metric
        else:
            raise ValueError(
                f"Cannot extract callable implementation from metric: {metric}"
            )

        return cls(
            feedback_fn=feedback_fn,
            transform=transform,
            app_name=app_name,
            app_version=app_version,
        )

    def evaluate_sample(
        self, prompt: str, completion: str, **kwargs: Any
    ) -> float:
        """Evaluate a single (prompt, completion) pair and return its scalar reward."""
        try:
            raw_result = self.feedback_fn(
                prompt=prompt, response=completion, **kwargs
            )
        except TypeError:
            try:
                raw_result = self.feedback_fn(prompt=prompt, **kwargs)
            except TypeError:
                raw_result = self.feedback_fn(prompt, completion, **kwargs)

        score = (
            float(raw_result[0])
            if isinstance(raw_result, tuple)
            else float(raw_result)
        )
        reward = self._transform_fn(score)

        if self._recorder is not None:
            self._log_reward(
                prompt=prompt, completion=completion, score=score, reward=reward
            )

        return reward

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """Evaluate a batch of prompts and completions, returning a list of float rewards."""
        if len(prompts) != len(completions):
            raise ValueError(
                f"Mismatched batch size: {len(prompts)} prompts vs {len(completions)} completions."
            )

        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            reward = self.evaluate_sample(prompt, completion, **kwargs)
            rewards.append(reward)

        return rewards

    def _log_reward(
        self, prompt: str, completion: str, score: float, reward: float
    ) -> None:
        """Log reward evaluation trajectory."""
        try:
            from trulens.apps.virtual import VirtualRecord
            from trulens.core import Select

            call_selector = Select.RecordCalls.reward_fn.evaluate
            record = VirtualRecord(
                main_input=f"Prompt: {prompt} | Completion: {completion}",
                main_output=str(reward),
                calls={
                    call_selector: {
                        "args": [prompt, completion],
                        "rets": {"score": score, "reward": reward},
                    }
                },
            )
            self._recorder.add_record(record)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to log RL reward evaluation to TruLens: %s", exc
            )


class TRLRewardAdapter(RewardFunction):
    """TRL (Transformer Reinforcement Learning) adapter wrapping TruLens metrics as TRL reward_funcs."""
