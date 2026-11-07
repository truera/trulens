"""`Metric.with_arguments` must not have its bound values overwritten.

`min_score_val`, `max_score_val` and `temperature` default to 0, 3 and 0.0
(not None) on `Metric.__init__`, so `__call__` always considered them "set"
and re-applied the constructor defaults on top of anything bound via
`with_arguments`, silently discarding the bound value on every call.
"""

import unittest
from unittest import mock

from trulens.core.feedback import selector as feedback_selector
from trulens.core.metric.metric import Metric


class TestMetricWithArguments(unittest.TestCase):
    def _imp(
        self,
        x: str,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> dict:
        return {
            "x": x,
            "min_score_val": min_score_val,
            "max_score_val": max_score_val,
            "temperature": temperature,
        }

    def test_bound_min_score_val_is_not_overwritten(self) -> None:
        metric = Metric(implementation=self._imp).with_arguments(
            min_score_val=5
        )
        self.assertEqual(metric(x="a")["min_score_val"], 5)

    def test_bound_max_score_val_is_not_overwritten(self) -> None:
        metric = Metric(implementation=self._imp).with_arguments(
            max_score_val=10
        )
        self.assertEqual(metric(x="a")["max_score_val"], 10)

    def test_bound_temperature_is_not_overwritten(self) -> None:
        metric = Metric(implementation=self._imp).with_arguments(
            temperature=0.7
        )
        self.assertEqual(metric(x="a")["temperature"], 0.7)

    def test_all_three_bound_together(self) -> None:
        metric = Metric(implementation=self._imp).with_arguments(
            min_score_val=1, max_score_val=9, temperature=0.9
        )
        result = metric(x="a")
        self.assertEqual(result["min_score_val"], 1)
        self.assertEqual(result["max_score_val"], 9)
        self.assertEqual(result["temperature"], 0.9)

    def test_unbound_fields_still_use_constructor_defaults(self) -> None:
        metric = Metric(
            implementation=self._imp,
            min_score_val=2,
            max_score_val=8,
            temperature=0.3,
        )
        result = metric(x="a")
        self.assertEqual(result["min_score_val"], 2)
        self.assertEqual(result["max_score_val"], 8)
        self.assertEqual(result["temperature"], 0.3)

    def test_on_default_with_bound_threshold_uses_output(self) -> None:
        def _score(text: str, threshold: float = 1.0) -> float:
            return len(text) * threshold

        with mock.patch(
            "trulens.core.metric.metric.is_otel_tracing_enabled",
            return_value=True,
        ):
            metric = (
                Metric(implementation=_score)
                .with_arguments(threshold=0.125)
                .on_default()
            )

            self.assertEqual(
                metric.selectors,
                {"text": feedback_selector.Selector.select_record_output()},
            )
            self.assertEqual(metric.implementation_kwargs, {"threshold": 0.125})
            metric.check_otel_selectors()
            self.assertEqual(metric(text="test"), 0.5)

    def test_on_default_with_bound_threshold_uses_input_output(self) -> None:
        def _score(prompt: str, response: str, threshold: float = 1.0) -> float:
            return (len(prompt) + len(response)) * threshold

        with mock.patch(
            "trulens.core.metric.metric.is_otel_tracing_enabled",
            return_value=True,
        ):
            metric = (
                Metric(implementation=_score)
                .with_arguments(threshold=0.125)
                .on_default()
            )

            self.assertEqual(
                metric.selectors,
                {
                    "prompt": (
                        feedback_selector.Selector.select_record_input()
                    ),
                    "response": (
                        feedback_selector.Selector.select_record_output()
                    ),
                },
            )
            self.assertEqual(metric.implementation_kwargs, {"threshold": 0.125})
            metric.check_otel_selectors()
            self.assertEqual(metric(prompt="in", response="reply"), 0.875)


if __name__ == "__main__":
    unittest.main()
