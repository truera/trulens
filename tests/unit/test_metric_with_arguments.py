"""`Metric.with_arguments` must not have its bound values overwritten.

`min_score_val`, `max_score_val` and `temperature` default to 0, 3 and 0.0
(not None) on `Metric.__init__`, so `__call__` always considered them "set"
and re-applied the constructor defaults on top of anything bound via
`with_arguments`, silently discarding the bound value on every call.
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
