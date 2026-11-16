"""`Metric` must pass its temperature to implementations that take `**kwargs`.

`Metric.__call__` drops every keyword argument the implementation does not
name, so an implementation that accepts `temperature` only through `**kwargs`
never saw the temperature set on the `Metric`.
"""

import unittest

from trulens.core.metric import metric as core_metric


def _kwargs_only(source: str, summary: str, **kwargs) -> dict:
    return kwargs


def _no_kwargs(source: str, summary: str) -> float:
    return 0.5


class TestMetricVarKeywordImplementation(unittest.TestCase):
    def test_temperature_reaches_kwargs_implementation(self) -> None:
        metric = core_metric.Metric(
            implementation=_kwargs_only, temperature=0.7
        )
        self.assertEqual(metric(source="a", summary="b")["temperature"], 0.7)

    def test_call_site_temperature_reaches_kwargs_implementation(
        self,
    ) -> None:
        metric = core_metric.Metric(implementation=_kwargs_only)
        result = metric(source="a", summary="b", temperature=0.2)
        self.assertEqual(result["temperature"], 0.2)

    def test_only_temperature_is_forwarded_to_kwargs(self) -> None:
        metric = core_metric.Metric(
            implementation=_kwargs_only,
            temperature=0.7,
            criteria="strict",
            additional_instructions="be brief",
            examples=[("a", 1)],
            min_score_val=1,
            max_score_val=10,
            groundedness_configs=core_metric.GroundednessConfigs(
                use_sent_tokenize=True, filter_trivial_statements=True
            ),
            enable_trace_compression=False,
        )
        self.assertEqual(metric(source="a", summary="b"), {"temperature": 0.7})

    def test_implementation_without_kwargs_gets_no_temperature(self) -> None:
        metric = core_metric.Metric(implementation=_no_kwargs, temperature=0.7)
        self.assertEqual(metric(source="a", summary="b"), 0.5)


if __name__ == "__main__":
    unittest.main()
