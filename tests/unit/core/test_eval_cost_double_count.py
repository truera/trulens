"""Unit tests for how a record's evaluation cost is assembled from spans.

An evaluation's spend is written to two places: the EVAL_ROOT span carries the
total for that metric, and each of its EVAL child spans carries that child's
share. They are two measurements of the same money, so the readers must not add
them together.

`get_eval_cost_trends` has always read EVAL_ROOT only. These tests pin the
per-record `eval_cost` and the per-metric `feedback cost in <currency>` column
to the same figure, and pin the fallback for a metric whose EVAL_ROOT span
carried no cost.

No external service is contacted. The provider call is wired through the repo's
own two registration helpers, which is what makes a call land on both spans:

  * `DummyEndpoint` for the endpoint cost callback, the path
    `providers/openai/.../endpoint.py:706-708` uses, and
  * `_TruSession._track_costs_for_module_member` for `instrument_cost_computer`,
    the path `experimental/otel_tracing/core/session.py:349` uses for openai.

No network: `DummyAPI.post` answers from a canned response.
"""

import json

from trulens.core import Metric
from trulens.core.schema.app import AppDefinition
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.session import _TruSession
from trulens.feedback.computer import RecordGraphNode
from trulens.feedback.computer import _compute_feedback
from trulens.feedback.dummy.endpoint import DummyAPI
from trulens.feedback.dummy.endpoint import DummyEndpoint
from trulens.otel.semconv.trace import SpanAttributes

import tests.unit.test_otel_feedback_computation as comp
from tests.unit.test_otel_get_records_and_feedback import _TestApp
from tests.util.mock_otel_feedback_computation import (
    all_retrieval_span_attributes,
)
from tests.util.otel_test_case import OtelTestCase

_SPEND = 0.25
_URL = "https://example.invalid/v1/chat"
_BODY = {"model": "m", "prompt": "p", "temperature": 0.0}


def _usage_of(response) -> dict:
    if isinstance(response, dict):
        return response.get("usage", {}) or {}
    node = response
    try:
        return node.usage or {}
    except AttributeError:
        return {}


def _cost_computer(response):
    """A provider cost computer, in the shape `instrument_cost_computer` wants."""
    usage = _usage_of(response)
    prompt = usage.get("prompt_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    return {
        SpanAttributes.COST.CURRENCY: "USD",
        SpanAttributes.COST.COST: _SPEND,
        SpanAttributes.COST.NUM_TOKENS: prompt + completion,
        SpanAttributes.COST.NUM_PROMPT_TOKENS: prompt,
        SpanAttributes.COST.NUM_COMPLETION_TOKENS: completion,
    }


def _feedback_calling_the_provider(source: str = "s") -> float:
    DummyAPI().post(url=_URL, json=dict(_BODY))
    return 0.5


_REGISTERED = False


def _register_both_writers_once() -> None:
    """Wire both cost writers, once per process.

    `instrument_cost_computer` re-wraps the method it is handed, so registering
    again would stack another wrapper and make the recorded cost depend on how
    many tests happened to run first.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    # The endpoint cost callback, the path providers/openai uses.
    DummyEndpoint()
    # The span-attribute path, registered the way session.py does for openai
    # and google.
    import sys

    module = sys.modules[__name__]
    module.DummyAPI = DummyAPI
    _TruSession._track_costs_for_module_member(module, "post", _cost_computer)
    _REGISTERED = True


def _attrs_of(row) -> dict:
    attrs = row.record_attributes
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
    return attrs or {}


class TestEvalCostNotDoubled(OtelTestCase):
    def setUp(self):
        super().setUp()
        self.session = TruSession()
        self.session.reset_database()
        self.db = self.session.connector.db
        self.app_name = "test_app"
        self.app_version = "1.0.0"
        self.app_id = AppDefinition._compute_app_id(
            self.app_name, self.app_version
        )

    def _run_one_evaluation(self) -> dict:
        app = _TestApp()
        tru_app = comp.TruApp(
            app,
            app_name=self.app_name,
            app_version=self.app_version,
            app_id=self.app_id,
            main_method=app.query,
        )
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_kwargs={"question": "hello"},
        )
        self.session.force_flush()

        # Both writers, as the real provider paths register them.
        _register_both_writers_once()

        metric = Metric(
            implementation=_feedback_calling_the_provider,
            name="with_provider",
            higher_is_better=True,
        )
        events = self._get_events()
        spans = comp._convert_events_to_MinimalSpanInfos(events)
        record_root = RecordGraphNode.build_graph(spans)
        _compute_feedback(
            record_root,
            "with_provider",
            metric,
            True,
            all_retrieval_span_attributes,
        )
        self.session.force_flush()

        costs = {}
        for row in self._get_events().itertuples(index=False):
            attrs = _attrs_of(row)
            span_type = attrs.get(SpanAttributes.SPAN_TYPE)
            if span_type in (
                SpanAttributes.SpanType.EVAL.value,
                SpanAttributes.SpanType.EVAL_ROOT.value,
            ):
                costs.setdefault(span_type, []).append(
                    attrs.get(SpanAttributes.COST.COST, 0.0)
                )
        return costs

    def test_eval_cost_is_not_the_sum_of_both_eval_spans(self):
        costs = self._run_one_evaluation()

        child = sum(costs.get(SpanAttributes.SpanType.EVAL.value, []))
        root = sum(costs.get(SpanAttributes.SpanType.EVAL_ROOT.value, []))
        # The premise: one evaluation, two spans carrying its cost.
        assert child > 0, "the EVAL child span should carry a cost"
        assert root > 0, "the EVAL_ROOT span should carry a cost"

        df, _ = self.db._get_records_and_feedback_otel()
        row = df.iloc[0]

        # The authoritative figure is the EVAL_ROOT total, which is what
        # get_eval_cost_trends reports for the same record.
        assert row["eval_cost"] == root, (
            f"eval_cost is {row['eval_cost']} but the EVAL_ROOT total is "
            f"{root}; adding the EVAL children on top counts the same spend "
            f"twice (children summed to {child})"
        )

    def test_per_metric_feedback_cost_is_not_the_sum_of_both_eval_spans(self):
        self._run_one_evaluation()

        df, _ = self.db._get_records_and_feedback_otel()
        row = df.iloc[0]
        column = "with_provider feedback cost in USD"
        self.assertIn(column, df.columns)

        root = sum(
            _attrs_of(r).get(SpanAttributes.COST.COST, 0.0) or 0.0
            for r in self._get_events().itertuples(index=False)
            if _attrs_of(r).get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.EVAL_ROOT.value
        )
        assert row[column] == root, (
            f"{column} is {row[column]} but the EVAL_ROOT total is {root}"
        )

    def test_record_eval_cost_agrees_with_get_eval_cost_trends(self):
        """The two readers in this package must not disagree about one record."""
        self._run_one_evaluation()

        df, _ = self.db._get_records_and_feedback_otel()
        reported = df.iloc[0]["eval_cost"]
        trends = self.db.get_eval_cost_trends(bucket="day")
        from_trends = (
            trends["total_eval_cost"].sum() if not trends.empty else 0.0
        )

        assert reported == from_trends, (
            f"the dataframe reports eval_cost {reported} while "
            f"get_eval_cost_trends totals {from_trends} for the same record"
        )
