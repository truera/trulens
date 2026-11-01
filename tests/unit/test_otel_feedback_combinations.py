"""Tests for FeedbackCombinations handling in OTEL feedback computation."""

from unittest.mock import patch

import pandas as pd
from trulens.core import Metric
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.feedback.selector import Selector
from trulens.core.schema import feedback as feedback_schema
from trulens.feedback import computer
from trulens.otel.semconv.trace import SpanAttributes

RECORD_ID = "record1"


def _implementation(a, b):
    return 1.0


def _make_metric(combinations: feedback_schema.FeedbackCombinations) -> Metric:
    return Metric(
        implementation=_implementation,
        name="combination_metric",
        selectors={
            "a": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
            ),
            "b": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ),
        },
    ).aggregate(combinations=combinations)


def _unflattened_inputs(*, a_values: int, b_values: int) -> dict:
    return {
        (RECORD_ID, None): {
            ("a",): [
                {"a": FeedbackFunctionInput(value=index)}
                for index in range(a_values)
            ],
            ("b",): [
                {"b": FeedbackFunctionInput(value=index)}
                for index in range(b_values)
            ],
        }
    }


def _run_feedback_computation(
    combinations: feedback_schema.FeedbackCombinations,
    *,
    a_values: int = 2,
    b_values: int = 2,
) -> list[dict]:
    """Run compute_feedback_by_span_group with mocked input collection."""
    captured = []

    def fake_run(flattened_inputs, *args, **kwargs):
        captured.extend(flattened_inputs)
        return len(flattened_inputs)

    unflattened_inputs = _unflattened_inputs(
        a_values=a_values, b_values=b_values
    )
    with (
        patch.object(
            computer,
            "_collect_inputs_from_events",
            side_effect=lambda *args, **kwargs: unflattened_inputs,
        ),
        patch.object(
            computer,
            "_map_record_id_to_record_roots",
            return_value={
                RECORD_ID: pd.Series({"record_attributes": {}}),
            },
        ),
        patch.object(computer, "_run_feedback_on_inputs", side_effect=fake_run),
    ):
        computer.compute_feedback_by_span_group(
            pd.DataFrame(),
            _make_metric(combinations),
            raise_error_on_no_feedbacks_computed=False,
        )
    return [
        {name: value.value for name, value in inputs.items()}
        for _, _, inputs in captured
    ]


def test_zip_combinations_pair_values_by_position() -> None:
    calls = _run_feedback_computation(feedback_schema.FeedbackCombinations.ZIP)
    assert calls == [{"a": 0, "b": 0}, {"a": 1, "b": 1}]


def test_zip_combinations_truncate_to_the_shortest_group() -> None:
    calls = _run_feedback_computation(
        feedback_schema.FeedbackCombinations.ZIP, b_values=3
    )
    assert calls == [{"a": 0, "b": 0}, {"a": 1, "b": 1}]


def test_product_still_drops_ambiguous_multi_value_inputs() -> None:
    # PRODUCT mode cannot pair two multi-value kwarg groups, so the existing
    # behavior of dropping those ambiguous inputs is unchanged.
    calls = _run_feedback_computation(
        feedback_schema.FeedbackCombinations.PRODUCT
    )
    assert calls == []


def test_flatten_inputs_zip_pairs_by_position() -> None:
    unflattened_inputs = _unflattened_inputs(a_values=2, b_values=2)
    flattened = computer._flatten_inputs(
        unflattened_inputs,
        combinations=feedback_schema.FeedbackCombinations.ZIP,
    )
    assert [
        {name: value.value for name, value in inputs.items()}
        for _, _, inputs in flattened
    ] == [{"a": 0, "b": 0}, {"a": 1, "b": 1}]


def test_validate_unflattened_inputs_zip_keeps_paired_inputs() -> None:
    unflattened_inputs = _unflattened_inputs(a_values=2, b_values=2)
    validated = computer._validate_unflattened_inputs(
        unflattened_inputs,
        [("a",), ("b",)],
        [RECORD_ID],
        "combination_metric",
        combinations=feedback_schema.FeedbackCombinations.ZIP,
    )
    assert list(validated) == [(RECORD_ID, None)]
