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


_GENERATION_INPUT_ATTRIBUTE = "test.generation.input"
_RECORD_ROOT_SPAN_ID = "span-root"


def _events_row(
    *,
    span_id: str,
    parent_id: str | None,
    start_timestamp: float,
    span_type: str,
    name: str,
    attributes: dict | None = None,
) -> dict:
    return {
        "record_attributes": {
            SpanAttributes.SPAN_TYPE: span_type,
            SpanAttributes.RECORD_ID: RECORD_ID,
            **(attributes or {}),
        },
        "trace": {"span_id": span_id, "parent_id": parent_id},
        "record": {"name": name, "events": []},
        "start_timestamp": start_timestamp,
    }


def _zip_pairs_from_events(events: pd.DataFrame) -> list[dict]:
    """Collect spans from events and flatten them with ZIP combinations."""
    selectors = {
        "q": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
        ),
        "a": Selector(
            span_type=SpanAttributes.SpanType.GENERATION,
            span_attribute=_GENERATION_INPUT_ATTRIBUTE,
        ),
    }
    groups: list[tuple[str]] = computer._group_kwargs_by_selectors(selectors)
    unflattened = computer._collect_inputs_from_events(
        events, groups, selectors
    )
    flattened = computer._flatten_inputs(
        unflattened,
        combinations=feedback_schema.FeedbackCombinations.ZIP,
    )
    return [
        {name: value.value for name, value in inputs.items()}
        for _, _, inputs in flattened
    ]


def test_zip_combinations_pair_spans_in_start_timestamp_order() -> None:
    """ZIP pairing must not depend on the events table row order.

    ``get_events`` has no ORDER BY, and concurrent retrievals can change
    the order of event rows. ZIP pairs values by position, so traversal
    must sort spans by start_timestamp (like the conversation tool spans
    in ``_conversation_tool_spans_by_record_id``) instead of row order.
    """
    rows = [
        _events_row(
            span_id=_RECORD_ROOT_SPAN_ID,
            parent_id=None,
            start_timestamp=0.0,
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            name="record_root",
        ),
        _events_row(
            span_id="span-r1",
            parent_id=_RECORD_ROOT_SPAN_ID,
            start_timestamp=1.0,
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            name="retrieve",
            attributes={SpanAttributes.RETRIEVAL.QUERY_TEXT: "q1"},
        ),
        _events_row(
            span_id="span-r2",
            parent_id=_RECORD_ROOT_SPAN_ID,
            start_timestamp=2.0,
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            name="retrieve",
            attributes={SpanAttributes.RETRIEVAL.QUERY_TEXT: "q2"},
        ),
        _events_row(
            span_id="span-g1",
            parent_id=_RECORD_ROOT_SPAN_ID,
            start_timestamp=1.5,
            span_type=SpanAttributes.SpanType.GENERATION,
            name="generate",
            attributes={_GENERATION_INPUT_ATTRIBUTE: "a1"},
        ),
        _events_row(
            span_id="span-g2",
            parent_id=_RECORD_ROOT_SPAN_ID,
            start_timestamp=2.5,
            span_type=SpanAttributes.SpanType.GENERATION,
            name="generate",
            attributes={_GENERATION_INPUT_ATTRIBUTE: "a2"},
        ),
    ]
    rows_by_span_id = {row["trace"]["span_id"]: row for row in rows}
    expected = [{"q": "q1", "a": "a1"}, {"q": "q2", "a": "a2"}]

    for order in (
        ("span-r2", "span-g1", "span-r1", "span-g2"),
        ("span-g2", "span-r1", "span-g1", "span-r2"),
        ("span-g1", "span-r2", "span-g2", "span-r1"),
    ):
        events = pd.DataFrame(
            [rows_by_span_id[_RECORD_ROOT_SPAN_ID]]
            + [rows_by_span_id[span_id] for span_id in order]
        )
        assert _zip_pairs_from_events(events) == expected
