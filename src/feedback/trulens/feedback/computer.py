from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from opentelemetry.trace import INVALID_SPAN_ID
import pandas as pd
from trulens.core.otel.instrument import OtelFeedbackComputationRecordingContext
from trulens.experimental.otel_tracing.core.span import (
    set_span_attribute_safely,
)
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.otel.semconv.trace import SpanAttributes


# If we could just have `opentelemetry.sdk.trace.ReadableSpan` it would be
# better, but this is all we need and it's easier to fill only this info
# from an event table row.
class MinimalSpanInfo:
    span_id: Optional[int] = None
    parent_span_id: Optional[int] = None
    attributes: Dict[str, Any] = {}


class RecordGraphNode:
    """Graph form of a record (i.e. a list of spans)."""

    current_span: MinimalSpanInfo
    parent_span: Optional[MinimalSpanInfo]
    children_spans: List["RecordGraphNode"]

    def __init__(self, span: MinimalSpanInfo):
        self.current_span = span
        self.parent_span = None
        self.children_spans = []

    @staticmethod
    def build_graph(spans: List[MinimalSpanInfo]) -> "RecordGraphNode":
        nodes = [RecordGraphNode(curr) for curr in spans]
        span_id_idx = {curr.span_id: i for i, curr in enumerate(spans)}
        root = None
        for i, span in enumerate(spans):
            if span.parent_span_id is None:
                if root is not None:
                    raise ValueError("Multiple roots found!")
                root = nodes[i]
                continue
            if span.parent_span_id == INVALID_SPAN_ID:
                raise ValueError()
            parent_idx = span_id_idx[span.parent_span_id]
            nodes[parent_idx].children_spans.append(nodes[i])
            nodes[i].parent_span = nodes[parent_idx]
        return root


def _compute_feedback(
    record_root: RecordGraphNode,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    feedback_name: str,
    selector_function: Callable[[RecordGraphNode], List[Dict[str, Any]]],
) -> None:
    """
    Compute feedback for a record. This is a utility function that can compute
    feedback functions quite arbitrarily and so is quite powerful.

    Args:
        record: Record to compute feedback for.
        feedback_function: Function to compute feedback.
        feedback_name: Name of feedback.
        selector_function:
            Function to select inputs for feedback computation. Given a record
            in graph form, it returns a list of inputs to the feedback
            function. Each entry in the list is a dictionary that represents
            the kwargs to the feedback function.
    """
    feedback_inputs = selector_function(record_root)
    record_root_attributes = record_root.current_span.attributes
    if SpanAttributes.APP_NAME in record_root_attributes:
        app_name = record_root_attributes[SpanAttributes.APP_NAME]
        app_version = record_root_attributes[SpanAttributes.APP_VERSION]
        run_name = record_root_attributes[SpanAttributes.RUN_NAME]
    elif f"snow.{BASE_SCOPE}.object.name" in record_root_attributes:
        # TODO(otel, dhuang): need to use these when getting the object entity!
        # database_name = record_root_attributes[
        #    f"snow.{BASE_SCOPE}.database.name"
        # ]
        # schema_name = record_root_attributes[
        #    f"snow.{BASE_SCOPE}.schema.name"
        # ]
        app_name = record_root_attributes[f"snow.{BASE_SCOPE}.object.name"]
        app_version = record_root_attributes[
            f"snow.{BASE_SCOPE}.object.version.name"
        ]
        run_name = record_root_attributes[f"snow.{BASE_SCOPE}.run.name"]
    input_id = record_root_attributes[SpanAttributes.INPUT_ID]
    target_record_id = record_root_attributes[SpanAttributes.RECORD_ID]
    for curr in feedback_inputs:
        context_manager = OtelFeedbackComputationRecordingContext(
            app_name=app_name,
            app_version=app_version,
            run_name=run_name,
            input_id=input_id,
            target_record_id=target_record_id,
            feedback_name=feedback_name,
        )
        with context_manager as eval_root_span:
            try:
                res = feedback_function(**curr)
            except Exception as e:
                eval_root_span.set_attribute(
                    SpanAttributes.EVAL_ROOT.ERROR, str(e)
                )
                raise e
            metadata = {}
            if isinstance(res, tuple):
                if (
                    len(res) != 2
                    or not isinstance(res[0], float)
                    or not isinstance(res[1], dict)
                    or not all([
                        isinstance(curr, str) for curr in res[1].keys()
                    ])
                ):
                    raise ValueError(
                        "Feedback functions must be of type `Callable[Any, Union[float, Tuple[float, Dict[str, Any]]]]`!"
                    )
                res, metadata = res[0], res[1]
            eval_root_span.set_attribute(SpanAttributes.EVAL_ROOT.RESULT, res)
            for k, v in metadata.items():
                set_span_attribute_safely(
                    eval_root_span,
                    f"{SpanAttributes.EVAL_ROOT.METADATA}.{k}",
                    v,
                )


@dataclass
class Selector:
    span_type: Optional[str] = None
    span_name: Optional[str] = None
    span_attribute: Optional[str] = None

    def describes_same_spans(self, other: Selector) -> bool:
        return (
            self.span_type == other.span_type
            and self.span_name == other.span_name
        )


def _group_kwargs_by_selectors(
    kwarg_to_selectors: Dict[str, Selector],
) -> List[List[str]]:
    """Group kwargs by by whether their selectors describe the same spans.

    Args:
        kwarg_to_selectors: kwarg to selector mapping.

    Returns:
        List[List[str]]: List of list of kwargs. Each sublist contains kwargs that describe the same spans in their selector.
    """
    ret = []
    for kwarg, selector in kwarg_to_selectors.items():
        new_kwarg_group = True
        for kwarg_group in ret:
            if selector.describes_same_spans(
                kwarg_to_selectors[kwarg_group[0]]
            ):
                kwarg_group.append(kwarg)
                new_kwarg_group = False
                break
        if new_kwarg_group:
            ret.append([kwarg])
    return ret


def compute_feedback_by_span_group(
    events: pd.DataFrame,
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    kwarg_to_selectors: Dict[str, Selector],
) -> None:
    kwarg_groups = _group_kwargs_by_selectors(kwarg_to_selectors)
    error_messages = []
    unflattened_inputs = {}
    for _, row in events.iterrows():
        record_attributes = row["record_attributes"]
        span_id = row["trace"]["span_id"]
        record_id = record_attributes[SpanAttributes.RECORD_ID]
        span_groups = record_attributes[SpanAttributes.SPAN_GROUPS]
        if span_groups is None:
            span_groups = [None]
        elif isinstance(span_groups, str):
            span_groups = [span_groups]
        for kwarg_group in kwarg_groups:
            row_satisfies_kwarg_group_selector = True
            if kwarg_to_selectors[kwarg_group[0]].span_name not in [
                None,
                record_attributes["name"],
            ]:
                row_satisfies_kwarg_group_selector = False
            if kwarg_to_selectors[kwarg_group[0]].span_type not in [
                None,
                record_attributes[SpanAttributes.SPAN_TYPE],
            ]:
                row_satisfies_kwarg_group_selector = False
            if row_satisfies_kwarg_group_selector:
                valid = True
                kwarg_group_inputs = {}
                for kwarg in kwarg_group:
                    span_attribute = kwarg_to_selectors[kwarg].span_attribute
                    if span_attribute not in record_attributes:
                        error_messages.append(
                            # TODO(this_pr): give better error message.
                            f"Span attribute {span_attribute} not found in record attributes for record id={record_id}, span_id={span_id}!"
                        )
                        valid = False
                    else:
                        kwarg_group_inputs[kwarg] = record_attributes[
                            span_attribute
                        ]
                if valid:
                    for span_group in span_groups:
                        unflattened_inputs[(record_id, span_group)][
                            kwarg_group
                        ] = kwarg_group_inputs
    # Pare down feedback function invocations.
    if len(kwarg_groups) > 1:
        keys_to_remove = []
        for (
            record_id,
            span_group,
        ), kwarg_group_to_inputs in unflattened_inputs.items():
            valid = True
            if len(kwarg_group_to_inputs) != len(kwarg_groups):
                error_messages.append(
                    # TODO(this_pr): give better error message.
                    f"For record id {record_id} and span group {span_group}, missing input to feedback {feedback_name}!"
                )
                valid = False
            len_kwarg_group_inputs = [
                len(inputs) for inputs in kwarg_group_to_inputs.values()
            ]
            len_kwarg_group_inputs = sorted(len_kwarg_group_inputs)
            if len_kwarg_group_inputs[-2] > 1:
                error_messages.append(
                    # TODO(this_pr): give better error message.
                    f"For record id {record_id} and span group {span_group}, multiple possible inputs to feedback {feedback_name}!"
                )
                valid = False
            if not valid:
                keys_to_remove.append((record_id, span_group))
        for key in keys_to_remove:
            del unflattened_inputs[key]
    # Flatten out all inputs to feedback function via cartesian product.
    flattened_inputs = []
    for (
        record_id,
        span_group,
    ), kwarg_group_to_inputs in unflattened_inputs.items():
        cartesian_product = itertools.product(kwarg_group_inputs.values())
        for entry in cartesian_product:
            res = {}
            for curr in entry:
                res.update(curr)
            flattened_inputs.append(res)
    # Invoke.
    for inputs in flattened_inputs:
        try:
            call(feedback_function, feedback_name, inputs)
        except Exception as e:
            error_messages.append(str(e))
    # Raise any error if necessary.
    if error_messages:
        error_message = "Found the following errors:\n"
        for i, curr in enumerate(error_messages):
            error_message += f"{i}. {curr}\n"
        raise ValueError(error_message)
