from __future__ import annotations

from collections import defaultdict
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
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    selector_function: Callable[[RecordGraphNode], List[Dict[str, Any]]],
) -> None:
    """
    Compute feedback for a record. This is a utility function that can compute
    feedback functions quite arbitrarily and so is quite powerful.

    Args:
        record: Record to compute feedback for.
        feedback_name: Name of feedback.
        feedback_function: Function to compute feedback.
        selector_function:
            Function to select inputs for feedback computation. Given a record
            in graph form, it returns a list of inputs to the feedback
            function. Each entry in the list is a dictionary that represents
            the kwargs to the feedback function.
    """
    feedback_inputs = selector_function(record_root)
    record_root_attributes = record_root.current_span.attributes
    for curr in feedback_inputs:
        _call_feedback_function(
            feedback_name, feedback_function, curr, record_root_attributes
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


def compute_feedback_by_span_group(
    events: pd.DataFrame,
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    kwarg_to_selector: Dict[str, Selector],
) -> None:
    """
    Compute feedback based on span groups in events.

    Args:
        events: DataFrame containing trace events.
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        kwarg_to_selector: Mapping from function kwargs to span selectors
    """
    error_messages = []
    kwarg_groups = _group_kwargs_by_selectors(kwarg_to_selector)
    unflattened_inputs = _collect_inputs_from_events(
        events, kwarg_groups, kwarg_to_selector, error_messages
    )
    record_id_to_record_roots = _map_record_id_to_record_roots(events)
    unflattened_inputs = _validate_unflattened_inputs(
        unflattened_inputs,
        kwarg_groups,
        list(record_id_to_record_roots.keys()),
        feedback_name,
        error_messages,
    )
    flattened_inputs = _flatten_inputs(unflattened_inputs)
    _run_feedback_on_inputs(
        flattened_inputs,
        feedback_name,
        feedback_function,
        record_id_to_record_roots,
        error_messages,
    )
    _report_errors(error_messages)


def _group_kwargs_by_selectors(
    kwarg_to_selector: Dict[str, Selector],
) -> List[Tuple[str]]:
    """Group kwargs by by whether their selectors describe the same spans.

    Args:
        kwarg_to_selector: kwarg to selector mapping.

    Returns:
        List of tuples of kwargs. Each tuple contains kwargs that describe the same spans in their selector.
    """
    ret = []
    for kwarg, selector in kwarg_to_selector.items():
        new_kwarg_group = True
        for kwarg_group in ret:
            if selector.describes_same_spans(kwarg_to_selector[kwarg_group[0]]):
                kwarg_group.append(kwarg)
                new_kwarg_group = False
                break
        if new_kwarg_group:
            ret.append([kwarg])
    ret = [tuple(curr) for curr in ret]
    return ret


def _collect_inputs_from_events(
    events: pd.DataFrame,
    kwarg_groups: List[Tuple[str]],
    kwarg_to_selector: Dict[str, Selector],
    error_messages: List[str],
) -> Dict[Tuple[str, Optional[str]], Dict[Tuple, List[Dict[str, Any]]]]:
    """Collect inputs from events based on selectors.

    Args:
        events: DataFrame containing trace events.
        kwarg_groups: List of list of kwargs. Each sublist contains kwargs that describe the same spans in their selector.
        kwarg_to_selector: Mapping from function kwargs to span selectors.
        error_messages: List of error messages to append to if any errors occur during this function.

    Returns:
        Mapping from (record_id, span_group) to kwarg group to inputs.
    """
    ret = defaultdict(lambda: defaultdict(list))

    for _, row in events.iterrows():
        record_attributes = row["record_attributes"]
        span_id = row["trace"]["span_id"]
        record_id = record_attributes[SpanAttributes.RECORD_ID]

        # Handle span groups.
        span_groups = record_attributes.get(SpanAttributes.SPAN_GROUPS, [None])
        if isinstance(span_groups, str):
            span_groups = [span_groups]
        elif span_groups is None:
            span_groups = [None]

        # Process each kwarg group.
        for kwarg_group in kwarg_groups:
            # Check if row satisfies selector conditions.
            selector = kwarg_to_selector[kwarg_group[0]]
            if _row_matches_selector(record_attributes, selector):
                # Collect inputs for this kwarg group.
                kwarg_group_inputs = {}
                valid = True

                for kwarg in kwarg_group:
                    span_attribute = kwarg_to_selector[kwarg].span_attribute
                    if span_attribute not in record_attributes:
                        error_messages.append(
                            f"Span attribute '{span_attribute}' not found in record_id={record_id}, span_id={span_id}"
                        )
                        valid = False
                        break
                    else:
                        kwarg_group_inputs[kwarg] = record_attributes[
                            span_attribute
                        ]

                if valid:
                    for span_group in span_groups:
                        ret[(record_id, span_group)][kwarg_group].append(
                            kwarg_group_inputs
                        )

    return ret


def _map_record_id_to_record_roots(
    events: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Map record_id to record roots.

    Args:
        events: DataFrame containing trace events.

    Returns:
        Mapping from record_id to record roots.
    """
    ret = {}
    for _, row in events.iterrows():
        record_attributes = row["record_attributes"]
        if (
            record_attributes.get(SpanAttributes.SPAN_TYPE, None)
            == SpanAttributes.SpanType.RECORD_ROOT
        ):
            record_id = record_attributes[SpanAttributes.RECORD_ID]
            ret[record_id] = row
    return ret


def _row_matches_selector(
    record_attributes: Dict[str, Any], selector: Selector
) -> bool:
    """Check if a record matches the given selector.

    Args:
        record_attributes: attributes of row/span.
        selector: Selector to check against.

    Returns:
        True iff the row/span matches the selector.
    """
    if selector.span_name is not None:
        if (
            "name" not in record_attributes
            or record_attributes.get("name") != selector.span_name
        ):
            return False
    if selector.span_type is not None:
        if (
            SpanAttributes.SPAN_TYPE not in record_attributes
            or record_attributes.get(SpanAttributes.SPAN_TYPE)
            != selector.span_type
        ):
            return False
    return True


def _validate_unflattened_inputs(
    unflattened_inputs: Dict[
        Tuple[str, Optional[str]], Dict[Tuple, List[Dict[str, Any]]]
    ],
    kwarg_groups: List[Tuple[str]],
    record_ids_with_record_roots: List[str],
    feedback_name: str,
    error_messages: List[str],
) -> Dict[Tuple[str, Optional[str]], Dict[Tuple, List[Dict[str, Any]]]]:
    """Validate collected inputs and remove invalid entries.

    Args:
        unflattened_inputs: Mapping from (record_id, span_group) to kwarg group to inputs.
        kwarg_groups: List of list of kwargs. Each sublist contains kwargs that describe the same spans in their selector.
        feedback_name: Name of the feedback function.
        error_messages: List of error messages to append to if any errors occur during this function.

    Returns:
        Validated mapping from (record_id, span_group) to kwarg group to inputs.
    """
    keys_to_remove = []

    if len(kwarg_groups) > 1:
        for (
            record_id,
            span_group,
        ), kwarg_group_to_inputs in unflattened_inputs.items():
            if len(kwarg_group_to_inputs) != len(kwarg_groups):
                error_messages.append(
                    f"record_id={record_id}, span_group={span_group} is missing required inputs for feedback '{feedback_name}'"
                )
                keys_to_remove.append((record_id, span_group))
                continue

            # Check for ambiguous inputs (multiple possible values).
            group_sizes = sorted([
                len(inputs) for inputs in kwarg_group_to_inputs.values()
            ])
            if (
                group_sizes[-2] > 1
            ):  # If second largest group has multiple items then unclear how to combine.
                error_messages.append(
                    # TODO(this_pr): go over all error messages.
                    f"record={record_id}, span_group={span_group} has ambiguous inputs for feedback '{feedback_name}'"
                )
                keys_to_remove.append((record_id, span_group))

    for record_id, span_group in unflattened_inputs:
        if record_id not in record_ids_with_record_roots:
            error_messages.append(
                f"record_id={record_id} not found in events for feedback '{feedback_name}'"
            )
            keys_to_remove.append((record_id, span_group))

    # Remove invalid entries.
    for key in keys_to_remove:
        if key in unflattened_inputs:
            del unflattened_inputs[key]

    return unflattened_inputs


def _flatten_inputs(
    unflattened_inputs: Dict[
        Tuple[str, Optional[str]], Dict[Tuple, List[Dict[str, Any]]]
    ],
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    """Flatten inputs via cartesian product.

    Args:
        unflattened_inputs: Mapping from (record_id, span_group) to kwarg group to inputs to flatten.

    Returns:
        Flattened inputs. Each entry is a tuple of (record_id, span_group, inputs).
    """
    ret = []
    for (
        record_id,
        span_group,
    ), kwarg_group_to_inputs in unflattened_inputs.items():
        # Create a list of dictionaries to use in the product.
        input_dicts = list(kwarg_group_to_inputs.values())
        if not input_dicts:
            continue
        # For each combination, merge the dictionaries.
        for combination in itertools.product(*input_dicts):
            merged_input = {}
            for input_dict in combination:
                merged_input.update(input_dict)
            ret.append((record_id, span_group, merged_input))
    return ret


def _run_feedback_on_inputs(
    flattened_inputs: List[Tuple[str, Optional[str], Dict[str, Any]]],
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    record_id_to_record_root: Dict[str, pd.Series],
    error_messages: List[str],
) -> None:
    """Run feedback function on all inputs.

    Args:
        flattened_inputs: Flattened inputs. Each entry is a tuple of (record_id, span_group, inputs).
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        error_messages: List of error messages to append to if any errors occur during this function.
    """
    for record_id, span_group, inputs in flattened_inputs:
        try:
            _call_feedback_function(
                feedback_name,
                feedback_function,
                inputs,
                record_id_to_record_root[record_id]["record_attributes"],
            )
        except Exception as e:
            error_messages.append(
                f"Error computing feedback '{feedback_name}': {str(e)} for span group {span_group}"
            )


def _call_feedback_function(
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    kwarg_inputs: Dict[str, Any],
    record_root_attributes: Dict[str, Any],
) -> None:
    """Call feedback function.

    Args:
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        kwarg_inputs: kwarg inputs to feedback function.
        record_root_attributes: Span attributes of record root.
    """
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
            res = feedback_function(**kwarg_inputs)
        except Exception as e:
            eval_root_span.set_attribute(SpanAttributes.EVAL_ROOT.ERROR, str(e))
            raise e
        metadata = {}
        if isinstance(res, tuple):
            if (
                len(res) != 2
                or not isinstance(res[0], float)
                or not isinstance(res[1], dict)
                or not all([isinstance(curr, str) for curr in res[1].keys()])
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


def _report_errors(error_messages: List[str]) -> None:
    """Report collected errors if any.

    Args:
        error_messages: List of error messages to report.
    """
    if error_messages:
        error_message = "Found the following errors:\n"
        for i, err in enumerate(error_messages, 1):
            error_message += f"{i}. {err}\n"
        raise ValueError(error_message)
