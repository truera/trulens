from __future__ import annotations

from collections import defaultdict
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from opentelemetry.trace import INVALID_SPAN_ID
import pandas as pd
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import OtelFeedbackComputationRecordingContext
from trulens.experimental.otel_tracing.core.span import (
    set_span_attribute_safely,
)
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.otel.semconv.trace import SpanAttributes

_logger = logging.getLogger(__name__)


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
        curr = {k: FeedbackFunctionInput(value=v) for k, v in curr.items()}
        _call_feedback_function(
            feedback_name, feedback_function, curr, record_root_attributes
        )


def compute_feedback_by_span_group(
    events: pd.DataFrame,
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    kwarg_to_selector: Dict[str, Selector],
    raise_error_on_no_feedbacks_computed: bool = True,
) -> None:
    """
    Compute feedback based on span groups in events.

    Args:
        events: DataFrame containing trace events.
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        kwarg_to_selector: Mapping from function kwargs to span selectors
        raise_error_on_no_feedbacks_computed:
            Raise an error if no feedbacks were computed. Default is True.
    """
    kwarg_groups = _group_kwargs_by_selectors(kwarg_to_selector)
    unflattened_inputs = _collect_inputs_from_events(
        events, kwarg_groups, kwarg_to_selector
    )
    record_id_to_record_roots = _map_record_id_to_record_roots(
        events["record_attributes"]
    )
    unflattened_inputs = _validate_unflattened_inputs(
        unflattened_inputs,
        kwarg_groups,
        list(record_id_to_record_roots.keys()),
        feedback_name,
    )
    flattened_inputs = _flatten_inputs(unflattened_inputs)
    flattened_inputs = _remove_already_computed_feedbacks(
        events, feedback_name, flattened_inputs
    )
    num_feedbacks_computed = _run_feedback_on_inputs(
        flattened_inputs,
        feedback_name,
        feedback_function,
        record_id_to_record_roots,
    )
    if raise_error_on_no_feedbacks_computed and num_feedbacks_computed == 0:
        raise ValueError("No feedbacks were computed!")


def _group_kwargs_by_selectors(
    kwarg_to_selector: Dict[str, Selector],
) -> List[Tuple[str]]:
    """Group kwargs by by whether their selectors describe the same spans.

    Args:
        kwarg_to_selector: kwarg to selector mapping.

    Returns:
        List of tuples of kwargs. Each tuple contains kwargs that describe the
        same spans in their selector.
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
) -> Dict[
    Tuple[str, Optional[str]],
    Dict[Tuple[str], List[Dict[str, FeedbackFunctionInput]]],
]:
    """Collect inputs from events based on selectors.

    Args:
        events: DataFrame containing trace events.
        kwarg_groups:
            List of list of kwargs. Each sublist contains kwargs that describe
            the same spans in their selector.
        kwarg_to_selector: Mapping from function kwargs to span selectors.

    Returns:
        Mapping from (record_id, span_group) to kwarg group to inputs.
    """
    ret = defaultdict(lambda: defaultdict(list))

    for _, curr in events.iterrows():
        record_attributes = curr["record_attributes"]
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
            if kwarg_to_selector[kwarg_group[0]].matches_span(
                record_attributes
            ):
                # Collect inputs for this kwarg group.
                kwarg_group_inputs = {
                    kwarg: kwarg_to_selector[kwarg].process_span(
                        curr["trace"]["span_id"], record_attributes
                    )
                    for kwarg in kwarg_group
                }
                # Place the inputs for this record id and every span group.
                for span_group in span_groups:
                    ret[(record_id, span_group)][kwarg_group].append(
                        kwarg_group_inputs
                    )
    return ret


def _map_record_id_to_record_roots(
    record_attributes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map record_id to record roots.

    Args:
        record_attributes: List containing record attributes of events.

    Returns:
        Mapping from record_id to record roots.
    """
    ret = {}
    for curr in record_attributes:
        if (
            curr.get(SpanAttributes.SPAN_TYPE, None)
            == SpanAttributes.SpanType.RECORD_ROOT
        ):
            record_id = curr[SpanAttributes.RECORD_ID]
            ret[record_id] = curr
    return ret


def _validate_unflattened_inputs(
    unflattened_inputs: Dict[
        Tuple[str, Optional[str]],
        Dict[Tuple[str], List[Dict[str, FeedbackFunctionInput]]],
    ],
    kwarg_groups: List[Tuple[str]],
    record_ids_with_record_roots: List[str],
    feedback_name: str,
) -> Dict[
    Tuple[str, Optional[str]],
    Dict[Tuple[str], List[Dict[str, FeedbackFunctionInput]]],
]:
    """Validate collected inputs and remove invalid entries.

    Args:
        unflattened_inputs:
            Mapping from (record_id, span_group) to kwarg group to inputs.
        kwarg_groups:
            List of list of kwargs. Each sublist contains kwargs that describe
            the same spans in their selector.
        record_ids_with_record_roots:
            List of record ids that have record roots.
        feedback_name: Name of the feedback function.

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
                keys_to_remove.append((record_id, span_group))
                continue

            # Check for ambiguous inputs (multiple possible values).
            group_sizes = sorted([
                len(inputs) for inputs in kwarg_group_to_inputs.values()
            ])
            if (
                group_sizes[-2] > 1
            ):  # If second largest group has multiple items then unclear how to combine.
                _logger.warning(
                    f"feedback_name={feedback_name}, record={record_id}, span_group={span_group} has ambiguous inputs!"
                )
                keys_to_remove.append((record_id, span_group))

    for record_id, span_group in unflattened_inputs:
        if record_id not in record_ids_with_record_roots:
            _logger.warning(f"record_id={record_id} has no known record root!")
            keys_to_remove.append((record_id, span_group))

    # Remove invalid entries.
    for key in keys_to_remove:
        if key in unflattened_inputs:
            del unflattened_inputs[key]

    return unflattened_inputs


def _flatten_inputs(
    unflattened_inputs: Dict[
        Tuple[str, Optional[str]],
        Dict[Tuple[str], List[Dict[str, FeedbackFunctionInput]]],
    ],
) -> List[Tuple[str, Optional[str], Dict[str, FeedbackFunctionInput]]]:
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


def _remove_already_computed_feedbacks(
    events: pd.DataFrame,
    feedback_name: str,
    flattened_inputs: List[
        Tuple[str, Optional[str], Dict[str, FeedbackFunctionInput]]
    ],
) -> List[Tuple[str, Optional[str], Dict[str, FeedbackFunctionInput]]]:
    """Remove inputs that have already been computed.

    Args:
        events: DataFrame containing trace events.
        feedback_name: Name of the feedback function.
        flattened_inputs:
            Flattened inputs to remove inputs that have already been computed
            from.

    Returns:
        List of inputs that have not already been computed.
    """
    attributes = events["record_attributes"]
    eval_root_attributes = attributes[
        attributes.apply(
            lambda curr: curr.get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.EVAL_ROOT
        )
    ]
    record_id_to_eval_root_attributes = eval_root_attributes.groupby(
        by=eval_root_attributes.apply(
            lambda curr: curr.get(SpanAttributes.RECORD_ID)
        )
    )
    ret = []
    for record_id, span_group, inputs in flattened_inputs:
        curr_eval_root_attributes = []
        if record_id in record_id_to_eval_root_attributes.groups:
            curr_eval_root_attributes = (
                record_id_to_eval_root_attributes.get_group(record_id)
            )
        if not _feedback_already_computed(
            span_group, inputs, feedback_name, curr_eval_root_attributes
        ):
            ret.append((record_id, span_group, inputs))
    return ret


def _feedback_already_computed(
    span_group: Optional[str],
    kwarg_inputs: Dict[str, FeedbackFunctionInput],
    feedback_name: str,
    eval_root_attributes: Sequence[Dict[str, Any]],
) -> bool:
    """Check if feedback has already been computed.

    Args:
        span_group: Span group of the invocation.
        kwarg_inputs: kwarg inputs to feedback function.
        feedback_name: Name of the feedback function.
        eval_root_attributes: List of eval root spans attributes.

    Returns:
        True iff feedback has already been computed.
    """
    for curr in eval_root_attributes:
        curr_span_group = curr.get(SpanAttributes.EVAL_ROOT.SPAN_GROUP)
        if isinstance(curr_span_group, list):
            valid = span_group in curr_span_group
        else:
            valid = span_group == curr_span_group
        valid = valid and feedback_name == curr.get(
            SpanAttributes.EVAL.METRIC_NAME
        )
        for k, v in kwarg_inputs.items():
            if not valid:
                break
            if v.span_id != curr.get(
                f"{SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID}.{k}"
            ):
                valid = False
            if v.span_attribute != curr.get(
                f"{SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE}.{k}"
            ):
                valid = False
        if valid:
            return True
    return False


def _run_feedback_on_inputs(
    flattened_inputs: List[
        Tuple[str, Optional[str], Dict[str, FeedbackFunctionInput]]
    ],
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    record_id_to_record_root: Dict[str, Dict[str, Any]],
) -> int:
    """Run feedback function on all inputs.

    Args:
        flattened_inputs: Flattened inputs. Each entry is a tuple of (record_id, span_group, inputs).
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        record_id_to_record_root: Mapping from record_id to record root.

    Returns:
        Number of feedbacks computed.
    """
    ret = 0
    for record_id, span_group, inputs in flattened_inputs:
        try:
            _call_feedback_function(
                feedback_name,
                feedback_function,
                inputs,
                record_id_to_record_root[record_id],
                span_group,
            )
            ret += 1
        except Exception as e:
            _logger.warning(
                f"feedback_name={feedback_name}, record={record_id}, span_group={span_group} had an error during computation:\n{str(e)}"
            )
    return ret


def _call_feedback_function(
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    kwarg_inputs: Dict[str, FeedbackFunctionInput],
    record_root_attributes: Dict[str, Any],
    span_group: Optional[str] = None,
) -> None:
    """Call feedback function.

    Args:
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        kwarg_inputs: kwarg inputs to feedback function.
        record_root_attributes: Span attributes of record root.
        span_group: Span group of the invocation.
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
    app_id = record_root_attributes[SpanAttributes.APP_ID]
    input_id = record_root_attributes[SpanAttributes.INPUT_ID]
    target_record_id = record_root_attributes[SpanAttributes.RECORD_ID]
    context_manager = OtelFeedbackComputationRecordingContext(
        app_name=app_name,
        app_version=app_version,
        app_id=app_id,
        run_name=run_name,
        input_id=input_id,
        target_record_id=target_record_id,
        feedback_name=feedback_name,
    )
    with context_manager as eval_root_span:
        try:
            for k, v in kwarg_inputs.items():
                if v.span_id is not None:
                    eval_root_span.set_attribute(
                        f"{SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID}.{k}",
                        v.span_id,
                    )
                if v.span_attribute is not None:
                    eval_root_span.set_attribute(
                        f"{SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE}.{k}",
                        v.span_attribute,
                    )
                if span_group is not None:
                    eval_root_span.set_attribute(
                        f"{SpanAttributes.EVAL_ROOT.SPAN_GROUP}",
                        span_group,
                    )
            res = feedback_function(**{
                k: v.value for k, v in kwarg_inputs.items()
            })
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
        eval_root_span.set_attribute(SpanAttributes.EVAL_ROOT.SCORE, res)
        for k, v in metadata.items():
            set_span_attribute_safely(
                eval_root_span,
                f"{SpanAttributes.EVAL_ROOT.METADATA}.{k}",
                v,
            )
