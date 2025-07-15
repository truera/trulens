from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import logging
import numbers
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from opentelemetry import trace
from opentelemetry.trace import INVALID_SPAN_ID
from opentelemetry.trace.span import Span
import pandas as pd
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.feedback.selector import ProcessedContentNode
from trulens.core.feedback.selector import Selector
from trulens.core.feedback.selector import Trace
from trulens.core.otel.instrument import OtelFeedbackComputationRecordingContext
from trulens.core.otel.instrument import get_func_name
from trulens.experimental.otel_tracing.core.session import TRULENS_SERVICE_NAME
from trulens.experimental.otel_tracing.core.span import (
    set_function_call_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_general_span_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_span_attribute_safely,
)
from trulens.otel.semconv.trace import BASE_SCOPE
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

_logger = logging.getLogger(__name__)


_EXPLANATION_KEYS = ["explanation", "explanations", "reason", "reasons"]


# If we could just have `opentelemetry.sdk.trace.ReadableSpan` it would be
# better, but this is all we need and it's easier to fill only this info
# from an event table row.
@dataclass
class MinimalSpanInfo:
    span_id: Optional[int]
    parent_span_id: Optional[int]
    attributes: Dict[str, Any]
    resource_attributes: Dict[str, Any]


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
    higher_is_better: bool,
    selector_function: Callable[[RecordGraphNode], List[Dict[str, Any]]],
) -> None:
    """
    Compute feedback for a record. This is a utility function that can compute
    feedback functions quite arbitrarily and so is quite powerful.

    Args:
        record_root: Record root of record to compute feedback for.
        feedback_name: Name of feedback.
        feedback_function: Function to compute feedback.
        higher_is_better: Whether higher values are better.
        selector_function:
            Function to select inputs for feedback computation. Given a record
            in graph form, it returns a list of inputs to the feedback
            function. Each entry in the list is a dictionary that represents
            the kwargs to the feedback function.
    """
    feedback_inputs = selector_function(record_root)
    record_root_attributes = record_root.current_span.attributes
    record_root_resource_attributes = (
        record_root.current_span.resource_attributes
    )
    for curr in feedback_inputs:
        curr = {k: FeedbackFunctionInput(value=v) for k, v in curr.items()}
        _call_feedback_function_with_record_root_info(
            feedback_name,
            feedback_function,
            higher_is_better,
            None,
            curr,
            record_root_attributes,
            record_root_resource_attributes,
        )


def compute_feedback_by_span_group(
    events: pd.DataFrame,
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    higher_is_better: bool,
    kwarg_to_selector: Dict[str, Selector],
    feedback_aggregator: Optional[Callable[[List[float]], float]] = None,
    raise_error_on_no_feedbacks_computed: bool = True,
) -> None:
    """
    Compute feedback based on span groups in events.

    Args:
        events: DataFrame containing trace events.
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        higher_is_better: Whether higher values are better.
        kwarg_to_selector: Mapping from function kwargs to span selectors
        feedback_aggregator: Aggregator function to combine feedback scores.
        raise_error_on_no_feedbacks_computed:
            Raise an error if no feedbacks were computed. Default is True.
    """
    kwarg_groups = _group_kwargs_by_selectors(kwarg_to_selector)
    unflattened_inputs = _collect_inputs_from_events(
        events, kwarg_groups, kwarg_to_selector
    )
    record_id_to_record_root = _map_record_id_to_record_roots(events)
    unflattened_inputs = _validate_unflattened_inputs(
        unflattened_inputs,
        kwarg_groups,
        list(record_id_to_record_root.keys()),
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
        higher_is_better,
        feedback_aggregator,
        record_id_to_record_root,
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
    span_id_to_child_events = defaultdict(list)
    for _, curr in events.iterrows():
        parent_span_id = curr["trace"]["parent_id"]
        span_id_to_child_events[parent_span_id].append(curr)
    record_roots = [
        curr
        for _, curr in events.iterrows()
        if curr["record_attributes"].get(SpanAttributes.SPAN_TYPE)
        == SpanAttributes.SpanType.RECORD_ROOT
    ]
    if _is_trace_level(kwarg_to_selector):
        sole_kwarg = kwarg_groups[0][0]
        ret = defaultdict(
            lambda: defaultdict(
                lambda: [{sole_kwarg: FeedbackFunctionInput(value=Trace())}]
            )
        )
        for record_root in record_roots:
            _dfs_collect_trace_level_inputs_from_events(
                sole_kwarg,
                kwarg_to_selector[sole_kwarg],
                span_id_to_child_events,
                record_root,
                None,
                ret,
            )
    else:
        ret = defaultdict(lambda: defaultdict(list))
        for kwarg_group in kwarg_groups:
            for record_root in record_roots:
                _dfs_collect_inputs_from_events(
                    kwarg_group,
                    kwarg_to_selector,
                    span_id_to_child_events,
                    record_root,
                    ret,
                )
    return ret


def _is_trace_level(kwarg_to_selector: Dict[str, Selector]) -> bool:
    """Check if any selectors are at a trace level.

    Args:
        kwarg_to_selector: Mapping from function kwargs to span selectors.

    Returns:
        True iff any selector is at a trace level. Will throw an error if there
        are multiple kwargs.
    """
    if any(curr.trace_level for curr in kwarg_to_selector.values()):
        if len(kwarg_to_selector) != 1:
            raise ValueError(
                "Cannot have multiple `Selectors` if any are trace level!"
            )
        return True
    return False


def _dfs_collect_trace_level_inputs_from_events(
    sole_kwarg,
    sole_selector,
    span_id_to_child_events: Dict[str, List[pd.Series]],
    curr_event: pd.Series,
    parent_processed_content_node: Optional[ProcessedContentNode],
    ret: Dict[
        Tuple[str, Optional[str]],
        Dict[Tuple[str], List[Dict[str, FeedbackFunctionInput]]],
    ],
) -> None:
    """DFS collect inputs from events based on a single trace level `Selector`.

    Args:
        sole_kwarg: Sole kwarg to the feedback function.
        sole_selector: Selector for sole kwarg to the feedback function.
        span_id_to_child_events: Mapping from span id to child events.
        curr_event: Current event to process.
        parent_processed_content_node:
            Parent `ProcessedContentNode` of the current event to process.
        ret:
            Mapping from (record_id, None) to kwarg group to inputs. This is
            what will be updated throughout the DFS.
    """
    # Convenience variables.
    record_attributes = curr_event["record_attributes"]
    record_id = record_attributes[SpanAttributes.RECORD_ID]
    span_id = curr_event["trace"]["span_id"]
    span_name = curr_event["record"]["name"]
    # Check if row satisfies selector conditions.
    curr_processed_content_node = None
    if sole_selector.matches_span(span_name, record_attributes):
        processed_content = sole_selector.process_span(
            span_id, record_attributes
        ).value
        if (
            processed_content is not None
            or not sole_selector.ignore_none_values
        ):
            curr_processed_content_node = ret[(record_id, None)][(sole_kwarg,)][
                0
            ][sole_kwarg].value.add_event(
                processed_content, curr_event, parent_processed_content_node
            )
    # Recurse on child events.
    for child_event in span_id_to_child_events[span_id]:
        _dfs_collect_trace_level_inputs_from_events(
            sole_kwarg,
            sole_selector,
            span_id_to_child_events,
            child_event,
            curr_processed_content_node or parent_processed_content_node,
            ret,
        )


def _dfs_collect_inputs_from_events(
    kwarg_group: Tuple[str],
    kwarg_to_selector: Dict[str, Selector],
    span_id_to_child_events: Dict[str, List[pd.Series]],
    curr_event: pd.Series,
    ret: Dict[
        Tuple[str, Optional[str]],
        Dict[Tuple[str], List[Dict[str, FeedbackFunctionInput]]],
    ],
) -> None:
    """DFS collect inputs from events.

    Args:
        kwarg_group:
            List of kwargs that describe the same spans in their selector.
        kwarg_to_selector: Mapping from function kwargs to span selectors.
        span_id_to_child_events: Mapping from span id to child events.
        curr_event: Current event to process.
        ret:
            Mapping from (record_id, span_group) to kwarg group to inputs. This
            is what will be updated throughout the DFS.
    """
    # Convenience variables.
    record_attributes = curr_event["record_attributes"]
    record_id = record_attributes[SpanAttributes.RECORD_ID]
    selector = kwarg_to_selector[kwarg_group[0]]
    span_id = curr_event["trace"]["span_id"]
    span_name = curr_event["record"]["name"]
    # Handle span groups.
    span_groups = record_attributes.get(SpanAttributes.SPAN_GROUPS, [None])
    if isinstance(span_groups, str):
        span_groups = [span_groups]
    elif span_groups is None:
        span_groups = [None]
    # Check if row satisfies selector conditions.
    matched = False
    if selector.matches_span(span_name, record_attributes):
        # Collect inputs for this kwarg group.
        ignore = False
        kwarg_group_inputs = {}
        for kwarg in kwarg_group:
            val = kwarg_to_selector[kwarg].process_span(
                span_id, record_attributes
            )
            kwarg_group_inputs[kwarg] = val
            if (
                val.value is None
                and kwarg_to_selector[kwarg].ignore_none_values
            ):
                ignore = True
                break
        if not ignore:
            # Place the inputs for this record id and every span group.
            for span_group in span_groups:
                ret[(record_id, span_group)][kwarg_group].append(
                    kwarg_group_inputs
                )
            matched = True
    # Recurse on child events if necessary.
    if not matched or not selector.match_only_if_no_ancestor_matched:
        for child_event in span_id_to_child_events[span_id]:
            _dfs_collect_inputs_from_events(
                kwarg_group,
                kwarg_to_selector,
                span_id_to_child_events,
                child_event,
                ret,
            )


def _map_record_id_to_record_roots(
    events: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Map record_id to record roots.

    Args:
        events: DataFrame containing trace events.

    Returns:
        Mapping from record_id to record root.
    """
    ret = {}
    for _, curr in events.iterrows():
        record_attributes = curr["record_attributes"]
        if (
            record_attributes.get(SpanAttributes.SPAN_TYPE)
            == SpanAttributes.SpanType.RECORD_ROOT
        ):
            record_id = record_attributes.get(SpanAttributes.RECORD_ID)
            if record_id in ret:
                _logger.warning(
                    f"Multiple record roots found for record_id={record_id}!"
                )
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
    if not events.empty and "record_attributes" in events.columns:
        attributes = events["record_attributes"]
    else:
        _logger.warning("Events is empty, returning flattened inputs.")
        return flattened_inputs
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
            SpanAttributes.EVAL_ROOT.METRIC_NAME
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
    higher_is_better: bool,
    feedback_aggregator: Optional[Callable[[List[float]], float]],
    record_id_to_record_root: Dict[str, pd.Series],
) -> int:
    """Run feedback function on all inputs.

    Args:
        flattened_inputs: Flattened inputs. Each entry is a tuple of (record_id, span_group, inputs).
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        higher_is_better: Whether higher values are better.
        feedback_aggregator: Aggregator function to combine feedback scores.
        record_id_to_record_root: Mapping from record_id to record root.

    Returns:
        Number of feedbacks computed.
    """
    ret = 0
    for record_id, span_group, inputs in flattened_inputs:
        try:
            _call_feedback_function_with_record_root_info(
                feedback_name,
                feedback_function,
                higher_is_better,
                feedback_aggregator,
                inputs,
                record_id_to_record_root[record_id]["record_attributes"],
                record_id_to_record_root[record_id]["resource_attributes"],
                span_group,
            )
            ret += 1
        except Exception as e:
            _logger.warning(
                f"feedback_name={feedback_name}, record={record_id}, span_group={span_group} had an error during computation:\n{str(e)}"
            )
    return ret


def _call_feedback_function_with_record_root_info(
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    higher_is_better: bool,
    feedback_aggregator: Optional[Callable[[List[float]], float]],
    kwarg_inputs: Dict[str, FeedbackFunctionInput],
    record_root_attributes: Dict[str, Any],
    record_root_resource_attributes: Dict[str, Any],
    span_group: Optional[str] = None,
) -> None:
    """Call feedback function.

    Args:
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        higher_is_better: Whether higher values are better.
        feedback_aggregator: Aggregator function to combine feedback scores.
        kwarg_inputs: kwarg inputs to feedback function.
        record_root_attributes: Span attributes of record root.
        record_root_resource_attributes: Resource attributes of record root.
        span_group: Span group of the invocation.
    """
    app_name, app_version, app_id, run_name = _get_app_and_run_info(
        record_root_attributes, record_root_resource_attributes
    )
    # TODO(otel, dhuang): need to use these when getting the object entity!
    # database_name = record_root_attributes[
    #    f"snow.{BASE_SCOPE}.database.name"
    # ]
    # schema_name = record_root_attributes[
    #    f"snow.{BASE_SCOPE}.schema.name"
    # ]
    input_id = record_root_attributes[SpanAttributes.INPUT_ID]
    target_record_id = record_root_attributes[SpanAttributes.RECORD_ID]
    _call_feedback_function(
        feedback_name,
        feedback_function,
        higher_is_better,
        feedback_aggregator,
        kwarg_inputs,
        app_name,
        app_version,
        app_id,
        run_name,
        input_id,
        target_record_id,
        span_group,
    )


def _call_feedback_function(
    feedback_name: str,
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    higher_is_better: bool,
    feedback_aggregator: Optional[Callable[[List[float]], float]],
    kwarg_inputs: Dict[str, FeedbackFunctionInput],
    app_name: str,
    app_version: str,
    app_id: str,
    run_name: str,
    input_id: str,
    target_record_id: str,
    span_group: Optional[str] = None,
) -> float:
    """Call feedback function.

    Args:
        feedback_name: Name of the feedback function.
        feedback_function: Function to compute feedback.
        higher_is_better: Whether higher values are better.
        feedback_aggregator: Aggregator function to combine feedback scores.
        kwarg_inputs: kwarg inputs to feedback function.
        app_name: Name of the app.
        app_version: Version of the app.
        app_id: ID of the app.
        run_name: Name of the run.
        input_id: ID of the input.
        target_record_id: ID of the target record.
        span_group: Span group of the invocation.

    Returns:
        The score returned by the feedback function.
    """
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
            if span_group is not None:
                eval_root_span.set_attribute(
                    f"{SpanAttributes.EVAL_ROOT.SPAN_GROUP}",
                    span_group,
                )
            eval_root_span.set_attribute(
                SpanAttributes.EVAL_ROOT.HIGHER_IS_BETTER, higher_is_better
            )
            expanded_kwargs_inputs = [{}]
            aggregate = False
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
                if isinstance(v.value, list) and not v.collect_list:
                    aggregate = True
                    new_expanded_kwargs_inputs = []
                    for curr in expanded_kwargs_inputs:
                        for val in v.value:
                            curr[k] = val
                            new_expanded_kwargs_inputs.append(curr.copy())
                    expanded_kwargs_inputs = new_expanded_kwargs_inputs
                else:
                    for curr in expanded_kwargs_inputs:
                        curr[k] = v.value
            res = []
            for i, curr in enumerate(expanded_kwargs_inputs):
                res.append(
                    _call_feedback_function_under_eval_span(
                        feedback_function,
                        curr,
                        eval_root_span,
                        is_only_child=len(expanded_kwargs_inputs) == 1,
                        eval_child_idx=i,
                    )
                )
            if aggregate:
                if feedback_aggregator is not None:
                    res = feedback_aggregator(res)
                else:
                    res = sum(res) / len(res) if res else 0.0
            else:
                res = res[0]
            eval_root_span.set_attribute(SpanAttributes.EVAL_ROOT.SCORE, res)
            return res
        except Exception as e:
            eval_root_span.set_attribute(SpanAttributes.EVAL_ROOT.ERROR, str(e))
            raise e


def _call_feedback_function_under_eval_span(
    feedback_function: Callable[
        [Any], Union[float, Tuple[float, Dict[str, Any]]]
    ],
    kwargs: Dict[str, Any],
    eval_root_span: Span,
    is_only_child: bool,
    eval_child_idx: int,
) -> float:
    """
    Call a feedback function with the provided kwargs and return the score.

    Args:
        feedback_function: The feedback function to call.
        kwargs: The keyword arguments to pass to the feedback function.
        eval_root_span: The root span for the evaluation.
        is_only_child:
            Whether this is the only child eval span of the evaluation root.
        eval_child_idx: Index of this eval child span in the evaluation root.

    Returns:
        The score returned by the feedback function.
    """
    with (
        trace.get_tracer_provider()
        .get_tracer(TRULENS_SERVICE_NAME)
        .start_as_current_span(f"eval-{eval_child_idx}")
    ) as eval_span:
        set_general_span_attributes(eval_span, SpanAttributes.SpanType.EVAL)
        res = None
        exc = None
        try:
            res = feedback_function(**kwargs)
            metadata = {}
            if isinstance(res, tuple):
                # If the result is a tuple, it must be (score, metadata) where
                # the metadata has string keys.
                if (
                    len(res) != 2
                    or not isinstance(res[0], numbers.Number)
                    or not isinstance(res[1], dict)
                    or not all([
                        isinstance(curr, str) for curr in res[1].keys()
                    ])
                ):
                    raise ValueError(
                        "Feedback functions must be of type `Callable[Any, Union[float, Tuple[float, Dict[str, Any]]]]`!"
                    )
                res, metadata = res[0], res[1]
            res = float(res)
            eval_span.set_attribute(SpanAttributes.EVAL.SCORE, res)
            _set_metadata_attributes(eval_span, metadata)
            if is_only_child:
                _set_metadata_attributes(eval_root_span, metadata)
            return res
        except Exception as e:
            exc = e
            eval_span.set_attribute(SpanAttributes.EVAL.ERROR, str(e))
            raise e
        finally:
            set_function_call_attributes(
                eval_span, res, get_func_name(feedback_function), exc, kwargs
            )


def _set_metadata_attributes(span: Span, metadata: Dict[str, Any]) -> None:
    """Set metadata attributes on a span.

    Args:
        span: Span to set attributes on.
        metadata: Metadata to extract attributes from.
    """
    for k, v in metadata.items():
        set_span_attribute_safely(
            span, f"{SpanAttributes.EVAL.METADATA}.{k}", v
        )
        if k in _EXPLANATION_KEYS:
            set_span_attribute_safely(span, SpanAttributes.EVAL.EXPLANATION, v)


def _get_app_and_run_info(
    attributes: Dict[str, Any], resource_attributes: Dict[str, Any]
) -> Tuple[str, str, str, str]:
    """Get app info from attributes.

    Args:
        attributes: Span attributes of record root.
        resource_attributes: Resource attributes of record root.

    Returns:
        Tuple of: app name, app version, and app id, run name.
    """

    def get_value(keys: List[str]) -> Optional[str]:
        for key in keys:
            for attr in [resource_attributes, attributes]:
                if key in attr:
                    return attr[key]
        return None

    app_name = get_value([
        f"snow.{BASE_SCOPE}.object.name",
        ResourceAttributes.APP_NAME,
    ])
    app_version = get_value([
        f"snow.{BASE_SCOPE}.object.version.name",
        ResourceAttributes.APP_VERSION,
    ])
    app_id = get_value([ResourceAttributes.APP_ID])
    run_name = get_value([
        f"snow.{BASE_SCOPE}.run.name",
        SpanAttributes.RUN_NAME,
    ])
    return app_name, app_version, app_id, run_name
