from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from opentelemetry.trace import INVALID_SPAN_ID
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
