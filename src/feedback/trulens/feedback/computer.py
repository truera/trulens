from typing import Any, Callable, Dict, List, Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import INVALID_SPAN_ID
from trulens.core.otel.instrument import OTELFeedbackComputationRecordingContext
from trulens.otel.semconv.trace import SpanAttributes


class RecordGraphNode:
    """Graph form of a record (i.e. a list of spans)."""

    current_span: ReadableSpan
    parent_span: Optional[ReadableSpan]
    children_spans: List["RecordGraphNode"]

    def __init__(self, span: ReadableSpan):
        self.current_span = span
        self.parent_span = None
        self.children_spans = []

    @staticmethod
    def build_graph(spans: List[ReadableSpan]) -> "RecordGraphNode":
        nodes = [RecordGraphNode(curr) for curr in spans]
        span_id_idx = {curr.context.span_id: i for i, curr in enumerate(spans)}
        root = None
        for i, span in enumerate(spans):
            if span.parent is None:
                if root is not None:
                    raise ValueError("Multiple roots found!")
                root = nodes[i]
                continue
            parent_span_id = span.parent.span_id
            if parent_span_id == INVALID_SPAN_ID:
                raise ValueError()
            parent_idx = span_id_idx[parent_span_id]
            nodes[parent_idx].children_spans.append(nodes[i])
            nodes[i].parent_span = nodes[parent_idx]
        return root


def _compute(
    record_root: RecordGraphNode,
    feedback_function: Callable[[Any], float],
    selector_function: Callable[[RecordGraphNode], List[Dict[str, Any]]],
) -> None:
    """
    Compute feedback for a record.

    Args:
        record: Record to compute feedback for.
        feedback_function: Function to compute feedback.
        selector_function: Function to select inputs for feedback computation.
    """
    feedback_inputs = selector_function(record_root)
    record_root_attributes = record_root.current_span.attributes
    app_name = record_root_attributes[SpanAttributes.APP_NAME]
    app_version = record_root_attributes[SpanAttributes.APP_VERSION]
    run_name = record_root_attributes[SpanAttributes.RUN_NAME]
    input_id = record_root_attributes[SpanAttributes.INPUT_ID]
    target_record_id = record_root_attributes[SpanAttributes.RECORD_ID]
    for curr in feedback_inputs:
        context_manager = OTELFeedbackComputationRecordingContext(
            app_name,
            app_version,
            run_name,
            input_id,
            target_record_id,
        )
        with context_manager:
            feedback_function(**curr)
