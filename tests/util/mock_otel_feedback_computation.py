from typing import Any, Dict, List, Tuple

from trulens.core.otel.instrument import instrument
from trulens.feedback.computer import RecordGraphNode
from trulens.otel.semconv.trace import SpanAttributes


def feedback_function(**kwargs):
    child_feedback_function("Is Kojikun the best baby?")
    child_feedback_function("Is Nolan the best baby?")
    return 0.99, {"best_baby": "Kojikun/Nolan"}


@instrument(
    span_type=SpanAttributes.SpanType.EVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.EVAL.CRITERIA: kwargs["criteria"],
        SpanAttributes.EVAL.EXPLANATION: ret[1],
        SpanAttributes.EVAL.SCORE: ret[0],
    },
)
def child_feedback_function(criteria: str) -> Tuple[float, str]:
    return {
        "Is Kojikun the best baby?": (
            1,
            "Kojikun is extraordinarily cute and fun and is the best baby!",
        ),
        "Is Nolan the best baby?": (
            0.98,
            "Nolan is just another name for Kojikun so he is the best baby!",
        ),
    }[criteria]


def all_retrieval_span_attributes(
    node: RecordGraphNode,
) -> List[Dict[str, Any]]:
    ret = []
    if (
        node.current_span.attributes.get(SpanAttributes.SPAN_TYPE)
        == SpanAttributes.SpanType.RETRIEVAL
    ):
        ret = [node.current_span.attributes]
    # Recurse on children.
    for child in node.children_spans:
        ret.extend(all_retrieval_span_attributes(child))
    return ret
