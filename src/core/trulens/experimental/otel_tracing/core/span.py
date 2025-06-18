"""
This file contains utility functions specific to certain span types.
"""

from __future__ import annotations

from inspect import signature
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from opentelemetry.baggage import get_baggage
from opentelemetry.context import Context
from opentelemetry.trace.span import Span
from opentelemetry.util.types import AttributeValue
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

if TYPE_CHECKING:
    from trulens.core.app import App

logger = logging.getLogger(__name__)

"""
Type definitions
"""
Attributes = Optional[
    Union[
        Dict[str, Any],
        Callable[
            [
                Optional[Any],
                Optional[Exception],
                List[Any],
                Optional[Dict[str, Any]],
            ],
            Dict[str, Any],
        ],
    ]
]

"""
General utilites for all spans
"""


def _stringify_span_attribute(o: Any) -> Tuple[bool, str]:
    """Converts an object to a string.

    Args:
        o: object to convert to a string

    Returns:
        A tuple containing:
        - A boolean indicating whether the object was jsonified (as opposed to
          simple stringification)
        - The string representation of the object.
    """
    try:
        return True, json.dumps(o)
    except Exception:
        pass
    return False, str(o)


def _convert_to_valid_span_attribute_type(val: Any) -> AttributeValue:
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        for curr_type in [bool, int, float, str]:
            if all([isinstance(curr, curr_type) for curr in val]):
                return val
        jsonifiable, j = _stringify_span_attribute(val)
        if jsonifiable:
            return j
        return [_stringify_span_attribute(curr)[1] for curr in val]
    return _stringify_span_attribute(val)[1]


def set_span_attribute_safely(
    span: Span,
    key: str,
    value: Any,
) -> None:
    if value is not None:
        span.set_attribute(key, _convert_to_valid_span_attribute_type(value))


def set_string_span_attribute_from_baggage(
    span: Span, key: str, context: Optional[Context] = None
) -> None:
    value = get_baggage(key, context)
    if value is not None:
        span.set_attribute(key, str(value))


def validate_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to validate span attributes based on the span type.
    """
    if not isinstance(attributes, dict) or any([
        not isinstance(key, str) for key in attributes.keys()
    ]):
        raise ValueError("Attributes must be a dictionary with string keys.")

    if SpanAttributes.SPAN_TYPE in attributes:
        raise ValueError("Span type should not be set in attributes.")

    # TODO: validate Span type attributes.
    return attributes


def set_general_span_attributes(
    span: Span,
    /,
    span_type: SpanAttributes.SpanType,
    context: Optional[Context] = None,
) -> None:
    span.set_attribute(SpanAttributes.SPAN_TYPE, span_type)

    set_string_span_attribute_from_baggage(
        span, ResourceAttributes.APP_NAME, context
    )
    set_string_span_attribute_from_baggage(
        span, ResourceAttributes.APP_VERSION, context
    )
    set_string_span_attribute_from_baggage(
        span, ResourceAttributes.APP_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.RECORD_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.EVAL.TARGET_RECORD_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.EVAL.EVAL_ROOT_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.EVAL.METRIC_NAME, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.RUN_NAME, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.INPUT_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.INPUT_RECORDS_COUNT, context
    )


def set_function_call_attributes(
    span: Span,
    ret: Any,
    func_name: str,
    func_exception: Optional[Exception],
    all_kwargs: Dict[str, Any],
) -> None:
    set_span_attribute_safely(span, SpanAttributes.CALL.RETURN, ret)
    set_span_attribute_safely(span, SpanAttributes.CALL.FUNCTION, func_name)
    set_span_attribute_safely(span, SpanAttributes.CALL.ERROR, func_exception)
    for k, v in all_kwargs.items():
        set_span_attribute_safely(span, f"{SpanAttributes.CALL.KWARGS}.{k}", v)


def set_user_defined_attributes(
    span: Span,
    *,
    attributes: Dict[str, Any],
) -> None:
    final_attributes = validate_attributes(attributes)

    cost_attributes = {}
    for key, value in final_attributes.items():
        if key.startswith(f"{SpanAttributes.COST.base}."):
            cost_attributes[key] = value
        else:
            span.set_attribute(key, value)
    if cost_attributes:
        attributes_so_far = dict(getattr(span, "attributes", {}))
        if attributes_so_far:
            for cost_field in [
                SpanAttributes.COST.COST,
                SpanAttributes.COST.NUM_TOKENS,
                SpanAttributes.COST.NUM_PROMPT_TOKENS,
                SpanAttributes.COST.NUM_COMPLETION_TOKENS,
            ]:
                cost_attributes[cost_field] = cost_attributes.get(
                    cost_field, 0
                ) + attributes_so_far.get(cost_field, 0)
            currency = attributes_so_far.get(SpanAttributes.COST.CURRENCY, None)
            model = attributes_so_far.get(SpanAttributes.COST.MODEL, None)
            if currency not in [
                None,
                cost_attributes[SpanAttributes.COST.CURRENCY],
            ]:
                cost_attributes[SpanAttributes.COST.CURRENCY] = "mixed"
            if model not in [
                None,
                cost_attributes.get(SpanAttributes.COST.MODEL),
            ]:
                cost_attributes[SpanAttributes.COST.MODEL] = "mixed"
            for k, v in cost_attributes.items():
                span.set_attribute(k, v)


"""
RECORD_ROOT SPAN
"""


def set_record_root_span_attributes(
    span: Span,
    /,
    func: Callable,
    args: tuple,
    kwargs: dict,
    ret: Any,
    exception: Optional[Exception],
) -> None:
    tru_app: App = get_baggage("__trulens_app__")
    sig = signature(func)
    set_span_attribute_safely(
        span,
        SpanAttributes.RECORD_ROOT.INPUT,
        tru_app.main_input(func, sig, sig.bind_partial(*args, **kwargs)),
    )
    ground_truth_output = get_baggage(
        SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT
    )
    if ground_truth_output:
        set_span_attribute_safely(
            span,
            SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
            ground_truth_output,
        )
    if exception:
        set_span_attribute_safely(
            span, SpanAttributes.RECORD_ROOT.ERROR, str(exception)
        )
    if ret is not None:
        set_span_attribute_safely(
            span,
            SpanAttributes.RECORD_ROOT.OUTPUT,
            tru_app.main_output(
                func, sig, sig.bind_partial(*args, **kwargs), ret
            ),
        )
