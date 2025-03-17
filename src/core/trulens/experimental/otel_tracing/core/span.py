"""
This file contains utility functions specific to certain span types.
"""

from inspect import signature
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from opentelemetry.baggage import get_baggage
from opentelemetry.context import Context
from opentelemetry.trace.span import Span
from opentelemetry.util.types import AttributeValue
from trulens.core.utils import signature as signature_utils
from trulens.otel.semconv.trace import SpanAttributes

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


def validate_selector_name(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to validate the selector name in the attributes.

    It does the following:
    1. Ensure that the selector name is a string.
    2. Ensure that the selector name is keyed with either the trulens/non-trulens key variations.
    3. Ensure that the selector name is not set in both the trulens and non-trulens key variations.
    """

    result = attributes.copy()

    if (
        SpanAttributes.SELECTOR_NAME_KEY in result
        and SpanAttributes.SELECTOR_NAME in result
    ):
        raise ValueError(
            f"Both {SpanAttributes.SELECTOR_NAME_KEY} and {SpanAttributes.SELECTOR_NAME} cannot be set."
        )

    if SpanAttributes.SELECTOR_NAME in result:
        # Transfer the trulens-namespaced key to the non-trulens-namespaced key.
        result[SpanAttributes.SELECTOR_NAME_KEY] = result[
            SpanAttributes.SELECTOR_NAME
        ]
        del result[SpanAttributes.SELECTOR_NAME]

    if SpanAttributes.SELECTOR_NAME_KEY in result:
        selector_name = result[SpanAttributes.SELECTOR_NAME_KEY]
        if not isinstance(selector_name, str):
            raise ValueError(
                f"Selector name must be a string, not {type(selector_name)}"
            )

    return result


def _convert_to_valid_span_attribute_type(val: Any) -> AttributeValue:
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        for curr_type in [bool, int, float, str]:
            if all([isinstance(curr, curr_type) for curr in val]):
                return val
        return [str(curr) for curr in val]
    return str(val)


def set_span_attribute_safely(
    span: Span,
    key: str,
    value: Any,
) -> None:
    if value is not None:
        span.set_attribute(key, _convert_to_valid_span_attribute_type(value))


def set_string_span_attribute_from_baggage(
    span: Span,
    key: str,
    context: Optional[Context] = None,
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

    return validate_selector_name(attributes)
    # TODO: validate Span type attributes.


def set_general_span_attributes(
    span: Span,
    /,
    span_type: SpanAttributes.SpanType,
    context: Optional[Context] = None,
) -> None:
    span.set_attribute(SpanAttributes.SPAN_TYPE, span_type)

    set_string_span_attribute_from_baggage(
        span, SpanAttributes.APP_NAME, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.APP_VERSION, context
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


def set_function_call_attributes(
    span: Span,
    ret: Any,
    func_exception: Optional[Exception],
    all_kwargs: Dict[str, Any],
) -> None:
    set_span_attribute_safely(span, SpanAttributes.CALL.RETURN, ret)
    set_span_attribute_safely(span, SpanAttributes.CALL.ERROR, func_exception)
    for k, v in all_kwargs.items():
        set_span_attribute_safely(span, f"{SpanAttributes.CALL.KWARGS}.{k}", v)


def set_user_defined_attributes(
    span: Span,
    *,
    span_type: SpanAttributes.SpanType,
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
            if model not in [None, cost_attributes[SpanAttributes.COST.MODEL]]:
                cost_attributes[SpanAttributes.COST.MODEL] = "mixed"
            for k, v in cost_attributes.items():
                span.set_attribute(k, v)


"""
RECORD_ROOT SPAN
"""


def get_main_input(func: Callable, args: tuple, kwargs: dict) -> str:
    sig = signature(func)
    bindings = signature(func).bind(*args, **kwargs)
    return signature_utils.main_input(func, sig, bindings)


def set_record_root_span_attributes(
    span: Span,
    /,
    func: Callable,
    args: tuple,
    kwargs: dict,
    ret: Any,
    exception: Optional[Exception],
) -> None:
    set_span_attribute_safely(
        span,
        SpanAttributes.RECORD_ROOT.INPUT,
        get_main_input(func, args, kwargs),
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
            signature_utils.main_output(func, ret),
        )
