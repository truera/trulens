"""
This file contains utility functions specific to certain span types.
"""

from inspect import signature
import logging
from typing import Any, Callable, Dict, Optional, Union

from opentelemetry.baggage import get_baggage
from opentelemetry.trace.span import Span
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
            [Optional[Any], Optional[Exception], Any, Any], Dict[str, Any]
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
    span: Span, /, span_type: SpanAttributes.SpanType
) -> Span:
    span.set_attribute("kind", "SPAN_KIND_TRULENS")
    span.set_attribute(SpanAttributes.SPAN_TYPE, span_type)
    span.set_attribute(
        SpanAttributes.APP_ID, str(get_baggage(SpanAttributes.APP_ID))
    )
    span.set_attribute(
        SpanAttributes.RECORD_ID, str(get_baggage(SpanAttributes.RECORD_ID))
    )

    return span


def set_user_defined_attributes(
    span: Span,
    *,
    span_type: SpanAttributes.SpanType,
    args: tuple,
    kwargs: dict,
    ret,
    func_exception: Optional[Exception],
    attributes: Attributes,
) -> None:
    attributes_to_add = {}

    # Set the user-provided attributes.
    if attributes:
        if callable(attributes):
            attributes_to_add = attributes(ret, func_exception, *args, **kwargs)
        else:
            attributes_to_add = attributes

    logger.info(f"Attributes to add: {attributes_to_add}")

    final_attributes = validate_attributes(attributes_to_add)

    prefix = f"trulens.{span_type.value}."

    for key, value in final_attributes.items():
        span.set_attribute(prefix + key, value)

        if (
            key != SpanAttributes.SELECTOR_NAME_KEY
            and SpanAttributes.SELECTOR_NAME_KEY in final_attributes
        ):
            span.set_attribute(
                f"trulens.{final_attributes[SpanAttributes.SELECTOR_NAME_KEY]}.{key}",
                value,
            )


"""
MAIN SPAN
"""


def get_main_input(func: Callable, args: tuple, kwargs: dict) -> str:
    sig = signature(func)
    bindings = signature(func).bind(*args, **kwargs)
    return signature_utils.main_input(func, sig, bindings)


def set_main_span_attributes(
    span: Span,
    /,
    func: Callable,
    args: tuple,
    kwargs: dict,
    ret: Any,
    exception: Optional[Exception],
) -> None:
    span.set_attribute(
        SpanAttributes.MAIN.MAIN_INPUT, get_main_input(func, args, kwargs)
    )

    if exception:
        span.set_attribute(SpanAttributes.MAIN.MAIN_ERROR, str(exception))

    if ret is not None:
        span.set_attribute(
            SpanAttributes.MAIN.MAIN_OUTPUT,
            signature_utils.main_output(func, ret),
        )
