from functools import wraps
import logging
from typing import Any, Callable, Dict, Optional, Union

from opentelemetry import trace
from trulens.experimental.otel_tracing.core.init import TRULENS_SERVICE_NAME
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


def instrument(
    *,
    attributes: Optional[
        Union[
            Dict[str, Any],
            Callable[
                [Optional[Any], Optional[Exception], Any, Any], Dict[str, Any]
            ],
        ]
    ] = {},
):
    """
    Decorator for marking functions to be instrumented in custom classes that are
    wrapped by TruCustomApp, with OpenTelemetry tracing.
    """

    def _validate_selector_name(attributes: Dict[str, Any]) -> Dict[str, Any]:
        result = attributes.copy()

        if (
            SpanAttributes.SELECTOR_NAME_KEY in result
            and SpanAttributes.SELECTOR_NAME in result
        ):
            raise ValueError(
                f"Both {SpanAttributes.SELECTOR_NAME_KEY} and {SpanAttributes.SELECTOR_NAME} cannot be set."
            )

        if SpanAttributes.SELECTOR_NAME in result:
            # Transfer the trulens namespaced to the non-trulens namespaced key.
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

    def _validate_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(attributes, dict) or any([
            not isinstance(key, str) for key in attributes.keys()
        ]):
            raise ValueError(
                "Attributes must be a dictionary with string keys."
            )
        return _validate_selector_name(attributes)
        # TODO: validate OTEL attributes.
        # TODO: validate span type attributes.

    def inner_decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with (
                trace.get_tracer_provider()
                .get_tracer(TRULENS_SERVICE_NAME)
                .start_as_current_span(
                    name=func.__name__,
                )
            ) as span:
                ret = None
                func_exception: Optional[Exception] = None
                attributes_exception: Optional[Exception] = None

                try:
                    ret = func(*args, **kwargs)
                except Exception as e:
                    # We want to get into the next clause to allow the users to still add attributes.
                    # It's on the user to deal with None as a return value.
                    func_exception = e

                try:
                    attributes_to_add = {}

                    # Set the user provider attributes.
                    if attributes:
                        if callable(attributes):
                            attributes_to_add = attributes(
                                ret, func_exception, *args, **kwargs
                            )
                        else:
                            attributes_to_add = attributes

                    logger.info(f"Attributes to add: {attributes_to_add}")

                    final_attributes = _validate_attributes(attributes_to_add)

                    prefix = "trulens."
                    if (
                        SpanAttributes.SPAN_TYPE in final_attributes
                        and final_attributes[SpanAttributes.SPAN_TYPE]
                        != SpanAttributes.SpanType.UNKNOWN
                    ):
                        prefix += (
                            final_attributes[SpanAttributes.SPAN_TYPE] + "."
                        )

                    for key, value in final_attributes.items():
                        span.set_attribute(prefix + key, value)

                        if (
                            key != SpanAttributes.SELECTOR_NAME_KEY
                            and SpanAttributes.SELECTOR_NAME_KEY
                            in final_attributes
                        ):
                            span.set_attribute(
                                f"trulens.{final_attributes[SpanAttributes.SELECTOR_NAME_KEY]}.{key}",
                                value,
                            )

                except Exception as e:
                    attributes_exception = e
                    logger.error(f"Error setting attributes: {e}")

                if func_exception:
                    raise func_exception
                if attributes_exception:
                    raise attributes_exception

                return ret

        return wrapper

    return inner_decorator
