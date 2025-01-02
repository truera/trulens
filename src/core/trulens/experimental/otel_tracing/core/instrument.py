from functools import wraps
import logging
from typing import Any, Callable, Dict, Optional
import uuid

from opentelemetry import trace
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from trulens.core import app as core_app
from trulens.experimental.otel_tracing.core.init import TRULENS_SERVICE_NAME
from trulens.experimental.otel_tracing.core.span import Attributes
from trulens.experimental.otel_tracing.core.span import (
    set_general_span_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_user_defined_attributes,
)
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


VALID_ATTR_VALUE_TYPES = (bool, str, int, float)
"""
Per the OTEL [documentation](https://opentelemetry.io/docs/specs/otel/common/#attribute),
valid attribute value types are either:
- A primitive type: string, boolean, double precision floating point (IEEE 754-1985) or signed 64 bit integer.
- An array of primitive type values. The array MUST be homogeneous, i.e., it MUST NOT contain values of different types.
"""


def validate_value_for_attribute(value):
    """
    Ensure that value is a valid attribute value type, and coerce it to a string if it is not.

    This is helpful for lists/etc because if any single value is not a valid attribute value type, the entire list
    will not be added as a span attribute.
    """
    arg_type = type(value)

    # Coerge the argument to a string if it is not a valid attribute value type.
    if arg_type not in VALID_ATTR_VALUE_TYPES:
        return str(value)

    return value


def validate_list_of_values_for_attribute(arguments: list):
    """
    Ensure that all values in a list are valid attribute value types, and coerce them to strings if they are not.
    """
    return list(map(validate_value_for_attribute, arguments))


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


def validate_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to validate span attributes based on the span type.
    """
    if not isinstance(attributes, dict) or any([
        not isinstance(key, str) for key in attributes.keys()
    ]):
        raise ValueError("Attributes must be a dictionary with string keys.")
    return validate_selector_name(attributes)
    # TODO: validate Span type attributes.


def instrument(
    *,
    span_type: SpanAttributes.SpanType = SpanAttributes.SpanType.UNKNOWN,
    attributes: Attributes = {},
):
    """
    Decorator for marking functions to be instrumented in custom classes that are
    wrapped by TruCustomApp, with OpenTelemetry tracing.
    """

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

                span.set_attribute("name", func.__name__)

                try:
                    ret = func(*args, **kwargs)
                except Exception as e:
                    # We want to get into the next clause to allow the users to still add attributes.
                    # It's on the user to deal with None as a return value.
                    func_exception = e

                set_general_span_attributes(span, span_type)
                attributes_exception = None

                try:
                    set_user_defined_attributes(
                        span,
                        span_type=span_type,
                        args=args,
                        kwargs=kwargs,
                        ret=ret,
                        func_exception=func_exception,
                        attributes=attributes,
                    )

                except Exception as e:
                    attributes_exception = e
                    logger.error(f"Error setting attributes: {e}")

                exception = func_exception or attributes_exception

                if exception:
                    raise exception

            return ret

        return wrapper

    return inner_decorator


class App(core_app.App):
    # For use as a context manager.
    def __enter__(self):
        logger.debug("Entering the OTEL app context.")

        # Note: This is not the same as the record_id in the core app since the OTEL
        # tracing is currently separate from the old records behavior
        otel_record_id = str(uuid.uuid4())

        tracer = trace.get_tracer_provider().get_tracer(TRULENS_SERVICE_NAME)

        # Calling set_baggage does not actually add the baggage to the current context, but returns a new one
        # To avoid issues with remembering to add/remove the baggage, we attach it to the runtime context.
        self.tokens.append(
            context_api.attach(
                set_baggage(SpanAttributes.RECORD_ID, otel_record_id)
            )
        )
        self.tokens.append(
            context_api.attach(set_baggage(SpanAttributes.APP_ID, self.app_id))
        )

        # Use start_as_current_span as a context manager
        self.span_context = tracer.start_as_current_span("root")
        root_span = self.span_context.__enter__()

        # Set general span attributes
        root_span.set_attribute("kind", "SPAN_KIND_TRULENS")
        root_span.set_attribute("name", "root")
        root_span.set_attribute(
            SpanAttributes.SPAN_TYPE, SpanAttributes.SpanType.RECORD_ROOT
        )
        root_span.set_attribute(SpanAttributes.APP_ID, self.app_id)
        root_span.set_attribute(SpanAttributes.RECORD_ID, otel_record_id)

        # Set record root specific attributes
        root_span.set_attribute(
            SpanAttributes.RECORD_ROOT.APP_NAME, self.app_name
        )
        root_span.set_attribute(
            SpanAttributes.RECORD_ROOT.APP_VERSION, self.app_version
        )
        root_span.set_attribute(SpanAttributes.RECORD_ROOT.APP_ID, self.app_id)
        root_span.set_attribute(
            SpanAttributes.RECORD_ROOT.RECORD_ID, otel_record_id
        )

        return root_span

    def __exit__(self, exc_type, exc_value, exc_tb):
        remove_baggage(SpanAttributes.RECORD_ID)
        remove_baggage(SpanAttributes.APP_ID)

        logger.debug("Exiting the OTEL app context.")

        while self.tokens:
            # Clearing the context once we're done with this root span.
            # See https://github.com/open-telemetry/opentelemetry-python/issues/2432#issuecomment-1593458684
            context_api.detach(self.tokens.pop())

        if self.span_context:
            self.span_context.__exit__(exc_type, exc_value, exc_tb)
