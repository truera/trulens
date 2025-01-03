from functools import wraps
import logging
from typing import Callable, Optional

from opentelemetry import trace
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
