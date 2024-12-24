from functools import wraps
import logging
<<<<<<< HEAD
from typing import Callable, Optional
=======
from typing import Any, Callable, Dict, Optional, Union
>>>>>>> d4bbf065b (remove artifacts)
import uuid

from opentelemetry import trace
from opentelemetry.baggage import get_baggage
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
        logging.debug("Entering the OTEL app context.")

        # Note: This is not the same as the record_id in the core app since the OTEL
        # tracing is currently separate from the old records behavior
        otel_record_id = str(uuid.uuid4())

        tracer = trace.get_tracer_provider().get_tracer(TRULENS_SERVICE_NAME)

        # Calling set_baggage does not actually add the baggage to the current context, but returns a new one
        # To avoid issues with remembering to add/remove the baggage, we attach it to the runtime context.
        self.token = context_api.attach(
            set_baggage(SpanAttributes.RECORD_ID, otel_record_id)
        )

        # Use start_as_current_span as a context manager
        self.span_context = tracer.start_as_current_span("root")
        root_span = self.span_context.__enter__()

        logger.debug(str(get_baggage(SpanAttributes.RECORD_ID)))

        root_span.set_attribute("kind", "SPAN_KIND_TRULENS")
        root_span.set_attribute("name", "root")
        root_span.set_attribute(
            SpanAttributes.SPAN_TYPE, SpanAttributes.SpanType.RECORD_ROOT
        )
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
        logging.debug("Exiting the OTEL app context.")

        if self.token:
            # Clearing the context once we're done with this root span.
            # See https://github.com/open-telemetry/opentelemetry-python/issues/2432#issuecomment-1593458684
            context_api.detach(self.token)

        if self.span_context:
            self.span_context.__exit__(exc_type, exc_value, exc_tb)
