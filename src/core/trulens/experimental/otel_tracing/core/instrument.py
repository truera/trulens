from functools import wraps
import logging
from typing import Any, Callable, Optional, Union
import uuid

from opentelemetry import trace
from opentelemetry.baggage import get_baggage
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from trulens.apps.custom import instrument as custom_instrument
from trulens.core import app as core_app
from trulens.experimental.otel_tracing.core.init import TRULENS_SERVICE_NAME
from trulens.experimental.otel_tracing.core.semantic import (
    TRULENS_SELECTOR_NAME,
)
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


def instrument(
    attributes: Optional[
        Union[dict[str, Any], Callable[[Any, Any, Any], dict[str, Any]]]
    ] = {},
):
    """
    Decorator for marking functions to be instrumented in custom classes that are
    wrapped by TruCustomApp, with OpenTelemetry tracing.
    """

    def _validate_selector_name(final_attributes: dict[str, Any]):
        if TRULENS_SELECTOR_NAME in final_attributes:
            selector_name = final_attributes[TRULENS_SELECTOR_NAME]
            if not isinstance(selector_name, str):
                raise ValueError(
                    f"Selector name must be a string, not {type(selector_name)}"
                )

    def _validate_attributes(final_attributes: dict[str, Any]):
        _validate_selector_name(final_attributes)
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
            ) as parent_span:
                span = trace.get_current_span()

                span.set_attribute("name", func.__name__)
                span.set_attribute("kind", "SPAN_KIND_TRULENS")
                span.set_attribute(
                    "parent_span_id", parent_span.get_span_context().span_id
                )
                span.set_attribute(
                    SpanAttributes.RECORD_ID,
                    str(get_baggage(SpanAttributes.RECORD_ID)),
                )

                ret = func(*args, **kwargs)

                try:
                    attributes_to_add = {}

                    # Set the user provider attributes.
                    if attributes:
                        if callable(attributes):
                            attributes_to_add = attributes(ret, *args, **kwargs)
                        else:
                            attributes_to_add = attributes

                    logger.info(f"Attributes to add: {attributes_to_add}")

                    _validate_attributes(attributes_to_add)

                    for key, value in attributes_to_add.items():
                        span.set_attribute(key, value)

                except Exception:
                    span.set_attribute("status", "STATUS_CODE_ERROR")
                    return None

                span.set_attribute("status", "STATUS_CODE_UNSET")
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
