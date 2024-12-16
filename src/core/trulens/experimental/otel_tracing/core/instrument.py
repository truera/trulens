from functools import wraps
import logging
from typing import Any, Callable, Optional, Union

from opentelemetry import trace
from trulens.apps.custom import instrument as custom_instrument
from trulens.experimental.otel_tracing.core.init import TRULENS_SERVICE_NAME
from trulens.experimental.otel_tracing.core.semantic import (
    TRULENS_SELECTOR_NAME,
)

logger = logging.getLogger(__name__)

type Attributes = Optional[
    Union[dict[str, Any], Callable[[Any, Any, Any], dict[str, Any]]]
]


def instrument(attributes: Attributes = {}):
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
                    name="test",
                )
            ) as parent_span:
                span = trace.get_current_span()
                _instrumented_object, *rest_args = args

                span.set_attribute("name", "test")
                span.set_attribute("kind", "SPAN_KIND_TRULENS")
                span.set_attribute(
                    "parent_span_id", parent_span.get_span_context().span_id
                )

                try:
                    ret = custom_instrument(func)(*args, **kwargs)

                    attributes_to_add = {}

                    # Set the user provider attributes.
                    if attributes:
                        if callable(attributes):
                            attributes_to_add = attributes(
                                ret, *rest_args, **kwargs
                            )
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
