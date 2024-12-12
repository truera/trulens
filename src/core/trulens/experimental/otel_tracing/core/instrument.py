from functools import wraps
import logging
from typing import Any, Callable, Union

from opentelemetry import trace
from trulens.apps.custom import instrument as custom_instrument
from trulens.core.utils import json as json_utils
from trulens.experimental.otel_tracing.core.init import TRULENS_SERVICE_NAME
from trulens.experimental.otel_tracing.core.semantic import (
    TRULENS_SELECTOR_NAME,
)

logger = logging.getLogger(__name__)

type Attributes = Union[
    dict[str, Any], Callable[[Any, Any, Any], dict[str, Any]]
]


def instrument(func: Callable, attributes: Attributes = {}):
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

    @wraps(func)
    def wrapper(*args, **kwargs):
        with (
            trace.get_tracer_provider()
            .get_tracer(TRULENS_SERVICE_NAME)
            .start_as_current_span(
                name=func.__name__,
            )
        ) as span:
            _instrumented_object, *rest_args = args

            # ? Do we have to validate the args/kwargs and make sure they are serializable?
            span.set_attribute("function", func.__name__)
            span.set_attribute("args", json_utils.json_str_of_obj(rest_args))
            ret = custom_instrument(func)(*args, **kwargs)
            span.set_attribute("return", ret)

            attributes_to_add = {}

            # Set the user provider attributes.
            if attributes:
                if callable(attributes):
                    attributes_to_add = attributes(ret, *args, **kwargs)
                else:
                    attributes_to_add = attributes

            _validate_attributes(attributes_to_add)

            for key, value in attributes_to_add.items():
                span.set_attribute(key, value)

            return ret

    return wrapper
