from typing import Callable

from opentelemetry import trace
from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument as custom_instrument
from trulens.core import instruments as core_instruments
from trulens.experimental.otel_tracing.core.init import TRULENS_SERVICE_NAME


class instrument2(core_instruments.instrument):
    """
    Decorator for marking methods to be instrumented in custom classes that are
    wrapped by TruCustomApp, with OpenTelemetry tracing.
    """

    @classmethod
    def method(cls, inst_cls: type, name: str) -> None:
        core_instruments.instrument.method(inst_cls, name)

        # Also make note of it for verification that it was found by the walk
        # after init.
        TruCustomApp.functions_to_instrument.add(getattr(inst_cls, name))

    # `_self` is used to avoid conflicts where `self` may be passed from the caller method
    def __call__(_self, *args, **kwargs):
        print("in call")
        with (
            trace.get_tracer_provider()
            .get_tracer(TRULENS_SERVICE_NAME)
            .start_as_current_span(
                name=_self.func.__name__,
            )
        ) as span:
            span.set_attribute("function", _self.func.__name__)
            span.set_attribute("args", args)
            span.set_attribute("kwargs", **kwargs)
            ret = super.__call__(_self, *args, **kwargs)
            span.set_attribute("return", ret)
            return ret


def instrument(func: Callable):
    """
    Decorator for marking functions to be instrumented in custom classes that are
    wrapped by TruCustomApp, with OpenTelemetry tracing.
    """

    def wrapper(*args, **kwargs):
        with (
            trace.get_tracer_provider()
            .get_tracer(TRULENS_SERVICE_NAME)
            .start_as_current_span(
                name=func.__name__,
            )
        ) as span:
            span.set_attribute("function", func.__name__)
            span.set_attribute("args", args)
            ret = custom_instrument(func)(*args, **kwargs)
            span.set_attribute("return", ret)
            return ret

    return wrapper
