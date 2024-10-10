from __future__ import annotations

import logging
from typing import Callable, TypeVar
import weakref

from trulens.core import instruments as core_instruments
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing.core import trace as mod_trace
from trulens.experimental.otel_tracing.core._utils import wrap as wrap_utils

logger = logging.getLogger(__name__)

T = TypeVar("T")


def deproxy(proxy: weakref.ProxyType[T]) -> T:
    """Return the object being proxied."""

    return proxy.__init__.__self__


class _Instrument(core_instruments.Instrument):
    def tracked_method_wrapper(
        self,
        query: serial_utils.Lens,
        func: Callable,
        method_name: str,
        cls: type,
        obj: object,
    ):
        """Wrap a method to capture its inputs/outputs/errors."""

        if self.app is None:
            raise ValueError("Instrumentation requires an app but is None.")

        if python_utils.safe_hasattr(func, "__func__"):
            raise ValueError("Function expected but method received.")

        if python_utils.safe_hasattr(func, mod_trace.INSTRUMENT):
            logger.debug("\t\t\t%s: %s is already instrumented", query, func)

        # Notify the app instrumenting this method where it is located:
        self.app.on_method_instrumented(obj, func, path=query)

        logger.debug("\t\t\t%s: instrumenting %s=%s", query, method_name, func)

        return wrap_utils.wrap_callable(
            func=func,
            callback_class=mod_trace.AppTracingCallbacks,
            func_name=method_name,
            app=deproxy(self.app),
        )
