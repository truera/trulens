from __future__ import annotations

import logging
from pprint import PrettyPrinter
from typing import Generic, Tuple, TypeVar

from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.schema import base as base_schema
from trulens.core.utils import asynchro as asynchro_utils
from trulens.core.utils import python as python_utils
from trulens.experimental.otel_tracing.core import trace as mod_trace
from trulens.experimental.otel_tracing.core._utils import wrap as wrap_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

Args = TypeVar("Args")
Ret = TypeVar("Ret")
Res = TypeVar("Res")


class _WrapperEndpointCallback(
    mod_trace.TracingCallbacks[Ret, mod_trace.LiveSpanCallWithCost],
    Generic[Ret, Res],
):
    """EXPERIMENTAL(otel_tracing): Extension to TracingCallbacks that tracks
    costs.

    Type Args:
        Ret: The return type of the wrapped callable.

        Res: The response type of the endpoint. This can be the same as Ret but
            does not need to.
    """

    # overriding CallableCallbacks
    def __init__(self, endpoint: core_endpoint.Endpoint, **kwargs):
        super().__init__(**kwargs, span_type=mod_trace.LiveSpanCallWithCost)

        self.endpoint: core_endpoint.Endpoint = endpoint
        self.span.endpoint = endpoint

        self.cost: base_schema.Cost = self.span.cost
        self.cost.n_requests += 1
        # Subclasses need to fill in n_*_requests either in their init or
        # on_callable_call when bindings are available.

    # overriding CallableCallbacks
    def on_callable_return(self, ret: Ret, **kwargs) -> Ret:
        """Called after a request returns.

        A return does not mean the request was successful. The return value can
        indicate failure of some sort.

        Subclasses need to override this method and extract response: Res to
        invoke on_endpoint_response on it.
        """

        return super().on_callable_return(ret=ret, **kwargs)
        # Fills in some general attributes from kwargs before the next callback
        # is called.

    # our optional
    def on_endpoint_response(self, response: Res) -> None:
        """Called after each non-error response."""

        logger.warning("No on_endpoint_response method defined for %s.", self)

    # our optional
    def on_endpoint_generation(self, response: Res) -> None:
        """Called after each completion request received a response."""

        self.cost.n_successful_requests += 1

    # our optional
    def on_endpoint_embedding(self, response: Res) -> None:
        """Called after each embedding request received a response."""

        self.cost.n_successful_requests += 1

    # our optional
    def on_endpoint_generation_chunk(self, response: Res) -> None:
        """Called after receiving a chunk from a completion request."""

        self.cost.n_stream_chunks += 1

    # our optional
    def on_endpoint_classification(self, response: Res) -> None:
        """Called after each classification request receives a response."""

        self.cost.n_successful_requests += 1


class _Endpoint(core_endpoint.Endpoint):
    def wrap_function(self, func):
        """Create a wrapper of the given function to perform cost tracking."""

        if self._experimental_wrapper_callback_class is None:
            logger.warning(
                "OTEL_TRACING costs callbacks for %s are not available. Will not track costs for this endpoint.",
                python_utils.class_name(type(self)),
            )
            return func

        return wrap_utils.wrap_callable(
            func=func,
            func_name=python_utils.callable_name(func),
            callback_class=self._experimental_wrapper_callback_class,
            endpoint=self,
        )

    @staticmethod
    def track_all_costs_tally(
        __func: asynchro_utils.CallableMaybeAwaitable[Args, Ret],
        *args,
        **kwargs,
    ) -> Tuple[Ret, python_utils.Thunk[base_schema.Cost]]:
        with mod_trace.trulens_tracer().cost(
            method_name=__func.__name__
        ) as span:
            ret = __func(*args, **kwargs)

            return ret, span.total_cost
