from __future__ import annotations

import logging
from pprint import PrettyPrinter
from typing import (
    Any,
    TypeVar,
)

from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.schema import base as base_schema
from trulens.experimental.otel_tracing.core import trace as mod_trace

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")


class _WrapperEndpointCallback(mod_trace.TracingCallbacks[T]):
    """EXPERIMENTAL(otel_tracing).

    Extension to TracingCallbacks that tracks costs.
    """

    # overriding CallableCallbacks
    def __init__(self, endpoint: core_endpoint.Endpoint, **kwargs):
        super().__init__(**kwargs, span_type=mod_trace.LiveSpanCallWithCost)

        self.endpoint: core_endpoint.Endpoint = endpoint
        self.span.endpoint = endpoint

        self.cost: base_schema.Cost = self.span.cost
        self.cost.n_requests += 1

    # overriding CallableCallbacks
    def on_callable_return(self, ret: T, **kwargs) -> T:
        """Called after a request returns."""

        self.cost.n_responses += 1

        ret = super().on_callable_return(ret=ret, **kwargs)
        # Fills in some general attributes from kwargs before the next callback
        # is called.

        self.on_endpoint_response(response=ret)

        return ret

    # our optional
    def on_endpoint_response(self, response: Any) -> None:
        """Called after each non-error response."""

        logger.warning("No on_endpoint_response method defined for %s.", self)

    # our optional
    def on_endpoint_generation(self, response: Any) -> None:
        """Called after each completion request."""

        self.cost.n_successful_requests += 1
        self.cost.n_generations += 1

    # our optional
    def on_endpoint_generation_chunk(self, response: Any) -> None:
        """Called after receiving a chunk from a completion request."""

        self.cost.n_stream_chunks += 1

    # our optional
    def on_endpoint_classification(self, response: Any) -> None:
        """Called after each classification response."""

        self.cost.n_successful_requests += 1
        self.cost.n_classifications += 1
