"""W3C trace context propagation for outbound HTTP requests.

TruLens instruments the caller, but the LLM provider runs on the other side of
an HTTP call. Nothing carries the active span's context across that boundary,
so the provider's server span starts a trace of its own and the two ends are
correlated only by timing. The helpers here put the active span's W3C trace
context on the outbound request headers, so the provider's span becomes a child
of the caller's span and both ends land in one trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, MutableMapping, Optional

from opentelemetry.propagate import inject
from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    import httpx


def inject_trace_context(headers: MutableMapping[str, str]) -> bool:
    """Inject the active span's trace context into `headers`.

    Uses the globally configured propagators, so a `traceparent` header is
    written when the default W3C propagator is installed, and nothing is
    written when the span context is invalid or propagation is disabled.

    Args:
        headers: Mutable header mapping to write into, modified in place.

    Returns:
        True when a valid span context was available and injected.
    """

    if not get_current_span().get_span_context().is_valid:
        return False

    inject(headers)
    return True


def tracing_transport(inner: Optional[Any] = None) -> Any:
    """Return an httpx transport that injects trace context on every request.

    Attach it to a client and hand that client to the provider SDK, so the
    provider's server span is parented under the active TruLens span:

        client = httpx.Client(transport=tracing_transport())
        provider = OpenAI(http_client=client)

    Args:
        inner: Transport to delegate to. Defaults to `httpx.HTTPTransport`,
            which is what a plain `httpx.Client` uses.

    Returns:
        An `httpx.BaseTransport` that injects before delegating.
    """

    # httpx is not a dependency of trulens-core. The openai and anthropic SDKs
    # already require it, so it is present wherever this is useful, and it is
    # imported here to keep it out of the core dependency set.
    import httpx

    class _TracingTransport(httpx.BaseTransport):
        def __init__(
            self, inner_transport: Optional[httpx.BaseTransport] = None
        ) -> None:
            self._inner = (
                inner_transport
                if inner_transport is not None
                else httpx.HTTPTransport()
            )

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            inject_trace_context(request.headers)
            return self._inner.handle_request(request)

    return _TracingTransport(inner)
