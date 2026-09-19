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
        True when the configured propagators actually wrote something.
    """

    if not get_current_span().get_span_context().is_valid:
        return False

    # Inject into a scratch mapping first. With a no-op propagator installed the
    # span context is valid but nothing is written, and the return value should
    # say so rather than report an injection that did not happen.
    injected: dict[str, str] = {}
    inject(injected)
    if not injected:
        return False

    headers.update(injected)
    return True


def tracing_transport(inner: Optional[Any] = None) -> Any:
    """Return an httpx transport that injects trace context on every request.

    This is the synchronous transport. For `httpx.AsyncClient`, use
    `async_tracing_transport`; the two are not interchangeable, because httpx
    dispatches sync and async requests through different base classes.

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

        def close(self) -> None:
            # The inner transport owns the connection pool, so it has to be the
            # one that closes it.
            self._inner.close()

        def __enter__(self) -> Any:
            self._inner.__enter__()
            return self

        def __exit__(self, *exc_info: Any) -> None:
            self.close()

    return _TracingTransport(inner)


def async_tracing_transport(inner: Optional[Any] = None) -> Any:
    """Return an async httpx transport that injects trace context.

    The async counterpart to `tracing_transport`, for the async clients the
    provider SDKs expose:

        client = httpx.AsyncClient(transport=async_tracing_transport())
        provider = AsyncOpenAI(http_client=client)

    Args:
        inner: Transport to delegate to. Defaults to
            `httpx.AsyncHTTPTransport`, which is what a plain
            `httpx.AsyncClient` uses.

    Returns:
        An `httpx.AsyncBaseTransport` that injects before delegating.
    """

    import httpx

    class _AsyncTracingTransport(httpx.AsyncBaseTransport):
        def __init__(
            self, inner_transport: Optional[httpx.AsyncBaseTransport] = None
        ) -> None:
            self._inner = (
                inner_transport
                if inner_transport is not None
                else httpx.AsyncHTTPTransport()
            )

        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            inject_trace_context(request.headers)
            return await self._inner.handle_async_request(request)

        async def aclose(self) -> None:
            await self._inner.aclose()

        async def __aenter__(self) -> Any:
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *exc_info: Any) -> None:
            await self.aclose()

    return _AsyncTracingTransport(inner)
