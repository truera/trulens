"""A cost scope nesting N provider calls must not multiply their tokens.

`Endpoint._track_costs` copied the per-call-class callback dict with a plain
`dict(...)`, which only copies the dict's top level. The lists of
`(endpoint, callback)` pairs inside it were still shared with the parent
scope, so appending a new callback for a nested call mutated the parent's
list in place. `contextvars.ContextVar.reset` only restores which list
object the var points to, so it couldn't undo that mutation once the nested
call returned, and every call made afterward in the same outer scope kept
notifying every callback that preceded it. Summed over N sequential nested
calls, each already-returned callback absorbs one extra call for every
sibling that runs after it, so the total requests counted across all N
per-call callbacks is N(N+1)/2 instead of N.
"""

import unittest

from trulens.core.feedback.endpoint import Endpoint
from trulens.core.feedback.endpoint import EndpointCallback


class _CountingCallback(EndpointCallback):
    pass


class _CountingEndpoint(Endpoint):
    def __init__(self, **kwargs):
        kwargs["callback_class"] = _CountingCallback
        super().__init__(**kwargs)

    def handle_wrapped_call(self, func, bindings, response, callback):
        callback.handle_generation(response)
        return response


def _make_endpoint() -> _CountingEndpoint:
    return _CountingEndpoint(name="test_nested_cost_scope")


class TestNestedCostScope(unittest.TestCase):
    def test_sibling_calls_in_one_scope_do_not_double_count(self) -> None:
        endpoint = _make_endpoint()
        wrapped = endpoint.wrap_function(lambda: "ok")

        def run_n_calls(n):
            callbacks = []
            for _ in range(n):
                _, callback = endpoint.track_cost(wrapped)
                callbacks.append(callback)
            return callbacks

        n = 4
        callbacks, _outer_callback = endpoint.track_cost(run_n_calls, n)

        # Each call's own callback should see exactly its own request, not
        # every sibling call made after it in the same outer scope.
        for i, callback in enumerate(callbacks):
            self.assertEqual(
                callback.cost.n_requests,
                1,
                f"callback {i} absorbed {callback.cost.n_requests} requests, "
                "expected exactly its own",
            )

        total = sum(callback.cost.n_requests for callback in callbacks)
        self.assertEqual(
            total,
            n,
            f"summed requests across all {n} per-call callbacks was "
            f"{total}, expected {n} (the buggy value is n(n+1)/2 = "
            f"{n * (n + 1) // 2})",
        )

    def test_outer_scope_still_sees_every_nested_call(self) -> None:
        """The outer callback is meant to see every nested call -- only the
        already-returned sibling callbacks should stop accumulating."""
        endpoint = _make_endpoint()
        wrapped = endpoint.wrap_function(lambda: "ok")

        def run_n_calls(n):
            for _ in range(n):
                endpoint.track_cost(wrapped)

        n = 3
        _, outer_callback = endpoint.track_cost(run_n_calls, n)

        self.assertEqual(outer_callback.cost.n_requests, n)


if __name__ == "__main__":
    unittest.main()
