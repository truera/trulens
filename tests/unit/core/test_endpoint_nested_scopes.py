"""Unit tests for how nested cost-tracking scopes hand callbacks to each other.

``_track_costs`` keeps the endpoints it is tracking in a contextvar, and copies
that dict on entry so a scope can add its own callbacks without disturbing its
parent. The copy has to include the lists the dict holds: sharing them means
every nested scope leaves a callback behind in its ancestors, and since
``tru_wrapper`` dispatches a response to every callback registered for the
endpoint's class, each leftover callback counts the request again.

No external service is contacted: these use the real ``DummyEndpoint`` and the
real instrumented ``DummyAPI.post``, which go through the same
``_track_costs`` / ``wrap_function`` / ``EndpointCallback`` code a real provider
call does.
"""

import pytest
from trulens.core.feedback.endpoint import Endpoint
from trulens.feedback.dummy.endpoint import DummyAPI
from trulens.feedback.dummy.endpoint import DummyEndpoint

_URL = "https://example.invalid/v1/chat"
_BODY = {"model": "m", "prompt": "p", "temperature": 0.0}


def _post_n(endpoint: Endpoint, api: DummyAPI, n: int) -> None:
    for _ in range(n):
        Endpoint._track_costs(
            api.post, url=_URL, json=dict(_BODY), with_endpoints=[endpoint]
        )


class TestNestedCostScopes:
    def test_single_scope_counts_each_request_once(self):
        endpoint = DummyEndpoint()
        api = DummyAPI()

        _post_n(endpoint, api, 1)

        assert endpoint.global_callback.cost.n_requests == 1

    def test_nested_scope_does_not_leak_a_callback_into_its_parent(self):
        """Regression: the inner scope appended to the parent's list in place,
        so the parent ended up holding two callbacks and every later request was
        counted twice."""
        endpoint = DummyEndpoint()
        api = DummyAPI()
        key = endpoint.callback_class
        seen = {}

        def outer() -> None:
            before = Endpoint._context_endpoints.get()[key]
            seen["before"] = len(before)
            seen["id"] = id(before)
            _post_n(endpoint, api, 1)
            after = Endpoint._context_endpoints.get()[key]
            seen["after"] = len(after)
            seen["same_list"] = before is after

        Endpoint._track_costs(outer, with_endpoints=[endpoint])

        assert seen["before"] == 1
        assert seen["after"] == 1, (
            "the inner scope grew the parent's callback list, so the parent "
            "counts this and every later request more than once; revert the "
            "deep-copy fix at endpoint.py:647"
        )
        assert seen["same_list"], (
            "the list object is expected to be the parent's own, since the "
            "scope copies the dict rather than replacing the parent's entry"
        )

    @pytest.mark.parametrize("n", [1, 2, 4, 8])
    def test_request_count_does_not_grow_with_the_number_of_nested_scopes(
        self, n
    ):
        """Each request is dispatched once to the callback of the scope that
        opened it and once to the callback of the enclosing scope, because
        `handle_wrapped_call` counts on `global_callback` for every dispatch
        and the enclosing scope's callback is still registered. That is the
        documented nesting behaviour and it is linear in the number of
        requests. Before the fix each nested scope also left its callback
        behind in the enclosing scope, so the dispatch list grew by one per
        scope and the total went as the sum of 2..n+1 instead."""
        endpoint = DummyEndpoint()
        api = DummyAPI()

        Endpoint._track_costs(
            lambda: _post_n(endpoint, api, n), with_endpoints=[endpoint]
        )

        assert endpoint.global_callback.cost.n_requests == 2 * n

    def test_tally_read_later_does_not_absorb_a_later_request(self):
        """`track_all_costs_tally` hands back a live thunk, and `Metric.run`
        reads it after the call returns, so a tally is read after later scopes
        have run. It must not pick up their requests."""
        api = DummyAPI()

        _, tally = Endpoint.track_all_costs_tally(
            api.post, url=_URL, json=dict(_BODY), with_dummy=True
        )
        assert tally().n_requests == 1

        # An independent scope for the same endpoint class, afterwards.
        Endpoint.track_all_costs_tally(
            api.post, url=_URL, json=dict(_BODY), with_dummy=True
        )

        assert tally().n_requests == 1
