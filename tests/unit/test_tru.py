from trulens.core import Tru
from trulens.core.feedback import Endpoint


def test_singleton():
    tru = Tru()
    tru2 = Tru()
    assert tru is tru2

    endpoint = Endpoint()
    endpoint2 = Endpoint()
    assert endpoint is not endpoint2
