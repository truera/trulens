from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession

from tests.util.otel_app_test_case import OtelAppTestCase


class _FailingApp:
    @instrument()
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"


def _failing_construct_event(*args, **kwargs):
    raise ValueError()


class TestOtelTruSession(OtelAppTestCase):
    def test_main_thread_fails_if_exporter_fails(self) -> None:
        tru_session = TruSession()
        tru_session.connector.add_events = _failing_construct_event
        failing_app = _FailingApp()
        custom_app = TruApp(failing_app, main_method=failing_app.greet)
        with custom_app(run_name="test run", input_id="456"):
            failing_app.greet("this is going to fail!")
        tru_session.force_flush()
        import time

        for i in range(1000):
            time.sleep(1)
